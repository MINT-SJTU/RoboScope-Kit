# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""GR00T N1.7 BEHAVIOR Policy example for RoboScope Kit.

Serves ``kmy17518/gr00t-n1.7-b1k-multitask``, a GR00T N1.7 fine-tune on the 100 tasks of the
BEHAVIOR Challenge 2026, for the ``behavior`` benchmark's ``challenge_2026`` suite. The model runs
in its own process: the checkpoint's native BEHAVIOR server, ``scripts/b1k/serve_b1k.py`` of the
pinned Isaac-GR00T fork, in that fork's Python 3.10 environment (``build-model-env.sh``). This
Policy, which runs on Python 3.11+ with the kit, starts that server on a free local port, waits for
its ``/healthz`` endpoint, and talks to it over the BEHAVIOR websocket protocol (msgpack with NumPy
arrays), exactly as the native evaluator's ``WebsocketPolicy`` does. The Python 3.10 process never
imports the kit.

Everything model-specific stays in the server, as natively: the per-task prompt resolved from
``task_id``, the 224x224 letter-box resize, the state slicing of the 61-dim proprioception, the
40-step action horizon, and ``temporal_ensemble`` (one inference per control step, actions averaged
over the five most recent chunks, grippers from the newest). Each request carries what the
server reads from a native observation, with the native key names and a leading batch dimension of
one: the three cameras' RGB (the native RGBA without alpha, which the server drops itself), the
proprioception and the task ID.

The server keeps one temporal-ensemble buffer for its whole connection, so one replica serves one
episode at a time; ``reset`` rejects an overlapping episode. Run with as many replicas as workers.

Configuration (environment variables of the Policy process, see ``roboscope-policy.json``):

- ``GR00T_ROOT``: the Isaac-GR00T checkout of ``build-model-env.sh``; ``GR00T_PYTHON``: its Python;
- ``HF_HOME``: the Hugging Face cache holding ``nvidia/Cosmos-Reason2-2B`` (the server runs offline);
- ``GR00T_READY_TIMEOUT_S`` (default 900): how long to wait for the server to load the model.

``--checkpoint-source`` is the snapshot directory, for example ``.../checkpoint-238000``.
"""

from __future__ import annotations

import functools
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import msgpack
import numpy as np

from roboscope_kit.policy import (
    ActionChunk,
    EpisodeContext,
    Observation,
    Policy,
    PolicyCapabilities,
    SessionContext,
)

BENCHMARK = "behavior"
SUITE = "challenge_2026"
ACTION_REPRESENTATION = "behavior.r1pro-joint-v1"
ACTION_DIM = 23
# SDK feature -> the native observation key the server reads.
NATIVE_KEYS = {
    "images.head": "robot_r1::robot_r1:zed_link:Camera:0::rgb",
    "images.left_wrist": "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
    "images.right_wrist": "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
    "state.proprioception": "robot_r1::proprio",
    "state.task_id": "task_id",
}
# Files serve_b1k.py loads from the snapshot directory.
CHECKPOINT_FILES = (
    "config.json",
    "processor_config.json",
    "statistics.json",
    "embodiment_id.json",
    "model.safetensors.index.json",
    "experiment_cfg/conf.yaml",
)


def _pack_array(obj: Any) -> Any:
    # gr00t.policy.websocket_b1k_server / omnigibson.eval.utils.network_utils array encoding.
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"unsupported dtype {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj: dict) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


packb = functools.partial(msgpack.packb, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def native_request(observation: Observation) -> dict[str, np.ndarray]:
    """The native evaluator's observation keys the server reads, batched as one environment."""

    request = {}
    for feature, key in NATIVE_KEYS.items():
        value = np.asarray(observation.features[feature])
        request[key] = np.ascontiguousarray(value[None])
    return request


class Gr00tBehaviorPolicy(Policy):
    def __init__(self) -> None:
        self.server: subprocess.Popen | None = None
        self.connection = None
        self.active_episode: str | None = None
        self.server_metadata: dict = {}
        # Per-episode time in seconds: packing requests, server round trips, and the server's own
        # infer_ms (preprocessing, model and temporal ensemble); logged by end_episode.
        self.timing = dict.fromkeys(("steps", "pack_s", "round_trip_s", "server_infer_s"), 0.0)

    def load(self, context: SessionContext) -> PolicyCapabilities:
        if context.benchmark.name != BENCHMARK or context.benchmark.suite != SUITE:
            raise ValueError(
                f"this example serves {BENCHMARK}/{SUITE}; "
                f"got {context.benchmark.name}/{context.benchmark.suite}"
            )
        from websockets.sync.client import connect  # noqa: PLC0415

        checkpoint = Path(context.checkpoint_path)
        missing = [name for name in CHECKPOINT_FILES if not (checkpoint / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{checkpoint} is not a GR00T snapshot; missing {', '.join(missing)}")
        root = Path(os.environ["GR00T_ROOT"])
        python = Path(os.environ["GR00T_PYTHON"])
        hf_home = Path(os.environ["HF_HOME"])
        if not (hf_home / "hub" / "models--nvidia--Cosmos-Reason2-2B").is_dir():
            raise FileNotFoundError(f"nvidia/Cosmos-Reason2-2B is not in the Hugging Face cache {hf_home}")
        port = _free_port()
        command = (
            str(python),
            str(root / "scripts" / "b1k" / "serve_b1k.py"),
            "--model-path",
            str(checkpoint),
            "--modality-config-path",
            str(root / "examples" / "b1k" / "r1pro.py"),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        )
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
        }
        env.update({"HF_HOME": str(hf_home), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        self.server = subprocess.Popen(command, cwd=root, env=env, stdout=sys.stderr, stderr=sys.stderr)
        deadline = time.monotonic() + float(os.environ.get("GR00T_READY_TIMEOUT_S", "900"))
        while True:
            if self.server.poll() is not None:
                raise RuntimeError(f"GR00T server exited with {self.server.returncode} while loading")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("GR00T server did not become healthy")
            time.sleep(2)
        self.connection = connect(
            f"ws://127.0.0.1:{port}", compression=None, max_size=None, ping_interval=60, ping_timeout=300
        )
        self.server_metadata = unpackb(self.connection.recv())
        return PolicyCapabilities(
            supported_benchmarks=(BENCHMARK,),
            supported_suites={BENCHMARK: (SUITE,)},
            required_features=tuple(NATIVE_KEYS),
            max_batch_size=1,
            supported_action_chunk_sizes=(1,),
            action_representations={BENCHMARK: ACTION_REPRESENTATION},
        )

    def reset(self, context: EpisodeContext) -> None:
        if self.active_episode is not None and self.active_episode != context.episode_id:
            raise RuntimeError(
                f"GR00T replica is serving episode {self.active_episode}; it cannot start "
                f"{context.episode_id} (use one replica per worker)"
            )
        self.active_episode = context.episode_id
        self.timing = dict.fromkeys(self.timing, 0.0)
        # The native evaluator's reset message: the server clears its temporal-ensemble buffers.
        self.connection.send(packb({"reset": True}))

    def act(self, observation: Observation) -> ActionChunk:
        if observation.episode_id != self.active_episode:
            raise RuntimeError(
                f"GR00T replica got a step of {observation.episode_id}, not {self.active_episode}"
            )
        started = time.perf_counter()
        request = packb(native_request(observation))
        packed = time.perf_counter()
        self.connection.send(request)
        response = self.connection.recv()
        received = time.perf_counter()
        if isinstance(response, str):
            raise RuntimeError(f"GR00T server error:\n{response}")
        reply = unpackb(response)
        action = np.asarray(reply["action"], dtype=np.float32)
        if action.shape != (1, ACTION_DIM):
            raise RuntimeError(
                f"GR00T server returned action shape {action.shape}, expected (1, {ACTION_DIM})"
            )
        self.timing["steps"] += 1
        self.timing["pack_s"] += packed - started
        self.timing["round_trip_s"] += received - packed
        self.timing["server_infer_s"] += float(reply.get("server_timing", {}).get("infer_ms", 0.0)) / 1000
        return ActionChunk(actions=action)

    def end_episode(self, episode_id: str) -> None:
        if self.active_episode == episode_id:
            timing = {"episode_id": episode_id, **self.timing}
            print(json.dumps({"gr00t_episode_timing": timing}), file=sys.stderr, flush=True)
            self.active_episode = None

    def unload(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.server is not None and self.server.poll() is None:
            self.server.terminate()
            try:
                self.server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.server.kill()
        self.server = None


def main() -> None:
    from roboscope_kit.server import run  # noqa: PLC0415

    run(Gr00tBehaviorPolicy())


if __name__ == "__main__":
    main()
