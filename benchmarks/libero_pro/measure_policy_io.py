# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Measure LIBERO worker Policy I/O on CPU, without a simulator or real model.

Run from an environment containing roboscope-kit and imageio. The server runs in a
separate process and verifies every observation. These are transport measurements,
not LIBERO-Pro throughput or success-rate results.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import statistics
import tempfile
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
from roboscope_kit import (
    ActionChunk,
    EpisodeContext,
    Policy,
    PolicyCapabilities,
    PolicyClient,
    SessionContext,
)
from roboscope_kit.benchmark_specs import BenchmarkName, get_benchmark_spec
from roboscope_kit.runtime.benchmarks.libero.worker import PolicyActionQueue, build_observation
from roboscope_kit.server import PolicyServer

INSTRUCTION = "put the bowl on the plate"
CHUNK_SIZE = 8
MODES = [(encoding, persistent) for encoding in ("json", "msgpack") for persistent in (False, True)]


def observation() -> dict:
    rng = np.random.default_rng(7)
    return {
        "agentview_image": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        **{name: rng.random(size).astype(np.float32) for name, size in (
            ("robot0_eef_pos", 3), ("robot0_eef_quat", 4),
            ("robot0_gripper_qpos", 2), ("robot0_joint_pos", 7),
        )},
    }


def actions(offset: float) -> np.ndarray:
    return np.arange(CHUNK_SIZE * 7, dtype=np.float32).reshape(CHUNK_SIZE, 7) / 100 + offset


class MeasurementPolicy(Policy):
    """Deterministic payload verifier; it performs no model inference."""

    def load(self, context):
        self.expected = build_observation("reference", 0, observation(), INSTRUCTION).features
        self.episode_id = ""
        self.step = 0
        self.offset = 0.0
        return PolicyCapabilities(
            supported_benchmarks=("libero_pro",),
            required_features=tuple(self.expected),
            supported_action_chunk_sizes=(CHUNK_SIZE,),
            action_representations={"libero_pro": "libero.delta-eef-v1"},
        )

    def reset(self, context):
        if self.episode_id:
            raise ValueError("previous episode was not ended")
        self.episode_id = context.episode_id
        self.step = 0
        self.offset = context.metadata["action_offset"]

    def act(self, request):
        if request.episode_id != self.episode_id or request.step != self.step:
            raise ValueError("episode state or action-chunk order changed")
        for name, expected in self.expected.items():
            value = request.features[name]
            if isinstance(expected, np.ndarray):
                if value.dtype != expected.dtype or not np.array_equal(value, expected):
                    raise ValueError(f"observation payload changed: {name}")
            elif value != expected:
                raise ValueError(f"instruction changed: {name}")
        self.step += CHUNK_SIZE
        return ActionChunk(actions(self.offset))

    def end_episode(self, episode_id):
        if episode_id != self.episode_id:
            raise ValueError("wrong episode ended")
        self.episode_id = ""

    def unload(self):
        if self.episode_id:
            raise ValueError("episode state leaked at unload")


class JsonOnlyAdvertisementServer(PolicyServer):
    """Exercise negotiation fallback using the old JSON-only health response."""

    def _dispatch(self, operation, payload):
        if operation == "health":
            return {"state": self.state.value}
        return super()._dispatch(operation, payload)


def serve(socket_path: Path, json_only: bool) -> None:
    server = JsonOnlyAdvertisementServer if json_only else PolicyServer
    server(MeasurementPolicy(), socket_path).serve()


def measure(encoding: str, persistent: bool, args) -> dict:
    with tempfile.TemporaryDirectory(prefix="libero-io-") as temporary:
        socket_path = Path(temporary) / "policy.sock"
        process = multiprocessing.get_context("spawn").Process(
            target=serve, args=(socket_path, args.json_only_server),
        )
        process.start()
        admin = PolicyClient(socket_path, encoding="json")
        previous_encoding = os.environ.get("ROBOSCOPE_POLICY_ENCODING")
        os.environ["ROBOSCOPE_POLICY_ENCODING"] = encoding
        queue = PolicyActionQueue(socket_path, persistent_connection=persistent)
        samples = []
        try:
            admin.wait_until_ready(timeout_s=20)
            admin.load(SessionContext(
                session_id="transport-measurement", checkpoint_path=temporary,
                benchmark=get_benchmark_spec(BenchmarkName.LIBERO_PRO, "libero_goal_object"),
            ))
            raw = observation()
            # End each episode mid-chunk, so the next reset also exercises cached-action disposal.
            for label, requests, offset in (("warmup", args.warmup, 0.0), ("measured", args.requests, 0.25)):
                queue.reset(EpisodeContext(label, "0", {"action_offset": offset}))
                expected = actions(offset)
                for request in range(requests):
                    count = CHUNK_SIZE - 3 if request == requests - 1 else CHUNK_SIZE
                    start = time.perf_counter()
                    received = [queue.step(raw, INSTRUCTION, request * CHUNK_SIZE + index)
                                for index in range(count)]
                    elapsed = time.perf_counter() - start
                    if label == "measured":
                        samples.append(elapsed)
                    if not np.array_equal(np.asarray(received), expected[:count]):
                        raise ValueError("returned actions changed or leaked across episodes")
                queue.end_episode()
            negotiated = queue.encoding
            expected_encoding = "json" if args.json_only_server else encoding
            if negotiated != expected_encoding:
                raise ValueError(f"expected {expected_encoding}, negotiated {negotiated}")
            admin.unload()
            process.join(timeout=5)
            if process.exitcode != 0:
                raise RuntimeError(f"measurement server exit code: {process.exitcode}")
        finally:
            queue.close()
            admin.close()
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if previous_encoding is None:
                os.environ.pop("ROBOSCOPE_POLICY_ENCODING", None)
            else:
                os.environ["ROBOSCOPE_POLICY_ENCODING"] = previous_encoding
    return {
        "requested_encoding": encoding, "negotiated_encoding": negotiated,
        "persistent_connection": persistent, "requests": args.requests,
        "mean_ms": statistics.mean(samples) * 1000,
        "median_ms": statistics.median(samples) * 1000,
        "p95_ms": float(np.percentile(samples, 95)) * 1000,
        "payloads_equal": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json-only-server", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.requests, args.warmup, args.repeats) < 1:
        parser.error("requests, warmup, and repeats must be positive")
    # Do not overwrite earlier measurement evidence.
    with args.output.open("x", encoding="utf-8") as output:
        rows = []
        for repeat in range(args.repeats):
            # Alternate mode order to reduce systematic warmup/thermal bias.
            for encoding, persistent in MODES if repeat % 2 == 0 else reversed(MODES):
                row = {"repeat": repeat + 1, **measure(encoding, persistent, args)}
                rows.append(row)
                print(json.dumps(row), flush=True)
        summary = []
        for encoding, persistent in MODES:
            values = [row["mean_ms"] for row in rows
                      if (row["requested_encoding"], row["persistent_connection"]) == (encoding, persistent)]
            summary.append({
                "requested_encoding": encoding, "persistent_connection": persistent,
                "mean_ms": statistics.mean(values),
                "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
            })
        json.dump({
            "kind": "cpu_policy_transport_only", "real_model": False, "simulator": False,
            "timestamp": datetime.now(UTC).isoformat(), "platform": platform.platform(),
            "python": platform.python_version(), "kit_version": version("roboscope-kit"),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "numpy_version": np.__version__, "msgpack_version": version("msgpack"),
            "json_only_server": args.json_only_server, "warmup_requests": args.warmup,
            "chunk_size": CHUNK_SIZE, "rows": rows, "summary": summary,
        }, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
