# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Measure LIBERO worker Policy socket I/O on CPU without a simulator or real model."""

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
from typing import Any

import numpy as np

from roboscope_kit import (
    ActionChunk,
    EpisodeContext,
    Observation,
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
ENCODINGS = ("json", "msgpack")


def _sample_observation() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    return {
        "agentview_image": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        **{
            name: rng.random(size, dtype=np.float32)
            for name, size in (
                ("robot0_eef_pos", 3),
                ("robot0_eef_quat", 4),
                ("robot0_gripper_qpos", 2),
                ("robot0_joint_pos", 7),
            )
        },
    }


def _actions(offset: float) -> np.ndarray:
    return np.arange(CHUNK_SIZE * 7, dtype=np.float32).reshape(CHUNK_SIZE, 7) / 100.0 + offset


class _MeasurementPolicy(Policy):
    """Deterministic payload verifier performing no model inference."""

    def load(self, _context: SessionContext) -> PolicyCapabilities:
        self.expected = build_observation("reference", 0, _sample_observation(), INSTRUCTION).features
        self.episode_id = ""
        self.step = 0
        self.offset = 0.0
        return PolicyCapabilities(
            supported_benchmarks=("libero_pro",),
            required_features=tuple(self.expected),
            supported_action_chunk_sizes=(CHUNK_SIZE,),
            action_representations={"libero_pro": "libero.delta-eef-v1"},
        )

    def reset(self, context: EpisodeContext) -> None:
        if self.episode_id:
            raise ValueError("previous episode was not ended")
        self.episode_id = context.episode_id
        self.step = 0
        self.offset = float(context.metadata["action_offset"])

    def act(self, request: Observation) -> ActionChunk:
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
        return ActionChunk(_actions(self.offset))

    def end_episode(self, episode_id: str) -> None:
        if episode_id != self.episode_id:
            raise ValueError("wrong episode ended")
        self.episode_id = ""

    def unload(self) -> None:
        if self.episode_id:
            raise ValueError("episode state leaked at unload")


class _JsonOnlyAdvertisementServer(PolicyServer):
    """Exercise negotiation fallback using a legacy JSON-only health reply."""

    def _dispatch(self, operation: str, payload: dict[str, Any]) -> Any:
        if operation == "health":
            return {"state": self.state.value}
        return super()._dispatch(operation, payload)


def _serve(socket_path: Path, json_only: bool) -> None:
    server_cls = _JsonOnlyAdvertisementServer if json_only else PolicyServer
    server_cls(_MeasurementPolicy(), socket_path).serve()


def _measure(encoding: str, args: argparse.Namespace) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="libero-io-") as temporary:
        socket_path = Path(temporary) / "policy.sock"
        process = multiprocessing.get_context("spawn").Process(
            target=_serve,
            args=(socket_path, args.json_only_server),
        )
        process.start()
        admin = PolicyClient(socket_path, encoding="json")
        previous_encoding = os.environ.get("ROBOSCOPE_POLICY_ENCODING")
        os.environ["ROBOSCOPE_POLICY_ENCODING"] = encoding
        queue = PolicyActionQueue(socket_path)
        samples: list[float] = []
        try:
            admin.wait_until_ready(timeout_s=20)
            admin.load(
                SessionContext(
                    session_id="transport-measurement",
                    checkpoint_path=temporary,
                    benchmark=get_benchmark_spec(BenchmarkName.LIBERO_PRO, "libero_goal_object"),
                )
            )
            raw = _sample_observation()
            for label, requests, offset in (("warmup", args.warmup, 0.0), ("measured", args.requests, 0.25)):
                queue.reset(EpisodeContext(label, "0", {"action_offset": offset}))
                expected = _actions(offset)
                for req_idx in range(requests):
                    count = CHUNK_SIZE - 3 if req_idx == requests - 1 else CHUNK_SIZE
                    start = time.perf_counter()
                    received = [
                        queue.step(raw, INSTRUCTION, req_idx * CHUNK_SIZE + step_idx)
                        for step_idx in range(count)
                    ]
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
            admin.close()
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if previous_encoding is None:
                os.environ.pop("ROBOSCOPE_POLICY_ENCODING", None)
            else:
                os.environ["ROBOSCOPE_POLICY_ENCODING"] = previous_encoding
    return {
        "requested_encoding": encoding,
        "negotiated_encoding": negotiated,
        "requests": args.requests,
        "mean_ms": statistics.mean(samples) * 1000.0,
        "median_ms": statistics.median(samples) * 1000.0,
        "p95_ms": float(np.percentile(samples, 95)) * 1000.0,
        "payloads_equal": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=300, help="measured action-chunk RPCs per repeat")
    parser.add_argument("--warmup", type=int, default=20, help="warmup action-chunk RPCs per repeat")
    parser.add_argument("--repeats", type=int, default=5, help="alternating JSON/MessagePack pairs")
    parser.add_argument("--json-only-server", action="store_true", help="verify fallback to JSON")
    parser.add_argument("--output", type=Path, required=True, help="new output JSON file path")
    args = parser.parse_args()
    if min(args.requests, args.warmup, args.repeats) < 1:
        parser.error("requests, warmup, and repeats must be positive")
    with args.output.open("x", encoding="utf-8") as output:
        rows: list[dict[str, Any]] = []
        for repeat in range(args.repeats):
            order = ENCODINGS if repeat % 2 == 0 else reversed(ENCODINGS)
            for encoding in order:
                row = {"repeat": repeat + 1, **_measure(encoding, args)}
                rows.append(row)
                print(json.dumps(row), flush=True)
        summary = [
            {
                "requested_encoding": encoding,
                "mean_ms": statistics.mean(
                    row["mean_ms"] for row in rows if row["requested_encoding"] == encoding
                ),
                "stdev_ms": (
                    statistics.stdev(
                        row["mean_ms"] for row in rows if row["requested_encoding"] == encoding
                    )
                    if args.repeats > 1
                    else 0.0
                ),
            }
            for encoding in ENCODINGS
        ]
        json.dump(
            {
                "kind": "cpu_policy_transport_only",
                "real_model": False,
                "simulator": False,
                "timestamp": datetime.now(UTC).isoformat(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "kit_version": version("roboscope-kit"),
                "numpy_version": np.__version__,
                "msgpack_version": version("msgpack"),
                "json_only_server": args.json_only_server,
                "warmup_requests": args.warmup,
                "chunk_size": CHUNK_SIZE,
                "rows": rows,
                "summary": summary,
            },
            output,
            indent=2,
        )
        output.write("\n")


if __name__ == "__main__":
    main()
