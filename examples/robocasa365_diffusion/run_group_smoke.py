# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Run one full-horizon, five-slot RoboCasa365 task through the public Kit runtime.

This checks the native batch-five path without running the full 2,500-rollout
matrix. It is a smoke workload, not a formal suite or a throughput comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from roboscope_kit.benchmark_specs import BenchmarkName, get_benchmark_spec
from roboscope_kit.policy_config import load_policy_runtime_config
from roboscope_kit.runtime.benchmark.factories import BenchmarkRunRequest
from roboscope_kit.runtime.benchmark.manifest import load_benchmark_manifest
from roboscope_kit.runtime.benchmark.orchestrator import (
    ParallelBenchmarkRuntime,
    RuntimeConfig,
    append_event,
)
from roboscope_kit.runtime.benchmark.output import reserve_output_directory
from roboscope_kit.runtime.benchmark.policy_process import PolicyProcessAdapter
from roboscope_kit.runtime.benchmark.queue import atomic_json
from roboscope_kit.runtime.benchmark.timing import BenchmarkTimingRecorder, benchmark_phase
from roboscope_kit.runtime.benchmarks.robocasa365.adapter import RoboCasa365BenchmarkAdapter
from roboscope_kit.runtime.benchmarks.robocasa365.factory import RoboCasa365BenchmarkFactory
from roboscope_kit.runtime.benchmarks.robocasa365.protocol import NATIVE_ENVS, SUITES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--policy-config", type=Path, required=True)
    parser.add_argument("--checkpoint-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-device", default="0", help="one physical GPU index; default: 0")
    parser.add_argument(
        "--task", choices=SUITES["target50"], default=SUITES["target50"][0],
        help="one official target50 task; default: CloseBlenderLid",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = BenchmarkName.ROBOCASA365
    suite = "target50"
    manifest = load_benchmark_manifest(args.benchmark_manifest)
    manifest.validate_request(benchmark, suite)
    policy = load_policy_runtime_config(args.policy_config)
    checkpoint = args.checkpoint_source
    if not checkpoint.is_absolute() or not checkpoint.exists():
        raise ValueError("--checkpoint-source must be an existing absolute local path")
    output_dir = args.output_dir.resolve()
    request = BenchmarkRunRequest(
        benchmark=benchmark, suite=suite, checkpoint_path=checkpoint,
        output_dir=output_dir, episodes=NATIVE_ENVS, quick_test=False,
    )
    # The public factory validates every frozen manifest setting. Restrict only
    # the task list; native slot count, reset streams and horizons stay intact.
    reference = RoboCasa365BenchmarkFactory().create(request, manifest)
    adapter = RoboCasa365BenchmarkAdapter(replace(
        reference.config, tasks=(args.task,), save_videos=False,
    ))
    ready_timeout = float(manifest.defaults.get("ready_timeout_s", 900))
    inference = PolicyProcessAdapter(
        command=policy.command, checkpoint_path=checkpoint,
        benchmark=benchmark, suite=suite, benchmark_spec=get_benchmark_spec(benchmark, suite),
        cwd=policy.cwd, load_timeout_s=ready_timeout,
        session_metadata={
            "validation_scope": "native-five-slot-smoke", "task": args.task,
            "native_envs": NATIVE_ENVS, "episodes": NATIVE_ENVS,
        },
    )
    runtime = ParallelBenchmarkRuntime(
        config=RuntimeConfig(
            output_dir=output_dir, gpu_devices=(args.gpu_device,),
            replicas=1, concurrency=1, ready_timeout_s=ready_timeout,
            base_port=int(manifest.defaults.get("base_port", 16694)),
        ),
        inference=inference, benchmark=adapter,
    )
    with reserve_output_directory(output_dir):
        recorder = BenchmarkTimingRecorder(output_dir / "metrics.json")
        started = time.monotonic()
        try:
            with benchmark_phase(recorder.handle_phase, "checkpoint_prepare_s"):
                if not checkpoint.exists():
                    raise FileNotFoundError(checkpoint)
            with benchmark_phase(recorder.handle_phase, "autotune_s"):
                pass
            result = runtime.run(phase_timing_callback=recorder.handle_phase)
            result["validation_scope"] = "one-task native-five-slot smoke; not a formal suite"
            result["validation_workload"] = {
                "task": args.task, "native_envs": NATIVE_ENVS, "episodes": NATIVE_ENVS,
            }
            result["autotune"] = {"k": 1, "g": 1, "mode": "fixed-group-smoke"}
            result["end_to_end_wall_s"] = time.monotonic() - started
            atomic_json(output_dir / "result.json", result)
            print(json.dumps({
                "validation_scope": result["validation_scope"],
                "task": args.task, "native_envs": NATIVE_ENVS, "episodes": NATIVE_ENVS,
                "full_suite": result["official_result"]["full_suite"],
                "result": result["result"], "phase_timings": result["phase_timings"],
                "end_to_end_wall_s": result["end_to_end_wall_s"],
            }, indent=2), flush=True)
        except Exception as exc:
            append_event(output_dir / "errors.jsonl", {
                "type": "group-smoke-error", "error_type": type(exc).__name__, "message": str(exc),
            })
            raise
        finally:
            recorder.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"group smoke failed: {error}", file=sys.stderr, flush=True)
        raise
