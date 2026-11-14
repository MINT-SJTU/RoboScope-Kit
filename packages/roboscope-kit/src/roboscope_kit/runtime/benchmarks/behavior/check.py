# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Check a BEHAVIOR runtime: pinned data, a headless launch, and one rendered instance load.

Runs in the BEHAVIOR runtime environment (``build-runtime.sh`` and ``--self-test`` call it)::

    PYTHONHASHSEED=0 OMNIGIBSON_DATA_PATH=<data> OMNI_KIT_ACCEPT_EULA=YES \
        python -m roboscope_kit.runtime.benchmarks.behavior.check --report <file> [--task turning_on_radio]

It checks the dataset versions, the challenge task metadata against the frozen digests, and the
decryption key; then it builds the worker's evaluator for the task (scene load), loads public test
instance 301 with the evaluator's own load path, and checks that the first observation maps to
the SDK features: every camera rendered a non-empty RGB image and finite depth, and the
proprioception and camera poses have the frozen sizes. No action is executed. Writes a JSON
report and exits non-zero on any mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .tasks import (
    BEHAVIOR_1K_COMMIT,
    CAMERAS,
    MAX_STEPS,
    PYTHON_HASH_SEED,
    QUICK_TEST_TASK,
    TASK_METADATA_SHA256,
    TASK_NAMES,
    instance_id,
    require_python_hash_seed,
)

DATASET_VERSIONS = {"behavior-1k-assets": "3.9.0", "omnigibson-robot-assets": "3.8.2"}


def check_data(data_path: Path) -> dict:
    report: dict = {"data_path": str(data_path)}
    for dataset, version in DATASET_VERSIONS.items():
        actual = (data_path / dataset / "VERSION").read_text(encoding="utf-8").strip()
        if actual != version:
            raise RuntimeError(f"{dataset} is version {actual}, expected {version}")
        report[dataset] = actual
    instances = data_path / "2026-challenge-task-instances"
    for name, expected in TASK_METADATA_SHA256.items():
        actual = hashlib.sha256((instances / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"2026-challenge-task-instances/{name} has SHA-256 {actual}, expected {expected}"
            )
    report["task_metadata_sha256"] = dict(TASK_METADATA_SHA256)
    if not (data_path / "omnigibson.key").is_file():
        raise RuntimeError(f"the BEHAVIOR dataset key is missing: {data_path / 'omnigibson.key'}")
    return report


def instance_report(evaluator, task: str, started: float) -> dict:
    from .worker import build_observation, episode_step_limit

    loaded = time.perf_counter()
    evaluator.load_batch({0: instance_id(0)})
    observation = build_observation(evaluator._batch_obs(), f"{task}-self-test", 0)
    report = {
        "task": task,
        "instance_id": instance_id(0),
        "max_steps": episode_step_limit(evaluator),
        "features": {
            name: {"dtype": value.dtype.name, "shape": list(value.shape)}
            for name, value in sorted(observation.features.items())
        },
        "scene_load_s": round(loaded - started, 3),
        "instance_load_s": round(time.perf_counter() - loaded, 3),
    }
    if report["max_steps"] != MAX_STEPS[task]:
        raise RuntimeError(f"{task} max_steps is {report['max_steps']}, expected {MAX_STEPS[task]}")
    for role, _, resolution in CAMERAS:
        rgb = observation.features[f"images.{role}"]
        depth = observation.features[f"state.{role}_depth"]
        if rgb.shape != (*resolution, 3) or not rgb.any():
            raise RuntimeError(f"camera {role} rendered an empty {rgb.shape} image")
        if depth.shape != resolution or not np.isfinite(depth).all():
            raise RuntimeError(f"camera {role} returned invalid depth {depth.shape}")
        report["features"][f"images.{role}"]["mean"] = round(float(rgb.mean()), 3)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=TASK_NAMES, default=QUICK_TEST_TASK)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--data-only", action="store_true", help="check the data without launching")
    args = parser.parse_args()
    require_python_hash_seed()
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    report: dict = {"behavior_1k_commit": BEHAVIOR_1K_COMMIT, "python_hash_seed": PYTHON_HASH_SEED}
    report["data"] = check_data(Path(os.environ["OMNIGIBSON_DATA_PATH"]))
    evaluator = None
    if not args.data_only:
        from .worker import create_evaluator

        started = time.perf_counter()
        evaluator = create_evaluator(args.task, write_video=False)
        report["instance"] = instance_report(evaluator, args.task, started)
    text = json.dumps(report, indent=2, sort_keys=True)
    args.report.write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    if evaluator is not None:
        # Only after a passing check: closes the simulator and removes OmniGibson's temporary
        # directory (decrypted assets). The application may end the process here, with status 0.
        evaluator.__exit__(None, None, None)
    return 0


if __name__ == "__main__":
    # The simulator application may not return control when the interpreter exits.
    try:
        code = main()
    except BaseException:
        traceback.print_exc()
        code = 1
        if "omnigibson" in sys.modules:
            # Remove OmniGibson's temporary directory (decrypted assets) and keep the failing status.
            shutil.rmtree(sys.modules["omnigibson"].tempdir, ignore_errors=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
