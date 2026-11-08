# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Check an RLBench runtime: frozen test episodes, headless launch, and one rendered reset.

Runs in the RLBench runtime environment (``build-runtime.sh`` and ``--self-test`` call it)::

    PYTHONHASHSEED=0 python -m roboscope_kit.runtime.benchmarks.rlbench.check \
        --test-set <root> --xvfb <Xvfb> [--task open_drawer ...]

For every requested task it recomputes the test-episode digest and compares it with the
frozen one, then launches CoppeliaSim on a private Xvfb display, loads every requested task and
checks its variation count against the frozen one, resets the first requested task to its test
episode 0 with the worker's own reset path, and checks that every camera
rendered a non-empty image and a finite point cloud. Prints a JSON report (CoppeliaSim also
writes its own messages to stdout; ``--report`` writes the report alone to a file) and exits
non-zero on any mismatch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from .tasks import (
    CAMERAS,
    EPISODES_PER_TASK,
    IMAGE_SIZE,
    PYTHON_HASH_SEED,
    TASK_NAMES,
    TASKS_BY_NAME,
    require_python_hash_seed,
)
from .test_set import load_test_episode, task_digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test-set", type=Path, required=True)
    parser.add_argument("--xvfb", type=Path, required=True)
    parser.add_argument("--task", action="append", choices=TASK_NAMES)
    parser.add_argument("--report", type=Path, help="also write the JSON report here")
    args = parser.parse_args()
    require_python_hash_seed()
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    tasks = args.task or list(TASK_NAMES)
    report: dict = {"python_hash_seed": PYTHON_HASH_SEED, "test_episodes": {}}
    mismatched = []
    for task in tasks:
        actual = task_digest(args.test_set, task)
        report["test_episodes"][task] = actual
        if actual != TASKS_BY_NAME[task].test_episodes_sha256:
            mismatched.append(task)
    if mismatched:
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"PerAct test-episode digests differ for: {', '.join(mismatched)}", file=sys.stderr)
        return 1

    from .worker import LoadedTask, create_environment, reset_to_test_episode, start_xvfb, task_class

    started = time.perf_counter()
    start_xvfb(args.xvfb)
    environment = create_environment(args.test_set)
    try:
        launched = time.perf_counter()
        # Each task's variation count must be the frozen one, and cover its test episodes.
        report["variations"] = {}
        for task in reversed(tasks):
            task_env = environment.get_task(task_class(task))
            count = int(task_env.variation_count())
            report["variations"][task] = count
            used = {
                load_test_episode(args.test_set, task, index).variation for index in range(EPISODES_PER_TASK)
            }
            if count != TASKS_BY_NAME[task].variations or max(used) >= count:
                raise RuntimeError(
                    f"RLBench task {task} has {count} variations, expected {TASKS_BY_NAME[task].variations}"
                )
        loaded = time.perf_counter()
        # The last task loaded is the first requested one; this is its first reset after loading.
        observation, instruction, diagnostics = reset_to_test_episode(
            LoadedTask(task_env), load_test_episode(args.test_set, tasks[0], 0)
        )
        for camera in CAMERAS:
            rgb = np.asarray(getattr(observation, f"{camera}_rgb"))
            point_cloud = np.asarray(getattr(observation, f"{camera}_point_cloud"))
            if rgb.shape != (*IMAGE_SIZE, 3) or not rgb.any():
                raise RuntimeError(f"camera {camera} rendered an empty {rgb.shape} image")
            if point_cloud.shape != (*IMAGE_SIZE, 3) or not np.isfinite(point_cloud).all():
                raise RuntimeError(f"camera {camera} returned an invalid point cloud")
        report["reset"] = {
            "task": tasks[0],
            "episode": 0,
            "instruction": instruction,
            "initial_scene_position_error_m": diagnostics.initial_scene_position_error_m,
            "launch_s": round(launched - started, 3),
            "load_tasks_s": round(loaded - launched, 3),
            "reset_s": round(time.perf_counter() - loaded, 3),
        }
    finally:
        environment.shutdown()
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.report is not None:
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
