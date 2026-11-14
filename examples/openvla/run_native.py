# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Run the pinned official OpenVLA evaluator and export episode results without videos."""

from __future__ import annotations

import argparse
import functools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

UPSTREAM_REVISION = "c8f03f48af692657d3060c19588038c7220e9af9"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openvla-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--env-seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.episodes_per_task <= 50:
        parser.error("--episodes-per-task must be between 1 and 50")
    root = args.openvla_root.resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_REVISION:
        parser.error(f"OpenVLA checkout must be at {UPSTREAM_REVISION}")
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if dirty.strip():
        parser.error("OpenVLA checkout has modified tracked files")
    checkpoint = args.checkpoint.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(root))
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    from experiments.robot.libero import run_libero_eval as upstream

    cfg = upstream.GenerateConfig(
        pretrained_checkpoint=str(checkpoint), task_suite_name=args.suite,
        num_trials_per_task=args.episodes_per_task, seed=args.seed,
        center_crop=True, num_steps_wait=10, local_log_dir=str(output / "logs"),
        use_wandb=False,
    )
    state = {"task_id": -1, "steps": 0}
    records = []
    environments = []
    original_env = upstream.get_libero_env

    def get_env(*positional, **keywords):
        env, description = original_env(*positional, **keywords)
        env.seed(args.env_seed)
        environments.append(env)
        state["task_id"] += 1
        original_step = env.step

        @functools.wraps(original_step)
        def step(action):
            result = original_step(action)
            state["steps"] += 1
            return result

        env.step = step
        return env, description

    def record_episode(images, index, success, task_description, log_file=None):
        # Called by upstream once per episode instead of writing a replay video.
        steps = state["steps"] - cfg.num_steps_wait
        if steps < 1 or len(images) != steps:
            raise RuntimeError("Incomplete episode or failed action execution; reject this run")
        row = {"suite": args.suite, "task_id": state["task_id"],
               "episode_index": (index - 1) % args.episodes_per_task,
               "success": bool(success), "steps": steps}
        records.append(row)
        with (output / "native-episodes.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        state["steps"] = 0

    upstream.get_libero_env = get_env
    upstream.save_rollout_video = record_episode
    run = {"upstream_revision": revision, "suite": args.suite,
           "checkpoint": str(checkpoint), "seed": args.seed, "env_seed": args.env_seed,
           "episodes_per_task": args.episodes_per_task, "settle_steps": 10,
           "preprocessing": "CPU", "video": False, "complete": False,
           "command": sys.argv, "started": time.time()}
    run_path = output / "run.json"
    run_path.write_text(json.dumps(run, indent=2) + "\n")
    started = time.perf_counter()
    try:
        upstream.eval_libero(cfg)
        logs = list((output / "logs").glob("*.txt"))
        if any("Caught exception:" in p.read_text() for p in logs):
            raise RuntimeError("Official evaluator caught an exception; reject this run")
        if len(records) != 10 * args.episodes_per_task:
            raise RuntimeError("Incomplete ten-task evaluation")
        run.update(complete=True, elapsed_s=time.perf_counter() - started, finished=time.time())
        run_path.write_text(json.dumps(run, indent=2) + "\n")
    finally:
        for env in environments:
            env.close()


if __name__ == "__main__":
    main()
