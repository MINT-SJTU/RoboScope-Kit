# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Run SimplerEnv's unmodified ``main_inference.py`` with read-only per-episode logging.

Usage (``run_native.py`` calls it once per official-script invocation, from the native root)::

    python native_hooks.py --record episodes.jsonl [--write-videos] -- simpler_env/main_inference.py ARGS...

The hooks wrap, without changing their behavior:

- ``maniskill2_evaluator.run_maniskill2_eval_single_episode``: the episode's arguments,
  its success, and its wall time;
- ``maniskill2_evaluator.build_maniskill2_env``: the returned environment's ``step`` (step
  count, a SHA-256 over every float64 action passed to ``env.step`` and an 8-hex-digit
  digest of each step's action, the last
  ``truncated`` and ``episode_stats``) and ``advance_to_next_subtask``;
- ``maniskill2_evaluator.get_image_from_maniskill2_obs_dict``: a SHA-256 of the episode's
  first policy image.

Without ``--write-videos`` the evaluator's ``write_video`` and ``RT1Inference.visualize_epoch``
(a matplotlib figure per episode) are skipped; they only write files and do not feed back
into the episode. Each episode appends one JSON line to ``--record``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class EpisodeLog:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.actions = hashlib.sha256()
        self.step_digests: list[str] = []
        self.truncated = False
        self.terminated = False
        self.episode_stats: dict[str, Any] = {}
        self.advances = 0
        self.first_image_sha256: str | None = None
        self.instructions: list[str] = []


def install(record: Path, write_videos: bool) -> None:
    from simpler_env.evaluation import maniskill2_evaluator as evaluator
    from simpler_env.policies.rt1 import rt1_model

    log = EpisodeLog()
    build_env = evaluator.build_maniskill2_env
    get_image = evaluator.get_image_from_maniskill2_obs_dict
    run_episode = evaluator.run_maniskill2_eval_single_episode

    def build_maniskill2_env(env_name, **kwargs):
        env = build_env(env_name, **kwargs)
        step = env.step
        advance = env.advance_to_next_subtask
        instruction = env.get_language_instruction

        def logged_step(action):
            raw = np.ascontiguousarray(action, dtype=np.float64).tobytes()
            log.actions.update(raw)
            log.step_digests.append(hashlib.sha256(raw).hexdigest()[:8])
            obs, reward, done, truncated, info = step(action)
            log.steps += 1
            log.terminated = bool(done)
            log.truncated = bool(truncated)
            log.episode_stats = _jsonable(info.get("episode_stats", {}))
            return obs, reward, done, truncated, info

        def logged_advance():
            log.advances += 1
            return advance()

        def logged_instruction(**kwargs):
            text = instruction(**kwargs)
            if not log.instructions or log.instructions[-1] != text:
                log.instructions.append(text)
            return text

        env.step = logged_step
        env.advance_to_next_subtask = logged_advance
        env.get_language_instruction = logged_instruction
        return env

    def get_image_from_maniskill2_obs_dict(env, obs, camera_name=None):
        image = get_image(env, obs, camera_name=camera_name)
        if log.first_image_sha256 is None:
            log.first_image_sha256 = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
        return image

    def run_maniskill2_eval_single_episode(**kwargs):
        log.reset()
        started = time.perf_counter()
        success = run_episode(**kwargs)
        elapsed = time.perf_counter() - started
        entry = {
            "env_name": kwargs["env_name"],
            "scene_name": kwargs["scene_name"],
            "robot": kwargs["robot_name"],
            "control_freq": kwargs["control_freq"],
            "sim_freq": kwargs["sim_freq"],
            "max_episode_steps": kwargs["max_episode_steps"],
            "rgb_overlay_path": kwargs["rgb_overlay_path"],
            "enable_raytracing": kwargs["enable_raytracing"],
            "additional_env_build_kwargs": _jsonable(kwargs["additional_env_build_kwargs"] or {}),
            "robot_init_x": float(kwargs["robot_init_x"]),
            "robot_init_y": float(kwargs["robot_init_y"]),
            "robot_init_quat": [float(value) for value in kwargs["robot_init_quat"]],
            "obj_init_x": None if kwargs.get("obj_init_x") is None else float(kwargs["obj_init_x"]),
            "obj_init_y": None if kwargs.get("obj_init_y") is None else float(kwargs["obj_init_y"]),
            "obj_episode_id": kwargs.get("obj_episode_id"),
            "success": bool(success),
            "steps": log.steps,
            "terminated_at_last_step": log.terminated,
            "truncated": log.truncated,
            "subtask_advances": log.advances,
            "instructions": log.instructions,
            "episode_stats": log.episode_stats,
            "actions_sha256": log.actions.hexdigest(),
            "step_action_digests": log.step_digests,
            "first_image_sha256": log.first_image_sha256,
            "elapsed_s": elapsed,
        }
        with record.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        return success

    evaluator.build_maniskill2_env = build_maniskill2_env
    evaluator.get_image_from_maniskill2_obs_dict = get_image_from_maniskill2_obs_dict
    evaluator.run_maniskill2_eval_single_episode = run_maniskill2_eval_single_episode
    if not write_videos:
        evaluator.write_video = lambda *args, **kwargs: None
        rt1_model.RT1Inference.visualize_epoch = lambda self, *args, **kwargs: None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--write-videos", action="store_true")
    parser.add_argument("script")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    args = options.args[1:] if options.args[:1] == ["--"] else options.args
    install(options.record, options.write_videos)
    sys.argv = [options.script, *args]
    runpy.run_path(options.script, run_name="__main__")


if __name__ == "__main__":
    main()
