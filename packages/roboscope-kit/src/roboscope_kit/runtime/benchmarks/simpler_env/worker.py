# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Persistent SimplerEnv worker using an SDK Policy replica.

Each work item is one episode of one frozen variant (``tasks.Variant``). The worker
reproduces the pinned upstream evaluator, ``simpler_env/evaluation/maniskill2_evaluator.py``:

1. the variant's ``main_inference.py`` arguments are parsed by upstream ``get_args``, and
   the episode's robot pose and object pose (or predefined episode id) are taken in
   ``maniskill2_evaluator``'s loop order;
2. a fresh environment is built with upstream ``build_maniskill2_env`` and the evaluator's
   keywords (``obs_mode="rgbd"``, segmentation on, the overlay image, ``shader_dir="rt"``
   first when ray tracing is enabled), then ``env.reset(options=...)`` with no seed, so the
   environment's default seed sequence (main seed 2022) applies as it does upstream;
3. until the policy terminates on the final subtask or the time limit truncates: one
   Policy action per observation; a terminate signal on a non-final subtask calls
   ``env.advance_to_next_subtask()`` instead; ``env.step`` gets the first 7 values;
4. the episode succeeds when the last executed step returned ``terminated`` (the task's
   success predicate). Upstream names its video ``success``/``failure`` the same way.

The Policy receives the policy camera image (``get_image_from_maniskill2_obs_dict``) and
the current instruction; the episode metadata carries the variant's ``--policy-setup``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from roboscope_kit import EpisodeContext, Observation
from roboscope_kit.client import PolicyClient
from roboscope_kit.runtime.benchmark import EvaluationRecord, WorkerLoop, WorkItem, WorkResult

from .tasks import ACTION_DIM, PROTOCOL_VERSION, SUCCESS_CRITERION, SUITE_VARIANTS, Variant, episode_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--policy-socket", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--worker-state-dir", type=Path, required=True)
    parser.add_argument("--poll-interval", type=float, default=0.05)
    parser.add_argument("--recycle-after-records", type=int, default=None)
    return parser.parse_args()


def upstream_args(variant: Variant) -> argparse.Namespace:
    """Parse the variant exactly as ``main_inference.py`` parses its command line."""

    from simpler_env.evaluation.argparse import get_args

    saved = sys.argv
    sys.argv = ["main_inference.py", *variant.argv]
    try:
        return get_args()
    finally:
        sys.argv = saved


def expand_episodes(args: argparse.Namespace) -> list[dict[str, Any]]:
    """The episodes ``maniskill2_evaluator`` runs for ``args``, in its loop order."""

    episodes: list[dict[str, Any]] = []
    for robot_init_x in args.robot_init_xs:
        for robot_init_y in args.robot_init_ys:
            for robot_init_quat in args.robot_init_quats:
                pose = {
                    "robot_init_x": float(robot_init_x),
                    "robot_init_y": float(robot_init_y),
                    "robot_init_quat": [float(value) for value in robot_init_quat],
                }
                if args.obj_variation_mode == "xy":
                    for obj_init_x in args.obj_init_xs:
                        for obj_init_y in args.obj_init_ys:
                            episodes.append(
                                {**pose, "obj_init_x": float(obj_init_x), "obj_init_y": float(obj_init_y)}
                            )
                elif args.obj_variation_mode == "episode":
                    for obj_episode_id in range(args.obj_episode_range[0], args.obj_episode_range[1]):
                        episodes.append({**pose, "obj_episode_id": obj_episode_id})
                else:
                    raise ValueError(f"unsupported obj_variation_mode {args.obj_variation_mode!r}")
    return episodes


def build_env(args: argparse.Namespace):
    """``run_maniskill2_eval_single_episode``'s environment construction."""

    from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode

    with contextlib.redirect_stdout(sys.stderr):  # upstream prints the control mode
        control_mode = get_robot_control_mode(args.robot, args.policy_model)
    additional_env_build_kwargs = dict(args.additional_env_build_kwargs or {})
    kwargs = dict(
        obs_mode="rgbd",
        robot=args.robot,
        sim_freq=args.sim_freq,
        control_mode=control_mode,
        control_freq=args.control_freq,
        max_episode_steps=args.max_episode_steps,
        scene_name=args.scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=args.rgb_overlay_path,
    )
    if args.enable_raytracing:
        ray_tracing_dict = {"shader_dir": "rt"}
        ray_tracing_dict.update(additional_env_build_kwargs)
        additional_env_build_kwargs = ray_tracing_dict
    return build_maniskill2_env(args.env_name, **additional_env_build_kwargs, **kwargs)


def reset_options(episode: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {
        "robot_init_options": {
            "init_xy": np.array([episode["robot_init_x"], episode["robot_init_y"]]),
            "init_rot_quat": np.array(episode["robot_init_quat"]),
        }
    }
    if "obj_init_x" in episode:
        options["obj_init_options"] = {"init_xy": np.array([episode["obj_init_x"], episode["obj_init_y"]])}
    else:
        options["obj_init_options"] = {"episode_id": episode["obj_episode_id"]}
    return options


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


def run_episode(
    client: PolicyClient,
    variant: Variant,
    args: argparse.Namespace,
    index: int,
    episode: dict[str, Any],
) -> tuple[EvaluationRecord, float]:
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    record_id = episode_id(variant, index)
    started = time.perf_counter()
    env = build_env(args)
    built = time.perf_counter()
    obs, _ = env.reset(options=reset_options(episode))
    is_final_subtask = env.is_final_subtask()
    instruction = env.get_language_instruction()
    instructions = [instruction]
    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=args.obs_camera_name)
    first_image_sha256 = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
    client.reset(
        EpisodeContext(
            episode_id=record_id,
            task_id=variant.task,
            metadata={
                "protocol": PROTOCOL_VERSION,
                "suite": variant.suite,
                "variant": variant.variant_id,
                "policy_setup": variant.policy_setup,
                "robot": args.robot,
                "env_name": args.env_name,
                "control_freq": args.control_freq,
                "max_episode_steps": args.max_episode_steps,
            },
        )
    )
    reset_done = time.perf_counter()
    actions_digest = hashlib.sha256()
    inference_wait_s = 0.0
    chunk = np.empty((0, ACTION_DIM), dtype=np.float64)
    predicted_terminated = truncated = done = False
    subtask_advances = 0
    timestep = 0
    info: dict[str, Any] = {}
    try:
        while not (predicted_terminated or truncated):
            if not chunk.size:
                request = Observation(
                    episode_id=record_id,
                    step=timestep,
                    timestamp_s=time.time(),
                    features={"images.primary": image, "language.instruction": instruction},
                )
                requested = time.perf_counter()
                chunk = np.asarray(client.act(request).actions, dtype=np.float64)
                inference_wait_s += time.perf_counter() - requested
                if chunk.ndim == 1:
                    chunk = chunk.reshape(1, -1)
                if chunk.ndim != 2 or chunk.shape[0] == 0 or chunk.shape[1] != ACTION_DIM:
                    raise RuntimeError(
                        f"SimplerEnv Policy action shape is {chunk.shape}, expected (N, {ACTION_DIM})"
                    )
            action, chunk = chunk[0], chunk[1:]
            predicted_terminated = bool(action[7] > 0)
            if predicted_terminated and not is_final_subtask:
                predicted_terminated = False
                env.advance_to_next_subtask()
                subtask_advances += 1
            env_action = np.ascontiguousarray(action[:7], dtype=np.float64)
            actions_digest.update(env_action.tobytes())
            obs, _reward, done, truncated, info = env.step(env_action)
            new_instruction = env.get_language_instruction()
            if new_instruction != instruction:
                instruction = new_instruction
                instructions.append(instruction)
                chunk = chunk[:0]  # remaining actions were for the previous instruction
            is_final_subtask = env.is_final_subtask()
            image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=args.obs_camera_name)
            timestep += 1
    finally:
        # Like upstream's evaluator, the episode's environment is dropped, not closed.
        client.end_episode(record_id)
    finished = time.perf_counter()
    record = EvaluationRecord(
        record_id=record_id,
        success=bool(done),
        steps=timestep,
        elapsed_s=finished - started,
        metadata={
            "protocol": PROTOCOL_VERSION,
            "suite": variant.suite,
            "task": variant.task,
            "subtask": variant.subtask,
            "factor": variant.factor,
            "variant": variant.variant_id,
            "index": index,
            "episode": episode,
            "env_name": args.env_name,
            "scene_name": args.scene_name,
            "max_episode_steps": args.max_episode_steps,
            "instructions": instructions,
            "subtask_advances": subtask_advances,
            "terminated_by_policy": predicted_terminated,
            "truncated": bool(truncated),
            "episode_stats": _jsonable(info.get("episode_stats", {})),
            "actions_sha256": actions_digest.hexdigest(),
            "first_image_sha256": first_image_sha256,
            "success_criterion": SUCCESS_CRITERION,
            "env_build_s": built - started,
            "env_reset_s": reset_done - built,
        },
    )
    return record, inference_wait_s


def main() -> None:
    args = parse_args()
    if args.worker_id < 0 or args.poll_interval <= 0:
        raise ValueError("SimplerEnv worker numeric arguments must be positive")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    # main_inference.py clears DISPLAY before building environments.
    os.environ["DISPLAY"] = ""
    variants = SUITE_VARIANTS[args.suite]
    parsed: dict[str, tuple[argparse.Namespace, list[dict[str, Any]]]] = {}

    def evaluate(item: WorkItem, client: PolicyClient) -> WorkResult:
        variant = variants[str(item.payload["variant"])]
        if variant.variant_id not in parsed:
            upstream = upstream_args(variant)
            episodes = expand_episodes(upstream)
            if len(episodes) != variant.episodes:
                raise RuntimeError(
                    f"SimplerEnv variant {variant.variant_id} expands to {len(episodes)} episodes, "
                    f"expected {variant.episodes}"
                )
            parsed[variant.variant_id] = (upstream, episodes)
        upstream, episodes = parsed[variant.variant_id]
        started = time.perf_counter()
        records = []
        inference_wait_s = 0.0
        for index in item.payload["indices"]:
            record, wait = run_episode(client, variant, upstream, int(index), episodes[int(index)])
            records.append(record)
            inference_wait_s += wait
        return WorkResult(
            item_id=item.item_id,
            group_id=item.group_id,
            worker_id=args.worker_id,
            records=tuple(records),
            metadata={
                "suite": args.suite,
                "variant": variant.variant_id,
                "elapsed_s": time.perf_counter() - started,
                "inference_wait_s": inference_wait_s,
            },
        )

    with PolicyClient(args.policy_socket) as client:
        WorkerLoop(
            worker_id=args.worker_id,
            queue_dir=args.queue_dir,
            state_dir=args.worker_state_dir,
            poll_interval_s=args.poll_interval,
            recycle_after_records=args.recycle_after_records,
        ).run(lambda item: evaluate(item, client))


if __name__ == "__main__":
    main()
