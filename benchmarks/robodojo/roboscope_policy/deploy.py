# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""RoboDojo's evaluation loops for a RoboScope Policy.

``eval_one_episode`` runs one environment. ``eval_one_episode_batch`` runs RoboDojo's
lockstep multi-environment batch (``--num_envs N``), with one Kit episode per
environment and each environment's action chunks, step observations and episode end
handled exactly as in the single-environment loop.
"""

from __future__ import annotations

import contextlib
import os
import time
from collections import deque
from copy import deepcopy

import numpy as np

from roboscope_kit import Observation
from roboscope_kit.policy import EpisodeControl


def _observation(episode_id, step, raw, native_status=None):
    vision, state = raw["vision"], raw["state"]
    joints = []
    for side in ("left", "right"):
        arm = np.asarray(state[f"{side}_arm_joint_state"], dtype=np.float32).reshape(-1)
        grip = np.asarray(state[f"{side}_ee_joint_state"], dtype=np.float32).reshape(-1)
        if arm.shape != (6,) or grip.shape != (1,):
            raise ValueError("RoboDojo X5 observation requires six joints and one gripper per arm")
        joints.extend((arm, grip))
    instruction = raw.get("instruction", "")
    if isinstance(instruction, (list, tuple)):
        instruction = instruction[0] if instruction else ""
    metadata = {}
    if native_status is not None:
        other = deepcopy({key: value for key, value in raw.items()
                          if key not in ("vision", "state", "instruction")})
        additional = dict(other.get("additional_info") or {})
        additional["pyrualean"] = native_status
        other["additional_info"] = additional
        metadata["robodojo_native"] = {
            "state": deepcopy(state),
            "vision": deepcopy({name: {key: value for key, value in camera.items() if key != "color"}
                                if name in ("cam_head", "cam_left_wrist", "cam_right_wrist") else camera
                                for name, camera in vision.items()}),
            "other": other,
        }
    return Observation(
        episode_id=episode_id,
        step=step,
        features={
            "images.head": np.array(vision["cam_head"]["color"], dtype=np.uint8, order="C", copy=True),
            "images.left_wrist": np.array(
                vision["cam_left_wrist"]["color"], dtype=np.uint8, order="C", copy=True
            ),
            "images.right_wrist": np.array(
                vision["cam_right_wrist"]["color"], dtype=np.uint8, order="C", copy=True
            ),
            "state.joints": np.concatenate(joints),
            "state.ee_pose": np.concatenate([
                np.asarray(state[f"{side}_ee_pose"], dtype=np.float32).reshape(7)
                for side in ("left", "right")
            ]),
            "language.instruction": str(instruction or ""),
        },
        timestamp_s=None,
        metadata=metadata,
    )


def _quaternion(qxyz):
    xyz = np.asarray(qxyz, dtype=np.float64)
    squared_norm = float(np.dot(xyz, xyz))
    if squared_norm > 1.0 + 1e-5:
        raise ValueError("RoboDojo compact quaternion xyz squared norm exceeds one")
    if squared_norm > 1.0:
        xyz = xyz / np.sqrt(squared_norm)
        squared_norm = 1.0
    return np.concatenate(([np.sqrt(1.0 - squared_norm)], xyz)).astype(np.float32)


def _action(row, representation):
    row = np.asarray(row, dtype=np.float32)
    if row.shape != (14,) or not np.isfinite(row).all():
        raise ValueError("RoboDojo Policy actions must contain 14 finite values")
    if representation == "robodojo.x5-joint-v1":
        return {
            "left_arm_joint_state": row[:6],
            "left_ee_joint_state": row[6:7],
            "right_arm_joint_state": row[7:13],
            "right_ee_joint_state": row[13:14],
        }
    if representation == "robodojo.x5-ee-compact-quat-v1":
        return {
            "left_ee_pose": np.concatenate((row[:3], _quaternion(row[3:6]))),
            "left_ee_joint_state": row[6:7],
            "right_ee_pose": np.concatenate((row[7:10], _quaternion(row[10:13]))),
            "right_ee_joint_state": row[13:14],
        }
    raise ValueError(f"unsupported RoboDojo action representation: {representation!r}")


def _chunk(actions):
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != 14:
        raise ValueError("RoboDojo Policy returned an invalid action chunk")
    return actions


def _native_status(task_env):
    """The same episode metadata as PyRUA-Lean's native RoboDojo adapter."""
    try:
        from env.camera_manager.capture import render_sync  # noqa: F401
        pictures_behind = 0
    except Exception:  # noqa: BLE001 - older evaluation client
        pictures_behind = 1
    info = {"pictures_behind": pictures_behind}
    taken = getattr(task_env, "take_action_cnt", None)
    try:
        info["step"] = int(taken[0])
    except (TypeError, IndexError, ValueError):
        if isinstance(getattr(task_env, "episode_step", None), int):
            info["step"] = task_env.episode_step
    for name in ("step_lim", "episode_step_limit"):
        try:
            info["step_limit"] = int(getattr(task_env, name))
            break
        except (TypeError, ValueError, AttributeError):
            continue
    for name in ("task_name", "eval_seed"):
        value = getattr(task_env, name, None)
        if value is not None:
            info[name] = value
    with contextlib.suppress(TypeError, IndexError, ValueError, AttributeError):
        info["layout"] = int(task_env.env_seeds[0])
    return info


def _judge_as_it_stands(task_env):
    if task_env.is_episode_end() or not hasattr(task_env, "success"):
        return
    running = task_env.get_running_env_idx_list() if hasattr(task_env, "get_running_env_idx_list") else [0]
    for env_idx in running:
        task_env.success[env_idx] = False
    task_env.is_episode_end()


def _close_episode(task_env, model_client, episode_id, interactive):
    try:
        if interactive:
            _judge_as_it_stands(task_env)
            outcome = _native_status(task_env)
            try:
                outcome["success"] = bool(task_env.success[0])
            except (TypeError, IndexError, AttributeError):
                outcome["success"] = False
            model_client.episode_outcome(episode_id, outcome)
    finally:
        if interactive:
            model_client.end_episode(episode_id)
        else:
            model_client.end_episode()


def eval_one_episode(TASK_ENV, model_client):
    task = str(TASK_ENV.task_name)
    layout_id = int(TASK_ENV.env_seeds[0])
    representation = os.environ["ROBOSCOPE_ROBODOJO_ACTION_REPRESENTATION"]
    step_updates = os.environ.get("ROBOSCOPE_ROBODOJO_STEP_UPDATES") == "1"
    interactive = os.environ.get("ROBOSCOPE_ROBODOJO_INTERACTIVE") == "1"
    stride = int(os.environ.get("ROBOSCOPE_ROBODOJO_OBSERVATION_STRIDE", "1"))
    episode_id = model_client.begin_episode(task, layout_id)
    started = time.monotonic()
    try:
        while not TASK_ENV.is_episode_end():
            step = int(TASK_ENV.take_action_cnt[0])
            observation = _observation(episode_id, step, TASK_ENV.get_obs(),
                                       _native_status(TASK_ENV) if interactive else None)
            answer = model_client.act(observation)
            while isinstance(answer, EpisodeControl) and answer.kind == "wait":
                if not interactive:
                    raise RuntimeError("unexpected interactive policy response")
                time.sleep(0.05)
                answer = model_client.poll(episode_id)
            if isinstance(answer, EpisodeControl):
                if not interactive or answer.kind != "finish":
                    raise RuntimeError("unexpected interactive policy response")
                break
            actions = _chunk(answer)
            for index, action in enumerate(actions):
                TASK_ENV.take_action(_action(action, representation))
                if TASK_ENV.is_episode_end():
                    break
                if step_updates and index + 1 < len(actions):
                    # get_obs() also renders and writes the video frame, so it runs on every
                    # step; the stride only skips delivering the observation to the Policy.
                    raw = TASK_ENV.get_obs()
                    step = int(TASK_ENV.take_action_cnt[0])
                    if step % stride == 0:
                        model_client.observe(_observation(episode_id, step, raw))
    except BaseException as primary:
        try:
            _close_episode(TASK_ENV, model_client, episode_id, interactive)
        except BaseException as cleanup:
            raise primary from cleanup
        raise
    finally:
        TASK_ENV._roboscope_episode_elapsed_s = time.monotonic() - started
    if model_client.episode_id:
        _close_episode(TASK_ENV, model_client, episode_id, interactive)


def eval_one_episode_batch(TASK_ENV, model_client):
    """Run one lockstep batch of RoboDojo environments, one Kit episode each.

    Every running environment requests a new chunk when its previous one is used up
    and, with step updates, sends the observations between a chunk's actions, in the
    same order as ``eval_one_episode``. An environment's episode ends as soon as it
    terminates, while the others continue.
    """
    if TASK_ENV.num_envs > 1 and os.environ.get("ROBOSCOPE_ROBODOJO_INTERACTIVE") == "1":
        raise ValueError("interactive RoboDojo policies require one environment per worker")
    if TASK_ENV.num_envs == 1:
        eval_one_episode(TASK_ENV, model_client)
        return
    task = str(TASK_ENV.task_name)
    representation = os.environ["ROBOSCOPE_ROBODOJO_ACTION_REPRESENTATION"]
    step_updates = os.environ.get("ROBOSCOPE_ROBODOJO_STEP_UPDATES") == "1"
    stride = int(os.environ.get("ROBOSCOPE_ROBODOJO_OBSERVATION_STRIDE", "1"))
    model_client.end_episode()
    episodes = {}
    started = time.monotonic()
    try:
        for env_idx in TASK_ENV.get_running_env_idx_list():
            episodes[env_idx] = model_client.open_episode(task, int(TASK_ENV.env_seeds[env_idx]))
        chunks = {env_idx: deque() for env_idx in episodes}
        while not TASK_ENV.is_episode_end():
            running = [env_idx for env_idx in TASK_ENV.get_running_env_idx_list() if env_idx in episodes]
            if not running:
                raise RuntimeError("RoboDojo batch has running environments without an episode")
            observed = [env_idx for env_idx in running if not chunks[env_idx] or step_updates]
            rows = dict(zip(observed, TASK_ENV.get_obs_batch(observed), strict=True)) if observed else {}
            for env_idx in running:
                if env_idx not in rows:
                    continue
                step = int(TASK_ENV.take_action_cnt[env_idx])
                observation = _observation(episodes[env_idx], step, rows[env_idx])
                if not chunks[env_idx]:
                    chunks[env_idx].extend(_chunk(model_client.act(observation)))
                elif step % stride == 0:
                    model_client.observe(observation)
            TASK_ENV.take_action_batch(
                [_action(chunks[env_idx].popleft(), representation) for env_idx in running], running
            )
            for env_idx in running:
                if TASK_ENV.end_flag[env_idx]:
                    chunks[env_idx].clear()
                    model_client.end_episode(episodes.pop(env_idx))
    except BaseException as primary:
        try:
            model_client.end_episode()
        except BaseException as cleanup:
            raise primary from cleanup
        raise
    finally:
        TASK_ENV._roboscope_episode_elapsed_s = time.monotonic() - started
    model_client.end_episode()
