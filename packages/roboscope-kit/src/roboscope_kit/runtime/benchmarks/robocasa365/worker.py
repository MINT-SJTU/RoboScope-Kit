# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""RoboCasa365 evaluation with native five-slot reset and termination groups.

Every task retains its constructor RNG streams across sequential rollouts. The
Policy receives the last two physical-step observations, with the initial frame
duplicated. Official Gym converters own image orientation and controller logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import deque
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from roboscope_kit import EpisodeContext, Observation, PolicyClient, profiling
from roboscope_kit.runtime.benchmark import EvaluationRecord, WorkerLoop, WorkItem, WorkQueue, WorkResult

from .protocol import (
    ACTION_DIMENSION,
    ACTION_STEPS,
    CONSTRUCTOR_SEED,
    EPISODES_PER_TASK,
    IMAGE_KEYS,
    NATIVE_ENVS,
    OBSERVATION_STEPS,
    ROLLOUT_LABEL_BASE,
    SCENE_SPLIT,
    STATE_DIMENSIONS,
    TASKS_BY_NAME,
    Task,
    validate_upstream,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--policy-socket", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--worker-state-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=0.05)
    return parser.parse_args()


def create_environment(task: Task, slot: int):
    import gymnasium as gym
    import robocasa.wrappers.gym_wrapper  # noqa: F401 -- registers pinned Gym environments

    return gym.make(
        f"robocasa/{task.name}", split=SCENE_SPLIT, seed=CONSTRUCTOR_SEED + slot,
        camera_widths=256, camera_heights=256,
    )


def episode_id(task: str, index: int) -> str:
    return f"{task}-rollout-{ROLLOUT_LABEL_BASE + index}"


def environment_range(name: str):
    """Use the pinned Kit's NVTX range only under an active profiler injection."""
    return profiling._range(name) if profiling.ENABLED else nullcontext()


def build_observation(identifier: str, step: int, history: deque) -> Observation:
    if len(history) != OBSERVATION_STEPS:
        raise RuntimeError("RoboCasa365 requires two consecutive observation frames")
    features: dict[str, Any] = {}
    for feature, key in IMAGE_KEYS.items():
        value = np.stack([np.asarray(frame[key]) for frame in history])
        if value.shape != (2, 256, 256, 3) or value.dtype != np.uint8:
            raise RuntimeError(f"RoboCasa365 image {key}: invalid {value.shape}/{value.dtype}")
        features[feature] = np.ascontiguousarray(value)
    for key, dimension in STATE_DIMENSIONS.items():
        value = np.stack([np.asarray(frame[f"state.{key}"]) for frame in history])
        if value.shape != (2, dimension) or not np.isfinite(value).all():
            raise RuntimeError(f"RoboCasa365 state {key}: invalid values or shape {value.shape}")
        features[f"state.{key}"] = np.ascontiguousarray(value, dtype=np.float32)
    instruction = history[-1]["annotation.human.task_description"]
    if not isinstance(instruction, str) or not instruction.strip():
        raise RuntimeError("RoboCasa365 instruction is missing or blank")
    features["language.instruction"] = instruction
    return Observation(identifier, step, features, time.time())


def validate_actions(value: np.ndarray) -> np.ndarray:
    actions = np.asarray(value)
    if actions.shape != (ACTION_STEPS, ACTION_DIMENSION) or not np.isfinite(actions).all():
        raise RuntimeError(f"RoboCasa365 requires finite (8, 12) actions; received {actions.shape}")
    return actions


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported scenario metadata type {type(value).__name__}")


def reset_fingerprint(environment, observation: Observation) -> dict[str, Any]:
    native = environment.unwrapped.env
    state = np.asarray(native.sim.get_state().flatten(), dtype=np.float64)
    meta = json.dumps(native.get_ep_meta(), sort_keys=True, default=_json_default, separators=(",", ":"))
    features = hashlib.sha256()
    for key, value in sorted(observation.features.items()):
        features.update(key.encode())
        features.update(value.encode() if isinstance(value, str) else value.tobytes())
    return {
        "initial_state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
        "scenario_metadata_sha256": hashlib.sha256(meta.encode()).hexdigest(),
        "initial_observation_sha256": features.hexdigest(),
    }


def save_first_batch(args: argparse.Namespace, task: str, observations: tuple, chunks: tuple) -> None:
    path = args.output_dir / "evidence" / f"worker-{args.worker_id:03d}" / f"{task}-first-batch.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {
        key: np.stack([observation.features[key] for observation in observations])
        for key in observations[0].features
    }
    values["actions"] = np.stack([chunk.actions for chunk in chunks])
    values["task_name"] = np.asarray(task)
    values["episode_ids"] = np.asarray([observation.episode_id for observation in observations])
    np.savez(path, **values)


def run_group(
    args: argparse.Namespace, task: Task, environments: list,
    stream_episode: int, client: PolicyClient,
) -> tuple[list[EvaluationRecord], dict[str, float]]:
    from robocasa.utils.env_utils import convert_action

    count = len(environments)
    indices = [stream_episode * count + slot for slot in range(count)]
    identifiers = [episode_id(task.name, index) for index in indices]
    histories: list[deque] = []
    fingerprints = []
    reset_started = time.perf_counter()
    with environment_range("environment.reset"):
        for slot, environment in enumerate(environments):
            # Native labels are not reset seeds: continue the constructor stream.
            native, _ = environment.reset()
            history = deque((native, native), maxlen=OBSERVATION_STEPS)
            histories.append(history)
            fingerprints.append(reset_fingerprint(
                environment, build_observation(identifiers[slot], 0, history)
            ))
    reset_s = time.perf_counter() - reset_started
    started = time.perf_counter()
    steps = [0] * count
    successes = [False] * count
    dones = [False] * count
    recent_success = [deque(maxlen=OBSERVATION_STEPS) for _ in environments]
    inference_s = 0.0
    stepping_s = 0.0
    video_s = 0.0
    termination = "horizon"
    with ExitStack() as stack:
        writers = []
        for identifier in identifiers:
            if args.save_videos:
                import imageio.v2 as imageio

                path = args.output_dir / "videos" / f"worker-{args.worker_id:03d}" / f"{identifier}.mp4"
                path.parent.mkdir(parents=True, exist_ok=True)
                writer = imageio.get_writer(
                    path, fps=10, codec="libx264", quality=None, ffmpeg_params=["-crf", "22"]
                )
                stack.callback(writer.close)
                writers.append(writer)
            else:
                writers.append(None)
        reset_ids: list[str] = []
        if profiling.ENABLED:
            profiling.episode_start()
        try:
            for slot, identifier in enumerate(identifiers):
                context = EpisodeContext(identifier, task.name, {
                    "episode_index": indices[slot], "slot": slot,
                    "stream_episode": stream_episode, "constructor_seed": CONSTRUCTOR_SEED + slot,
                    "scene_split": SCENE_SPLIT, "native_envs": count,
                })
                if profiling.ENABLED:
                    # PolicyClient's automatic range assumes serial episodes.
                    # Keep one capture open for the entire native group instead.
                    client._request("reset", {"context": context.to_dict()})
                else:
                    client.reset(context)
                reset_ids.append(identifier)
            while not all(dones):
                observations = tuple(
                    build_observation(identifier, steps[slot], histories[slot])
                    for slot, identifier in enumerate(identifiers)
                )
                inference_started = time.perf_counter()
                # Native inference includes done slots; their wrapper ignores actions.
                chunks = client.act_batch(observations)
                inference_s += time.perf_counter() - inference_started
                if len(chunks) != count:
                    raise RuntimeError("RoboCasa365 Policy returned an incomplete action batch")
                if stream_episode == 0 and not any(steps):
                    save_first_batch(args, task.name, observations, chunks)
                for slot, chunk in enumerate(chunks):
                    actions = validate_actions(chunk.actions)
                    for action in actions:
                        if dones[slot]:
                            break
                        step_started = time.perf_counter()
                        with environment_range("environment.step"):
                            native, _, terminated, truncated, info = environments[slot].step(
                                convert_action(action)
                            )
                        stepping_s += time.perf_counter() - step_started
                        histories[slot].append(native)
                        steps[slot] += 1
                        success = bool(info["success"])
                        recent_success[slot].append(success)
                        successes[slot] |= success
                        dones[slot] = bool(terminated or truncated or steps[slot] >= task.horizon)
                        # Native VideoRecordingWrapper starts its counter at 1
                        # after reset, so a 20 Hz / 10 fps run records steps 1, 3, ...
                        if writers[slot] is not None and steps[slot] % 2 == 1:
                            video_started = time.perf_counter()
                            writers[slot].append_data(native["video.robot0_agentview_right"])
                            video_s += time.perf_counter() - video_started
                # Native info['success'][0] is FIRST of its latest two step values.
                if all(values and values[0] for values in recent_success):
                    termination = "native_group_success"
                    break
            if termination != "native_group_success" and any(step < task.horizon for step in steps):
                termination = "simulator_done"
        finally:
            try:
                for identifier in reset_ids:
                    if profiling.ENABLED:
                        client._request("end_episode", {"episode_id": identifier})
                    else:
                        client.end_episode(identifier)
            finally:
                if profiling.ENABLED:
                    profiling.episode_end()
    elapsed_s = time.perf_counter() - started
    records = [
        EvaluationRecord(
            identifiers[slot], successes[slot], steps[slot], elapsed_s,
            {
                "task": task.name, "split": task.split, "scene_split": SCENE_SPLIT,
                "episode_index": indices[slot], "rollout_label": ROLLOUT_LABEL_BASE + indices[slot],
                "slot": slot, "stream_episode": stream_episode,
                "constructor_seed": CONSTRUCTOR_SEED + slot,
                "horizon": task.horizon, "termination": termination,
                "group_elapsed_s": elapsed_s, "group_reset_s": reset_s,
                **fingerprints[slot],
            },
        )
        for slot in range(count)
    ]
    return records, {
        "reset_s": reset_s, "inference_wait_s": inference_s,
        "environment_step_s": stepping_s, "video_write_s": video_s,
    }


def run_task(args: argparse.Namespace, item: WorkItem) -> WorkResult:
    task = TASKS_BY_NAME[str(item.payload["task"])]
    episodes = int(item.payload["episodes"])
    count = int(item.payload["native_envs"])
    if count not in (1, NATIVE_ENVS) or not 1 <= episodes <= EPISODES_PER_TASK or episodes % count:
        raise ValueError("invalid RoboCasa365 task workload")
    environments = []
    records = []
    timings = {
        "reset_s": 0.0, "inference_wait_s": 0.0, "environment_step_s": 0.0, "video_write_s": 0.0,
    }
    started = time.perf_counter()
    try:
        with environment_range("environment.create"):
            for slot in range(count):
                environments.append(create_environment(task, slot))
                context = environments[-1].unwrapped.env.sim._render_context_offscreen
                if getattr(context, "_roboscope_owning_context_cleanup", None) != 2:
                    raise RuntimeError("rebuild RoboCasa365 runtime with the owning-context cleanup patches")
        timings["environment_create_s"] = time.perf_counter() - started
        with PolicyClient(args.policy_socket, timeout_s=300) as client:
            for group in range(episodes // count):
                group_records, group_timing = run_group(args, task, environments, group, client)
                records.extend(group_records)
                for key, value in group_timing.items():
                    timings[key] += value
                WorkQueue(args.queue_dir).publish_partial(item, WorkResult(
                    item.item_id, item.group_id, args.worker_id, tuple(records),
                    {"task": task.name, "timings": dict(timings)},
                ))
    finally:
        close_started = time.perf_counter()
        for environment in environments:
            environment.close()
        timings["environment_close_s"] = time.perf_counter() - close_started
    timings["task_elapsed_s"] = time.perf_counter() - started
    return WorkResult(
        item.item_id, item.group_id, args.worker_id, tuple(records),
        {"task": task.name, "native_envs": count, "timings": timings},
    )


def main() -> None:
    args = parse_args()
    validate_upstream()
    WorkerLoop(
        worker_id=args.worker_id, queue_dir=args.queue_dir,
        state_dir=args.worker_state_dir, poll_interval_s=args.poll_interval,
    ).run(lambda item: run_task(args, item))


if __name__ == "__main__":
    main()
