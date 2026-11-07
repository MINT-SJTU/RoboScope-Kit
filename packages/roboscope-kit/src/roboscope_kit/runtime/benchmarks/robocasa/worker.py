# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Persistent RoboCasa GR1 Tabletop worker using an SDK Policy replica."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from roboscope_kit import EpisodeContext, Observation, PolicyClient
from roboscope_kit.runtime.benchmark import (
    EvaluationRecord,
    WorkerLoop,
    WorkItem,
    WorkQueue,
    WorkResult,
)

ACTION_DIMENSION = 29
STATE_KEYS = ("state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand", "state.waist")
IMAGE_KEYS = ("video.ego_view", "video.ego_view_bg_crop_pad_res256_freq20")
LANGUAGE_KEY = "annotation.human.coarse_action"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--policy-socket", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--worker-state-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--action-steps", type=int, default=12)
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument("--disable-video", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--defer-observations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--poll-interval", type=float, default=0.05)
    return parser.parse_args()


def _latest(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim < 2 or array.shape[0] != 1:
        raise RuntimeError(f"RoboCasa observation must have one environment, received {array.shape}")
    return np.asarray(array[0, -1] if array.ndim >= 3 else array[0])


def _instruction(value: Any) -> str:
    current = value
    while isinstance(current, (list, tuple, np.ndarray)):
        if len(current) == 0:
            raise RuntimeError("RoboCasa instruction cannot be empty")
        current = current[0]
    instruction = str(current).strip()
    if not instruction:
        raise RuntimeError("RoboCasa instruction cannot be blank")
    return instruction


def build_observation(episode_id: str, step: int, native: dict[str, Any]) -> Observation:
    image_key = next((name for name in IMAGE_KEYS if name in native), None)
    if image_key is None:
        raise RuntimeError("RoboCasa observation is missing the ego-view image")
    missing_state = [name for name in STATE_KEYS if name not in native]
    if missing_state:
        raise RuntimeError(f"RoboCasa observation is missing state features: {', '.join(missing_state)}")
    state = np.concatenate([_latest(native[name]).reshape(-1) for name in STATE_KEYS])
    if state.shape != (ACTION_DIMENSION,):
        raise RuntimeError(f"RoboCasa state shape is {state.shape}, expected {(ACTION_DIMENSION,)}")
    return Observation(
        episode_id=episode_id,
        step=step,
        timestamp_s=time.time(),
        features={
            "images.primary": np.ascontiguousarray(_latest(native[image_key]), dtype=np.uint8),
            "state.proprioception": np.ascontiguousarray(state, dtype=np.float32),
            "language.instruction": _instruction(native[LANGUAGE_KEY]),
        },
    )


def native_action(chunk: np.ndarray, action_steps: int) -> dict[str, np.ndarray]:
    actions = np.asarray(chunk, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIMENSION:
        raise RuntimeError(
            f"RoboCasa Policy action shape is {actions.shape}, expected (N, {ACTION_DIMENSION})"
        )
    if actions.shape[0] < action_steps:
        raise RuntimeError(
            f"RoboCasa Policy returned {actions.shape[0]} actions, but {action_steps} are required"
        )
    selected = np.ascontiguousarray(actions[:action_steps])
    return {
        "action.left_arm": selected[None, :, 0:7],
        "action.right_arm": selected[None, :, 7:14],
        "action.left_hand": selected[None, :, 14:20],
        "action.right_hand": selected[None, :, 20:26],
        "action.waist": selected[None, :, 26:29],
    }


def create_seeded_environment(config: Any, seed: int):
    """Seed native construction while preserving the pinned upstream wrapper stack."""
    import gymnasium as gym
    from examples.simBenchmarks.Robocasa_tabletop.eval_files.simulation_env import (
        _create_single_env,
    )

    # Gym accepts an EnvSpec as well as an id. Copy it so this task's seed never
    # mutates the process-wide registry used by subsequent work items.
    spec = deepcopy(gym.spec(config.env_name))
    spec.kwargs["seed"] = seed
    np.random.seed(seed)
    return _create_single_env(config=replace(config, env_name=spec), idx=0)


def run_task(args: argparse.Namespace, item: WorkItem) -> WorkResult:
    import gymnasium as gym
    from examples.simBenchmarks.Robocasa_tabletop.eval_files.simulation_env import (
        MultiStepConfig,
        SimulationConfig,
        VideoConfig,
    )

    from .rendering import enable_chunk_observations

    task_name = str(item.payload["task_name"])
    env_name = str(item.payload["env_name"])
    episodes = int(item.payload["episodes"])
    video_dir = None
    if not args.disable_video:
        video_dir = str(args.output_dir / "video" / f"worker-{args.worker_id:03d}" / task_name)
    config = SimulationConfig(
        env_name=env_name,
        n_episodes=1,
        n_envs=1,
        video=VideoConfig(video_dir=video_dir),
        multistep=MultiStepConfig(
            n_action_steps=args.action_steps,
            max_episode_steps=args.max_episode_steps,
        ),
    )
    # Seeds depend only on the manifest seed, task and episode, never on which
    # Worker claims the task or in which order, so any K/g reproduces the scenes.
    task_seed = int.from_bytes(hashlib.sha256(task_name.encode()).digest()[:4], "big")
    construction_seed = (args.seed + task_seed) % 2**32
    initialize_started = time.perf_counter()
    single_environment = create_seeded_environment(config, construction_seed)
    rendering = None
    if args.defer_observations and args.disable_video:
        rendering = enable_chunk_observations(single_environment)
    environment = gym.vector.SyncVectorEnv([lambda: single_environment])
    initialize_s = time.perf_counter() - initialize_started
    records: list[EvaluationRecord] = []
    queue = WorkQueue(args.queue_dir)
    started = time.perf_counter()
    try:
        for episode_index in range(episodes):
            # Seed both upstream RNGs: the Gym wrapper seeds numpy, while robosuite
            # owns a separate Generator used for scene and object placement on reset.
            episode_seed = (construction_seed + episode_index) % 2**32
            native = single_environment.unwrapped.env
            native.seed = episode_seed
            native.rng = np.random.default_rng(episode_seed)
            reset_started = time.perf_counter()
            native_observation, _ = environment.reset(seed=episode_seed)
            reset_s = time.perf_counter() - reset_started
            episode_id = f"{task_name}-episode-{episode_index:04d}"
            episode_started = time.perf_counter()
            native_steps = 0
            success = False
            done = False
            policy_s = 0.0
            simulator_s = 0.0
            with PolicyClient(args.policy_socket) as client:
                client.reset(
                    EpisodeContext(
                        episode_id=episode_id,
                        task_id=task_name,
                        metadata={"env_name": env_name, "episode_index": episode_index},
                    )
                )
                try:
                    while not done:
                        policy_started = time.perf_counter()
                        action_chunk = client.act(
                            build_observation(episode_id, native_steps, native_observation)
                        ).actions
                        policy_s += time.perf_counter() - policy_started
                        simulator_started = time.perf_counter()
                        native_observation, _, terminations, truncations, infos = environment.step(
                            native_action(action_chunk, args.action_steps)
                        )
                        simulator_s += time.perf_counter() - simulator_started
                        # The final chunk is shorter when the step limit truncates it.
                        native_steps += np.asarray(infos["rewards"])[0].size
                        success |= bool(np.asarray(infos["success"])[0, 0])
                        done = bool(terminations[0] or truncations[0])
                finally:
                    client.end_episode(episode_id)
            records.append(
                EvaluationRecord(
                    record_id=f"{task_name}-episode-{episode_index:04d}",
                    success=success,
                    steps=native_steps,
                    elapsed_s=time.perf_counter() - episode_started,
                    metadata={
                        "task_name": task_name,
                        "env_name": env_name,
                        "episode_index": episode_index,
                        "seed": episode_seed,
                        "policy_encoding": client.encoding,
                        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
                        "reset_s": reset_s,
                        "policy_s": policy_s,
                        "simulator_s": simulator_s,
                        "deferred_native_steps": rendering.deferred_steps if rendering else 0,
                    },
                )
            )
            queue.publish_partial(
                item,
                WorkResult(
                    item_id=item.item_id,
                    group_id=item.group_id,
                    worker_id=args.worker_id,
                    records=tuple(records),
                    metadata={"task_name": task_name, "env_name": env_name},
                ),
            )
    finally:
        environment.close()
    return WorkResult(
        item_id=item.item_id,
        group_id=item.group_id,
        worker_id=args.worker_id,
        records=tuple(records),
        metadata={
            "task_name": task_name,
            "env_name": env_name,
            "elapsed_s": time.perf_counter() - started,
            "initialize_s": initialize_s,
        },
    )


def main() -> None:
    args = parse_args()
    if args.action_steps < 1 or args.max_episode_steps < 1 or args.poll_interval <= 0:
        raise ValueError("RoboCasa worker numeric arguments must be positive")
    if not 0 <= args.seed < 2**32:
        raise ValueError("RoboCasa seed must be an unsigned 32-bit integer")
    WorkerLoop(
        worker_id=args.worker_id,
        queue_dir=args.queue_dir,
        state_dir=args.worker_state_dir,
        poll_interval_s=args.poll_interval,
    ).run(lambda item: run_task(args, item))


if __name__ == "__main__":
    main()
