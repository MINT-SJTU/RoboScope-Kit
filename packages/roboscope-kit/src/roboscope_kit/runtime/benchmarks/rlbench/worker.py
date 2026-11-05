# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Persistent RLBench worker for the PerAct 18-task profile, using an SDK Policy replica.

The worker reproduces one test episode of the pinned PerAct ``eval.py`` (YARR's
``RolloutGenerator`` driving PerAct's ``CustomMultiTaskRLBenchEnv``):

1. ``set_variation`` to the test episode's variation, restore its stored NumPy random state, and
   ``reset()`` the task, twice (see ``reset_to_test_episode``), which samples the recorded scene of
   the test episode; the instruction is ``get_task_descriptions()[0]``, read after the reset as
   PerAct does;
2. up to 25 key-frame actions. Each action is executed by RLBench's ``MoveArmThenGripper``
   (``EndEffectorPoseViaPlanning`` then ``Discrete``). An ``IKError``,
   ``ConfigurationPathError`` or ``InvalidActionError`` ends the episode as a failure, as
   PerAct's environment wrapper does;
3. the episode succeeds when a step returns reward >= 1 (all success conditions met), which also
   terminates it; it fails when a fail condition terminates it, on a planning error, or after the
   25th action.

Each worker owns one CoppeliaSim instance, rendered headless through its own Xvfb server.
"""

from __future__ import annotations

import argparse
import atexit
import ctypes
import os
import select
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from roboscope_kit import EpisodeContext, Observation
from roboscope_kit.client import PolicyClient
from roboscope_kit.runtime.benchmark import EvaluationRecord, WorkerLoop, WorkItem, WorkResult

from .tasks import (
    ACTION_DIM,
    CAMERAS,
    EPISODE_LENGTH,
    IMAGE_SIZE,
    PLANNING_ERRORS,
    PROTOCOL_VERSION,
    SUCCESS_CRITERION,
)
from .test_set import TestEpisode, load_test_episode

_PR_SET_PDEATHSIG = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--policy-socket", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--worker-state-dir", type=Path, required=True)
    parser.add_argument("--test-set", type=Path, required=True)
    parser.add_argument("--xvfb", type=Path, required=True)
    parser.add_argument("--poll-interval", type=float, default=0.05)
    return parser.parse_args()


def _die_with_parent() -> None:
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)


def start_xvfb(xvfb: Path, timeout_s: float = 30.0) -> subprocess.Popen:
    """Start a private Xvfb server, export its ``DISPLAY``, and stop it when the worker exits."""

    read_fd, write_fd = os.pipe()
    process = subprocess.Popen(
        (
            str(xvfb),
            "-displayfd",
            str(write_fd),
            "-screen",
            "0",
            "640x480x24",
            "-nolisten",
            "tcp",
            "+extension",
            "GLX",
            "-noreset",
        ),
        pass_fds=(write_fd,),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=_die_with_parent,
    )
    os.close(write_fd)
    atexit.register(process.terminate)
    display = b""
    deadline = time.monotonic() + timeout_s
    while not display.endswith(b"\n"):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or process.poll() is not None:
            raise RuntimeError(f"Xvfb ({xvfb}) did not report a display")
        ready, _, _ = select.select([read_fd], [], [], remaining)
        if ready:
            chunk = os.read(read_fd, 64)
            if not chunk:
                raise RuntimeError(f"Xvfb ({xvfb}) exited before reporting a display")
            display += chunk
    os.close(read_fd)
    os.environ["DISPLAY"] = f":{int(display.strip())}"
    return process


def observation_config():
    """PerAct's ``create_obs_config`` for its four cameras, with depth kept for the Policy.

    Keeping depth does not change rendering or the point clouds: RLBench captures the depth
    buffer for the point cloud either way and only drops it when depth is not requested.
    """

    from pyrep.const import RenderMode
    from rlbench.observation_config import CameraConfig, ObservationConfig

    unused = CameraConfig()
    unused.set_all(False)
    used = CameraConfig(
        rgb=True,
        point_cloud=True,
        mask=False,
        depth=True,
        image_size=IMAGE_SIZE,
        render_mode=RenderMode.OPENGL,
    )
    cameras = {name: used for name in CAMERAS}
    return ObservationConfig(
        front_camera=cameras.get("front", unused),
        left_shoulder_camera=cameras.get("left_shoulder", unused),
        right_shoulder_camera=cameras.get("right_shoulder", unused),
        wrist_camera=cameras.get("wrist", unused),
        overhead_camera=cameras.get("overhead", unused),
        joint_forces=False,
        joint_positions=True,
        joint_velocities=True,
        task_low_dim_state=False,
        gripper_touch_forces=False,
        gripper_pose=True,
        gripper_open=True,
        gripper_matrix=True,
        gripper_joint_positions=True,
    )


def create_environment(test_set: Path):
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment

    environment = Environment(
        action_mode=MoveArmThenGripper(EndEffectorPoseViaPlanning(), Discrete()),
        obs_config=observation_config(),
        dataset_root=str(test_set),
        headless=True,
    )
    environment.launch()
    return environment


def task_class(name: str):
    from rlbench.backend.utils import task_file_to_task_class

    return task_file_to_task_class(name)


@dataclass(frozen=True)
class ResetDiagnostics:
    # Largest object displacement (m) after the clearing reset, from the recorded first frame.
    clearing_reset_position_error_m: float
    # The same after the episode's reset: how far the observed scene is from the recorded one.
    initial_scene_position_error_m: float
    # Largest object displacement (m) made by get_task_descriptions() after the observation.
    descriptions_moved_objects_m: float


def object_positions(task, state: np.ndarray) -> np.ndarray | None:
    """The positions of the task's objects in an RLBench task low-dimensional state vector.

    ``Task.get_low_dim_state`` concatenates, per object, its pose (x, y, z, qx, qy, qz, qw),
    then a joint's position or a force sensor's six readings. Positions compare scenes without
    quaternion sign ambiguity or contact-force noise.
    """

    from pyrep.const import ObjectType

    positions = []
    index = 0
    for _object, object_type in task._initial_objs_in_scene:
        positions.append(state[index : index + 3])
        index += 7 + {ObjectType.JOINT: 1, ObjectType.FORCE_SENSOR: 6}.get(object_type, 0)
    if index != state.size:
        return None
    return np.asarray(positions, dtype=np.float64)


def max_displacement(task, actual: np.ndarray, expected: np.ndarray) -> float:
    """Largest per-object position difference (m) between two task states; -1 if incomparable."""

    left = object_positions(task, np.asarray(actual, dtype=np.float64))
    right = object_positions(task, np.asarray(expected, dtype=np.float64))
    if left is None or right is None or left.shape != right.shape:
        return -1.0
    return float(np.max(np.abs(left - right))) if left.size else 0.0


def _seeded_reset(task_env, episode: TestEpisode):
    task_env.set_variation(episode.variation)
    np.random.set_state(episode.random_state)
    _, observation = task_env.reset()
    return observation, np.asarray(task_env._task.get_low_dim_state(), dtype=np.float64)


def reset_to_test_episode(task_env, episode: TestEpisode):
    """PerAct ``CustomMultiTaskRLBenchEnv.reset_to_demo`` without loading the recorded images.

    PerAct restores the episode's variation and random state and resets once. A reset that starts
    from the end of a played episode can place the objects differently (RLBench's placement
    rejects samples against objects the previous episode left behind), so the scene would depend on
    which episode a worker ran before. The worker therefore resets twice with the same variation
    and random state: the first reset clears the previous episode, and the second samples the
    recorded scene of the test episode, whatever ran before it.

    Like PerAct, the instruction comes from ``get_task_descriptions()``, called after the reset
    has produced the first observation. That call runs the task's ``init_episode`` once more, and
    tasks whose ``init_episode`` samples object poses move those objects again without a new
    observation. The protocol keeps this upstream behaviour and records how far it moved them.
    """

    recorded = np.asarray(episode.recorded_task_state, dtype=np.float64)
    _, cleared = _seeded_reset(task_env, episode)
    observation, sampled = _seeded_reset(task_env, episode)
    instruction = task_env.get_task_descriptions()[0]
    described = np.asarray(task_env._task.get_low_dim_state(), dtype=np.float64)
    task = task_env._task
    diagnostics = ResetDiagnostics(
        clearing_reset_position_error_m=max_displacement(task, cleared, recorded),
        initial_scene_position_error_m=max_displacement(task, sampled, recorded),
        descriptions_moved_objects_m=max_displacement(task, described, sampled),
    )
    return observation, instruction, diagnostics


def build_observation(
    observation: Any,
    instruction: str,
    episode_id: str,
    step: int,
) -> Observation:
    features: dict[str, Any] = {"language.instruction": instruction}
    for camera in CAMERAS:
        misc = observation.misc
        depth = np.asarray(getattr(observation, f"{camera}_depth"), dtype=np.float64)
        near = float(misc[f"{camera}_camera_near"])
        far = float(misc[f"{camera}_camera_far"])
        features[f"images.{camera}"] = np.ascontiguousarray(
            getattr(observation, f"{camera}_rgb"), dtype=np.uint8
        )
        features[f"state.{camera}_depth"] = (near + depth * (far - near)).astype(np.float32)
        features[f"state.{camera}_point_cloud"] = np.asarray(
            getattr(observation, f"{camera}_point_cloud"), dtype=np.float32
        )
        features[f"state.{camera}_camera_extrinsics"] = np.asarray(
            misc[f"{camera}_camera_extrinsics"], dtype=np.float32
        )
        features[f"state.{camera}_camera_intrinsics"] = np.asarray(
            misc[f"{camera}_camera_intrinsics"], dtype=np.float32
        )
    features["state.gripper_open"] = np.asarray([observation.gripper_open], dtype=np.float32)
    for name in ("gripper_joint_positions", "gripper_pose", "joint_positions", "joint_velocities"):
        features[f"state.{name}"] = np.asarray(getattr(observation, name), dtype=np.float32)
    return Observation(episode_id=episode_id, step=step, timestamp_s=time.time(), features=features)


def episode_id_for(task: str, episode: int) -> str:
    return f"{task}-episode-{episode:02d}"


class StepCounter:
    """Counts simulation steps through the scene's step callback."""

    def __init__(self) -> None:
        self.steps = 0

    def __call__(self) -> None:
        self.steps += 1


def run_episode(
    client: PolicyClient,
    task_env,
    counter: StepCounter,
    episode: TestEpisode,
) -> tuple[EvaluationRecord, float]:
    from pyrep.errors import ConfigurationPathError, IKError
    from rlbench.backend.exceptions import InvalidActionError

    episode_id = episode_id_for(episode.task, episode.episode)
    observation, instruction, diagnostics = reset_to_test_episode(task_env, episode)
    client.reset(
        EpisodeContext(
            episode_id=episode_id,
            task_id=episode.task,
            metadata={
                "protocol": PROTOCOL_VERSION,
                "episode": episode.episode,
                "variation": episode.variation,
                "episode_length": EPISODE_LENGTH,
                "instruction": instruction,
                "cameras": list(CAMERAS),
            },
        )
    )
    counter.steps = 0
    started = time.perf_counter()
    inference_wait_s = 0.0
    success = False
    steps = 0
    termination = "timeout"
    error = None
    actions: list[list[float]] = []
    try:
        for step in range(EPISODE_LENGTH):
            request = build_observation(observation, instruction, episode_id, step)
            requested = time.perf_counter()
            chunk = np.asarray(client.act(request).actions, dtype=np.float64)
            inference_wait_s += time.perf_counter() - requested
            if chunk.shape not in ((ACTION_DIM,), (1, ACTION_DIM)):
                raise RuntimeError(
                    f"RLBench Policy action shape is {chunk.shape}, expected (1, {ACTION_DIM})"
                )
            action = chunk.reshape(ACTION_DIM)
            actions.append(action.tolist())
            steps = step + 1
            try:
                observation, reward, terminate = task_env.step(action)
            except (IKError, ConfigurationPathError, InvalidActionError) as exc:
                termination = type(exc).__name__
                error = str(exc)
                break
            if reward >= 1:
                success = True
            if terminate:
                termination = "success" if success else "fail_condition"
                break
    finally:
        client.end_episode(episode_id)
    elapsed_s = time.perf_counter() - started
    if termination not in ("success", "fail_condition", "timeout", *PLANNING_ERRORS):
        raise RuntimeError(f"unexpected RLBench termination {termination}")
    record = EvaluationRecord(
        record_id=episode_id,
        success=success,
        steps=steps,
        elapsed_s=elapsed_s,
        metadata={
            "task": episode.task,
            "episode": episode.episode,
            "variation": episode.variation,
            "episode_state_sha256": episode.state_sha256,
            "instruction": instruction,
            "termination": termination,
            "error": error,
            "simulation_steps": counter.steps,
            "clearing_reset_position_error_m": diagnostics.clearing_reset_position_error_m,
            "initial_scene_position_error_m": diagnostics.initial_scene_position_error_m,
            "descriptions_moved_objects_m": diagnostics.descriptions_moved_objects_m,
            "actions": actions,
            "protocol": PROTOCOL_VERSION,
            "success_criterion": SUCCESS_CRITERION,
        },
    )
    return record, inference_wait_s


def main() -> None:
    args = parse_args()
    if args.worker_id < 0 or args.poll_interval <= 0:
        raise ValueError("RLBench worker numeric arguments must be positive")
    # Runtimes may live on read-only or inode-limited storage; never write pyc.
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    start_xvfb(args.xvfb)
    environment = create_environment(args.test_set)
    counter = StepCounter()
    current: dict[str, Any] = {"task": None, "env": None}

    def task_environment(name: str):
        if current["task"] != name:
            # Loading a task replaces the scene's previous one, as PerAct's task swap does.
            current["env"] = environment.get_task(task_class(name))
            current["task"] = name
            current["env"]._scene.register_step_callback(counter)
        return current["env"]

    def evaluate(item: WorkItem, client: PolicyClient) -> WorkResult:
        task = str(item.payload["task"])
        episodes = [int(episode) for episode in item.payload["episodes"]]
        task_env = task_environment(task)
        started = time.perf_counter()
        records = []
        inference_wait_s = 0.0
        for index in episodes:
            episode = load_test_episode(args.test_set, task, index)
            record, wait = run_episode(client, task_env, counter, episode)
            records.append(record)
            inference_wait_s += wait
        return WorkResult(
            item_id=item.item_id,
            group_id=item.group_id,
            worker_id=args.worker_id,
            records=tuple(records),
            metadata={
                "suite": args.suite,
                "task": task,
                "elapsed_s": time.perf_counter() - started,
                "inference_wait_s": inference_wait_s,
            },
        )

    try:
        with PolicyClient(args.policy_socket) as client:
            WorkerLoop(
                worker_id=args.worker_id,
                queue_dir=args.queue_dir,
                state_dir=args.worker_state_dir,
                poll_interval_s=args.poll_interval,
            ).run(lambda item: evaluate(item, client))
    finally:
        environment.shutdown()


if __name__ == "__main__":
    main()
