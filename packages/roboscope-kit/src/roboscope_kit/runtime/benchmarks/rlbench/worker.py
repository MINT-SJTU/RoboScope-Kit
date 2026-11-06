# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Persistent RLBench worker for the PerAct 18-task profile, using an SDK Policy replica.

The worker reproduces one test episode of the pinned PerAct ``eval.py`` (YARR's
``RolloutGenerator`` driving PerAct's ``CustomMultiTaskRLBenchEnv``):

1. ``set_variation`` to the test episode's variation, restore its stored NumPy random state, and
   ``reset()`` the task once (see ``reset_to_test_episode``), which samples the recorded scene of
   the test episode; the instruction is ``get_task_descriptions()[0]``, read after the reset as
   PerAct does;
2. up to 25 key-frame actions. Each action is executed by RLBench's ``MoveArmThenGripper``
   (``EndEffectorPoseViaPlanning`` then ``Discrete``). An ``IKError``,
   ``ConfigurationPathError`` or ``InvalidActionError`` ends the episode as a failure, as
   PerAct's environment wrapper does;
3. the episode succeeds when a step returns reward >= 1 (all success conditions met), which also
   terminates it; it fails when a fail condition terminates it, on a planning error, or after the
   25th action.

Each record also carries how far the starting scene is from the test episode's recorded first
frame, compared object by object through the handles captured when the task was loaded
(``LoadedTask``). RLBench's task low-dimensional state lists the objects in an order a grasp
changes within one loaded task, so after a grasp its order is not the recorded one; the
profile does not give that state to the Policy (``task_low_dim_state=False``, which PerAct also
does not use).

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
    # Largest object displacement (m) after the reset from the recorded first frame, by object.
    initial_scene_position_error_m: float
    # The object at that distance.
    initial_scene_worst_object: str
    # Largest object displacement (m) made by get_task_descriptions() after the observation.
    descriptions_moved_objects_m: float


def _state_lengths(objects) -> list[int]:
    """Per-object slot lengths of ``Task.get_low_dim_state`` (``rlbench/backend/task.py``).

    Each object contributes its pose (x, y, z, qx, qy, qz, qw), then a joint's position (+1) or a
    force sensor's three forces and three torques (+6); a deleted object contributes zeros of the
    same length.
    """

    from pyrep.const import ObjectType

    return [7 + {ObjectType.JOINT: 1, ObjectType.FORCE_SENSOR: 6}.get(kind, 0) for _, kind in objects]


def recorded_positions(objects, state) -> np.ndarray:
    """Per-object positions in a recorded task state laid out over ``objects``.

    Raises ``ValueError`` when the state's length is not the layout's: the recorded state was then
    not taken over this object list, and no object-by-object comparison is possible.
    """

    state = np.asarray(state, dtype=np.float64).reshape(-1)
    lengths = _state_lengths(objects)
    if sum(lengths) != state.size:
        raise ValueError(
            f"recorded task state has {state.size} values; the {len(objects)} task objects "
            f"lay out {sum(lengths)}"
        )
    offsets = np.cumsum([0, *lengths[:-1]], dtype=np.int64)
    return np.asarray([state[offset : offset + 3] for offset in offsets], dtype=np.float64).reshape(-1, 3)


def scene_positions(objects) -> np.ndarray:
    """Current positions of ``objects`` (zeros for a deleted object, as RLBench records it)."""

    return np.asarray(
        [obj.get_position() if obj.still_exists() else (0.0, 0.0, 0.0) for obj, _ in objects],
        dtype=np.float64,
    ).reshape(-1, 3)


def max_displacement(objects, actual: np.ndarray, expected: np.ndarray) -> tuple[float, str]:
    """Largest per-coordinate position difference (m) of any object, and that object's name."""

    if not len(objects):
        return 0.0, ""
    distance = np.max(np.abs(np.asarray(actual) - np.asarray(expected)), axis=1)
    worst = int(np.argmax(distance))
    obj = objects[worst][0]
    return float(distance[worst]), obj.get_name() if obj.still_exists() else f"handle {obj.get_handle()}"


class LoadedTask:
    """A task loaded into the scene, with the object list its recorded task states are laid out over.

    ``Task.get_low_dim_state`` (``rlbench/backend/task.py:84``) concatenates one slot per object of
    ``Task._initial_objs_in_scene``, which every ``Scene.reset`` rebuilds from the task's object
    tree (``rlbench/backend/scene.py:168``, ``task.py:169``). That order is not stable within one
    loaded task: a grasp re-parents the object to the gripper (``pyrep/robots/end_effectors/
    gripper.py:50``), the ``release()`` at the start of the next reset (``scene.py:152``) re-parents
    it to its old parent (``gripper.py:63``), where CoppeliaSim lists it after its siblings, and
    ``Task.restore_state`` (``task.py:342``) restores poses but not that order. The list is
    therefore captured once, at the first reset after the task was loaded, and every comparison
    goes through those object handles. The data generator loads the task afresh for each episode
    (``tools/dataset_generator.py:375``), so a recorded state is laid out over that order unless a
    failed demonstration was retried after a grasp in the same loaded task; ``initial_scenes``
    lists the test episodes where that happened. A changed object set raises.
    """

    def __init__(self, task_env) -> None:
        self.env = task_env
        self.objects: tuple | None = None

    def capture_objects(self) -> tuple:
        current = tuple(self.env._task._initial_objs_in_scene)
        if self.objects is None:
            self.objects = current
        elif sorted(obj.get_handle() for obj, _ in current) != sorted(
            obj.get_handle() for obj, _ in self.objects
        ):
            raise RuntimeError(
                f"RLBench task {self.env.get_name()} changed its object set after it was loaded"
            )
        return self.objects


def reset_scene(task: LoadedTask, episode: TestEpisode):
    """One seeded reset to a test episode: the first observation and the scene, object by object.

    Returns the observation, the task's objects (``LoadedTask.objects``), their positions after the
    reset, and their positions in the episode's recorded first frame.
    """

    task_env = task.env
    task_env.set_variation(episode.variation)
    np.random.set_state(episode.random_state)
    _, observation = task_env.reset()
    objects = task.capture_objects()
    recorded = recorded_positions(objects, episode.recorded_task_state)
    return observation, objects, scene_positions(objects), recorded


def reset_to_test_episode(task: LoadedTask, episode: TestEpisode):
    """PerAct ``CustomMultiTaskRLBenchEnv.reset_to_demo`` without loading the recorded images.

    Like PerAct, it restores the episode's variation and random state and resets once, whatever
    episode ran before on this worker, and takes the instruction from ``get_task_descriptions()``
    after the reset has produced the first observation. That call runs the task's
    ``init_episode`` once more, and tasks whose ``init_episode`` samples object poses move those
    objects again without a new observation. The protocol keeps this upstream behaviour and records
    how far it moved them.

    The initial-scene diagnostic compares each object's position, by object handle (see
    ``LoadedTask``), with its position in the recorded first frame; the observation the Policy
    receives does not include the task state.
    """

    observation, objects, sampled, recorded = reset_scene(task, episode)
    instruction = task.env.get_task_descriptions()[0]
    described = scene_positions(objects)
    error, worst = max_displacement(objects, sampled, recorded)
    diagnostics = ResetDiagnostics(
        initial_scene_position_error_m=error,
        initial_scene_worst_object=worst,
        descriptions_moved_objects_m=max_displacement(objects, described, sampled)[0],
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
    task: LoadedTask,
    counter: StepCounter,
    episode: TestEpisode,
) -> tuple[EvaluationRecord, float]:
    from pyrep.errors import ConfigurationPathError, IKError
    from rlbench.backend.exceptions import InvalidActionError

    episode_id = episode_id_for(episode.task, episode.episode)
    task_env = task.env
    observation, instruction, diagnostics = reset_to_test_episode(task, episode)
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
            "initial_scene_position_error_m": diagnostics.initial_scene_position_error_m,
            "initial_scene_worst_object": diagnostics.initial_scene_worst_object,
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
            current["env"] = LoadedTask(environment.get_task(task_class(name)))
            current["task"] = name
            current["env"].env._scene.register_step_callback(counter)
        return current["env"]

    def evaluate(item: WorkItem, client: PolicyClient) -> WorkResult:
        task = str(item.payload["task"])
        episodes = [int(episode) for episode in item.payload["episodes"]]
        loaded = task_environment(task)
        started = time.perf_counter()
        records = []
        inference_wait_s = 0.0
        for index in episodes:
            episode = load_test_episode(args.test_set, task, index)
            record, wait = run_episode(client, loaded, counter, episode)
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
