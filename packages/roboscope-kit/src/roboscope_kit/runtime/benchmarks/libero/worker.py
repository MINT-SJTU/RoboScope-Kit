# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Run persistent LIBERO episodes against an SDK-native user policy."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import imageio
import numpy as np

from roboscope_kit import EpisodeContext, Observation, PolicyClient, profiling
from roboscope_kit.runtime.benchmark import EvaluationRecord, WorkerLoop, WorkItem, WorkResult

from .camera import ChunkCameraGate

LIBERO_DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
LIBERO_DELTA_ACTION = "libero.delta-eef-v1"
LIBERO_ABSOLUTE_ACTION = "libero.absolute-eef-v1"
LIBERO_ENV_RESOLUTION = 256
# Observation feature -> robosuite camera; the order is the order robosuite renders them.
LIBERO_CAMERAS = (("images.primary", "agentview"), ("images.wrist", "robot0_eye_in_hand"))
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclass
class TaskRuntime:
    task_id: int
    task_description: str
    initial_states: np.ndarray
    env: object


@dataclass(frozen=True)
class _CacheSnapshot:
    model_hits: int = 0
    model_misses: int = 0
    egl_hits: int = 0


class _SimulatorCaches:
    """Optional pinned-runtime MuJoCo model and EGL context caches."""

    def __init__(self, *, compiled_model_cache: bool, egl_context_reuse: bool) -> None:
        self._model_cache: Any = None
        self._egl_cache: Any = None
        if egl_context_reuse:
            from robosuite.renderers.context import egl_context

            if not hasattr(egl_context, "_CONTEXT_REUSE_HITS"):
                raise RuntimeError("rebuild the LIBERO-Pro runtime before enabling egl_context_reuse")
            self._egl_cache = egl_context
        if compiled_model_cache:
            from robosuite.utils import binding_utils

            if version("robosuite") != "1.4.0" or version("mujoco") != "3.2.3":
                raise RuntimeError("compiled model caching requires the pinned LIBERO-Pro runtime")
            model_cache = getattr(binding_utils, "_cached_xml_model", None)
            if model_cache is None or os.environ.get("ROBOSCOPE_LIBERO_MODEL_CACHE") != "1":
                raise RuntimeError("rebuild the LIBERO-Pro runtime before enabling compiled_model_cache")
            self._model_cache = model_cache

    def snapshot(self) -> _CacheSnapshot:
        info = self._model_cache.cache_info() if self._model_cache is not None else None
        return _CacheSnapshot(
            model_hits=info.hits if info is not None else 0,
            model_misses=info.misses if info is not None else 0,
            egl_hits=self._egl_cache._CONTEXT_REUSE_HITS if self._egl_cache is not None else 0,
        )

    def metadata(self, before: _CacheSnapshot) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "egl_context_reuse": self._egl_cache is not None,
            "egl_context_reuse_hits": (
                self._egl_cache._CONTEXT_REUSE_HITS - before.egl_hits
                if self._egl_cache is not None
                else 0
            ),
        }
        if self._model_cache is not None:
            after = self._model_cache.cache_info()
            meta.update(
                compiled_model_cache_hits=after.hits - before.model_hits,
                compiled_model_cache_misses=after.misses - before.model_misses,
                compiled_model_cache_entries=after.currsize,
            )
        return meta

    def close(self) -> None:
        if self._egl_cache is not None:
            self._egl_cache._release_idle_context()


class PolicyActionQueue:
    def __init__(self, socket_path: Path) -> None:
        self.socket_path = socket_path
        self.encoding: str | None = None
        self.episode_id = ""
        self.actions = np.empty((0, 7), dtype=np.float32)
        self.inference_wait_s = 0.0
        self._client: PolicyClient | None = None

    def _open_client(self) -> PolicyClient:
        client = PolicyClient(self.socket_path)
        if hasattr(client, "connect"):
            client.connect()
        else:
            client = client.__enter__()
        self._client = client
        return client

    def _call(self, fn_name: str, *args: Any) -> Any:
        client = self._client if self._client is not None else self._open_client()
        try:
            result = getattr(client, fn_name)(*args)
        except OSError:
            self.close()
            client = self._open_client()
            result = getattr(client, fn_name)(*args)
        self.encoding = getattr(client, "encoding", None)
        return result

    def reset(self, context: EpisodeContext) -> None:
        self.episode_id = context.episode_id
        self.actions = np.empty((0, 7), dtype=np.float32)
        self._call("reset", context)

    def step(self, observation: dict, task_description: str, step: int) -> np.ndarray:
        if not self.actions.size:
            if profiling.TIMING_ENABLED:
                with profiling.span("worker.build_observation"):
                    request = build_observation(self.episode_id, step, observation, task_description)
            else:
                request = build_observation(self.episode_id, step, observation, task_description)
            started = time.perf_counter()
            chunk = self._call("act", request).actions
            self.inference_wait_s += time.perf_counter() - started
            self.actions = chunk.reshape(1, -1) if chunk.ndim == 1 else chunk
        # The chunk is owned exclusively by this queue and discarded as rows are consumed.
        action = self.actions[0]
        self.actions = self.actions[1:]
        return action

    def end_episode(self) -> None:
        if not self.episode_id:
            return
        episode_id = self.episode_id
        self.episode_id = ""
        self.actions = np.empty((0, 7), dtype=np.float32)
        self._call("end_episode", episode_id)

    def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            if hasattr(client, "close"):
                client.close()
            else:
                client.__exit__(None, None, None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--policy-socket", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument(
        "--compiled-model-cache",
        action="store_true",
        help="use the LIBERO-Pro builder's bounded pristine MuJoCo model cache",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="policy action steps per episode; required for suites outside MAX_STEPS",
    )
    parser.add_argument(
        "--instruction-source",
        choices=("task", "bddl"),
        default="task",
        help="task: the benchmark's Task.language; bddl: the (:language ...) of the task's BDDL file",
    )
    parser.add_argument(
        "--action-representation",
        choices=(LIBERO_DELTA_ACTION, LIBERO_ABSOLUTE_ACTION),
        default=LIBERO_DELTA_ACTION,
        help="negotiated action representation; absolute targets switch the OSC controller after settling",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--worker-state-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument(
        "--chunk-camera-rendering",
        action="store_true",
        help="skip unused RGB renders inside action chunks (disabled when saving video)",
    )
    parser.add_argument("--poll-interval", type=float, default=0.05)
    parser.add_argument(
        "--primary-size",
        type=_image_size,
        default=(LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION),
        help="agent-view render size as HEIGHTxWIDTH",
    )
    parser.add_argument(
        "--wrist-size",
        type=_image_size,
        default=(LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION),
        help="wrist render size as HEIGHTxWIDTH",
    )
    args = parser.parse_args()
    if args.max_steps is None:
        if args.suite not in MAX_STEPS:
            parser.error(f"--max-steps is required for suite {args.suite!r}")
        args.max_steps = MAX_STEPS[args.suite]
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    return args


def _image_size(value: str) -> tuple[int, int]:
    try:
        height, width = (int(part) for part in value.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected HEIGHTxWIDTH, got {value!r}") from exc
    if height < 1 or width < 1:
        raise argparse.ArgumentTypeError(f"image size must be positive, got {value!r}")
    return height, width


def build_observation(
    episode_id: str,
    step: int,
    observation: dict,
    task_description: str,
) -> Observation:
    primary = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1], dtype=np.uint8)
    wrist = np.ascontiguousarray(
        observation["robot0_eye_in_hand_image"][::-1, ::-1],
        dtype=np.uint8,
    )
    state = np.empty(16, dtype=np.float32)
    state[0:3] = _state_vector(observation, "robot0_eef_pos", 3)
    state[3:7] = _state_vector(observation, "robot0_eef_quat", 4)
    state[7:9] = _state_vector(observation, "robot0_gripper_qpos", 2)
    state[9:16] = _state_vector(observation, "robot0_joint_pos", 7)
    return Observation(
        episode_id=episode_id,
        step=step,
        timestamp_s=time.time(),
        features={
            "images.primary": primary,
            "images.wrist": wrist,
            "state.proprioception": state,
            "language.instruction": task_description,
        },
    )


def _state_vector(observation: dict, name: str, expected: int) -> np.ndarray:
    if name not in observation:
        raise KeyError(f"LIBERO observation is missing {name!r}")
    value = np.ravel(observation[name])
    if value.size != expected:
        raise ValueError(f"LIBERO observation {name!r} has {value.size} values; expected {expected}")
    return value


def create_task_runtime(
    task_suite: object,
    task_id: int,
    seed: int,
    instruction_source: str = "task",
    image_sizes: tuple[tuple[int, int], tuple[int, int]] = (
        (LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION),
        (LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION),
    ),
) -> TaskRuntime:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = task_suite.get_task(task_id)
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    canonical = (LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION)
    if all(size == canonical for size in image_sizes):
        env = OffScreenRenderEnv(
            bddl_file_name=str(task_bddl_file),
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
        )
    else:
        env = OffScreenRenderEnv(
            bddl_file_name=str(task_bddl_file),
            camera_names=[camera for _feature, camera in LIBERO_CAMERAS],
            camera_heights=[height for height, _width in image_sizes],
            camera_widths=[width for _height, width in image_sizes],
        )
    env.seed(seed)
    # LIBERO-Pro keeps the original BDDL file names while its perturbed (:language ...) differs, so
    # its instruction is the one the environment parsed from the BDDL file, not Task.language.
    instruction = env.language_instruction if instruction_source == "bddl" else task.language
    return TaskRuntime(
        task_id=task_id,
        task_description=instruction,
        initial_states=task_suite.get_task_init_states(task_id),
        env=env,
    )


def set_delta_control(env: object, use_delta: bool) -> None:
    """Make every robot's OSC controller read actions as deltas (LIBERO's default) or absolute targets."""
    for robot in env.env.robots:
        robot.controller.use_delta = use_delta


def run_episode(
    runtime: TaskRuntime,
    policy: PolicyActionQueue,
    episode_index: int,
    *,
    max_steps: int,
    settle_steps: int,
    save_frames: bool,
    absolute_actions: bool = False,
    chunk_camera_rendering: bool = False,
    camera_counters: dict[str, int] | None = None,
) -> tuple[bool, int, float, list[np.ndarray]]:
    episode_id = f"task-{runtime.task_id:04d}-episode-{episode_index:04d}"
    timed = profiling.TIMING_ENABLED
    policy.reset(
        EpisodeContext(
            episode_id=episode_id,
            task_id=str(runtime.task_id),
            metadata={"episode_index": episode_index, "instruction": runtime.task_description},
        )
    )
    try:
        if timed:
            with profiling.span("worker.reset"):
                runtime.env.reset()
            with profiling.span("worker.set_init_state"):
                observation = runtime.env.set_init_state(runtime.initial_states[episode_index])
        else:
            runtime.env.reset()
            observation = runtime.env.set_init_state(runtime.initial_states[episode_index])
        frames: list[np.ndarray] = []
        started = time.perf_counter()

        # The settle steps are zero deltas. robosuite rebuilds the controller on every reset, so delta
        # control is already on here and setting it is a guard. Absolute-target policies then get absolute
        # control, as the X-VLA recipe does.
        if absolute_actions:
            set_delta_control(runtime.env, True)

        gate_enabled = chunk_camera_rendering and not save_frames
        cameras = ChunkCameraGate()
        try:
            with cameras.install(runtime.env, observation, enabled=gate_enabled):
                for settle_idx in range(settle_steps):
                    if gate_enabled:
                        cameras.required = settle_idx == settle_steps - 1
                    if timed:
                        with profiling.span("worker.settle_step"):
                            observation, _reward, task_done, _info = runtime.env.step(LIBERO_DUMMY_ACTION)
                    else:
                        observation, _reward, task_done, _info = runtime.env.step(LIBERO_DUMMY_ACTION)
                    if task_done or runtime.env.env.done:
                        return bool(task_done), 0, time.perf_counter() - started, frames
                if absolute_actions:
                    set_delta_control(runtime.env, False)

                for action_step in range(max_steps):
                    if save_frames:
                        frames.append(
                            np.ascontiguousarray(observation["agentview_image"][::-1, ::-1], dtype=np.uint8)
                        )
                    action = policy.step(observation, runtime.task_description, action_step)
                    # step() popped one action. When that exhausts the returned chunk, the next
                    # policy request needs this step's native camera samples.
                    if gate_enabled:
                        cameras.required = not getattr(policy, "actions", ()).__len__()
                    if timed:
                        with profiling.span("worker.action_step"):
                            observation, _reward, task_done, _info = runtime.env.step(action)
                    else:
                        observation, _reward, task_done, _info = runtime.env.step(action)
                    if task_done or runtime.env.env.done:
                        return bool(task_done), action_step + 1, time.perf_counter() - started, frames
        finally:
            if camera_counters is not None:
                camera_counters["camera_renders"] += cameras.renders
                camera_counters["camera_renders_skipped"] += cameras.skipped

        return False, max_steps, time.perf_counter() - started, frames
    finally:
        policy.end_episode()


def save_video(
    video_dir: Path | None,
    task_id: int,
    episode_index: int,
    success: bool,
    frames: list[np.ndarray],
) -> None:
    if video_dir is None:
        return
    video_dir.mkdir(parents=True, exist_ok=True)
    suffix = "success" if success else "failure"
    path = video_dir / f"task-{task_id:02d}-episode-{episode_index:02d}-{suffix}.mp4"
    imageio.mimwrite(path, frames, fps=10)


def run_chunk(
    args: argparse.Namespace,
    item: WorkItem,
    runtime: TaskRuntime,
    policy: PolicyActionQueue,
) -> WorkResult:
    records: list[EvaluationRecord] = []
    started = time.perf_counter()
    inference_wait_before = policy.inference_wait_s
    camera_counters = {"camera_renders": 0, "camera_renders_skipped": 0}
    start_episode = int(item.payload["start_episode"])
    num_trials = int(item.payload["num_trials"])
    for episode_index in range(start_episode, start_episode + num_trials):
        success, steps, elapsed_s, frames = run_episode(
            runtime,
            policy,
            episode_index,
            max_steps=args.max_steps,
            settle_steps=args.settle_steps,
            save_frames=args.video_dir is not None,
            absolute_actions=args.action_representation == LIBERO_ABSOLUTE_ACTION,
            chunk_camera_rendering=args.chunk_camera_rendering,
            camera_counters=camera_counters,
        )
        save_video(args.video_dir, runtime.task_id, episode_index, success, frames)
        records.append(
            EvaluationRecord(
                record_id=f"task-{runtime.task_id:04d}-episode-{episode_index:04d}",
                success=success,
                steps=steps,
                elapsed_s=elapsed_s,
                metadata={"task_id": runtime.task_id, "episode_index": episode_index},
            )
        )
    return WorkResult(
        item_id=item.item_id,
        group_id=item.group_id,
        worker_id=args.worker_id,
        records=tuple(records),
        metadata={
            "suite": args.suite,
            "task_id": runtime.task_id,
            "task_description": runtime.task_description,
            "elapsed_s": time.perf_counter() - started,
            "inference_wait_s": policy.inference_wait_s - inference_wait_before,
            "policy_encoding": policy.encoding,
            "chunk_camera_rendering": args.chunk_camera_rendering and args.video_dir is None,
            **camera_counters,
        },
    )


def main() -> None:
    args = parse_args()
    if args.poll_interval <= 0:
        raise ValueError("poll-interval must be positive")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from libero.libero import benchmark

    caches = _SimulatorCaches(
        compiled_model_cache=args.compiled_model_cache,
        egl_context_reuse=os.environ.get("ROBOSCOPE_LIBERO_EGL_CONTEXT_REUSE") == "1",
    )
    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    policy = PolicyActionQueue(args.policy_socket)
    runtime: TaskRuntime | None = None

    def evaluate(item: WorkItem) -> WorkResult:
        nonlocal runtime
        with profiling.capture("worker", item_id=item.item_id, worker_id=args.worker_id):
            task_id = int(item.payload["task_id"])
            created = runtime is None or runtime.task_id != task_id
            before = caches.snapshot()
            create_s = 0.0
            if created:
                started = time.perf_counter()
                if runtime is not None:
                    runtime.env.close()
                runtime = create_task_runtime(
                    task_suite,
                    task_id,
                    args.seed,
                    args.instruction_source,
                    image_sizes=(args.primary_size, args.wrist_size),
                )
                create_s = time.perf_counter() - started
            result = run_chunk(args, item, runtime, policy)
            result.metadata.update(
                task_runtime_created=created,
                task_runtime_create_s=create_s,
                **caches.metadata(before),
            )
            return result

    try:
        WorkerLoop(
            worker_id=args.worker_id,
            queue_dir=args.queue_dir,
            state_dir=args.worker_state_dir,
            poll_interval_s=args.poll_interval,
        ).run(evaluate)
    finally:
        policy.close()
        if runtime is not None:
            runtime.env.close()
        caches.close()


if __name__ == "__main__":
    main()
