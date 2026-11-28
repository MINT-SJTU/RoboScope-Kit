# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Evo-DexBench evaluation through the upstream wave evaluator and a Kit Policy.

A work item is one upstream wave: four consecutive episodes of the upstream execution
order, stepped by the upstream ``CPUParallelEnv`` and ``CPUProtocolEvaluator``. Only
the Policy session is replaced. It sends the two most recent control-step frames per
slot over the Kit socket, executes the returned chunk, and asks the environments to
render only the frames the next request needs, as the upstream DP session does.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import resource
import time
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from roboscope_kit import EpisodeContext, Observation, PolicyClient
from roboscope_kit.runtime.benchmark import EvaluationRecord, WorkerLoop, WorkItem, WorkResult

from .protocol import (
    ACTION_BOUNDS_POLICY,
    CAMERAS,
    CONTROL_FREQ_HZ,
    CONTROL_MODE,
    EPISODES_PER_TASK,
    EVO_DEXBENCH_REVISION,
    JOINTS_PER_ROLE,
    NATIVE_ENVS,
    OBS_MODE,
    OBSERVATION_STEPS,
    REQUIRED_CAMERAS,
    ROLES,
    SIM_FREQ_HZ,
    TASKS,
    TASKS_BY_ID,
    Task,
    action_dimension,
    episode_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--policy-socket", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--worker-state-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-timeout", type=float, default=300.0)
    parser.add_argument("--poll-interval", type=float, default=0.05)
    return parser.parse_args()


def validate_upstream() -> None:
    import dex_benchmark
    from dex_benchmark.evaluation.protocol_parallel import EvaluationProtocol

    protocol = EvaluationProtocol.load()
    frozen = {
        "version": "evaluation_v1", "num_envs": NATIVE_ENVS, "sim_freq": SIM_FREQ_HZ,
        "control_freq": CONTROL_FREQ_HZ, "action_bounds_policy": ACTION_BOUNDS_POLICY,
        "observation_capture": "policy",
    }
    for key, expected in frozen.items():
        if getattr(protocol, key) != expected:
            raise RuntimeError(f"Evo-DexBench protocol {key} is {getattr(protocol, key)!r}, not {expected!r}")
    marker = Path(dex_benchmark.__file__).resolve().parents[1] / ".roboscope-revision"
    if not marker.is_file() or marker.read_text().strip() != EVO_DEXBENCH_REVISION:
        raise RuntimeError(f"Evo-DexBench runtime must be built from {EVO_DEXBENCH_REVISION}")


def task_schedule(task: Task, episodes: int):
    from dex_benchmark.evaluation.protocol_parallel import (
        EvaluationProtocol,
        build_protocol_schedule,
        execution_order,
    )
    from dex_benchmark.sim.generalization import GeneralizationProtocol
    from dex_benchmark.sim.tasks.bootstrap import load_builtin_task

    settings = load_builtin_task(task.task_id)
    if settings.max_steps != task.horizon:
        raise RuntimeError(f"{task.task_id} horizon is {settings.max_steps}, not {task.horizon}")
    layouts = tuple(layout.value for layout in settings.supported_layouts)
    schedule = build_protocol_schedule(
        task_id=task.task_id, layout_ids=layouts, episodes=episodes,
        protocol=GeneralizationProtocol.load(), evaluation_protocol=EvaluationProtocol.load(),
    )
    return execution_order(schedule)


def base_env_kwargs() -> dict[str, Any]:
    # What `dexbench evaluate --obs-mode rgb --control-mode CONTROL_MODE` passes without video.
    return {
        "obs_mode": OBS_MODE, "render_mode": None, "control_mode": CONTROL_MODE,
        "sim_config": {"sim_freq": SIM_FREQ_HZ, "control_freq": CONTROL_FREQ_HZ},
    }


def frame(model_input: Mapping[str, Any], embodiment: str) -> dict[str, Any]:
    from dex_benchmark.policy_api.observation import as_finite_numpy, required_value

    observation = model_input["observation"]
    cameras = required_value(observation, "cameras")
    manipulators = required_value(observation, "manipulators")
    features: dict[str, Any] = {}
    for camera in CAMERAS[embodiment]:
        if camera not in cameras:
            if camera in REQUIRED_CAMERAS[embodiment]:
                raise RuntimeError(f"Evo-DexBench observation lacks camera {camera!r}")
            continue
        rgb = as_finite_numpy(required_value(cameras, camera, "rgb"), name=f"{camera}.rgb")
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise RuntimeError(f"Evo-DexBench camera {camera}: invalid {rgb.shape}/{rgb.dtype}")
        features[f"images.{camera}"] = rgb
    for role in ROLES[embodiment]:
        pose = as_finite_numpy(
            required_value(manipulators, role, "end_effector_pose"), name=f"{role}.end_effector_pose",
        ).astype(np.float32, copy=False)
        joints = as_finite_numpy(
            required_value(manipulators, role, "joint_position"), name=f"{role}.joint_position",
        ).astype(np.float32, copy=False)
        if pose.shape != (7,) or joints.shape != (JOINTS_PER_ROLE,):
            raise RuntimeError(f"Evo-DexBench {role} state has shapes {pose.shape}/{joints.shape}")
        features[f"state.{role}.end_effector_pose"] = pose
        features[f"state.{role}.joint_position"] = joints
    return features


class KitWaveSession:
    """Upstream ``BatchedPolicySession`` whose model runs behind the Kit Policy socket."""

    def __init__(self, client: PolicyClient, task: Task) -> None:
        self.client = client
        self.task = task
        self.embodiment = task.embodiment
        self.dimension = action_dimension(task.embodiment)
        self.contexts: tuple[EpisodeContext, ...] = ()
        self.started: list[str] = []
        self.histories: list[deque] = []
        self.actions: list[deque] = []
        self.steps: list[int] = []
        self.inference_s = 0.0
        self.requests = 0

    def begin(self, contexts: Sequence[EpisodeContext]) -> None:
        self.contexts = tuple(contexts)

    def reset(self) -> None:
        # Upstream calls reset() after the environments reset, once per wave.
        if len(self.contexts) != NATIVE_ENVS:
            raise RuntimeError("Evo-DexBench wave started without its episode contexts")
        for context in self.contexts:
            self.client.reset(context)
            self.started.append(context.episode_id)
        self.histories = [deque(maxlen=OBSERVATION_STEPS) for _ in self.contexts]
        self.actions = [deque() for _ in self.contexts]
        self.steps = [0] * len(self.contexts)

    def end(self) -> None:
        for identifier in self.started:
            self.client.end_episode(identifier)
        self.started.clear()

    def _remaining(self) -> int:
        remaining = {len(buffer) for buffer in self.actions}
        if len(remaining) != 1:
            raise RuntimeError("Evo-DexBench action chunks became misaligned")
        return remaining.pop()

    def act_batch(self, model_inputs: Sequence[Mapping[str, Any]]) -> tuple[dict, ...]:
        from dex_benchmark.policy_api.policy_features import decode_semantic_action

        inputs = tuple(model_inputs)
        if len(inputs) != len(self.contexts):
            raise RuntimeError("Evo-DexBench Policy batch does not match the wave")
        remaining = self._remaining()
        if remaining < OBSERVATION_STEPS:
            for history, model_input in zip(self.histories, inputs, strict=True):
                current = frame(model_input, self.embodiment)
                if not history:
                    history.extend([current] * (OBSERVATION_STEPS - 1))
                history.append(current)
        if remaining == 0:
            self._request(inputs)
        for index in range(len(self.steps)):
            self.steps[index] += 1
        return tuple(decode_semantic_action(buffer.popleft(), self.embodiment) for buffer in self.actions)

    def _request(self, inputs: Sequence[Mapping[str, Any]]) -> None:
        observations = []
        for context, history, model_input, step in zip(
            self.contexts, self.histories, inputs, self.steps, strict=True,
        ):
            features = {
                key: np.ascontiguousarray(np.stack([entry[key] for entry in history]))
                for key in history[-1]
            }
            instruction = model_input["instruction"]
            if not isinstance(instruction, str) or not instruction.strip():
                raise RuntimeError("Evo-DexBench instruction is missing or blank")
            features["language.instruction"] = instruction
            observations.append(Observation(context.episode_id, step, features, time.time()))
        started = time.perf_counter()
        chunks = self.client.act_batch(tuple(observations))
        self.inference_s += time.perf_counter() - started
        self.requests += 1
        if len(chunks) != len(observations):
            raise RuntimeError("Evo-DexBench Policy returned an incomplete action batch")
        lengths = set()
        for buffer, chunk in zip(self.actions, chunks, strict=True):
            actions = np.asarray(chunk.actions, dtype=np.float32)
            if actions.ndim == 1:
                actions = actions[None]
            if actions.ndim != 2 or actions.shape[1] != self.dimension or not np.isfinite(actions).all():
                raise RuntimeError(f"Evo-DexBench requires finite (k, {self.dimension}) actions")
            lengths.add(len(actions))
            buffer.extend(actions)
        if len(lengths) != 1:
            raise RuntimeError("Evo-DexBench Policy returned chunks of different lengths")

    def _capture(self) -> bool:
        # Render only the frames the next request reads, as the upstream DP session does.
        return not self.actions or self._remaining() < OBSERVATION_STEPS

    capture_observation_after_action = _capture
    capture_visual_observation_after_action = _capture
    capture_lowdim_observation_after_action = _capture

    def requires_control_period_tactile(self) -> bool:
        return False

    def close(self) -> None:
        pass


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported Evo-DexBench record type {type(value).__name__}")


class TaskEnvironment:
    """One upstream ``CPUParallelEnv`` kept across the waves of a task."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.task: Task | None = None
        self.env: Any = None

    def get(self, task: Task, initial_kwargs: dict[str, Any]) -> tuple[Any, float]:
        if self.task == task:
            return self.env, 0.0
        from dex_benchmark.evaluation.cpu_parallel import CPUParallelEnv

        self.close()
        started = time.perf_counter()
        self.env = CPUParallelEnv(
            task_id=task.task_id, env_kwargs=initial_kwargs, num_envs=NATIVE_ENVS,
            action_bounds_policy=ACTION_BOUNDS_POLICY, worker_timeout=self.args.worker_timeout,
        )
        self.task = task
        return self.env, time.perf_counter() - started

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
        self.env = None
        self.task = None


def run_wave(
    args: argparse.Namespace, item: WorkItem, environments: TaskEnvironment, client: PolicyClient,
) -> WorkResult:
    from dex_benchmark.evaluation.protocol_parallel import (
        CPUProtocolEvaluator,
        SingleEvaluationLog,
    )
    from dex_benchmark.policy_api.runner import CentralizedBatchPolicyController
    from dex_benchmark.sim.generalization import GeneralizationProtocol

    task = TASKS_BY_ID[str(item.payload["task"])]
    episodes = int(item.payload["episodes"])
    wave = int(item.payload["wave"])
    ordered = task_schedule(task, episodes)
    rows = ordered[wave * NATIVE_ENVS:(wave + 1) * NATIVE_ENVS]
    if len(rows) != NATIVE_ENVS:
        raise RuntimeError(f"{item.item_id} is outside the {task.task_id} schedule")
    base = base_env_kwargs()
    started = time.perf_counter()
    env, create_s = environments.get(task, rows[0].env_kwargs(base))
    session = KitWaveSession(client, task)
    controller = CentralizedBatchPolicyController.from_session(session, num_envs=NATIVE_ENVS)
    # One directory for all Workers, so a retry on any Worker finds the failed attempt's log.
    log_path = args.output_dir / "native-logs" / f"{item.item_id}.jsonl"
    if log_path.exists():
        # A retried wave: the upstream log refuses to overwrite, so keep the failed attempt aside.
        attempt = 1
        while log_path.with_name(f"{log_path.stem}.failed-{attempt}.jsonl").exists():
            attempt += 1
        log_path.rename(log_path.with_name(f"{log_path.stem}.failed-{attempt}.jsonl"))
    log = SingleEvaluationLog(log_path)
    timing: dict[str, float] = {"environment_create_s": create_s}

    def start(episode_ids: Sequence[int]):
        timing["prepare_s"] = time.perf_counter() - started - create_s
        timing["rollout_started"] = time.perf_counter()
        session.begin([
            EpisodeContext(episode_id(task.task_id, int(index)), task.task_id, {
                "task_name": task.name, "embodiment": task.embodiment, "wave": wave,
                "episode_index": int(index),
            })
            for index in episode_ids
        ])
        return None

    evaluator = CPUProtocolEvaluator(
        env, controller, schedule=rows, base_env_kwargs=base, max_steps=task.horizon, log=log,
        allow_observation_elision=True,
        max_resample_attempts=GeneralizationProtocol.load().max_resample_attempts,
    )
    try:
        result = evaluator.run(on_wave_start=start, on_wave_end=session.end)
    except BaseException:
        # Keep the evaluator's error; a failed Policy connection would also fail end_episode.
        with contextlib.suppress(Exception):
            session.end()
        raise
    finally:
        log.close()
    rollout_s = time.perf_counter() - timing.pop("rollout_started")
    records = []
    for raw in result.records:
        metadata = _plain(raw)
        metadata.update(wave=wave, task=task.task_id, embodiment=task.embodiment, horizon=task.horizon)
        records.append(EvaluationRecord(
            episode_id(task.task_id, int(raw["episode_id"])), bool(raw["success"]),
            int(raw["episode_steps"]), float(raw["wave_wall_time_s"]), metadata,
        ))
    timing.update(
        rollout_s=rollout_s, inference_wait_s=session.inference_s,
        environment_step_s=rollout_s - session.inference_s, policy_requests=session.requests,
        wave_elapsed_s=time.perf_counter() - started,
    )
    return WorkResult(item.item_id, item.group_id, args.worker_id, tuple(records), {
        "task": task.task_id, "wave": wave, "native_log": str(log_path), "timings": timing,
    })


def raise_open_file_limit() -> None:
    """Lift the soft descriptor limit to the hard one, for this process and its env workers.

    SAPIEN keeps every mesh of a scene open while loading it; a PartNet-heavy scene such as
    m01's exceeds the common 1024 default and fails with "Failed to open file".
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = hard
    if hard == resource.RLIM_INFINITY:
        # An unlimited soft descriptor limit is rejected; use the kernel's per-process ceiling.
        target = int(Path("/proc/sys/fs/nr_open").read_text())
    if soft == resource.RLIM_INFINITY or soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))


def check_runtime() -> None:
    """Load every task's schedule and step one nominal environment once."""
    from dex_benchmark.evaluation.cpu_parallel import CPUParallelEnv

    raise_open_file_limit()
    validate_upstream()
    for task in TASKS:
        rows = task_schedule(task, EPISODES_PER_TASK)
        if len(rows) != EPISODES_PER_TASK or len({row.layout_id for row in rows}) != 3:
            raise RuntimeError(f"{task.task_id} schedule does not match evaluation_v1")
    task = TASKS[0]
    row = task_schedule(task, NATIVE_ENVS)[0]
    env = CPUParallelEnv(
        task_id=task.task_id, env_kwargs=row.env_kwargs(base_env_kwargs()), num_envs=1,
        action_bounds_policy=ACTION_BOUNDS_POLICY,
    )
    try:
        env.prepare([row.env_kwargs(base_env_kwargs())], [row.reset_seed])
        observation, _ = env.reset(seeds=[row.reset_seed])
        from dex_benchmark.policy_api.model_input import build_model_input
        from dex_benchmark.policy_api.runner import _slice_observation

        frame(build_model_input(_slice_observation(observation, 0, 1), env.instructions()[0]), "single")
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    raise_open_file_limit()
    validate_upstream()
    environments = TaskEnvironment(args)
    with PolicyClient(args.policy_socket, timeout_s=600) as client:
        try:
            WorkerLoop(
                worker_id=args.worker_id, queue_dir=args.queue_dir,
                state_dir=args.worker_state_dir, poll_interval_s=args.poll_interval,
            ).run(lambda item: run_wave(args, item, environments, client))
        finally:
            environments.close()


if __name__ == "__main__":
    main()
