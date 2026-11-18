# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""BEHAVIOR Challenge 2026 worker: one public test instance per process, using an SDK Policy replica.

The worker runs the pinned evaluator itself (BEHAVIOR-1K ``v3.9.3-post1``,
``omnigibson.eval.evaluator.BatchedEvaluator``) the way ``python -m omnigibson.eval.eval`` runs
it for one instance (``--instance-indices i --num-envs 1 --num-rollouts 1``):

1. headless OmniGibson without the unused viewer camera, ``seed_everything(0)``, the default R1Pro
   robot config, the official ``RGBDFullResWrapper`` (head 720x720, wrists 480x480, RGB and
   linear depth) and ``max_steps`` of 1.5x the task's mean human demonstration length;
2. ``BatchedEvaluator.run([instance])``: the evaluator resets the robot, loads the instance's stored
   object states and robot pose, settles physics, resets the environment, and then queries the
   policy once per control step (30 Hz) until the task succeeds, OmniGibson's task termination
   fires, or the timeout ends it (after ``max_steps + 1`` actions, see ``tasks.py``);
3. the evaluator's own metrics (``TaskMetric`` Q score, ``AgentMetric`` distance and time) and its
   per-rollout JSON are the record.

Only the policy differs from the native command: instead of ``WebsocketPolicy`` the evaluator calls
:class:`KitPolicyBridge`, which turns each evaluator observation into an SDK observation, asks the
Policy replica for one action, and returns it to the evaluator unchanged. Loading a scene shuts down
the simulator application when the process ends, so each worker process evaluates one instance and
then exits for the runtime to start the next one.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from roboscope_kit import EpisodeContext, Observation
from roboscope_kit.client import PolicyClient
from roboscope_kit.runtime.benchmark import EvaluationRecord, WorkerLoop, WorkItem, WorkResult

from .tasks import (
    ACTION_DIM,
    BEHAVIOR_1K_COMMIT,
    CAMERA_POSES_DIM,
    CAMERAS,
    ENV_WRAPPER,
    EVAL_SEED,
    MAX_STEPS,
    PROPRIOCEPTION_DIM,
    PROTOCOL_VERSION,
    PYTHON_HASH_SEED,
    ROBOT_NAME,
    ROLLOUT_ID,
    TASK_IDS,
    episode_id_for,
    instance_id,
    require_python_hash_seed,
)

# Actions kept in each record, from the first; enough to locate an early divergence.
RECORDED_ACTIONS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--policy-socket", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--worker-state-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--write-video", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=0.05)
    return parser.parse_args()


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def observation_keys() -> dict[str, str]:
    """Flattened evaluator observation key behind each SDK feature."""

    keys = {}
    for role, sensor, _ in CAMERAS:
        keys[f"images.{role}"] = f"{ROBOT_NAME}::{sensor}::rgb"
        keys[f"state.{role}_depth"] = f"{ROBOT_NAME}::{sensor}::depth_linear"
    keys["state.proprioception"] = f"{ROBOT_NAME}::proprio"
    keys["state.camera_poses"] = f"{ROBOT_NAME}::cam_rel_poses"
    keys["state.task_id"] = "task_id"
    return keys


OBSERVATION_KEYS = observation_keys()


def build_observation(batched: dict[str, Any], episode_id: str, step: int) -> Observation:
    """SDK observation of the single environment in an evaluator batch (leading dimension 1).

    RGB drops the renderer's alpha channel; every other value is the evaluator's, cast to the
    feature dtype without rescaling.
    """

    features: dict[str, Any] = {}
    for feature, key in OBSERVATION_KEYS.items():
        if key not in batched:
            raise RuntimeError(f"BEHAVIOR evaluator observation lacks {key}")
        value = _numpy(batched[key])
        if value.shape[0] != 1:
            raise RuntimeError(
                f"BEHAVIOR evaluator batch of {key} has {value.shape[0]} environments, expected 1"
            )
        value = value[0]
        if feature.startswith("images."):
            if value.ndim != 3 or value.shape[-1] not in (3, 4) or value.dtype != np.uint8:
                raise RuntimeError(f"BEHAVIOR {key} is {value.dtype} {value.shape}, expected uint8 RGB(A)")
            value = np.ascontiguousarray(value[..., :3])
        elif feature == "state.task_id":
            value = value.astype(np.int64).reshape(1)
        else:
            value = np.ascontiguousarray(value, dtype=np.float32)
        features[feature] = value
    if features["state.proprioception"].shape != (PROPRIOCEPTION_DIM,):
        raise RuntimeError(f"BEHAVIOR proprioception has shape {features['state.proprioception'].shape}")
    if features["state.camera_poses"].shape != (CAMERA_POSES_DIM,):
        raise RuntimeError(f"BEHAVIOR camera poses have shape {features['state.camera_poses'].shape}")
    return Observation(episode_id=episode_id, step=step, timestamp_s=time.time(), features=features)


class KitPolicyBridge:
    """The evaluator's policy object, backed by an SDK Policy replica.

    ``BatchedEvaluator`` calls ``reset()`` once after loading an instance and ``forward(obs=...)``
    once per control step with a batch of one environment; both forward to the replica.
    """

    def __init__(self, client: PolicyClient) -> None:
        self.client = client
        self.context: EpisodeContext | None = None
        self.step = 0
        self.inference_wait_s = 0.0
        self.actions: list[list[float]] = []

    def start(self, context: EpisodeContext) -> None:
        self.context = context
        self.step = 0
        self.inference_wait_s = 0.0
        self.actions = []

    def set_action_dim(self, action_dim: int) -> None:
        if action_dim != ACTION_DIM:
            raise RuntimeError(f"BEHAVIOR robot action dimension is {action_dim}, expected {ACTION_DIM}")

    def reset(self) -> None:
        if self.context is None:
            raise RuntimeError("BEHAVIOR evaluator reset the policy before an episode started")
        self.client.reset(self.context)

    def forward(self, obs: dict[str, Any], *args: Any, **kwargs: Any):
        import torch as th

        if self.context is None:
            raise RuntimeError("BEHAVIOR evaluator queried the policy outside an episode")
        observation = build_observation(obs, self.context.episode_id, self.step)
        requested = time.perf_counter()
        chunk = np.asarray(self.client.act(observation).actions)
        self.inference_wait_s += time.perf_counter() - requested
        if chunk.shape not in ((ACTION_DIM,), (1, ACTION_DIM)):
            raise RuntimeError(f"BEHAVIOR Policy action shape is {chunk.shape}, expected (1, {ACTION_DIM})")
        action = chunk.reshape(1, ACTION_DIM).astype(np.float32)
        if len(self.actions) < RECORDED_ACTIONS:
            self.actions.append(action[0].tolist())
        self.step += 1
        return th.from_numpy(action)


def create_evaluator(task: str, write_video: bool):
    """``omnigibson.eval.eval.main`` up to the evaluator, for one task and one environment."""

    from omegaconf import OmegaConf
    from omnigibson.macros import gm

    gm.HEADLESS = True
    gm.RENDER_VIEWER_CAMERA = False

    from omnigibson.eval.evaluator import BatchedEvaluator
    from omnigibson.eval.utils.eval_utils import DEFAULT_EVAL_SEED, TASK_NAMES_TO_INDICES, seed_everything

    if DEFAULT_EVAL_SEED != EVAL_SEED or TASK_NAMES_TO_INDICES.get(task) != TASK_IDS[task]:
        raise RuntimeError("the installed BEHAVIOR evaluator does not match the frozen task table")
    seed = seed_everything(DEFAULT_EVAL_SEED)
    cfg = OmegaConf.create(
        {
            "env_wrapper": {"_target_": ENV_WRAPPER},
            "policy_name": "roboscope",
            # Replaced by the bridge below; LocalPolicy holds no connection.
            "model": {"_target_": "omnigibson.eval.policies.LocalPolicy", "action_dim": None},
            "headless": True,
            "partial_scene_load": True,
            "max_steps": None,
            "write_video": write_video,
            "mode": "public_test",
            "seed": seed,
            "num_envs": 1,
            "task": {"name": task},
            "robot": None,
        }
    )
    evaluator = BatchedEvaluator(cfg)
    limit = episode_step_limit(evaluator)
    if limit != MAX_STEPS[task]:
        raise RuntimeError(f"BEHAVIOR {task} max_steps is {limit}, expected frozen {MAX_STEPS[task]}")
    return evaluator


def episode_step_limit(evaluator) -> int:
    """The task's ``Timeout`` termination: the evaluator's ``max_steps`` (1.5x human length)."""

    return int(evaluator.env.task._termination_conditions["timeout"]._max_steps)


def run_instance(
    evaluator,
    bridge: KitPolicyBridge,
    task: str,
    index: int,
    output_dir: Path,
    write_video: bool,
    scene_load_s: float,
) -> EvaluationRecord:
    instance = instance_id(index)
    episode_id = episode_id_for(task, index)
    json_dir = output_dir / "json"
    video_dir = output_dir / "videos"
    json_dir.mkdir(parents=True, exist_ok=True)
    if write_video:
        video_dir.mkdir(parents=True, exist_ok=True)
    bridge.start(
        EpisodeContext(
            episode_id=episode_id,
            task_id=task,
            metadata={
                "protocol": PROTOCOL_VERSION,
                "task_index": TASK_IDS[task],
                "instance_index": index,
                "instance_id": instance,
                "max_steps": MAX_STEPS[task],
            },
        )
    )
    started = time.perf_counter()
    try:
        results = evaluator.run(
            [instance],
            write_video=write_video,
            video_path=str(video_dir),
            metrics_dir=str(json_dir),
            rollout_id=ROLLOUT_ID,
        )
    finally:
        bridge.client.end_episode(episode_id)
    elapsed_s = time.perf_counter() - started
    native = json.loads(json.dumps(results[instance], default=float))
    steps = int(native["steps"])
    if steps != bridge.step:
        raise RuntimeError(
            f"BEHAVIOR evaluator reports {steps} steps; the Policy was queried {bridge.step} times"
        )
    success = bool(native["success"])
    if success:
        termination = "success"
    elif steps > MAX_STEPS[task]:
        termination = "max_steps"
    else:
        termination = "task_termination"
    return EvaluationRecord(
        record_id=episode_id,
        success=success,
        steps=steps,
        elapsed_s=elapsed_s,
        metadata={
            "task": task,
            "task_index": TASK_IDS[task],
            "instance_index": index,
            "instance_id": instance,
            "rollout_id": ROLLOUT_ID,
            "max_steps": MAX_STEPS[task],
            "termination": termination,
            "q_score": float(native["q_score"]["final"]),
            "native_result": native,
            "first_actions": bridge.actions,
            "scene_load_s": scene_load_s,
            "inference_wait_s": bridge.inference_wait_s,
            "protocol": PROTOCOL_VERSION,
            "behavior_1k_commit": BEHAVIOR_1K_COMMIT,
            "python_hash_seed": PYTHON_HASH_SEED,
        },
    )


def main() -> None:
    args = parse_args()
    require_python_hash_seed()
    if args.worker_id < 0 or args.poll_interval <= 0:
        raise ValueError("BEHAVIOR worker numeric arguments must be positive")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    state: dict[str, Any] = {"evaluator": None}

    def evaluate(item: WorkItem, client: PolicyClient) -> WorkResult:
        if state["evaluator"] is not None:
            raise RuntimeError("a BEHAVIOR worker process evaluates one instance")
        task = str(item.payload["task"])
        index = int(item.payload["instance_index"])
        started = time.perf_counter()
        evaluator = create_evaluator(task, args.write_video)
        state["evaluator"] = evaluator
        bridge = KitPolicyBridge(client)
        evaluator.policy = bridge
        scene_load_s = time.perf_counter() - started
        record = run_instance(evaluator, bridge, task, index, args.output_dir, args.write_video, scene_load_s)
        return WorkResult(
            item_id=item.item_id,
            group_id=item.group_id,
            worker_id=args.worker_id,
            records=(record,),
            metadata={
                "suite": args.suite,
                "task": task,
                "elapsed_s": time.perf_counter() - started,
                "scene_load_s": scene_load_s,
                "inference_wait_s": bridge.inference_wait_s,
            },
        )

    exit_code = 0
    try:
        with PolicyClient(args.policy_socket) as client:
            WorkerLoop(
                worker_id=args.worker_id,
                queue_dir=args.queue_dir,
                state_dir=args.worker_state_dir,
                poll_interval_s=args.poll_interval,
                recycle_after_records=1,
            ).run(lambda item: evaluate(item, client))
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if state["evaluator"] is not None:
            if exit_code == 0:
                # Closes the environment and the simulator application, as the native runner's
                # ``with BatchedEvaluator(...)`` does; the application may end the process here.
                try:
                    state["evaluator"].__exit__(None, None, None)
                except BaseException:
                    traceback.print_exc()
            else:
                # Keep the failing status: only remove OmniGibson's temporary directory, which holds
                # decrypted assets.
                import omnigibson

                shutil.rmtree(omnigibson.tempdir, ignore_errors=True)
        os._exit(exit_code)


if __name__ == "__main__":
    main()
