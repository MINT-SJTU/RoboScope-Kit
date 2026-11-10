"""Compose unchanged official 4D PPO actors for canonical dual-Panda state tasks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from ppo_model import LoadedPPO, load_model_bundle

from roboscope_kit import (
    ActionChunk,
    EpisodeContext,
    Observation,
    Policy,
    PolicyCapabilities,
    SessionContext,
    run,
)

TASK_PROFILES = {
    "TwoRobotPickCube-v1": "two_robot_pickcube_state",
    "TwoRobotStackCube-v1": "two_robot_stackcube_state",
}
SOURCE_TASKS = {
    "TwoRobotPickCube-v1": ("PushCube-v1", "PushCube-v1"),
    "TwoRobotStackCube-v1": ("StackCube-v1", "PushCube-v1"),
}
SOURCE_DIMENSIONS = {"PushCube-v1": 35, "StackCube-v1": 48}
# Final poses set by the pinned TableSceneBuilder.initialize, not _load_agent.
SOURCE_BASE = np.array([-0.615, 0.0, 0.0], dtype=np.float32)
TARGET_BASES = np.array([[0.0, -0.75, 0.0], [0.0, 0.75, 0.0]], dtype=np.float32)
ROTATIONS = np.array([
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
], dtype=np.float32)
HALF_YAW = np.float32(np.sqrt(0.5))
QUATERNIONS = np.array([
    [HALF_YAW, 0.0, 0.0, -HALF_YAW],
    [HALF_YAW, 0.0, 0.0, HALF_YAW],
], dtype=np.float32)


def _point(value: np.ndarray, arm: int) -> np.ndarray:
    return SOURCE_BASE + ROTATIONS[arm] @ (value - TARGET_BASES[arm])


def _pose(value: np.ndarray, arm: int) -> np.ndarray:
    """Apply T_source_base * inverse(T_target_base), including wxyz orientation."""
    if not np.isclose(np.linalg.norm(value[3:]), 1.0, atol=1e-3, rtol=0.0):
        raise ValueError("Expected a measured unit wxyz pose quaternion")
    w, x, y, z = QUATERNIONS[arm]
    a, b, c, d = value[3:]
    quaternion = np.array([
        w*a - x*b - y*c - z*d,
        w*b + x*a + y*d - z*c,
        w*c - x*d + y*a + z*b,
        w*d + x*c - y*b + z*a,
    ], dtype=np.float32)
    return np.concatenate((_point(value[:3], arm), quaternion))


def project_actor_inputs(task_id: str, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use only the unchanged canonical observation and pinned static base calibration."""
    if task_id not in TASK_PROFILES:
        raise ValueError("Expected a supported canonical dual-Panda task")
    value = np.asarray(state, dtype=np.float32)
    dimension = 66 if task_id == "TwoRobotPickCube-v1" else 76
    if value.shape != (dimension,) or not np.isfinite(value).all():
        raise ValueError(f"Expected the complete finite canonical state of {dimension} values")
    tcp = (_pose(value[36:43], 0), _pose(value[43:50], 1))
    if task_id == "TwoRobotPickCube-v1":
        # The goal is observed as cube-to-goal, not a separately exposed world point.
        # Specify the FP32 reconstruction; it need not invert the hidden subtraction bitwise.
        goal = np.add(value[50:53], value[63:66], dtype=np.float32)
        return tuple(np.ascontiguousarray(np.concatenate((
            value[18*arm:18*(arm+1)], tcp[arm], _point(goal, arm),
            _pose(value[50:57], arm),
        )), dtype=np.float32) for arm in (0, 1))
    cube_a, cube_b = value[53:60], value[60:67]
    left = np.concatenate((
        value[:18], tcp[0], _pose(cube_a, 0), _pose(cube_b, 0),
        ROTATIONS[0] @ value[67:70],
        ROTATIONS[0] @ np.subtract(cube_b[:3], value[36:39], dtype=np.float32),
        ROTATIONS[0] @ value[73:76],
    ))
    right = np.concatenate((
        value[18:36], tcp[1], _point(value[50:53], 1), _pose(cube_b, 1),
    ))
    return np.ascontiguousarray(left, dtype=np.float32), np.ascontiguousarray(right, dtype=np.float32)


class ManiSkillDualPandaPolicy(Policy):
    def __init__(self) -> None:
        self.models: dict[str, LoadedPPO] = {}
        self.episodes: dict[str, str] = {}
        self.device = "cuda"

    def load(self, context: SessionContext) -> PolicyCapabilities:
        if context.benchmark.name != "maniskill" or context.benchmark.protocol_version != "maniskill-v1":
            raise ValueError("The dual-Panda transfer requires ManiSkill v1")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        loaded = load_model_bundle(context.checkpoint_path, self.device)
        catalog = json.loads(Path(__file__).with_name("official-models.json").read_text(encoding="utf-8"))
        for task_id, dimension in SOURCE_DIMENSIONS.items():
            model = loaded.get(task_id)
            if model is None or (model.observation_dim, model.action_dim) != (dimension, 4):
                raise ValueError(f"Expected the complete official {task_id} {dimension}D/4D actor")
            if model.sha256 != catalog["models"][task_id]["sha256"]:
                raise ValueError(f"The {task_id} weights differ from the pinned official checkpoint")
        self.models = {task_id: loaded[task_id] for task_id in SOURCE_DIMENSIONS}
        self.episodes.clear()
        return PolicyCapabilities(
            supported_benchmarks=("maniskill",),
            supported_suites={"maniskill": (context.benchmark.suite,)},
            required_features=("state.proprioception",),
            max_batch_size=256,
            supported_action_chunk_sizes=(1,),
            action_representations={"maniskill": "maniskill.native-controller-v1"},
        )

    def reset(self, context: EpisodeContext) -> None:
        if not self.models or context.task_id not in TASK_PROFILES:
            raise ValueError("Load the real source actors before a supported dual-Panda episode")
        for key, expected in (
            ("profile_id", TASK_PROFILES[context.task_id]), ("obs_mode", "state"),
            ("control_mode", "pd_ee_delta_pos"), ("action_dimension", 8),
            ("sim_freq", 100), ("control_freq", 20),
        ):
            if context.metadata.get(key) != expected:
                raise ValueError(f"The canonical dual-Panda transfer requires {key}={expected!r}")
        if context.metadata.get("robot_uid") != ["panda_wristcam", "panda_wristcam"]:
            raise ValueError("Expected the pinned left/right Panda wrist-camera articulations")
        self.episodes[context.episode_id] = context.task_id

    def act(self, observation: Observation) -> ActionChunk:
        return self.act_batch([observation])[0]

    def act_batch(self, observations: list[Observation]) -> list[ActionChunk]:
        if not self.models or not 1 <= len(observations) <= 256:
            raise ValueError("Expected a loaded Policy and between 1 and 256 observations")
        groups: dict[str, list[tuple[int, int, np.ndarray]]] = {}
        for index, observation in enumerate(observations):
            task_id = self.episodes[observation.episode_id]
            inputs = project_actor_inputs(task_id, observation.features["state.proprioception"])
            for arm, (source_task, values) in enumerate(zip(SOURCE_TASKS[task_id], inputs, strict=True)):
                groups.setdefault(source_task, []).append((index, arm, values))
        actions = np.empty((len(observations), 8), dtype=np.float32)
        for source_task, items in groups.items():
            batch = torch.as_tensor(np.stack([row[2] for row in items]),
                                    device=self.device, dtype=torch.float32)
            with torch.inference_mode():
                output = self.models[source_task].agent.actor_mean(batch).cpu().numpy()
            if output.shape != (len(items), 4) or not np.isfinite(output).all():
                raise ValueError("The source actor must return four finite native controller values")
            for (index, arm, _), action in zip(items, output, strict=True):
                # Actor output is already in the corresponding root frame. Native bounds apply.
                actions[index, 4*arm:4*(arm+1)] = action
        return [ActionChunk(np.ascontiguousarray(action[None], dtype=np.float32)) for action in actions]

    def end_episode(self, episode_id: str) -> None:
        self.episodes.pop(episode_id, None)

    def unload(self) -> None:
        self.models.clear()
        self.episodes.clear()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    run(ManiSkillDualPandaPolicy())
