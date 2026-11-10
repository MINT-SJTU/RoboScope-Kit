"""Explicit physical torque-to-PD transfer of pretrained TriFinger cube experts."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

import numpy as np
import torch
from trifinger_model import TriFingerExpert, load_expert

from roboscope_kit import (
    ActionChunk,
    EpisodeContext,
    Observation,
    Policy,
    PolicyCapabilities,
    SessionContext,
    run,
)

TASK_KIND = {f"TriFingerRotateCubeLevel{level}-v1": "push" if level < 2 else "lift"
             for level in range(5)}
# Native observed joints are breadth-first; RRC and controller commands are finger-first.
QPOS_TO_CONTROLLER = np.array([0, 3, 6, 1, 4, 7, 2, 5, 8])
CORNERS = np.array([[1.0 if (index >> axis) & 1 == 0 else -1.0 for axis in range(3)]
                    for index in range(8)], dtype=np.float32) * np.float32(0.0325)


@dataclass
class EpisodeState:
    kind: str
    previous_torque: np.ndarray = field(default_factory=lambda: np.zeros(9, dtype=np.float32))
    steps: int = 0
    controller_clipped_steps: int = 0
    controller_clipped_joints: int = 0
    source_clipped_features: int = 0


def cube_keypoints(position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
    """The source's eight .065 m cube corners, in its binary/reversed-axis order."""
    quaternion = quaternion_wxyz.astype(np.float64)
    quaternion /= np.linalg.norm(quaternion)
    vector = quaternion[1:]
    corners = CORNERS.astype(np.float64)
    crossed = 2.0 * np.cross(vector, corners)
    rotated = corners + quaternion[0] * crossed + np.cross(vector, crossed)
    return np.asarray(rotated + position, dtype=np.float32).reshape(-1)


def consumed_observation(value: np.ndarray, state: EpisodeState) -> tuple[np.ndarray, np.ndarray]:
    qpos = value[:9][QPOS_TO_CONTROLLER]
    qvel = value[9:18][QPOS_TO_CONTROLLER]
    # Native state: qpos9, qvel9, tip poses21, tip velocities9, goal pose7, object pose7.
    goal_pos, goal_q = value[48:51], value[51:55]
    obj_pos, obj_q = value[55:58], value[58:62]
    goal = goal_pos if state.kind == "push" else cube_keypoints(goal_pos, goal_q)
    features = np.concatenate((qpos, qvel, cube_keypoints(obj_pos, obj_q), goal, state.previous_torque))
    return features, qvel


def native_action(state: EpisodeState, torque: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    """Invert native Kp=100, Kd=10 at observation time, then honor its delta bounds."""
    state.previous_torque = np.clip(torque, -0.397, 0.397).astype(np.float32)
    desired_delta = (state.previous_torque + np.float32(10.0) * qvel) / np.float32(100.0)
    action = desired_delta / np.float32(0.1)
    clipped = int(np.count_nonzero(np.abs(action) > 1.0))
    state.controller_clipped_steps += int(clipped > 0)
    state.controller_clipped_joints += clipped
    state.steps += 1
    return np.ascontiguousarray(np.clip(action, -1.0, 1.0), dtype=np.float32)


class ManiSkillTriFingerPolicy(Policy):
    def __init__(self) -> None:
        self.models: dict[str, TriFingerExpert] = {}
        self.episodes: dict[str, EpisodeState] = {}
        self.device = "cpu"

    def load(self, context: SessionContext) -> PolicyCapabilities:
        if context.benchmark.name != "maniskill" or context.benchmark.protocol_version != "maniskill-v1":
            raise ValueError("The TriFinger example serves ManiSkill native cube rotation profiles")
        torch.set_num_threads(1)
        self.models = {kind: load_expert(context.checkpoint_path, kind, self.device)
                       for kind in ("push", "lift")}
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
        if context.task_id not in TASK_KIND:
            raise ValueError(f"The TriFinger experts do not serve {context.task_id}")
        expected = {"robot_uid": "trifingerpro", "control_mode": "pd_joint_delta_pos",
                    "obs_mode": "state", "action_dimension": 9, "control_freq": 20}
        for name, value in expected.items():
            actual = context.metadata.get(name)
            if actual is not None and actual != value:
                raise ValueError(f"The TriFinger adapter requires {name}={value!r}")
        self.episodes[context.episode_id] = EpisodeState(TASK_KIND[context.task_id])

    def act(self, observation: Observation) -> ActionChunk:
        return self.act_batch([observation])[0]

    def act_batch(self, observations: list[Observation]) -> list[ActionChunk]:
        if not self.models:
            raise RuntimeError("Load the pinned real TriFinger experts before inference")
        if not 1 <= len(observations) <= 256:
            raise ValueError("Expected between 1 and 256 observations")
        groups: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {}
        for index, observation in enumerate(observations):
            value = np.asarray(observation.features["state.proprioception"], dtype=np.float32)
            if value.shape != (62,) or not np.isfinite(value).all():
                raise ValueError("Native TriFinger rotation requires a finite 62D state")
            state = self.episodes[observation.episode_id]
            features, qvel = consumed_observation(value, state)
            groups.setdefault(state.kind, []).append((index, features, qvel))
        result: list[ActionChunk | None] = [None] * len(observations)
        for kind, entries in groups.items():
            model = self.models[kind]
            physical = torch.as_tensor(np.stack([entry[1] for entry in entries]), dtype=torch.float32)
            clipped = ((physical < model.c0) | (physical > model.c0 + model.c1)).sum(dim=-1).tolist()
            with torch.inference_mode():
                torques = model(physical).numpy()
            if not np.isfinite(torques).all():
                raise ValueError("TriFinger expert produced non-finite torques")
            for (index, _, qvel), torque, clip_count in zip(entries, torques, clipped, strict=True):
                state = self.episodes[observations[index].episode_id]
                state.source_clipped_features += clip_count
                result[index] = ActionChunk(native_action(state, torque, qvel)[None])
        return [chunk for chunk in result if chunk is not None]

    def end_episode(self, episode_id: str) -> None:
        state = self.episodes.pop(episode_id, None)
        if state is not None:
            print(json.dumps({"policy": "rrc-trifinger-zero-shot-transfer", "episode_id": episode_id,
                              "source_expert": state.kind, "steps": state.steps,
                              "controller_clipped_steps": state.controller_clipped_steps,
                              "controller_clipped_joints": state.controller_clipped_joints,
                              "source_clipped_features": state.source_clipped_features}),
                  file=sys.stderr, flush=True)

    def unload(self) -> None:
        self.models.clear()
        self.episodes.clear()


if __name__ == "__main__":
    run(ManiSkillTriFingerPolicy())
