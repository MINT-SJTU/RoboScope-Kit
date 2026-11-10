"""HORA-RMA zero-shot transfer to native ManiSkill Allegro rotation profiles."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import numpy as np
import torch
from hora_model import HORAModel, load_hora

from roboscope_kit import (
    ActionChunk,
    EpisodeContext,
    Observation,
    Policy,
    PolicyCapabilities,
    SessionContext,
    run,
)

TASKS = tuple(f"RotateSingleObjectInHandLevel{level}-v1" for level in range(4))
# SAPIEN uses breadth-first articulation order; the controller uses joint_0.0..15.0.
QPOS_TO_CONTROLLER = np.array([0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15])
# HORA deploy.py orders fingers as index, thumb, middle, ring.
CONTROLLER_TO_HORA = np.array([0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10, 11])
HORA_TO_CONTROLLER = np.argsort(CONTROLLER_TO_HORA)
LOWER = np.array([
    -0.470, -0.196, -0.174, -0.227, 0.263, -0.105, -0.189, -0.162,
    -0.470, -0.196, -0.174, -0.227, -0.470, -0.196, -0.174, -0.227,
], dtype=np.float32)
UPPER = np.array([
    0.470, 1.610, 1.709, 1.618, 1.396, 1.163, 1.644, 1.719,
    0.470, 1.610, 1.709, 1.618, 0.470, 1.610, 1.709, 1.618,
], dtype=np.float32)


@dataclass
class EpisodeState:
    target: np.ndarray | None = None
    history: np.ndarray | None = None
    steps: int = 0
    controller_clipped_steps: int = 0
    controller_clipped_joints: int = 0
    source_target_clipped_joints: int = 0


def update_history(state: EpisodeState, qpos: np.ndarray) -> np.ndarray:
    normalized = (2.0 * qpos - UPPER - LOWER) / (UPPER - LOWER)
    if state.target is None:
        state.target = qpos.copy()
        state.history = np.tile(np.concatenate((normalized, state.target)), (30, 1))
    else:
        state.history = np.concatenate((
            state.history[1:], np.concatenate((normalized, state.target))[None],
        ))
    return state.history


def native_action(state: EpisodeState, qpos: np.ndarray, source_action: np.ndarray) -> np.ndarray:
    """Translate HORA cumulative radian targets into the frozen current-relative PD command."""
    desired = state.target + np.clip(source_action, -1.0, 1.0) / np.float32(24.0)
    state.source_target_clipped_joints += int(np.count_nonzero((desired < LOWER) | (desired > UPPER)))
    state.target = np.clip(desired, LOWER, UPPER)
    action = (state.target[HORA_TO_CONTROLLER] - qpos[HORA_TO_CONTROLLER]) / np.float32(0.1)
    clipped = int(np.count_nonzero(np.abs(action) > 1.0))
    state.controller_clipped_joints += clipped
    state.controller_clipped_steps += int(clipped > 0)
    state.steps += 1
    return np.ascontiguousarray(np.clip(action, -1.0, 1.0), dtype=np.float32)


class ManiSkillHORAPolicy(Policy):
    def __init__(self) -> None:
        self.model: HORAModel | None = None
        self.episodes: dict[str, EpisodeState] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self, context: SessionContext) -> PolicyCapabilities:
        if context.benchmark.name != "maniskill" or context.benchmark.protocol_version != "maniskill-v1":
            raise ValueError("The HORA example serves ManiSkill Allegro rotation profiles")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.model = load_hora(context.checkpoint_path, self.device)
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
        if context.task_id not in TASKS:
            raise ValueError(f"The HORA Allegro student does not serve {context.task_id}")
        expected = {"robot_uid": "allegro_hand_right_touch", "control_mode": "pd_joint_delta_pos",
                    "obs_mode": "state", "action_dimension": 16, "control_freq": 20}
        for name, value in expected.items():
            actual = context.metadata.get(name)
            if actual is not None and actual != value:
                raise ValueError(f"The HORA adapter requires {name}={value!r}")
        self.episodes[context.episode_id] = EpisodeState()

    def act(self, observation: Observation) -> ActionChunk:
        return self.act_batch([observation])[0]

    def act_batch(self, observations: list[Observation]) -> list[ActionChunk]:
        if self.model is None:
            raise RuntimeError("Load the pinned real HORA student before inference")
        if not 1 <= len(observations) <= 256:
            raise ValueError("Expected between 1 and 256 observations")
        histories, joint_positions = [], []
        for observation in observations:
            value = np.asarray(observation.features["state.proprioception"], dtype=np.float32)
            if value.shape != (105,) or not np.isfinite(value).all():
                raise ValueError("Native Allegro rotation requires a finite 105D state")
            qpos = value[:16][QPOS_TO_CONTROLLER][CONTROLLER_TO_HORA]
            joint_positions.append(qpos)
            histories.append(update_history(self.episodes[observation.episode_id], qpos))
        history = torch.as_tensor(np.stack(histories), device=self.device, dtype=torch.float32)
        with torch.inference_mode():
            actions = self.model(history).cpu().numpy()
        if not np.isfinite(actions).all():
            raise ValueError("HORA produced non-finite actions")
        return [
            ActionChunk(native_action(self.episodes[observation.episode_id], qpos, action)[None])
            for observation, qpos, action in zip(observations, joint_positions, actions, strict=True)
        ]

    def end_episode(self, episode_id: str) -> None:
        state = self.episodes.pop(episode_id, None)
        if state is not None:
            print(json.dumps({"policy": "hora-rma-zero-shot-transfer", "episode_id": episode_id,
                              "steps": state.steps,
                              "controller_clipped_steps": state.controller_clipped_steps,
                              "controller_clipped_joints": state.controller_clipped_joints,
                              "source_target_clipped_joints": state.source_target_clipped_joints}),
                  file=sys.stderr, flush=True)

    def unload(self) -> None:
        self.model = None
        self.episodes.clear()


if __name__ == "__main__":
    run(ManiSkillHORAPolicy())
