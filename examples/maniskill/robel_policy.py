"""Signed joint and valve-axis transfer of a pretrained nine-DOF ROBEL DClaw actor."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

import numpy as np
from robel_model import ROBELActor, load_actor

from roboscope_kit import (
    ActionChunk,
    EpisodeContext,
    Observation,
    Policy,
    PolicyCapabilities,
    SessionContext,
    run,
)

TASKS = tuple(f"RotateValveLevel{level}-v1" for level in range(5))
# Native observation/controller: f1_0,f2_0,f3_0,f1_1,f2_1,f3_1,f1_2,f2_2,f3_2.
# ROBEL FF/MF/TH fingers correspond to native f1/f3/f2, including hinge-axis signs.
NATIVE_TO_SOURCE = np.array([0, 3, 6, 2, 5, 8, 1, 4, 7])
JOINT_SIGNS = np.array([1, -1, 1, -1, 1, 1, -1, 1, -1], dtype=np.float32)
SOURCE_TARGET_RANGE = np.tile(np.array([0.48, np.pi / 3, np.pi / 2], dtype=np.float32), 3)


@dataclass
class EpisodeState:
    previous_action: np.ndarray = field(default_factory=lambda: np.zeros(9, dtype=np.float32))
    desired_angle: float | None = None
    steps: int = 0
    controller_clipped_steps: int = 0
    controller_clipped_joints: int = 0


def source_observation(value: np.ndarray, state: EpisodeState) -> np.ndarray:
    qpos = value[:9][NATIVE_TO_SOURCE] * JOINT_SIGNS
    # Native revolute +X is rotated by Ry(pi/2), giving physical axis -Z.
    # The homologous ROBEL palm frame is rotated +pi/6 from the native world frame.
    w, x, y, z = value[47:51]
    valve_root_yaw = np.arctan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))
    angle = float(valve_root_yaw - value[40] + np.pi / 6)
    if state.desired_angle is None:
        state.desired_angle = angle
    target_error = (state.desired_angle - angle + np.pi) % (2.0 * np.pi) - np.pi
    return np.concatenate((qpos, np.array([np.cos(angle), np.sin(angle)], dtype=np.float32),
                           state.previous_action, np.array([target_error], dtype=np.float32)))


def native_action(value: np.ndarray, state: EpisodeState, action: np.ndarray) -> np.ndarray:
    normalized = np.clip(action, -1.0, 1.0).astype(np.float32)
    source_target = normalized * SOURCE_TARGET_RANGE
    target = np.empty(9, dtype=np.float32)
    target[NATIVE_TO_SOURCE] = source_target * JOINT_SIGNS
    native = (target - value[:9]) / np.float32(0.1)
    clipped = int(np.count_nonzero(np.abs(native) > 1.0))
    state.controller_clipped_steps += int(clipped > 0)
    state.controller_clipped_joints += clipped
    state.previous_action = normalized
    # Original screw policies track an internal moving target. Respect native direction and dt.
    state.desired_angle -= float(value[39]) * 0.5 / 20.0
    state.steps += 1
    return np.ascontiguousarray(np.clip(native, -1.0, 1.0), dtype=np.float32)


class ManiSkillROBELPolicy(Policy):
    def __init__(self) -> None:
        self.actor: ROBELActor | None = None
        self.episodes: dict[str, EpisodeState] = {}

    def load(self, context: SessionContext) -> PolicyCapabilities:
        if context.benchmark.name != "maniskill" or context.benchmark.protocol_version != "maniskill-v1":
            raise ValueError("The ROBEL transfer example serves ManiSkill DClaw valve profiles")
        self.actor = load_actor(context.checkpoint_path)
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
            raise ValueError(f"The ROBEL DClaw actor does not serve {context.task_id}")
        expected = {"robot_uid": "dclaw", "control_mode": "pd_joint_delta_pos",
                    "obs_mode": "state", "action_dimension": 9, "control_freq": 20}
        for name, value in expected.items():
            actual = context.metadata.get(name)
            if actual is not None and actual != value:
                raise ValueError(f"The ROBEL adapter requires {name}={value!r}")
        self.episodes[context.episode_id] = EpisodeState()

    def act(self, observation: Observation) -> ActionChunk:
        return self.act_batch([observation])[0]

    def act_batch(self, observations: list[Observation]) -> list[ActionChunk]:
        if self.actor is None:
            raise RuntimeError("Load the real ROBEL checkpoint before inference")
        if not 1 <= len(observations) <= 256:
            raise ValueError("Expected between 1 and 256 observations")
        values, inputs = [], []
        for observation in observations:
            value = np.asarray(observation.features["state.proprioception"], dtype=np.float32)
            if value.shape != (51,) or not np.isfinite(value).all():
                raise ValueError("Native RotateValve requires a finite 51D state")
            values.append(value)
            inputs.append(source_observation(value, self.episodes[observation.episode_id]))
        actions = self.actor(np.stack(inputs))
        if not np.isfinite(actions).all():
            raise ValueError("ROBEL actor produced non-finite actions")
        return [
            ActionChunk(native_action(value, self.episodes[observation.episode_id], action)[None])
            for observation, value, action in zip(observations, values, actions, strict=True)
        ]

    def end_episode(self, episode_id: str) -> None:
        state = self.episodes.pop(episode_id, None)
        if state is not None:
            print(json.dumps({"policy": "robel-dclaw-zero-shot-transfer", "episode_id": episode_id,
                              "steps": state.steps,
                              "controller_clipped_steps": state.controller_clipped_steps,
                              "controller_clipped_joints": state.controller_clipped_joints}),
                  file=sys.stderr, flush=True)

    def unload(self) -> None:
        self.actor = None
        self.episodes.clear()


if __name__ == "__main__":
    run(ManiSkillROBELPolicy())
