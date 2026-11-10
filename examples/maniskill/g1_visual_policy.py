"""Image-conditioned Octo transfer to the native fixed-leg G1 upper body."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from g1_kinematics import URDF_NAME, G1Kinematics, rotation
from panda_octo_model import ManiSkillFinetunedOcto

from roboscope_kit import (
    ActionChunk,
    EpisodeContext,
    Observation,
    Policy,
    PolicyCapabilities,
    SessionContext,
    run,
)

TASK_HANDS = {
    "UnitreeG1PlaceAppleInBowl-v1": ("right",),
    "UnitreeG1TransportBox-v1": ("left", "right"),
}
FLEXION_JOINTS = ("one", "two", "three", "four", "five", "six")
HAND_JOINTS = {f"{side}_{finger}_joint" for side in ("left", "right")
               for finger in ("zero", *FLEXION_JOINTS)}


@dataclass
class EpisodeState:
    kinematics: G1Kinematics
    controller_indices: np.ndarray
    controller_ranges: np.ndarray
    hands: tuple[str, ...]
    stream_ids: list[str]
    instruction: str | None = None
    steps: int = 0
    clipped_joints: int = 0
    maximum_commanded_position_residual_m: float = 0.0


def cartesian_goal(current: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Retarget the pinned GPU controller's scaled root-frame differential intent."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape != (7,) or not np.isfinite(raw).all():
        raise ValueError("The source policy must provide seven finite EE/gripper values")
    result = np.asarray(current, dtype=np.float64).copy()
    result[:3, 3] += 0.1 * np.clip(raw[:3], -1, 1)
    # The pinned controller norm-clips raw rotation, then uses rot_lower=-0.1.
    # Its GPU differential IK consumes this as a root-frame angular displacement.
    angular = -0.1 * raw[3:6] / max(1.0, float(np.linalg.norm(raw[3:6])))
    result[:3, :3] = rotation(angular) @ result[:3, :3]
    return result


def joint_action(qpos: np.ndarray, commands: dict[str, np.ndarray], episode: EpisodeState) -> np.ndarray:
    kinematics = episode.kinematics
    transforms, _ = kinematics.forward(qpos)
    goals = {f"{side}_tcp_link": cartesian_goal(transforms[f"{side}_tcp_link"], raw)
             for side, raw in commands.items()}
    target = kinematics.inverse(qpos, goals)
    for side, raw in commands.items():
        amount = (1.0 - float(np.clip(raw[6], -1, 1))) / 2.0
        for finger in FLEXION_JOINTS:
            index = kinematics.indices[f"{side}_{finger}_joint"]
            lower, upper = kinematics.lower[index], kinematics.upper[index]
            closed = lower if abs(lower) > abs(upper) else upper
            target[index] = amount * closed
        # The zero/thumb-abduction joint and the uncommanded hand keep current targets.
    normalized = (target-qpos)[episode.controller_indices] / episode.controller_ranges
    episode.clipped_joints += int(np.count_nonzero(np.abs(normalized) > 1))
    normalized = np.clip(normalized, -1, 1)
    commanded = qpos.astype(np.float64).copy()
    commanded[episode.controller_indices] += normalized * episode.controller_ranges
    achieved, _ = kinematics.forward(commanded)
    episode.maximum_commanded_position_residual_m = max(episode.maximum_commanded_position_residual_m,
        *(float(np.linalg.norm(achieved[link][:3, 3]-goal[:3, 3])) for link, goal in goals.items()))
    episode.steps += 1
    return np.ascontiguousarray(normalized, dtype=np.float32)


class ManiSkillG1VisualPolicy(Policy):
    def __init__(self) -> None:
        self.model: ManiSkillFinetunedOcto | None = None
        self.checkpoint: Path | None = None
        self.episodes: dict[str, EpisodeState] = {}
        self.kinematics: dict[tuple[str, ...], G1Kinematics] = {}

    def load(self, context: SessionContext) -> PolicyCapabilities:
        if context.benchmark.name != "maniskill" or context.benchmark.protocol_version != "maniskill-v1":
            raise ValueError("G1 visual transfer requires the ManiSkill v1 protocol")
        self.checkpoint = Path(context.checkpoint_path).resolve(strict=True)
        self.model = ManiSkillFinetunedOcto(self.checkpoint)
        self.episodes.clear()
        self.kinematics.clear()
        return PolicyCapabilities(
            supported_benchmarks=("maniskill",),
            supported_suites={"maniskill": (context.benchmark.suite,)},
            required_features=("images.primary", "language.instruction", "state.proprioception"),
            max_batch_size=16,
            supported_action_chunk_sizes=(1,),
            action_representations={"maniskill": "maniskill.native-controller-v1"},
        )

    def reset(self, context: EpisodeContext) -> None:
        if self.model is None or self.checkpoint is None or context.task_id not in TASK_HANDS:
            raise ValueError("Load the real model before resetting a supported G1 manipulation task")
        if context.metadata.get("robot_uid") not in {
            "unitree_g1_simplified_upper_body", "unitree_g1_simplified_upper_body_with_head_camera",
        }:
            raise ValueError("The transfer requires the pinned native G1 upper-body articulation")
        for key, expected in (("control_mode", "pd_joint_delta_pos"), ("action_dimension", 25),
                              ("sim_freq", 100), ("control_freq", 20)):
            if context.metadata.get(key) != expected:
                raise ValueError(f"G1 visual transfer requires {key}={expected!r}")
        if context.metadata.get("obs_mode") not in {"rgb", "rgb+segmentation"}:
            raise ValueError("G1 visual transfer requires a declared RGB profile")
        self.end_episode(context.episode_id)
        names = tuple(context.metadata["active_joint_names"])
        action_names = list(context.metadata["action_joint_names"])
        if names not in self.kinematics:
            self.kinematics[names] = G1Kinematics(self.checkpoint / URDF_NAME, list(names))
        kinematics = self.kinematics[names]
        if len(action_names) != 25 or set(action_names) != set(names):
            raise ValueError("Actual native controller identities must match all 25 G1 joints")
        hands = TASK_HANDS[context.task_id]
        self.episodes[context.episode_id] = EpisodeState(
            kinematics, np.asarray([names.index(name) for name in action_names]),
            np.asarray([0.5 if name in HAND_JOINTS else 0.2 for name in action_names]), hands,
            [f"{context.episode_id}/{side}" for side in hands],
        )

    def act(self, observation: Observation) -> ActionChunk:
        return self.act_batch([observation])[0]

    def act_batch(self, observations: list[Observation]) -> list[ActionChunk]:
        if self.model is None or not 1 <= len(observations) <= 16:
            raise ValueError("Load the model before a batch of one to sixteen observations")
        stream_ids, images = [], []
        for observation in observations:
            episode = self.episodes[observation.episode_id]
            instruction = observation.features["language.instruction"]
            if episode.instruction is None:
                if not isinstance(instruction, str) or not instruction:
                    raise ValueError("The native task instruction must be nonempty")
                episode.instruction = instruction
                for side, stream in zip(episode.hands, episode.stream_ids, strict=True):
                    self.model.reset(stream, f"Using the {side} hand: {instruction}")
            elif instruction != episode.instruction:
                raise ValueError("The native instruction changed during the episode")
            for stream in episode.stream_ids:
                stream_ids.append(stream)
                images.append(np.asarray(observation.features["images.primary"]))
        # This checkpoint has no proprioception tokenizer. Only real images/language
        # are submitted to Octo; G1 qpos is used separately by physical retargeting.
        raw_actions = iter(self.model.act_batch(stream_ids, images))
        results = []
        for observation in observations:
            episode = self.episodes[observation.episode_id]
            state = np.asarray(observation.features["state.proprioception"], dtype=np.float32)
            if state.ndim != 1 or len(state) < 50 or not np.isfinite(state).all():
                raise ValueError("Expected actual G1 joint positions/velocities in native state order")
            commands = {side: next(raw_actions) for side in episode.hands}
            action = joint_action(state[:25], commands, episode)
            results.append(ActionChunk(action[None]))
        return results

    def end_episode(self, episode_id: str) -> None:
        episode = self.episodes.pop(episode_id, None)
        if episode is not None:
            if self.model is not None:
                for stream in episode.stream_ids:
                    self.model.end_episode(stream)
            print(json.dumps({"episode_id": episode_id, "transfer": "panda-octo-to-g1",
                "steps": episode.steps, "controller_clipped_joints": episode.clipped_joints,
                "maximum_commanded_position_residual_m": episode.maximum_commanded_position_residual_m}),
                file=sys.stderr, flush=True)

    def unload(self) -> None:
        for episode_id in list(self.episodes):
            self.end_episode(episode_id)
        self.kinematics.clear()
        self.model = None
        self.checkpoint = None


if __name__ == "__main__":
    run(ManiSkillG1VisualPolicy())
