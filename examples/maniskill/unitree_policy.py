"""Unitree RL Gym standing transfer policies with explicit root-state inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from unitree_model import LoadedUnitree, load_model_bundle, native_action, policy_observation

from roboscope_kit import (
    ActionChunk,
    EpisodeContext,
    Observation,
    Policy,
    PolicyCapabilities,
    SessionContext,
    run,
)


@dataclass
class EpisodeState:
    model: LoadedUnitree
    actor: torch.jit.ScriptModule
    leg_indices: np.ndarray
    controller_indices: np.ndarray
    previous_action: np.ndarray
    control_freq: float


class ManiSkillUnitreePolicy(Policy):
    def __init__(self) -> None:
        self.models: dict[str, LoadedUnitree] = {}
        self.episodes: dict[str, EpisodeState] = {}
        self.device = "cuda"

    def load(self, context: SessionContext) -> PolicyCapabilities:
        if context.benchmark.name != "maniskill" or context.benchmark.protocol_version != "maniskill-v1":
            raise ValueError("The Unitree example requires the ManiSkill v1 protocol")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.models = load_model_bundle(context.checkpoint_path, self.device)
        self.episodes.clear()
        return PolicyCapabilities(
            supported_benchmarks=("maniskill",),
            supported_suites={"maniskill": (context.benchmark.suite,)},
            required_features=(
                "state.proprioception", "state.root_pose", "state.root_angular_velocity",
            ),
            max_batch_size=256,
            supported_action_chunk_sizes=(1,),
            action_representations={"maniskill": "maniskill.native-controller-v1"},
        )

    def reset(self, context: EpisodeContext) -> None:
        if context.task_id not in self.models:
            raise ValueError(f"No pinned Unitree transfer checkpoint for {context.task_id}")
        model = self.models[context.task_id]
        for key in ("robot_uid", "control_mode", "action_dimension"):
            actual = context.metadata.get(key)
            if actual is not None and actual != model.metadata[key]:
                raise ValueError(f"{context.task_id} requires {key}={model.metadata[key]!r}")
        names = context.metadata.get("active_joint_names")
        expected = model.metadata["native_controller_joint_names"]
        if not isinstance(names, (tuple, list)) or len(names) != len(expected) or set(names) != set(expected):
            raise ValueError("Active native joint names must match the pinned robot")
        action_names = context.metadata.get("action_joint_names")
        if action_names is not None and list(action_names) != expected:
            raise ValueError("Native controller joint order differs from the pinned robot")
        control_freq = float(context.metadata["control_freq"])
        if not np.isfinite(control_freq) or control_freq <= 0:
            raise ValueError("Native control frequency must be positive")
        self.episodes[context.episode_id] = EpisodeState(
            model=model,
            actor=model.episode_actor(),
            leg_indices=np.asarray([names.index(name) for name in model.metadata["policy_joint_names"]]),
            controller_indices=np.asarray([names.index(name) for name in expected]),
            previous_action=np.zeros(len(model.metadata["policy_joint_names"]), dtype=np.float32),
            control_freq=control_freq,
        )

    def act(self, observation: Observation) -> ActionChunk:
        episode = self.episodes[observation.episode_id]
        metadata = episode.model.metadata
        state = np.asarray(observation.features["state.proprioception"], dtype=np.float32)
        actor_input = policy_observation(
            state,
            np.asarray(observation.features["state.root_pose"], dtype=np.float32),
            np.asarray(observation.features["state.root_angular_velocity"], dtype=np.float32),
            episode.previous_action,
            episode.leg_indices,
            observation.step / episode.control_freq,
            metadata,
        )
        with torch.inference_mode():
            action = episode.actor(torch.as_tensor(actor_input[None], device=self.device))[0].cpu().numpy()
        episode.previous_action = action.copy()
        return ActionChunk(native_action(action, state, episode.controller_indices, metadata)[None])

    def act_batch(self, observations: list[Observation]) -> list[ActionChunk]:
        if not 1 <= len(observations) <= 256:
            raise ValueError("Expected a batch of 1 to 256 observations")
        return [self.act(observation) for observation in observations]

    def end_episode(self, episode_id: str) -> None:
        self.episodes.pop(episode_id, None)

    def unload(self) -> None:
        self.episodes.clear()
        self.models.clear()


if __name__ == "__main__":
    run(ManiSkillUnitreePolicy())
