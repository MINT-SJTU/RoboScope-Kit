"""Community Go2 velocity actor transferred to the observed native Reach goal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from go2_model import Go2ActorCritic, load_model, policy_observation

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
    observation_indices: np.ndarray
    action_indices: np.ndarray
    previous_action: np.ndarray


class ManiSkillGo2Policy(Policy):
    def __init__(self) -> None:
        self.model: Go2ActorCritic | None = None
        self.metadata: dict[str, Any] = {}
        self.episodes: dict[str, EpisodeState] = {}
        self.device = "cuda"

    def load(self, context: SessionContext) -> PolicyCapabilities:
        if context.benchmark.name != "maniskill" or context.benchmark.protocol_version != "maniskill-v1":
            raise ValueError("The Go2 example requires the ManiSkill v1 protocol")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.model, self.metadata = load_model(context.checkpoint_path, self.device)
        self.episodes.clear()
        return PolicyCapabilities(
            supported_benchmarks=("maniskill",),
            supported_suites={"maniskill": (context.benchmark.suite,)},
            required_features=("state.proprioception", "state.root_pose", "state.root_angular_velocity"),
            max_batch_size=256,
            supported_action_chunk_sizes=(1,),
            action_representations={"maniskill": "maniskill.native-controller-v1"},
        )

    def reset(self, context: EpisodeContext) -> None:
        if self.model is None or context.task_id != self.metadata["task_id"]:
            raise ValueError("Load the pinned Go2 Reach transfer model before reset")
        for key in ("robot_uid", "control_mode"):
            actual = context.metadata.get(key)
            if actual is not None and actual != self.metadata[key]:
                raise ValueError(f"Go2 model requires {key}={self.metadata[key]!r}")
        state_names = context.metadata.get("active_joint_names")
        action_names = context.metadata.get("action_joint_names")
        expected = self.metadata["policy_joint_names"]
        for names in (state_names, action_names):
            if not isinstance(names, (tuple, list)) or len(names) != 12 or set(names) != set(expected):
                raise ValueError("Go2 requires explicit native state and controller joint names")
        self.episodes[context.episode_id] = EpisodeState(
            np.asarray([state_names.index(name) for name in expected]),
            np.asarray([action_names.index(name) for name in expected]),
            np.zeros(12, dtype=np.float32),
        )

    def act(self, observation: Observation) -> ActionChunk:
        if self.model is None:
            raise RuntimeError("Policy has not loaded the pinned checkpoint")
        episode = self.episodes[observation.episode_id]
        state = np.asarray(observation.features["state.proprioception"], dtype=np.float32)
        actor_input = policy_observation(
            state,
            np.asarray(observation.features["state.root_pose"], dtype=np.float32),
            np.asarray(observation.features["state.root_angular_velocity"], dtype=np.float32),
            episode.observation_indices, episode.previous_action, self.metadata,
        )
        with torch.inference_mode():
            batch = torch.as_tensor(actor_input[None], device=self.device)
            predicted = self.model.actor(batch)[0].cpu().numpy()
        if not np.isfinite(predicted).all():
            raise ValueError("Go2 actor produced non-finite actions")
        predicted = np.clip(predicted, -self.metadata["clip_actions"], self.metadata["clip_actions"])
        # Reproduce the published one-step actuator latency. Observation history
        # retains the newly predicted action, while this tick executes its predecessor.
        targets = (
            np.asarray(self.metadata["policy_default_angles"], dtype=np.float32)
            + self.metadata["action_scale"] * episode.previous_action
        )
        delta = (targets - state[episode.observation_indices]) / self.metadata["native_delta_range"]
        action = np.empty(12, dtype=np.float32)
        action[episode.action_indices] = np.clip(delta, -1, 1)
        episode.previous_action = np.ascontiguousarray(predicted, dtype=np.float32)
        return ActionChunk(action[None])

    def act_batch(self, observations: list[Observation]) -> list[ActionChunk]:
        if not 1 <= len(observations) <= 256:
            raise ValueError("Expected a batch of 1 to 256 observations")
        return [self.act(observation) for observation in observations]

    def end_episode(self, episode_id: str) -> None:
        self.episodes.pop(episode_id, None)

    def unload(self) -> None:
        self.episodes.clear()
        self.model = None
        self.metadata.clear()


if __name__ == "__main__":
    run(ManiSkillGo2Policy())
