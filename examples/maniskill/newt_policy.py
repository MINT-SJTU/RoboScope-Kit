"""SDK policy for pinned NEWT state policy priors on native ManiSkill tasks."""

from __future__ import annotations

import numpy as np
import torch
from newt_model import LoadedNewt, load_model_bundle

from roboscope_kit import (
    ActionChunk,
    EpisodeContext,
    Observation,
    Policy,
    PolicyCapabilities,
    SessionContext,
    run,
)


class ManiSkillNewtPolicy(Policy):
    def __init__(self) -> None:
        self.models: dict[str, LoadedNewt] = {}
        self.episode_tasks: dict[str, str] = {}
        self.held_actions: dict[str, np.ndarray] = {}
        self.device = "cuda"

    def load(self, context: SessionContext) -> PolicyCapabilities:
        if context.benchmark.name != "maniskill" or context.benchmark.protocol_version != "maniskill-v1":
            raise ValueError("The NEWT example requires the ManiSkill v1 state protocol")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_num_threads(1)
        self.models = load_model_bundle(context.checkpoint_path, self.device)
        self.episode_tasks.clear()
        self.held_actions.clear()
        return PolicyCapabilities(
            supported_benchmarks=("maniskill",),
            supported_suites={"maniskill": (context.benchmark.suite,)},
            required_features=("state.proprioception",),
            max_batch_size=256,
            supported_action_chunk_sizes=(1,),
            action_representations={"maniskill": "maniskill.native-controller-v1"},
        )

    def reset(self, context: EpisodeContext) -> None:
        if context.task_id not in self.models:
            raise ValueError(f"No pinned NEWT checkpoint for {context.task_id}")
        model = self.models[context.task_id]
        mode = context.metadata.get("obs_mode", context.metadata.get("observation_mode"))
        if mode is not None and mode not in {"state", "state_dict"}:
            raise ValueError("NEWT requires native numeric state observations")
        for key in ("robot_uid", "control_mode", "action_dimension"):
            expected = model.metadata[key]
            actual = context.metadata.get(key)
            if actual is not None and actual != expected:
                raise ValueError(f"{context.task_id} requires {key}={expected!r}")
        self.episode_tasks[context.episode_id] = context.task_id
        self.held_actions.pop(context.episode_id, None)

    def act(self, observation: Observation) -> ActionChunk:
        return self.act_batch([observation])[0]

    def act_batch(self, observations: list[Observation]) -> list[ActionChunk]:
        if not self.models or not 1 <= len(observations) <= 256:
            raise ValueError("Load the policy before requesting a batch of 1 to 256 observations")
        groups: dict[str, list[int]] = {}
        result: list[ActionChunk | None] = [None] * len(observations)
        for index, observation in enumerate(observations):
            episode = observation.episode_id
            task = self.episode_tasks[episode]
            if episode in self.held_actions:
                result[index] = ActionChunk(self.held_actions.pop(episode)[None].copy())
            else:
                groups.setdefault(task, []).append(index)
        for task, indices in groups.items():
            model = self.models[task]
            padded = np.zeros((len(indices), 128), dtype=np.float32)
            for row, index in enumerate(indices):
                state = np.asarray(observations[index].features["state.proprioception"], dtype=np.float32)
                if state.shape != (model.observation_dim,) or not np.isfinite(state).all():
                    raise ValueError(f"{task} requires finite native state shape ({model.observation_dim},)")
                padded[row, :model.observation_dim] = state
            with torch.inference_mode():
                actions = model.model(torch.as_tensor(padded, device=self.device)).cpu().numpy()
            if not np.isfinite(actions).all():
                raise ValueError(f"{task} produced non-finite actions")
            for index, padded_action in zip(indices, actions, strict=True):
                action = np.ascontiguousarray(padded_action[:model.action_dim], dtype=np.float32)
                episode = observations[index].episode_id
                self.held_actions[episode] = action.copy()
                result[index] = ActionChunk(action[None])
        # Each actor action is held for two native ticks, matching NEWT's wrapper.
        # The worker still observes, evaluates rewards/termination, and steps physics each tick.
        return [chunk for chunk in result if chunk is not None]

    def end_episode(self, episode_id: str) -> None:
        self.episode_tasks.pop(episode_id, None)
        self.held_actions.pop(episode_id, None)

    def unload(self) -> None:
        self.models.clear()
        self.episode_tasks.clear()
        self.held_actions.clear()


if __name__ == "__main__":
    run(ManiSkillNewtPolicy())
