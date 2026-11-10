"""SDK Policy for task-indexed ManiSkill PPO state checkpoint bundles."""

from __future__ import annotations

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

ACTION_REPRESENTATION = "maniskill.native-controller-v1"


class ManiSkillPPOPolicy(Policy):
    def __init__(self) -> None:
        self.models: dict[str, LoadedPPO] = {}
        self.episode_tasks: dict[str, str] = {}
        self.device = "cuda"

    def load(self, context: SessionContext) -> PolicyCapabilities:
        if context.benchmark.name != "maniskill":
            raise ValueError("The PPO example serves ManiSkill state suites")
        if context.benchmark.protocol_version != "maniskill-v1":
            raise ValueError("Unsupported ManiSkill protocol version")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_num_threads(1)
        self.models = load_model_bundle(context.checkpoint_path, self.device)
        self.episode_tasks.clear()
        return PolicyCapabilities(
            supported_benchmarks=("maniskill",),
            supported_suites={"maniskill": (context.benchmark.suite,)},
            required_features=("state.proprioception",),
            max_batch_size=256,
            supported_action_chunk_sizes=(1,),
            action_representations={"maniskill": ACTION_REPRESENTATION},
        )

    def reset(self, context: EpisodeContext) -> None:
        if context.task_id not in self.models:
            raise ValueError(f"No checkpoint configured for task {context.task_id}")
        model = self.models[context.task_id]
        observation_mode = context.metadata.get("obs_mode", context.metadata.get("observation_mode"))
        if observation_mode is not None and observation_mode not in {"state", "state_dict"}:
            raise ValueError("The PPO example requires a numeric state profile")
        for key in ("robot_uid", "control_mode"):
            expected = model.metadata.get(key)
            actual = context.metadata.get(key)
            if expected is not None and actual is not None and expected != actual:
                raise ValueError(f"{context.task_id} checkpoint requires {key}={expected!r}")
        actual_dimension = context.metadata.get("action_dimension")
        if actual_dimension is not None and actual_dimension != model.action_dim:
            raise ValueError("Configured controller action dimension differs from the checkpoint")
        self.episode_tasks[context.episode_id] = context.task_id

    def act(self, observation: Observation) -> ActionChunk:
        return self.act_batch([observation])[0]

    def act_batch(self, observations: list[Observation]) -> list[ActionChunk]:
        if not self.models:
            raise RuntimeError("Policy has not loaded the real checkpoint")
        if not 1 <= len(observations) <= 256:
            raise ValueError("Expected between 1 and 256 observations")
        groups: dict[str, list[int]] = {}
        for index, observation in enumerate(observations):
            task_id = self.episode_tasks[observation.episode_id]
            groups.setdefault(task_id, []).append(index)
        result: list[ActionChunk | None] = [None] * len(observations)
        for task_id, indices in groups.items():
            model = self.models[task_id]
            values = []
            for index in indices:
                state = np.asarray(observations[index].features["state.proprioception"], dtype=np.float32)
                native_dim = model.metadata.get("native_observation_dimension", model.observation_dim)
                if state.shape != (native_dim,) or not np.isfinite(state).all():
                    raise ValueError(f"{task_id} requires finite native state shape ({native_dim},)")
                prefix = model.metadata.get("state_prefix_length")
                if prefix is not None:
                    state = state[:prefix]
                if state.shape != (model.observation_dim,):
                    raise ValueError(f"{task_id} observation projection differs from the checkpoint")
                values.append(state)
            batch = torch.as_tensor(np.stack(values), device=self.device, dtype=torch.float32)
            with torch.inference_mode():
                actions = model.agent.actor_mean(batch).cpu().numpy()
            if not np.isfinite(actions).all():
                raise ValueError(f"{task_id} model produced non-finite actions")
            bounds = model.metadata.get("action_clip_bounds")
            if bounds is not None:
                actions = np.clip(actions, bounds[0], bounds[1])
            for index, action in zip(indices, actions, strict=True):
                result[index] = ActionChunk(np.ascontiguousarray(action[None], dtype=np.float32))
        # Preserve each model's declared actor semantics; official actors stay unclipped.
        # The controller also enforces its native action bounds.
        return [chunk for chunk in result if chunk is not None]

    def end_episode(self, episode_id: str) -> None:
        self.episode_tasks.pop(episode_id, None)

    def unload(self) -> None:
        self.models.clear()
        self.episode_tasks.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    run(ManiSkillPPOPolicy())
