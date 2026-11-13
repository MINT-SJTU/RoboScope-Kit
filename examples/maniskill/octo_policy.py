"""Serve the pinned public Octo-small model on ManiSkill Bridge RGB tasks."""

from __future__ import annotations

import numpy as np
from octo_model import CONTROL_MODE, INSTRUCTIONS, BridgeOcto

from roboscope_kit import (
    ActionChunk,
    EpisodeContext,
    Observation,
    Policy,
    PolicyCapabilities,
    SessionContext,
    run,
)


class ManiSkillOctoPolicy(Policy):
    def __init__(self) -> None:
        self.model: BridgeOcto | None = None
        self.tasks: dict[str, str] = {}

    def load(self, context: SessionContext) -> PolicyCapabilities:
        if context.benchmark.name != "maniskill" or context.benchmark.protocol_version != "maniskill-v1":
            raise ValueError("Octo requires the ManiSkill v1 Bridge profiles")
        self.model = BridgeOcto(context.checkpoint_path)
        self.tasks.clear()
        return PolicyCapabilities(
            supported_benchmarks=("maniskill",),
            supported_suites={"maniskill": (context.benchmark.suite,)},
            required_features=("images.primary", "language.instruction"),
            max_batch_size=16,
            supported_action_chunk_sizes=(1,),
            action_representations={"maniskill": "maniskill.native-controller-v1"},
        )

    def reset(self, context: EpisodeContext) -> None:
        if self.model is None:
            raise RuntimeError("Octo has not loaded its checkpoint")
        if context.task_id not in INSTRUCTIONS:
            raise ValueError(f"Octo Bridge checkpoint does not declare {context.task_id}")
        if context.metadata.get("control_mode") != CONTROL_MODE:
            raise ValueError("Octo Bridge requires the native seven-dimensional WidowX controller")
        if context.metadata.get("obs_mode") != "rgb+segmentation":
            raise ValueError("Octo Bridge requires the native visual observation mode")
        self.model.reset(context.episode_id, context.task_id)
        self.tasks[context.episode_id] = context.task_id

    def act(self, observation: Observation) -> ActionChunk:
        return self.act_batch([observation])[0]

    def act_batch(self, observations: list[Observation]) -> list[ActionChunk]:
        if self.model is None:
            raise RuntimeError("Octo has not loaded its checkpoint")
        for observation in observations:
            instruction = INSTRUCTIONS[self.tasks[observation.episode_id]]
            if observation.features["language.instruction"] != instruction:
                raise ValueError("Bridge task instruction differs from the pinned upstream prompt")
        actions = self.model.act_batch(
            [observation.episode_id for observation in observations],
            [np.asarray(observation.features["images.primary"]) for observation in observations],
        )
        return [ActionChunk(action[None]) for action in actions]

    def end_episode(self, episode_id: str) -> None:
        self.tasks.pop(episode_id, None)
        if self.model is not None:
            self.model.end_episode(episode_id)

    def unload(self) -> None:
        self.tasks.clear()
        self.model = None


if __name__ == "__main__":
    run(ManiSkillOctoPolicy())
