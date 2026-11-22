"""Author-task ManiSkill Octo candidate with its complete native EE-pose action."""

from __future__ import annotations

import numpy as np
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

AUTHOR_TASKS = {
    "PegInsertionSide-v1", "PickCube-v1", "StackCube-v1", "PlugCharger-v1", "PushCube-v1",
}


class ManiSkillPandaOctoPolicy(Policy):
    def __init__(self) -> None:
        self.model: ManiSkillFinetunedOcto | None = None
        self.episodes: dict[str, str | None] = {}

    def load(self, context: SessionContext) -> PolicyCapabilities:
        if context.benchmark.name != "maniskill" or context.benchmark.protocol_version != "maniskill-v1":
            raise ValueError("Panda Octo requires the ManiSkill v1 protocol")
        self.model = ManiSkillFinetunedOcto(context.checkpoint_path)
        self.episodes.clear()
        return PolicyCapabilities(
            supported_benchmarks=("maniskill",),
            supported_suites={"maniskill": (context.benchmark.suite,)},
            required_features=("images.primary", "language.instruction"),
            max_batch_size=16,
            supported_action_chunk_sizes=(1,),
            action_representations={"maniskill": "maniskill.native-controller-v1"},
        )

    def reset(self, context: EpisodeContext) -> None:
        self.end_episode(context.episode_id)
        if context.task_id not in AUTHOR_TASKS:
            raise ValueError("This Octo candidate is restricted to the author's five ManiSkill tasks")
        robot = context.metadata["robot_uid"]
        if not isinstance(robot, str) or robot not in {"panda", "panda_wristcam"}:
            raise ValueError("This Octo candidate requires one native Panda")
        if (context.metadata["control_mode"] != "pd_ee_delta_pose"
                or context.metadata["action_dimension"] != 7):
            raise ValueError("This Octo candidate requires the author's complete 7D EE-pose controller")
        self.episodes[context.episode_id] = None

    def act(self, observation: Observation) -> ActionChunk:
        return self.act_batch([observation])[0]

    def act_batch(self, observations: list[Observation]) -> list[ActionChunk]:
        if self.model is None:
            raise RuntimeError("Load the real ManiSkill Octo checkpoint before inference")
        ids, images = [], []
        for observation in observations:
            previous_instruction = self.episodes[observation.episode_id]
            instruction = observation.features["language.instruction"]
            if previous_instruction is None:
                self.episodes[observation.episode_id] = instruction
                self.model.reset(observation.episode_id, instruction)
            elif previous_instruction != instruction:
                raise ValueError("The instruction changed during an episode")
            ids.append(observation.episode_id)
            images.append(np.asarray(observation.features["images.primary"]))
        return [ActionChunk(np.ascontiguousarray(action[None], dtype=np.float32))
                for action in self.model.act_batch(ids, images)]

    def end_episode(self, episode_id: str) -> None:
        if episode_id in self.episodes:
            self.episodes.pop(episode_id)
            if self.model is not None:
                self.model.end_episode(episode_id)

    def unload(self) -> None:
        self.episodes.clear()
        self.model = None


if __name__ == "__main__":
    run(ManiSkillPandaOctoPolicy())
