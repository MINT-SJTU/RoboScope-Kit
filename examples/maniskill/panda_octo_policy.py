"""Visual ManiSkill Octo baseline and explicit projections onto frozen Panda controllers."""

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


class ManiSkillPandaOctoPolicy(Policy):
    def __init__(self) -> None:
        self.model: ManiSkillFinetunedOcto | None = None
        self.episodes: dict[str, dict] = {}

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
        robots = context.metadata["robot_uid"]
        robots = [robots] if isinstance(robots, str) else robots
        modes = context.metadata["control_mode"]
        modes = [modes] * len(robots) if isinstance(modes, str) else modes
        if len(robots) != len(modes) or not 1 <= len(robots) <= 2:
            raise ValueError("Expected one or two frozen Panda controllers")
        for robot, mode in zip(robots, modes, strict=True):
            if robot not in {"panda", "panda_wristcam", "panda_stick"}:
                raise ValueError(f"Unsupported Panda embodiment {robot}")
            if mode not in {"pd_ee_delta_pose", "pd_ee_delta_pos"}:
                raise ValueError(f"Unsupported Panda Octo controller {mode}")
            if robot == "panda_stick" and mode != "pd_ee_delta_pos":
                raise ValueError("The fixed PandaStick profile has translation-only control")
        self.episodes[context.episode_id] = {
            "robots": robots, "modes": modes, "instruction": None,
            "ids": [f"{context.episode_id}/arm{index}" for index in range(len(robots))],
        }

    def act(self, observation: Observation) -> ActionChunk:
        return self.act_batch([observation])[0]

    def act_batch(self, observations: list[Observation]) -> list[ActionChunk]:
        if self.model is None:
            raise RuntimeError("Load the real ManiSkill Octo checkpoint before inference")
        ids, images = [], []
        for observation in observations:
            state = self.episodes[observation.episode_id]
            instruction = observation.features["language.instruction"]
            if state["instruction"] is None:
                state["instruction"] = instruction
                for index, episode_id in enumerate(state["ids"]):
                    prompt = instruction if len(state["ids"]) == 1 else f"Robot {index + 1}: {instruction}"
                    self.model.reset(episode_id, prompt)
            elif state["instruction"] != instruction:
                raise ValueError("The instruction changed during an episode")
            for episode_id in state["ids"]:
                ids.append(episode_id)
                images.append(np.asarray(observation.features["images.primary"]))
        raw_actions = iter(self.model.act_batch(ids, images))
        result = []
        for observation in observations:
            state = self.episodes[observation.episode_id]
            actions = []
            for robot, mode in zip(state["robots"], state["modes"], strict=True):
                raw = next(raw_actions)
                if robot == "panda_stick":
                    actions.append(raw[:3])
                elif mode == "pd_ee_delta_pos":
                    actions.append(raw[[0, 1, 2, 6]])
                else:
                    actions.append(raw)
            result.append(ActionChunk(np.ascontiguousarray(np.concatenate(actions)[None], dtype=np.float32)))
        return result

    def end_episode(self, episode_id: str) -> None:
        state = self.episodes.pop(episode_id, None)
        if state is not None and self.model is not None:
            for stream in state["ids"]:
                self.model.end_episode(stream)

    def unload(self) -> None:
        self.episodes.clear()
        self.model = None


if __name__ == "__main__":
    run(ManiSkillPandaOctoPolicy())
