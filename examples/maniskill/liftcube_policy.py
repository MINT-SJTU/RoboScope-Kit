"""Serve an explicit learned LiftCube actor prior on the existing LiftPeg4 profile."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

import numpy as np
import torch
from liftcube_model import load_actor
from liftcube_projection import SourceActionRepeat, project_actor_input

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
    repeat: SourceActionRepeat = field(default_factory=SourceActionRepeat)
    steps: int = 0
    saturated_xyz_components: int = 0


class ManiSkillLiftCubePolicy(Policy):
    def __init__(self) -> None:
        self.actor = None
        self.device = "cuda"
        self.episodes: dict[str, EpisodeState] = {}

    def load(self, context: SessionContext) -> PolicyCapabilities:
        if context.benchmark.name != "maniskill" or context.benchmark.protocol_version != "maniskill-v1":
            raise ValueError("LiftCube prior requires ManiSkill v1")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.actor = load_actor(context.checkpoint_path, self.device)
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
        expected = {"profile_id": "lift_peg_upright_state", "robot_uid": "panda",
                    "control_mode": "pd_ee_delta_pos", "action_dimension": 4,
                    "sim_freq": 100, "control_freq": 20}
        if context.task_id != "LiftPegUpright-v1" or any(context.metadata.get(key) != value
                                                        for key, value in expected.items()):
            raise ValueError("LiftCube prior requires the unchanged canonical LiftPeg4 profile")
        if context.metadata.get("obs_mode") not in {"state", "state_dict"}:
            raise ValueError("LiftCube prior requires the complete native numeric state")
        self.episodes[context.episode_id] = EpisodeState()

    def act(self, observation: Observation) -> ActionChunk:
        episode = self.episodes[observation.episode_id]
        state = np.asarray(observation.features["state.proprioception"], dtype=np.float32)
        features = project_actor_input(state)
        source_action = None
        if episode.repeat.needs_source_action:
            with torch.inference_mode():
                tensor = torch.as_tensor(features[None], device=self.device)
                source_action = self.actor(tensor)[0].cpu().numpy()
        # Hold the complete source4 for two ticks; remap using each current TCP quaternion.
        action = episode.repeat.next_action(state, source_action)
        episode.steps += 1
        episode.saturated_xyz_components += int(np.count_nonzero(np.abs(action[:3]) > 1))
        return ActionChunk(action[None])

    def act_batch(self, observations: list[Observation]) -> list[ActionChunk]:
        if not 1 <= len(observations) <= 256:
            raise ValueError("Expected a batch of 1 to 256 observations")
        return [self.act(observation) for observation in observations]

    def end_episode(self, episode_id: str) -> None:
        episode = self.episodes.pop(episode_id, None)
        if episode is not None:
            print(json.dumps({"episode_id": episode_id, "transfer": "liftcube-actor-prior",
                              "steps": episode.steps,
                              "native_saturated_xyz_components": episode.saturated_xyz_components}),
                  file=sys.stderr, flush=True)

    def unload(self) -> None:
        self.episodes.clear()
        self.actor = None


if __name__ == "__main__":
    run(ManiSkillLiftCubePolicy())
