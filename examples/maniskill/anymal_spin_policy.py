"""Zero-shot counterclockwise orbit transfer of the official ANYmal Reach PPO."""

from __future__ import annotations

import numpy as np
import torch
from ppo_model import PPOAgent, load_agent

from roboscope_kit import (
    ActionChunk,
    EpisodeContext,
    Observation,
    Policy,
    PolicyCapabilities,
    SessionContext,
    run,
)

CHECKPOINT_SHA256 = "e679a1ec3aa9cf2eeee8f6b2c233a49fa340480ab614af6b06b6b56393857ba9"
ORBIT_RADIUS_M = 0.75
REACHED_GOAL_RADIUS_M = 0.35


class ManiSkillAnymalSpinPolicy(Policy):
    def __init__(self) -> None:
        self.agent: PPOAgent | None = None
        self.centers: dict[str, np.ndarray | None] = {}
        self.device = "cuda"

    def load(self, context: SessionContext) -> PolicyCapabilities:
        if context.benchmark.name != "maniskill" or context.benchmark.protocol_version != "maniskill-v1":
            raise ValueError("The ANYmal orbit example requires the ManiSkill v1 protocol")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.agent, observation_dim, action_dim = load_agent(
            context.checkpoint_path, self.device, expected_sha256=CHECKPOINT_SHA256,
        )
        if observation_dim != 35 or action_dim != 12:
            raise ValueError("Expected the official ANYmal Reach 35-state/12-action checkpoint")
        self.centers.clear()
        return PolicyCapabilities(
            supported_benchmarks=("maniskill",),
            supported_suites={"maniskill": (context.benchmark.suite,)},
            required_features=("state.proprioception", "state.root_pose"),
            max_batch_size=256,
            supported_action_chunk_sizes=(1,),
            action_representations={"maniskill": "maniskill.native-controller-v1"},
        )

    def reset(self, context: EpisodeContext) -> None:
        if context.task_id != "AnymalC-Spin-v1":
            raise ValueError("The ANYmal orbit example serves AnymalC-Spin-v1")
        for key, expected in (("robot_uid", "anymal_c"), ("control_mode", "pd_joint_delta_pos")):
            actual = context.metadata.get(key)
            if actual is not None and actual != expected:
                raise ValueError(f"ANYmal orbit requires {key}={expected!r}")
        dimension = context.metadata.get("action_dimension")
        if dimension is not None and dimension != 12:
            raise ValueError("ANYmal orbit requires the native 12-joint controller")
        self.centers[context.episode_id] = None

    def reach_observation(self, observation: Observation) -> np.ndarray:
        state = np.asarray(observation.features["state.proprioception"], dtype=np.float32)
        pose = np.asarray(observation.features["state.root_pose"], dtype=np.float32)
        if state.shape != (30,) or not np.isfinite(state).all():
            raise ValueError("Expected the complete finite native Spin state of 30 values")
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError("Expected native root pose xyz+wxyz")
        quaternion = pose[3:]
        norm = np.linalg.norm(quaternion)
        if norm < 1e-8 or abs(float(norm) - 1) > 1e-3:
            raise ValueError("Native root quaternion must have unit norm")
        position = pose[:2]
        center = self.centers[observation.episode_id]
        if center is None:
            w, x, y, z = quaternion / norm
            yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            center = position + ORBIT_RADIUS_M * np.array([-np.sin(yaw), np.cos(yaw)], dtype=np.float32)
            self.centers[observation.episode_id] = center
        relative_position = position - center
        angle = np.arctan2(relative_position[1], relative_position[0]) + np.pi / 2
        waypoint = center + ORBIT_RADIUS_M * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        goal_delta = waypoint - position
        reached = float(np.linalg.norm(goal_delta) < REACHED_GOAL_RADIUS_M)
        # Reach and Spin share exactly this 30-value native physical prefix.
        # The final 5 values represent the policy's real internal waypoint goal.
        return np.concatenate((state, np.array([reached], dtype=np.float32), waypoint, goal_delta))

    def act(self, observation: Observation) -> ActionChunk:
        return self.act_batch([observation])[0]

    def act_batch(self, observations: list[Observation]) -> list[ActionChunk]:
        if self.agent is None:
            raise RuntimeError("Load the official ANYmal checkpoint before inference")
        if not 1 <= len(observations) <= 256:
            raise ValueError("Expected a batch of 1 to 256 observations")
        values = np.stack([self.reach_observation(observation) for observation in observations])
        with torch.inference_mode():
            actions = self.agent.actor_mean(torch.as_tensor(values, device=self.device)).cpu().numpy()
        if not np.isfinite(actions).all():
            raise ValueError("ANYmal Reach actor produced non-finite actions")
        return [ActionChunk(np.ascontiguousarray(action[None], dtype=np.float32)) for action in actions]

    def end_episode(self, episode_id: str) -> None:
        self.centers.pop(episode_id, None)

    def unload(self) -> None:
        self.centers.clear()
        self.agent = None


if __name__ == "__main__":
    run(ManiSkillAnymalSpinPolicy())
