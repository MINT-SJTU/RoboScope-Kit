"""Transfer published DM-Control Humanoid priors to native ManiSkill PD control."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import numpy as np
import torch
from humanoid_model import LoadedHumanoid, catalog, load_models

from roboscope_kit import (
    ActionChunk,
    EpisodeContext,
    Observation,
    Policy,
    PolicyCapabilities,
    SessionContext,
    run,
)

CANONICAL_PROFILES = {
    "MS-HumanoidWalk-v1": "ms_humanoid_walk_state",
    "MS-HumanoidRun-v1": "ms_humanoid_run_state",
}


@dataclass
class EpisodeState:
    task_id: str
    source_state_indices: np.ndarray
    controller_state_indices: np.ndarray
    requires_root_kinematics: bool = True
    held_source_action: np.ndarray | None = None
    steps: int = 0
    clipped_joints: int = 0


def rotation_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or norm < 1e-8:
        raise ValueError("Expected a finite measured root quaternion")
    w, x, y, z = quaternion / norm
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def source_observation(features: dict, episode: EpisodeState, dimension: int) -> np.ndarray:
    value = np.asarray(features["state.proprioception"], dtype=np.float32)
    if value.shape != (dimension,) or not np.isfinite(value).all():
        raise ValueError(f"Expected the complete finite native Humanoid state of {dimension} values")
    if episode.requires_root_kinematics:
        pose = np.asarray(features["state.root_pose"], dtype=np.float32)
        velocity = np.asarray(features["state.root_linear_velocity"], dtype=np.float32)
        angular_velocity = np.asarray(features["state.root_angular_velocity"], dtype=np.float32)
        if pose.shape != (7,) or any(x.shape != (3,) for x in (velocity, angular_velocity)):
            raise ValueError("Expected measured root pose and world linear/angular velocities")
        if not all(np.isfinite(x).all() for x in (pose, velocity, angular_velocity)):
            raise ValueError("Root kinematics must be finite")
        quaternion = pose[3:]
    else:
        if episode.task_id not in CANONICAL_PROFILES or dimension != 185:
            raise ValueError("Only canonical Humanoid Walk/Run supply complete root kinematics in state")
        # These are the same dummy_root_0 measurements exposed by the root-state profiles.
        quaternion = value[181:185]
        velocity, angular_velocity = value[42:45], value[45:48]
    rotation = rotation_matrix(quaternion)
    if episode.task_id == "MS-HumanoidStand-v1":
        # Stand stores 16 link linear + 16 angular + COM velocities in the root frame.
        height, extremities = value[42:43], value[142:154]
        com_velocity = rotation @ value[139:142]
    else:
        height, com_velocity, extremities = value[48:49], value[49:52], value[52:64]
    # MuJoCo's free-joint translation velocity is world-frame, angular velocity body-frame.
    generalized_velocity = np.concatenate((velocity, rotation.T @ angular_velocity,
                                            value[21:42][episode.source_state_indices]))
    return np.asarray(np.concatenate((value[:21][episode.source_state_indices], height,
        extremities, rotation[2], com_velocity, generalized_velocity)), dtype=np.float32)


def native_action(value: np.ndarray, source_action: np.ndarray, episode: EpisodeState,
                  contract: dict) -> np.ndarray:
    """Match source actuator+passive joint torque at the observed state, before clipping."""
    qpos = value[:21][episode.source_state_indices]
    qvel = value[21:42][episode.source_state_indices]
    stiffness = np.asarray(contract["stiffness"], dtype=np.float32)
    damping = np.asarray(contract["damping"], dtype=np.float32)
    spring_reference = np.asarray(contract["spring_reference"], dtype=np.float32)
    torque = -stiffness * (qpos-spring_reference) - damping*qvel
    state_names = contract["observation_joint_names"]
    actuator_indices = [state_names.index(name) for name in contract["action_joint_names"]]
    torque[actuator_indices] += np.asarray(contract["gears"], dtype=np.float32) * source_action
    controller_indices = [state_names.index(name) for name in contract["native_controller_joint_names"]]
    native_kp = np.asarray(contract["native_stiffness"], dtype=np.float32)
    native_kd = np.asarray(contract["native_damping"], dtype=np.float32)
    current_qvel = value[21:42][episode.controller_state_indices]
    delta = (torque[controller_indices] + native_kd*current_qvel) / native_kp
    normalized = delta / np.float32(contract["native_delta_range"])
    episode.clipped_joints += int(np.count_nonzero(np.abs(normalized) > 1.0))
    return np.ascontiguousarray(np.clip(normalized, -1.0, 1.0), dtype=np.float32)


class ManiSkillHumanoidPolicy(Policy):
    def __init__(self) -> None:
        self.models: dict[str, LoadedHumanoid] = {}
        self.episodes: dict[str, EpisodeState] = {}
        self.contract: dict = {}
        self.device = "cuda"

    def load(self, context: SessionContext) -> PolicyCapabilities:
        if context.benchmark.name != "maniskill" or context.benchmark.protocol_version != "maniskill-v1":
            raise ValueError("TD-MPC2 Humanoid transfer requires ManiSkill v1")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.models = load_models(context.checkpoint_path, self.device)
        self.contract = catalog()["physics_contract"]
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
        if context.task_id not in self.models:
            raise ValueError(f"No selected Humanoid checkpoint for {context.task_id}")
        for key, expected in (("robot_uid", "humanoid"), ("control_mode", "pd_joint_delta_pos"),
                              ("action_dimension", 21), ("sim_freq", 200), ("control_freq", 40)):
            if context.metadata.get(key) != expected:
                raise ValueError(f"Humanoid transfer requires {key}={expected!r}")
        if context.metadata.get("obs_mode") not in {"state", "state_dict"}:
            raise ValueError("The Humanoid transfer requires a numeric native state profile")
        profile_id = context.metadata.get("profile_id")
        if profile_id == "ms_humanoid_stand_state":
            raise ValueError(
                "Canonical Humanoid Stand omits the world root kinematics required by this model"
            )
        names = context.metadata.get("active_joint_names")
        source_names = self.contract["observation_joint_names"]
        if not isinstance(names, (list, tuple)) or len(names) != 21 or set(names) != set(source_names):
            raise ValueError("Actual native joint identities must match the pinned Humanoid")
        controller = self.contract["native_controller_joint_names"]
        if list(context.metadata.get("action_joint_names", ())) != controller:
            raise ValueError("Native Humanoid controller joint ordering differs from its source pin")
        self.episodes[context.episode_id] = EpisodeState(
            context.task_id, np.asarray([names.index(name) for name in source_names]),
            np.asarray([names.index(name) for name in controller]),
            requires_root_kinematics=not (
                context.task_id in CANONICAL_PROFILES
                and profile_id == CANONICAL_PROFILES[context.task_id]
            ),
        )

    def act(self, observation: Observation) -> ActionChunk:
        episode = self.episodes[observation.episode_id]
        loaded = self.models[episode.task_id]
        features = source_observation(observation.features, episode,
                                      loaded.metadata["observation_dimension"])
        if episode.held_source_action is None:
            with torch.inference_mode():
                output = loaded.actor(torch.as_tensor(features[None], device=self.device))[0].cpu().numpy()
            if output.shape != (21,) or not np.isfinite(output).all():
                raise ValueError("Humanoid actor must return 21 finite normalized motor commands")
            episode.held_source_action = np.ascontiguousarray(output, dtype=np.float32)
        output = episode.held_source_action
        # The source wrapper holds a motor command for two 25 ms control ticks.
        # Recompute the native PD target each tick to retain the requested physical torque.
        value = np.asarray(observation.features["state.proprioception"], dtype=np.float32)
        action = native_action(value, output, episode, self.contract)
        episode.steps += 1
        if episode.steps % loaded.metadata["action_repeat"] == 0:
            episode.held_source_action = None
        return ActionChunk(action[None])

    def act_batch(self, observations: list[Observation]) -> list[ActionChunk]:
        if not 1 <= len(observations) <= 256:
            raise ValueError("Expected a batch of 1 to 256 observations")
        return [self.act(observation) for observation in observations]

    def end_episode(self, episode_id: str) -> None:
        episode = self.episodes.pop(episode_id, None)
        if episode is not None:
            print(json.dumps({"episode_id": episode_id, "transfer": "dmcontrol-humanoid-prior",
                              "steps": episode.steps, "controller_clipped_joints": episode.clipped_joints}),
                  file=sys.stderr, flush=True)

    def unload(self) -> None:
        self.episodes.clear()
        self.models.clear()


if __name__ == "__main__":
    run(ManiSkillHumanoidPolicy())
