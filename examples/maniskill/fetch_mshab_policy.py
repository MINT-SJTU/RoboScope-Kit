"""Transfer official MS-HAB Fetch Open PPO policies to native cabinet tasks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
from fetch_mshab_model import LoadedFetch, load_models

from roboscope_kit import (
    ActionChunk,
    EpisodeContext,
    Observation,
    Policy,
    PolicyCapabilities,
    SessionContext,
    run,
)

BASE_JOINTS = ("root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint")
ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint",
              "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint")
SOURCE_JOINTS = (*BASE_JOINTS, "torso_lift_joint", "head_pan_joint", "shoulder_pan_joint",
                 "head_tilt_joint", *ARM_JOINTS[1:], "r_gripper_finger_joint", "l_gripper_finger_joint")
CONTROLLER_JOINTS = (*ARM_JOINTS, "l_gripper_finger_joint", "r_gripper_finger_joint",
                     "head_pan_joint", "head_tilt_joint", "torso_lift_joint", *BASE_JOINTS)
RAW_FEATURES = ("state.proprioception", "state.robot_base_pose", "state.robot_tcp_pose",
                "state.target_handle_position", "state.left_finger_pose", "state.right_finger_pose",
                "state.left_target_contact_force", "state.right_target_contact_force",
                "state.depth_head", "state.depth_hand")


@dataclass
class EpisodeState:
    task_id: str
    state_indices: np.ndarray
    frames: dict[str, deque] = field(default_factory=lambda: {
        key: deque(maxlen=3) for key in ("fetch_head_depth", "fetch_hand_depth")})


def raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw), -1)


def rotate(quaternion: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    value = torch.cat((point.new_zeros(point.shape[:-1]+(1,)), point), -1)
    inverse = quaternion * quaternion.new_tensor([1, -1, -1, -1])
    return raw_multiply(raw_multiply(quaternion, value), inverse)[..., 1:]


def rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternion, -1)
    scale = 2.0 / (quaternion*quaternion).sum(-1)
    return torch.stack((1-scale*(j*j+k*k), scale*(i*j-k*r), scale*(i*k+j*r),
                        scale*(i*j+k*r), 1-scale*(i*i+k*k), scale*(j*k-i*r),
                        scale*(i*k-j*r), scale*(j*k+i*r), 1-scale*(i*i+j*j)), -1).reshape(3, 3)


def raw_tensor(features: dict, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    value = np.asarray(features[name], dtype=np.float32)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"Expected finite measured {name} with shape {shape}")
    if name.endswith("_pose") and abs(float(np.linalg.norm(value[3:]))-1) > 1e-3:
        raise ValueError(f"Expected a measured unit quaternion in {name}")
    return torch.from_numpy(value.copy())


def grasped(features: dict) -> torch.Tensor:
    """Preserve native finger-contact magnitude and angle checks on the handle link."""
    flags = []
    for side, sign in (("left", -1), ("right", 1)):
        pose = raw_tensor(features, f"state.{side}_finger_pose", (7,))
        force = raw_tensor(features, f"state.{side}_target_contact_force", (3,))
        direction = sign * rotation_matrix(pose[3:])[:, 1]
        magnitude = torch.linalg.norm(force)
        direction_norm = torch.linalg.norm(direction)
        if direction_norm < 1e-6:
            direction_norm = torch.ones_like(direction_norm)
        force_norm = magnitude if magnitude >= 1e-6 else torch.ones_like(magnitude)
        normalized_direction = direction * (1/direction_norm)
        normalized_force = force * (1/force_norm)
        cosine = torch.clip(torch.einsum("i,i->", normalized_direction, normalized_force), -1, 1)
        angle = torch.rad2deg(torch.arccos(cosine))
        flags.append((magnitude >= 0.5) & (angle <= 30))
    return (flags[0] & flags[1]).reshape(1).to(torch.float32)


def source_state(features: dict, indices: np.ndarray) -> torch.Tensor:
    native = np.asarray(features["state.proprioception"], dtype=np.float32)
    if native.ndim != 1 or len(native) < 30 or not np.isfinite(native).all():
        raise ValueError("Expected native Fetch qpos15 and qvel15 at the state prefix")
    base = raw_tensor(features, "state.robot_base_pose", (7,))
    tcp = raw_tensor(features, "state.robot_tcp_pose", (7,))
    handle = raw_tensor(features, "state.target_handle_position", (3,))
    inverse = base[3:] * base.new_tensor([1, -1, -1, -1])
    inverse_position = rotate(inverse, -base[:3])
    relative_position = inverse_position + rotate(inverse, tcp[:3])
    relative_quaternion = raw_multiply(inverse, tcp[3:])
    relative_quaternion = torch.where(relative_quaternion[0:1] < 0,
                                      -relative_quaternion, relative_quaternion)
    relative_handle = inverse_position + rotate(inverse, handle)
    # Open/Close have no loose subtask object; their upstream obj_pose is exactly zeros.
    return torch.cat((torch.from_numpy(native[:15][indices][3:].copy()),
                      torch.from_numpy(native[15:30][indices][3:].copy()),
                      relative_position, relative_quaternion, torch.zeros(7), relative_handle,
                      grasped(features)))


def depth_frame(features: dict, name: str) -> torch.Tensor:
    value = raw_tensor(features, name, (128, 128, 1))
    return value.permute(2, 0, 1).contiguous()


class ManiSkillFetchMshabPolicy(Policy):
    def __init__(self) -> None:
        self.models: dict[str, LoadedFetch] = {}
        self.episodes: dict[str, EpisodeState] = {}
        self.device = "cuda"

    def load(self, context: SessionContext) -> PolicyCapabilities:
        if context.benchmark.name != "maniskill" or context.benchmark.protocol_version != "maniskill-v1":
            raise ValueError("MS-HAB Fetch transfer requires the ManiSkill v1 protocol")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.models = load_models(context.checkpoint_path, self.device)
        self.episodes.clear()
        return PolicyCapabilities(
            supported_benchmarks=("maniskill",),
            supported_suites={"maniskill": (context.benchmark.suite,)},
            required_features=RAW_FEATURES,
            max_batch_size=256,
            supported_action_chunk_sizes=(1,),
            action_representations={"maniskill": "maniskill.native-controller-v1"},
        )

    def reset(self, context: EpisodeContext) -> None:
        if context.task_id not in self.models:
            raise ValueError("Load the task's pinned real Fetch model before reset")
        for key, expected in (("robot_uid", "fetch"), ("control_mode", "pd_joint_delta_pos"),
                              ("action_dimension", 13), ("sim_freq", 100), ("control_freq", 20)):
            if context.metadata.get(key) != expected:
                raise ValueError(f"MS-HAB Fetch transfer requires {key}={expected!r}")
        names = tuple(context.metadata["active_joint_names"])
        if len(names) != 15 or set(names) != set(SOURCE_JOINTS):
            raise ValueError("Expected the original fifteen Fetch active joints")
        if tuple(context.metadata["action_joint_names"]) != CONTROLLER_JOINTS:
            raise ValueError("Expected the original native Fetch controller component order")
        self.episodes[context.episode_id] = EpisodeState(
            context.task_id, np.asarray([names.index(name) for name in SOURCE_JOINTS]))

    def act(self, observation: Observation) -> ActionChunk:
        return self.act_batch([observation])[0]

    def act_batch(self, observations: list[Observation]) -> list[ActionChunk]:
        if not self.models or not 1 <= len(observations) <= 256:
            raise ValueError("Load a real Fetch model before a batch of one to 256 observations")
        if len({obs.episode_id for obs in observations}) != len(observations):
            raise ValueError("A Fetch batch must contain distinct episodes")
        grouped: dict[str, list[tuple[int, torch.Tensor, dict[str, torch.Tensor]]]] = {}
        for index, observation in enumerate(observations):
            episode = self.episodes[observation.episode_id]
            pixels = {}
            for key, field_name in (("fetch_head_depth", "state.depth_head"),
                                    ("fetch_hand_depth", "state.depth_hand")):
                frame = depth_frame(observation.features, field_name)
                history = episode.frames[key]
                if not history:
                    history.extend([frame]*3)
                else:
                    history.append(frame)
                pixels[key] = torch.stack(tuple(history), dim=0)
            state = source_state(observation.features, episode.state_indices)
            grouped.setdefault(episode.task_id, []).append((index, state, pixels))
        results: list[ActionChunk | None] = [None] * len(observations)
        with torch.inference_mode():
            for task, records in grouped.items():
                batch = {"state": torch.stack([item[1] for item in records]).to(self.device),
                         "pixels": {key: torch.stack([item[2][key] for item in records]).to(self.device)
                                    for key in ("fetch_head_depth", "fetch_hand_depth")}}
                actions = self.models[task].actor(batch).cpu().numpy()
                # Preserve FetchActionWrapper: source head stationary, base/torso movable.
                actions[:, 8:10] = 0
                if actions.shape != (len(records), 13) or not np.isfinite(actions).all():
                    raise ValueError("The real Fetch model returned invalid controller actions")
                for (index, _, _), action in zip(records, actions, strict=True):
                    # Native controllers perform their own clipping/scaling; preserve actor means.
                    results[index] = ActionChunk(np.ascontiguousarray(action[None], dtype=np.float32))
        return results

    def end_episode(self, episode_id: str) -> None:
        self.episodes.pop(episode_id, None)

    def unload(self) -> None:
        self.episodes.clear()
        self.models.clear()


if __name__ == "__main__":
    run(ManiSkillFetchMshabPolicy())
