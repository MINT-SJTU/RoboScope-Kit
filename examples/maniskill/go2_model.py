"""Strict community Go2 locomotion actor and its explicit observation contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from unitree_model import body_vector


class Go2ActorCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.std = nn.Parameter(torch.ones(12))
        self.actor = nn.Sequential(
            nn.Linear(45, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 12),
        )
        self.critic = nn.Sequential(
            nn.Linear(45, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1),
        )


def load_model(source: str | Path, device: str = "cuda") -> tuple[Go2ActorCritic, dict[str, Any]]:
    metadata = json.loads(Path(__file__).with_name("go2-checkpoint.json").read_text("utf-8"))
    checkpoint = Path(source).expanduser().resolve(strict=True)
    if checkpoint.is_dir():
        checkpoint = checkpoint / metadata["path"]
    if checkpoint.stat().st_size != metadata["bytes"]:
        raise ValueError("Go2 checkpoint size differs from its pin")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("Go2 checkpoint SHA-256 differs from its pin")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"]
    model = Go2ActorCritic()
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False), metadata


def policy_observation(
    state: np.ndarray,
    root_pose: np.ndarray,
    world_angular_velocity: np.ndarray,
    joint_indices: np.ndarray,
    previous_action: np.ndarray,
    metadata: dict[str, Any],
) -> np.ndarray:
    if state.shape != (35,) or not np.isfinite(state).all():
        raise ValueError("Go2 requires the complete finite native Reach state of 35 values")
    if root_pose.shape != (7,) or world_angular_velocity.shape != (3,):
        raise ValueError("Go2 requires native root pose xyz+wxyz and world angular velocity")
    if not np.isfinite(root_pose).all() or not np.isfinite(world_angular_velocity).all():
        raise ValueError("Root kinematics must be finite")
    quaternion = root_pose[3:]
    norm = np.linalg.norm(quaternion)
    if norm < 1e-8 or abs(float(norm) - 1) > 1e-3:
        raise ValueError("Native root quaternion must have unit norm")
    quaternion = quaternion / norm
    omega = body_vector(world_angular_velocity, quaternion)
    gravity = body_vector(np.array([0, 0, -1], dtype=np.float32), quaternion)
    # Native QuadrupedReach order after qpos/qvel: root linear/angular velocity,
    # reached_goal, goal_pos.xy, robot_to_goal.xy. No observation values are padded.
    goal_delta = state[33:35]
    body_goal = body_vector(np.array([*goal_delta, 0], dtype=np.float32), quaternion)
    commands = np.array([
        body_goal[0], body_goal[1], np.arctan2(body_goal[1], body_goal[0]),
    ], dtype=np.float32)
    bounds = np.asarray(metadata["command_ranges"], dtype=np.float32)
    commands = np.clip(commands, bounds[:, 0], bounds[:, 1])
    if state[30] > 0.5:
        commands[:] = 0
    scales = metadata["observation_scales"]
    relative_qpos = state[joint_indices] - np.asarray(metadata["policy_default_angles"], dtype=np.float32)
    return np.concatenate((
        omega * scales["ang_vel"], gravity,
        commands * np.array([scales["lin_vel"], scales["lin_vel"], scales["ang_vel"]], dtype=np.float32),
        relative_qpos * scales["dof_pos"],
        state[12 + joint_indices] * scales["dof_vel"], previous_action,
    )).astype(np.float32, copy=False)
