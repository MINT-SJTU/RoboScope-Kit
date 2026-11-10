"""Pinned official Unitree RL Gym TorchScript locomotion actors.

The models are transfer references for native ManiSkill standing tasks, not
ManiSkill-trained policies. Joint names and numeric conventions are explicit.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class LoadedUnitree:
    actor: torch.jit.ScriptModule
    metadata: dict[str, Any]

    def episode_actor(self) -> torch.jit.ScriptModule:
        actor = copy.deepcopy(self.actor)
        actor.reset_memory()
        return actor


def load_model_bundle(source: str | Path, device: str = "cuda") -> dict[str, LoadedUnitree]:
    source = Path(source).expanduser().resolve(strict=True)
    catalog = json.loads(Path(__file__).with_name("unitree-checkpoints.json").read_text("utf-8"))
    if source.is_dir():
        source = source / "models.json"
    if source.suffix == ".pt":
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        records = {
            task: {**record, "checkpoint": source}
            for task, record in catalog["models"].items() if record["sha256"] == digest
        }
        if len(records) != 1:
            raise ValueError("Unknown Unitree checkpoint; use a pinned artifact")
    else:
        bundle = json.loads(source.read_text("utf-8"))
        if bundle.get("schema_version") != 1 or not bundle.get("models"):
            raise ValueError("Expected a schema_version=1 bundle with a nonempty models mapping")
        records = {}
        for task, record in bundle["models"].items():
            if record != catalog["models"].get(task):
                raise ValueError(f"{task} differs from its pinned Unitree model contract")
            records[task] = {
                **record, "checkpoint": (source.parent / record["path"]).resolve(strict=True),
            }
    models = {}
    for task, record in records.items():
        checkpoint = record["checkpoint"]
        if checkpoint.stat().st_size != record["bytes"]:
            raise ValueError(f"{task} checkpoint size differs from its pin")
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"{task} checkpoint SHA-256 differs from its pin")
        actor = torch.jit.load(str(checkpoint), map_location=device).eval()
        for parameter in actor.parameters():
            parameter.requires_grad_(False)
        actor.reset_memory()
        if actor.memory.input_size != record["policy_observation_dimension"]:
            raise ValueError(f"{task} actor input dimension differs from its contract")
        models[task] = LoadedUnitree(actor, record)
    return models


def body_vector(world_vector: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """Rotate a world-frame vector into the root frame; quaternion is wxyz."""
    w, xyz = quaternion[0], quaternion[1:]
    return world_vector + 2 * np.cross(xyz, np.cross(xyz, world_vector) - w * world_vector)


def policy_observation(
    state: np.ndarray,
    root_pose: np.ndarray,
    root_angular_velocity: np.ndarray,
    previous_action: np.ndarray,
    leg_indices: np.ndarray,
    time_s: float,
    metadata: dict[str, Any],
) -> np.ndarray:
    dimension = metadata["observation_dimension"]
    if state.shape != (dimension,) or not np.isfinite(state).all():
        raise ValueError(f"Expected finite native joint state shape ({dimension},)")
    if root_pose.shape != (7,) or root_angular_velocity.shape != (3,):
        raise ValueError("Unitree requires native root pose xyz+wxyz and world angular velocity")
    if not np.isfinite(root_pose).all() or not np.isfinite(root_angular_velocity).all():
        raise ValueError("Root kinematics must be finite")
    quaternion = root_pose[3:]
    norm = np.linalg.norm(quaternion)
    if norm < 1e-8 or abs(float(norm) - 1) > 1e-3:
        raise ValueError("Native root quaternion must have unit norm")
    quaternion = quaternion / norm
    angular_velocity = body_vector(root_angular_velocity, quaternion)
    gravity = body_vector(np.array([0, 0, -1], dtype=np.float32), quaternion)
    joint_count = dimension // 2
    qpos, qvel = state[:joint_count], state[joint_count:]
    relative_qpos = qpos[leg_indices] - np.asarray(metadata["policy_default_angles"], dtype=np.float32)
    phase = 2 * np.pi * (time_s % metadata["gait_period_s"]) / metadata["gait_period_s"]
    command = np.asarray(metadata["velocity_command"], dtype=np.float32)
    return np.concatenate((
        angular_velocity * metadata["angular_velocity_scale"],
        gravity,
        command * np.array([2, 2, 0.25], dtype=np.float32),
        relative_qpos,
        qvel[leg_indices] * metadata["joint_velocity_scale"],
        previous_action,
        np.array([np.sin(phase), np.cos(phase)], dtype=np.float32),
    )).astype(np.float32, copy=False)


def native_action(
    actor_action: np.ndarray,
    state: np.ndarray,
    controller_state_indices: np.ndarray,
    metadata: dict[str, Any],
) -> np.ndarray:
    """Convert absolute actor leg targets to the native normalized delta control."""
    expected = len(metadata["policy_joint_names"])
    if actor_action.shape != (expected,) or not np.isfinite(actor_action).all():
        raise ValueError(f"Expected finite actor action shape ({expected},)")
    targets = np.asarray(metadata["native_nominal_positions"], dtype=np.float32).copy()
    controller_names = metadata["native_controller_joint_names"]
    for index, name in enumerate(metadata["policy_joint_names"]):
        targets[controller_names.index(name)] = (
            metadata["policy_default_angles"][index] + metadata["action_scale"] * actor_action[index]
        )
    delta = (targets - state[controller_state_indices]) / metadata["native_delta_range"]
    return np.ascontiguousarray(np.clip(delta, -1, 1), dtype=np.float32)
