"""Retarget the original ANYmal Reach PPO actor to canonical Go2 Reach.

Both native tasks expose qpos12, qvel12, world linear/angular velocity6,
reached-goal1, world goal xy2 and robot-to-goal xy2. All 35 values are retained;
this transfer constructs no root orientation or missing sensor measurement.

The frozen asset joint axes agree in the base frame at zero configuration.
Home-centered coordinates associate corresponding joints without a sign flip.
This is a cross-morphology learned prior, not exact kinematics, dynamics, or a
Go2-trained expert. Geometry, limits and PD gains remain those of native Go2.
"""

from __future__ import annotations

import numpy as np
import torch

SOURCE_JOINT_NAMES = (
    "LF_HAA", "RF_HAA", "LH_HAA", "RH_HAA",
    "LF_HFE", "RF_HFE", "LH_HFE", "RH_HFE",
    "LF_KFE", "RF_KFE", "LH_KFE", "RH_KFE",
)
TARGET_JOINT_NAMES = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
)
SOURCE_HOME = np.asarray(
    [.03, -.03, .03, -.03, .4, .4, -.4, -.4, -.8, -.8, .8, .8], dtype=np.float32,
)
TARGET_HOME = np.asarray([0.] * 4 + [.9] * 4 + [-1.8] * 4, dtype=np.float32)
SOURCE_LOW, SOURCE_HIGH = np.float32(-.225), np.float32(.225)
TARGET_LOW, TARGET_HIGH = np.float32(-.7), np.float32(.7)


def project_actor_input(state: np.ndarray) -> np.ndarray:
    """Map every native Go2 joint measurement into source home coordinates."""
    value = np.asarray(state, dtype=np.float32)
    if value.shape != (35,) or not np.isfinite(value).all():
        raise ValueError("The Go2 Reach prior requires the complete finite native 35-vector")
    result = value.copy()
    result[:12] = SOURCE_HOME + (value[:12] - TARGET_HOME)
    return result


def project_native_actor_input(state: torch.Tensor) -> torch.Tensor:
    """Apply the same FP32 projection to direct-native batches."""
    if state.ndim != 2 or state.shape[1] != 35 or state.dtype != torch.float32:
        raise ValueError("Expected a native Go2 FP32 batch of shape (N, 35)")
    result = state.clone()
    source_home = torch.as_tensor(SOURCE_HOME, device=state.device)
    target_home = torch.as_tensor(TARGET_HOME, device=state.device)
    result[:, :12] = source_home + (state[:, :12] - target_home)
    return result


def project_action(actions: np.ndarray) -> np.ndarray:
    """Preserve all source joint deltas using the unchanged target controller."""
    value = np.asarray(actions, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 12 or not np.isfinite(value).all():
        raise ValueError("The Go2 Reach prior requires finite actor batches of shape (N, 12)")
    # Match the source native controller's clip-and-scale operation before the
    # target inverse normalization. Scaling unclipped actors would change intent.
    physical = (
        np.float32(.5) * (SOURCE_HIGH + SOURCE_LOW)
        + np.float32(.5) * (SOURCE_HIGH - SOURCE_LOW) * np.clip(value, -1, 1)
    )
    return np.ascontiguousarray(
        (physical - np.float32(.5) * (TARGET_HIGH + TARGET_LOW))
        / (np.float32(.5) * (TARGET_HIGH - TARGET_LOW)), dtype=np.float32,
    )


def project_native_action(actions: torch.Tensor) -> torch.Tensor:
    """Apply the original source clipping and complete target normalization."""
    if actions.ndim != 2 or actions.shape[1] != 12 or actions.dtype != torch.float32:
        raise ValueError("Expected an ANYmal FP32 actor batch of shape (N, 12)")
    physical = (
        float(np.float32(.5) * (SOURCE_HIGH + SOURCE_LOW))
        + float(np.float32(.5) * (SOURCE_HIGH - SOURCE_LOW)) * torch.clip(actions, -1, 1)
    )
    return (
        physical - float(np.float32(.5) * (TARGET_HIGH + TARGET_LOW))
    ) / float(np.float32(.5) * (TARGET_HIGH - TARGET_LOW))
