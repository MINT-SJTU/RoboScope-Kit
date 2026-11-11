"""Direct-pull PPO prior using the native tool task's fixed workspace target."""

from __future__ import annotations

import numpy as np
import torch

# TableSceneBuilder's fixed Panda base plus PullCubeTool.compute_dense_reward's
# workspace offset at the pinned simulator revision. This is not a sampled goal.
WORKSPACE_TARGET = np.add(
    np.array([-0.615, 0.0, 0.0], dtype=np.float32),
    np.array([0.05, 0.0, 0.0], dtype=np.float32),
    dtype=np.float32,
)


def project_actor_input(state: np.ndarray) -> np.ndarray:
    """Keep actual joint/TCP/cube state; the unchanged actor ignores the tool."""
    value = np.asarray(state, dtype=np.float32)
    if value.shape != (39,) or not np.isfinite(value).all():
        raise ValueError("Expected the finite complete canonical PullCubeTool state39")
    return np.ascontiguousarray(
        np.concatenate((value[:25], WORKSPACE_TARGET, value[25:32])), dtype=np.float32,
    )


def project_native_actor_input(state: torch.Tensor) -> torch.Tensor:
    """Apply the same field mapping to native GPU batches without host copies."""
    if state.ndim != 2 or state.shape[1] != 39:
        raise ValueError("Expected a native PullCubeTool batch with 39 fields")
    target = torch.as_tensor(WORKSPACE_TARGET, dtype=state.dtype, device=state.device)
    return torch.cat((state[:, :25], target.expand(state.shape[0], 3), state[:, 25:32]), dim=1)
