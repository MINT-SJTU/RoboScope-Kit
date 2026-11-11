"""Model-side full-action projection for the LiftCube prior on canonical LiftPeg.

This helper preserves the frozen 32D input and 4D native controller. Source pins:
ManiSkill2 90a628059cbdbcd76303b899b192efc4a38beddc; native ManiSkill
baab60ede2e89167c1b7aaed41a9aa8e690a9d1e. Both Panda bases are truly
[-0.615, 0, 0] with identity orientation in their pinned table scene builders.

The source arm translates in the current TCP frame; the frozen native arm
translates in the root frame. Bounds are +/-0.1 m for both. Rotated components
can exceed native [-1, 1]; this helper deliberately does not hide that mismatch
by all-axis rescaling or pre-clipping. The native controller clips components.
Source fourth-channel output remains the active learned gripper command.
"""

from __future__ import annotations

import numpy as np

SOURCE_BASE_POSE = np.array([-0.615, 0, 0, 1, 0, 0, 0], dtype=np.float32)
SOURCE_BASE_POSE.setflags(write=False)
ACTION_REPEAT = 2


def _vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float32)
    if out.shape != (size,) or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be a finite {size}-vector")
    return out


def project_actor_input(canonical_state: np.ndarray) -> np.ndarray:
    """Retain real native fields and add only true base pose/derived displacement."""
    state = _vector(canonical_state, 32, "canonical_state")
    return np.concatenate(
        (state[:18], SOURCE_BASE_POSE, state[18:25], state[25:32],
         state[25:28] - state[18:21])
    ).astype(np.float32, copy=False)


def _root_from_tcp(wxyz: np.ndarray) -> np.ndarray:
    # Geometric quaternion rotation, independent of SciPy used by the oracle.
    q = np.asarray(wxyz, dtype=np.float64)
    length = np.linalg.norm(q)
    if not np.isfinite(length) or length < 1e-12:
        raise ValueError("TCP quaternion must have nonzero finite norm")
    w, x, y, z = q / length
    return np.array(
        [[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
         [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
         [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]],
        dtype=np.float64,
    )


def project_action(canonical_state: np.ndarray, source_action: np.ndarray) -> np.ndarray:
    """Rotate complete source XYZ into native root axes; retain its fourth value.

    Source control clips normalized XYZ before translating. The returned XYZ
    may exceed native component bounds; the actual native controller, not this
    helper, clips it. The raw fourth value is copied without slicing/replacing
    the learned grip; identical source/native gripper clipping and physical
    bounds then retain its effective target. In each repeated source tick call
    this function again with the new real TCP orientation and the same source4.
    """
    state = _vector(canonical_state, 32, "canonical_state")
    action = _vector(source_action, 4, "source_action")
    result = np.empty(4, dtype=np.float32)
    result[:3] = _root_from_tcp(state[21:25]) @ np.clip(action[:3], -1, 1)
    result[3] = action[3]
    return result


class SourceActionRepeat:
    """Hold the original source4 and recompute its frame on each repeated tick.

    This does not claim full MPC/source-evaluator parity. A parent model wrapper
    must preserve its chosen original inference mode and per-episode state.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._source_action = None
        self._remaining = 0

    @property
    def needs_source_action(self) -> bool:
        return self._remaining == 0

    def next_action(self, canonical_state: np.ndarray,
                    new_source_action: np.ndarray | None = None) -> np.ndarray:
        if self.needs_source_action:
            if new_source_action is None:
                raise ValueError("a new source action is required at a repeat boundary")
            self._source_action = _vector(new_source_action, 4, "new_source_action").copy()
            self._remaining = ACTION_REPEAT
        elif new_source_action is not None:
            raise ValueError("do not replace source action inside its two-tick repeat")
        result = project_action(canonical_state, self._source_action)
        self._remaining -= 1
        return result
