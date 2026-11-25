# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Record per-step privileged LIBERO state for offline failure attribution.

The tracer only reads simulator state: it evaluates the task's BDDL goal predicates, gripper contacts and
grasps, and object poses after every policy step, so a traced episode has the same outcome as an untraced one.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

TRACE_SCHEMA = "roboscope-libero-trace-v1"
UNARY_ARTICULATION_PREDICATES = frozenset({"open", "close", "turnon", "turnoff"})


def _round(values: Any, digits: int = 4) -> list[float]:
    return [round(float(value), digits) for value in np.asarray(values, dtype=np.float64).reshape(-1)]


def observation_digest(observation: dict) -> str:
    digest = hashlib.blake2b(digest_size=8)
    for name in ("agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos", "robot0_gripper_qpos"):
        if name in observation:
            digest.update(np.ascontiguousarray(observation[name]).tobytes())
    return digest.hexdigest()


def action_digest(action: np.ndarray) -> str:
    payload = np.ascontiguousarray(action, dtype=np.float32).tobytes()
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


class LiberoEpisodeTracer:
    """Collect one episode's trace from a LIBERO ``ControlEnv``."""

    def __init__(self, env: object) -> None:
        self.control_env = env
        self.domain = env.env
        self.goal_state = [list(state) for state in self.domain.parsed_problem["goal_state"]]
        self.movable = [name for name in self.domain.objects_dict if name in self.domain.object_states_dict]
        self.goal_objects = self._goal_objects()
        tracked = list(dict.fromkeys(self.movable + [name for name in self.goal_objects if name]))
        self.tracked = [name for name in tracked if self._model(name) is not None]
        self.gripper = self.domain.robots[0].gripper
        self.steps: list[dict[str, Any]] = []
        self.first_observation_digest = ""

    def _model(self, name: str) -> object | None:
        state = self.domain.object_states_dict.get(name)
        if state is None:
            return None
        if getattr(state, "object_state_type", "object") == "site":
            return self.domain.get_object(state.parent_name)
        return self.domain.get_object(name)

    def _goal_objects(self) -> list[str]:
        """Name the object the robot must act on for each goal predicate."""
        names: list[str] = []
        for state in self.goal_state:
            name = state[1]
            object_state = self.domain.object_states_dict.get(name)
            if object_state is not None and getattr(object_state, "object_state_type", "") == "site":
                name = object_state.parent_name
            names.append(name)
        return names

    def _predicates(self) -> list[bool]:
        return [bool(self.domain._eval_predicate(state)) for state in self.goal_state]

    def _body_pos(self, name: str) -> list[float]:
        state = self.domain.object_states_dict[name]
        if getattr(state, "object_state_type", "object") == "site":
            return _round(self.domain.sim.data.get_site_xpos(name))
        return _round(self.domain.sim.data.body_xpos[self.domain.obj_body_id[name]])

    def start(self, observation: dict) -> None:
        self.steps = []
        self.first_observation_digest = observation_digest(observation)

    def record(self, step: int, action: np.ndarray, observation: dict, inferred: bool) -> None:
        contact: list[str] = []
        grasp: list[str] = []
        positions: dict[str, list[float]] = {}
        for name in self.tracked:
            model = self._model(name)
            if self.domain.check_contact(self.gripper, model):
                contact.append(name)
                if name in self.domain.objects_dict and self.domain._check_grasp(self.gripper, model):
                    grasp.append(name)
            if name in self.domain.objects_dict:
                positions[name] = self._body_pos(name)
        self.steps.append(
            {
                "step": step,
                "action": _round(action),
                "action_digest": action_digest(action),
                "inferred": inferred,
                "observation_digest": observation_digest(observation),
                "eef_pos": _round(observation["robot0_eef_pos"]),
                "gripper_qpos": _round(observation["robot0_gripper_qpos"]),
                "predicates": self._predicates(),
                "contact": contact,
                "grasp": grasp,
                "positions": positions,
            }
        )

    def goal_positions(self) -> dict[str, list[float]]:
        positions: dict[str, list[float]] = {}
        for state in self.goal_state:
            for name in state[1:]:
                if name in self.domain.object_states_dict:
                    positions[name] = self._body_pos(name)
        return positions

    def write(self, path: Path, header: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "schema": TRACE_SCHEMA,
            **header,
            "goal_state": self.goal_state,
            "goal_objects": self.goal_objects,
            "movable_objects": self.movable,
            "goal_positions_final": self.goal_positions(),
            "first_observation_digest": self.first_observation_digest,
            "steps": self.steps,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            json.dump(document, handle, separators=(",", ":"))
        temporary.replace(path)


__all__ = [
    "LiberoEpisodeTracer",
    "TRACE_SCHEMA",
    "UNARY_ARTICULATION_PREDICATES",
    "action_digest",
    "observation_digest",
]
