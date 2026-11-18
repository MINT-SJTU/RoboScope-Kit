# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Record per-slot rollout digests at the RoboCasa Gym wrapper boundary.

The hook wraps ``RoboCasaGymEnv`` without changing resets, actions or scoring.
Native and Kit runs install the same hook, so their traces compare directly:
one ``reset`` row per episode, one ``chunk`` row every eight physical steps (and
at termination), and one ``episode_end`` row. Each row carries SHA-256 digests of
the two-frame Policy features, the actions executed since the previous row, and
the MuJoCo integration state.

Set ``ROBOCASA365_TRACE_DIR`` and ``ROBOCASA365_TRACE_ROLE`` (``native`` or
``kit``) before calling ``install()``.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from collections import deque
from pathlib import Path

import numpy as np

CONSTRUCTOR_SEED = 1111111
NATIVE_ENVS = 5
CHUNK = 8
FEATURES = {
    "images.agentview_left": "video.robot0_agentview_left",
    "images.agentview_right": "video.robot0_agentview_right",
    "images.wrist": "video.robot0_eye_in_hand",
    "state.end_effector_position_relative": "state.end_effector_position_relative",
    "state.end_effector_rotation_relative": "state.end_effector_rotation_relative",
    "state.gripper_qpos": "state.gripper_qpos",
    "state.base_position": "state.base_position",
    "state.base_rotation": "state.base_rotation",
    "language.instruction": "annotation.human.task_description",
}


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def digest_features(history) -> tuple[str, dict[str, str]]:
    fields = {}
    combined = hashlib.sha256()
    for feature, key in sorted(FEATURES.items()):
        if feature == "language.instruction":
            data = history[-1][key].encode()
        else:
            dtype = np.uint8 if feature.startswith("images.") else np.float32
            data = np.stack([np.asarray(frame[key], dtype=dtype) for frame in history]).tobytes()
        fields[feature] = hashlib.sha256(data).hexdigest()
        combined.update(feature.encode())
        combined.update(data)
    return combined.hexdigest(), fields


def integration_digest(environment) -> str:
    import mujoco

    sim = environment.env.sim
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(sim.model._model, kind))
    mujoco.mj_getState(sim.model._model, sim.data._data, state, kind)
    return hashlib.sha256(state.tobytes()).hexdigest()


def _record(environment, event: str, **values) -> None:
    state = environment.__dict__["_rollout_trace"]
    row = {
        "event": event, "task": environment.env_name, "slot": state["slot"],
        "episode_index": state["group"] * NATIVE_ENVS + state["slot"],
        "stream_episode": state["group"], "step": state["step"], **values,
    }
    state["stream"].write(json.dumps(row, default=_json_default, sort_keys=True) + "\n")
    state["stream"].flush()


def _finish_episode(environment) -> None:
    state = environment.__dict__.get("_rollout_trace")
    if state is not None and state["group"] >= 0 and state["step"]:
        observation, fields = digest_features(state["history"])
        _record(environment, "episode_end", success=state["success"],
                integration_sha256=integration_digest(environment),
                observation_sha256=observation, feature_sha256=fields,
                action_sha256=state["actions"].hexdigest())


def install() -> None:
    from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv

    if getattr(RoboCasaGymEnv, "_rollout_trace_installed", False):
        return
    root = Path(os.environ["ROBOCASA365_TRACE_DIR"])
    role = os.environ["ROBOCASA365_TRACE_ROLE"]
    if role not in ("native", "kit"):
        raise ValueError("ROBOCASA365_TRACE_ROLE must be native or kit")
    original_init = RoboCasaGymEnv.__init__
    original_reset = RoboCasaGymEnv.reset
    original_step = RoboCasaGymEnv.step
    original_close = RoboCasaGymEnv.close

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # The native runner builds one metadata-only environment in its main process.
        if role == "native" and multiprocessing.current_process().name == "MainProcess":
            return
        slot = int(kwargs["seed"]) - CONSTRUCTOR_SEED
        if not 0 <= slot < NATIVE_ENVS:
            raise RuntimeError(f"unexpected RoboCasa365 constructor seed {kwargs['seed']}")
        directory = root / str(os.getpid()) / self.env_name
        directory.mkdir(parents=True, exist_ok=True)
        self._rollout_trace = {
            "slot": slot, "group": -1, "step": 0, "success": False,
            "stream": (directory / f"slot-{slot}.jsonl").open("x", buffering=1),
            "history": deque(maxlen=2), "actions": hashlib.sha256(),
        }

    def reset(self, *args, **kwargs):
        state = self.__dict__.get("_rollout_trace")
        if state is not None:
            _finish_episode(self)
        result = original_reset(self, *args, **kwargs)
        if state is None:
            return result
        frame, _ = result
        state.update(group=state["group"] + 1, step=0, success=False,
                     history=deque((frame, frame), maxlen=2), actions=hashlib.sha256())
        meta = json.dumps(self.env.get_ep_meta(), sort_keys=True, default=_json_default,
                          separators=(",", ":"))
        observation, fields = digest_features(state["history"])
        state_bytes = np.asarray(self.env.sim.get_state().flatten(), dtype=np.float64).tobytes()
        _record(self, "reset", initial_state_sha256=hashlib.sha256(state_bytes).hexdigest(),
                scenario_metadata_sha256=hashlib.sha256(meta.encode()).hexdigest(),
                initial_observation_sha256=observation, feature_sha256=fields,
                integration_sha256=integration_digest(self))
        return result

    def step(self, action):
        state = self.__dict__.get("_rollout_trace")
        if state is not None:
            state["actions"].update(json.dumps(
                action, sort_keys=True, default=_json_default, separators=(",", ":"),
            ).encode())
        result = original_step(self, action)
        if state is None:
            return result
        frame, _, terminated, truncated, info = result
        state["step"] += 1
        state["history"].append(frame)
        state["success"] |= bool(info["success"])
        if state["step"] % CHUNK == 0 or terminated or truncated:
            observation, fields = digest_features(state["history"])
            _record(self, "chunk", observation_sha256=observation, feature_sha256=fields,
                    action_sha256=state["actions"].hexdigest(),
                    integration_sha256=integration_digest(self), success=state["success"],
                    current_success=bool(info["success"]),
                    terminated=bool(terminated), truncated=bool(truncated))
            state["actions"] = hashlib.sha256()
        return result

    def close(self):
        state = self.__dict__.pop("_rollout_trace", None)
        if state is not None:
            self._rollout_trace = state
            _finish_episode(self)
            del self._rollout_trace
            state["stream"].close()
        return original_close(self)

    RoboCasaGymEnv.__init__ = initialize
    RoboCasaGymEnv.reset = reset
    RoboCasaGymEnv.step = step
    RoboCasaGymEnv.close = close
    RoboCasaGymEnv._rollout_trace_installed = True
