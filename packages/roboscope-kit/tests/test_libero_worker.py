# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest


def _worker(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    # The worker saves videos with imageio, which only the LIBERO simulator environment installs.
    if importlib.util.find_spec("imageio") is None:
        monkeypatch.setitem(sys.modules, "imageio", types.ModuleType("imageio"))
    return importlib.import_module("roboscope_kit.runtime.benchmarks.libero.worker")


class _FakeEnv:
    """Records each step's action and whether the robot's OSC controller read it as a delta."""

    def __init__(self) -> None:
        self.controller = SimpleNamespace(use_delta=True)
        self.env = SimpleNamespace(robots=[SimpleNamespace(controller=self.controller)], done=False)
        self.steps: list[tuple[tuple[float, ...], bool]] = []

    def reset(self) -> None:
        pass

    def set_init_state(self, _state: np.ndarray) -> dict:
        return {}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        self.steps.append((tuple(float(value) for value in action), self.controller.use_delta))
        return {}, 0.0, False, {}


class _FakePolicy:
    def reset(self, _context: object) -> None:
        pass

    def step(self, _observation: dict, _task_description: str, _step: int) -> np.ndarray:
        return np.ones(7, dtype=np.float32)

    def end_episode(self) -> None:
        pass


@pytest.mark.parametrize("absolute_actions", (False, True))
def test_absolute_actions_switch_the_controller_after_settling(
    monkeypatch: pytest.MonkeyPatch,
    absolute_actions: bool,
) -> None:
    worker = _worker(monkeypatch)
    env = _FakeEnv()
    runtime = worker.TaskRuntime(task_id=0, task_description="task", initial_states=np.zeros((2, 1)), env=env)

    # Two consecutive episodes on one environment, as a worker runs them.
    for episode_index in range(2):
        success, steps, _elapsed_s, _frames = worker.run_episode(
            runtime,
            _FakePolicy(),
            episode_index,
            max_steps=4,
            settle_steps=3,
            save_frames=False,
            absolute_actions=absolute_actions,
        )
        assert (success, steps) == (False, 4)

    dummy = tuple(float(value) for value in worker.LIBERO_DUMMY_ACTION)
    episode = [(dummy, True)] * 3 + [((1.0,) * 7, not absolute_actions)] * 4
    assert env.steps == episode * 2


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("chunk_size", (1, 2))
def test_policy_action_queue_preserves_action_dtype_through_env_step(
    monkeypatch: pytest.MonkeyPatch,
    dtype: type[np.floating],
    chunk_size: int,
) -> None:
    worker = _worker(monkeypatch)
    raw = np.array(
        [
            [0.1234567890123456, -0.9876543210987654, 0.3333333333333333, 0.1, -0.2, 0.3, -1.0],
            [-0.4567890123456789, 0.6543210987654321, -0.7777777777777777, -0.1, 0.2, -0.3, 1.0],
        ],
        dtype=dtype,
    )
    chunk_value = raw[0] if chunk_size == 1 else raw[:chunk_size]
    expected_actions = [raw[0], raw[0]] if chunk_size == 1 else [raw[0], raw[1]]

    class _FakeClient:
        def __init__(self, _socket_path: object) -> None:
            pass

        def __enter__(self) -> _FakeClient:
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def reset(self, _context: object) -> None:
            pass

        def act(self, _observation: object) -> SimpleNamespace:
            return SimpleNamespace(actions=chunk_value.copy())

        def end_episode(self, _episode_id: str) -> None:
            pass

    monkeypatch.setattr(worker, "PolicyClient", _FakeClient)

    received: list[np.ndarray] = []

    class _RecordingEnv:
        def __init__(self) -> None:
            self.env = SimpleNamespace(robots=[], done=False)

        def reset(self) -> None:
            pass

        def set_init_state(self, _state: np.ndarray) -> dict:
            return {
                "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
                "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
                "robot0_eef_pos": np.zeros(3, dtype=np.float64),
                "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
                "robot0_gripper_qpos": np.zeros(2, dtype=np.float64),
                "robot0_joint_pos": np.zeros(7, dtype=np.float64),
            }

        def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
            received.append(action.copy())
            return self.set_init_state(np.zeros(1)), 0.0, False, {}

    env = _RecordingEnv()
    runtime = worker.TaskRuntime(task_id=0, task_description="task", initial_states=np.zeros((1, 1)), env=env)
    queue = worker.PolicyActionQueue(socket_path=worker.Path("/unused.sock"))

    success, steps, _elapsed_s, _frames = worker.run_episode(
        runtime,
        queue,
        0,
        max_steps=2,
        settle_steps=1,
        save_frames=False,
    )
    assert (success, steps) == (False, 2)
    policy_actions = received[1:]
    assert len(policy_actions) == 2
    for actual, expected in zip(policy_actions, expected_actions, strict=True):
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()
