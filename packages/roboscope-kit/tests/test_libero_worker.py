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
