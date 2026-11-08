# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""RoboTwin's independent client closes episode ranges on RPC failures."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def launcher(monkeypatch: pytest.MonkeyPatch):
    # The simulator supplies PyYAML; these client tests do not exercise YAML.
    monkeypatch.setitem(sys.modules, "yaml", types.ModuleType("yaml"))
    monkeypatch.delenv("ROBOSCOPE_PROFILE", raising=False)
    path = Path(__file__).resolve().parents[3] / "benchmarks/robotwin/roboscope_robotwin_launcher.py"
    spec = importlib.util.spec_from_file_location("robotwin_profiling_test_launcher", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Ranges:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def range_start(self, name: str) -> int:
        self.events.append(("start", name))
        return 7

    def range_end(self, token: int) -> None:
        self.events.append(("end", token))


class _Client:
    def __init__(self, *, fail_reset: bool, fail_end: bool = False) -> None:
        self.fail_reset = fail_reset
        self.fail_end = fail_end

    def request(self, operation: str, _payload: object) -> None:
        if operation == "reset" and self.fail_reset:
            raise RuntimeError("reset failed")
        if operation == "end_episode" and self.fail_end:
            raise RuntimeError("end failed")


def _model(launcher, *, fail_reset: bool, fail_end: bool = False):
    model = object.__new__(launcher.ModelClient)
    model.episode_id = ""
    model.profile_episode = None
    model.client = _Client(fail_reset=fail_reset, fail_end=fail_end)
    return model


def test_reset_failure_closes_episode_range(launcher, monkeypatch: pytest.MonkeyPatch) -> None:
    ranges = _Ranges()
    monkeypatch.setattr(launcher, "PROFILE_NVTX", ranges)
    model = _model(launcher, fail_reset=True)

    with pytest.raises(RuntimeError, match="reset failed"):
        model.reset("task")

    assert ranges.events == [("start", "episode"), ("end", 7)]
    assert model.profile_episode is None
    model.end_episode()
    assert ranges.events == [("start", "episode"), ("end", 7)]


def test_successful_episode_closes_range_once(launcher, monkeypatch: pytest.MonkeyPatch) -> None:
    ranges = _Ranges()
    monkeypatch.setattr(launcher, "PROFILE_NVTX", ranges)
    model = _model(launcher, fail_reset=False)

    model.reset("task")
    assert ranges.events == [("start", "episode")]
    model.end_episode()
    assert ranges.events == [("start", "episode"), ("end", 7)]
    assert model.profile_episode is None


def test_end_failure_closes_episode_range(launcher, monkeypatch: pytest.MonkeyPatch) -> None:
    ranges = _Ranges()
    monkeypatch.setattr(launcher, "PROFILE_NVTX", ranges)
    model = _model(launcher, fail_reset=False, fail_end=True)

    model.reset("task")
    with pytest.raises(RuntimeError, match="end failed"):
        model.end_episode()
    assert ranges.events == [("start", "episode"), ("end", 7)]
    assert model.profile_episode is None
