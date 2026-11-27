# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Skip unconsumed RGB renders within action chunks while preserving sensor clocks."""

from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import wraps
from importlib.metadata import version
from typing import Any

import numpy as np

from roboscope_kit import profiling

CAMERA_OBSERVATIONS = ("agentview_image", "robot0_eye_in_hand_image")
PINNED_ROBOSUITE_VERSION = "1.4.0"
_VERSION_VERIFIED = False


def _verify_robosuite_version() -> None:
    global _VERSION_VERIFIED
    if _VERSION_VERIFIED:
        return
    actual = version("robosuite")
    if actual != PINNED_ROBOSUITE_VERSION:
        raise RuntimeError(
            f"ChunkCameraGate requires robosuite {PINNED_ROBOSUITE_VERSION}, found {actual}"
        )
    _VERSION_VERIFIED = True


def _verify_camera_observables(observables: Any, observation: Mapping[str, Any]) -> None:
    if not isinstance(observables, Mapping):
        raise RuntimeError("robosuite environment does not expose an _observables mapping")
    for name in CAMERA_OBSERVATIONS:
        if name not in observables:
            raise RuntimeError(f"robosuite environment is missing camera observable {name!r}")
        if not isinstance(observation.get(name), np.ndarray):
            raise RuntimeError(f"initial observation is missing RGB ndarray {name!r}")
    rs_obs = sys.modules.get("robosuite.utils.observables")
    no_corruption = getattr(rs_obs, "NO_CORRUPTION", None)
    no_filter = getattr(rs_obs, "NO_FILTER", None)
    no_delay = getattr(rs_obs, "NO_DELAY", None)
    for name, observable in observables.items():
        if not getattr(observable, "_active", True):
            continue
        sensor = getattr(observable, "_sensor", None)
        modality = getattr(observable, "modality", getattr(sensor, "__modality__", None))
        is_image = modality == "image" or name.endswith("_image")
        if name in CAMERA_OBSERVATIONS:
            if not callable(sensor):
                raise RuntimeError(f"camera observable {name!r} has no callable _sensor")
            corrupter = getattr(observable, "_corrupter", getattr(observable, "_corruptor", None))
            if corrupter not in (None, no_corruption):
                raise RuntimeError(f"camera observable {name!r} uses a corruptor; gating is unsafe")
            if getattr(observable, "_filter", None) not in (None, no_filter):
                raise RuntimeError(f"camera observable {name!r} uses a filter; gating is unsafe")
            if (
                getattr(observable, "_delayer", None) not in (None, no_delay)
                or getattr(observable, "_delay", 0) != 0
                or getattr(observable, "_current_delay", 0.0) != 0.0
            ):
                raise RuntimeError(f"camera observable {name!r} uses sensor delay; gating is unsafe")
        elif is_image:
            raise RuntimeError(
                f"unexpected active image observable {name!r}; expected only {CAMERA_OBSERVATIONS}"
            )


class ChunkCameraGate:
    """Gate RGB sensor execution while robosuite's native Observable clock advances.

    Toggling ``Observable.enabled`` would reset its sampling clock, and rendering after
    ``env.step()`` would sample at a different physics substep. Replacing ``_sensor`` for
    the episode preserves exact native sampling substeps for every consumed observation.
    """

    def __init__(self) -> None:
        self.required = True
        self.renders = 0
        self.skipped = 0

    def _make_sensor(self, original: Any, initial_image: np.ndarray, *, timed: bool) -> Any:
        cached = initial_image

        if timed:

            @wraps(original)
            def timed_sample(obs_cache: Any) -> np.ndarray:
                nonlocal cached
                if self.required:
                    with profiling.span("worker.rgb_render"):
                        cached = original(obs_cache)
                    self.renders += 1
                else:
                    self.skipped += 1
                return cached

            return timed_sample

        @wraps(original)
        def sample(obs_cache: Any) -> np.ndarray:
            nonlocal cached
            if self.required:
                cached = original(obs_cache)
                self.renders += 1
            else:
                self.skipped += 1
            return cached

        return sample

    @contextmanager
    def install(
        self,
        env: Any,
        observation: Mapping[str, Any],
        *,
        enabled: bool,
        verify_version: bool = True,
    ) -> Iterator[None]:
        timed = profiling.TIMING_ENABLED
        if not enabled and not timed:
            yield
            return

        if verify_version:
            _verify_robosuite_version()
        observables = getattr(getattr(env, "env", None), "_observables", None)
        _verify_camera_observables(observables, observation)
        originals = {name: observables[name]._sensor for name in CAMERA_OBSERVATIONS}

        try:
            for name, original in originals.items():
                observables[name]._sensor = self._make_sensor(
                    original,
                    observation[name],
                    timed=timed,
                )
            yield
        finally:
            self.required = True
            for name, original in originals.items():
                observables[name]._sensor = original
