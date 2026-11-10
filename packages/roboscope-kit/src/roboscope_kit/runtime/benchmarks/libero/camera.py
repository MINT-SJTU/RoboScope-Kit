# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Skip unconsumed RGB renders without resetting robosuite's sensor clocks."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps

import numpy as np

from roboscope_kit import timing

CAMERA_OBSERVATIONS = ("agentview_image", "robot0_eye_in_hand_image")


class ChunkCameraGate:
    """Gate only RGB sensor work; the native Observable still advances every substep.

    This deliberately uses the pinned robosuite 1.4.0 Observable implementation. Toggling
    ``enabled`` would reset its clock, while forcing a render after env.step() would sample
    at a different physics timestep. Neither preserves the baseline's policy observations.
    """

    def __init__(self, counters: dict[str, int]) -> None:
        self.required = True
        self._installing = False
        self.counters = counters

    def _sensor(self, original, initial_image: np.ndarray):
        cached = initial_image

        @wraps(original)
        def sample(obs_cache):
            nonlocal cached
            if self._installing:
                return cached
            if self.required:
                with timing.span("worker.rgb_render"):
                    cached = original(obs_cache)
                self.counters["camera_renders"] += 1
            else:
                self.counters["camera_renders_skipped"] += 1
            return cached

        return sample

    @contextmanager
    def install(self, env: object, observation: dict, *, enabled: bool) -> Iterator[None]:
        if not enabled and not timing.ENABLED:
            yield
            return

        observables = env.env._observables
        originals = {name: observables[name]._sensor for name in CAMERA_OBSERVATIONS}

        try:
            for name, original in originals.items():
                observables[name]._sensor = self._sensor(original, observation[name])
            yield
        finally:
            self.required = True
            for name, original in originals.items():
                observables[name]._sensor = original
