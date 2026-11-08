# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Skip unconsumed RGB renders without resetting robosuite's sensor clocks."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from functools import wraps
from importlib.metadata import version

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
        if version("robosuite") != "1.4.0":
            raise RuntimeError("chunk camera rendering requires pinned robosuite 1.4.0")
        from robosuite.utils.observables import NO_CORRUPTION, NO_DELAY, NO_FILTER

        native = env.env
        observables = native._observables
        images = {name for name, obs in observables.items() if obs.modality == "image"}
        if images != set(CAMERA_OBSERVATIONS):
            raise RuntimeError("chunk camera rendering requires the two standard LIBERO RGB sensors")
        originals = {}
        for name in CAMERA_OBSERVATIONS:
            obs = observables[name]
            image = observation[name]
            if (
                not obs.is_enabled()
                or not obs.is_active()
                or obs._corrupter is not NO_CORRUPTION
                or obs._filter is not NO_FILTER
                or obs._delayer is not NO_DELAY
                or image.dtype != np.uint8
                or image.ndim != 3
                or image.shape[-1] != 3
            ):
                raise RuntimeError(f"unsupported camera configuration for chunk rendering: {name}")
            originals[name] = obs._sensor

        with ExitStack() as stack:
            for name, original in originals.items():
                # set_sensor() validates the replacement by calling it once. Supply the
                # already available image for that call, without rendering or counting it.
                stack.callback(native.modify_observable, name, "sensor", original)
                self._installing = True
                try:
                    native.modify_observable(name, "sensor", self._sensor(original, observation[name]))
                finally:
                    self._installing = False
            try:
                yield
            finally:
                self.required = True
                # Restoring through the public API validates each original sensor once.
                # Those two cleanup renders are outside the step counters, but included
                # in the formal evaluation wall time. No installed source is modified.
