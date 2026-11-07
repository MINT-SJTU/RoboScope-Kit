# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Avoid preprocessing observations discarded by the single-frame action wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import gymnasium as gym


def enable_chunk_observations(environment: gym.Wrapper) -> gym.Wrapper:
    """Install underneath the pinned MultiStepWrapper without eager simulator imports."""
    import gymnasium as gym

    class ChunkObservationWrapper(gym.Wrapper):
        """Keep every native sensor call and preprocess only consumed steps.

        The upstream MultiStepWrapper uses only the final observation of each action
        chunk (delta indices [0]). With video disabled, intermediate transformed
        observations are not consumed by the policy. Still render each native
        camera sample and run the basic observation path, including its cache
        updates; only defer the final Groot conversion until a consumed step.
        """

        def __init__(self, env: gym.Env, action_steps: int, max_episode_steps: int) -> None:
            super().__init__(env)
            self.action_steps = action_steps
            self.max_episode_steps = max_episode_steps
            self.steps = 0
            self.deferred_steps = 0
            self.native = env.unwrapped.env
            self.last_observation = None

        def reset(self, **kwargs):
            self.steps = 0
            self.deferred_steps = 0
            result = self.env.reset(**kwargs)
            self.last_observation = result[0]
            return result

        def _observed_step(self, action):
            result = self.env.step(action)
            self.last_observation = result[0]
            return result

        def step(self, action):
            from robosuite.utils.observables import NO_CORRUPTION, NO_DELAY, NO_FILTER

            self.steps += 1
            consumed = self.steps % self.action_steps == 0 or self.steps >= self.max_episode_steps
            # Early native termination, temporal filters, depth and segmentation
            # require intermediate images. Retain the upstream path in those cases.
            if (
                consumed
                or not self.native.ignore_done
                or any(self.native.camera_depths)
                or any(self.native.camera_segmentations)
            ):
                return self._observed_step(action)
            cameras = [
                observable
                for observable in self.native._observables.values()
                if observable.modality == "image" and observable.is_enabled()
            ]
            if not cameras or any(
                observable._corrupter is not NO_CORRUPTION
                or observable._filter is not NO_FILTER
                or observable._delayer is not NO_DELAY
                or np.asarray(observable.obs).ndim != 3
                for observable in cameras
            ):
                return self._observed_step(action)
            groot = self.env.unwrapped
            groot_observation = groot.get_groot_observation
            try:
                # No video recorder or history consumes the intermediate Groot
                # conversions. Native sensors and the basic conversion still run.
                groot.get_groot_observation = lambda _raw: self.last_observation
                self.deferred_steps += 1
                return self.env.step(action)
            finally:
                groot.get_groot_observation = groot_observation

    if not np.array_equal(environment.video_delta_indices, [0]) or not np.array_equal(
        environment.state_delta_indices, [0]
    ):
        raise ValueError("RoboCasa chunk rendering requires single-frame observation histories")
    wrapper = ChunkObservationWrapper(
        environment.env, environment.n_action_steps, environment.max_episode_steps
    )
    environment.env = wrapper
    return wrapper
