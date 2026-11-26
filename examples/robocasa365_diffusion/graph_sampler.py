# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Replay the official DDPM sampling loop as one CUDA graph per batch shape.

The graph launches the same kernels with the same inputs as the pinned
``conditional_sample``, and draws its noise from the default CUDA generator at
the same offsets, so actions and the generator state after a call are
bit-identical to eager sampling. It removes the per-kernel launch cost of the
100 denoising steps. Observation encoding stays eager.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

WARMUP_CALLS = 2


@contextmanager
def _known_causal_masks():
    """Reuse the eager causal-mask decision while capturing.

    ``nn.Transformer*`` checks whether a mask is causal by comparing it on the
    device, which synchronizes and cannot be captured. The policy's masks are
    constant buffers, so the eager answer from warm-up applies unchanged.
    """
    from torch.nn.modules import transformer

    detect = transformer._detect_is_causal_mask
    answers: dict[tuple, bool] = {}

    def cached(mask, is_causal=None, size=None):
        if mask is None or is_causal is not None:
            return detect(mask, is_causal, size)
        key = (mask.data_ptr(), tuple(mask.shape), size)
        if key not in answers:
            answers[key] = detect(mask, is_causal, size)
        return answers[key]

    transformer._detect_is_causal_mask = cached
    try:
        yield
    finally:
        transformer._detect_is_causal_mask = detect


class GraphedSampler:
    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self.graphs: dict[tuple, tuple] = {}

    def _loop(self, condition_data, condition_mask, cond, timesteps_device, timesteps):
        import torch

        policy, scheduler = self.policy, self.policy.noise_scheduler
        trajectory = torch.randn(
            size=condition_data.shape, dtype=condition_data.dtype, device=condition_data.device,
        )
        for index, timestep in enumerate(timesteps):
            # Equal to the masked assignment of the eager loop, without its host sync.
            trajectory = torch.where(condition_mask, condition_data, trajectory)
            output = policy.model(trajectory, timesteps_device[index], cond)
            trajectory = scheduler.step(output, timestep, trajectory, **policy.kwargs).prev_sample
        return torch.where(condition_mask, condition_data, trajectory)

    def _capture(self, condition_data, condition_mask, cond) -> tuple:
        import torch

        scheduler = self.policy.noise_scheduler
        scheduler.set_timesteps(self.policy.num_inference_steps)
        timesteps = list(scheduler.timesteps)
        timesteps_device = scheduler.timesteps.to(condition_data.device)
        inputs = [condition_data.clone(), condition_mask.clone(), None if cond is None else cond.clone()]
        state = torch.cuda.get_rng_state(condition_data.device)
        with _known_causal_masks():
            stream = torch.cuda.Stream(condition_data.device)
            stream.wait_stream(torch.cuda.current_stream(condition_data.device))
            with torch.cuda.stream(stream):
                for _ in range(WARMUP_CALLS):
                    self._loop(*inputs, timesteps_device, timesteps)
            torch.cuda.current_stream(condition_data.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            # torch.cuda.graph's default capture stream is on the current device, not the Policy's.
            with torch.cuda.graph(graph, stream=stream):
                output = self._loop(*inputs, timesteps_device, timesteps)
        torch.cuda.set_rng_state(state, condition_data.device)
        # The graph reads every tensor below by address; keep them alive with it.
        return graph, inputs, output, timesteps_device

    def __call__(self, condition_data, condition_mask, cond=None, generator=None, **kwargs):
        if generator is not None or kwargs:
            raise ValueError("the graphed sampler supports the official default generator only")
        key = (tuple(condition_data.shape), None if cond is None else tuple(cond.shape))
        if key not in self.graphs:
            self.graphs[key] = self._capture(condition_data, condition_mask, cond)
        graph, inputs, output, _ = self.graphs[key]
        inputs[0].copy_(condition_data)
        inputs[1].copy_(condition_mask)
        if cond is not None:
            inputs[2].copy_(cond)
        graph.replay()
        return output.clone()


def install(policy: Any) -> None:
    """Make ``policy.predict_action`` sample through CUDA graphs."""
    policy.conditional_sample = GraphedSampler(policy)
