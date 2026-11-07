"""Synthetic latency-bound Policy for measuring RoboDojo concurrency.

Stands in for an agent-based Policy: it uses no GPU, holds the current joint
positions, and spends a fixed time in every ``act()`` call::

    latency_policy.py --latency-s 3 --chunk 10

It measures how throughput scales with concurrent episodes, not task success:
holding still never solves a task, so every episode runs to its step limit.
Run it with one replica per Worker (``--concurrency K --replicas K``), because
a replica serves one request at a time.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from roboscope_kit.policy import (
    ActionChunk,
    Observation,
    Policy,
    PolicyCapabilities,
    SessionContext,
)

BENCHMARK = "robodojo"
SUITES = ("arx_x5", "arx_x5_randomized")
ACTION_REPRESENTATION = "robodojo.x5-joint-v1"


class LatencyPolicy(Policy):
    def __init__(self, latency_s: float, chunk: int) -> None:
        self.latency_s = latency_s
        self.chunk = chunk

    def load(self, context: SessionContext) -> PolicyCapabilities:
        if context.benchmark.name != BENCHMARK:
            raise ValueError(f"this example supports only {BENCHMARK!r}; got {context.benchmark.name!r}")
        return PolicyCapabilities(
            supported_benchmarks=(BENCHMARK,),
            supported_suites={BENCHMARK: SUITES},
            required_features=("state.joints",),
            max_batch_size=1,
            supported_action_chunk_sizes=(self.chunk,),
            action_representations={BENCHMARK: ACTION_REPRESENTATION},
        )

    def act(self, observation: Observation) -> ActionChunk:
        time.sleep(self.latency_s)
        joints = np.asarray(observation.features["state.joints"], dtype=np.float32)
        return ActionChunk(actions=np.tile(joints, (self.chunk, 1)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--latency-s", type=float, default=3.0, help="time spent in each act() call")
    parser.add_argument("--chunk", type=int, default=10, help="actions returned by each act() call")
    args = parser.parse_args()
    if args.latency_s < 0 or args.chunk < 1:
        parser.error("--latency-s must be nonnegative and --chunk positive")

    from roboscope_kit.server import run  # noqa: PLC0415

    run(LatencyPolicy(args.latency_s, args.chunk))


if __name__ == "__main__":
    main()
