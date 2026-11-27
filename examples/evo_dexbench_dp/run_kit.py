# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Run the public Kit CLI on a subset of one Evo-DexBench suite, for native comparisons.

All arguments go to ``python -m roboscope_kit.runtime.benchmark`` unchanged. Set
``EVO_DEXBENCH_TASKS`` to comma-separated task ids of the chosen ``--suite``
(for example ``m02_v1``); each keeps its full evaluation_v1 schedule, so the
result pairs with ``dexbench evaluate --task-id`` on the same task. The run is
recorded as a subset unless it lists the whole suite in order. ``--quick-test``
runs one wave of the first listed task: upstream's four-episode schedule.
"""

from __future__ import annotations

import os

from roboscope_kit.runtime.benchmarks.evo_dexbench.factory import EvoDexBenchBenchmarkFactory
from roboscope_kit.runtime.benchmarks.evo_dexbench.protocol import (
    EPISODES_PER_TASK,
    NATIVE_ENVS,
    SUITES,
)


def main() -> int:
    tasks = tuple(task for task in os.environ["EVO_DEXBENCH_TASKS"].split(",") if task)

    def create(self, request, manifest):
        self.validate_suite(request.suite)
        if not tasks or set(tasks) - set(SUITES[request.suite]):
            raise SystemExit(f"EVO_DEXBENCH_TASKS {tasks} are not tasks of suite {request.suite}")
        if request.quick_test:
            return self._create(request, manifest, tasks[:1], NATIVE_ENVS)
        episodes = request.episodes or int(manifest.defaults.get("episodes", EPISODES_PER_TASK))
        return self._create(request, manifest, tasks, episodes)

    EvoDexBenchBenchmarkFactory.create = create
    from roboscope_kit.runtime.benchmark.cli import main as kit_main

    return kit_main()


if __name__ == "__main__":
    raise SystemExit(main())
