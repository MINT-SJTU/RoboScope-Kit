# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Run the public Kit CLI with ``rollout_trace`` installed in every simulator process.

All arguments go to ``python -m roboscope_kit.runtime.benchmark`` unchanged; set
``ROBOCASA365_TRACE_DIR``. Workers start through ``traced_worker.py``, which
installs the trace hook before the production worker's ``main()``. Spawned
parallel slots re-import that file as their main module, so they install it too.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from roboscope_kit.runtime.benchmarks.robocasa365.adapter import (
    RoboCasa365BenchmarkAdapter,
)

WORKER_MODULE = "roboscope_kit.runtime.benchmarks.robocasa365.worker"


def main() -> int:
    production = RoboCasa365BenchmarkAdapter.worker_process
    trace_dir = os.environ["ROBOCASA365_TRACE_DIR"]

    def traced(self, context):
        spec = production(self, context)
        if spec.command[1:3] != ("-m", WORKER_MODULE):
            raise RuntimeError("unexpected RoboCasa365 worker command")
        return replace(
            spec, command=(spec.command[0], str(Path(__file__).with_name("traced_worker.py")),
                           *spec.command[3:]),
            env={**spec.env, "ROBOCASA365_TRACE_DIR": trace_dir, "ROBOCASA365_TRACE_ROLE": "kit"},
        )

    RoboCasa365BenchmarkAdapter.worker_process = traced
    from roboscope_kit.runtime.benchmark.cli import main as kit_main

    return kit_main()


if __name__ == "__main__":
    raise SystemExit(main())
