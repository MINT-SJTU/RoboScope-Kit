# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Start the production RoboCasa365 worker with ``rollout_trace`` installed."""

import rollout_trace

rollout_trace.install()

if __name__ == "__main__":
    from roboscope_kit.runtime.benchmarks.robocasa365.worker import main

    main()
