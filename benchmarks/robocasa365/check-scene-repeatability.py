# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Show whether one constructor seed gives the same initial scene in fresh processes.

Run with a runtime's interpreter from its ``src/robocasa`` directory. Each task is
built with constructor seed 1111111 and reset once, as slot 0 of a native group,
in ``--processes`` separate interpreters. The digests of the episode metadata and
of the initial simulator state are printed per process; an uncorrected runtime
prints differing digests, the corrected runtime identical ones.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

CHILD = """
import hashlib, json, sys
import numpy as np
import gymnasium as gym
import robocasa.wrappers.gym_wrapper  # noqa: F401

def default(value):
    return value.tolist() if isinstance(value, np.ndarray) else value.item()

env = gym.make(f"robocasa/{sys.argv[1]}", split="pretrain", seed=1111111,
               camera_widths=256, camera_heights=256)
env.reset()
native = env.unwrapped.env
meta = json.dumps(native.get_ep_meta(), sort_keys=True, default=default, separators=(",", ":"))
state = np.asarray(native.sim.get_state().flatten(), dtype=np.float64)
print(json.dumps({"scenario": hashlib.sha256(meta.encode()).hexdigest()[:16],
                  "state": hashlib.sha256(state.tobytes()).hexdigest()[:16]}))
env.close()
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", nargs="+")
    parser.add_argument("--processes", type=int, default=2)
    args = parser.parse_args()
    rows = {}
    for task in args.tasks:
        runs = []
        for _ in range(args.processes):
            output = subprocess.run((sys.executable, "-c", CHILD, task), check=True,
                                    capture_output=True, text=True).stdout
            runs.append(json.loads(output.strip().splitlines()[-1]))
        rows[task] = {"runs": runs, "identical": all(run == runs[0] for run in runs)}
        print(json.dumps({task: rows[task]}), flush=True)
    print(json.dumps({"identical_tasks": sum(row["identical"] for row in rows.values()),
                      "tasks": len(rows)}))


if __name__ == "__main__":
    main()
