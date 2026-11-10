# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Read the PerAct test episodes and check them against the frozen digests.

A test episode directory of the released PerAct test set holds, among the recorded images,
``low_dim_obs.pkl`` (an RLBench ``Demo`` whose ``random_seed`` is the NumPy random state the
scene was sampled from), ``variation_number.pkl`` and ``variation_descriptions.pkl``. PerAct's
``reset_to_demo`` reads only the variation and the random state to rebuild the scene; the worker
does the same and never loads the images.

The pickles are read with an unpickler that only resolves the RLBench and NumPy classes such a
file contains, so a modified test set cannot run code in the worker.

Runs in the RLBench runtime environment (``build-runtime.sh`` and ``--self-test`` call it)::

    python -m roboscope_kit.runtime.benchmarks.rlbench.test_set <test-set-root> \\
        [--task close_jar ...] [--print]

A per-task digest that differs from ``tasks.TASKS`` exits with status 1.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tasks import (
    EPISODES_PER_TASK,
    TASK_NAMES,
    TASKS_BY_NAME,
    episode_state_sha256,
    test_episodes_sha256,
)

_ALLOWED_GLOBALS = {
    ("rlbench.demo", "Demo"),
    ("rlbench.backend.observation", "Observation"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
}


class _TestSetUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"PerAct test-set pickle references {module}.{name}")
        return super().find_class(module, name)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as stream:
        return _TestSetUnpickler(stream).load()


@dataclass(frozen=True)
class TestEpisode:
    task: str
    episode: int
    variation: int
    random_state: tuple
    descriptions: tuple[str, ...]
    # RLBench's task low-dimensional state (object poses) in the recorded first frame.
    recorded_task_state: Any = None

    @property
    def state_sha256(self) -> str:
        return episode_state_sha256(self.variation, self.random_state)


def episode_dir(root: Path, task: str, episode: int) -> Path:
    return root / task / "all_variations" / "episodes" / f"episode{episode}"


def load_test_episode(root: Path, task: str, episode: int) -> TestEpisode:
    """The variation and random state PerAct's ``reset_to_demo`` restores for one test episode."""

    path = episode_dir(root, task, episode)
    demo = _load_pickle(path / "low_dim_obs.pkl")
    variation = _load_pickle(path / "variation_number.pkl")
    descriptions = _load_pickle(path / "variation_descriptions.pkl")
    random_state = tuple(demo.random_seed)
    if len(random_state) != 5 or random_state[0] != "MT19937":
        raise ValueError(f"{path} does not store a NumPy MT19937 random state")
    return TestEpisode(
        task=task,
        episode=episode,
        variation=int(variation),
        random_state=random_state,
        descriptions=tuple(str(item) for item in descriptions),
        recorded_task_state=demo[0].task_low_dim_state,
    )


def check_task_layout(root: Path, task: str) -> None:
    """PerAct picks episode ``i`` as the ``i``-th natural-sorted entry; require exactly episode0..24."""

    episodes = root / task / "all_variations" / "episodes"
    names = {entry.name for entry in episodes.iterdir() if ".DS_Store" not in entry.name}
    expected = {f"episode{index}" for index in range(EPISODES_PER_TASK)}
    if names != expected:
        raise ValueError(f"{episodes} must contain exactly episode0..episode{EPISODES_PER_TASK - 1}")


def task_digest(root: Path, task: str) -> str:
    check_task_layout(root, task)
    return test_episodes_sha256(
        (episode, load_test_episode(root, task, episode).state_sha256) for episode in range(EPISODES_PER_TASK)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="the PerAct test split root (<root>/<task>/all_variations)")
    parser.add_argument("--task", action="append", choices=TASK_NAMES)
    parser.add_argument("--print", action="store_true", help="print the digests without comparing")
    args = parser.parse_args()
    report = {}
    mismatched = []
    for name in args.task or TASK_NAMES:
        actual = task_digest(args.root, name)
        expected = TASKS_BY_NAME[name].test_episodes_sha256
        report[name] = {"episodes": EPISODES_PER_TASK, "actual": actual, "expected": expected}
        if actual != expected:
            mismatched.append(name)
    print(json.dumps(report, indent=2, sort_keys=True))
    if mismatched and not args.print:
        print(f"PerAct test-episode digests differ for: {', '.join(mismatched)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
