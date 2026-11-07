# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""RLBench PerAct 18-task work manifest and SDK-native worker process adapter."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboscope_kit.runtime.benchmark import ProcessSpec, WorkerContext, WorkItem, WorkResult
from roboscope_kit.runtime.benchmark.kit_path import kit_import_root

from .tasks import (
    ACTION_REPRESENTATION,
    CAMERAS,
    EPISODE_LENGTH,
    EPISODES_PER_TASK,
    IMAGE_SIZE,
    PLANNING_ERRORS,
    PROTOCOL_VERSION,
    SUCCESS_CRITERION,
    SUITES,
    TASK_NAMES,
    TASKS_BY_NAME,
    TEST_SET_SPLIT,
    test_episodes_sha256,
)

TERMINATIONS = ("success", "fail_condition", "timeout", *PLANNING_ERRORS)


@dataclass(frozen=True)
class RLBenchRuntimeConfig:
    python: Path
    coppeliasim: Path
    test_set: Path
    xvfb: Path
    suite: str
    tasks: tuple[str, ...]
    episodes_per_task: int
    chunk_size: int = 1
    cpu_threads: int = 1

    def __post_init__(self) -> None:
        if self.suite not in SUITES:
            raise ValueError(f"unsupported RLBench suite: {self.suite}")
        if not self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise ValueError("RLBench tasks must be unique and non-empty")
        unknown = sorted(set(self.tasks) - set(TASK_NAMES))
        if unknown:
            raise ValueError(f"unknown RLBench tasks: {', '.join(unknown)}")
        if not 1 <= self.episodes_per_task <= EPISODES_PER_TASK:
            raise ValueError(
                f"RLBench episodes per task must be in [1, {EPISODES_PER_TASK}] "
                "(episode i is PerAct test episode i)"
            )
        if self.chunk_size < 1 or self.cpu_threads < 1:
            raise ValueError("RLBench chunk_size and cpu_threads must be positive")

    @property
    def formal(self) -> bool:
        """Only all 18 tasks with all 25 test episodes are the full ``peract_18`` evaluation."""

        return self.tasks == TASK_NAMES and self.episodes_per_task == EPISODES_PER_TASK


class RLBenchBenchmarkAdapter:
    """Run PerAct's RLBench test-set protocol against an SDK Policy."""

    def __init__(self, config: RLBenchRuntimeConfig) -> None:
        self.config = config

    def work_items(self) -> Sequence[WorkItem]:
        items: list[WorkItem] = []
        episodes = self.config.episodes_per_task
        for start in range(0, episodes, self.config.chunk_size):
            chunk = list(range(start, min(start + self.config.chunk_size, episodes)))
            for task in self.config.tasks:
                items.append(
                    WorkItem(
                        ordinal=len(items),
                        item_id=f"{task}-episodes-{chunk[0]:02d}-{chunk[-1] + 1:02d}",
                        group_id=task,
                        payload={"task": task, "episodes": chunk},
                        expected_results=len(chunk),
                    )
                )
        return items

    def worker_process(self, context: WorkerContext) -> ProcessSpec:
        threads = str(self.config.cpu_threads)
        coppeliasim = str(self.config.coppeliasim)
        command = (
            str(self.config.python),
            "-m",
            "roboscope_kit.runtime.benchmarks.rlbench.worker",
            "--worker-id",
            str(context.worker_id),
            "--policy-socket",
            str(context.replica_socket),
            "--suite",
            self.config.suite,
            "--queue-dir",
            str(context.queue_dir),
            "--worker-state-dir",
            str(context.state_dir),
            "--test-set",
            str(self.config.test_set),
            "--xvfb",
            str(self.config.xvfb),
        )
        return ProcessSpec(
            command=command,
            env={
                "PYTHONPATH": ":".join((str(kit_import_root()), os.environ.get("PYTHONPATH", ""))),
                "COPPELIASIM_ROOT": coppeliasim,
                "LD_LIBRARY_PATH": ":".join((coppeliasim, os.environ.get("LD_LIBRARY_PATH", ""))),
                "QT_QPA_PLATFORM_PLUGIN_PATH": coppeliasim,
                # The simulator renders with Mesa's llvmpipe on the worker's own Xvfb; it never
                # touches CUDA.
                "CUDA_VISIBLE_DEVICES": "",
                "LP_NUM_THREADS": threads,
                "OMP_NUM_THREADS": threads,
                "MKL_NUM_THREADS": threads,
                "OPENBLAS_NUM_THREADS": threads,
                "NUMEXPR_NUM_THREADS": threads,
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )

    def aggregate(self, results: Sequence[WorkResult]) -> dict[str, Any]:
        per_task = self._collect(results)
        rows = [self._row(task, per_task.get(task, [])) for task in self.config.tasks]
        missing = [row["task"] for row in rows if row["episodes"] != self.config.episodes_per_task]
        if self.config.formal and missing:
            raise RuntimeError(f"RLBench formal result is incomplete: {', '.join(missing)}")
        official: dict[str, Any] = {
            "metric": "per_task_success_rate",
            "formal": self.config.formal,
            "tasks": rows,
            # PerAct reports each task's success rate and their unweighted mean over the tasks.
            "mean_success_rate": sum(row["success_rate"] for row in rows) / len(rows),
            # Not part of the official metric, and not merged across shards.
            "diagnostics": {task: self._diagnostics(per_task.get(task, [])) for task in self.config.tasks},
        }
        if self.config.formal:
            for task in self.config.tasks:
                actual = test_episodes_sha256(
                    (int(record.metadata["episode"]), str(record.metadata["episode_state_sha256"]))
                    for record in per_task[task]
                )
                expected = TASKS_BY_NAME[task].test_episodes_sha256
                if actual != expected:
                    raise RuntimeError(
                        f"RLBench task {task} replayed test episodes {actual}, expected frozen {expected}"
                    )
            official["test_episodes_verified"] = True
        return official

    def aggregate_partial(self, results: Sequence[WorkResult]) -> dict[str, Any]:
        """Keep episode identities for result-only merging; no shard is a formal result."""

        per_task = self._collect(results)
        return {
            "partial": True,
            "formal": False,
            "tasks": [self._row(task, per_task.get(task, [])) for task in self.config.tasks],
            "episode_states": [
                {
                    "task": task,
                    "episode": int(record.metadata["episode"]),
                    "state_sha256": str(record.metadata["episode_state_sha256"]),
                    "success": record.success,
                }
                for task in self.config.tasks
                for record in per_task.get(task, [])
            ],
        }

    @staticmethod
    def _row(task: str, records: list) -> dict[str, Any]:
        episodes = len(records)
        successes = sum(record.success for record in records)
        return {
            "task": task,
            "episodes": episodes,
            "successes": successes,
            "success_rate": successes / episodes if episodes else 0.0,
        }

    @staticmethod
    def _diagnostics(records: list) -> dict[str, Any]:
        terminations = dict.fromkeys(TERMINATIONS, 0)
        for record in records:
            terminations[str(record.metadata["termination"])] += 1
        return {
            "actions": sum(record.steps for record in records),
            "simulation_steps": sum(int(record.metadata["simulation_steps"]) for record in records),
            "terminations": terminations,
        }

    def _collect(self, results: Sequence[WorkResult]) -> dict[str, list]:
        per_task: dict[str, list] = {}
        seen: set[str] = set()
        expected_episodes = set(range(self.config.episodes_per_task))
        for result in results:
            task = result.group_id
            if task not in self.config.tasks:
                raise RuntimeError(f"unexpected RLBench task result {task!r}")
            for record in result.records:
                episode = int(record.metadata["episode"])
                if record.metadata.get("task") != task or episode not in expected_episodes:
                    raise RuntimeError(f"RLBench record {record.record_id} is outside the work manifest")
                if record.record_id != f"{task}-episode-{episode:02d}" or record.record_id in seen:
                    raise RuntimeError(f"duplicate or malformed RLBench record {record.record_id}")
                if not 1 <= record.steps <= EPISODE_LENGTH:
                    raise RuntimeError(f"RLBench record {record.record_id} has invalid steps {record.steps}")
                termination = record.metadata.get("termination")
                if termination not in TERMINATIONS or record.success != (termination == "success"):
                    raise RuntimeError(f"RLBench record {record.record_id} has inconsistent termination")
                if termination == "timeout" and record.steps != EPISODE_LENGTH:
                    raise RuntimeError(
                        f"RLBench record {record.record_id} timed out before {EPISODE_LENGTH} actions"
                    )
                seen.add(record.record_id)
                per_task.setdefault(task, []).append(record)
        return per_task

    def metadata(self) -> dict[str, Any]:
        return {
            "adapter": type(self).__name__,
            "benchmark": "rlbench",
            "suite": self.config.suite,
            "protocol": PROTOCOL_VERSION,
            "metric": "per_task_success_rate",
            "full_profile": self.config.formal,
            "task_order": list(self.config.tasks),
            "tasks": [
                {
                    "task": name,
                    "variations": TASKS_BY_NAME[name].variations,
                    "test_episodes_sha256": TASKS_BY_NAME[name].test_episodes_sha256,
                }
                for name in self.config.tasks
            ],
            "test_set": {"source": "PerAct pre-generated RLBench demonstrations", "split": TEST_SET_SPLIT},
            "episodes_per_task": self.config.episodes_per_task,
            "episodes": list(range(self.config.episodes_per_task)),
            "episode_length": EPISODE_LENGTH,
            "cameras": list(CAMERAS),
            "image_size": list(IMAGE_SIZE),
            "action_representation": ACTION_REPRESENTATION,
            "success_criterion": SUCCESS_CRITERION,
            "chunk_size": self.config.chunk_size,
        }


__all__ = ["RLBenchBenchmarkAdapter", "RLBenchRuntimeConfig"]
