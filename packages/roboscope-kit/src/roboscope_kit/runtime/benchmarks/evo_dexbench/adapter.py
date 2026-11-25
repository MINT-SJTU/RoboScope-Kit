# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Wave-level scheduling for Evo-DexBench ``evaluation_v1``."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboscope_kit.runtime.benchmark import ProcessSpec, WorkerContext, WorkItem, WorkResult
from roboscope_kit.runtime.benchmark.kit_path import kit_import_root

from .protocol import (
    ACTION_REPRESENTATION,
    CONDITIONS,
    CONTROL_FREQ_HZ,
    EPISODES_PER_TASK,
    EVO_DEXBENCH_REVISION,
    NATIVE_ENVS,
    OBSERVATION_STEPS,
    PROTOCOL_VERSION,
    SIM_FREQ_HZ,
    SUITES,
    TASKS_BY_ID,
    episode_id,
    plan_sha256,
)


@dataclass(frozen=True)
class EvoDexBenchRuntimeConfig:
    python: Path
    benchmark_root: Path
    asset_root: Path
    suite: str
    tasks: tuple[str, ...]
    episodes_per_task: int
    cpu_threads: int = 1

    def __post_init__(self) -> None:
        if self.suite not in SUITES:
            raise ValueError(f"unsupported Evo-DexBench suite: {self.suite}")
        if not self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise ValueError("Evo-DexBench tasks must be unique and non-empty")
        if set(self.tasks) - set(SUITES[self.suite]):
            raise ValueError("Evo-DexBench tasks are outside the selected suite")
        if not 1 <= self.episodes_per_task <= EPISODES_PER_TASK:
            raise ValueError(f"Evo-DexBench episodes must be in [1, {EPISODES_PER_TASK}]")
        if self.episodes_per_task % NATIVE_ENVS:
            raise ValueError(f"Evo-DexBench episodes must be a multiple of {NATIVE_ENVS}; no padded waves")
        if self.cpu_threads < 1:
            raise ValueError("Evo-DexBench cpu_threads must be positive")

    @property
    def full_suite(self) -> bool:
        return self.tasks == SUITES[self.suite] and self.episodes_per_task == EPISODES_PER_TASK

    @property
    def standard_protocol(self) -> bool:
        return self.episodes_per_task == EPISODES_PER_TASK


class EvoDexBenchBenchmarkAdapter:
    def __init__(self, config: EvoDexBenchRuntimeConfig) -> None:
        self.config = config

    def work_items(self) -> Sequence[WorkItem]:
        # One item is one upstream wave: NATIVE_ENVS consecutive episodes in the upstream
        # execution order, stepped together and batched together for the Policy.
        waves = self.config.episodes_per_task // NATIVE_ENVS
        return tuple(
            WorkItem(
                ordinal=index * waves + wave, item_id=f"{task}-wave-{wave:02d}", group_id=task,
                payload={"task": task, "episodes": self.config.episodes_per_task, "wave": wave},
                expected_results=NATIVE_ENVS,
            )
            for index, task in enumerate(self.config.tasks)
            for wave in range(waves)
        )

    def worker_process(self, context: WorkerContext) -> ProcessSpec:
        threads = str(self.config.cpu_threads)
        return ProcessSpec(
            command=(
                str(self.config.python), "-m", "roboscope_kit.runtime.benchmarks.evo_dexbench.worker",
                "--worker-id", str(context.worker_id),
                "--policy-socket", str(context.replica_socket),
                "--queue-dir", str(context.queue_dir),
                "--worker-state-dir", str(context.state_dir),
                "--output-dir", str(context.output_dir),
            ),
            cwd=self.config.benchmark_root,
            env={
                "PYTHONPATH": os.pathsep.join(
                    filter(None, (str(kit_import_root()), os.environ.get("PYTHONPATH"))),
                ),
                "CUDA_VISIBLE_DEVICES": context.gpu_device,
                **asset_environment(self.config.asset_root),
                "OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads,
                "OPENBLAS_NUM_THREADS": threads, "NUMEXPR_NUM_THREADS": threads,
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )

    def aggregate(self, results: Sequence[WorkResult]) -> dict[str, Any]:
        per_task: dict[str, list] = {task: [] for task in self.config.tasks}
        seen: set[str] = set()
        for result in results:
            if result.group_id not in per_task:
                raise RuntimeError(f"unexpected Evo-DexBench task {result.group_id!r}")
            task = TASKS_BY_ID[result.group_id]
            for record in result.records:
                index = int(record.metadata["episode_id"])
                if record.record_id != episode_id(task.task_id, index) or record.record_id in seen:
                    raise RuntimeError(f"duplicate or malformed Evo-DexBench record {record.record_id}")
                if not 0 <= index < self.config.episodes_per_task:
                    raise RuntimeError(f"Evo-DexBench episode {index} is outside the schedule")
                if record.metadata.get("condition") not in CONDITIONS:
                    raise RuntimeError(f"Evo-DexBench record {record.record_id} has an unknown condition")
                if not 1 <= record.steps <= task.horizon:
                    raise RuntimeError(f"Evo-DexBench record {record.record_id} has invalid step count")
                score = record.metadata.get("stage_score")
                if not isinstance(score, (int, float)) or not 0.0 <= score <= 1.0:
                    raise RuntimeError(f"Evo-DexBench record {record.record_id} has invalid stage score")
                if record.success and score != 1.0:
                    raise RuntimeError(f"Evo-DexBench record {record.record_id} succeeded below stage 1.0")
                seen.add(record.record_id)
                per_task[task.task_id].append(record)
        rows = []
        for task_id in self.config.tasks:
            records = per_task[task_id]
            rows.append({
                "task": task_id, "name": TASKS_BY_ID[task_id].name, "split": TASKS_BY_ID[task_id].split,
                **summarize(records),
                "by_condition": {
                    condition: summarize([r for r in records if r.metadata["condition"] == condition])
                    for condition in CONDITIONS
                },
            })
        expected = len(self.config.tasks) * self.config.episodes_per_task
        return official_result(
            rows, expected, full_suite=self.config.full_suite,
            standard_protocol=self.config.standard_protocol,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "adapter": type(self).__name__, "benchmark": "evo_dexbench", "suite": self.config.suite,
            "protocol": PROTOCOL_VERSION, "evo_dexbench_revision": EVO_DEXBENCH_REVISION,
            "full_suite": self.config.full_suite, "standard_protocol": self.config.standard_protocol,
            "native_equivalence_verified": False, "native_envs_per_worker": NATIVE_ENVS,
            "task_order": list(self.config.tasks),
            "horizons": {task: TASKS_BY_ID[task].horizon for task in self.config.tasks},
            "episodes_per_task": self.config.episodes_per_task,
            "observation_steps": OBSERVATION_STEPS, "action_representation": ACTION_REPRESENTATION,
            "control_freq_hz": CONTROL_FREQ_HZ, "sim_freq_hz": SIM_FREQ_HZ,
            "plan_sha256": plan_sha256(self.config.tasks, self.config.episodes_per_task),
        }


def asset_environment(root: Path) -> dict[str, str]:
    """The variables upstream ``activate_evodex_assets.sh`` exports for one asset root."""
    return {
        "EVODEX_ASSET_DIR": str(root), "MS_ASSET_DIR": str(root / "maniskill"),
        "PARTNET_MOBILITY_DATASET": str(root / "partnet_mobility"),
        "PARTNET_MOBILITY_MODIFIED_DATASET": str(root / "partnet_mobility_modified"),
        "YCB_DATASET": str(root / "ycb"),
    }


def summarize(records: Sequence[Any]) -> dict[str, Any]:
    return _rates(
        len(records), sum(record.success for record in records),
        sum(float(record.metadata["stage_score"]) for record in records),
    )


def _rates(episodes: int, successes: int, score_sum: float) -> dict[str, Any]:
    return {
        "episodes": episodes, "successes": successes, "stage_score_sum": score_sum,
        "success_rate": successes / episodes if episodes else 0.0,
        "stage_score": score_sum / episodes if episodes else 0.0,
    }


def official_result(
    rows: list[dict[str, Any]], expected: int, *, full_suite: bool, standard_protocol: bool,
) -> dict[str, Any]:
    completed = sum(row["episodes"] for row in rows)
    return {
        "metric": "success_rate_and_stage_score", "full_suite": full_suite,
        "standard_protocol": standard_protocol, "native_equivalence_verified": False,
        "coverage": {"expected": expected, "completed": completed, "complete": completed == expected},
        "overall": _rates(
            completed, sum(row["successes"] for row in rows),
            sum(row["stage_score_sum"] for row in rows),
        ),
        "mean_task_success_rate": sum(row["success_rate"] for row in rows) / len(rows),
        "mean_task_stage_score": sum(row["stage_score"] for row in rows) / len(rows),
        "tasks": rows,
    }


def merge_shard_results(partials: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Sum the per-task and per-condition counts of shards that ran parts of one workload."""
    first = partials[0]
    layout = [row["task"] for row in first["tasks"]]
    for partial in partials:
        if [row["task"] for row in partial["tasks"]] != layout:
            raise ValueError("Evo-DexBench shards list different task sets")
        for key in ("full_suite", "standard_protocol"):
            if partial[key] != first[key]:
                raise ValueError(f"Evo-DexBench shards disagree on {key}")
        if partial["coverage"]["expected"] != first["coverage"]["expected"]:
            raise ValueError("Evo-DexBench shards disagree on the expected workload")

    def merged(cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
        episodes = sum(cell["episodes"] for cell in cells)
        successes = sum(cell["successes"] for cell in cells)
        score_sum = sum(cell["stage_score_sum"] for cell in cells)
        if not 0 <= successes <= episodes or not 0.0 <= score_sum <= episodes:
            raise ValueError("Evo-DexBench shard counts are inconsistent")
        return _rates(episodes, successes, score_sum)

    per_task, remainder = divmod(first["coverage"]["expected"], len(layout))
    per_condition = per_task // len(CONDITIONS)
    if remainder or per_condition * len(CONDITIONS) != per_task:
        raise ValueError("Evo-DexBench expected workload is not a whole schedule per task")
    rows = []
    for index, task_id in enumerate(layout):
        cells = [partial["tasks"][index] for partial in partials]
        rows.append({
            "task": task_id, "name": cells[0]["name"], "split": cells[0]["split"], **merged(cells),
            "by_condition": {
                condition: merged([cell["by_condition"][condition] for cell in cells])
                for condition in CONDITIONS
            },
        })
    # Overlapping shards would push some task or condition past its schedule.
    if any(
        row["episodes"] > per_task
        or any(cell["episodes"] > per_condition for cell in row["by_condition"].values())
        for row in rows
    ):
        raise ValueError("Evo-DexBench shards overlap: a task or condition exceeds its schedule")
    return official_result(
        rows, first["coverage"]["expected"], full_suite=first["full_suite"],
        standard_protocol=first["standard_protocol"],
    )
