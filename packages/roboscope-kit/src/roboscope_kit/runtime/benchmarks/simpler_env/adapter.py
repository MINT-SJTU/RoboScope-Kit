# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""SimplerEnv work manifest, worker process, and official aggregation."""

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
    BRIDGE,
    PROTOCOL_VERSION,
    SUCCESS_CRITERION,
    SUITE_EPISODES,
    SUITE_TASKS,
    SUITES,
    Variant,
    episode_id,
)


@dataclass(frozen=True)
class SimplerEnvRuntimeConfig:
    python: Path
    simpler_env_root: Path
    suite: str
    # (variant id, episode indices) in work-item order
    selection: tuple[tuple[str, tuple[int, ...]], ...]
    cpu_threads: int = 1
    recycle_after_records: int | None = None

    def __post_init__(self) -> None:
        if self.suite not in SUITES:
            raise ValueError(f"unsupported SimplerEnv suite: {self.suite}")
        variants = {
            variant.variant_id: variant for task in SUITE_TASKS[self.suite] for variant in task.variants
        }
        seen: set[tuple[str, int]] = set()
        for variant_id, indices in self.selection:
            variant = variants.get(variant_id)
            if variant is None:
                raise ValueError(f"unknown SimplerEnv variant {variant_id!r} for suite {self.suite}")
            for index in indices:
                if not 0 <= index < variant.episodes or (variant_id, index) in seen:
                    raise ValueError(f"invalid or repeated SimplerEnv episode {variant_id}/{index}")
                seen.add((variant_id, index))
        if not seen:
            raise ValueError("SimplerEnv selection cannot be empty")
        if self.cpu_threads < 1:
            raise ValueError("SimplerEnv cpu_threads must be positive")

    @property
    def episodes(self) -> int:
        return sum(len(indices) for _variant, indices in self.selection)

    @property
    def formal(self) -> bool:
        """Whether the selection is the suite's complete matrix.

        A result is formal only when it also holds a record for every selected episode; a shard
        of a formal selection is partial until the shards are merged.
        """

        return self.episodes == SUITE_EPISODES[self.suite]


class SimplerEnvBenchmarkAdapter:
    """Run one frozen SimplerEnv suite against an SDK Policy, one episode per work item."""

    def __init__(self, config: SimplerEnvRuntimeConfig) -> None:
        self.config = config
        self.variants: dict[str, Variant] = {
            variant.variant_id: variant for task in SUITE_TASKS[config.suite] for variant in task.variants
        }

    def work_items(self) -> Sequence[WorkItem]:
        items: list[WorkItem] = []
        for variant_id, indices in self.config.selection:
            variant = self.variants[variant_id]
            for index in indices:
                items.append(
                    WorkItem(
                        ordinal=len(items),
                        item_id=episode_id(variant, index),
                        group_id=variant.task,
                        payload={"variant": variant_id, "indices": [index]},
                        expected_results=1,
                    )
                )
        return items

    def worker_process(self, context: WorkerContext) -> ProcessSpec:
        threads = str(self.config.cpu_threads)
        command = [
            str(self.config.python),
            "-m",
            "roboscope_kit.runtime.benchmarks.simpler_env.worker",
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
        ]
        if self.config.recycle_after_records is not None:
            command += ["--recycle-after-records", str(self.config.recycle_after_records)]
        return ProcessSpec(
            command=tuple(command),
            # Upstream scripts run from the SimplerEnv checkout; overlay paths are relative to it.
            cwd=self.config.simpler_env_root,
            env={
                "PYTHONPATH": ":".join((str(kit_import_root()), os.environ.get("PYTHONPATH", ""))),
                # SAPIEN renders through Vulkan on this GPU; the worker never runs CUDA kernels.
                "CUDA_VISIBLE_DEVICES": context.gpu_device,
                "OMP_NUM_THREADS": threads,
                "MKL_NUM_THREADS": threads,
                "OPENBLAS_NUM_THREADS": threads,
                "NUMEXPR_NUM_THREADS": threads,
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )

    def aggregate(self, results: Sequence[WorkResult]) -> dict[str, Any]:
        return self._aggregate(results, shard=False)

    def aggregate_partial(self, results: Sequence[WorkResult]) -> dict[str, Any]:
        """One shard's records: never formal, whatever the selection; the merge decides."""

        return self._aggregate(results, shard=True)

    def _aggregate(self, results: Sequence[WorkResult], *, shard: bool) -> dict[str, Any]:
        records = self._collect(results)
        recorded = sum(len(rows) for rows in records.values())
        complete = not shard and recorded == self.config.episodes
        variant_rows = []
        for task in SUITE_TASKS[self.config.suite]:
            for variant in task.variants:
                rows = records.get(variant.variant_id, [])
                if not rows and variant.variant_id not in dict(self.config.selection):
                    continue
                successes = sum(record.success for record in rows)
                variant_rows.append(
                    {
                        "variant": variant.variant_id,
                        "task": variant.task,
                        "subtask": variant.subtask,
                        "factor": variant.factor,
                        "episodes": len(rows),
                        "successes": successes,
                        "success_rate": successes / len(rows) if rows else 0.0,
                    }
                )
        official: dict[str, Any] = {
            "metric": "per_task_success_rate",
            "aggregation": "mean of variant success rates (calc_metrics_evaluation_videos.py)",
            "suite": self.config.suite,
            # Formal only for the complete matrix with every episode recorded in this result.
            "formal": self.config.formal and complete,
            "partial": not complete,
            "episodes": recorded,
            "selected_episodes": self.config.episodes,
            "tasks": self._task_rows(variant_rows),
            "subtasks": self._subtask_rows(variant_rows),
            "variants": variant_rows,
        }
        if self.config.suite == BRIDGE:
            official["partial_success"] = self._partial_success(records)
        return official

    def _task_rows(self, variant_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for task in SUITE_TASKS[self.config.suite]:
            members = [row for row in variant_rows if row["task"] == task.name]
            if not members:
                continue
            episodes = sum(row["episodes"] for row in members)
            successes = sum(row["successes"] for row in members)
            measured = [row["success_rate"] for row in members if row["episodes"]]
            rows.append(
                {
                    "task": task.name,
                    "episodes": episodes,
                    "successes": successes,
                    # The official number: every variant of a task has the same episode count,
                    # so this also equals successes / episodes on a formal run.
                    "success_rate": sum(measured) / len(measured) if measured else 0.0,
                    "variants": len(members),
                }
            )
        return rows

    def _subtask_rows(self, variant_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        keys = dict.fromkeys((row["task"], row["subtask"]) for row in variant_rows if row["subtask"])
        for task, subtask in keys:
            members = [row for row in variant_rows if row["task"] == task and row["subtask"] == subtask]
            measured = [row["success_rate"] for row in members if row["episodes"]]
            rows.append(
                {
                    "task": task,
                    "subtask": subtask,
                    "episodes": sum(row["episodes"] for row in members),
                    "successes": sum(row["successes"] for row in members),
                    "success_rate": sum(measured) / len(measured) if measured else 0.0,
                }
            )
        return rows

    @staticmethod
    def _partial_success(records: dict[str, list]) -> list[dict[str, Any]]:
        # calc_bridge_put_on_env_stats: "partial success" is is_src_obj_grasped in the
        # final episode statistics (the object was grasped at some step).
        rows = []
        for variant_id, members in records.items():
            grasped = sum(
                bool(record.metadata["episode_stats"].get("is_src_obj_grasped")) for record in members
            )
            rows.append(
                {
                    "task": members[0].metadata["task"],
                    "variant": variant_id,
                    "episodes": len(members),
                    "grasped": grasped,
                    "grasp_rate": grasped / len(members),
                }
            )
        return rows

    def _collect(self, results: Sequence[WorkResult]) -> dict[str, list]:
        expected = {
            episode_id(self.variants[variant_id], index): (variant_id, index)
            for variant_id, indices in self.config.selection
            for index in indices
        }
        per_variant: dict[str, list] = {}
        seen: set[str] = set()
        for result in results:
            for record in result.records:
                key = expected.get(record.record_id)
                if key is None:
                    raise RuntimeError(f"SimplerEnv record {record.record_id} is outside the work manifest")
                if record.record_id in seen:
                    raise RuntimeError(f"duplicate SimplerEnv record {record.record_id}")
                variant = self.variants[key[0]]
                metadata = record.metadata
                if metadata.get("variant") != variant.variant_id or metadata.get("index") != key[1]:
                    raise RuntimeError(f"SimplerEnv record {record.record_id} has inconsistent metadata")
                if result.group_id != variant.task or metadata.get("task") != variant.task:
                    raise RuntimeError(f"SimplerEnv record {record.record_id} belongs to another task")
                if not 1 <= record.steps <= variant.max_episode_steps:
                    raise RuntimeError(
                        f"SimplerEnv record {record.record_id} has invalid steps {record.steps}"
                    )
                terminated = bool(metadata.get("terminated_by_policy"))
                truncated = bool(metadata.get("truncated"))
                # The episode loop stops on a policy termination or on truncation at the limit
                # (both when the policy terminates on the last allowed step), never otherwise.
                if not (terminated or truncated) or (truncated and record.steps != variant.max_episode_steps):
                    raise RuntimeError(
                        f"SimplerEnv record {record.record_id} ended inconsistently "
                        f"(terminated={terminated}, truncated={truncated}, steps={record.steps})"
                    )
                seen.add(record.record_id)
                per_variant.setdefault(variant.variant_id, []).append(record)
        return per_variant

    def metadata(self) -> dict[str, Any]:
        return {
            "adapter": type(self).__name__,
            "benchmark": "simpler_env",
            "suite": self.config.suite,
            "protocol": PROTOCOL_VERSION,
            "metric": "per_task_success_rate",
            # The workload's selection, identical on every shard; whether a result is formal is
            # official_result.formal, which also requires every selected episode's record.
            "formal_selection": self.config.formal,
            "episodes": self.config.episodes,
            "suite_episodes": SUITE_EPISODES[self.config.suite],
            "tasks": [
                {"task": task.name, "variants": len(task.variants), "episodes": task.episodes}
                for task in SUITE_TASKS[self.config.suite]
            ],
            "action_representation": ACTION_REPRESENTATION,
            "success_criterion": SUCCESS_CRITERION,
            "simpler_env_root": str(self.config.simpler_env_root),
        }


__all__ = ["SimplerEnvBenchmarkAdapter", "SimplerEnvRuntimeConfig"]
