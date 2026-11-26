# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""BEHAVIOR Challenge 2026 work manifest and SDK-native worker process adapter."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboscope_kit.runtime.benchmark import ProcessSpec, WorkerContext, WorkItem, WorkResult
from roboscope_kit.runtime.benchmark.kit_path import kit_import_root

from .tasks import (
    ACTION_DIM,
    ACTION_REPRESENTATION,
    BEHAVIOR_1K_COMMIT,
    CAMERAS,
    CONTROL_HZ,
    EVAL_SEED,
    EVAL_TIMEOUT_MULTIPLIER,
    INSTANCES_PER_TASK,
    MAX_STEPS,
    PROTOCOL_VERSION,
    PYTHON_HASH_SEED,
    ROBOT,
    ROLLOUT_ID,
    SUITE_OBSERVATIONS,
    SUITES,
    TASK_IDS,
    TASK_NAMES,
    TASK_SCENES,
    TERMINATIONS,
    episode_id_for,
    instance_id,
)


@dataclass(frozen=True)
class BehaviorRuntimeConfig:
    python: Path
    data_path: Path
    suite: str
    tasks: tuple[str, ...]
    instances_per_task: int
    write_video: bool = False
    cpu_threads: int = 4

    def __post_init__(self) -> None:
        if self.suite not in SUITES:
            raise ValueError(f"unsupported BEHAVIOR suite: {self.suite}")
        if not self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise ValueError("BEHAVIOR tasks must be unique and non-empty")
        unknown = sorted(set(self.tasks) - set(TASK_NAMES))
        if unknown:
            raise ValueError(f"unknown BEHAVIOR tasks: {', '.join(unknown)}")
        if not 1 <= self.instances_per_task <= INSTANCES_PER_TASK:
            raise ValueError(
                f"BEHAVIOR instances per task must be in [1, {INSTANCES_PER_TASK}] "
                "(instance index i is public test instance 301 + i)"
            )
        if self.cpu_threads < 1:
            raise ValueError("BEHAVIOR cpu_threads must be positive")

    @property
    def formal(self) -> bool:
        """Only all 100 tasks with instance indices 0-9 are the suite's full evaluation."""

        return self.tasks == TASK_NAMES and self.instances_per_task == INSTANCES_PER_TASK


class BehaviorBenchmarkAdapter:
    """Run the BEHAVIOR Challenge 2026 public-test protocol against an SDK Policy."""

    def __init__(self, config: BehaviorRuntimeConfig) -> None:
        self.config = config

    def work_items(self) -> Sequence[WorkItem]:
        # One item per instance: each worker process loads one scene and evaluates one instance, as
        # the native command with one instance index does. One scene after another, so the run keeps
        # few scene texture caches (see the worker), the scene with the most steps first and, within a
        # scene, the longest tasks first, so the tail of a run is not one long episode.
        scene_steps: dict[str, int] = {}
        for task in self.config.tasks:
            scene_steps[TASK_SCENES[task]] = scene_steps.get(TASK_SCENES[task], 0) + MAX_STEPS[task]
        tasks = sorted(
            self.config.tasks,
            key=lambda task: (
                -scene_steps[TASK_SCENES[task]],
                TASK_SCENES[task],
                -MAX_STEPS[task],
                TASK_IDS[task],
            ),
        )
        items: list[WorkItem] = []
        for index in range(self.config.instances_per_task):
            for task in tasks:
                items.append(
                    WorkItem(
                        ordinal=len(items),
                        item_id=episode_id_for(task, index),
                        group_id=task,
                        payload={"task": task, "instance_index": index},
                        expected_results=1,
                    )
                )
        return items

    def worker_process(self, context: WorkerContext) -> ProcessSpec:
        threads = str(self.config.cpu_threads)
        command = (
            str(self.config.python),
            "-m",
            "roboscope_kit.runtime.benchmarks.behavior.worker",
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
            "--output-dir",
            str(context.output_dir / "behavior"),
            "--texture-cache-dir",
            str(context.output_dir / "texture-cache"),
        )
        if self.config.write_video:
            command += ("--write-video",)
        return ProcessSpec(
            command=command,
            env={
                "PYTHONPATH": ":".join((str(kit_import_root()), os.environ.get("PYTHONPATH", ""))),
                "OMNIGIBSON_DATA_PATH": str(self.config.data_path),
                # Building the runtime with --accept-licenses accepted the Isaac Sim EULA.
                "OMNI_KIT_ACCEPT_EULA": "YES",
                "CUDA_VISIBLE_DEVICES": context.gpu_device,
                "OMP_NUM_THREADS": threads,
                "MKL_NUM_THREADS": threads,
                "OPENBLAS_NUM_THREADS": threads,
                "PYTHONHASHSEED": str(PYTHON_HASH_SEED),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )

    def aggregate(self, results: Sequence[WorkResult]) -> dict[str, Any]:
        per_task = self._collect(results)
        rows = [self._row(task, per_task.get(task, [])) for task in self.config.tasks]
        missing = [row["task"] for row in rows if row["episodes"] != self.config.instances_per_task]
        if self.config.formal and missing:
            raise RuntimeError(f"BEHAVIOR formal result is incomplete: {', '.join(missing)}")
        return {
            # The challenge ranks by Q: each task's mean Q over its instances, averaged over the
            # tasks (score_utils.compute_final_q_score). A full success has Q = 1; success rate is
            # reported alongside and is not the ranking metric.
            "metric": "q_score",
            "formal": self.config.formal,
            "tasks": rows,
            "mean_q_score": sum(row["mean_q_score"] for row in rows) / len(rows),
            "mean_success_rate": sum(row["success_rate"] for row in rows) / len(rows),
            "diagnostics": {task: self._diagnostics(per_task.get(task, [])) for task in self.config.tasks},
        }

    def aggregate_partial(self, results: Sequence[WorkResult]) -> dict[str, Any]:
        """Keep episode identities for result-only merging; no shard is a formal result."""

        per_task = self._collect(results)
        return {
            "partial": True,
            "formal": False,
            "tasks": [self._row(task, per_task.get(task, [])) for task in self.config.tasks],
            "episodes": [
                {
                    "task": task,
                    "instance_id": int(record.metadata["instance_id"]),
                    "success": record.success,
                    "q_score": float(record.metadata["q_score"]),
                    "steps": record.steps,
                }
                for task in self.config.tasks
                for record in per_task.get(task, [])
            ],
        }

    @staticmethod
    def _row(task: str, records: list) -> dict[str, Any]:
        episodes = len(records)
        successes = sum(record.success for record in records)
        q_sum = sum(float(record.metadata["q_score"]) for record in records)
        return {
            "task": task,
            "task_id": TASK_IDS[task],
            "episodes": episodes,
            "successes": successes,
            "success_rate": successes / episodes if episodes else 0.0,
            # Sharded runs sum score_sum per task (shard_reduce) into mean_score.
            "score_sum": q_sum,
            "mean_q_score": q_sum / episodes if episodes else 0.0,
        }

    @staticmethod
    def _diagnostics(records: list) -> dict[str, Any]:
        terminations = dict.fromkeys(TERMINATIONS, 0)
        for record in records:
            terminations[str(record.metadata["termination"])] += 1
        return {
            "actions": sum(record.steps for record in records),
            "terminations": terminations,
            "scene_load_s": sum(float(record.metadata["scene_load_s"]) for record in records),
            "inference_wait_s": sum(float(record.metadata["inference_wait_s"]) for record in records),
            "observation_s": sum(float(record.metadata["observation_s"]) for record in records),
        }

    def _collect(self, results: Sequence[WorkResult]) -> dict[str, list]:
        per_task: dict[str, list] = {}
        env_wrapper = SUITE_OBSERVATIONS[self.config.suite].env_wrapper
        seen: set[str] = set()
        for result in results:
            if result.group_id not in self.config.tasks:
                raise RuntimeError(f"unexpected BEHAVIOR task result {result.group_id!r}")
            for record in result.records:
                task = record.metadata.get("task")
                index = int(record.metadata["instance_index"])
                if task != result.group_id or not 0 <= index < self.config.instances_per_task:
                    raise RuntimeError(f"BEHAVIOR record {record.record_id} is outside the work manifest")
                if (
                    record.record_id != episode_id_for(task, index)
                    or int(record.metadata["instance_id"]) != instance_id(index)
                    or record.record_id in seen
                ):
                    raise RuntimeError(f"duplicate or malformed BEHAVIOR record {record.record_id}")
                if (
                    record.metadata.get("protocol") != PROTOCOL_VERSION
                    or record.metadata.get("behavior_1k_commit") != BEHAVIOR_1K_COMMIT
                    or record.metadata.get("python_hash_seed") != PYTHON_HASH_SEED
                    or int(record.metadata.get("max_steps", -1)) != MAX_STEPS[task]
                    or record.metadata.get("suite") != self.config.suite
                    or record.metadata.get("env_wrapper") != env_wrapper
                ):
                    raise RuntimeError(
                        f"BEHAVIOR record {record.record_id} has an unexpected execution contract"
                    )
                # The timeout fires on the step after max_steps (see tasks.py), so an episode executes at
                # most max_steps + 1 actions.
                if not 1 <= record.steps <= MAX_STEPS[task] + 1:
                    raise RuntimeError(f"BEHAVIOR record {record.record_id} has invalid steps {record.steps}")
                termination = record.metadata.get("termination")
                if termination not in TERMINATIONS or record.success != (termination == "success"):
                    raise RuntimeError(f"BEHAVIOR record {record.record_id} has inconsistent termination")
                if termination == "max_steps" and record.steps != MAX_STEPS[task] + 1:
                    raise RuntimeError(f"BEHAVIOR record {record.record_id} hit max_steps early")
                q_score = float(record.metadata["q_score"])
                if (
                    not math.isfinite(q_score)
                    or not 0.0 <= q_score <= 1.0
                    or (record.success and q_score != 1.0)
                ):
                    raise RuntimeError(f"BEHAVIOR record {record.record_id} has invalid Q score {q_score}")
                seen.add(record.record_id)
                per_task.setdefault(task, []).append(record)
        return per_task

    def metadata(self) -> dict[str, Any]:
        observation = SUITE_OBSERVATIONS[self.config.suite]
        return {
            "adapter": type(self).__name__,
            "benchmark": "behavior",
            "suite": self.config.suite,
            "protocol": PROTOCOL_VERSION,
            "behavior_1k_commit": BEHAVIOR_1K_COMMIT,
            "python_hash_seed": PYTHON_HASH_SEED,
            "eval_seed": EVAL_SEED,
            "metric": "q_score",
            "full_profile": self.config.formal,
            "tasks": [
                {"task": task, "task_id": TASK_IDS[task], "max_steps": MAX_STEPS[task]}
                for task in self.config.tasks
            ],
            "instances_per_task": self.config.instances_per_task,
            "instance_ids": [instance_id(index) for index in range(self.config.instances_per_task)],
            "rollout_id": ROLLOUT_ID,
            "mode": "public_test",
            "robot": ROBOT,
            "env_wrapper": observation.env_wrapper,
            "observes_depth": observation.depth,
            "cameras": [
                {"role": role, "sensor": sensor, "resolution": list(observation.resolution[role])}
                for role, sensor in CAMERAS
            ],
            "control_hz": CONTROL_HZ,
            "eval_timeout_multiplier": EVAL_TIMEOUT_MULTIPLIER,
            "action_dim": ACTION_DIM,
            "action_representation": ACTION_REPRESENTATION,
            "write_video": self.config.write_video,
        }


def complete_merged_result(official: dict[str, Any], partials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Finish the per-task table ``shard_reduce`` summed from BEHAVIOR shard partials.

    Recomputes each task's mean Q from its summed ``score_sum`` and adds what ``aggregate`` reports for
    an unsharded run (metric, overall mean Q and success rate, formal), after checking that no episode
    appears in two shards and that the episode identities match the per-task counts.
    """

    episodes = [episode for partial in partials for episode in partial.get("episodes", [])]
    identities = [(episode["task"], int(episode["instance_id"])) for episode in episodes]
    if len(identities) != len(set(identities)):
        raise ValueError("BEHAVIOR shards report the same episode more than once")
    rows = official["tasks"]
    for row in rows:
        counted = sum(task == row["task"] for task, _ in identities)
        if counted != row["episodes"]:
            raise ValueError(
                f"BEHAVIOR shards list {counted} episodes of {row['task']}, not {row['episodes']}"
            )
        row["mean_q_score"] = row["score_sum"] / row["episodes"] if row["episodes"] else 0.0
    official.update(
        {
            "metric": "q_score",
            "formal": sorted(row["task"] for row in rows) == sorted(TASK_NAMES)
            and all(row["episodes"] == INSTANCES_PER_TASK for row in rows),
            "mean_q_score": sum(row["mean_q_score"] for row in rows) / len(rows),
            "mean_success_rate": sum(row["success_rate"] for row in rows) / len(rows),
            "episodes": sorted(episodes, key=lambda episode: (episode["task"], int(episode["instance_id"]))),
        }
    )
    return official


__all__ = ["BehaviorBenchmarkAdapter", "BehaviorRuntimeConfig", "complete_merged_result"]
