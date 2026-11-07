# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Task-level scheduling for the fixed RoboCasa365 native environment groups."""

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
    ACTION_STEPS,
    CONSTRUCTOR_SEED,
    CONTROL_FREQ_HZ,
    EPISODES_PER_TASK,
    NATIVE_ENVS,
    OBSERVATION_STEPS,
    PROTOCOL_VERSION,
    ROBOCASA_REVISION,
    ROLLOUT_LABEL_BASE,
    SCENE_SPLIT,
    SUITES,
    TASKS_BY_NAME,
    plan_sha256,
    scenario_plan,
)


@dataclass(frozen=True)
class RoboCasa365RuntimeConfig:
    python: Path
    benchmark_root: Path
    suite: str
    tasks: tuple[str, ...]
    episodes_per_task: int
    native_envs: int = NATIVE_ENVS
    cpu_threads: int = 1
    save_videos: bool = False

    def __post_init__(self) -> None:
        if self.suite not in SUITES:
            raise ValueError(f"unsupported RoboCasa365 suite: {self.suite}")
        if not self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise ValueError("RoboCasa365 tasks must be unique and non-empty")
        if set(self.tasks) - set(SUITES[self.suite]):
            raise ValueError("RoboCasa365 tasks are outside the selected suite")
        if not 1 <= self.episodes_per_task <= EPISODES_PER_TASK:
            raise ValueError(f"RoboCasa365 episodes must be in [1, {EPISODES_PER_TASK}]")
        if self.native_envs not in (1, NATIVE_ENVS):
            raise ValueError("RoboCasa365 native_envs must be 5 (native) or 1 (smoke)")
        if self.episodes_per_task % self.native_envs:
            raise ValueError("RoboCasa365 episodes must be divisible by native_envs; no padded records")
        if self.cpu_threads < 1:
            raise ValueError("RoboCasa365 cpu_threads must be positive")

    @property
    def full_suite(self) -> bool:
        return (
            self.tasks == SUITES[self.suite]
            and self.episodes_per_task == EPISODES_PER_TASK
            and self.native_envs == NATIVE_ENVS
        )


class RoboCasa365BenchmarkAdapter:
    def __init__(self, config: RoboCasa365RuntimeConfig) -> None:
        self.config = config

    def work_items(self) -> Sequence[WorkItem]:
        # Each task owns its five constructor RNG streams and sequential resets.
        # Do not split the native group's termination barrier between workers.
        return tuple(
            WorkItem(
                ordinal=index, item_id=f"{self.config.suite}-{name}", group_id=name,
                payload={
                    "task": name, "episodes": self.config.episodes_per_task,
                    "native_envs": self.config.native_envs,
                },
                expected_results=self.config.episodes_per_task,
            )
            for index, name in enumerate(self.config.tasks)
        )

    def worker_process(self, context: WorkerContext) -> ProcessSpec:
        threads = str(self.config.cpu_threads)
        return ProcessSpec(
            command=(
                str(self.config.python), "-m", "roboscope_kit.runtime.benchmarks.robocasa365.worker",
                "--worker-id", str(context.worker_id),
                "--policy-socket", str(context.replica_socket),
                "--queue-dir", str(context.queue_dir),
                "--worker-state-dir", str(context.state_dir),
                "--output-dir", str(context.output_dir),
                *(("--save-videos",) if self.config.save_videos else ()),
            ),
            cwd=self.config.benchmark_root,
            env={
                "PYTHONPATH": ":".join((str(kit_import_root()), os.environ.get("PYTHONPATH", ""))),
                "CUDA_VISIBLE_DEVICES": context.gpu_device,
                "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
                "MUJOCO_EGL_DEVICE_ID": context.gpu_device,
                "OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads,
                "OPENBLAS_NUM_THREADS": threads, "NUMEXPR_NUM_THREADS": threads,
                "NUMBA_NUM_THREADS": threads, "TOKENIZERS_PARALLELISM": "false",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )

    def aggregate(self, results: Sequence[WorkResult]) -> dict[str, Any]:
        per_task: dict[str, list] = {name: [] for name in self.config.tasks}
        seen: set[str] = set()
        for result in results:
            if result.group_id not in per_task:
                raise RuntimeError(f"unexpected RoboCasa365 task {result.group_id!r}")
            task = TASKS_BY_NAME[result.group_id]
            for record in result.records:
                index = int(record.metadata["episode_index"])
                slot = index % self.config.native_envs
                expected_id = f"{task.name}-rollout-{ROLLOUT_LABEL_BASE + index}"
                if record.record_id != expected_id or record.record_id in seen:
                    raise RuntimeError(f"duplicate or malformed RoboCasa365 record {record.record_id}")
                if not 0 <= index < self.config.episodes_per_task:
                    raise RuntimeError(f"RoboCasa365 episode {index} is outside the manifest")
                if (
                    record.metadata.get("task") != task.name
                    or record.metadata.get("scene_split") != SCENE_SPLIT
                    or record.metadata.get("slot") != slot
                    or record.metadata.get("constructor_seed") != CONSTRUCTOR_SEED + slot
                    or record.metadata.get("stream_episode") != index // self.config.native_envs
                ):
                    raise RuntimeError(f"RoboCasa365 record {record.record_id} has invalid provenance")
                if not 1 <= record.steps <= task.horizon:
                    raise RuntimeError(f"RoboCasa365 record {record.record_id} has invalid step count")
                reason = record.metadata.get("termination")
                if reason not in {"horizon", "native_group_success", "simulator_done"}:
                    raise RuntimeError(f"RoboCasa365 record {record.record_id} has invalid termination")
                if reason == "horizon" and record.steps != task.horizon:
                    raise RuntimeError(f"RoboCasa365 record {record.record_id} stopped before its horizon")
                if reason == "native_group_success" and not record.success:
                    raise RuntimeError(f"RoboCasa365 record {record.record_id} lacks its group success")
                for key in (
                    "initial_state_sha256", "scenario_metadata_sha256", "initial_observation_sha256",
                ):
                    digest = str(record.metadata.get(key, ""))
                    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                        raise RuntimeError(f"RoboCasa365 record {record.record_id} lacks valid {key}")
                seen.add(record.record_id)
                per_task[task.name].append(record)
        rows = []
        for name in self.config.tasks:
            records = per_task[name]
            successes = sum(record.success for record in records)
            rows.append({
                "task": name, "split": TASKS_BY_NAME[name].split,
                "episodes": len(records), "successes": successes,
                "success_rate": successes / len(records) if records else 0.0,
            })
        expected = len(self.config.tasks) * self.config.episodes_per_task
        return {
            "metric": "episode_success_any_step", "full_suite": self.config.full_suite,
            "native_equivalence_verified": False,
            "coverage": {"expected": expected, "completed": len(seen), "complete": len(seen) == expected},
            "tasks": rows, "splits": split_results(rows),
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "adapter": type(self).__name__, "benchmark": "robocasa365", "suite": self.config.suite,
            "protocol": PROTOCOL_VERSION, "robocasa_revision": ROBOCASA_REVISION,
            "scene_split": SCENE_SPLIT, "full_suite": self.config.full_suite,
            "native_equivalence_verified": False,
            "native_envs_per_worker": self.config.native_envs,
            "task_order": list(self.config.tasks),
            "horizons": {name: TASKS_BY_NAME[name].horizon for name in self.config.tasks},
            "episodes_per_task": self.config.episodes_per_task,
            "action_steps": ACTION_STEPS, "observation_steps": OBSERVATION_STEPS,
            "action_representation": ACTION_REPRESENTATION, "control_freq_hz": CONTROL_FREQ_HZ,
            "video_logging": self.config.save_videos,
            "scenario_plan": scenario_plan(
                self.config.tasks, self.config.episodes_per_task, self.config.native_envs
            ),
            "scenario_plan_sha256": plan_sha256(
                self.config.tasks, self.config.episodes_per_task, self.config.native_envs
            ),
        }


def split_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    splits = []
    for split in ("atomic_seen", "composite_seen", "composite_unseen"):
        selected = [row for row in rows if row["split"] == split]
        if not selected:
            continue
        episodes = sum(row["episodes"] for row in selected)
        successes = sum(row["successes"] for row in selected)
        splits.append({
            "split": split, "tasks": len(selected), "episodes": episodes,
            "successes": successes,
            "success_rate": successes / episodes if episodes else 0.0,
            "mean_task_success_rate": sum(row["success_rate"] for row in selected) / len(selected),
        })
    return splits
