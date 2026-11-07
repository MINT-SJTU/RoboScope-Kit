# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0
"""Queue adapter for configured ManiSkill tasks and GPU vector groups."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboscope_kit.runtime.benchmark import ProcessSpec, WorkerContext, WorkItem, WorkResult

from .profile import PROTOCOL_ID


@dataclass(frozen=True)
class ManiSkillRuntimeConfig:
    python: Path
    manifest_path: Path
    runtime_home: Path
    asset_dir: Path
    suite: str
    profiles: tuple[dict[str, Any], ...]
    envs_per_worker: int = 10
    policy_batch_size: int = 1
    cpu_threads: int = 1
    expected_profile_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.profiles or not self.suite:
            raise ValueError("ManiSkill needs a named suite with at least one task profile")
        if min(self.envs_per_worker, self.policy_batch_size, self.cpu_threads) < 1:
            raise ValueError("ManiSkill worker sizes must be positive")
        ids = [profile["profile_id"] for profile in self.profiles]
        if len(set(ids)) != len(ids) or any(not profile["seeds"] for profile in self.profiles):
            raise ValueError("ManiSkill needs distinct task profiles with nonempty episode seeds")


class ManiSkillBenchmarkAdapter:
    def __init__(self, config: ManiSkillRuntimeConfig) -> None:
        self.config = config

    def work_items(self) -> Sequence[WorkItem]:
        items = []
        for profile in self.config.profiles:
            profile_id = profile["profile_id"]
            for start in range(0, len(profile["seeds"]), self.config.envs_per_worker):
                seeds = profile["seeds"][start:start + self.config.envs_per_worker]
                items.append(WorkItem(
                    ordinal=len(items),
                    item_id=f"{profile_id}-episodes-{start:06d}-{start + len(seeds):06d}",
                    group_id=profile_id,
                    payload={"profile_id": profile_id, "episode_start": start, "seeds": list(seeds)},
                    expected_results=len(seeds),
                ))
        return items

    def worker_process(self, context: WorkerContext) -> ProcessSpec:
        threads = str(self.config.cpu_threads)
        return ProcessSpec(
            command=(
                str(self.config.python), "-m", "roboscope_kit.runtime.benchmarks.maniskill.worker",
                "--benchmark-manifest", str(self.config.manifest_path), "--suite", self.config.suite,
                "--worker-id", str(context.worker_id),
                "--policy-socket", str(context.replica_socket),
                "--queue-dir", str(context.queue_dir),
                "--worker-state-dir", str(context.state_dir),
                "--policy-batch-size", str(self.config.policy_batch_size),
            ),
            env={
                "CUDA_VISIBLE_DEVICES": context.gpu_device,
                "HOME": str(self.config.runtime_home),
                "MS_ASSET_DIR": str(self.config.asset_dir),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads,
                "OPENBLAS_NUM_THREADS": threads, "NUMEXPR_NUM_THREADS": threads,
            },
        )

    def aggregate(self, results: Sequence[WorkResult]) -> dict[str, Any]:
        by_profile: dict[str, list] = {profile["profile_id"]: [] for profile in self.config.profiles}
        for result in results:
            by_profile[result.group_id].extend(result.records)
        rows = []
        for profile in self.config.profiles:
            records = by_profile[profile["profile_id"]]
            count = len(records)
            successes = sum(record.success for record in records)
            once = sum(bool(record.metadata["success_once"]) for record in records)
            at_end = sum(bool(record.metadata["success_at_end"]) for record in records)
            rows.append({
                "profile_id": profile["profile_id"], "task_id": profile["task_id"],
                "episodes": count, "successes": successes, "failures": count - successes,
                "success_metric": profile["success_metric"],
                "success_rate": successes / count if count else 0.0,
                "success_once_successes": once,
                "success_once_rate": once / count if count else 0.0,
                "success_at_end_successes": at_end,
                "success_at_end_rate": at_end / count if count else 0.0,
                "actual_steps": sum(record.steps for record in records),
            })
        records = [record for result in results for record in result.records]
        count = len(records)
        return {
            "tasks": rows,
            "overall": {
                "episodes": count, "successes": sum(record.success for record in records),
                "success_rate": sum(record.success for record in records) / count if count else 0.0,
                "task_mean_success_rate": (
                    sum(row["success_rate"] for row in rows) / len(rows) if rows else 0.0
                ),
            },
            "runtime_metrics": {
                "actual_policy_steps": sum(record.steps for record in records),
                "inference_wait_s": sum(
                    float(result.metadata.get("inference_wait_s", 0.0)) for result in results
                ),
                "vector_groups": len(results),
            },
        }

    def metadata(self) -> dict[str, Any]:
        profiles = self.config.profiles
        ids = tuple(profile["profile_id"] for profile in profiles)
        counts = {profile["profile_id"]: len(profile["seeds"]) for profile in profiles}
        expected_ids = self.config.expected_profile_ids or ids
        complete = ids == expected_ids and all(
            len(profile["seeds"]) == profile["episodes_per_task"] for profile in profiles
        )
        uniform_counts = set(counts.values())
        return {
            "adapter": type(self).__name__, "benchmark": "maniskill", "suite": self.config.suite,
            "protocol": PROTOCOL_ID, "tasks": [profile["task_id"] for profile in profiles],
            "profiles": list(ids), "suite_profiles": list(expected_ids),
            "episodes_per_task": next(iter(uniform_counts)) if len(uniform_counts) == 1 else None,
            "episodes_by_profile": counts,
            "profile_episodes_per_task": {
                profile["profile_id"]: profile["episodes_per_task"] for profile in profiles
            },
            "complete_named_suite": complete,
            "episode_seeds": {profile["profile_id"]: list(profile["seeds"]) for profile in profiles},
            "expected_records": sum(counts.values()),
            "envs_per_worker": self.config.envs_per_worker,
            "policy_batch_size": self.config.policy_batch_size,
            "simulation_backend": "physx_cuda",
            "profile_config": [dict(profile) for profile in profiles],
            "scope": "Explicit task/robot/observation/controller suite; no universal ManiSkill score",
        }
