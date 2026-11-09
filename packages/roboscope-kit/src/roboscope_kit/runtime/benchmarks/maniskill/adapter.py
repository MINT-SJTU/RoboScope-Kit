# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0
"""Queue adapter for configured ManiSkill tasks and GPU vector groups."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
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
        return self._aggregate(results, require_complete=True)

    def aggregate_partial(self, results: Sequence[WorkResult]) -> dict[str, Any]:
        return self._aggregate(results, require_complete=False)

    def _aggregate(
        self, results: Sequence[WorkResult], *, require_complete: bool,
    ) -> dict[str, Any]:
        by_profile: dict[str, list] = {profile["profile_id"]: [] for profile in self.config.profiles}
        profiles = {profile["profile_id"]: profile for profile in self.config.profiles}
        expected = {item.item_id: item for item in self.work_items()}
        seen: set[str] = set()
        for result in results:
            item = expected.get(result.item_id)
            if item is None or result.item_id in seen or result.group_id != item.group_id:
                raise RuntimeError(f"Unexpected or duplicate ManiSkill work item {result.item_id!r}")
            seen.add(result.item_id)
            profile = profiles[result.group_id]
            steps = profile.get("diagnostic_step_limit", profile["max_episode_steps"])
            diagnostic = "diagnostic_step_limit" in profile
            if len(result.records) != item.expected_results:
                raise RuntimeError(f"Incomplete ManiSkill vector group {result.item_id!r}")
            for lane, (record, seed) in enumerate(zip(
                result.records, item.payload["seeds"], strict=True,
            )):
                index = item.payload["episode_start"] + lane
                record_id = f"{result.group_id}-episode-{index:06d}-seed-{seed}"
                metadata = record.metadata
                if record.record_id != record_id or any(
                    metadata.get(key) != value for key, value in {
                        "task_id": profile["task_id"], "profile_id": result.group_id,
                        "suite": self.config.suite, "protocol": PROTOCOL_ID,
                        "episode_index": index, "seed": seed, "vector_lane": lane,
                        "success_metric": profile["success_metric"],
                        "terminated": False, "truncated": steps == profile["max_episode_steps"],
                        **({"diagnostic_step_limit": steps,
                            "native_max_episode_steps": profile["max_episode_steps"]}
                           if diagnostic else {}),
                    }.items()
                ):
                    raise RuntimeError(f"ManiSkill record {record.record_id!r} differs from its manifest")
                if record.steps != steps:
                    raise RuntimeError(f"Invalid ManiSkill steps or score in record {record.record_id!r}")
                if profile["success_metric"] == "return":
                    if record.success or not isfinite(metadata.get("episode_return", float("nan"))):
                        raise RuntimeError(f"Invalid ManiSkill return in record {record.record_id!r}")
                elif any(
                    type(metadata.get(key)) is not bool
                    for key in ("success_once", "success_at_end")
                ) or record.success != metadata[profile["success_metric"]]:
                    raise RuntimeError(f"Invalid ManiSkill success in record {record.record_id!r}")
            by_profile[result.group_id].extend(result.records)
        if require_complete and seen != set(expected):
            raise RuntimeError(f"Missing {len(set(expected) - seen)} ManiSkill work items")
        rows = []
        for profile in self.config.profiles:
            records = by_profile[profile["profile_id"]]
            count = len(records)
            scores_success = profile["success_metric"] != "return"
            successes = sum(record.success for record in records) if scores_success else None
            has_success = bool(records) and all(
                type(record.metadata.get("success_once")) is bool for record in records
            )
            once = sum(record.metadata["success_once"] for record in records) if has_success else None
            at_end = sum(record.metadata["success_at_end"] for record in records) if has_success else None
            returns = [record.metadata.get("episode_return") for record in records]
            rows.append({
                "profile_id": profile["profile_id"], "task_id": profile["task_id"],
                "episodes": count, "successes": successes,
                "failures": count - successes if scores_success else None,
                "success_metric": profile["success_metric"],
                "success_rate": successes / count if scores_success and count else None,
                "success_once_successes": once,
                "success_once_rate": once / count if has_success else None,
                "success_at_end_successes": at_end,
                "success_at_end_rate": at_end / count if has_success else None,
                "mean_return": sum(returns) / count if count and None not in returns else None,
                "actual_steps": sum(record.steps for record in records),
            })
        records = [record for result in results for record in result.records]
        count = len(records)
        success_rows = [row for row in rows if row["success_metric"] != "return"]
        success_count = sum(row["episodes"] for row in success_rows)
        successes = sum(row["successes"] for row in success_rows)
        return {
            "formal": require_complete and self.metadata()["complete_named_suite"],
            "evaluation_kind": self.metadata()["evaluation_kind"],
            "tasks": rows,
            "overall": {
                "episodes": count, "successes": successes if success_count else None,
                "success_evaluated_episodes": success_count,
                "success_rate": successes / success_count if success_count else None,
                "task_mean_success_rate": (
                    sum(row["success_rate"] for row in success_rows) / len(success_rows)
                    if success_rows and all(row["episodes"] for row in success_rows) else None
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
        ) and not any("diagnostic_step_limit" in profile for profile in profiles)
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
            "evaluation_kind": (
                "integration_smoke" if any("diagnostic_step_limit" in profile for profile in profiles)
                else "full_horizon"
            ),
            "episode_seeds": {profile["profile_id"]: list(profile["seeds"]) for profile in profiles},
            "expected_records": sum(counts.values()),
            "envs_per_worker": self.config.envs_per_worker,
            "policy_batch_size": self.config.policy_batch_size,
            "simulation_backend": "physx_cuda",
            "profile_config": [dict(profile) for profile in profiles],
            "scope": "Explicit task/robot/observation/controller suite; no universal ManiSkill score",
        }
