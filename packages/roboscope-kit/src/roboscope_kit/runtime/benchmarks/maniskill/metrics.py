# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Simulator-free arithmetic for ManiSkill's distinct scoring contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isclose, isfinite
from typing import Any

SCORE_METRICS = {"success_once", "success_at_end", "return"}


def _count(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"ManiSkill {field} must be a nonnegative integer")
    return value


def _number(value: Any, field: str) -> float:
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError(f"ManiSkill {field} must be finite")
    return value


def _rate(value: Any, expected: float | None, field: str) -> None:
    if expected is None:
        if value is not None:
            raise ValueError(f"ManiSkill {field} is undefined for this sample set")
    elif not isclose(_number(value, field), expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"ManiSkill {field} disagrees with its sample counts")


def validate_task_row(row: Mapping[str, Any]) -> None:
    """Validate raw counts before combining them; return can be negative."""
    if not isinstance(row.get("profile_id"), str) or not row["profile_id"]:
        raise ValueError("ManiSkill row has no profile identity")
    if not isinstance(row.get("task_id"), str) or not row["task_id"]:
        raise ValueError("ManiSkill row has no task identity")
    episodes = _count(row.get("episodes"), "episodes")
    metric = row.get("success_metric")
    if metric not in SCORE_METRICS:
        raise ValueError("ManiSkill row has an unknown success metric")
    count = _count(row.get("return_count"), "return_count")
    total = _number(row.get("return_sum"), "return_sum")
    if count != episodes or (not count and total != 0):
        raise ValueError("ManiSkill return coverage differs from its episode count")
    _rate(row.get("mean_return"), total / count if count else None, "mean_return")
    _count(row.get("actual_steps"), "actual_steps")
    if metric == "return":
        for field in ("successes", "failures", "success_rate", "success_once_successes",
                      "success_once_rate", "success_at_end_successes", "success_at_end_rate"):
            if row.get(field) is not None:
                raise ValueError("ManiSkill return-only rows cannot report success or failure")
        return
    successes = _count(row.get("successes"), "successes")
    failures = _count(row.get("failures"), "failures")
    if successes + failures != episodes:
        raise ValueError("ManiSkill successes and failures do not cover scored episodes")
    _rate(row.get("success_rate"), successes / episodes if episodes else None, "success_rate")
    for kind in ("success_once", "success_at_end"):
        value = row.get(kind + "_successes")
        if not episodes:
            if value is not None:
                raise ValueError("ManiSkill empty rows cannot report observed success counts")
        elif _count(value, kind) > episodes:
            raise ValueError("ManiSkill success count exceeds its sample count")
        _rate(row.get(kind + "_rate"), value / episodes if episodes else None, kind + "_rate")
    if episodes and successes != row[metric + "_successes"]:
        raise ValueError("ManiSkill selected score differs from its declared metric")


def overall_task_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Keep return-only samples out of every success denominator."""
    scored = [row for row in rows if row["success_metric"] != "return"]
    episodes = sum(row["episodes"] for row in rows)
    count = sum(row["episodes"] for row in scored)
    successes = sum(row["successes"] for row in scored)
    return {
        "episodes": episodes,
        "successes": successes if count else None,
        "success_evaluated_episodes": count,
        "return_evaluated_episodes": episodes - count,
        "success_rate": successes / count if count else None,
        "task_mean_success_rate": (
            sum(row["success_rate"] for row in scored) / len(scored)
            if scored and all(row["episodes"] for row in scored) else None
        ),
    }


def validate_task_results(partial: Mapping[str, Any]) -> None:
    rows = partial.get("tasks")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("ManiSkill result must retain its profile table")
    ids = [row.get("profile_id") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("ManiSkill result contains duplicate profiles")
    for row in rows:
        validate_task_row(row)
    expected = overall_task_results(rows)
    if partial.get("overall") != expected:
        raise ValueError("ManiSkill overall result disagrees with its per-profile samples")
    runtime = partial.get("runtime_metrics", {})
    if _count(runtime.get("actual_policy_steps"), "actual_policy_steps") != sum(
        row["actual_steps"] for row in rows
    ):
        raise ValueError("ManiSkill runtime and per-profile step counts disagree")
    _count(runtime.get("vector_groups"), "vector_groups")
    if _number(runtime.get("inference_wait_s"), "inference_wait_s") < 0:
        raise ValueError("ManiSkill inference wait must be nonnegative")


def merge_task_results(partials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge profile sufficient statistics without averaging shard means or rates."""
    if not partials:
        raise ValueError("no ManiSkill partial results to merge")
    for partial in partials:
        validate_task_results(partial)
    layout = [(row["profile_id"], row["task_id"], row["success_metric"])
              for row in partials[0]["tasks"]]
    for partial in partials:
        if [(r["profile_id"], r["task_id"], r["success_metric"]) for r in partial["tasks"]] != layout:
            raise ValueError("ManiSkill shards disagree on profiles, task IDs or score metrics")
        if partial.get("evaluation_kind") != partials[0].get("evaluation_kind"):
            raise ValueError("ManiSkill shards disagree on evaluation kind")
    rows = []
    for index, (_, _, metric) in enumerate(layout):
        parts = [partial["tasks"][index] for partial in partials]
        row = dict(parts[0])
        for field in ("episodes", "return_count", "return_sum", "actual_steps"):
            row[field] = sum(part[field] for part in parts)
        count = row["episodes"]
        row["mean_return"] = row["return_sum"] / row["return_count"] if count else None
        if metric != "return":
            row["successes"] = sum(part["successes"] for part in parts)
            row["failures"] = count - row["successes"]
            row["success_rate"] = row["successes"] / count if count else None
            for kind in ("success_once", "success_at_end"):
                value = sum(part[kind + "_successes"] or 0 for part in parts) if count else None
                row[kind + "_successes"] = value
                row[kind + "_rate"] = value / count if count else None
        rows.append(row)
    return {
        "formal": False,  # The caller must additionally validate the complete protocol coverage.
        "evaluation_kind": partials[0]["evaluation_kind"],
        "tasks": rows,
        "overall": overall_task_results(rows),
        "runtime_metrics": {
            field: sum(partial["runtime_metrics"][field] for partial in partials)
            for field in ("actual_policy_steps", "inference_wait_s", "vector_groups")
        },
    }
