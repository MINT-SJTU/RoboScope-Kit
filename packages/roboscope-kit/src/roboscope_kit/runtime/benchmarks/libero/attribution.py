# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Attribute LIBERO episode failures from privileged-state traces.

Every failed episode gets one cause, taken from the goal predicate that is still false at the end and whose
object the instruction names first. The cause comes from simulator state only (predicates, gripper contacts
and grasps, object poses), so it needs no human or VLM labels:

- ``no_contact``: the gripper never touched the goal object.
- ``wrong_object``: the gripper never touched the goal object but touched another movable object.
- ``grasp_failed``: the gripper touched the goal object but never grasped it.
- ``holding_at_timeout``: the goal object was still grasped when the step budget ran out.
- ``dropped_before_target``: the object was released more than ``--near-target-m`` from its target.
- ``placement_failed``: the object was released near its target but the predicate stayed false.
- ``articulation_incomplete``: the gripper touched the drawer, door, or knob but did not finish the motion.
- ``undone``: the predicate held at some step and was false again at the end.

When the run used ``repeats`` > 1, attempts of the same episode are compared step by step against the
first attempt. Each pair is attributed to its earliest difference: the initial observation
(``initial_state``), a different action for the same observation (``policy``), or a different state after the
same actions (``simulator``). Both the first bitwise difference and the first material one (beyond a
tolerance) are reported.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

ARTICULATION_PREDICATES = frozenset({"open", "close", "turnon", "turnoff"})
PLACEMENT_PREDICATES = frozenset({"on", "in"})
FAILURE_CAUSES = (
    "no_contact",
    "wrong_object",
    "grasp_failed",
    "holding_at_timeout",
    "dropped_before_target",
    "placement_failed",
    "articulation_incomplete",
    "undone",
)


def load_traces(trace_dir: Path) -> list[dict[str, Any]]:
    traces = []
    for path in sorted(trace_dir.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            traces.append(json.load(handle))
    return traces


def _first(steps: Sequence[dict[str, Any]], predicate) -> int | None:
    for row in steps:
        if predicate(row):
            return int(row["step"])
    return None


def _last(steps: Sequence[dict[str, Any]], predicate) -> int | None:
    for row in reversed(steps):
        if predicate(row):
            return int(row["step"])
    return None


def _xy_distance(first: Sequence[float] | None, second: Sequence[float] | None) -> float | None:
    if not first or not second:
        return None
    return math.dist(first[:2], second[:2])


def attribute_goal(trace: dict[str, Any], index: int, near_target_m: float) -> dict[str, Any]:
    """Attribute why goal predicate ``index`` is false at the end of a trace."""
    steps = trace["steps"]
    state = trace["goal_state"]
    predicate = str(state[index][0]).lower()
    target_object = trace["goal_objects"][index]
    goal_objects = set(trace["goal_objects"])
    result: dict[str, Any] = {
        "goal_index": index,
        "predicate": state[index],
        "object": target_object,
    }

    achieved = _last(steps, lambda row: row["predicates"][index])
    if achieved is not None:
        result.update(cause="undone", decisive_step=achieved + 1)
        return result

    first_contact = _first(steps, lambda row: target_object in row["contact"])
    if first_contact is None:
        others = [
            name
            for row in steps
            for name in row["contact"]
            if name not in goal_objects and name in trace["movable_objects"]
        ]
        if others:
            wrong = Counter(others).most_common(1)[0][0]
            step = _first(steps, lambda row: wrong in row["contact"])
            result.update(cause="wrong_object", decisive_step=step, touched=wrong)
        else:
            result.update(cause="no_contact", decisive_step=None)
        return result

    if predicate in ARTICULATION_PREDICATES:
        result.update(cause="articulation_incomplete", decisive_step=first_contact)
        return result

    last_grasp = _last(steps, lambda row: target_object in row["grasp"])
    if last_grasp is None:
        result.update(cause="grasp_failed", decisive_step=first_contact)
        return result
    if target_object in steps[-1]["grasp"]:
        result.update(cause="holding_at_timeout", decisive_step=int(steps[-1]["step"]))
        return result

    release_step = last_grasp + 1
    release_row = next((row for row in steps if int(row["step"]) == release_step), steps[-1])
    destination = state[index][2] if len(state[index]) > 2 else None
    distance = None
    if predicate in PLACEMENT_PREDICATES and destination is not None:
        distance = _xy_distance(
            release_row["positions"].get(target_object),
            trace["goal_positions_final"].get(destination),
        )
    result["release_distance_m"] = None if distance is None else round(distance, 4)
    if distance is not None and distance > near_target_m:
        result.update(cause="dropped_before_target", decisive_step=release_step)
    else:
        result.update(cause="placement_failed", decisive_step=release_step)
    return result


def instruction_order(trace: dict[str, Any], index: int) -> int:
    """Rank a goal by where its object is first named in the instruction.

    BDDL lists goals in no fixed order ("close" can precede the "in" it depends on), while instructions
    name the objects in the order the robot handles them.
    """
    words = str(trace.get("task_description", "")).lower().replace(",", " ").split()
    tokens = [token for token in str(trace["goal_objects"][index]).lower().split("_") if len(token) > 2]
    if not tokens:
        return len(words) + index
    width = len(tokens) + 2
    best_count, best_position = 0, len(words) + index
    for position in range(len(words)):
        count = len(set(tokens) & set(words[position : position + width]))
        if count > best_count:
            best_count, best_position = count, position
    return best_position


def attribute_episode(trace: dict[str, Any], near_target_m: float = 0.10) -> dict[str, Any]:
    steps = trace["steps"]
    row = {
        "task_id": trace["task_id"],
        "episode_index": trace["episode_index"],
        "attempt": trace.get("attempt", 0),
        "success": bool(trace["success"]),
        "steps": int(trace["steps_taken"]),
    }
    if trace["success"] or not steps:
        return row
    final = steps[-1]["predicates"]
    unmet = sorted(
        (index for index, value in enumerate(final) if not value),
        key=lambda index: (instruction_order(trace, index), index),
    )
    goals = [attribute_goal(trace, index, near_target_m) for index in unmet]
    row.update(
        cause=goals[0]["cause"],
        goals_met=len(final) - len(unmet),
        goals_total=len(final),
        goals=goals,
    )
    return row


def _max_difference(left: Sequence[float], right: Sequence[float]) -> float:
    return max((abs(a - b) for a, b in zip(left, right, strict=False)), default=0.0)


def _state_difference(left: dict[str, Any], right: dict[str, Any]) -> float:
    difference = _max_difference(left["eef_pos"], right["eef_pos"])
    for name, position in left["positions"].items():
        if name in right["positions"]:
            difference = max(difference, _max_difference(position, right["positions"][name]))
    return difference


def divergence(
    first: dict[str, Any],
    second: dict[str, Any],
    action_tolerance: float = 1e-3,
    position_tolerance_m: float = 1e-3,
) -> dict[str, Any]:
    """Locate where two attempts of one episode first differ, bitwise and materially.

    ``bitwise`` is the first step whose action or observation bytes differ. ``material`` is the first step
    whose action differs by more than ``action_tolerance`` or whose end-effector or object positions differ
    by more than ``position_tolerance_m``; tiny floating-point differences often grow into it.
    """
    result: dict[str, Any] = {"bitwise": None, "material": None}
    if first["first_observation_digest"] != second["first_observation_digest"]:
        result["bitwise"] = {"origin": "initial_state", "step": 0}
    for left, right in zip(first["steps"], second["steps"], strict=False):
        step = int(left["step"])
        if result["bitwise"] is None:
            if left["action_digest"] != right["action_digest"]:
                result["bitwise"] = {"origin": "policy", "step": step}
            elif left["observation_digest"] != right["observation_digest"]:
                result["bitwise"] = {"origin": "simulator", "step": step}
        if result["material"] is None:
            if _max_difference(left["action"], right["action"]) > action_tolerance:
                result["material"] = {"origin": "policy", "step": step}
            elif _state_difference(left, right) > position_tolerance_m:
                result["material"] = {"origin": "simulator", "step": step}
        if result["bitwise"] is not None and result["material"] is not None:
            break
    if result["material"] is None and len(first["steps"]) != len(second["steps"]):
        result["material"] = {
            "origin": "length",
            "step": min(len(first["steps"]), len(second["steps"])),
        }
    return result


def stability(traces: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    episodes: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        episodes[(int(trace["task_id"]), int(trace["episode_index"]))].append(trace)
    rows = []
    for (task_id, episode_index), attempts in sorted(episodes.items()):
        attempts.sort(key=lambda trace: int(trace.get("attempt", 0)))
        if len(attempts) < 2:
            continue
        outcomes = [bool(trace["success"]) for trace in attempts]
        if all(outcomes):
            verdict = "stable_success"
        elif not any(outcomes):
            verdict = "stable_failure"
        else:
            verdict = "flaky"
        rows.append(
            {
                "task_id": task_id,
                "episode_index": episode_index,
                "outcomes": outcomes,
                "verdict": verdict,
                "divergence": [divergence(attempts[0], other) for other in attempts[1:]],
            }
        )
    return rows


def summarize(traces: Sequence[dict[str, Any]], near_target_m: float = 0.10) -> dict[str, Any]:
    episodes = [attribute_episode(trace, near_target_m) for trace in traces]
    first_attempts = [row for row in episodes if row["attempt"] == 0]
    failures = [row for row in first_attempts if not row["success"]]
    causes = Counter(row["cause"] for row in failures)
    partial = sum(1 for row in failures if row.get("goals_met", 0) > 0)
    per_task: dict[int, Counter] = defaultdict(Counter)
    for row in failures:
        per_task[int(row["task_id"])][row["cause"]] += 1
    stable = stability(traces)
    verdicts = Counter(row["verdict"] for row in stable)
    origins = Counter(
        (entry["material"] or {"origin": "identical"})["origin"]
        for row in stable
        for entry in row["divergence"]
    )
    bitwise = Counter(
        (entry["bitwise"] or {"origin": "identical"})["origin"]
        for row in stable
        for entry in row["divergence"]
    )
    return {
        "episodes": len(first_attempts),
        "successes": len(first_attempts) - len(failures),
        "failures": len(failures),
        "failure_causes": {cause: causes.get(cause, 0) for cause in FAILURE_CAUSES},
        "partial_failures": partial,
        "failure_causes_by_task": {str(task): dict(counter) for task, counter in sorted(per_task.items())},
        "stability": {
            "episodes": len(stable),
            "verdicts": dict(verdicts),
            "divergence_origins": dict(origins),
            "bitwise_divergence_origins": dict(bitwise),
        },
        "episode_attributions": episodes,
        "episode_stability": stable,
    }


def render_markdown(summary: dict[str, Any], task_names: dict[int, str] | None = None) -> str:
    lines = [
        f"Episodes: {summary['episodes']}, successes: {summary['successes']}, "
        f"failures: {summary['failures']} ({summary['partial_failures']} met some goals)",
        "",
        "| Cause | Failures | Share |",
        "| --- | ---: | ---: |",
    ]
    total = max(summary["failures"], 1)
    for cause, count in summary["failure_causes"].items():
        if count:
            lines.append(f"| `{cause}` | {count} | {count / total:.0%} |")
    stable = summary["stability"]
    if stable["episodes"]:
        lines += ["", "| Stability | Episodes |", "| --- | ---: |"]
        for verdict in ("stable_success", "stable_failure", "flaky"):
            lines.append(f"| `{verdict}` | {stable['verdicts'].get(verdict, 0)} |")
        lines += ["", "| Divergence origin | Material | Bitwise |", "| --- | ---: | ---: |"]
        origins = sorted(set(stable["divergence_origins"]) | set(stable["bitwise_divergence_origins"]))
        for origin in origins:
            material = stable["divergence_origins"].get(origin, 0)
            bitwise = stable["bitwise_divergence_origins"].get(origin, 0)
            lines.append(f"| `{origin}` | {material} | {bitwise} |")
    if summary["failure_causes_by_task"]:
        lines += ["", "| Task | Failures by cause |", "| --- | --- |"]
        for task, counter in summary["failure_causes_by_task"].items():
            name = (task_names or {}).get(int(task), task)
            causes = ", ".join(f"`{cause}` {count}" for cause, count in sorted(counter.items()))
            lines.append(f"| {name} | {causes} |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("trace_dir", type=Path, help="the trace directory of a LIBERO run with trace enabled")
    parser.add_argument("--output", type=Path, help="write the full attribution as JSON")
    parser.add_argument("--near-target-m", type=float, default=0.10)
    args = parser.parse_args(argv)
    traces = load_traces(args.trace_dir)
    if not traces:
        raise SystemExit(f"no traces under {args.trace_dir}")
    summary = summarize(traces, args.near_target_m)
    names = {int(trace["task_id"]): f"{trace['task_id']}: {trace['task_description']}" for trace in traces}
    if args.output is not None:
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(render_markdown(summary, names), end="")


if __name__ == "__main__":
    main()
