# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Compare a RoboScope Kit ``rlbench``/``peract_18`` run with a ``run_native.py`` run, episode by episode.

    python examples/peract/compare_native.py --kit-run /runs/peract-full \\
        --native /runs/peract-native --output /runs/peract-compare.json

Both sides replay the same test episodes (task, episode index, variation and stored random
state), so every episode is paired by (task, episode). The comparison rejects missing,
duplicate, unexpected or mismatched inputs. For each pair it reports how far each side's
initial scene is from the recorded first frame, the largest first-action difference, the length
of the common prefix of identical actions, and both outcomes and executed action counts. Per
task it reports both success counts and executed action and simulation-step totals.

RLBench's motion planner samples with CoppeliaSim's own random generator, which neither side
seeds, so trajectories may diverge after the first planned motion; differing outcomes in single
episodes are expected and only the per-task rates are compared.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_native import TASKS

from roboscope_kit.runtime.benchmarks.rlbench.tasks import TASKS_BY_NAME, test_episodes_sha256

ACTION_TOLERANCE = 1e-6
# This is a diagnostic against the recorded vector, which may use another object order.
SCENE_TOLERANCE_M = 0.01


def expected_records(run_dir: Path) -> set[tuple[str, int]]:
    """Reconstruct the intended workload independently of either run's terminal records."""

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    benchmark = manifest["benchmark"]
    tasks = benchmark["task_order"]
    count = benchmark["episodes_per_task"]
    if (
        benchmark.get("benchmark") != "rlbench"
        or benchmark.get("suite") != "peract_18"
        or not isinstance(tasks, list)
        or not tasks
        or len(tasks) != len(set(tasks))
        or any(task not in TASKS for task in tasks)
        or type(count) is not int
        or not 1 <= count <= 25
    ):
        raise SystemExit("Kit manifest has an invalid RLBench workload")
    expected = {(task, episode) for task in tasks for episode in range(count)}
    recorded = []
    for item in manifest["items"]:
        payload = item["payload"]
        if item["group_id"] != payload["task"] or item["expected_results"] != len(payload["episodes"]):
            raise SystemExit("Kit manifest has an inconsistent work item")
        recorded.extend((payload["task"], episode) for episode in payload["episodes"])
    if (len(recorded) != len(expected) or set(recorded) != expected
            or manifest["total_records"] != len(expected)):
        raise SystemExit("Kit manifest does not cover its declared workload exactly once")
    return expected


def kit_records(run_dir: Path) -> dict[tuple[str, int], dict]:
    records: dict[tuple[str, int], dict] = {}
    for path in sorted((run_dir / "queue" / "completed").glob("item-*.json")):
        for record in json.loads(path.read_text(encoding="utf-8"))["records"]:
            key = (record["metadata"]["task"], int(record["metadata"]["episode"]))
            if key in records:
                raise SystemExit(f"Kit run {run_dir} has a duplicate record for {key}")
            records[key] = record
    if not records:
        raise SystemExit(f"no completed Kit records under {run_dir}/queue/completed")
    return records


def native_records(native_dir: Path) -> dict[tuple[str, int], dict]:
    records: dict[tuple[str, int], dict] = {}
    for line in (native_dir / "native-episodes.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        key = (record["task"], int(record["episode"]))
        if key in records:
            raise SystemExit(f"native run {native_dir} has a duplicate record for {key}")
        records[key] = record
    return records


def max_difference(left: list[float], right: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(left, right, strict=True))


def common_prefix(left: list[list[float]], right: list[list[float]]) -> int:
    count = 0
    for a, b in zip(left, right, strict=False):
        if max_difference(a, b) > ACTION_TOLERANCE:
            break
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kit-run", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = expected_records(args.kit_run)
    kit = kit_records(args.kit_run)
    native = native_records(args.native)
    for name, records in (("Kit", kit), ("native", native)):
        if set(records) != expected:
            difference = sorted(set(records) ^ expected)
            raise SystemExit(f"{name} run does not cover the declared workload: {difference[:10]}")
    for task in {task for task, _episode in expected}:
        indices = sorted(episode for name, episode in expected if name == task)
        if len(indices) == 25:
            digest = test_episodes_sha256(
                (episode, str(kit[(task, episode)]["metadata"]["episode_state_sha256"]))
                for episode in indices
            )
            if digest != TASKS_BY_NAME[task].test_episodes_sha256:
                raise SystemExit(f"Kit test episode digest differs from the frozen task: {task}")
    episodes = []
    tasks: dict[str, dict] = {}
    for key in sorted(kit):
        task, episode = key
        k, n = kit[key], native[key]
        km = k["metadata"]
        if (
            not n.get("episode_state_sha256")
            or km.get("episode_state_sha256") != n["episode_state_sha256"]
            or km["variation"] != n["variation"]
            or km["instruction"] != n["instruction"]
        ):
            raise SystemExit(f"Kit and native episode inputs differ or lack state digests: {key}")
        row = {
            "task": task,
            "episode": episode,
            "variation_equal": km["variation"] == n["variation"],
            "instruction_equal": km["instruction"] == n["instruction"],
            "episode_state_equal": True,
            "first_action_max_abs_diff": max_difference(km["actions"][0], n["actions"][0]),
            "identical_action_prefix": common_prefix(km["actions"], n["actions"]),
            "kit_initial_scene_position_error_m": km["initial_scene_position_error_m"],
            "native_initial_scene_position_error_m": n["initial_scene_position_error_m"],
            "kit": {"success": k["success"], "steps": k["steps"], "termination": km["termination"]},
            "native": {"success": n["success"], "steps": n["steps"], "termination": n["termination"]},
        }
        episodes.append(row)
        summary = tasks.setdefault(
            task,
            {
                "episodes": 0,
                "kit_successes": 0,
                "native_successes": 0,
                "same_outcome": 0,
                "first_action_equal": 0,
                "kit_actions": 0,
                "native_actions": 0,
                "kit_simulation_steps": 0,
                "native_simulation_steps": 0,
                "kit_initial_scene_position_error_m": 0.0,
                "native_initial_scene_position_error_m": 0.0,
                "kit_scene_differs": 0,
                "native_scene_differs": 0,
            },
        )
        summary["episodes"] += 1
        summary["kit_successes"] += int(k["success"])
        summary["native_successes"] += int(n["success"])
        summary["same_outcome"] += int(k["success"] == n["success"])
        summary["first_action_equal"] += int(row["first_action_max_abs_diff"] <= ACTION_TOLERANCE)
        summary["kit_actions"] += k["steps"]
        summary["native_actions"] += n["steps"]
        summary["kit_simulation_steps"] += int(km["simulation_steps"])
        summary["native_simulation_steps"] += int(n["simulation_steps"])
        for side in ("kit", "native"):
            name = f"{side}_initial_scene_position_error_m"
            summary[f"{side}_scene_differs"] += int(row[name] > SCENE_TOLERANCE_M)
            summary[name] = max(summary[name], row[name])
    totals = {name: sum(summary[name] for summary in tasks.values()) for name in next(iter(tasks.values()))}
    report = {
        "kit_run": str(args.kit_run),
        "native_run": str(args.native),
        "action_tolerance": ACTION_TOLERANCE,
        "scene_tolerance_m": SCENE_TOLERANCE_M,
        "variations_equal": all(row["variation_equal"] for row in episodes),
        "instructions_equal": all(row["instruction_equal"] for row in episodes),
        "tasks": tasks,
        "totals": totals,
        "kit_mean_success_rate": sum(s["kit_successes"] / s["episodes"] for s in tasks.values()) / len(tasks),
        "native_mean_success_rate": sum(s["native_successes"] / s["episodes"] for s in tasks.values())
        / len(tasks),
        "episodes": episodes,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{'task':32} {'kit':>5} {'native':>6} {'same':>5} {'1st=':>5} {'kit act':>8} {'nat act':>8}")
    for task, s in tasks.items():
        print(
            f"{task:32} {s['kit_successes']:>5} {s['native_successes']:>6} {s['same_outcome']:>5} "
            f"{s['first_action_equal']:>5} {s['kit_actions']:>8} {s['native_actions']:>8}"
        )
    print(
        f"{'total':32} {totals['kit_successes']:>5} {totals['native_successes']:>6} "
        f"{totals['same_outcome']:>5} {totals['first_action_equal']:>5} "
        f"{totals['kit_actions']:>8} {totals['native_actions']:>8}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
