# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Compare a RoboScope Kit ``rlbench``/``peract_18`` run with a ``run_native.py`` run, episode by episode.

    python examples/peract/compare_native.py --kit-run /runs/peract-full \\
        --native /runs/peract-native --output /runs/peract-compare.json

Both sides replay the same test episodes (task, episode index, variation and stored random
state), so every episode is paired by (task, episode). For each pair it reports whether the
variation and instruction agree, how far each side's initial scene is from the recorded first
frame of the test episode, the largest difference between the first actions (the
first action depends only on the initial observation, so equal initial scenes give equal first
actions), the length of the common prefix of identical actions, and both outcomes and executed
action counts. Per task it reports both success counts and executed action and simulation-step
totals.

RLBench's motion planner samples with CoppeliaSim's own random generator, which neither side
seeds, so trajectories may diverge after the first planned motion; differing outcomes in single
episodes are expected and only the per-task rates are compared.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ACTION_TOLERANCE = 1e-6
# Above this, an episode did not start from its recorded test scene (settling noise is below 1e-2).
SCENE_TOLERANCE = 1e-2


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
    kit = kit_records(args.kit_run)
    native = native_records(args.native)
    if set(kit) != set(native):
        missing = sorted(set(kit) ^ set(native))
        raise SystemExit(f"the runs cover different episodes: {missing[:10]}")
    episodes = []
    tasks: dict[str, dict] = {}
    for key in sorted(kit):
        task, episode = key
        k, n = kit[key], native[key]
        km = k["metadata"]
        row = {
            "task": task,
            "episode": episode,
            "variation_equal": km["variation"] == n["variation"],
            "instruction_equal": km["instruction"] == n["instruction"],
            "first_action_max_abs_diff": max_difference(km["actions"][0], n["actions"][0]),
            "identical_action_prefix": common_prefix(km["actions"], n["actions"]),
            "kit_initial_scene_max_abs_diff": km["initial_scene_max_abs_diff"],
            "kit_clearing_reset_max_abs_diff": km["clearing_reset_max_abs_diff"],
            "native_initial_scene_max_abs_diff": n["initial_scene_max_abs_diff"],
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
                "kit_initial_scene_max_abs_diff": 0.0,
                "native_initial_scene_max_abs_diff": 0.0,
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
        summary["native_scene_differs"] += int(row["native_initial_scene_max_abs_diff"] > SCENE_TOLERANCE)
        for side in ("kit", "native"):
            name = f"{side}_initial_scene_max_abs_diff"
            summary[name] = max(summary[name], row[name])
    totals = {name: sum(summary[name] for summary in tasks.values()) for name in next(iter(tasks.values()))}
    report = {
        "kit_run": str(args.kit_run),
        "native_run": str(args.native),
        "action_tolerance": ACTION_TOLERANCE,
        "scene_tolerance": SCENE_TOLERANCE,
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
