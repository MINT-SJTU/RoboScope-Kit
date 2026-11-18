# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Compare complete OpenVLA Kit/Native results by suite, task and initial-state index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def index_records(rows, label):
    result = {}
    for row in rows:
        metadata = row.get("metadata", row)
        key = (metadata["task_id"], metadata["episode_index"])
        if key in result:
            raise ValueError(f"Duplicate {label} episode: {key}")
        if type(row["success"]) is not bool or type(row["steps"]) is not int or row["steps"] < 1:
            raise ValueError(f"Invalid {label} result: {key}")
        result[key] = row
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit-run", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.kit_run / "manifest.json")
    benchmark = manifest["benchmark"]
    native_run = read_json(args.native / "run.json")
    count = native_run["episodes_per_task"]
    if (native_run.get("complete") is not True or benchmark.get("benchmark") != "libero"
            or benchmark["suite"] != native_run["suite"]
            or benchmark["seed"] != native_run["env_seed"]
            or benchmark["episodes_per_task"] != count
            or sorted(benchmark["task_ids"]) != list(range(10))):
        raise ValueError("Incomplete Native run or mismatched suite, environment seed or workload")
    expected = {(task, episode) for task in range(10) for episode in range(count)}
    declared = []
    for item in manifest["items"]:
        p = item["payload"]
        declared.extend((p["task_id"], e) for e in range(p["start_episode"], p["start_episode"] + p["num_trials"]))
    if len(declared) != len(expected) or set(declared) != expected:
        raise ValueError("Kit manifest must cover the same ten tasks and states 0..N-1 exactly once")
    kit = index_records(
        [r for p in sorted((args.kit_run / "queue/completed").glob("item-*.json"))
         for r in read_json(p)["records"]], "Kit")
    rows = [json.loads(line) for line in (args.native / "native-episodes.jsonl").read_text().splitlines()]
    if any(r["suite"] != native_run["suite"] for r in rows):
        raise ValueError("Unexpected Native suite")
    native = index_records(rows, "Native")
    if set(kit) != expected or set(native) != expected:
        raise ValueError("Missing or unexpected episodes; incomplete runs cannot be compared")
    tasks, differences = [], []
    for task in range(10):
        pairs = [(episode, kit[task, episode], native[task, episode]) for episode in range(count)]
        tasks.append({"task_id": task, "episodes": count,
                      "kit_successes": sum(k["success"] for _, k, n in pairs),
                      "native_successes": sum(n["success"] for _, k, n in pairs),
                      "kit_steps": sum(k["steps"] for _, k, n in pairs),
                      "native_steps": sum(n["steps"] for _, k, n in pairs),
                      "kit_only": sum(k["success"] and not n["success"] for _, k, n in pairs),
                      "native_only": sum(n["success"] and not k["success"] for _, k, n in pairs)})
        for episode, k, n in pairs:
            if (k["success"], k["steps"]) != (n["success"], n["steps"]):
                differences.append({"task_id": task, "episode_index": episode,
                                    "kit": {f: k[f] for f in ("success", "steps")},
                                    "native": {f: n[f] for f in ("success", "steps")}})
    totals = {f: sum(t[f] for t in tasks) for f in tasks[0] if f != "task_id"}
    report = {"suite": native_run["suite"], "tasks": tasks, "totals": totals,
              "differences": differences,
              "scope": "Identity/outcome comparison only; does not verify equal scene, inputs, checkpoint or trajectories."}
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
