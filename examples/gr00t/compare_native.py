# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Pair a Kit BEHAVIOR run with a ``run_native.py`` run, episode by episode.

Reconstructs the Kit workload from the run's ``manifest.json``, requires exactly one Kit record and
one native result per expected (task, instance) and nothing else, from runs with the same observation
wrapper, and reports per task the successes, mean Q score and mean executed steps of each side,
and every episode whose success, Q score or step count differs. The simulator and a sampling
policy (a diffusion or flow-matching action head) are not deterministic (BEHAVIOR's evaluation
docs: different rollouts of the same policy may differ), so a difference is reported, not an
error; the acceptance tolerance is stated in the protocol document. Any example's
``run_native.py`` output can be compared.

Usage::

    python compare_native.py --kit-run <kit-output-dir> --native <run_native-output-dir> [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages" / "roboscope-kit" / "src"))

from roboscope_kit.runtime.benchmarks.behavior.tasks import (  # noqa: E402
    BEHAVIOR_1K_COMMIT,
    PROTOCOL_VERSION,
    TASK_NAMES,
    instance_id,
)


def expected_records(run_dir: Path, native_dir: Path) -> set[tuple[str, int]]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    benchmark = manifest["benchmark"]
    native = json.loads((native_dir / "run.json").read_text(encoding="utf-8"))
    if native.get("env_wrapper") != benchmark.get("env_wrapper"):
        raise SystemExit(
            f"the native run observed through {native.get('env_wrapper')}, "
            f"the Kit run through {benchmark.get('env_wrapper')}"
        )
    tasks = [row["task"] for row in benchmark["tasks"]]
    count = benchmark["instances_per_task"]
    if (
        benchmark.get("benchmark") != "behavior"
        or benchmark.get("protocol") != PROTOCOL_VERSION
        or benchmark.get("behavior_1k_commit") != BEHAVIOR_1K_COMMIT
        or not tasks
        or len(tasks) != len(set(tasks))
        or any(task not in TASK_NAMES for task in tasks)
    ):
        raise SystemExit("Kit manifest has an invalid BEHAVIOR workload")
    expected = {(task, instance_id(index)) for task in tasks for index in range(count)}
    recorded = [
        (item["payload"]["task"], instance_id(int(item["payload"]["instance_index"])))
        for item in manifest["items"]
    ]
    if (
        len(recorded) != len(expected)
        or set(recorded) != expected
        or manifest["total_records"] != len(expected)
    ):
        raise SystemExit("Kit manifest does not cover its declared workload exactly once")
    return expected


def kit_records(run_dir: Path) -> dict[tuple[str, int], dict]:
    records: dict[tuple[str, int], dict] = {}
    for path in sorted((run_dir / "queue" / "completed").glob("item-*.json")):
        for record in json.loads(path.read_text(encoding="utf-8"))["records"]:
            key = (record["metadata"]["task"], int(record["metadata"]["instance_id"]))
            if key in records:
                raise SystemExit(f"Kit run {run_dir} has a duplicate record for {key}")
            records[key] = {
                "success": bool(record["success"]),
                "q_score": float(record["metadata"]["q_score"]),
                "steps": int(record["steps"]),
            }
    return records


def native_records(native_dir: Path) -> dict[tuple[str, int], dict]:
    records: dict[tuple[str, int], dict] = {}
    for path in sorted((native_dir / "json").glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        key = (result["task"], int(result["instance_id"]))
        if key in records or int(result.get("rollout_id", 0)) != 0:
            raise SystemExit(f"native run has a duplicate or extra result {path}")
        records[key] = {
            "success": bool(result["success"]),
            "q_score": float(result["q_score"]["final"]),
            "steps": int(result["steps"]),
        }
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kit-run", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    expected = expected_records(args.kit_run, args.native)
    kit = kit_records(args.kit_run)
    native = native_records(args.native)
    for name, records in (("Kit", kit), ("native", native)):
        if set(records) != expected:
            difference = sorted(set(records) ^ expected)
            raise SystemExit(f"{name} records do not match the workload: {difference}")
    rows = []
    disagreements = []
    for task in [task for task in TASK_NAMES if any(key[0] == task for key in expected)]:
        keys = sorted(key for key in expected if key[0] == task)
        row = {"task": task, "episodes": len(keys)}
        for name, records in (("kit", kit), ("native", native)):
            row[f"{name}_successes"] = sum(records[key]["success"] for key in keys)
            row[f"{name}_mean_q_score"] = sum(records[key]["q_score"] for key in keys) / len(keys)
            row[f"{name}_mean_steps"] = sum(records[key]["steps"] for key in keys) / len(keys)
        rows.append(row)
        for key in keys:
            if kit[key] != native[key]:
                disagreements.append(
                    {"task": key[0], "instance_id": key[1], "kit": kit[key], "native": native[key]}
                )
    report = {
        "episodes": len(expected),
        "kit_successes": sum(record["success"] for record in kit.values()),
        "native_successes": sum(record["success"] for record in native.values()),
        "kit_mean_q_score": sum(row["kit_mean_q_score"] for row in rows) / len(rows),
        "native_mean_q_score": sum(row["native_mean_q_score"] for row in rows) / len(rows),
        "success_disagreements": sum(d["kit"]["success"] != d["native"]["success"] for d in disagreements),
        "tasks": rows,
        "episode_differences": disagreements,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.json is not None:
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
