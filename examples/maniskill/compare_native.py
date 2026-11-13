"""Compare ManiSkill SDK records with an independently evaluated native run.

The expected profile, episode, seed and horizon matrix comes from the benchmark
manifest. Optional trajectory directories compare each consumed input and action.
This command uses only the Python standard library and does not load the simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare(args: argparse.Namespace) -> dict:
    errors = []
    manifest = read_json(args.benchmark_manifest)
    protocol = manifest["benchmark_protocols"]["maniskill"]
    width = protocol.get("vector_envs")
    if type(width) is not int or width < 1:
        raise ValueError("The comparison requires the frozen protocol.vector_envs")
    profile_ids = protocol["suites"][args.suite]
    if len(profile_ids) != len(set(profile_ids)):
        raise ValueError("The manifest suite contains duplicate profiles")
    profiles = {name: {
        **protocol["profiles"][name],
        **({"diagnostic_step_limit": protocol["diagnostic_step_limit"]}
           if "diagnostic_step_limit" in protocol
           and "diagnostic_step_limit" not in protocol["profiles"][name] else {}),
    } for name in profile_ids}
    expected = {}
    for name, profile in profiles.items():
        count = profile.get("episodes_per_task", protocol.get("episodes_per_task", 100))
        seeds = profile.get("seeds", protocol.get("seeds", list(range(count))))
        if len(seeds) != count or len(set(seeds)) != count:
            raise ValueError(f"The manifest seed list is incomplete or duplicated for {name}")
        count = args.episodes or count
        if count > len(seeds):
            raise ValueError(f"Requested episodes exceed the frozen seeds for {name}")
        if count % width:
            raise ValueError("The comparison must preserve complete protocol vector groups")
        for index, seed in enumerate(seeds[:count]):
            expected[name, index, seed] = profile

    native = read_json(args.native_result)
    if native.get("mode") != "native_gpu_vectorized":
        raise ValueError("The native result must come from the GPU-vectorized evaluator")
    sdk_manifest = read_json(args.sdk_output / "manifest.json")["benchmark"]
    sdk_result = read_json(args.sdk_output / "result.json")
    if sdk_result["result"].get("state") != "complete":
        errors.append({"side": "sdk", "field": "evaluation_state", "reason": "Run is incomplete"})
    for side, value, key in (
        ("native", native.get("suite"), "suite"),
        ("sdk", sdk_manifest.get("suite"), "suite"),
        ("native", native.get("protocol"), "protocol"),
        ("sdk", sdk_manifest.get("protocol"), "protocol"),
    ):
        wanted = args.suite if key == "suite" else protocol["protocol_id"]
        if value != wanted:
            errors.append({"side": side, "field": key, "expected": wanted, "actual": value})
    if (type(width) is not int or width < 1 or native["num_envs"] != width
            or (args.num_envs is not None and args.num_envs != width)
            or sdk_manifest.get("envs_per_worker") != width):
        raise ValueError("Native vector width differs from the requested comparison width")
    task_profiles = [task["profile_id"] for task in native["tasks"]]
    if set(task_profiles) != set(profiles) or len(task_profiles) != len(profiles):
        errors.append({"side": "native", "field": "task_profile_coverage", "actual": task_profiles})
    for task in native["tasks"]:
        profile = profiles.get(task["profile_id"])
        if profile is None:
            continue
        for field in ("task_id", "robot_uid", "control_mode"):
            if task.get(field) != profile[field]:
                errors.append({"side": "native", "profile_id": task["profile_id"], "field": field})

    sdk = []
    item_ids = []
    for path in sorted((args.sdk_output / "queue/completed").glob("*.json")):
        group = read_json(path)
        item_ids.append(group["item_id"])
        if (group["metadata"].get("vector_envs") != width
                or len(group["records"]) != width):
            errors.append({"side": "sdk", "item_id": group["item_id"], "field": "vector_envs"})
        for raw in group["records"]:
            row = {**raw["metadata"], "steps": raw["steps"], "record_id": raw["record_id"]}
            row["episode"] = row["episode_index"]
            if row["profile_id"] != group["group_id"]:
                errors.append({"side": "sdk", "record_id": raw["record_id"], "field": "group_id"})
            score = row.get("success_metric", "success_once")
            if score != "return" and raw["success"] != row.get(score):
                errors.append({"side": "sdk", "record_id": raw["record_id"], "field": "success"})
            sdk.append(row)
    duplicates = [item for item, count in Counter(item_ids).items() if count > 1]
    if duplicates:
        errors.append({"side": "sdk", "duplicate_item_ids": duplicates})

    indexed = {}
    for side, rows in (("sdk", sdk), ("native", native["records"])):
        groups = {}
        for row in rows:
            key = row["profile_id"], row["episode"], row["seed"]
            if key in groups:
                errors.append({"side": side, "duplicate_episode": key})
            groups[key] = row
            profile = expected.get(key)
            if profile is None:
                errors.append({"side": side, "unexpected_episode": key})
                continue
            name, index, seed = key
            horizon = profile["max_episode_steps"]
            steps = profile.get("diagnostic_step_limit", horizon)
            checks = {
                "task_id": profile["task_id"], "steps": steps,
                "vector_lane": index % width,
                "terminated": False, "truncated": steps == horizon,
            }
            if "diagnostic_step_limit" in profile:
                checks.update({"diagnostic_step_limit": steps, "native_max_episode_steps": horizon})
            if side == "sdk":
                checks.update({
                    "record_id": f"{name}-episode-{index:06d}-seed-{seed}",
                    "suite": args.suite, "protocol": protocol["protocol_id"],
                    "robot_uid": profile["robot_uid"], "control_mode": profile["control_mode"],
                    "obs_mode": profile["obs_mode"],
                })
                if "action_dimension" in profile:
                    checks["action_dimension"] = profile["action_dimension"]
            for field, value in checks.items():
                if row.get(field) != value or (
                    field in {"terminated", "truncated"} and type(row.get(field)) is not bool
                ):
                    errors.append({"side": side, "episode": key, "field": field,
                                   "expected": value, "actual": row.get(field)})
        missing = sorted(set(expected) - set(groups))
        if missing:
            errors.append({"side": side, "missing_episodes": missing})
        indexed[side] = groups

    per_task = []
    maximum_action_difference = maximum_return_difference = 0.0
    for key in sorted(set(expected) & set(indexed["sdk"]) & set(indexed["native"])):
        left, right = indexed["sdk"][key], indexed["native"][key]
        digest_fields = ["initial_state_sha256"]
        if (expected[key]["obs_mode"] != "state"
                or expected[key].get("include_root_kinematics", False)):
            digest_fields.append("initial_observation_sha256")
        for field in (*digest_fields, "success_available", "success_once", "success_at_end"):
            if field not in left or field not in right or left[field] != right[field]:
                errors.append({"episode": key, "field": field,
                               "sdk": left.get(field), "native": right.get(field)})
        for side, row in (("sdk", left), ("native", right)):
            for field in digest_fields:
                digest = row.get(field, "")
                if (not isinstance(digest, str) or len(digest) != 64
                        or any(c not in "0123456789abcdef" for c in digest)):
                    errors.append({"side": side, "episode": key, "field": f"{field}_format"})
            success_type = bool if row.get("success_available") else type(None)
            if type(row.get("success_available")) is not bool or any(
                type(row.get(field)) is not success_type
                for field in ("success_once", "success_at_end")
            ):
                errors.append({"side": side, "episode": key, "field": "success_metric_type"})
        actions = left.get("first_action"), right.get("first_action")
        valid_actions = all(
            isinstance(value, list) and value
            and all(type(number) in (int, float) and math.isfinite(number) for number in value)
            for value in actions
        )
        if not valid_actions or len(actions[0]) != len(actions[1]):
            errors.append({"episode": key, "field": "first_action_shape"})
        else:
            if len(actions[0]) != left.get("action_dimension"):
                errors.append({"episode": key, "field": "first_action_dimension"})
            differences = [abs(a - b) for a, b in zip(*actions, strict=True)]
            maximum_action_difference = max(maximum_action_difference, max(differences))
            if not all(math.isfinite(x) and x <= args.action_atol for x in differences):
                errors.append({"episode": key, "field": "first_action",
                               "max_absolute_difference": max(differences)})
        values = left.get("episode_return"), right.get("episode_return")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            errors.append({"episode": key, "field": "episode_return", "reason": "Missing or nonfinite"})
        else:
            maximum_return_difference = max(maximum_return_difference, abs(values[0] - values[1]))
            if not math.isclose(*values, rel_tol=args.return_rtol, abs_tol=args.return_atol):
                errors.append({"episode": key, "field": "episode_return",
                               "sdk": values[0], "native": values[1]})

    summaries = {"sdk": sdk_result["official_result"]["tasks"], "native": native["tasks"]}
    for side, rows in summaries.items():
        names = [row["profile_id"] for row in rows]
        if set(names) != set(profiles) or len(names) != len(profiles):
            errors.append({"side": side, "field": "summary_profile_coverage", "actual": names})
    for name, profile in profiles.items():
        row = {"profile_id": name, "task_id": profile["task_id"],
               "expected_episodes": sum(key[0] == name for key in expected)}
        for side, groups in indexed.items():
            records = [value for key, value in groups.items() if key[0] == name]
            row[side] = {
                "episodes": len(records), "steps": sum(record["steps"] for record in records),
                "success_once": sum(record["success_once"] for record in records)
                if records and all(type(record.get("success_once")) is bool for record in records) else None,
                "success_at_end": sum(record["success_at_end"] for record in records)
                if records and all(type(record.get("success_at_end")) is bool for record in records)
                else None,
            }
            summary = next((value for value in summaries[side] if value["profile_id"] == name), None)
            score = profile.get("success_metric", "success_once")
            successes = sum(record[score] for record in records) if score != "return" else None
            if summary is not None and (summary.get("episodes") != len(records)
                                        or summary.get("successes") != successes):
                errors.append({"side": side, "profile_id": name, "field": "summary_counts"})
        per_task.append(row)

    trajectory_comparison = None
    if args.sdk_trajectories is not None:
        trajectory_comparison = compare_trajectories(args, expected, errors)
    if native.get("episodes") != len(native["records"]) or native.get("actual_simulation_steps") != sum(
        row["steps"] for row in native["records"]
    ):
        errors.append({"side": "native", "field": "aggregate_counts"})
    if (sdk_result["result"].get("total_episodes") != len(sdk)
            or sdk_result["result"].get("total_steps") != sum(row["steps"] for row in sdk)
            or sdk_result["official_result"]["overall"].get("episodes") != len(sdk)):
        errors.append({"side": "sdk", "field": "aggregate_counts"})
    return {
        "passed": not errors, "suite": args.suite, "protocol": protocol["protocol_id"],
        "native_configuration_source": native.get("configuration_source", "legacy_manifest_configured"),
        "original_upstream_paired_acceptance": False,
        "expected_episodes": len(expected), "sdk_episodes": len(sdk),
        "native_episodes": len(native["records"]), "vector_envs": width,
        "max_first_action_absolute_difference": maximum_action_difference,
        "max_episode_return_absolute_difference": maximum_return_difference,
        "tolerances": {"action_atol": args.action_atol, "return_atol": args.return_atol,
                       "return_rtol": args.return_rtol},
        "per_task": per_task, "trajectories": trajectory_comparison, "errors": errors,
        "sources": {
            "sdk_output": str(args.sdk_output.resolve()),
            "native_result": str(args.native_result.resolve()),
            "benchmark_manifest": str(args.benchmark_manifest.resolve()),
            "benchmark_manifest_sha256": hashlib.sha256(args.benchmark_manifest.read_bytes()).hexdigest(),
        },
    }


def compare_trajectories(args: argparse.Namespace, expected: dict, errors: list) -> dict:
    filenames = {
        f"{name}-episode-{index:06d}-seed-{seed}.jsonl": (name, index, seed)
        for name, index, seed in expected
    }
    counts = {}
    paths_by_side = {}
    for side, directory in (("sdk", args.sdk_trajectories), ("native", args.native_trajectories)):
        paths = [path for path in directory.rglob("*.jsonl") if path.is_file()]
        paths_by_side[side] = {path.name: path for path in paths}
        actual = set(paths_by_side[side])
        counts[side] = len(paths)
        duplicates = sorted(name for name, count in Counter(path.name for path in paths).items() if count > 1)
        if duplicates:
            errors.append({"side": side, "trajectory_duplicate": duplicates})
        if actual != set(filenames):
            errors.append({"side": side, "trajectory_missing": sorted(set(filenames) - actual),
                           "trajectory_unexpected": sorted(actual - set(filenames))})
    for filename, key in filenames.items():
        paths = paths_by_side["sdk"].get(filename), paths_by_side["native"].get(filename)
        if any(path is None for path in paths):
            continue
        traces = [
            [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            for path in paths
        ]
        horizon = expected[key].get("diagnostic_step_limit", expected[key]["max_episode_steps"])
        for side, rows in zip(("sdk", "native"), traces, strict=True):
            if [row["step"] for row in rows] != list(range(horizon + 1)):
                errors.append({"side": side, "episode": key, "field": "trajectory_step_coverage"})
        for left, right in zip(*traces, strict=False):
            if left != right:
                errors.append({"episode": key, "field": "trajectory", "first_divergent_step": left["step"],
                               "sdk": left, "native": right})
                break
    return {"sdk_files": counts["sdk"], "native_files": counts["native"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-output", type=Path, required=True)
    parser.add_argument("--native-result", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-atol", type=float, default=0.0)
    parser.add_argument("--return-atol", type=float, default=0.0)
    parser.add_argument("--return-rtol", type=float, default=0.0)
    parser.add_argument("--sdk-trajectories", type=Path)
    parser.add_argument("--native-trajectories", type=Path)
    args = parser.parse_args()
    if any(value is not None and value < 1 for value in (args.episodes, args.num_envs)):
        parser.error("Episode counts and vector width must be positive")
    if any(not math.isfinite(value) or value < 0 for value in (
        args.action_atol, args.return_atol, args.return_rtol,
    )):
        parser.error("Comparison tolerances must be finite and nonnegative")
    if (args.sdk_trajectories is None) != (args.native_trajectories is None):
        parser.error("Provide both trajectory directories")
    try:
        report = compare(args)
    except (KeyError, TypeError, ValueError, OSError) as error:
        report = {"passed": False, "errors": [{"error": type(error).__name__, "message": str(error)}]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "report": str(args.output),
                      "errors": len(report["errors"])}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
