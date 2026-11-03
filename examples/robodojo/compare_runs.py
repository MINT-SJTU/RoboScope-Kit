"""Compare complete RoboDojo runs by layout, including native printed action counts."""

import argparse
import json
import re
from pathlib import Path


def load_run(directory):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if not summary.get("complete"):
        raise ValueError(f"Incomplete run: {directory}")
    details = json.loads((directory / "result.json").read_text(encoding="utf-8"))["details"]
    rows = list(details.values())
    if summary["num_envs"] == 1 and any("executed_steps" not in row for row in rows):
        log = (directory / "client.log").read_text(encoding="utf-8", errors="replace")
        log = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", log)
        counts, current, limit = [], 0, None
        for index, step, cap in re.findall(r"env(\d+) step: (\d+) / (\d+)", log):
            if int(index) != 0:
                raise ValueError("Unexpected environment index in serial native run")
            step, cap = int(step), int(cap)
            if step == 1 and current:
                counts.append((current, limit))
                current = 0
            if step != current + 1 or (current and cap != limit):
                raise ValueError("Missing, duplicated or inconsistent native step records")
            current, limit = step, cap
        if current:
            counts.append((current, limit))
        if len(counts) != len(rows):
            raise ValueError("Native step sequences do not match completed episode outcomes")
        for row, (count, limit) in zip(rows, counts, strict=True):
            row["executed_steps"] = count
            row["step_limit"] = limit
            row["termination"] = (
                "success" if row["success"] else "step_limit" if count >= limit else "failure"
            )
    by_layout = {int(row["layout_id"]): row for row in rows}
    if len(by_layout) != len(rows):
        raise ValueError("Duplicate layout outcomes")
    if len(rows) != summary["requested_episodes"] or sorted(by_layout) != list(range(len(rows))):
        raise ValueError("Missing or substituted fixed-layout outcomes")
    return summary, by_layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="external run evidence directory")
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--allow-topology-change", action="store_true",
                        help="explicitly compare different worker/environment counts; no parity claim")
    args = parser.parse_args()
    reference, before = load_run(args.reference)
    candidate, after = load_run(args.candidate)
    if reference["seed"] != candidate["seed"] or before.keys() != after.keys():
        raise ValueError("Comparison requires the same seed and distinct layout IDs")
    topology_change = (
        reference["num_envs"], reference.get("g", 1)
    ) != (candidate["num_envs"], candidate.get("g", 1))
    if topology_change and not args.allow_topology_change:
        raise ValueError("Different concurrency requires explicit --allow-topology-change")
    if not reference.get("profile_sha256") or reference["profile_sha256"] != candidate.get("profile_sha256"):
        raise ValueError("Comparison requires the same pinned recipe profile")
    if reference.get("runtime_versions") != candidate.get("runtime_versions"):
        raise ValueError("Comparison requires the same simulator/interpreter/PyTorch versions")
    if reference.get("compatibility_patch_sha256") != candidate.get("compatibility_patch_sha256"):
        raise ValueError("Comparison requires the same simulator compatibility patch")
    if reference.get("source_revisions") != candidate.get("source_revisions"):
        raise ValueError("Comparison requires the same pinned sources")
    for name in ("measurement", "episode-count", "isaac51-deps"):
        a = reference.get("runtime_patches", {}).get(name, {}).get("sha256")
        b = candidate.get("runtime_patches", {}).get(name, {}).get("sha256")
        if a != b:
            raise ValueError(f"Comparison requires the same {name} patch version")
    lost, gained, differences = [], [], []
    audit_fields = (
        "layout_sha256", "initial_state_sha256", "initial_vision_sha256",
        "instruction_sha256", "action_sha256", "control_sha256",
    )
    audit = {field: {"compared": 0, "different_layouts": []} for field in audit_fields}
    for layout_id in sorted(before):
        a, b = before[layout_id], after[layout_id]
        if a["success"] and not b["success"]:
            lost.append(layout_id)
        if not a["success"] and b["success"]:
            gained.append(layout_id)
        for key in ("success", "executed_steps", "step_limit", "termination"):
            if key not in a or key not in b:
                raise ValueError(f"Missing comparison field {key} for layout {layout_id}")
            if a[key] != b[key]:
                differences.append({"layout_id": layout_id, "field": key, "before": a[key], "after": b[key]})
        for field in audit_fields:
            if field in a and field in b:
                audit[field]["compared"] += 1
                if a[field] != b[field]:
                    audit[field]["different_layouts"].append(layout_id)
    report = {
        "episodes": len(before),
        "reference_num_envs": reference["num_envs"],
        "candidate_num_envs": candidate["num_envs"],
        "reference_replicas": reference.get("g", 1),
        "candidate_replicas": candidate.get("g", 1),
        "topology_change": topology_change,
        "baseline_successes": reference["successes"],
        "candidate_successes": candidate["successes"],
        "success_rate_change_pp": 100 * (candidate["success_rate"] - reference["success_rate"]),
        "lost_successes": lost,
        "gained_successes": gained,
        "outcome_and_step_fields_equal": not differences,
        "differences": differences,
        "input_and_trajectory_audit": audit,
        "total_wall_speedup": reference["total_wall_s"] / candidate["total_wall_s"],
        "limitation": "Outcome and step parity do not prove pixel or action-vector equivalence.",
    }
    if "evaluation_s" in reference and "evaluation_s" in candidate:
        report["episode_wall_speedup"] = reference["evaluation_s"] / candidate["evaluation_s"]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
