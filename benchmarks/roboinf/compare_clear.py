"""Compare authorized GR00T clear-861 judgments; never load judgments in the policy."""

import argparse
import hashlib
import json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path


def index(rows, cases):
    output = {}
    for row in rows:
        key = row["case_id"]
        if (
            row["domain"] != "test"
            or key not in cases
            or key in output
            or any(
                type(row[k]) is not bool
                for k in ("task_success", "instruction_success")
            )
        ):
            raise ValueError(f"Invalid clear judgment: {key}")
        output[key] = (
            row["task_success"],
            row["task_success"] and row["instruction_success"],
        )
    if output.keys() != cases.keys():
        raise ValueError("Incomplete clear judgments")
    return output


def compare(cases, baseline, candidate):
    a, b = index(baseline, cases), index(candidate, cases)
    groups = defaultdict(list)
    for key, task in cases.items():
        groups[task].append(key)
    report = {"tasks": {}, "macro": {}, "passed": True}
    for i, name in enumerate(("TR", "JR")):
        differences = []
        for task, keys in sorted(groups.items()):
            old, new = (sum(rows[k][i] for k in keys) for rows in (a, b))
            diff = Fraction(100 * (new - old), len(keys))
            report["tasks"].setdefault(task, {})[name] = {
                "n": len(keys),
                "baseline": old,
                "candidate": new,
                "delta_pp": float(diff),
                "disagreements": sum(a[k][i] != b[k][i] for k in keys),
            }
            differences.append(diff)
            report["passed"] &= abs(diff) <= 5
        macro = sum(differences) / len(differences)
        report["macro"][name] = {"delta_pp": float(macro)}
        report["passed"] &= abs(macro) <= 2
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("cases", "baseline", "candidate"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    raw = args.cases.read_bytes()
    if (
        hashlib.sha256(raw).hexdigest()
        != "9a88101906acce3ecc55011121bd012e85047ba4a9b75f9e7a670309d7d0951b"
    ):
        parser.error("Expected the official RoboInF v1.0.0 test-cases.jsonl")
    cases = {row["case_id"]: row["task"] for row in map(json.loads, raw.splitlines())}
    report = compare(
        cases,
        *(
            map(json.loads, p.read_bytes().splitlines())
            for p in (args.baseline, args.candidate)
        ),
    )
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
