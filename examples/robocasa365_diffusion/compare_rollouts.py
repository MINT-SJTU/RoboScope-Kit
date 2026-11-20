# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Pair full-horizon RoboCasa365 rollouts from native and Kit runs.

Every arm is a directory holding ``trace/`` written by ``rollout_trace``. Native
arms (``run_native.py``, one sub-directory per task) also hold the upstream
``result/<task>/eval_log.json``; Kit arms (``run_kit.py``) hold the CLI's
``result/queue/completed/*.json``. Each episode is identified by task and episode
index (slot = index mod 5, constructor seed 1111111 + slot).

The protocol is nondeterministic: a one-level MSAA RGB difference changes the
next diffusion action, and full-horizon trajectories then diverge, even native
against native. The tolerance, fixed before the acceptance runs, is therefore:

1. Coverage: every arm has each task's five episodes once, with no unknown or
   duplicate record, and the score in the trace equals the arm's own score.
2. Inputs: the initial scenario metadata and initial simulator state of every
   pair are identical (the same constructor stream produces the same scene).
3. Kit against native: the paired outcome disagreements between a Kit arm and a
   native pass are no more frequent than between the two native passes. Given
   ``d(Kit, native)`` and ``d(native A, native B)`` discordant pairs, the
   two-sided exact binomial test of ``d(Kit, native)`` among their sum, at
   p = 0.5, must not reject at 0.05.
4. No direction: among the Kit/native discordant pairs, Kit-only and
   native-only successes must not differ by the two-sided exact sign test at 0.05.

Per-task success differences larger than the largest per-task native A/B
difference are listed with each pair's first divergence for explanation.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

SPLITS = ("atomic_seen", "composite_seen", "composite_unseen")


def read_trace(root: Path, episodes_per_task: int) -> dict[tuple[str, int], dict]:
    episodes: dict[tuple[str, int], dict] = {}
    for path in sorted(root.glob("trace/*/*/slot-*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = row["task"], row["episode_index"]
            if key[1] >= episodes_per_task and row["event"] == "reset":
                continue  # The native runner resets once more before it closes.
            episode = episodes.setdefault(key, {"chunks": []})
            if row["event"] == "chunk":
                episode["chunks"].append(row)
            elif row["event"] in episode:
                raise ValueError(f"{root}: duplicate {row['event']} for {key}")
            else:
                episode[row["event"]] = row
    return episodes


def native_scores(root: Path, tasks: list[str]) -> dict[tuple[str, int], dict]:
    scores = {}
    for task in tasks:
        log = json.loads((root / task / "result" / task / "eval_log.json").read_text())
        seeds = sorted(int(key.rsplit("_", 1)[1]) for key in log if key.startswith("test/sim_max_reward_"))
        for index, seed in enumerate(seeds):
            scores[task, index] = {"success": float(log[f"test/sim_max_reward_{seed}"]) > 0}
    return scores


def kit_scores(root: Path) -> dict[tuple[str, int], dict]:
    scores = {}
    for path in sorted((root / "result/queue/completed").glob("*.json")):
        for record in json.loads(path.read_text())["records"]:
            meta = record["metadata"]
            key = meta["task"], meta["episode_index"]
            if key in scores:
                raise ValueError(f"{root}: duplicate Kit record {key}")
            scores[key] = {"success": record["success"], "steps": record["steps"],
                           "termination": meta["termination"]}
    return scores


def load_arm(root: Path, kind: str, tasks: list[str], episodes: int) -> dict[tuple[str, int], dict]:
    if kind == "native":
        traces = {}
        for task in tasks:
            traces.update(read_trace(root / task, episodes))
        scores = native_scores(root, tasks)
    else:
        traces, scores = read_trace(root, episodes), kit_scores(root)
    expected = {(task, index) for task in tasks for index in range(episodes)}
    for name, keys in (("trace", set(traces)), ("score", set(scores))):
        if keys != expected:
            raise ValueError(f"{root}: {name} coverage differs: missing {sorted(expected - keys)[:5]}, "
                             f"unknown {sorted(keys - expected)[:5]}")
    arm = {}
    for key in sorted(expected):
        trace = traces[key]
        if "reset" not in trace or "episode_end" not in trace:
            raise ValueError(f"{root}: incomplete trace for {key}")
        score = scores[key]
        if score["success"] != trace["episode_end"]["success"]:
            raise ValueError(f"{root}: score and trace disagree for {key}")
        if "steps" in score and score["steps"] != trace["episode_end"]["step"]:
            raise ValueError(f"{root}: step count and trace disagree for {key}")
        arm[key] = {**trace, "success": score["success"], "steps": trace["episode_end"]["step"],
                    "termination": score.get("termination")}
    return arm


def first_divergence(a: dict, b: dict) -> dict | None:
    rows = [("reset", a["reset"], b["reset"])]
    rows += [("chunk", x, y) for x, y in zip(a["chunks"], b["chunks"])]
    rows.append(("episode_end", a["episode_end"], b["episode_end"]))
    for event, x, y in rows:
        if x["step"] != y["step"]:
            return {"event": event, "step": min(x["step"], y["step"]), "fields": ["step"]}
        fields = sorted(
            [name for name in x["feature_sha256"] if x["feature_sha256"][name] != y["feature_sha256"][name]]
            + [name for name in ("action_sha256", "integration_sha256") if x.get(name) != y.get(name)]
        )
        if fields:
            return {"event": event, "step": x["step"], "fields": fields}
    return None


def binomial_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial test at p = 0.5."""
    if n == 0:
        return 1.0
    probabilities = [math.comb(n, i) / 2**n for i in range(n + 1)]
    return min(1.0, sum(p for p in probabilities if p <= probabilities[k] * (1 + 1e-9)))


def pair(a: dict, b: dict) -> dict:
    rows = []
    for key in sorted(a):
        x, y = a[key], b[key]
        rows.append({
            "task": key[0], "episode_index": key[1],
            "success": [x["success"], y["success"]], "steps": [x["steps"], y["steps"]],
            "scenario_equal": x["reset"]["scenario_metadata_sha256"] == y["reset"]["scenario_metadata_sha256"],
            "initial_state_equal": x["reset"]["initial_state_sha256"] == y["reset"]["initial_state_sha256"],
            "first_divergence": first_divergence(x, y),
        })
    discordant = [row for row in rows if row["success"][0] != row["success"][1]]
    return {
        "pairs": len(rows),
        "successes": [sum(row["success"][0] for row in rows), sum(row["success"][1] for row in rows)],
        "discordant": len(discordant),
        "only_first": sum(row["success"][0] for row in discordant),
        "only_second": sum(row["success"][1] for row in discordant),
        "step_differences": sum(row["steps"][0] != row["steps"][1] for row in rows),
        "scenario_mismatches": sum(not row["scenario_equal"] for row in rows),
        "initial_state_mismatches": sum(not row["initial_state_equal"] for row in rows),
        "identical_traces": sum(row["first_divergence"] is None for row in rows),
        "rows": rows,
    }


def per_task(arm: dict) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for (task, _), episode in arm.items():
        counts[task] += episode["success"]
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--native", type=Path, nargs=2, required=True, metavar=("A", "B"))
    parser.add_argument("--kit", action="append", default=[], metavar="NAME=DIR")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from roboscope_kit.runtime.benchmarks.robocasa365.protocol import (
        SUITES,
        TASKS_BY_NAME,
    )

    tasks = list(SUITES["target50"])
    arms = {"native_a": load_arm(args.native[0], "native", tasks, args.episodes),
            "native_b": load_arm(args.native[1], "native", tasks, args.episodes)}
    for value in args.kit:
        name, _, path = value.partition("=")
        arms[name] = load_arm(Path(path), "kit", tasks, args.episodes)
    reference = pair(arms["native_a"], arms["native_b"])
    counts = {name: per_task(arm) for name, arm in arms.items()}
    native_spread = max(abs(counts["native_a"][task] - counts["native_b"][task]) for task in tasks)
    comparisons = {"native_a/native_b": reference}
    verdicts = {}
    for name in [name for name in arms if not name.startswith("native_")]:
        for native in ("native_a", "native_b"):
            result = pair(arms[name], arms[native])
            result["repeat_test_p"] = binomial_two_sided(
                result["discordant"], result["discordant"] + reference["discordant"])
            result["direction_test_p"] = binomial_two_sided(result["only_first"], result["discordant"])
            comparisons[f"{name}/{native}"] = result
            verdicts[f"{name}/{native}"] = {
                "inputs_equal": not result["scenario_mismatches"] and not result["initial_state_mismatches"],
                "within_native_repeat": result["repeat_test_p"] >= 0.05,
                "no_direction": result["direction_test_p"] >= 0.05,
            }
        for left, right in ((name, other) for other in arms if not other.startswith("native_") and other > name):
            comparisons[f"{left}/{right}"] = pair(arms[left], arms[right])
    table = []
    for task in tasks:
        row = {"task": task, "split": TASKS_BY_NAME[task].split, "horizon": TASKS_BY_NAME[task].horizon}
        for name, arm in arms.items():
            row[name] = {"successes": counts[name][task],
                         "steps": sum(arm[task, i]["steps"] for i in range(args.episodes))}
        row["exceeds_native_spread"] = [
            name for name in arms if not name.startswith("native_")
            and min(abs(counts[name][task] - counts[n][task]) for n in ("native_a", "native_b")) > native_spread
        ]
        table.append(row)
    splits = {
        split: {name: sum(counts[name][t] for t in tasks if TASKS_BY_NAME[t].split == split) for name in arms}
        for split in SPLITS
    }
    report = {"episodes_per_task": args.episodes, "tasks": len(tasks), "native_task_spread": native_spread,
              "totals": {name: sum(counts[name].values()) for name in arms}, "splits": splits,
              "verdicts": verdicts, "per_task": table, "comparisons": comparisons}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    summary = {key: value for key, value in report.items() if key not in ("per_task", "comparisons")}
    summary["comparisons"] = {key: {k: v for k, v in value.items() if k != "rows"}
                              for key, value in comparisons.items()}
    print(json.dumps(summary, indent=2))
    if not all(all(verdict.values()) for verdict in verdicts.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
