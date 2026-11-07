# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Pair a Kit SimplerEnv run with a native ``run_native.py`` run, episode by episode.

Usage (with the SimplerEnv runtime's Python and the kit source on ``PYTHONPATH``)::

    PYTHONPATH=packages/roboscope-kit/src /sim/simpler-env/env/bin/python examples/rt1/compare_native.py \\
        --suite google_robot_visual_matching --kit-run /runs/kit-vm --native-run /runs/native-vm \\
        --output /runs/compare-vm.json

1. **Protocol.** Every RT-1-X invocation that the official scripts made (``plan.jsonl``) is
   parsed with upstream ``get_args`` and matched to exactly one frozen Kit variant with the
   same environment, build keywords, overlay, frequencies, step limit, and episode list.
2. **Episodes.** Native and Kit episodes are paired on (variant, robot pose, object pose or
   episode id). For each pair it compares success, executed steps, how the episode ended,
   the final ``episode_stats``, the instructions, a SHA-256 of the first policy image, and a
   SHA-256 over every float64 action passed to ``env.step``; with per-step action digests on
   both sides, also the first step whose action differs.
3. **Metrics.** Per task: both sides' official success rate (mean of variant rates) and the
   paired agreement counts.

Writes the full comparison to ``--output`` and prints the per-task table.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from roboscope_kit.runtime.benchmarks.simpler_env.tasks import SUITE_TASKS
from roboscope_kit.runtime.benchmarks.simpler_env.worker import expand_episodes, upstream_args


def _parse(argv: list[str]) -> argparse.Namespace:
    import sys

    from simpler_env.evaluation.argparse import get_args

    saved = sys.argv
    sys.argv = ["main_inference.py", *argv]
    try:
        return get_args()
    finally:
        sys.argv = saved


def variant_key(
    env_name: str,
    scene_name: str,
    robot: str,
    control_freq: int,
    sim_freq: int,
    max_episode_steps: int,
    rgb_overlay_path: str | None,
    enable_raytracing: bool,
    build_kwargs: dict[str, Any] | None,
) -> str:
    overlay = None if rgb_overlay_path is None else os.path.normpath(rgb_overlay_path)
    return json.dumps(
        [env_name, scene_name, robot, control_freq, sim_freq, max_episode_steps, overlay, enable_raytracing,
         build_kwargs or {}],
        sort_keys=True,
    )  # fmt: skip


def args_key(args: argparse.Namespace) -> str:
    return variant_key(
        args.env_name,
        args.scene_name,
        args.robot,
        args.control_freq,
        args.sim_freq,
        args.max_episode_steps,
        args.rgb_overlay_path,
        bool(args.enable_raytracing),
        dict(args.additional_env_build_kwargs or {}),
    )


def episode_key(episode: dict[str, Any]) -> str:
    pose = [round(episode["robot_init_x"], 9), round(episode["robot_init_y"], 9)]
    pose += [round(value, 9) for value in episode["robot_init_quat"]]
    if episode.get("obj_episode_id") is not None:
        obj: list[Any] = ["episode", int(episode["obj_episode_id"])]
    else:
        obj = ["xy", round(episode["obj_init_x"], 9), round(episode["obj_init_y"], 9)]
    return json.dumps([pose, obj])


def first_divergence(kit: list[str] | None, native: list[str] | None) -> int | None:
    """The first step whose env.step action differs (-1 if none), or None without per-step digests."""

    if kit is None or native is None:
        return None
    for index, (left, right) in enumerate(zip(kit, native, strict=False)):
        if left != right:
            return index
    return -1 if len(kit) == len(native) else min(len(kit), len(native))


def kit_records(run_dir: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for path in sorted((run_dir / "queue" / "completed").glob("item-*.json")):
        for record in json.loads(path.read_text(encoding="utf-8"))["records"]:
            if record["record_id"] in records:
                raise SystemExit(f"duplicate Kit record {record['record_id']}")
            records[record["record_id"]] = record
    if not records:
        raise SystemExit(f"no completed Kit records under {run_dir}/queue/completed")
    return records


def native_entries(run_dir: Path) -> list[dict]:
    entries = []
    for path in sorted((run_dir / "episodes").glob("*.jsonl")):
        entries += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", choices=sorted(SUITE_TASKS), required=True)
    parser.add_argument("--kit-run", type=Path, required=True)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # 1. The native invocations against the frozen variants.
    variants = [variant for task in SUITE_TASKS[args.suite] for variant in task.variants]
    by_key: dict[str, Any] = {}
    episodes_by_variant: dict[str, list[dict]] = {}
    for variant in variants:
        parsed = upstream_args(variant)
        key = args_key(parsed)
        if key in by_key:
            raise SystemExit(f"Kit variants {by_key[key].variant_id} and {variant.variant_id} share a key")
        by_key[key] = variant
        episodes_by_variant[variant.variant_id] = expand_episodes(parsed)
    plan = [
        json.loads(line) for line in (args.native_run / "plan.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    kept = [entry for entry in plan if entry["kept"]]
    matched: set[str] = set()
    protocol_mismatches = []
    for entry in kept:
        parsed = _parse(entry["argv"])
        variant = by_key.get(args_key(parsed))
        if variant is None:
            protocol_mismatches.append({"invocation": entry["index"], "reason": "no Kit variant"})
            continue
        native_episodes = [episode_key(episode) for episode in expand_episodes(parsed)]
        kit_episodes = [episode_key(episode) for episode in episodes_by_variant[variant.variant_id]]
        if native_episodes != kit_episodes:
            protocol_mismatches.append({"invocation": entry["index"], "variant": variant.variant_id})
        if variant.variant_id in matched:
            protocol_mismatches.append({"invocation": entry["index"], "reason": "variant matched twice"})
        matched.add(variant.variant_id)
    unmatched_variants = sorted({variant.variant_id for variant in variants} - matched)

    # 2. Episode pairing.
    kit = kit_records(args.kit_run)
    kit_by_episode: dict[tuple[str, str], dict] = {}
    for record in kit.values():
        metadata = record["metadata"]
        kit_by_episode[(metadata["variant"], episode_key(metadata["episode"]))] = record
    native_by_episode: dict[tuple[str, str], dict] = {}
    duplicates = 0
    unknown = 0
    for entry in native_entries(args.native_run):
        variant = by_key.get(
            variant_key(
                entry["env_name"],
                entry["scene_name"],
                entry["robot"],
                entry["control_freq"],
                entry["sim_freq"],
                entry["max_episode_steps"],
                entry["rgb_overlay_path"],
                entry["enable_raytracing"],
                entry["additional_env_build_kwargs"],
            )
        )
        if variant is None:
            unknown += 1
            continue
        key = (variant.variant_id, episode_key(entry))
        if key in native_by_episode:
            duplicates += 1
        native_by_episode[key] = entry

    pairs = []
    for key in sorted(set(kit_by_episode) & set(native_by_episode)):
        k, n = kit_by_episode[key], native_by_episode[key]
        km = k["metadata"]
        pairs.append(
            {
                "variant": key[0],
                "task": km["task"],
                "record_id": k["record_id"],
                "kit_success": k["success"],
                "native_success": n["success"],
                "kit_steps": k["steps"],
                "native_steps": n["steps"],
                "same_outcome": k["success"] == n["success"],
                "same_steps": k["steps"] == n["steps"],
                "same_actions": km["actions_sha256"] == n["actions_sha256"],
                "first_divergent_step": first_divergence(
                    km.get("step_action_digests"), n.get("step_action_digests")
                ),
                "same_first_image": km["first_image_sha256"] == n["first_image_sha256"],
                "same_episode_stats": km["episode_stats"] == n["episode_stats"],
                "same_instructions": km["instructions"] == n["instructions"],
                # Both loops stop on a policy termination or on truncation at the step limit.
                "same_ending": bool(km["truncated"]) == bool(n["truncated"]),
            }
        )

    # 3. Per-task metrics: the official mean of variant rates on each side.
    def official(rows: list[tuple[str, bool]]) -> float | None:
        per_variant: dict[str, list[bool]] = defaultdict(list)
        for variant_id, success in rows:
            per_variant[variant_id].append(success)
        rates = [sum(values) / len(values) for values in per_variant.values()]
        return sum(rates) / len(rates) if rates else None

    tasks = []
    for task in SUITE_TASKS[args.suite]:
        members = [pair for pair in pairs if pair["task"] == task.name]
        kit_rows = [
            (r["metadata"]["variant"], r["success"])
            for r in kit.values()
            if r["metadata"]["task"] == task.name
        ]
        native_rows = [
            (key[0], entry["success"])
            for key, entry in native_by_episode.items()
            if any(variant.variant_id == key[0] for variant in task.variants)
        ]
        row = {
            "task": task.name,
            "frozen_episodes": task.episodes,
            "kit_episodes": len(kit_rows),
            "native_episodes": len(native_rows),
            "paired": len(members),
            "kit_successes": sum(success for _variant, success in kit_rows),
            "native_successes": sum(success for _variant, success in native_rows),
            "kit_success_rate": official(kit_rows),
            "native_success_rate": official(native_rows),
            "kit_steps": sum(pair["kit_steps"] for pair in members),
            "native_steps": sum(pair["native_steps"] for pair in members),
        }
        for field in (
            "same_outcome",
            "same_steps",
            "same_actions",
            "same_first_image",
            "same_episode_stats",
            "same_instructions",
            "same_ending",
        ):
            row[field] = sum(pair[field] for pair in members)
        tasks.append(row)

    report = {
        "suite": args.suite,
        "kit_run": str(args.kit_run),
        "native_run": str(args.native_run),
        "protocol": {
            "native_invocations": len(kept),
            "kit_variants": len(variants),
            "matched_variants": len(matched),
            "unmatched_kit_variants": unmatched_variants,
            "mismatches": protocol_mismatches,
        },
        "native_unknown_episodes": unknown,
        "native_duplicate_episodes": duplicates,
        "kit_unpaired": len(set(kit_by_episode) - set(native_by_episode)),
        "native_unpaired": len(set(native_by_episode) - set(kit_by_episode)),
        "tasks": tasks,
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "pairs"}, indent=2))
    return 0 if not protocol_mismatches and not unmatched_variants else 1


if __name__ == "__main__":
    raise SystemExit(main())
