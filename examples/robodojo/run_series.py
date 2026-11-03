"""Run explicit repeated layout blocks, preserving every result outside Git."""

import argparse
import hashlib
import json
import signal
import subprocess
import sys
import threading
import time
import tomllib
from dataclasses import asdict
from pathlib import Path

from roboscope_kit.runtime.benchmark.resources import cpu_quota, read_cpu_stat

from run_eval import gpu_summary, sample_gpu, stop

HERE = Path(__file__).resolve().parent


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def sample_resources(finished, path):
    with path.open("x", encoding="utf-8") as output:
        while not finished.is_set():
            memory = Path("/sys/fs/cgroup/memory.current")
            row = {
                "time_unix_s": time.time(),
                "cpu": asdict(read_cpu_stat()),
                "memory_current_bytes": int(memory.read_text()) if memory.is_file() else None,
            }
            output.write(json.dumps(row) + "\n")
            output.flush()
            finished.wait(1)


def resource_summary(rows, windows):
    intervals = []
    for previous, current in zip(rows, rows[1:]):
        start, end = previous["time_unix_s"], current["time_unix_s"]
        if any(low <= start < end <= high for low, high in windows):
            intervals.append((end - start, previous, current))
    wall = sum(interval[0] for interval in intervals)
    if not wall:
        return None
    usage = sum(b["cpu"].get("usage_usec", 0) - a["cpu"].get("usage_usec", 0) for _, a, b in intervals)
    periods = sum(b["cpu"]["nr_periods"] - a["cpu"]["nr_periods"] for _, a, b in intervals)
    throttled = sum(b["cpu"]["nr_throttled"] - a["cpu"]["nr_throttled"] for _, a, b in intervals)
    memory = [(dt, row["memory_current_bytes"]) for dt, _, row in intervals if row["memory_current_bytes"] is not None]
    return {
        "covered_s": wall,
        "mean_cpu_cores": usage / 1e6 / wall,
        "throttled_period_ratio": throttled / periods if periods else None,
        "memory_mean_bytes": sum(dt * value for dt, value in memory) / sum(dt for dt, _ in memory) if memory else None,
        "memory_peak_bytes": max(value for _, value in memory) if memory else None,
        "memory_scope": "container cgroup, includes file cache and other container processes",
    }


def aggregate(record, summaries, resources):
    completed = sum(row["completed_episodes"] for row in summaries)
    successes = sum(row["successes"] for row in summaries)
    steps = sum(row["executed_steps"] for row in summaries)
    formal = sum(row["evaluation_s"] for row in summaries)
    wall = time.time() - record["started_unix_s"]
    record.update(
        completed_rounds=len(summaries), completed_episodes=completed, successes=successes,
        success_rate=successes / completed if completed else None, executed_steps=steps,
        evaluation_s=formal, total_wall_s=wall,
        evaluation_episodes_per_s=completed / formal if formal else None,
        evaluation_steps_per_s=steps / formal if formal else None,
        total_episodes_per_s=completed / wall, total_steps_per_s=steps / wall,
        evaluation_resources=resource_summary(resources, [
            (row["evaluation_start_unix_s"], row["evaluation_end_unix_s"]) for row in summaries
        ]),
        total_resources=resource_summary(resources, [(record["started_unix_s"], time.time())]),
    )
    for key in ("gpu_total", "gpu_evaluation"):
        values = [row[key] for row in summaries if row.get(key)]
        count = sum(row["samples"] for row in values)
        record[key] = {
            "samples": count,
            **{name: sum(row[name] * row["samples"] for row in values) / count for name in (
                "mean_utilization_percent", "mean_memory_mib",
            )},
            "peak_memory_mib": max(row["peak_memory_mib"] for row in values),
        } if count else None
    record["gpu_total_scope"] = "sample-weighted child-run windows; excludes between-round gaps"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--env-root", type=Path, default=Path("/workspace/robodojo-envs"))
    parser.add_argument("--profile", type=Path, default=HERE / "recipe.toml")
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--episodes-per-round", type=int, choices=range(1, 51), default=50)
    parser.add_argument("--num-envs", type=int, choices=[1, 2], default=1)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-per-round-s", type=float, default=14400)
    parser.add_argument("--vulkan-icd", type=Path)
    parser.add_argument("--async-video", action="store_true")
    parser.add_argument("--control-state-reuse", action="store_true", help="reuse control expansion within one dual-arm call")
    parser.add_argument("--unpaced-render", action="store_true")
    parser.add_argument("--physics-threads", type=int, choices=[1, 4, 8])
    args = parser.parse_args()
    if args.rounds < 1 or args.episodes_per_round < args.num_envs:
        parser.error("Positive rounds and at least one real episode per environment are required")
    if not args.label.replace("-", "").replace("_", "").isalnum():
        parser.error("Invalid label")
    root, profile_path, output = args.root.resolve(), args.profile.resolve(), args.output.resolve()
    profile = tomllib.loads(profile_path.read_text(encoding="utf-8"))
    cfg = profile["protocol"]
    if (cfg["task"], cfg["policy"], cfg["env_cfg"]) != ("stack_blocks", "Simate_beta", "arx_x5"):
        parser.error("Only the declared stack_blocks / Simate_beta / arx_x5 profile is supported")
    output.mkdir(parents=True, exist_ok=False)
    if not Path("/sys/fs/cgroup/cpu.stat").is_file():
        parser.error("A cgroup v2 CPU accounting file is required for full-container measurements")
    layout_root = root / "Assets/Eval_Layout/RoboDojo/arx_x5" / str(args.seed)
    layout_hashes = {str(i): file_digest(layout_root / f"stack_blocks_{i}.json") for i in range(args.episodes_per_round)}
    record = {
        "complete": False,
        "scope": "single-task fixed-layout evaluation" if args.rounds == 1 else "single-task repeated-layout experiment",
        "task": cfg["task"], "seed": args.seed, "num_envs": args.num_envs,
        "async_video": args.async_video,
        "control_state_reuse": args.control_state_reuse,
        "unpaced_render": args.unpaced_render,
        "physics_threads": args.physics_threads,
        "rounds": args.rounds, "episodes_per_round": args.episodes_per_round,
        "requested_episodes": args.rounds * args.episodes_per_round,
        "unique_layouts": args.episodes_per_round, "layout_sha256": layout_hashes,
        "profile_sha256": file_digest(profile_path), "source_revisions": profile["sources"],
        "measurement_sha256": file_digest(HERE / "patches/0001-measurement.patch"),
        "cpu_quota": cpu_quota(), "started_unix_s": time.time(), "runs": [],
        "rng": "native continuous stream within each round; fresh model process with the same seed each round",
    }
    atomic_json(output / "series.json", record)
    finished = threading.Event()
    sampler = threading.Thread(target=sample_resources, args=(finished, output / "resources.jsonl"), daemon=True)
    sampler.start()
    monitor = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits", "-i", "0", "-lms", "500"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, start_new_session=True,
    )
    gpu_thread = threading.Thread(target=sample_gpu, args=(monitor, output / "gpu.jsonl"), daemon=True)
    gpu_thread.start()
    process, summaries = None, []

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        for index in range(args.rounds):
            if file_digest(profile_path) != record["profile_sha256"]:
                raise ValueError("Profile changed during the series")
            if any(file_digest(layout_root / f"stack_blocks_{i}.json") != digest for i, digest in layout_hashes.items()):
                raise ValueError("Layout changed during the series")
            pointer = output / f"round-{index:03d}.json"
            command = [
                sys.executable, str(HERE / "run_eval.py"), str(root),
                "--env-root", str(args.env_root), "--profile", str(profile_path),
                "--mode", "measured", "--episodes", str(args.episodes_per_round),
                "--num-envs", str(args.num_envs), "--seed", str(args.seed),
                "--timeout-s", str(args.timeout_per_round_s),
                "--label", f"{args.label}-r{index:03d}", "--summary-path", str(pointer),
            ]
            if args.vulkan_icd:
                command.extend(["--vulkan-icd", str(args.vulkan_icd)])
            if args.control_state_reuse:
                command.append("--control-state-reuse")
            if args.async_video:
                command.append("--async-video")
            if args.unpaced_render:
                command.append("--unpaced-render")
            if args.physics_threads is not None:
                command.extend(["--physics-threads", str(args.physics_threads)])
            record["active_round"] = index
            record["active_command"] = command
            atomic_json(output / "series.json", record)
            with (output / f"round-{index:03d}.log").open("x", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                code = process.wait()
            process = None
            if code:
                raise RuntimeError(f"Round {index} exited {code}; no retry or missing result is hidden")
            summary_path = Path(json.loads(pointer.read_text())["summary_path"])
            summary = json.loads(summary_path.read_text())
            if not summary["complete"] or summary["completed_episodes"] != args.episodes_per_round:
                raise ValueError(f"Incomplete round {index}")
            result = json.loads((summary_path.parent / "result.json").read_text())
            rows = list(result["details"].values())
            if sorted(row["layout_id"] for row in rows) != list(range(args.episodes_per_round)):
                raise ValueError("Duplicate, missing or unexpected layouts")
            for row in rows:
                for name in ("initial_state_sha256", "initial_vision_sha256", "instruction_sha256", "layout_sha256", "action_sha256", "control_sha256"):
                    if len(row[name]) != 64:
                        raise ValueError(f"Missing audit digest: {name}")
                if not 0 < row["executed_steps"] <= cfg["step_limit"] or row["step_limit"] != cfg["step_limit"]:
                    raise ValueError("Unexpected action count or step limit")
            record["runs"].append({"round": index, "summary_path": str(summary_path), "run_id": summary["run_id"]})
            summaries.append(summary)
            resources = [json.loads(line) for line in (output / "resources.jsonl").read_text().splitlines() if line.endswith("}")]
            aggregate(record, summaries, resources)
            atomic_json(output / "series.json", record)
            print(f"Round {index + 1}/{args.rounds}: {record['completed_episodes']}/{record['requested_episodes']} episodes", flush=True)
        record["complete"] = record["completed_episodes"] == record["requested_episodes"]
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError, KeyboardInterrupt) as error:
        record["error"] = f"{type(error).__name__}: {error}"
    finally:
        # The launcher drains several separately owned process groups in turn.
        # Give it time to finish before killing it, or its children can survive.
        stop(process, grace_s=120)
        stop(monitor)
        gpu_thread.join(timeout=5)
        finished.set()
        sampler.join(timeout=5)
        resources = [json.loads(line) for line in (output / "resources.jsonl").read_text().splitlines()]
        aggregate(record, summaries, resources)
        samples = [json.loads(line) for line in (output / "gpu.jsonl").read_text().splitlines()]
        record["gpu_total"] = gpu_summary(samples, record["started_unix_s"], time.time())
        record["gpu_total_scope"] = "entire series, including between-round gaps, loading and cleanup"
        record["ended_unix_s"] = time.time()
        atomic_json(output / "series.json", record)
        signal.signal(signal.SIGTERM, previous)
    if not record["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
