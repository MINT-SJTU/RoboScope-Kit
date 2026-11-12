"""Run the official OpenVLA evaluator with independent, explicit measurements."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

from instrument import run_child, write_json

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
REVISIONS = {
    "libero_spatial": "962318cec55ac10993ff0f5f43eda9a270b4c873",
    "libero_object": "287d6cfdf12d07b1449505f66d9bf3550257e9b3",
    "libero_goal": "fa5ae1e7509348889295bba8e08621d8b55e9baf",
    "libero_10": "80970322773f81baa2e22fe495d0487b93a05cfa",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-inventory", type=Path, required=True)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--task-id", type=int, action="append")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--env-seed", type=int)
    parser.add_argument("--video", choices=("on", "off"), default="on")
    parser.add_argument(
        "--profile-name", choices=("upstream-reference", "matched-serial"), default="upstream-reference"
    )
    parser.add_argument("--repeat-id", default="0")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument(
        "--diagnostic-trace", action="store_true", help="save per-step images, processor tensors and actions"
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("runtime", "checkpoint", "checkpoint_inventory", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if not 1 <= args.episodes_per_task <= 50:
        parser.error("--episodes-per-task must be between 1 and 50")
    if args.cpu_threads < 1 or args.sample_interval <= 0:
        parser.error("thread count and sampling interval must be positive")
    if args.task_id is not None and (
        len(set(args.task_id)) != len(args.task_id) or any(i not in range(10) for i in args.task_id)
    ):
        parser.error("--task-id must contain unique task ids in 0..9")
    if args.profile_name == "upstream-reference" and (args.env_seed is not None or args.video != "on"):
        parser.error(
            "upstream-reference preserves env seed 0 and videos on; use matched-serial for deviations"
        )
    if args.profile_name == "matched-serial" and (args.env_seed != 7 or args.video != "off"):
        parser.error("matched-serial requires --env-seed 7 --video off")
    return args


def command_output(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def read_jsonl(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def distribution(values):
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    ordered = sorted(values)

    def percentile(q):
        position = (len(ordered) - 1) * q
        lo = int(position)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)

    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": max(values),
    }


def sample_resources(args, child, stop, rows):
    import psutil

    fields = (
        "unix_s",
        "monotonic_s",
        "query_elapsed_s",
        "phase",
        "gpu_uuid",
        "gpu_index",
        "gpu_memory_mib",
        "gpu_util_pct",
        "process_tree_rss_bytes",
        "process_tree_cpu_s",
        "host_available_ram_bytes",
        "cgroup_memory_bytes",
        "error",
    )
    with (args.output_dir / "resources.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        while not stop.is_set():
            started = time.monotonic()
            row = {"unix_s": time.time(), "monotonic_s": started, "phase": "startup"}
            try:
                phase_path = args.output_dir / "phase.json"
                if phase_path.is_file():
                    row["phase"] = json.loads(phase_path.read_text())["phase"]
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={args.gpu}",
                        "--query-gpu=uuid,index,memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                )
                uuid, index, memory, utilization = result.stdout.strip().split(",")
                row.update(
                    gpu_uuid=uuid.strip(),
                    gpu_index=index.strip(),
                    gpu_memory_mib=float(memory),
                    gpu_util_pct=float(utilization),
                )
                processes = [psutil.Process(child.pid)]
                processes.extend(processes[0].children(recursive=True))
                row["process_tree_rss_bytes"] = sum(p.memory_info().rss for p in processes)
                cpu = [p.cpu_times() for p in processes]
                row["process_tree_cpu_s"] = sum(p.user + p.system for p in cpu)
                row["host_available_ram_bytes"] = psutil.virtual_memory().available
                memory_path = Path("/sys/fs/cgroup/memory.current")
                if memory_path.is_file():
                    row["cgroup_memory_bytes"] = int(memory_path.read_text())
            except (OSError, ValueError, subprocess.SubprocessError, psutil.Error) as error:
                row["error"] = f"{type(error).__name__}: {error}"
            row["query_elapsed_s"] = time.monotonic() - started
            rows.append(row)
            writer.writerow(row)
            file.flush()
            stop.wait(args.sample_interval)


def summarize(args, returncode, elapsed, samples):
    records = read_jsonl(args.output_dir / "episodes.jsonl")
    requests = read_jsonl(args.output_dir / "requests.jsonl")
    expected = {
        f"task-{task:04d}-episode-{episode:04d}"
        for task in (args.task_id if args.task_id is not None else range(10))
        for episode in json.loads(os.environ["DIAGNOSTIC_SELECTION"])[str(task)]
    }
    counts = Counter(row["episode_id"] for row in records)
    duplicates = sorted(key for key, value in counts.items() if value != 1)
    missing, unexpected = sorted(expected - set(counts)), sorted(set(counts) - expected)
    timings_path = args.output_dir / "native-timings.json"
    timings = json.loads(timings_path.read_text()) if timings_path.is_file() else {}
    duration = timings.get("evaluation_s")
    successes = sum(row["success"] for row in records)
    steps = sum(row["policy_steps"] for row in records)
    errors = read_jsonl(args.output_dir / "errors.jsonl")
    complete = returncode == 0 and not missing and not duplicates and not unexpected
    valid = complete and not errors
    per_task = {}
    for task in args.task_id if args.task_id is not None else range(10):
        task_rows = [row for row in records if row["task_id"] == task]
        per_task[str(task)] = {
            "completed": len(task_rows),
            "successes": sum(r["success"] for r in task_rows),
            "success_rate": sum(r["success"] for r in task_rows) / len(task_rows) if task_rows else None,
        }
    phases = defaultdict(list)
    for sample in samples:
        if "gpu_memory_mib" in sample:
            phases[sample["phase"]].append(sample)
    resources = {
        name: {
            "samples": len(values),
            "observed_peak_gpu_memory_mib": max(v["gpu_memory_mib"] for v in values),
            "mean_gpu_util_pct": statistics.fmean(v["gpu_util_pct"] for v in values),
        }
        for name, values in phases.items()
    }
    summary = {
        "suite": args.suite,
        "profile": args.profile_name,
        "repeat_id": args.repeat_id,
        "exit_code": returncode,
        "complete_unique_coverage": complete,
        "execution_valid": valid,
        "diagnostic_trace": args.diagnostic_trace,
        "performance_evidence": valid and not args.diagnostic_trace,
        "expected_episodes": len(expected),
        "completed_episodes": len(records),
        "unique_episodes": len(counts),
        "missing": missing,
        "duplicates": duplicates,
        "unexpected": unexpected,
        "successes": successes,
        "observed_success_rate": successes / len(records) if records else None,
        "policy_steps": steps,
        "settling_steps": sum(row["settling_steps"] for row in records),
        "errors": len(errors),
        "oom_errors": sum(
            "out of memory" in row["message"].lower() or row["class"] == "OutOfMemoryError" for row in errors
        ),
        "retries": 0,
        "per_task": per_task,
        "phase_timings": timings,
        "formal_episodes_per_s": (
            len(records) / duration if valid and duration and not args.diagnostic_trace else None
        ),
        "formal_steps_per_s": steps / duration if valid and duration and not args.diagnostic_trace else None,
        "formal_boundary": (
            "before first task creation through last task upstream logs; excludes model/processor loading"
        ),
        "child_process_elapsed_s": elapsed,
        "resources_by_phase": resources,
        "resource_samples": len(samples),
        "resource_error_samples": sum(bool(row.get("error")) for row in samples),
        "sampler_interval_requested_s": args.sample_interval,
        "observed_sample_gap_s": distribution(
            [b["monotonic_s"] - a["monotonic_s"] for a, b in zip(samples, samples[1:], strict=False)]
        ),
    }
    successful_requests = [row["service_ms"] for row in requests if row["success"]]
    latency = {
        "service_ms": distribution(successful_requests),
        "service_after_first_request_ms": distribution(successful_requests[1:]),
        "first_request_ms": successful_requests[0] if successful_requests else None,
        "failed_requests": sum(not row["success"] for row in requests),
        "boundary": (
            "upstream get_action: crop, prompt, processor, H2D, predict_action and CPU action; "
            "excludes JPEG/resize"
        ),
        "synchronization": (
            "no additional CUDA synchronization; native predict_action returns a CPU NumPy action"
        ),
        "first_input_digest_cost": (
            "first request of each episode records input hashes inside the service boundary"
        ),
        "diagnostic_trace": args.diagnostic_trace,
        "queue_wait_ms": None,
        "transport_ms": None,
        "unavailable_reason": "native serial path has no socket queue",
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "latency.json", latency)
    print(json.dumps(summary, indent=2))
    return 0 if valid else 1


def main():
    args = parse_args()
    if args.child:
        return run_child(args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("--output-dir must be new or empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory = json.loads(args.checkpoint_inventory.read_text())
    if (args.checkpoint / "VERIFIED.json").exists() and inventory["checkpoint"] != str(args.checkpoint):
        # A relocated diagnostic cache must revalidate bytes before recording new stat identity.
        inventory["original_checkpoint"] = inventory["checkpoint"]
        for name, value in inventory["files"].items():
            h = hashlib.sha256()
            with (args.checkpoint / name).open("rb") as stream:
                for data in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                    h.update(data)
            if h.hexdigest() != value["sha256"]:
                raise ValueError(f"Relocated checkpoint checksum mismatch: {name}")
            stat = (args.checkpoint / name).stat()
            value.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        inventory["checkpoint"] = str(args.checkpoint)
        write_json(args.output_dir / "relocated-checkpoint-inventory.json", inventory)
    if inventory["revision"] != REVISIONS[args.suite] or inventory["checkpoint"] != str(args.checkpoint):
        raise ValueError("Checkpoint inventory does not match the suite, revision or path")
    for name, value in inventory["files"].items():
        stat = (args.checkpoint / name).stat()
        if stat.st_size != value["size"] or stat.st_mtime_ns != value["mtime_ns"]:
            raise ValueError(f"Immutable checkpoint changed: {name}; regenerate and verify the inventory")
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "PYTHONHASHSEED": str(args.seed),
        "PYTHONPATH": f"{args.runtime / 'openvla'}:{args.runtime / 'LIBERO'}",
        "LIBERO_CONFIG_PATH": str(args.runtime / "config"),
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "TF_FORCE_GPU_ALLOW_GROWTH": "true",
        "OMP_NUM_THREADS": str(args.cpu_threads),
        "MKL_NUM_THREADS": str(args.cpu_threads),
        "TOKENIZERS_PARALLELISM": "false",
    }
    manifest = {
        "arm": "native",
        "profile": args.profile_name,
        "suite": args.suite,
        "repeat_id": args.repeat_id,
        "resolved_config": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "argv": sys.argv,
        "checkpoint_inventory": inventory,
        "build": json.loads((args.runtime / "build-manifest.json").read_text()),
        "hardware": command_output(
            ["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total", "--format=csv"]
        ),
        "initial_gpu_processes": command_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"]
        ),
        "mps_processes": command_output(["pgrep", "-x", "nvidia-cuda-mps"]),
        "platform": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cgroup_limits": {
            name: Path(f"/sys/fs/cgroup/{name}").read_text().strip()
            for name in ("cpu.max", "memory.max")
            if Path(f"/sys/fs/cgroup/{name}").is_file()
        },
        "environment": {
            key: environment[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "PYTHONHASHSEED",
                "PYTHONPATH",
                "LIBERO_CONFIG_PATH",
                "MUJOCO_GL",
                "PYOPENGL_PLATFORM",
                "TF_FORCE_GPU_ALLOW_GROWTH",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "runner_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(__file__).with_name("instrument.py"),
                Path(__file__).with_name("trace_artifacts.py"),
            )
        },
        "deviations": [
            "observational evaluator hooks saved in instrumentation.patch",
            "pinned external model code revision",
            "TensorFlow GPU memory growth enabled",
            "declared CPU thread limits",
            "first-input content digests",
            *(["per-step diagnostic trace copies and disk writes"] if args.diagnostic_trace else []),
        ],
        "sample_interval_s": args.sample_interval,
        "profiled": False,
        "concurrency": 1,
        "replicas": 1,
    }
    write_json(args.output_dir / "run-manifest.json", manifest)
    stop, samples = threading.Event(), []
    started = time.monotonic()
    command = [
        str(args.runtime / "env/bin/python"),
        "-u",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--child",
    ]
    with (args.output_dir / "upstream.stdout.log").open("w") as log:
        child = subprocess.Popen(
            command, cwd=args.output_dir, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
        sampler = threading.Thread(target=sample_resources, args=(args, child, stop, samples), daemon=True)
        sampler.start()
        try:
            returncode = child.wait()
        except BaseException:
            child.terminate()
            child.wait()
            raise
        finally:
            elapsed = time.monotonic() - started
            stop.set()
            sampler.join(timeout=10)
    return summarize(args, returncode, elapsed, samples)


if __name__ == "__main__":
    raise SystemExit(main())
