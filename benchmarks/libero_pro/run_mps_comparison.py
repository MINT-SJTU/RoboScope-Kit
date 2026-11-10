"""Run matched, sequential LIBERO-Pro MPS arms using the Kit's owned MPS lifecycle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

from roboscope_kit.policy_config import load_policy_runtime_config
from roboscope_kit.runtime.benchmark.cuda_mps import running_mps_processes
from roboscope_kit.runtime.benchmarks.libero_pro.adapter import LIBERO_PRO_SUITES


def write_json(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def require_idle_gpu(gpu: str) -> None:
    if running_mps_processes():
        raise RuntimeError("an MPS session is already running; no existing service will be stopped")
    result = subprocess.run(
        ["nvidia-smi", "--id=" + gpu, "--query-compute-apps=pid", "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    if result.stdout.strip():
        raise RuntimeError("the selected GPU has compute clients; run the comparison on an idle GPU")


def command(args: argparse.Namespace, arm: str, *, warmup: bool) -> list[str]:
    output = args.output_dir / arm / ("warmup" if warmup else "run")
    result = [
        sys.executable, "-m", "roboscope_kit.runtime.benchmark",
        "--benchmark", "libero_pro", "--suite", args.suite,
        "--checkpoint-source", str(args.checkpoint_source),
        "--policy-config", str(args.output_dir / "policy.json"),
        "--benchmark-manifest", str(args.output_dir / "benchmark-manifest.json"),
        "--output-dir", str(output),
    ]
    if warmup:
        result += ["--quick-test"]
    else:
        result += ["--episodes", str(args.episodes), "--concurrency", str(args.concurrency),
                   "--replicas", str(args.replicas)]
    if arm == "mps-on":
        result.append("--cuda-mps")
    return result


def validate_arm(
    folder: Path, *, episodes: int, concurrency: int, replicas: int, mps: bool,
) -> tuple[dict, dict]:
    result = json.loads((folder / "run/result.json").read_text())
    metrics = json.loads((folder / "run/metrics.json").read_text())
    records = [record for path in (folder / "run/queue/completed").glob("*.json")
               for record in json.loads(path.read_text())["records"]]
    expected = {f"task-{task:04d}-episode-{episode:04d}"
                for task in range(10) for episode in range(episodes)}
    if len(records) != len(expected) or {r["record_id"] for r in records} != expected:
        raise RuntimeError(f"incomplete or duplicate episode coverage: {folder}")
    by_id = {r["record_id"]: r for r in records}
    official = result["result"]
    if (official["episodes"] != len(records)
            or sum(r["success"] for r in records) != official["successes"]
            or sum(r["steps"] for r in records) != official["total_steps"]):
        raise RuntimeError(f"record totals differ from result: {folder}")
    if (result["runtime"]["concurrency"] != concurrency
            or result["runtime"]["replicas"] != replicas
            or result["autotune"]["mode"] != "fixed"):
        raise RuntimeError(f"the evaluator did not use the requested fixed K/g: {folder}")
    if result["cuda_mps"]["enabled"] is not mps:
        raise RuntimeError(f"unexpected MPS mode: {folder}")
    if mps and result["cuda_mps"].get("max_observed_clients", 0) < replicas:
        raise RuntimeError(f"MPS did not observe all replicas: {folder}")
    seconds = metrics["evaluation_s"]
    if seconds <= 0:
        raise RuntimeError("formal evaluation time must be positive")
    return {
        "episodes": len(records), "successes": official["successes"],
        "steps": official["total_steps"], "formal_s": seconds,
        "episodes_per_s": len(records) / seconds,
        "steps_per_s": official["total_steps"] / seconds,
        "phase_timings": metrics, "cuda_mps": result["cuda_mps"],
        "recovery": result["recovery"], "resources": result["resources"],
    }, by_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--policy-config", type=Path, required=True)
    parser.add_argument("--checkpoint-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", choices=sorted(LIBERO_PRO_SUITES), default="libero_goal_object")
    parser.add_argument("--episodes", type=int, default=50, help="episodes per task (default: 50)")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--gpu", default="0", help="one GPU index or UUID")
    parser.add_argument("--on-first", action="store_true", help="reverse arm order for a repeat")
    parser.add_argument("--stage-timing", action="store_true", help="diagnostic timing in BOTH arms")
    parser.add_argument("--dry-run", action="store_true", help="print the plan; write/start nothing")
    args = parser.parse_args()
    if not 1 <= args.episodes <= 50 or not 1 <= args.replicas <= args.concurrency:
        parser.error("require 1..50 episodes and 1 <= replicas <= concurrency")
    if not args.gpu or "," in args.gpu or args.gpu in {"-1", "all"}:
        parser.error("select one GPU index or UUID")
    for key in ("benchmark_manifest", "policy_config", "checkpoint_source"):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    args.output_dir = args.output_dir.resolve()
    policy = load_policy_runtime_config(args.policy_config)
    manifest = json.loads(args.benchmark_manifest.read_text())
    if args.suite not in manifest["supported_suites"].get("libero_pro", []):
        parser.error("the supplied manifest does not support the selected LIBERO-Pro suite")
    manifest["runtime"] = {
        key: str((args.benchmark_manifest.parent / value).resolve(strict=True))
        for key, value in manifest["runtime"].items()
    }
    manifest["defaults"].update(policy_encoding="msgpack", persistent_policy_connection=False,
                                chunk_camera_rendering=True, save_videos=False)
    arms = ["mps-on", "mps-off"] if args.on_first else ["mps-off", "mps-on"]
    plan = {
        "kit_version": version("roboscope-kit"), "gpu": args.gpu,
        "suite": args.suite, "episodes_per_task": args.episodes,
        "k": args.concurrency, "g": args.replicas, "stage_timing": args.stage_timing,
        "manifest": manifest, "policy": policy.summary(),
        "commands": {arm: {phase: command(args, arm, warmup=phase == "warmup")
                           for phase in ("warmup", "formal")} for arm in arms},
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if shutil.which("nvidia-cuda-mps-control") is None:
        raise RuntimeError("nvidia-cuda-mps-control is required for the MPS arm")
    require_idle_gpu(args.gpu)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "plan.json", plan)
    write_json(args.output_dir / "benchmark-manifest.json", manifest)
    # summary() includes provenance path, which is not a field in the policy contract.
    policy_path = args.output_dir / "policy.json"
    policy_config = policy.summary()
    policy_config.pop("path")
    write_json(policy_path, policy_config)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("CUDA_MPS_") and key not in {"ROBOSCOPE_TIMING_DIR", "ROBOSCOPE_PROFILE"}}
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED="1")
    summaries, records = {}, {}
    for arm in arms:
        folder = args.output_dir / arm
        folder.mkdir()
        walls = {}
        for phase in ("warmup", "formal"):
            require_idle_gpu(args.gpu)
            if args.stage_timing:
                env["ROBOSCOPE_TIMING_DIR"] = str(folder / f"timing-{phase}")
            print(f"Starting {arm} {phase}", flush=True)
            started = time.perf_counter()
            with (folder / f"{phase}.log").open("x") as log:
                completed = subprocess.run(plan["commands"][arm][phase], env=env,
                                           stdout=log, stderr=subprocess.STDOUT, check=False)
            walls[phase] = {"wall_s": time.perf_counter() - started, "exit_code": completed.returncode}
            write_json(folder / f"{phase}-status.json", walls[phase])
            if completed.returncode:
                raise RuntimeError(f"{arm} {phase} failed; inspect {folder / (phase + '.log')}")
            require_idle_gpu(args.gpu)
        summaries[arm], records[arm] = validate_arm(
            folder, episodes=args.episodes, concurrency=args.concurrency,
            replicas=args.replicas, mps=arm == "mps-on",
        )
        summaries[arm]["CLI_wall_s"] = walls["formal"]["wall_s"]
        write_json(folder / "summary.json", summaries[arm])
    before, after = summaries["mps-off"], summaries["mps-on"]
    comparison = {
        "arms": summaries,
        "formal_throughput_speedup": after["episodes_per_s"] / before["episodes_per_s"],
        "actual_steps_per_s_speedup": after["steps_per_s"] / before["steps_per_s"],
        "CLI_wall_speedup": before["CLI_wall_s"] / after["CLI_wall_s"],
        "paired_success_differences": sum(records["mps-off"][key]["success"] !=
                                          records["mps-on"][key]["success"] for key in records["mps-off"]),
        "paired_step_differences": sum(records["mps-off"][key]["steps"] !=
                                       records["mps-on"][key]["steps"] for key in records["mps-off"]),
        "note": "CLI wall excludes warmup/setup; one pair does not prove equivalence or stable gain.",
    }
    write_json(args.output_dir / "comparison.json", comparison)
    print(f"Completed: {args.output_dir / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
