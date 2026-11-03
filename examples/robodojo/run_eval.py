"""Run the native Simate server/client and collect a checked RoboDojo baseline."""

import argparse
import contextlib
import csv
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import tomllib
import uuid
from pathlib import Path

from apply_runtime_patch import patch_metadata

HERE = Path(__file__).resolve().parent


def validate_episode_scope(simulator_python, root, cfg, seed, episodes):
    """Reject upstream EVAL_NUM clamping and missing layouts before GPU startup."""
    task_limit = json.loads(subprocess.check_output(
        [str(simulator_python), "-c",
         "import json,sys,yaml; from pathlib import Path; "
         "info=yaml.safe_load(Path(sys.argv[1]).read_text()); "
         "print(json.dumps(info['tasks'].get(sys.argv[2], {}).get("
         "'eval_nums', info.get('common', {}).get('eval_nums', 50))))",
         str(root / "task/RoboDojo/config/_task.yml"), cfg["task"]],
        text=True,
    ))
    if type(task_limit) is not int or task_limit < episodes:
        raise ValueError(
            f"Requested {episodes} episodes exceed upstream task limit {task_limit}; "
            "apply the declared episode-count patch before running the 50-layout scope"
        )
    layout_root = root / "Assets/Eval_Layout/RoboDojo" / cfg["env_cfg"] / str(seed)
    missing = [
        index for index in range(episodes)
        if not (layout_root / f"{cfg['task']}_{index}.json").is_file()
    ]
    if missing:
        raise ValueError(f"Missing fixed evaluation layouts: {missing}")
    return {"task_limit": task_limit, "requested_episodes": episodes, "layout_ids": list(range(episodes))}


def stop(process, *, grace_s=20):
    if process is None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGINT)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=grace_s)
    # A shell may exit before its children; clean up the owned process group too.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def sample_gpu(process, path):
    with path.open("w") as output:
        for line in process.stdout:
            row = next(csv.reader([line]))
            if len(row) == 3:
                try:
                    record = dict(
                        zip(("gpu", "utilization_percent", "memory_mib"), map(float, row), strict=True)
                    )
                except ValueError:
                    continue
                output.write(json.dumps({"time_unix_s": time.time(), **record}) + "\n")
                output.flush()


def gpu_summary(samples, start, end):
    selected = [s for s in samples if start <= s["time_unix_s"] <= end]
    if not selected:
        return None
    return {
        "samples": len(selected),
        "mean_utilization_percent": sum(s["utilization_percent"] for s in selected) / len(selected),
        "mean_memory_mib": sum(s["memory_mib"] for s in selected) / len(selected),
        "peak_memory_mib": max(s["memory_mib"] for s in selected),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    # Defaults are the conda environments created by the upstream installers.
    parser.add_argument("--simulator-env", type=Path, default=Path.home() / "miniconda3/envs/RoboDojo")
    parser.add_argument("--policy-env", type=Path, default=Path.home() / "miniconda3/envs/simate-beta-eval")
    parser.add_argument("--vulkan-icd", type=Path, help="NVIDIA Vulkan ICD JSON on this host")
    parser.add_argument("--profile", type=Path, default=HERE / "recipe.toml")
    parser.add_argument("--episodes", type=int, choices=range(1, 51), default=1)
    parser.add_argument("--label", required=True)
    parser.add_argument("--mode", choices=["native", "measured"], default="measured")
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], default=None)
    parser.add_argument(
        "--num-envs", type=int, choices=[1, 2], default=1,
        help="one for the declared baseline; two for the upstream vectorization reference",
    )
    parser.add_argument("--timeout-s", type=float, default=7200)
    parser.add_argument("--metrics-root", type=Path, default=Path("/workspace/robodojo-metrics"))
    parser.add_argument("--summary-path", type=Path, help="exclusive result pointer for a series runner")
    parser.add_argument("--async-video", action="store_true", help="enable the experimental bounded video pipe")
    parser.add_argument("--control-state-reuse", action="store_true", help="reuse control expansion within one dual-arm call")
    parser.add_argument("--execution-profile", action="store_true", help="diagnostic host timings; separate from acceptance runs")
    parser.add_argument("--python-profile", action="store_true", help="also save cProfile; requires execution profiling")
    parser.add_argument("--unpaced-render", action="store_true", help="remove wall-clock pacing during fixed-step renders")
    parser.add_argument("--physics-threads", type=int, choices=[1, 4, 8], help="fixed PhysX CPU worker limit; no automatic search")
    args = parser.parse_args()
    if os.environ.get("OMNI_KIT_ACCEPT_EULA", "").upper() not in {"YES", "Y"}:
        parser.error(
            "Isaac Sim is NVIDIA software under its own EULA "
            "(https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html). "
            "Read it and export OMNI_KIT_ACCEPT_EULA=YES to accept it yourself; this recipe does not accept it for you."
        )
    if (args.execution_profile and args.mode != "measured") or (args.python_profile and not args.execution_profile):
        parser.error("Execution profiling requires measured mode; Python profiling also requires --execution-profile")
    if args.summary_path is not None and args.summary_path.exists():
        parser.error("--summary-path already exists; refusing to replace previous evidence")
    if not args.label.replace("-", "").replace("_", "").isalnum():
        parser.error("--label must contain only letters, numbers, hyphens or underscores")
    if args.episodes < args.num_envs:
        parser.error("Every environment must receive a real episode")
    vulkan_icd = args.vulkan_icd or next(
        (path for path in (
            Path("/etc/vulkan/icd.d/nvidia_icd.json"),
            Path("/usr/share/vulkan/icd.d/nvidia_icd.json"),
        ) if path.is_file()), None,
    )
    if vulkan_icd is None or not vulkan_icd.is_file():
        parser.error("Set --vulkan-icd to the host's existing NVIDIA Vulkan ICD JSON")
    vulkan_icd = vulkan_icd.resolve()
    root = args.root.resolve()
    simulator_env, policy_env = args.simulator_env.absolute(), args.policy_env.absolute()
    profile = tomllib.loads(args.profile.read_text(encoding="utf-8"))
    cfg = profile["protocol"]
    if (cfg["policy"], cfg["task"], cfg["env_cfg"]) != ("Simate_beta", "stack_blocks", "arx_x5"):
        parser.error("This recipe is scoped to Simate_beta / stack_blocks / arx_x5")
    if args.seed is None:
        args.seed = cfg["seed"]
    runtime_versions = json.loads(subprocess.check_output(
        [str(simulator_env / "bin/python"), "-c",
         "import importlib.metadata as m,json,platform; "
         "print(json.dumps(dict(python=platform.python_version(), "
         "isaacsim=m.version('isaacsim'), torch=m.version('torch'))))"], text=True,
    ))
    expected_runtime = profile.get("runtime", {"isaacsim": "5.1.0.0"})
    if runtime_versions["isaacsim"] != expected_runtime["isaacsim"]:
        raise ValueError("Simulator version does not match the declared recipe")
    if expected_runtime.get("simulator_torch", runtime_versions["torch"]) != runtime_versions["torch"]:
        raise ValueError("Simulator PyTorch version does not match the declared recipe")
    if not runtime_versions["python"].startswith(expected_runtime["simulator_python"] + "."):
        raise ValueError("Simulator Python version does not match the declared recipe")
    patch_states = {
        "measurement": "after" if args.mode == "measured" else "before",
    }
    if args.control_state_reuse:
        patch_states["control-state"] = "after"
    if args.async_video:
        patch_states["async-video"] = "after"
    if args.unpaced_render:
        patch_states["unpaced-render"] = "after"
    if args.physics_threads is not None:
        patch_states["physics-threads"] = "after"
    for key in ("compatibility_patch", "dependency_patch", "episode_count_patch"):
        if key in expected_runtime:
            patch_states[expected_runtime[key]] = "after"
    runtime_patches = {}
    allowed_changes = set()
    for name, expected in patch_states.items():
        state = subprocess.check_output(
            [sys.executable, str(HERE / "apply_runtime_patch.py"), "check", str(root), "--patch", name],
            text=True,
        ).strip()
        if state != f"{name} patch: {expected}":
            raise ValueError(f"Unexpected runtime patch state: {state}")
        manifest = patch_metadata(name)
        runtime_patches[name] = {"state": expected, "sha256": manifest["sha256"]}
        if expected == "after":
            allowed_changes.update((root / path).resolve() for path in manifest["files"])

    source_paths = {
        "robodojo": root,
        "xpolicylab": root / "XPolicyLab",
        "isaaclab": root / "third_party/IsaacLab",
        "curobo": root / "third_party/curobo",
    }
    source_revisions = {}
    for name, path in source_paths.items():
        # Record what the upstream installer checked out; its submodules follow their main branches.
        source_revisions[name] = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
        ).strip()
        changed = subprocess.check_output(
            ["git", "-C", str(path), "diff", "--name-only", "--ignore-submodules=all", "HEAD"],
            text=True,
        ).splitlines()
        added = subprocess.check_output(
            ["git", "-C", str(path), "ls-files", "--others", "--exclude-standard"], text=True,
        ).splitlines()
        changed.extend(p for p in added if Path(p).suffix in {".py", ".sh", ".yaml", ".yml", ".kit"})
        unexpected = [p for p in changed if (path / p).resolve() not in allowed_changes]
        if unexpected:
            raise ValueError(f"Unrecognized runtime edits in {name}: {unexpected}")
    subprocess.run(
        ["git", "-C", str(root / "XPolicyLab"), "diff", "--exit-code", "HEAD", "--", "policy/Simate_beta"],
        check=True,
    )
    episode_scope = validate_episode_scope(
        simulator_env / "bin/python", root, cfg, args.seed, args.episodes,
    )
    print(f"Episode scope preflight: {json.dumps(episode_scope)}", flush=True)
    run_id = f"{args.label}-{uuid.uuid4().hex[:12]}"
    output = root.parent / "evidence" / "evaluation" / run_id
    output.mkdir(parents=True)
    metrics_path = args.metrics_root.resolve() / run_id
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        port = available.getsockname()[1]
    env = dict(os.environ)
    env.update(
        PYTHONPATH=f"{root}:{root / 'XPolicyLab'}",
        PYTHONUNBUFFERED="1",
        EVAL_NUM=str(args.episodes),
        ROBODOJO_RUN_ID=run_id,
        CUDA_VISIBLE_DEVICES="0",
        # The caller accepted the NVIDIA EULA (checked in main); PRIVACY_CONSENT stays theirs to set.
        ACCEPT_EULA="Y",
        OMNI_KIT_ALLOW_ROOT="1",
        PYTHONNOUSERSITE="1",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="1",
        CUDA_HOME="/usr/local/cuda",
        VK_ICD_FILENAMES=str(vulkan_icd),
        VK_DRIVER_FILES=str(vulkan_icd),
    )
    env["PATH"] = f"{simulator_env / 'bin'}:/usr/local/cuda/bin:{env['PATH']}"
    env.pop("ROBODOJO_METRICS_DIR", None)
    env.pop("ROBODOJO_LAYOUT_IDS", None)
    env.pop("ROBODOJO_SCOPED_RESET", None)
    env.pop("ROBODOJO_FATAL_RESTART_COUNT", None)
    env.pop("ROBODOJO_EXECUTION_PROFILE", None)
    env.pop("ROBODOJO_PYTHON_PROFILE", None)
    if args.execution_profile:
        env["ROBODOJO_EXECUTION_PROFILE"] = "1"
    if args.python_profile:
        env["ROBODOJO_PYTHON_PROFILE"] = "1"
    env.pop("ROBODOJO_UNPACED_RENDER", None)
    env.pop("ROBODOJO_PHYSICS_THREADS", None)
    if args.physics_threads is not None:
        env["ROBODOJO_PHYSICS_THREADS"] = str(args.physics_threads)
    if args.unpaced_render:
        env["ROBODOJO_UNPACED_RENDER"] = "1"
    env.pop("ROBODOJO_CONTROL_STATE_REUSE", None)
    if args.control_state_reuse:
        env["ROBODOJO_CONTROL_STATE_REUSE"] = "1"
    env.pop("ROBODOJO_ASYNC_VIDEO", None)
    if args.async_video:
        env["ROBODOJO_ASYNC_VIDEO"] = "1"
    if args.mode == "measured":
        env["ROBODOJO_METRICS_DIR"] = str(metrics_path)
    server_command = [
        "bash",
        str(root / "XPolicyLab/policy/Simate_beta/setup_eval_policy_server.sh"),
        "RoboDojo",
        cfg["task"],
        str(root / profile["dataset"]["checkpoint"]),
        cfg["env_cfg"],
        cfg["action_type"],
        str(args.seed),
        "0",
        str(policy_env),
        str(port),
        "127.0.0.1",
    ]
    client_command = [
        str(simulator_env / "bin/python"),
        "-u",
        expected_runtime.get("entrypoint", "src/eval_client/main.py"),
        "--task_name",
        cfg["task"],
        "--env_cfg_type",
        cfg["env_cfg"],
        "--num_envs",
        str(args.num_envs),
        "--enable_cameras",
        "--kit_args",
        "--allow-root --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera",
        "--device_id",
        "0",
        "--policy_name",
        cfg["policy"],
        "--port",
        str(port),
        "--protocol",
        "ws",
        "--policy_server_url",
        f"ws://127.0.0.1:{port}",
        "--additional_info",
        args.label,
        "--seed",
        str(args.seed),
        "--host",
        "127.0.0.1",
        "--headless",
    ]
    record = {
        "run_id": run_id,
        "mode": args.mode,
        "profile_sha256": hashlib.sha256(args.profile.read_bytes()).hexdigest(),
        "source_revisions": source_revisions,
        "requested_episodes": args.episodes,
        "episode_scope": episode_scope,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "async_video": args.async_video,
        "control_state_reuse": args.control_state_reuse,
        "unpaced_render": args.unpaced_render,
        "physics_threads": args.physics_threads,
        "execution_profile": args.execution_profile,
        "python_profile": args.python_profile,
        "simulator_env": str(simulator_env),
        "policy_env": str(policy_env),
        "vulkan_icd": str(vulkan_icd),
        "runtime_versions": runtime_versions,
        "runtime_patches": runtime_patches,
        "compatibility_patch_sha256": runtime_patches.get(
            expected_runtime.get("compatibility_patch"), {}
        ).get("sha256"),
        "server_command": server_command,
        "client_command": client_command,
        "complete": False,
    }
    (output / "invocation.json").write_text(json.dumps(record, indent=2) + "\n")
    server = client = monitor = None
    monitor_thread = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    previous_sigterm = signal.signal(signal.SIGTERM, interrupted)
    started = time.time()
    print(f"Run {run_id}; evidence: {output}", flush=True)
    try:
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
                "-lms",
                "500",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        monitor_thread = threading.Thread(
            target=sample_gpu, args=(monitor, output / "gpu.jsonl"), daemon=True
        )
        monitor_thread.start()
        with (output / "server.log").open("w") as server_log, (output / "client.log").open("w") as client_log:
            server = subprocess.Popen(
                server_command,
                cwd=root,
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + 1200
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Policy server exited {server.returncode}; see server.log")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Policy server readiness deadline exceeded") from None
                    time.sleep(1)
            ready = time.time()
            record["server_start_to_ready_s"] = ready - started
            client = subprocess.Popen(
                client_command,
                cwd=root,
                env=env,
                stdout=client_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            record["client_exit_code"] = client.wait(timeout=args.timeout_s)
            record["client_wall_s"] = time.time() - ready
            if client.returncode:
                raise RuntimeError(f"Simulator exited {client.returncode}; see client.log")
        result_path = (
            root
            / "eval_result/RoboDojo"
            / cfg["task"]
            / cfg["policy"]
            / cfg["env_cfg"]
            / f"{args.seed}_{args.label}"
            / run_id
            / "_result.json"
        )
        result = json.loads(result_path.read_text())
        details = list(result["details"].values())
        layout_ids = [int(item["layout_id"]) for item in details]
        record.update(
            result_path=str(result_path),
            completed_episodes=len(details),
            successes=sum(bool(item["success"]) for item in details),
            success_rate=sum(bool(item["success"]) for item in details) / len(details) if details else None,
            layout_ids=layout_ids,
            fixed_layouts_match=sorted(layout_ids) == list(range(args.episodes)),
        )
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        if len(details) != args.episodes or len(set(layout_ids)) != len(details):
            raise RuntimeError("Missing or duplicate episode outcomes")
        if not record["fixed_layouts_match"]:
            raise RuntimeError("Completed layouts differ from the predeclared comparison set")
        if args.mode == "measured":
            events = [
                json.loads(line) for p in metrics_path.glob("*.jsonl") for line in p.read_text().splitlines()
            ]
            starts = [e for e in events if e["event"] == "evaluation_start"]
            ends = [e for e in events if e["event"] == "evaluation_end"]
            if len(starts) != 1 or len(ends) != 1:
                raise RuntimeError("Missing evaluation phase markers or unexpected restart")
            record["evaluation_start_unix_s"] = starts[0]["time_unix_s"]
            record["evaluation_end_unix_s"] = ends[0]["time_unix_s"]
            record["evaluation_s"] = ends[0]["elapsed_s"]
            record["executed_steps"] = sum(item["executed_steps"] for item in details)
            record["phases_s"] = {}
            for event in events:
                if event["event"] == "phase":
                    phase = event["phase"]
                    record["phases_s"][phase] = record["phases_s"].get(phase, 0) + event["elapsed_s"]
            record["model_calls"] = {}
            for event in events:
                if event["event"] == "model_call":
                    values = record["model_calls"].setdefault(event["method"], {"count": 0, "total_s": 0})
                    values["count"] += 1
                    values["total_s"] += event["elapsed_s"]
            record["unstable_episodes"] = ends[0]["unstable"]
            record["abandoned_layout_ids"] = ends[0]["abandoned_layout_ids"]
        record["complete"] = True
    except (
        OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError, KeyboardInterrupt
    ) as error:
        record["complete"] = False
        record["error"] = f"{type(error).__name__}: {error}"
    finally:
        stop(client)
        stop(server)
        stopped = time.time()
        stop(monitor)
        if monitor_thread is not None:
            monitor_thread.join(timeout=5)
        record["total_wall_s"] = stopped - started
        record["started_unix_s"] = started
        record["ended_unix_s"] = stopped
        if metrics_path.exists():
            archive_start = time.perf_counter()
            shutil.copytree(metrics_path, output / "events")
            record["metrics_archive_s"] = time.perf_counter() - archive_start
            record["local_metrics_path"] = str(metrics_path)
        gpu_path = output / "gpu.jsonl"
        samples = (
            [json.loads(line) for line in gpu_path.read_text().splitlines()] if gpu_path.exists() else []
        )
        record["gpu_total"] = gpu_summary(samples, started, stopped)
        if record.get("complete"):
            record["total_episodes_per_s"] = args.episodes / record["total_wall_s"]
            if "evaluation_s" in record:
                record["evaluation_episodes_per_s"] = args.episodes / record["evaluation_s"]
                record["evaluation_steps_per_s"] = record["executed_steps"] / record["evaluation_s"]
                record["total_steps_per_s"] = record["executed_steps"] / record["total_wall_s"]
                record["gpu_evaluation"] = gpu_summary(
                    samples, record["evaluation_start_unix_s"], record["evaluation_end_unix_s"]
                )
        (output / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
        if args.summary_path is not None:
            with args.summary_path.open("x", encoding="utf-8") as pointer:
                json.dump({"summary_path": str(output / "summary.json")}, pointer)
                pointer.write("\n")
        print(json.dumps(record, indent=2), flush=True)
        signal.signal(signal.SIGTERM, previous_sigterm)
    if not record["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
