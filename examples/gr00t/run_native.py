# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Run the BEHAVIOR Challenge 2026 native evaluator against the GR00T server, without the kit.

For each requested task and instance index this runs the pinned BEHAVIOR-1K evaluator's own
command in a fresh process, the way the challenge documents it for one instance::

    python -m omnigibson.eval.eval --task-name <task> --instance-indices <i> --num-envs 1 \\
        --num-rollouts 1 --mode public_test \\
        --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper \\
        --host 127.0.0.1 --port <port> --output-dir <out>

against one ``serve_b1k.py`` server of the GR00T checkpoint that this script starts (the same
command, checkpoint and environment as ``gr00t_behavior_policy.py``). The evaluator writes
``<out>/json/<task>_<instance>_0.json``; this script also records each process's command, wall time
and exit code in ``<out>/runs.jsonl`` and the workload in ``<out>/run.json``, and fails if an
expected result is missing. ``compare_native.py`` pairs these results with a Kit run.

Usage (BEHAVIOR runtime Python; GR00T_ROOT, GR00T_PYTHON and HF_HOME as for the Policy)::

    <behavior>/env/bin/python run_native.py --behavior-runtime <behavior> \\
        --checkpoint <snapshot> --output-dir <out> [--task turning_on_radio ...] [--instances 1]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages" / "roboscope-kit" / "src"))

from roboscope_kit.runtime.benchmarks.behavior.tasks import (  # noqa: E402
    BEHAVIOR_1K_COMMIT,
    INSTANCES_PER_TASK,
    PYTHON_HASH_SEED,
    SUITE,
    SUITE_OBSERVATIONS,
    TASK_NAMES,
    instance_id,
)

# The challenge_2026 suite's observation, as gr00t_behavior_policy.py is evaluated.
ENV_WRAPPER = SUITE_OBSERVATIONS[SUITE].env_wrapper


def start_server(checkpoint: Path, port: int, log: Path) -> subprocess.Popen:
    root = Path(os.environ["GR00T_ROOT"])
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    server = subprocess.Popen(
        (
            os.environ["GR00T_PYTHON"],
            str(root / "scripts" / "b1k" / "serve_b1k.py"),
            "--model-path",
            str(checkpoint),
            "--modality-config-path",
            str(root / "examples" / "b1k" / "r1pro.py"),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ),
        cwd=root,
        env=env,
        stdout=log.open("ab"),
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + 900
    while True:
        if server.poll() is not None:
            raise SystemExit(f"GR00T server exited with {server.returncode}; see {log}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
                if response.status == 200:
                    return server
        except OSError:
            pass
        if time.monotonic() > deadline:
            server.terminate()
            raise SystemExit("GR00T server did not become healthy")
        time.sleep(2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--behavior-runtime", type=Path, required=True, help="build-runtime.sh install dir")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", action="append", choices=TASK_NAMES, help="default: every task")
    parser.add_argument("--instances", type=int, default=INSTANCES_PER_TASK, help="instance indices 0..N-1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.instances <= INSTANCES_PER_TASK:
        raise SystemExit(f"--instances must be in [1, {INSTANCES_PER_TASK}]")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"{args.output_dir} is not empty")
    runtime = args.behavior_runtime.resolve()
    # The evaluator runs in the BEHAVIOR checkout, so pass it absolute paths.
    args.output_dir = args.output_dir.resolve()
    tasks = args.task or list(TASK_NAMES)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir()
    workload = {
        "behavior_1k_commit": BEHAVIOR_1K_COMMIT,
        "python_hash_seed": PYTHON_HASH_SEED,
        "env_wrapper": ENV_WRAPPER,
        "checkpoint": str(args.checkpoint.resolve()),
        "gr00t_root": os.environ["GR00T_ROOT"],
        "tasks": tasks,
        "instances": args.instances,
        "instance_ids": [instance_id(index) for index in range(args.instances)],
    }
    (args.output_dir / "run.json").write_text(json.dumps(workload, indent=2) + "\n", encoding="utf-8")
    env = {
        **{k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")},
        "OMNIGIBSON_DATA_PATH": str(runtime / "data"),
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "PYTHONHASHSEED": str(PYTHON_HASH_SEED),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    server = start_server(args.checkpoint.resolve(), args.port, args.output_dir / "logs" / "gr00t-server.log")
    missing = []
    try:
        for index in range(args.instances):
            for task in tasks:
                command = (
                    str(runtime / "env" / "bin" / "python"),
                    "-m",
                    "omnigibson.eval.eval",
                    "--task-name",
                    task,
                    "--instance-indices",
                    str(index),
                    "--num-envs",
                    "1",
                    "--num-rollouts",
                    "1",
                    "--mode",
                    "public_test",
                    "--env-wrapper",
                    ENV_WRAPPER,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.port),
                    "--output-dir",
                    str(args.output_dir),
                )
                log = args.output_dir / "logs" / f"{task}_{instance_id(index)}.log"
                started = time.time()
                with log.open("wb") as handle:
                    returncode = subprocess.run(
                        command,
                        cwd=runtime / "src" / "BEHAVIOR-1K",
                        env=env,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                    ).returncode
                result = args.output_dir / "json" / f"{task}_{instance_id(index)}_0.json"
                entry = {
                    "task": task,
                    "instance_index": index,
                    "instance_id": instance_id(index),
                    "command": list(command),
                    "returncode": returncode,
                    "wall_s": time.time() - started,
                    "result": str(result) if result.is_file() else None,
                }
                with (args.output_dir / "runs.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry) + "\n")
                if not result.is_file():
                    missing.append(f"{task}/{instance_id(index)}")
                print(json.dumps(entry), flush=True)
    finally:
        server.terminate()
        server.wait(timeout=60)
    if missing:
        print(f"native results missing for: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
