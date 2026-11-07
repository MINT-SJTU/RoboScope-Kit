"""Launch the unmodified pinned ManiSkill PPO evaluator and retain smoke evidence.

This launcher validates immutable input hashes and starts the upstream file as a
separate Python process. It does not patch or instrument upstream code. Its reset
schedule is the upstream default, not the Kit's frozen per-episode seed list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

from ppo_model import checkpoint_records

SOURCE_REVISION = "baab60ede2e89167c1b7aaed41a9aa8e690a9d1e"
PPO_SOURCE_GIT_BLOB = "3eae2120f84445a0938efe7be41db79d2a821dac"
PHYSX_SHA256 = "4c582a16509a71faf5592fe9708586dfcc7ab61ae932eabc1ddd81f290818706"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-id", default="PickCube-v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-eval-envs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.num_eval_envs < 1:
        raise ValueError("num-eval-envs must be positive")
    runtime = args.runtime.resolve(strict=True)
    models = checkpoint_records(args.checkpoint)
    if args.task_id not in models:
        raise ValueError(f"No checkpoint configured for {args.task_id}")
    model = models[args.task_id]
    checkpoint = model["checkpoint"]
    horizon = int(model["max_episode_steps"])
    python = runtime / "env/bin/python"
    entrypoint = runtime / "ManiSkill/examples/baselines/ppo/ppo.py"
    source = entrypoint.read_bytes()
    actual_blob = hashlib.sha1(f"blob {len(source)}\0".encode() + source).hexdigest()
    if actual_blob != PPO_SOURCE_GIT_BLOB:
        raise RuntimeError("The original pinned PPO entrypoint has changed")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != model["sha256"]:
        raise RuntimeError("The checkpoint differs from the declared PPO model")
    runtime_home = runtime / "home"
    physics = runtime_home / ".sapien/physx/105.1-physx-5.3.1.patch0/libPhysXGpu_64.so"
    if hashlib.sha256(physics.read_bytes()).hexdigest() != PHYSX_SHA256:
        raise RuntimeError("The install-local PhysX GPU library has changed")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    command = [
        str(python), "-u", str(entrypoint),
        "--evaluate", "--checkpoint", str(checkpoint),
        "--env-id", args.task_id, "--control-mode", model["control_mode"],
        "--num-eval-envs", str(args.num_eval_envs), "--num-eval-steps", str(horizon),
        "--num-envs", "1", "--num-steps", str(horizon), "--total-timesteps", str(horizon),
        "--num-minibatches", "1", "--eval-freq", "25",
        "--eval-reconfiguration-freq", "1", "--seed", str(args.seed),
        "--no-capture-video", "--no-save-model", "--no-track",
    ]
    environment = os.environ.copy()
    environment.update({
        "HOME": str(runtime_home), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "MS_ASSET_DIR": str(runtime / "assets"),
    })
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    print(f"Running unmodified upstream PPO evaluator; log: {output / 'upstream.log'}", flush=True)
    started = time.perf_counter()
    with (output / "upstream.log").open("w") as log:
        completed = subprocess.run(
            command, cwd=output, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    process_wall_s = time.perf_counter() - started
    log_text = (output / "upstream.log").read_text(errors="replace")
    coverage = re.findall(r"Evaluated\s+(\d+)\s+steps resulting in\s+(\d+)\s+episodes", log_text)
    expected_steps = horizon * args.num_eval_envs
    evaluated = [(int(steps), int(episodes)) for steps, episodes in coverage]
    valid = (
        completed.returncode == 0 and evaluated == [(expected_steps, args.num_eval_envs)]
        and "Running evaluation" in log_text and "Running training" not in log_text
        and "model saved to" not in log_text and not (output / "runs").exists()
    )
    metrics = {
        name: float(value)
        for name, value in re.findall(r"eval_([A-Za-z_]+)_mean=([-+0-9.eE]+)", log_text)
    }
    result = {
        "formal": False, "mode": "unmodified_upstream_entrypoint_smoke",
        "passed": valid, "exit_code": completed.returncode,
        "entrypoint": str(entrypoint), "source_revision": SOURCE_REVISION,
        "entrypoint_git_blob_sha1": actual_blob, "weight_sha256": model["sha256"],
        "task_id": args.task_id, "control_mode": model["control_mode"],
        "runtime_home": str(runtime_home), "physx_gpu_library_sha256": PHYSX_SHA256,
        "num_eval_envs": args.num_eval_envs, "control_steps_per_env": horizon,
        "completed_evaluations": evaluated, "metrics": metrics,
        "process_wall_s": process_wall_s,
        "process_wall_episodes_per_s": args.num_eval_envs / process_wall_s if valid else None,
        "process_wall_steps_per_s": expected_steps / process_wall_s if valid else None,
        "timing_boundary": (
            "Child Python process startup through exit, including imports, model loading, "
            f"one auxiliary 1-env allocation, {args.num_eval_envs}-env evaluation allocation, "
            "reset/rollout/close. "
            "This boundary differs from Kit formal_wall_s and the native harness."
        ),
        "seed_protocol": (
            "Upstream seeds random/NumPy/Torch and initial env resets with the scalar seed, "
            "then performs an unseeded eval reset before its first evaluation. The native "
            "ManiSkillVectorEnv uses its original auto-reset behavior at the horizon."
        ),
        "equal_inputs_to_kit_verified": False,
        "comparison_scope": (
            "Genuine upstream GPU-vectorized evaluator smoke and supplemental resource baseline; "
            "not the frozen Kit seeds 0..99, not a paired correctness or statistical-equivalence claim."
        ),
        "training_performed": False if valid else None,
        "video_or_trajectory_recording_enabled": False,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if valid else (completed.returncode or 1)


if __name__ == "__main__":
    raise SystemExit(main())
