"""Run task-indexed PPO actors directly on native ManiSkill GPU tensors.

The manifest, checkpoint, seed groups and task rules match the SDK path. There
are no Policy RPCs. Formal time includes simulator lifecycle and bookkeeping;
loading is reported separately. The original evaluator has its own launcher.
"""

# ruff: noqa: E402 -- retain heavyweight import/setup cost separately
from __future__ import annotations

import time

PROCESS_ENTRY_STARTED = time.perf_counter()

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import torch
from ppo_model import load_model_bundle

PHYSX_SHA256 = "4c582a16509a71faf5592fe9708586dfcc7ab61ae932eabc1ddd81f290818706"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--suite", default="baseline_state_v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-home", type=Path)
    parser.add_argument("--episodes", type=int, help="Episodes per task; defaults to the manifest")
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--seed", type=int, help="Optional diagnostic seed override")
    parser.add_argument("--reuse-environment", action="store_true")
    parser.add_argument("--trace-episodes", type=int, nargs="*", default=[0, 1, 49, 99])
    args = parser.parse_args()
    if args.num_envs < 1 or (args.episodes is not None and args.episodes < 1):
        parser.error("episodes and num-envs must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Select asset/PhysX locations before the first import of ManiSkill/SAPIEN.
    from roboscope_kit.runtime.benchmark.manifest import load_benchmark_manifest
    from roboscope_kit.runtime.benchmarks.maniskill.profile import validate_profiles

    manifest = load_benchmark_manifest(args.benchmark_manifest)
    runtime_home = (args.runtime_home or manifest.runtime_path("home")).resolve(strict=True)
    library = runtime_home / ".sapien/physx/105.1-physx-5.3.1.patch0/libPhysXGpu_64.so"
    if hashlib.sha256(library.read_bytes()).hexdigest() != PHYSX_SHA256:
        raise RuntimeError("Install-local PhysX GPU library differs from the pinned build")
    os.environ["HOME"] = str(runtime_home)
    os.environ["MS_ASSET_DIR"] = str(manifest.runtime_path("asset_dir"))
    protocol = manifest.benchmark_protocols["maniskill"]
    profiles = validate_profiles(protocol, args.suite)
    if any(profile["obs_mode"] != "state" for profile in profiles):
        raise ValueError("This PPO native example accepts state profiles only")
    if args.episodes is not None and any(args.episodes > len(p["seeds"]) for p in profiles):
        raise ValueError("Requested episodes exceed the frozen manifest")

    from roboscope_kit.runtime.benchmarks.maniskill.worker import create_environment

    imports_and_setup_s = time.perf_counter() - PROCESS_ENTRY_STARTED
    torch.manual_seed(0 if args.seed is None else args.seed)
    np.random.seed(0 if args.seed is None else args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(1)
    model_started = time.perf_counter()
    models = load_model_bundle(args.checkpoint)
    for profile in profiles:
        if profile["task_id"] not in models:
            raise ValueError(f"No checkpoint configured for {profile['task_id']}")
        model = models[profile["task_id"]]
        for key in ("robot_uid", "control_mode"):
            expected = model.metadata.get(key)
            if expected is not None and profile[key] != expected:
                raise ValueError(f"{profile['task_id']} checkpoint requires {key}={expected!r}")
    torch.cuda.synchronize()
    model_load_s = time.perf_counter() - model_started

    records = []
    tasks = []
    rollout_s = reset_s = create_s = close_s = diagnostics_s = inference_host_s = 0.0
    formal_started = time.perf_counter()
    for profile in profiles:
        task_records = []
        model = models[profile["task_id"]]
        episodes = args.episodes or len(profile["seeds"])
        selected_seeds = list(profile["seeds"][:episodes])
        if args.seed is not None:
            selected_seeds = list(range(args.seed, args.seed + episodes))
        horizon = int(profile["max_episode_steps"])
        env = None
        env_count = None
        try:
            for start_episode in range(0, episodes, args.num_envs):
                seeds = selected_seeds[start_episode:start_episode + args.num_envs]
                count = len(seeds)
                if env is not None and env_count != count:
                    phase_started = time.perf_counter()
                    env.close()
                    torch.cuda.synchronize()
                    close_s += time.perf_counter() - phase_started
                    env = None
                if env is None:
                    phase_started = time.perf_counter()
                    env = create_environment(profile, count)
                    env_count = count
                    torch.cuda.synchronize()
                    create_s += time.perf_counter() - phase_started
                    if tuple(env.single_observation_space.shape) != (model.observation_dim,):
                        raise ValueError(f"{profile['task_id']} observation differs from its checkpoint")
                    if tuple(env.single_action_space.shape) != (model.action_dim,):
                        raise ValueError(f"{profile['task_id']} controller differs from its checkpoint")
                phase_started = time.perf_counter()
                observation, _ = env.reset(seed=seeds)
                torch.cuda.synchronize()
                reset_s += time.perf_counter() - phase_started
                initial_states = observation.detach().clone()
                first_actions = None
                rollout_started = time.perf_counter()
                for step in range(horizon):
                    inference_started = time.perf_counter()
                    with torch.inference_mode():
                        action = model.agent.actor_mean(observation)
                    inference_host_s += time.perf_counter() - inference_started
                    if step == 0:
                        first_actions = action.detach().clone()
                    observation, _, terminated, truncated, info = env.step(action)
                    if bool(terminated.any()):
                        raise RuntimeError("The fixed-horizon native PPO profile unexpectedly terminated")
                    if step < horizon - 1 and bool(truncated.any()):
                        raise RuntimeError("Native vector lane truncated before the frozen horizon")
                torch.cuda.synchronize()
                rollout_s += time.perf_counter() - rollout_started
                phase_started = time.perf_counter()
                if not bool(truncated.all()):
                    raise RuntimeError("Not all native lanes reached the frozen horizon")
                metrics = info["episode"]
                if not bool((metrics["episode_len"] == horizon).all()):
                    raise RuntimeError("Native metrics do not cover the actual horizon")
                once = metrics["success_once"].cpu().tolist()
                at_end = metrics["success_at_end"].cpu().tolist()
                states_cpu = initial_states.cpu().numpy().astype("<f4", copy=False)
                first_actions_cpu = first_actions.cpu().numpy()
                for lane, seed in enumerate(seeds):
                    episode = start_episode + lane
                    state = np.ascontiguousarray(states_cpu[lane])
                    record = {
                        "profile_id": profile["profile_id"], "task_id": profile["task_id"],
                        "episode": episode, "seed": seed, "vector_lane": lane,
                        "steps": horizon, "success_once": bool(once[lane]),
                        "success_at_end": bool(at_end[lane]),
                        "initial_state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                        "first_action": first_actions_cpu[lane].tolist(),
                    }
                    if episode in args.trace_episodes:
                        record["initial_state"] = state.tolist()
                    task_records.append(record)
                diagnostics_s += time.perf_counter() - phase_started
                if not args.reuse_environment:
                    phase_started = time.perf_counter()
                    env.close()
                    torch.cuda.synchronize()
                    close_s += time.perf_counter() - phase_started
                    env = None
        finally:
            if env is not None:
                phase_started = time.perf_counter()
                env.close()
                torch.cuda.synchronize()
                close_s += time.perf_counter() - phase_started
        if [record["seed"] for record in task_records] != selected_seeds:
            raise RuntimeError("Native per-task seed coverage has gaps or duplicates")
        records.extend(task_records)
        primary_metric = profile["success_metric"]
        tasks.append({
            "profile_id": profile["profile_id"], "task_id": profile["task_id"],
            "robot_uid": profile["robot_uid"], "control_mode": profile["control_mode"],
            "horizon": horizon, "episodes": episodes,
            "state_dimension": model.observation_dim, "action_dimension": model.action_dim,
            "success_metric": primary_metric,
            "successes": sum(record[primary_metric] for record in task_records),
            "success_rate": sum(record[primary_metric] for record in task_records) / episodes,
            "success_once_rate": sum(record["success_once"] for record in task_records) / episodes,
            "success_at_end_rate": sum(record["success_at_end"] for record in task_records) / episodes,
            "checkpoint_sha256": model.sha256, "seed_manifest": selected_seeds,
        })

    evaluation_wall_s = time.perf_counter() - formal_started
    formal_wall_s = imports_and_setup_s + evaluation_wall_s
    total_episodes = len(records)
    actual_steps = sum(record["steps"] for record in records)
    result = {
        "formal": False, "mode": "native_gpu_vectorized", "suite": args.suite,
        "protocol": protocol["protocol_id"], "benchmark_manifest": str(manifest.path),
        "tasks": tasks, "num_envs": args.num_envs, "episodes": total_episodes,
        "actual_simulation_steps": actual_steps,
        "successes": sum(task["successes"] for task in tasks),
        "success_rate": sum(task["successes"] for task in tasks) / total_episodes,
        "success_once": sum(record["success_once"] for record in records) / total_episodes,
        "success_at_end": sum(record["success_at_end"] for record in records) / total_episodes,
        "macro_success_rate": sum(task["success_rate"] for task in tasks) / len(tasks),
        "environment_lifecycle": "persistent" if args.reuse_environment else "recreate_per_vector_group",
        "runtime_home": str(runtime_home), "asset_dir": os.environ["MS_ASSET_DIR"],
        "physx_gpu_library_sha256": PHYSX_SHA256,
        "maniskill_source_revision": protocol["maniskill_source_commit"],
        "model_kind": "official_rl_trained_ppo_state", "precision": "float32",
        "versions": {name: importlib.metadata.version(name) for name in ("mani_skill", "torch", "sapien")},
        "timing": {
            "model_load_s": model_load_s, "env_create_s": create_s, "reset_s": reset_s,
            "rollout_s": rollout_s, "env_close_s": close_s, "diagnostics_s": diagnostics_s,
            "inference_host_submission_s": inference_host_s,
            "imports_and_setup_s": imports_and_setup_s, "evaluation_wall_s": evaluation_wall_s,
            "formal_wall_s": formal_wall_s,
            "end_to_end_s": time.perf_counter() - PROCESS_ENTRY_STARTED,
        },
        "episodes_per_s": total_episodes / formal_wall_s,
        "actual_steps_per_s": actual_steps / formal_wall_s,
        "comparison_note": (
            "Compare matching task/seed groups at the same N and verify input hashes, first actions "
            "and outcomes. Changing N can change initial inputs. Pure GPU actor latency is not "
            "measured by the asynchronous host-submission timer. No SDK RPC is used; environment "
            "configuration shares the task-driven adapter factory. The original upstream evaluator "
            "is exercised separately by run_upstream.py."
        ),
        "records": records,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
