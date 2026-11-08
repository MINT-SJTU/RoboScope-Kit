"""Evaluate Octo on native ManiSkill Bridge Gym environments with an isolated actor.

Run this script with the simulator Python and pass --policy-python from the
Octo environment. The persistent actor pipe keeps JAX dependencies isolated.
This correctness reference does not use the SDK Worker or Policy client.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def features(observation, profile: dict, count: int) -> list[dict]:
    from mani_skill.utils.common import flatten_state_dict

    state = flatten_state_dict({
        key: value for key, value in observation.items()
        if key not in {"sensor_data", "sensor_param"}
    }, use_torch=True).detach().cpu().numpy().astype(np.float32, copy=False)
    camera = profile["cameras"]["images.primary"]
    image = observation["sensor_data"][camera]["rgb"].detach().cpu().numpy()
    if not np.issubdtype(image.dtype, np.integer) or image.min() < 0 or image.max() > 255:
        raise ValueError("Native Bridge overlay must contain integer RGB pixels in [0, 255]")
    return [{
        "state.proprioception": np.ascontiguousarray(state[lane]),
        "images.primary": np.ascontiguousarray(image[lane], dtype=np.uint8),
        "language.instruction": profile["instruction"],
    } for lane in range(count)]


def digest(row: dict) -> str:
    result = hashlib.sha256()
    for name, value in sorted(row.items()):
        result.update(name.encode() + b"\0")
        if isinstance(value, np.ndarray):
            result.update(value.dtype.str.encode() + b"\0")
            result.update(json.dumps(value.shape).encode() + b"\0")
            result.update(value.tobytes())
        else:
            result.update(str(value).encode())
        result.update(b"\0")
    return result.hexdigest()


def exchange(actor: subprocess.Popen, request: dict) -> dict:
    actor.stdin.write(json.dumps(request) + "\n")
    actor.stdin.flush()
    line = actor.stdout.readline()
    if not line:
        raise RuntimeError(f"Native Octo actor exited with code {actor.poll()}; inspect its policy log")
    return json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy-python", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--suite", default="bridge_rgb_v1")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path)
    args = parser.parse_args()
    if args.num_envs < 1 or (args.episodes is not None and args.episodes < 1):
        parser.error("Episode and environment counts must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.trajectory_dir:
        args.trajectory_dir.mkdir(parents=True, exist_ok=False)
    from roboscope_kit.runtime.benchmark.manifest import load_benchmark_manifest
    from roboscope_kit.runtime.benchmarks.maniskill.profile import validate_profiles

    manifest = load_benchmark_manifest(args.benchmark_manifest)
    profiles = validate_profiles(manifest.benchmark_protocols["maniskill"], args.suite)
    runtime_home = manifest.runtime_path("home")
    os.environ["HOME"] = str(runtime_home)
    os.environ["MS_ASSET_DIR"] = str(manifest.runtime_path("asset_dir"))
    import torch
    from native_eval import PHYSX_SHA256, native_environment

    library = runtime_home / ".sapien/physx/105.1-physx-5.3.1.patch0/libPhysXGpu_64.so"
    if hashlib.sha256(library.read_bytes()).hexdigest() != PHYSX_SHA256:
        raise RuntimeError("Native Bridge PhysX library differs from its pinned build")
    if any(profile["obs_mode"] != "rgb+segmentation" for profile in profiles):
        raise ValueError("The native Octo example serves the Bridge visual profiles")
    if args.episodes and any(args.episodes > len(profile["seeds"]) for profile in profiles):
        raise ValueError("Requested episodes exceed the frozen Bridge profile")
    started = time.perf_counter()
    records = []
    tasks = []
    create_s = reset_s = rollout_s = close_s = inference_s = 0.0
    pin = json.loads(Path(__file__).with_name("octo-checkpoint.json").read_text(encoding="utf-8"))
    checkpoint_hash = pin["files"]["270000/default/checkpoint"]["sha256"]
    command = [str(args.policy_python), str(Path(__file__).with_name("octo_model.py")),
               "--serve", "--checkpoint", str(args.checkpoint)]
    with args.output.with_suffix(".policy.log").open("w", encoding="utf-8") as log:
        actor = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=log, text=True, bufsize=1)
        try:
            line = actor.stdout.readline()
            if not line or json.loads(line) != {"ready": True}:
                raise RuntimeError("Native Octo actor failed to load; inspect its policy log")
            model_load_s = time.perf_counter() - started
            formal_started = time.perf_counter()
            for profile in profiles:
                count_before = len(records)
                seeds = profile["seeds"][:args.episodes]
                for start in range(0, len(seeds), args.num_envs):
                    group = seeds[start:start + args.num_envs]
                    count = len(group)
                    episode_ids = [
                        f"{profile['profile_id']}-episode-{start + lane:06d}-seed-{seed}"
                        for lane, seed in enumerate(group)
                    ]
                    phase_started = time.perf_counter()
                    env = native_environment(profile, count)
                    torch.cuda.synchronize()
                    create_s += time.perf_counter() - phase_started
                    try:
                        phase_started = time.perf_counter()
                        observation, _ = env.reset(seed=group)
                        torch.cuda.synchronize()
                        reset_s += time.perf_counter() - phase_started
                        initial_rows = features(observation, profile, count)
                        initial_hashes = [digest(row) for row in initial_rows]
                        initial_state_hashes = [hashlib.sha256(
                            row["state.proprioception"].tobytes(),
                        ).hexdigest() for row in initial_rows]
                        exchange(actor, {"op": "reset", "episodes": [
                            {"episode_id": episode_id, "task_id": profile["task_id"]}
                            for episode_id in episode_ids
                        ]})
                        traces = [[] for _ in group] if args.trajectory_dir else None
                        first_actions = None
                        phase_started = time.perf_counter()
                        for step in range(profile["max_episode_steps"]):
                            rows = initial_rows if step == 0 else features(observation, profile, count)
                            images = [row["images.primary"] for row in rows]
                            inference_started = time.perf_counter()
                            reply = exchange(actor, {
                                "op": "act", "episode_ids": episode_ids,
                                "images": [base64.b64encode(image.tobytes()).decode() for image in images],
                                "shapes": [image.shape for image in images],
                            })
                            inference_s += time.perf_counter() - inference_started
                            actions = np.asarray(reply["actions"], dtype=np.float32)
                            if actions.shape != (count, 7) or not np.isfinite(actions).all():
                                raise ValueError("Native Octo actor returned invalid controller actions")
                            if first_actions is None:
                                first_actions = actions.copy()
                            if traces is not None:
                                for lane in range(count):
                                    traces[lane].append({"step": step,
                                        "observation_sha256": digest(rows[lane]),
                                        "action_sha256": hashlib.sha256(actions[lane].tobytes()).hexdigest()})
                            observation, _, terminated, truncated, info = env.step(
                                torch.as_tensor(actions, device=env.device),
                            )
                            if bool(terminated.any()) or (
                                step < profile["max_episode_steps"] - 1 and bool(truncated.any())
                            ):
                                raise RuntimeError("Native Bridge lane ended before its full horizon")
                        torch.cuda.synchronize()
                        rollout_s += time.perf_counter() - phase_started
                        metrics = info["episode"]
                        horizon = profile["max_episode_steps"]
                        if not bool(truncated.all()) or not bool((metrics["episode_len"] == horizon).all()):
                            raise RuntimeError("Native Bridge metrics do not cover the full group")
                        terminal_rows = features(observation, profile, count) if traces is not None else None
                        for lane, seed in enumerate(group):
                            records.append({"profile_id": profile["profile_id"],
                                "task_id": profile["task_id"], "episode": start + lane,
                                "seed": seed, "vector_lane": lane, "steps": horizon,
                                "success_available": True,
                                "success_once": bool(metrics["success_once"][lane]),
                                "success_at_end": bool(metrics["success_at_end"][lane]),
                                "episode_return": float(metrics["return"][lane]),
                                "initial_state_sha256": initial_state_hashes[lane],
                                "initial_observation_sha256": initial_hashes[lane],
                                "first_action": first_actions[lane].tolist(),
                                "terminated": False, "truncated": True})
                            if traces is not None:
                                traces[lane].append({"step": horizon,
                                    "observation_sha256": digest(terminal_rows[lane]), "action_sha256": None})
                                path = args.trajectory_dir / f"{episode_ids[lane]}.jsonl"
                                path.write_text("".join(json.dumps(row) + "\n" for row in traces[lane]),
                                                encoding="utf-8")
                        exchange(actor, {"op": "end", "episode_ids": episode_ids})
                    finally:
                        phase_started = time.perf_counter()
                        env.close()
                        env = None
                        torch.cuda.synchronize()
                        close_s += time.perf_counter() - phase_started
                    print(json.dumps({
                        "native_progress": True, "profile_id": profile["profile_id"],
                        "completed_episodes": len(records),
                        "actual_steps": sum(row["steps"] for row in records),
                    }), file=sys.stderr, flush=True)
                task_records = records[count_before:]
                if [row["seed"] for row in task_records] != seeds:
                    raise RuntimeError("Native Bridge task coverage has gaps or duplicates")
                episodes = len(task_records)
                successes = sum(row[profile["success_metric"]] for row in task_records)
                tasks.append({
                    "task_id": profile["task_id"], "profile_id": profile["profile_id"],
                    "robot_uid": profile["robot_uid"], "control_mode": profile["control_mode"],
                    "obs_mode": profile["obs_mode"], "horizon": profile["max_episode_steps"],
                    "episodes": episodes, "seed_manifest": seeds,
                    "state_dimension": len(initial_rows[0]["state.proprioception"]),
                    "action_dimension": 7, "success_metric": profile["success_metric"],
                    "successes": successes, "success_rate": successes / episodes,
                    "success_once_rate": sum(row["success_once"] for row in task_records) / episodes,
                    "success_at_end_rate": sum(row["success_at_end"] for row in task_records) / episodes,
                    "mean_return": sum(row["episode_return"] for row in task_records) / episodes,
                    "actual_steps": sum(row["steps"] for row in task_records),
                    "checkpoint_sha256": checkpoint_hash,
                })
            formal_s = time.perf_counter() - formal_started
        finally:
            if actor.poll() is None:
                actor.stdin.write(json.dumps({"op": "close"}) + "\n")
                actor.stdin.flush()
                actor.wait(timeout=60)
    actual_steps = sum(row["steps"] for row in records)
    successes = sum(task["successes"] for task in tasks)
    total_s = time.perf_counter() - started
    result = {
        "mode": "native_gpu_vectorized", "protocol": "maniskill-v1",
        "suite": args.suite, "model": "octo-small-1.0", "num_envs": args.num_envs,
        "model_kind": "public_oxe_pretrained_octo_small_bridge", "precision": "float32",
        "benchmark_manifest": str(manifest.path), "runtime_home": str(runtime_home),
        "asset_dir": os.environ["MS_ASSET_DIR"],
        "maniskill_source_revision": manifest.benchmark_protocols["maniskill"]["maniskill_source_commit"],
        "physx_gpu_library_sha256": PHYSX_SHA256,
        "rng_protocol": "Independent init_rng=0 stream per episode, five initial key splits",
        "formal": False,
        "complete_named_suite": all(
            len(profile["seeds"]) == (args.episodes or len(profile["seeds"])) for profile in profiles
        ),
        "episodes": len(records), "actual_simulation_steps": actual_steps,
        "successes": successes, "success_rate": successes / len(records),
        "success_once": sum(row["success_once"] for row in records) / len(records),
        "success_at_end": sum(row["success_at_end"] for row in records) / len(records),
        "macro_success_rate": sum(task["success_rate"] for task in tasks) / len(tasks),
        "mean_return": sum(row["episode_return"] for row in records) / len(records),
        "trajectory_diagnostics": args.trajectory_dir is not None,
        "environment_lifecycle": "recreate_per_vector_group",
        "versions": {name: importlib.metadata.version(name) for name in ("mani_skill", "torch", "sapien")},
        "timing": {
            "actor_startup_and_load_s": model_load_s,
            "evaluation_wall_s": formal_s, "end_to_end_after_imports_s": total_s,
            "env_create_s": create_s, "reset_s": reset_s, "rollout_s": rollout_s,
            "env_close_s": close_s, "actor_round_trip_s": inference_s,
        },
        "episodes_per_s": len(records) / formal_s, "actual_steps_per_s": actual_steps / formal_s,
        "comparison_note": (
            "Native Gym construction and vector wrappers are independent of the SDK Worker. "
            "The actor pipe preserves raw RGB bytes and executes the same pinned model. "
            "This reference's timing includes its own transport and is used for correctness."
        ),
        "tasks": tasks, "records": records,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
