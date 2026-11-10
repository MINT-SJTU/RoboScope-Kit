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
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from ppo_model import load_model_bundle

PHYSX_SHA256 = "4c582a16509a71faf5592fe9708586dfcc7ab61ae932eabc1ddd81f290818706"


def native_environment(profile: dict, count: int):
    """Build through upstream Gym and wrappers independently of the Kit worker."""
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 -- register pinned upstream tasks
    from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    kwargs = deepcopy(profile.get("env_kwargs", {}))
    simulation = kwargs.pop("sim_config", {})
    for frequency in ("sim_freq", "control_freq"):
        if frequency in profile:
            simulation[frequency] = profile[frequency]
    robots = profile["robot_uid"]
    controller = profile["control_mode"]
    if isinstance(controller, list):
        if len(set(controller)) != 1:
            raise ValueError("The upstream multi-robot constructor requires one shared controller")
        controller = controller[0]
    robot_options = {} if profile.get("use_default_robot", False) else {
        "robot_uids": tuple(robots) if isinstance(robots, list) else robots,
    }
    raw = gym.make(
        profile["task_id"], num_envs=count,
        control_mode=controller, obs_mode=profile["obs_mode"],
        sim_backend=profile["sim_backend"], render_backend=profile["render_backend"],
        render_mode="rgb_array", sim_config=simulation,
        sensor_configs=deepcopy(profile.get("sensor_configs", {})),
        max_episode_steps=profile["max_episode_steps"],
        reconfiguration_freq=profile["reconfiguration_freq"],
        enhanced_determinism=profile["enhanced_determinism"], **robot_options, **kwargs,
    )
    try:
        if profile.get("use_default_robot", False) and raw.unwrapped.agent.uid != robots:
            raise ValueError("Upstream default robot differs from the frozen profile")
        if isinstance(raw.get_wrapper_attr("single_action_space"), gym.spaces.Dict):
            raw = FlattenActionSpaceWrapper(raw)
        return ManiSkillVectorEnv(
            raw, num_envs=count, auto_reset=False,
            ignore_terminations=profile["ignore_terminations"], record_metrics=True,
        )
    except Exception:
        raw.close()
        raise


def native_observation_digest(state: np.ndarray, instruction: str) -> str:
    """Hash the native state and declared instruction in the SDK feature format."""
    features = {"state.proprioception": state}
    if instruction:
        features["language.instruction"] = instruction
    digest = hashlib.sha256()
    for name, value in sorted(features.items()):
        digest.update(name.encode() + b"\0")
        if isinstance(value, np.ndarray):
            digest.update(value.dtype.str.encode() + b"\0")
            digest.update(json.dumps(value.shape).encode() + b"\0")
            digest.update(np.ascontiguousarray(value).tobytes())
        else:
            digest.update(str(value).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def run_task_processes(args: argparse.Namespace, profiles: tuple[dict, ...]) -> None:
    """Use the upstream one-task process lifetime for a multi-task reference."""
    root = args.output.with_suffix(".tasks")
    root.mkdir(parents=True, exist_ok=False)
    children = []
    provenance = []
    for profile in profiles:
        destination = root / profile["profile_id"]
        destination.mkdir()
        output = destination / "result.json"
        log = destination / "native.log"
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--checkpoint", str(args.checkpoint.resolve()),
            "--benchmark-manifest", str(args.benchmark_manifest.resolve()),
            "--suite", args.suite, "--profile-id", profile["profile_id"],
            "--output", str(output.resolve()), "--num-envs", str(args.num_envs),
        ]
        for option in ("runtime_home", "episodes", "seed"):
            value = getattr(args, option)
            if value is not None:
                command.extend(["--" + option.replace("_", "-"), str(value)])
        if args.reuse_environment:
            command.append("--reuse-environment")
        command.extend(["--trace-episodes", *map(str, args.trace_episodes)])
        if args.trajectory_dir is not None:
            command.extend(["--trajectory-dir", str(args.trajectory_dir / profile["profile_id"])])
        started = time.perf_counter()
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        if process.returncode:
            raise RuntimeError(f"Native task {profile['profile_id']} failed; see {log}")
        child = json.loads(output.read_text(encoding="utf-8"))
        count = args.episodes or len(profile["seeds"])
        seeds = profile["seeds"][:count] if args.seed is None else list(range(args.seed, args.seed + count))
        expected = [(profile["profile_id"], index, seed) for index, seed in enumerate(seeds)]
        actual = [(row["profile_id"], row["episode"], row["seed"]) for row in child["records"]]
        if (actual != expected or child["episodes"] != count
                or [row["profile_id"] for row in child["tasks"]] != [profile["profile_id"]]
                or any(row["steps"] != profile["max_episode_steps"] for row in child["records"])):
            raise RuntimeError(f"Native task {profile['profile_id']} has incomplete or unexpected records")
        children.append(child)
        provenance.append({
            "profile_id": profile["profile_id"], "command": command, "log": str(log),
            "result": str(output), "result_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "process_wall_s": time.perf_counter() - started,
        })

    records = [record for child in children for record in child["records"]]
    tasks = [task for child in children for task in child["tasks"]]
    successes = [record for record in records if record["success_available"]]
    scored_tasks = [task for task in tasks if task["successes"] is not None]
    timing = {
        key: sum(child["timing"][key] for child in children)
        for key in children[0]["timing"] if key != "end_to_end_s"
    }
    timing.update({
        "scope": "Sequential child phase sums; end_to_end_s is the complete parent command",
        "end_to_end_s": time.perf_counter() - PROCESS_ENTRY_STARTED,
    })
    steps = sum(record["steps"] for record in records)
    result = {
        **children[0], "formal": False, "tasks": tasks, "records": records,
        "episodes": len(records), "actual_simulation_steps": steps,
        "successes": sum(task["successes"] for task in scored_tasks) if scored_tasks else None,
        "success_rate": (
            sum(task["successes"] for task in scored_tasks) / len(successes) if successes else None
        ),
        "success_once": (
            sum(record["success_once"] for record in successes) / len(successes) if successes else None
        ),
        "success_at_end": (
            sum(record["success_at_end"] for record in successes) / len(successes) if successes else None
        ),
        "macro_success_rate": (
            sum(task["success_rate"] for task in scored_tasks) / len(scored_tasks) if scored_tasks else None
        ),
        "mean_return": sum(record["episode_return"] for record in records) / len(records),
        "timing": timing, "episodes_per_s": len(records) / timing["formal_wall_s"],
        "actual_steps_per_s": steps / timing["formal_wall_s"],
        "process_isolation": "one_task_per_process", "task_references": provenance,
        "comparison_note": (
            "Each task uses a separate native process, matching the upstream evaluator lifetime. "
            "Rates use sequential child formal intervals. Parent end-to-end time includes process "
            "startup and repeated model loading. Use the matching per-task records for correctness."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


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
    parser.add_argument("--trajectory-dir", type=Path, help="Diagnostic per-step hashes outside timed runs")
    parser.add_argument("--profile-id", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.num_envs < 1 or (args.episodes is not None and args.episodes < 1):
        parser.error("episodes and num-envs must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.trajectory_dir is not None:
        args.trajectory_dir.mkdir(parents=True, exist_ok=False)

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
    if args.profile_id is not None:
        profiles = tuple(profile for profile in profiles if profile["profile_id"] == args.profile_id)
        if len(profiles) != 1:
            parser.error("--profile-id must identify exactly one profile in the selected suite")
    if any("diagnostic_step_limit" in profile for profile in profiles):
        raise ValueError("Diagnostic step limits require native_eval_policy.py")
    if any(profile["obs_mode"] != "state" for profile in profiles):
        raise ValueError("This PPO native example accepts state profiles only")
    if args.episodes is not None and any(args.episodes > len(p["seeds"]) for p in profiles):
        raise ValueError("Requested episodes exceed the frozen manifest")
    if len(profiles) > 1:
        run_task_processes(args, profiles)
        return

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
                    env = native_environment(profile, count)
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
                trajectories = [[] for _ in seeds] if args.trajectory_dir is not None else None
                rollout_started = time.perf_counter()
                for step in range(horizon):
                    inference_started = time.perf_counter()
                    with torch.inference_mode():
                        action = model.agent.actor_mean(observation)
                    inference_host_s += time.perf_counter() - inference_started
                    if step == 0:
                        first_actions = action.detach().clone()
                    if trajectories is not None:
                        states = observation.detach().cpu().numpy().astype("<f4", copy=False)
                        actions = action.detach().cpu().numpy().astype("<f4", copy=False)
                        for lane in range(count):
                            trajectories[lane].append({
                                "step": step,
                                "observation_sha256": native_observation_digest(
                                    np.ascontiguousarray(states[lane]), profile.get("instruction", ""),
                                ),
                                "action_sha256": hashlib.sha256(
                                    np.ascontiguousarray(actions[lane]).tobytes(),
                                ).hexdigest(),
                            })
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
                success_available = "success_once" in metrics
                once = metrics["success_once"].cpu().tolist() if success_available else [None] * count
                at_end = metrics["success_at_end"].cpu().tolist() if success_available else [None] * count
                returns = metrics["return"].cpu().tolist()
                states_cpu = initial_states.cpu().numpy().astype("<f4", copy=False)
                first_actions_cpu = first_actions.cpu().numpy()
                terminal_states = (
                    observation.detach().cpu().numpy().astype("<f4", copy=False)
                    if trajectories is not None else None
                )
                for lane, seed in enumerate(seeds):
                    episode = start_episode + lane
                    state = np.ascontiguousarray(states_cpu[lane])
                    record = {
                        "profile_id": profile["profile_id"], "task_id": profile["task_id"],
                        "episode": episode, "seed": seed, "vector_lane": lane,
                        "steps": horizon, "success_once": bool(once[lane]) if success_available else None,
                        "success_at_end": bool(at_end[lane]) if success_available else None,
                        "success_available": success_available, "episode_return": float(returns[lane]),
                        "initial_state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                        "first_action": first_actions_cpu[lane].tolist(),
                        "terminated": bool(terminated[lane]), "truncated": bool(truncated[lane]),
                    }
                    if episode in args.trace_episodes:
                        record["initial_state"] = state.tolist()
                    if trajectories is not None:
                        trajectories[lane].append({
                            "step": horizon,
                            "observation_sha256": native_observation_digest(
                                np.ascontiguousarray(terminal_states[lane]), profile.get("instruction", ""),
                            ),
                            "action_sha256": None,
                        })
                        episode_id = f"{profile['profile_id']}-episode-{episode:06d}-seed-{seed}"
                        trace = "".join(json.dumps(row) + "\n" for row in trajectories[lane])
                        (args.trajectory_dir / f"{episode_id}.jsonl").write_text(trace, encoding="utf-8")
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
        scored_success = primary_metric != "return"
        if scored_success and not all(record["success_available"] for record in task_records):
            raise RuntimeError(f"{profile['task_id']} does not emit the declared success metric")
        successes = sum(record[primary_metric] for record in task_records) if scored_success else None
        tasks.append({
            "profile_id": profile["profile_id"], "task_id": profile["task_id"],
            "robot_uid": profile["robot_uid"], "control_mode": profile["control_mode"],
            "horizon": horizon, "episodes": episodes,
            "state_dimension": model.observation_dim, "action_dimension": model.action_dim,
            "success_metric": primary_metric,
            "successes": successes,
            "success_rate": successes / episodes if scored_success else None,
            "success_once_rate": (
                sum(record["success_once"] for record in task_records) / episodes if scored_success else None
            ),
            "success_at_end_rate": (
                sum(record["success_at_end"] for record in task_records) / episodes
                if scored_success else None
            ),
            "mean_return": sum(record["episode_return"] for record in task_records) / episodes,
            "checkpoint_sha256": model.sha256, "seed_manifest": selected_seeds,
        })

    evaluation_wall_s = time.perf_counter() - formal_started
    formal_wall_s = imports_and_setup_s + evaluation_wall_s
    total_episodes = len(records)
    actual_steps = sum(record["steps"] for record in records)
    success_tasks = [task for task in tasks if task["successes"] is not None]
    success_records = [record for record in records if record["success_available"]]
    result = {
        "formal": False, "mode": "native_gpu_vectorized", "suite": args.suite,
        "protocol": protocol["protocol_id"], "benchmark_manifest": str(manifest.path),
        "tasks": tasks, "num_envs": args.num_envs, "episodes": total_episodes,
        "actual_simulation_steps": actual_steps,
        "successes": sum(task["successes"] for task in success_tasks) if success_tasks else None,
        "success_rate": (
            sum(task["successes"] for task in success_tasks) / len(success_records)
            if success_records else None
        ),
        "success_once": (
            sum(record["success_once"] for record in success_records) / len(success_records)
            if success_records else None
        ),
        "success_at_end": (
            sum(record["success_at_end"] for record in success_records) / len(success_records)
            if success_records else None
        ),
        "macro_success_rate": (
            sum(task["success_rate"] for task in success_tasks) / len(success_tasks)
            if success_tasks else None
        ),
        "mean_return": sum(record["episode_return"] for record in records) / total_episodes,
        "environment_lifecycle": "persistent" if args.reuse_environment else "recreate_per_vector_group",
        "runtime_home": str(runtime_home), "asset_dir": os.environ["MS_ASSET_DIR"],
        "physx_gpu_library_sha256": PHYSX_SHA256,
        "maniskill_source_revision": protocol["maniskill_source_commit"],
        "model_kind": "official_rl_trained_ppo_state", "precision": "float32",
        "trajectory_diagnostics": args.trajectory_dir is not None,
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
            "measured by the asynchronous host-submission timer. No SDK RPC is used; the environment "
            "is constructed directly with upstream Gym and wrappers. The original upstream evaluator "
            "is exercised separately by run_upstream.py."
        ),
        "records": records,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
