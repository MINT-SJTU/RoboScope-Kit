"""Compare any public Policy against native Gym observations, without SDK workers or RPCs."""

# ruff: noqa: E402 -- include heavyweight imports in the process time boundary
from __future__ import annotations

import time

PROCESS_STARTED = time.perf_counter()

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def native_features(observation, environment, profile: dict, count: int) -> list[dict]:
    """Read upstream observation fields directly; no Kit worker extraction is imported."""
    if isinstance(observation, Mapping):
        from mani_skill.utils.common import flatten_state_dict

        state = flatten_state_dict(
            {key: value for key, value in observation.items() if key not in {"sensor_data", "sensor_param"}},
            use_torch=True,
        )
    else:
        state = observation
    state = numpy(state).astype(np.float32, copy=False)
    rows = [{"state.proprioception": np.ascontiguousarray(state[lane])} for lane in range(count)]
    for name, camera in profile.get("cameras", {}).items():
        images = numpy(observation["sensor_data"][camera]["rgb"])
        if (
            not np.issubdtype(images.dtype, np.integer)
            or images.shape != (count, *images.shape[1:3], 3)
            or images.min() < 0
            or images.max() > 255
        ):
            raise ValueError(f"Invalid native RGB camera {camera}")
        for lane, row in enumerate(rows):
            row[name] = np.ascontiguousarray(images[lane], dtype=np.uint8)
    if profile.get("include_root_kinematics", False):
        robot = environment.unwrapped.agent.robot
        for name, value in (
            ("state.root_pose", robot.pose.raw_pose),
            ("state.root_linear_velocity", robot.root_linear_velocity),
            ("state.root_angular_velocity", robot.root_angular_velocity),
        ):
            values = numpy(value).astype(np.float32, copy=False)
            for lane, row in enumerate(rows):
                row[name] = np.ascontiguousarray(values[lane])
    if profile.get("include_cabinet_contact_state", False):
        native = environment.unwrapped
        robot = native.agent
        for name, value in (
            ("state.robot_base_pose", robot.base_link.pose.raw_pose),
            ("state.robot_tcp_pose", robot.tcp.pose.raw_pose),
            ("state.target_handle_position", native.handle_link_positions()),
            ("state.left_finger_pose", robot.finger1_link.pose.raw_pose),
            ("state.right_finger_pose", robot.finger2_link.pose.raw_pose),
            ("state.left_target_contact_force",
             native.scene.get_pairwise_contact_forces(robot.finger1_link, native.handle_link)),
            ("state.right_target_contact_force",
             native.scene.get_pairwise_contact_forces(robot.finger2_link, native.handle_link)),
        ):
            values = numpy(value).astype(np.float32, copy=False)
            for lane, row in enumerate(rows):
                row[name] = np.ascontiguousarray(values[lane])
    for name, camera in profile.get("depth_cameras", {}).items():
        depths = numpy(observation["sensor_data"][camera]["depth"])
        if depths.ndim != 4 or depths.shape[0] != count or depths.shape[-1] != 1:
            raise ValueError(f"Invalid native depth camera {camera}")
        for lane, row in enumerate(rows):
            row[name] = np.ascontiguousarray(depths[lane], dtype=np.float32)
    if profile.get("instruction"):
        for row in rows:
            row["language.instruction"] = profile["instruction"]
    return rows


def digest(features: dict) -> str:
    value = hashlib.sha256()
    for name, item in sorted(features.items()):
        value.update(name.encode() + b"\0")
        if isinstance(item, np.ndarray):
            value.update(item.dtype.str.encode() + b"\0")
            value.update(json.dumps(item.shape).encode() + b"\0")
            value.update(np.ascontiguousarray(item).tobytes())
        else:
            value.update(str(item).encode())
        value.update(b"\0")
    return value.hexdigest()


def exchange(actor: subprocess.Popen, request: dict) -> dict:
    actor.stdin.write(json.dumps(request) + "\n")
    actor.stdin.flush()
    line = actor.stdout.readline()
    if not line:
        raise RuntimeError(f"Native Policy actor exited ({actor.poll()}); inspect the policy log")
    return json.loads(line)


def encode(features: dict) -> dict:
    return {
        name: {
            "dtype": item.dtype.str,
            "shape": item.shape,
            "data": base64.b64encode(np.ascontiguousarray(item).tobytes()).decode(),
        }
        if isinstance(item, np.ndarray)
        else item
        for name, item in features.items()
    }


def episode_metadata(environment, profile: dict, suite: str, index: int, seed: int, lane: int) -> dict:
    metadata = {
        name: profile[name]
        for name in ("robot_uid", "control_mode", "obs_mode", "sim_freq", "control_freq", "profile_id")
    }
    metadata.update(
        seed=seed,
        episode_index=index,
        vector_lane=lane,
        suite=suite,
        action_dimension=int(environment.single_action_space.shape[0]),
    )
    if (profile.get("include_root_kinematics", False)
            or profile.get("include_joint_names", False)
            or profile.get("include_cabinet_contact_state", False)):
        agent = environment.unwrapped.agent
        configs = getattr(agent.controller, "configs", None)
        configs = configs.values() if configs is not None else (agent.controller.config,)
        metadata["active_joint_names"] = [joint.name for joint in agent.robot.get_active_joints()]
        metadata["action_joint_names"] = [name for config in configs for name in config.joint_names]
    return metadata


def episode_segments(
    profile: dict, episodes: int | None, num_envs: int, episode_start: int = 0,
) -> tuple[tuple[int, int, int], ...]:
    """Keep full-width groups together and isolate a smaller final group."""
    stop = len(profile["seeds"]) if episodes is None else episode_start + episodes
    if num_envs < 1 or not 0 <= episode_start < stop <= len(profile["seeds"]):
        raise ValueError("Requested episode interval exceeds the frozen seeds")
    count = stop - episode_start
    full = count // num_envs * num_envs
    segments = [(episode_start, full, num_envs)] if full else []
    if count > full:
        segments.append((episode_start + full, count - full, count - full))
    return tuple(segments)


def run_task_processes(args: argparse.Namespace, profiles: tuple[dict, ...]) -> None:
    """Isolate native profile and vector-width lifetimes, preserving episode identities."""
    root = args.output.with_suffix(".tasks")
    root.mkdir(parents=True, exist_ok=False)
    children, provenance, tasks = [], [], []
    for profile in profiles:
        segments = episode_segments(profile, args.episodes, args.num_envs, args.episode_start)
        profile_tasks = []
        for episode_start, count, width in segments:
            segment_name = f"episodes-{episode_start:06d}-{episode_start + count:06d}-envs-{width}"
            relative = Path(profile["profile_id"])
            if len(segments) > 1:
                relative /= segment_name
            destination = root / relative
            destination.mkdir(parents=True)
            output = destination / "result.json"
            log = destination / "native.log"
            # Resolving the Python symlink would bypass its virtual environment.
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--checkpoint", str(args.checkpoint.resolve()),
                "--policy-python", str(args.policy_python.absolute()),
                "--policy-module", args.policy_module, "--policy-class", args.policy_class,
                "--benchmark-manifest", str(args.benchmark_manifest.resolve()),
                "--suite", args.suite, "--profile-id", profile["profile_id"],
                "--output", str(output.resolve()), "--num-envs", str(width),
                "--episodes", str(count), "--episode-start", str(episode_start),
            ]
            if args.runtime_home is not None:
                command.extend(["--runtime-home", str(args.runtime_home.resolve())])
            if args.trajectory_dir is not None:
                command.extend(["--trajectory-dir", str((args.trajectory_dir / relative).resolve())])
            (destination / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
            started = time.perf_counter()
            with log.open("w", encoding="utf-8") as stream:
                process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
            process_wall_s = time.perf_counter() - started
            if process.returncode:
                raise RuntimeError(f"Native task {profile['profile_id']} segment failed; see {log}")
            result_bytes = output.read_bytes()
            child = json.loads(result_bytes)
            seeds = profile["seeds"][episode_start:episode_start + count]
            limit = int(profile.get("diagnostic_step_limit", profile["max_episode_steps"]))
            expected = [
                (profile["profile_id"], profile["task_id"], episode_start + index,
                 seed, index % width, limit)
                for index, seed in enumerate(seeds)
            ]
            actual = [
                (row["profile_id"], row["task_id"], row["episode"], row["seed"],
                 row["vector_lane"], row["steps"])
                for row in child["records"]
            ]
            if (actual != expected or child["episodes"] != count
                    or child["actual_simulation_steps"] != count * limit
                    or child["suite"] != args.suite or child["num_envs"] != width
                    or [(task["profile_id"], task["task_id"], task["episodes"], task["horizon"])
                        for task in child["tasks"]] != [
                            (profile["profile_id"], profile["task_id"], count, profile["max_episode_steps"])
                        ]):
                raise RuntimeError(f"Native task {profile['profile_id']} segment has unexpected records")
            children.append(child)
            profile_tasks.extend(child["tasks"])
            provenance.append({
                "profile_id": profile["profile_id"], "episode_start": episode_start,
                "episodes": count, "vector_envs": width, "command": command, "log": str(log),
                "result": str(output), "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
                "process_wall_s": process_wall_s,
            })
        task = dict(profile_tasks[0])
        task["episodes"] = sum(row["episodes"] for row in profile_tasks)
        task["successes"] = (None if profile["success_metric"] == "return"
                             else sum(row["successes"] for row in profile_tasks))
        task["success_rate"] = (None if task["successes"] is None
                                else task["successes"] / task["episodes"])
        tasks.append(task)

    records = [record for child in children for record in child["records"]]
    timing = {
        key: sum(child["timing"][key] for child in children)
        for key, value in children[0]["timing"].items()
        if key != "end_to_end_s" and type(value) in (int, float)
    }
    timing.update({
        "scope": ("Sequential profile/width-segment child phase sums; "
                  "end_to_end_s is the complete parent command"),
        "end_to_end_s": time.perf_counter() - PROCESS_STARTED,
    })
    result = {
        **children[0], "tasks": tasks, "num_envs": args.num_envs,
        "episode_start": args.episode_start,
        "records": records, "episodes": len(records),
        "actual_simulation_steps": sum(record["steps"] for record in records),
        "evaluation_kind": "integration_smoke"
        if any("diagnostic_step_limit" in profile for profile in profiles) else "full_horizon",
        "timing": timing, "process_isolation": "one_profile_and_vector_width_segment_per_process",
        "task_references": provenance,
        "comparison_note": (
            "Each profile and vector-width segment uses a separate native process. Full-width "
            "groups stay together; a smaller final group retains its original episode indices and seeds. "
            "Formal timing sums child evaluation intervals. Parent end-to-end time includes process "
            "startup and repeated model loading. Use the per-task records for correctness."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "policy-python", "benchmark-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--policy-module", required=True)
    parser.add_argument("--policy-class", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--num-envs", type=int, help="must match the frozen protocol.vector_envs")
    parser.add_argument("--trajectory-dir", type=Path)
    parser.add_argument("--runtime-home", type=Path)
    parser.add_argument("--profile-id", help=argparse.SUPPRESS)
    parser.add_argument("--episode-start", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if ((args.num_envs is not None and args.num_envs < 1) or args.episode_start < 0
            or (args.episodes is not None and args.episodes < 1)):
        parser.error("Episode and environment counts must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.trajectory_dir:
        args.trajectory_dir.mkdir(parents=True, exist_ok=False)
    from roboscope_kit.runtime.benchmark.manifest import load_benchmark_manifest
    from roboscope_kit.runtime.benchmarks.maniskill.profile import (
        physx_library_sha256,
        validate_profiles,
        vector_envs,
    )

    manifest = load_benchmark_manifest(args.benchmark_manifest)
    protocol = manifest.benchmark_protocols["maniskill"]
    expected_physx_hash = physx_library_sha256(protocol)
    width = vector_envs(protocol)
    if args.num_envs is not None and args.num_envs != width:
        raise ValueError("Native vector width differs from the frozen protocol")
    args.num_envs = width
    if (args.episodes is not None and args.episodes % width) or args.episode_start % width:
        raise ValueError("Native episode selection must preserve complete protocol vector groups")
    profiles = validate_profiles(protocol, args.suite)
    if args.profile_id is not None:
        profiles = tuple(profile for profile in profiles if profile["profile_id"] == args.profile_id)
        if len(profiles) != 1:
            parser.error("--profile-id must identify exactly one profile in the selected suite")
    segments = [episode_segments(p, args.episodes, args.num_envs, args.episode_start) for p in profiles]
    if len(profiles) > 1 or len(segments[0]) > 1:
        run_task_processes(args, profiles)
        return
    os.environ["HOME"] = str(args.runtime_home or manifest.runtime_path("home"))
    os.environ["MS_ASSET_DIR"] = str(manifest.runtime_path("asset_dir"))
    import torch
    from native_eval import native_environment

    library = Path(os.environ["HOME"]) / ".sapien/physx/105.1-physx-5.3.1.patch0/libPhysXGpu_64.so"
    if hashlib.sha256(library.read_bytes()).hexdigest() != expected_physx_hash:
        raise RuntimeError("Native PhysX GPU library differs from the pinned build")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    records, tasks = [], []
    timing = dict(env_create_s=0.0, reset_s=0.0, rollout_s=0.0, env_close_s=0.0, inference_wait_s=0.0)
    command = [
        str(args.policy_python),
        str(Path(__file__).with_name("native_policy_actor.py")),
        "--policy-module",
        args.policy_module,
        "--policy-class",
        args.policy_class,
        "--checkpoint",
        str(args.checkpoint),
        "--suite",
        args.suite,
    ]
    model_started = time.perf_counter()
    with args.output.with_suffix(".policy.log").open("w", encoding="utf-8") as log:
        actor = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1
        )
        try:
            if json.loads(actor.stdout.readline()) != {"ready": True}:
                raise RuntimeError("The native Policy actor failed to load")
            timing["model_load_s"] = time.perf_counter() - model_started
            evaluation_started = time.perf_counter()
            for profile in profiles:
                before = len(records)
                stop = None if args.episodes is None else args.episode_start + args.episodes
                seeds = profile["seeds"][args.episode_start:stop]
                horizon = int(profile["max_episode_steps"])
                limit = int(profile.get("diagnostic_step_limit", horizon))
                diagnostic = "diagnostic_step_limit" in profile
                for start in range(0, len(seeds), args.num_envs):
                    group = seeds[start : start + args.num_envs]
                    count = len(group)
                    ids = [
                        f"{profile['profile_id']}-episode-{args.episode_start + start + lane:06d}-seed-{seed}"
                        for lane, seed in enumerate(group)
                    ]
                    phase = time.perf_counter()
                    env = native_environment(profile, count)
                    torch.cuda.synchronize()
                    timing["env_create_s"] += time.perf_counter() - phase
                    try:
                        phase = time.perf_counter()
                        observation, _ = env.reset(seed=group)
                        torch.cuda.synchronize()
                        timing["reset_s"] += time.perf_counter() - phase
                        rows = native_features(observation, env, profile, count)
                        initial_hashes = [digest(row) for row in rows]
                        state_hashes = [
                            hashlib.sha256(row["state.proprioception"].tobytes()).hexdigest() for row in rows
                        ]
                        exchange(
                            actor,
                            {
                                "op": "reset",
                                "episodes": [
                                    {
                                        "episode_id": ids[lane],
                                        "task_id": profile["task_id"],
                                        "metadata": episode_metadata(
                                            env, profile, args.suite, args.episode_start + start + lane,
                                            seed, lane,
                                        ),
                                    }
                                    for lane, seed in enumerate(group)
                                ],
                            },
                        )
                        traces = [[] for _ in group] if args.trajectory_dir else None
                        first_actions = None
                        phase = time.perf_counter()
                        for step in range(limit):
                            rows = rows if step == 0 else native_features(observation, env, profile, count)
                            inference_started = time.perf_counter()
                            reply = exchange(
                                actor,
                                {
                                    "op": "act",
                                    "observations": [
                                        {"episode_id": episode_id, "step": step, "features": encode(row)}
                                        for episode_id, row in zip(ids, rows, strict=True)
                                    ],
                                },
                            )
                            timing["inference_wait_s"] += time.perf_counter() - inference_started
                            actions = np.asarray(reply["actions"], dtype=np.float32)
                            if (
                                actions.shape != (count, env.single_action_space.shape[0])
                                or not np.isfinite(actions).all()
                            ):
                                raise ValueError("Native Policy returned invalid actions")
                            if first_actions is None:
                                first_actions = actions.copy()
                            if traces is not None:
                                for lane in range(count):
                                    traces[lane].append(
                                        {
                                            "step": step,
                                            "observation_sha256": digest(rows[lane]),
                                            "action_sha256": hashlib.sha256(
                                                actions[lane].tobytes()
                                            ).hexdigest(),
                                        }
                                    )
                            observation, _, terminated, truncated, info = env.step(
                                torch.as_tensor(actions, device=env.device)
                            )
                            if bool(terminated.any()) or (step < horizon - 1 and bool(truncated.any())):
                                raise RuntimeError("Native lane ended before its frozen horizon")
                        torch.cuda.synchronize()
                        timing["rollout_s"] += time.perf_counter() - phase
                        metrics = info["episode"]
                        if not bool((truncated == (limit == horizon)).all()) or not bool(
                            (metrics["episode_len"] == limit).all()
                        ):
                            raise RuntimeError(
                                "Native termination masks or episode lengths differ from the prefix"
                            )
                        terminal_rows = (
                            native_features(observation, env, profile, count) if traces is not None else None
                        )
                        available = "success_once" in metrics
                        for lane, seed in enumerate(group):
                            records.append(
                                {
                                    "profile_id": profile["profile_id"],
                                    "task_id": profile["task_id"],
                                    "episode": args.episode_start + start + lane,
                                    "seed": seed,
                                    "vector_lane": lane,
                                    "steps": limit,
                                    "success_available": available,
                                    "success_once": bool(metrics["success_once"][lane])
                                    if available
                                    else None,
                                    "success_at_end": bool(metrics["success_at_end"][lane])
                                    if available
                                    else None,
                                    "episode_return": float(metrics["return"][lane]),
                                    "initial_state_sha256": state_hashes[lane],
                                    "initial_observation_sha256": initial_hashes[lane],
                                    "first_action": first_actions[lane].tolist(),
                                    "terminated": bool(terminated[lane]),
                                    "truncated": bool(truncated[lane]),
                                    "formal": False,
                                    "evaluation_kind": "integration_smoke" if diagnostic else "full_horizon",
                                    **(
                                        {"native_max_episode_steps": horizon, "diagnostic_step_limit": limit}
                                        if diagnostic
                                        else {}
                                    ),
                                }
                            )
                            if traces is not None:
                                traces[lane].append(
                                    {
                                        "step": limit,
                                        "observation_sha256": digest(terminal_rows[lane]),
                                        "action_sha256": None,
                                    }
                                )
                                (args.trajectory_dir / f"{ids[lane]}.jsonl").write_text(
                                    "".join(json.dumps(row) + "\n" for row in traces[lane]), encoding="utf-8"
                                )
                        exchange(actor, {"op": "end", "episode_ids": ids})
                    finally:
                        phase = time.perf_counter()
                        env.close()
                        env = None
                        torch.cuda.synchronize()
                        timing["env_close_s"] += time.perf_counter() - phase
                    print(
                        json.dumps(
                            {
                                "native_progress": True,
                                "profile_id": profile["profile_id"],
                                "completed_episodes": len(records),
                            }
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                task_rows = records[before:]
                scored = profile["success_metric"] != "return"
                successes = sum(row[profile["success_metric"]] for row in task_rows) if scored else None
                tasks.append(
                    {
                        "profile_id": profile["profile_id"],
                        "task_id": profile["task_id"],
                        "robot_uid": profile["robot_uid"],
                        "control_mode": profile["control_mode"],
                        "horizon": horizon,
                        "episodes": len(task_rows),
                        "successes": successes,
                        "success_rate": successes / len(task_rows) if scored else None,
                    }
                )
            timing["formal_wall_s"] = time.perf_counter() - evaluation_started
            exchange(actor, {"op": "quit"})
            actor.wait(timeout=30)
        finally:
            if actor.poll() is None:
                actor.terminate()
                actor.wait(timeout=10)
    timing["end_to_end_s"] = time.perf_counter() - PROCESS_STARTED
    result = {
        "formal": False,
        "mode": "native_gpu_vectorized", "configuration_source": "kit_manifest_diagnostic",
        "protocol": protocol["protocol_id"],
        "suite": args.suite,
        "num_envs": args.num_envs,
        "episode_start": args.episode_start,
        "episodes": len(records),
        "actual_simulation_steps": sum(row["steps"] for row in records),
        "evaluation_kind": "integration_smoke"
        if any("diagnostic_step_limit" in p for p in profiles)
        else "full_horizon",
        "tasks": tasks,
        "records": records,
        "timing": timing,
        "policy_module": args.policy_module,
        "policy_class": args.policy_class,
        "benchmark_manifest": str(manifest.path),
        "physx_gpu_library_sha256": expected_physx_hash,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
