# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0
"""Queue-backed ManiSkill task worker with native GPU vector environments.

Every vector group keeps the pinned upstream fixed-horizon evaluation rules.
Task, robot, observations and controller are selected by the manifest; model
preprocessing and inference remain in the Policy process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from roboscope_kit import EpisodeContext, Observation, PolicyClient
from roboscope_kit.benchmark_specs import BenchmarkName
from roboscope_kit.runtime.benchmark import EvaluationRecord, WorkerLoop, WorkItem, WorkResult
from roboscope_kit.runtime.benchmark.manifest import load_benchmark_manifest

from .profile import PROTOCOL_ID, physx_library_sha256, self_test_profiles, validate_profiles

# Raw states sample the first two, midpoint and final episodes of the 100-episode
# reference. Every episode still retains complete initial-state/observation hashes.
INITIAL_STATE_SNAPSHOT_EPISODE_INDICES = frozenset((0, 1, 49, 99))


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def create_environment(profile: dict[str, Any], num_envs: int):
    """Construct one task exactly as declared, with upstream controller semantics."""
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 -- register pinned upstream tasks
    from mani_skill.utils import assets
    from mani_skill.utils.registration import REGISTERED_ENVS
    from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    robots = profile["robot_uid"]
    control = profile["control_mode"]
    robot_ids = robots if isinstance(robots, list) else [robots]
    required_assets = [
        *(REGISTERED_ENVS[profile["task_id"]].asset_download_ids or []),
        *profile.get("asset_download_ids", []),
        *(uid for uid in robot_ids if uid in assets.DATA_GROUPS),
    ]
    source_ids: set[str] = set()
    for uid in required_assets:
        source_ids.update(
            [uid] if uid in assets.DATA_SOURCES
            else assets.expand_data_group_into_individual_data_source_ids(uid)
        )
    missing_assets = sorted(uid for uid in source_ids if not assets.is_data_source_downloaded(uid))
    if missing_assets:
        raise FileNotFoundError(
            f"ManiSkill {profile['task_id']} requires missing assets {missing_assets}. "
            "Install them with benchmarks/maniskill/prepare-assets.py before evaluation."
        )
    if isinstance(robots, list):
        robots = tuple(robots)
    if isinstance(control, list):
        # BaseEnv passes its constructor control_mode to each individual agent.
        # MultiAgent.set_control_mode's list API is not the Gym constructor API.
        if len(set(control)) != 1:
            raise ValueError("The pinned ManiSkill constructor needs one shared robot controller")
        control = control[0]
    kwargs = deepcopy(profile.get("env_kwargs", {}))
    sim_config = kwargs.pop("sim_config", {})
    for frequency in ("sim_freq", "control_freq"):
        if frequency in profile:
            sim_config[frequency] = profile[frequency]
    kwargs.update(
        num_envs=num_envs, obs_mode=profile["obs_mode"],
        control_mode=control, sim_backend=profile["sim_backend"],
        render_backend=profile["render_backend"], render_mode="rgb_array",
        max_episode_steps=profile["max_episode_steps"],
        reconfiguration_freq=profile["reconfiguration_freq"],
        enhanced_determinism=profile["enhanced_determinism"],
        sim_config=sim_config, sensor_configs=deepcopy(profile.get("sensor_configs", {})),
    )
    if not profile.get("use_default_robot", False):
        kwargs["robot_uids"] = robots
    raw = gym.make(profile["task_id"], **kwargs)
    try:
        if profile.get("use_default_robot", False) and raw.unwrapped.agent.uid != robots:
            raise RuntimeError("ManiSkill task default robot differs from the manifest")
        if isinstance(raw.get_wrapper_attr("single_action_space"), gym.spaces.Dict):
            raw = FlattenActionSpaceWrapper(raw)
        vector = ManiSkillVectorEnv(
            raw, num_envs=num_envs, auto_reset=False,
            ignore_terminations=profile["ignore_terminations"], record_metrics=True,
        )
        shape = vector.single_action_space.shape
        if not isinstance(vector.single_action_space, gym.spaces.Box) or len(shape) != 1:
            raise ValueError(f"Expected a flat controller action space, found {vector.single_action_space}")
        expected_dimension = profile.get("action_dimension")
        if expected_dimension is not None and shape != (expected_dimension,):
            raise ValueError(f"Configured action dimension {expected_dimension} differs from {shape}")
        for frequency in ("sim_freq", "control_freq"):
            if frequency in profile and getattr(vector.unwrapped, frequency) != profile[frequency]:
                raise RuntimeError(f"ManiSkill {frequency} differs from the manifest")
        if bool(vector.unwrapped._enhanced_determinism) != profile["enhanced_determinism"]:
            raise RuntimeError("ManiSkill determinism configuration differs from the manifest")
        if vector.device.type != "cuda":
            raise RuntimeError(f"ManiSkill requires GPU physics, found {vector.device}")
        return vector
    except Exception:
        raw.close()
        raise


def state_rows(observation: Any, count: int) -> np.ndarray:
    """Preserve upstream state order; visual observations expose only their own state fields."""
    if isinstance(observation, Mapping):
        from mani_skill.utils.common import flatten_state_dict

        observation = flatten_state_dict({
            key: value for key, value in observation.items()
            if key not in {"sensor_data", "sensor_param"}
        }, use_torch=True)
    states = as_numpy(observation).astype(np.float32, copy=False)
    if states.ndim != 2 or states.shape[0] != count or not np.isfinite(states).all():
        raise ValueError(f"Invalid upstream state observation: {states.shape}")
    return np.ascontiguousarray(states)


def observation_rows(
    observation: Any, count: int, profile: dict[str, Any], *, environment: Any = None,
) -> tuple[dict[str, Any], ...]:
    states = state_rows(observation, count)
    rows = [{"state.proprioception": states[lane].copy()} for lane in range(count)]
    if profile.get("include_root_kinematics", False):
        robot = environment.unwrapped.agent.robot
        root_values = {
            "state.root_pose": (robot.pose.raw_pose, 7),
            "state.root_linear_velocity": (robot.root_linear_velocity, 3),
            "state.root_angular_velocity": (robot.root_angular_velocity, 3),
        }
        for name, (value, dimension) in root_values.items():
            array = as_numpy(value).astype(np.float32, copy=False)
            if array.shape != (count, dimension) or not np.isfinite(array).all():
                raise ValueError(f"Invalid native root kinematics {name}: {array.shape}")
            for lane, row in enumerate(rows):
                row[name] = np.ascontiguousarray(array[lane])
    if profile.get("include_cabinet_contact_state", False):
        native = environment.unwrapped
        agent = native.agent
        contact_values = {
            "state.robot_base_pose": (agent.base_link.pose.raw_pose, 7),
            "state.robot_tcp_pose": (agent.tcp.pose.raw_pose, 7),
            "state.target_handle_position": (native.handle_link_positions(), 3),
            "state.left_finger_pose": (agent.finger1_link.pose.raw_pose, 7),
            "state.right_finger_pose": (agent.finger2_link.pose.raw_pose, 7),
            "state.left_target_contact_force": (
                native.scene.get_pairwise_contact_forces(agent.finger1_link, native.handle_link), 3),
            "state.right_target_contact_force": (
                native.scene.get_pairwise_contact_forces(agent.finger2_link, native.handle_link), 3),
        }
        for name, (value, dimension) in contact_values.items():
            array = as_numpy(value).astype(np.float32, copy=False)
            if array.shape != (count, dimension) or not np.isfinite(array).all():
                raise ValueError(f"Invalid native cabinet state {name}: {array.shape}")
            for lane, row in enumerate(rows):
                row[name] = np.ascontiguousarray(array[lane])
    for feature, camera in profile.get("cameras", {}).items():
        images = as_numpy(observation["sensor_data"][camera]["rgb"])
        if images.ndim != 4 or images.shape[0] != count or images.shape[-1] != 3:
            raise ValueError(f"Camera {camera!r} returned invalid RGB shape {images.shape}")
        if images.dtype != np.uint8:
            # Bridge's overlay promotes RGB through its integer segmentation mask.
            if (profile["obs_mode"] != "rgb+segmentation"
                    or not np.issubdtype(images.dtype, np.integer)
                    or images.min() < 0 or images.max() > 255):
                raise ValueError(f"Camera {camera!r} must return uint8 RGB pixels, found {images.dtype}")
            images = images.astype(np.uint8)
        for lane in range(count):
            rows[lane][feature] = np.ascontiguousarray(images[lane])
    for feature, camera in profile.get("depth_cameras", {}).items():
        depth = as_numpy(observation["sensor_data"][camera]["depth"]).astype(np.float32, copy=False)
        if (depth.ndim != 4 or depth.shape[0] != count or depth.shape[-1] != 1
                or not np.isfinite(depth).all()):
            raise ValueError(f"Camera {camera!r} returned invalid native depth {depth.shape}")
        for lane, row in enumerate(rows):
            row[feature] = np.ascontiguousarray(depth[lane])
    if profile.get("instruction"):
        for row in rows:
            row["language.instruction"] = profile["instruction"]
    return tuple(rows)


def root_joint_metadata(environment: Any, profile: dict[str, Any]) -> dict[str, Any]:
    if not (profile.get("include_root_kinematics", False)
            or profile.get("include_joint_names", False)
            or profile.get("include_cabinet_contact_state", False)):
        return {}
    agent = environment.unwrapped.agent
    configs = getattr(agent.controller, "configs", None)
    configs = configs.values() if configs is not None else (agent.controller.config,)
    return {
        "active_joint_names": [joint.name for joint in agent.robot.get_active_joints()],
        "action_joint_names": [name for config in configs for name in config.joint_names],
    }


def observation_digest(features: dict[str, Any]) -> str:
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


def _episode_id(profile_id: str, index: int, seed: int) -> str:
    return f"{profile_id}-episode-{index:06d}-seed-{seed}"


def evaluate(
    item: WorkItem, *, worker_id: int, client: PolicyClient,
    profile: dict[str, Any], policy_batch_size: int, suite: str,
) -> WorkResult:
    import torch

    profile_id = profile["profile_id"]
    seeds = [int(seed) for seed in item.payload["seeds"]]
    start = int(item.payload["episode_start"])
    if (item.payload["profile_id"] != profile_id or item.group_id != profile_id
            or not seeds or len(seeds) != item.expected_results):
        raise ValueError("ManiSkill work item differs from its task profile")
    if seeds != profile["seeds"][start:start + len(seeds)]:
        raise ValueError("ManiSkill work item differs from the frozen seed manifest")
    count = len(seeds)
    episode_ids = [_episode_id(profile_id, start + lane, seed) for lane, seed in enumerate(seeds)]
    initialized = []
    trajectory_files = []
    trajectory_paths = []
    environment = None
    metadata: dict[str, Any] = {}
    records: tuple[EvaluationRecord, ...] = ()
    inference_wait_s = observation_prepare_s = step_with_observation_s = 0.0
    rpc_calls = 0
    try:
        phase_started = time.perf_counter()
        environment = create_environment(profile, count)
        metadata["environment_initialization_s"] = time.perf_counter() - phase_started
        action_dimension = int(environment.single_action_space.shape[0])
        phase_started = time.perf_counter()
        observation, _ = environment.reset(seed=seeds)
        metadata["environment_reset_s"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        initial_features = observation_rows(observation, count, profile, environment=environment)
        initial_states = [row["state.proprioception"] for row in initial_features]
        initial_state_digests = [
            hashlib.sha256(np.ascontiguousarray(state, dtype="<f4").tobytes()).hexdigest()
            for state in initial_states
        ]
        initial_observation_digests = [observation_digest(row) for row in initial_features]
        metadata["initial_observation_prepare_s"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        for lane, episode_id in enumerate(episode_ids):
            client.reset(EpisodeContext(
                episode_id=episode_id, task_id=profile["task_id"],
                metadata={
                    "seed": seeds[lane], "episode_index": start + lane, "vector_lane": lane,
                    "profile_id": profile_id, "suite": suite,
                    "robot_uid": profile["robot_uid"], "control_mode": profile["control_mode"],
                    "obs_mode": profile["obs_mode"], "action_dimension": action_dimension,
                    "sim_freq": profile["sim_freq"], "control_freq": profile["control_freq"],
                    **root_joint_metadata(environment, profile),
                },
            ))
            initialized.append(episode_id)
        metadata["policy_reset_s"] = time.perf_counter() - phase_started
        native_horizon = int(profile["max_episode_steps"])
        horizon = int(profile.get("diagnostic_step_limit", native_horizon))
        diagnostic = "diagnostic_step_limit" in profile
        trajectory_dir = os.environ.get("ROBOSCOPE_MANISKILL_TRAJECTORY_DIR")
        if trajectory_dir:
            destination = Path(trajectory_dir)
            destination.mkdir(parents=True, exist_ok=True)
            for episode_id in episode_ids:
                # Exclusive attempt files survive failures without blocking a retry.
                # Comparisons only discover canonical *.jsonl files from complete groups.
                trajectory = tempfile.NamedTemporaryFile(  # noqa: SIM115 -- group closes in finally.
                    mode="w", encoding="utf-8", dir=destination,
                    prefix=f"{episode_id}.attempt-{os.getpid()}-", suffix=".jsonl.partial",
                    delete=False,
                )
                trajectory_files.append(trajectory)
                trajectory_paths.append((Path(trajectory.name), destination / f"{episode_id}.jsonl"))
        first_actions = None
        final_info = None
        truncation_values = None
        started = time.perf_counter()
        for step in range(horizon):
            phase_started = time.perf_counter()
            features = observation_rows(observation, count, profile, environment=environment)
            observations = tuple(Observation(
                episode_id=episode_ids[lane], step=step, timestamp_s=time.time(),
                features=features[lane],
            ) for lane in range(count))
            observation_prepare_s += time.perf_counter() - phase_started
            actions = []
            for offset in range(0, count, policy_batch_size):
                batch = observations[offset:offset + policy_batch_size]
                phase_started = time.perf_counter()
                chunks = client.act_batch(batch) if len(batch) > 1 else (client.act(batch[0]),)
                inference_wait_s += time.perf_counter() - phase_started
                rpc_calls += 1
                if len(chunks) != len(batch):
                    raise RuntimeError("Policy batch returned a different number of actions")
                for chunk in chunks:
                    values = np.asarray(chunk.actions, dtype=np.float32)
                    if values.shape == (1, action_dimension):
                        values = values[0]
                    if values.shape != (action_dimension,) or not np.isfinite(values).all():
                        raise ValueError(
                            f"Expected one {action_dimension}D controller action, found {values.shape}"
                        )
                    actions.append(values)
            if step == 0:
                first_actions = [value.tolist() for value in actions]
            for lane, trajectory in enumerate(trajectory_files):
                trajectory.write(json.dumps({
                    "step": step, "observation_sha256": observation_digest(features[lane]),
                    "action_sha256": hashlib.sha256(
                        np.ascontiguousarray(actions[lane], dtype="<f4").tobytes(),
                    ).hexdigest(),
                }) + "\n")
            action_tensor = torch.as_tensor(
                np.stack(actions), dtype=torch.float32, device=environment.device,
            )
            phase_started = time.perf_counter()
            observation, _reward, terminated, truncated, info = environment.step(action_tensor)
            step_with_observation_s += time.perf_counter() - phase_started
            termination_values = as_numpy(terminated).reshape(-1)
            truncation_values = as_numpy(truncated).reshape(-1)
            if termination_values.size != count or truncation_values.size != count:
                raise RuntimeError("ManiSkill returned invalid per-environment terminal masks")
            if termination_values.any():
                raise RuntimeError("Upstream ignore_terminations profile unexpectedly terminated")
            if step < native_horizon - 1 and truncation_values.any():
                raise RuntimeError("ManiSkill episode truncated before the configured horizon")
            final_info = info
        expected_truncated = horizon == native_horizon
        if (truncation_values is None or final_info is None
                or not np.all(truncation_values == expected_truncated)):
            raise RuntimeError("ManiSkill did not complete every lane at the configured horizon")
        if trajectory_files:
            terminal_features = observation_rows(observation, count, profile, environment=environment)
            for lane, trajectory in enumerate(trajectory_files):
                trajectory.write(json.dumps({
                    "step": horizon,
                    "observation_sha256": observation_digest(terminal_features[lane]),
                    "action_sha256": None,
                }) + "\n")
        elapsed_s = time.perf_counter() - started
        metrics = final_info["episode"]
        returns = as_numpy(metrics["return"]).reshape(-1)
        success_available = "success_once" in metrics and "success_at_end" in metrics
        success_once = as_numpy(metrics["success_once"]).reshape(-1) if success_available else None
        success_end = as_numpy(metrics["success_at_end"]).reshape(-1) if success_available else None
        lengths = as_numpy(metrics["episode_len"]).reshape(-1)
        if (len(returns) != count or not np.isfinite(returns).all()
                or len(lengths) != count or not np.all(lengths == horizon)):
            raise RuntimeError("ManiSkill episode metrics do not cover the vector group")
        if success_available and (len(success_once) != count or len(success_end) != count):
            raise RuntimeError("ManiSkill success metrics do not cover the vector group")
        if profile["success_metric"] != "return" and not success_available:
            raise RuntimeError("ManiSkill task has no success metric; select its native return metric")
        selected_success = {
            "success_once": success_once, "success_at_end": success_end,
            "return": np.zeros(count, dtype=bool),
        }[profile["success_metric"]]
        records = tuple(EvaluationRecord(
            record_id=episode_ids[lane], success=bool(selected_success[lane]),
            steps=horizon, elapsed_s=elapsed_s,
            metadata={
                "task_id": profile["task_id"], "profile_id": profile_id, "suite": suite,
                "episode_index": start + lane, "seed": seed, "vector_lane": lane,
                "robot_uid": profile["robot_uid"], "control_mode": profile["control_mode"],
                "obs_mode": profile["obs_mode"], "action_dimension": action_dimension,
                "initial_state_sha256": initial_state_digests[lane],
                "initial_observation_sha256": initial_observation_digests[lane],
                "first_action": first_actions[lane],
                **({"initial_state": initial_states[lane].tolist()}
                   if start + lane in INITIAL_STATE_SNAPSHOT_EPISODE_INDICES else {}),
                "success_metric": profile["success_metric"],
                "episode_return": float(returns[lane]), "success_available": success_available,
                "success_once": bool(success_once[lane]) if success_available else None,
                "success_at_end": bool(success_end[lane]) if success_available else None,
                "terminated": False, "truncated": bool(truncation_values[lane]),
                "protocol": PROTOCOL_ID,
                **({"diagnostic_step_limit": horizon, "native_max_episode_steps": native_horizon}
                   if diagnostic else {}),
            },
        ) for lane, seed in enumerate(seeds))
        metadata.update({
            "task_id": profile["task_id"], "profile_id": profile_id, "suite": suite,
            "vector_envs": count, "policy_batch_size": min(policy_batch_size, count),
            "action_dimension": action_dimension,
            "observation_dimension": int(initial_states[0].shape[0]),
            "inference_wait_s": inference_wait_s, "inference_rpc_calls": rpc_calls,
            "inference_round_trip_mean_s": inference_wait_s / rpc_calls if rpc_calls else 0.0,
            "observation_prepare_cpu_wall_s": observation_prepare_s,
            "step_with_observation_cpu_wall_s": step_with_observation_s,
            "rollout_s": elapsed_s, "actual_simulation_steps": count * horizon,
            "vector_step_calls": horizon,
            "evaluation_kind": "integration_smoke" if diagnostic else "full_horizon",
            "trajectory_diagnostics": bool(trajectory_files),
            "timing_note": "CPU wall intervals; env.step includes physics and observation/render work. "
                           "GPU execution is asynchronous; these are not isolated GPU kernel timings.",
        })
    finally:
        for trajectory in trajectory_files:
            trajectory.close()
        try:
            phase_started = time.perf_counter()
            for episode_id in initialized:
                client.end_episode(episode_id)
            metadata["policy_cleanup_s"] = time.perf_counter() - phase_started
        finally:
            if environment is not None:
                phase_started = time.perf_counter()
                environment.close()
                metadata["environment_close_s"] = time.perf_counter() - phase_started
    # Publish only after every lane completed and the group cleanup succeeded.
    for attempt, destination in trajectory_paths:
        attempt.replace(destination)
    return WorkResult(
        item_id=item.item_id, group_id=item.group_id, worker_id=worker_id,
        records=records, metadata=metadata,
    )


def self_test(profiles: tuple[dict[str, Any], ...], destination: Path) -> None:
    import torch

    evidence = []
    for profile in profiles:
        for count in sorted({1, min(2, len(profile["seeds"]))}):
            environment = create_environment(profile, count)
            try:
                observation, _ = environment.reset(seed=profile["seeds"][:count])
                first_features = observation_rows(observation, count, profile, environment=environment)
                dimension = int(environment.single_action_space.shape[0])
                action = torch.zeros((count, dimension), dtype=torch.float32, device=environment.device)
                for _ in range(int(profile["max_episode_steps"])):
                    observation, _, terminated, truncated, info = environment.step(action)
                    final_features = observation_rows(observation, count, profile, environment=environment)
                if as_numpy(terminated).any() or not as_numpy(truncated).all():
                    raise RuntimeError("Self-test did not preserve fixed-horizon terminal masks")
                render_frame = as_numpy(environment.render())
                if (render_frame.ndim != 4 or render_frame.shape[0] != count or render_frame.shape[-1] != 3
                        or render_frame.dtype != np.uint8 or not np.ptp(render_frame)):
                    raise RuntimeError("ManiSkill self-test did not render a valid nonconstant RGB frame")
                images = {}
                for feature, camera in profile.get("cameras", {}).items():
                    camera_frame = final_features[0][feature]
                    if not np.ptp(camera_frame):
                        raise RuntimeError(f"RGB camera {camera!r} produced a constant frame")
                    images[feature] = {
                        "camera": camera, "shape": list(camera_frame.shape), "dtype": str(camera_frame.dtype),
                        "source_device": str(observation["sensor_data"][camera]["rgb"].device),
                        "nonconstant": True, "first_sha256": hashlib.sha256(
                            first_features[0][feature].tobytes()).hexdigest(),
                        "last_sha256": hashlib.sha256(camera_frame.tobytes()).hexdigest(),
                    }
                evidence.append({
                    "profile_id": profile["profile_id"], "task_id": profile["task_id"],
                    "obs_mode": profile["obs_mode"], "robot_uid": profile["robot_uid"],
                    "control_mode": profile["control_mode"], "vector_envs": count,
                    "state_dimension": int(first_features[0]["state.proprioception"].shape[0]),
                    "action_dimension": dimension, "simulation_device": str(environment.device),
                    "steps": count * int(profile["max_episode_steps"]),
                    "sim_freq": environment.unwrapped.sim_freq,
                    "control_freq": environment.unwrapped.control_freq,
                    "images": images,
                    "render_frame": {
                        "shape": list(render_frame.shape), "dtype": str(render_frame.dtype),
                        "nonconstant": True,
                    },
                    "success_once": (
                        as_numpy(info["episode"]["success_once"]).tolist()
                        if "success_once" in info["episode"] else None
                    ),
                    "episode_return": as_numpy(info["episode"]["return"]).tolist(),
                })
            finally:
                environment.close()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({
        "benchmark": "maniskill", "protocol": PROTOCOL_ID, "python": sys.version,
        "gpu_vector_checks": evidence, "real_model_loaded": False,
        "scope": "Representative state, dynamic controller and RGB integration checks",
    }, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--suite")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-output", type=Path)
    parser.add_argument("--worker-id", type=int)
    parser.add_argument("--policy-socket", type=Path)
    parser.add_argument("--queue-dir", type=Path)
    parser.add_argument("--worker-state-dir", type=Path)
    parser.add_argument("--policy-batch-size", type=int, default=1)
    args = parser.parse_args()
    manifest = load_benchmark_manifest(args.benchmark_manifest)
    os.environ["HOME"] = str(manifest.runtime_path("home"))
    os.environ["MS_ASSET_DIR"] = str(manifest.runtime_path("asset_dir"))
    library = manifest.runtime_path("physx_gpu_library")
    protocol = manifest.benchmark_protocols.get(BenchmarkName.MANISKILL, {})
    expected_library_hash = physx_library_sha256(protocol)
    if not library.is_file() or hashlib.sha256(library.read_bytes()).hexdigest() != expected_library_hash:
        raise RuntimeError("Pinned install-local PhysX GPU library is missing or has changed")
    if args.self_test:
        if args.self_test_output is None:
            parser.error("--self-test requires --self-test-output")
        self_test(self_test_profiles(protocol), args.self_test_output)
        return
    if not args.suite or any(value is None for value in (
        args.worker_id, args.policy_socket, args.queue_dir, args.worker_state_dir,
    )):
        parser.error("worker execution requires suite, id, Policy socket, queue, and state directory")
    manifest.validate_request(BenchmarkName.MANISKILL, args.suite)
    profiles = {profile["profile_id"]: profile for profile in validate_profiles(protocol, args.suite)}
    if args.policy_batch_size < 1:
        raise ValueError("Policy batch size must be positive")
    with PolicyClient(args.policy_socket, timeout_s=300) as client:
        WorkerLoop(
            worker_id=args.worker_id, queue_dir=args.queue_dir, state_dir=args.worker_state_dir,
            recycle_on_context_change=True,
        ).run(lambda item: evaluate(
            item, worker_id=args.worker_id, client=client,
            profile=profiles[item.payload["profile_id"]],
            policy_batch_size=args.policy_batch_size, suite=args.suite,
        ))


if __name__ == "__main__":
    main()
