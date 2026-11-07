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

from .profile import PROTOCOL_ID, self_test_profiles, validate_profiles


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def create_environment(profile: dict[str, Any], num_envs: int):
    """Construct one task exactly as declared, with upstream controller semantics."""
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 -- register pinned upstream tasks
    from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    robots = profile["robot_uid"]
    control = profile["control_mode"]
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
        num_envs=num_envs, robot_uids=robots, obs_mode=profile["obs_mode"],
        control_mode=control, sim_backend=profile["sim_backend"],
        render_backend=profile["render_backend"], render_mode="rgb_array",
        max_episode_steps=profile["max_episode_steps"],
        reconfiguration_freq=profile["reconfiguration_freq"],
        enhanced_determinism=profile["enhanced_determinism"],
        sim_config=sim_config, sensor_configs=deepcopy(profile.get("sensor_configs", {})),
    )
    raw = gym.make(profile["task_id"], **kwargs)
    try:
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
    observation: Any, count: int, profile: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    states = state_rows(observation, count)
    rows = [{"state.proprioception": states[lane].copy()} for lane in range(count)]
    for feature, camera in profile.get("cameras", {}).items():
        images = as_numpy(observation["sensor_data"][camera]["rgb"])
        if images.ndim != 4 or images.shape[0] != count or images.shape[-1] != 3:
            raise ValueError(f"Camera {camera!r} returned invalid RGB shape {images.shape}")
        if images.dtype != np.uint8:
            raise ValueError(f"Camera {camera!r} must return native uint8 RGB, found {images.dtype}")
        for lane in range(count):
            rows[lane][feature] = np.ascontiguousarray(images[lane])
    if profile.get("instruction"):
        for row in rows:
            row["language.instruction"] = profile["instruction"]
    return tuple(rows)


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
        initial_features = observation_rows(observation, count, profile)
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
                },
            ))
            initialized.append(episode_id)
        metadata["policy_reset_s"] = time.perf_counter() - phase_started
        horizon = int(profile["max_episode_steps"])
        first_actions = None
        final_info = None
        truncation_values = None
        started = time.perf_counter()
        for step in range(horizon):
            phase_started = time.perf_counter()
            features = observation_rows(observation, count, profile)
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
            if step < horizon - 1 and truncation_values.any():
                raise RuntimeError("ManiSkill episode truncated before the configured horizon")
            final_info = info
        if truncation_values is None or not truncation_values.all() or final_info is None:
            raise RuntimeError("ManiSkill did not complete every lane at the configured horizon")
        elapsed_s = time.perf_counter() - started
        metrics = final_info["episode"]
        success_once = as_numpy(metrics["success_once"]).reshape(-1)
        success_end = as_numpy(metrics["success_at_end"]).reshape(-1)
        lengths = as_numpy(metrics["episode_len"]).reshape(-1)
        if len(success_once) != count or len(success_end) != count or not np.all(lengths == horizon):
            raise RuntimeError("ManiSkill episode metrics do not cover the vector group")
        selected_success = success_once if profile["success_metric"] == "success_once" else success_end
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
                   if start + lane in {0, 1, 49, 99} else {}),
                "success_metric": profile["success_metric"],
                "success_once": bool(success_once[lane]), "success_at_end": bool(success_end[lane]),
                "terminated": False, "truncated": True, "protocol": PROTOCOL_ID,
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
            "timing_note": "CPU wall intervals; env.step includes physics and observation/render work. "
                           "GPU execution is asynchronous; these are not isolated GPU kernel timings.",
        })
    finally:
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
                first_features = observation_rows(observation, count, profile)
                dimension = int(environment.single_action_space.shape[0])
                action = torch.zeros((count, dimension), dtype=torch.float32, device=environment.device)
                for _ in range(int(profile["max_episode_steps"])):
                    observation, _, terminated, truncated, info = environment.step(action)
                    final_features = observation_rows(observation, count, profile)
                if as_numpy(terminated).any() or not as_numpy(truncated).all():
                    raise RuntimeError("Self-test did not preserve fixed-horizon terminal masks")
                images = {}
                for feature, camera in profile.get("cameras", {}).items():
                    frame = final_features[0][feature]
                    if not np.ptp(frame):
                        raise RuntimeError(f"RGB camera {camera!r} produced a constant frame")
                    images[feature] = {
                        "camera": camera, "shape": list(frame.shape), "dtype": str(frame.dtype),
                        "source_device": str(observation["sensor_data"][camera]["rgb"].device),
                        "nonconstant": True, "first_sha256": hashlib.sha256(
                            first_features[0][feature].tobytes()).hexdigest(),
                        "last_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
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
                    "images": images, "success_once": as_numpy(info["episode"]["success_once"]).tolist(),
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
    expected_library_hash = "4c582a16509a71faf5592fe9708586dfcc7ab61ae932eabc1ddd81f290818706"
    if not library.is_file() or hashlib.sha256(library.read_bytes()).hexdigest() != expected_library_hash:
        raise RuntimeError("Pinned install-local PhysX GPU library is missing or has changed")
    protocol = manifest.benchmark_protocols.get(BenchmarkName.MANISKILL, {})
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
        ).run(lambda item: evaluate(
            item, worker_id=args.worker_id, client=client,
            profile=profiles[item.payload["profile_id"]],
            policy_batch_size=args.policy_batch_size, suite=args.suite,
        ))


if __name__ == "__main__":
    main()
