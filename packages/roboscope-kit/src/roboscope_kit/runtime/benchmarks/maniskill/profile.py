# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0
"""Resolve named ManiSkill task profiles without importing the simulator."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

PROTOCOL_ID = "maniskill-v1"
_DEFAULTS: dict[str, Any] = {
    "sim_backend": "physx_cuda",
    "render_backend": "gpu",
    "reconfiguration_freq": 1,
    "ignore_terminations": True,
    "success_metric": "success_once",
    "action_chunk_size": 1,
    "enhanced_determinism": False,
    "use_default_robot": False,
    "include_root_kinematics": False,
    "include_joint_names": False,
    "include_cabinet_contact_state": False,
    "sim_freq": 100,
    "control_freq": 20,
    "env_kwargs": {},
    "sensor_configs": {},
    "cameras": {},
    "depth_cameras": {},
}
_CAMERA_FEATURES = {"images.primary", "images.wrist", "images.auxiliary"}
_DEPTH_FEATURES = {"state.depth_head", "state.depth_hand"}


def _profiles(protocol: dict[str, Any]) -> dict[str, Any]:
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"ManiSkill protocol_id must be {PROTOCOL_ID}")
    vector_envs(protocol)
    physx_library_sha256(protocol)
    profiles = protocol.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("ManiSkill manifest must define nonempty profiles")
    return profiles


def vector_envs(protocol: dict[str, Any]) -> int:
    """Vector width changes seeded initialization and is a protocol value, not a tuning bound."""
    width = protocol.get("vector_envs")
    if type(width) is not int or width < 1:
        raise ValueError("ManiSkill protocol.vector_envs must be a positive integer")
    return width


def physx_library_sha256(protocol: dict[str, Any]) -> str:
    """Read the single declared binary pin used by the builder and all evaluation paths."""
    digest = protocol.get("physx_gpu_library_sha256")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)):
        raise ValueError("ManiSkill protocol requires a valid physx_gpu_library_sha256")
    return digest


def _resolve(protocol: dict[str, Any], profile_id: str) -> dict[str, Any]:
    raw = _profiles(protocol).get(profile_id)
    if not isinstance(raw, dict):
        raise ValueError(f"ManiSkill profile {profile_id!r} is missing")
    profile = {**deepcopy(_DEFAULTS), **deepcopy(raw), "profile_id": profile_id}
    for key in ("task_id", "obs_mode"):
        if not isinstance(profile.get(key), str) or not profile[key]:
            raise ValueError(f"ManiSkill {profile_id}.{key} must be a nonempty string")
    for key in ("robot_uid", "control_mode"):
        value = profile.get(key)
        if not (isinstance(value, str) and value) and not (
            isinstance(value, list) and value
            and all(isinstance(item, str) and item for item in value)
        ):
            raise ValueError(f"ManiSkill {profile_id}.{key} must be a string or string list")
    if isinstance(profile["control_mode"], list) and (
        not isinstance(profile["robot_uid"], list)
        or len(profile["control_mode"]) != len(profile["robot_uid"])
    ):
        raise ValueError(f"ManiSkill {profile_id} must pair each robot with one controller")
    for key in ("enhanced_determinism", "use_default_robot", "include_root_kinematics",
                "include_cabinet_contact_state", "include_joint_names"):
        if type(profile[key]) is not bool:
            raise ValueError(f"ManiSkill {profile_id}.{key} must be a boolean")
    if profile["use_default_robot"] and not isinstance(profile["robot_uid"], str):
        raise ValueError(f"ManiSkill {profile_id}.use_default_robot requires one robot UID")
    if type(profile["reconfiguration_freq"]) is not int or profile["reconfiguration_freq"] < 0:
        raise ValueError(f"ManiSkill {profile_id}.reconfiguration_freq must be non-negative")
    if profile["obs_mode"] not in {"state", "state_dict", "rgb", "rgb+segmentation", "depth", "rgb+depth"}:
        raise ValueError(f"Unsupported ManiSkill observation mode {profile['obs_mode']!r}")
    if profile["ignore_terminations"] is not True or profile["action_chunk_size"] != 1:
        raise ValueError("ManiSkill v1 requires full-horizon episodes and one-step action chunks")
    if profile["success_metric"] not in {"success_once", "success_at_end", "return"}:
        raise ValueError("ManiSkill success_metric must be success_once, success_at_end or return")
    if profile["sim_backend"] != "physx_cuda":
        raise ValueError("ManiSkill v1 requires physx_cuda simulation")
    for key in ("max_episode_steps", "sim_freq", "control_freq"):
        if type(profile.get(key)) is not int or profile[key] < 1:
            raise ValueError(f"ManiSkill {profile_id}.{key} must be a positive integer")
    diagnostic_limit = profile.get("diagnostic_step_limit", protocol.get("diagnostic_step_limit"))
    if "diagnostic_step_limit" in profile or "diagnostic_step_limit" in protocol:
        if type(diagnostic_limit) is not int or not 1 <= diagnostic_limit <= profile["max_episode_steps"]:
            raise ValueError("ManiSkill diagnostic_step_limit must be within the native episode horizon")
        profile["diagnostic_step_limit"] = diagnostic_limit
    dimension = profile.get("action_dimension")
    if dimension is not None and (type(dimension) is not int or dimension < 1):
        raise ValueError(f"ManiSkill {profile_id}.action_dimension must be positive when specified")
    for key in ("env_kwargs", "sensor_configs", "cameras", "depth_cameras"):
        if not isinstance(profile[key], dict):
            raise ValueError(f"ManiSkill {profile_id}.{key} must be an object")
    if set(profile["cameras"]) - _CAMERA_FEATURES or any(
        not isinstance(camera, str) or not camera for camera in profile["cameras"].values()
    ):
        raise ValueError(f"ManiSkill {profile_id}.cameras must map declared image features to cameras")
    if set(profile["depth_cameras"]) - _DEPTH_FEATURES or any(
        not isinstance(camera, str) or not camera for camera in profile["depth_cameras"].values()
    ):
        raise ValueError(f"ManiSkill {profile_id}.depth_cameras must map declared depth features to cameras")
    if profile["depth_cameras"] and "depth" not in profile["obs_mode"]:
        raise ValueError(f"ManiSkill {profile_id}.depth_cameras requires native depth observations")
    profile.setdefault("instruction", profile["task_id"])
    if not isinstance(profile["instruction"], str):
        raise ValueError(f"ManiSkill {profile_id}.instruction must be a string")
    episodes = profile.get("episodes_per_task", protocol.get("episodes_per_task", 100))
    if type(episodes) is not int or episodes < 1:
        raise ValueError("ManiSkill episodes_per_task must be a positive integer")
    if episodes % vector_envs(protocol):
        raise ValueError("ManiSkill prescribed episodes must contain complete frozen vector groups")
    seeds = profile.get("seeds", protocol.get("seeds", list(range(episodes))))
    if not isinstance(seeds, list) or len(seeds) != episodes:
        raise ValueError(f"ManiSkill {profile_id}.seeds must cover every episode exactly")
    if any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError(f"ManiSkill {profile_id}.seeds must be unique non-negative integers")
    profile["episodes_per_task"] = episodes
    profile["seeds"] = list(seeds)
    return profile


def validate_profiles(protocol: dict[str, Any], suite: str) -> tuple[dict[str, Any], ...]:
    """Resolve every task configuration in one explicitly declared suite."""
    _profiles(protocol)
    suites = protocol.get("suites", {})
    profile_ids = suites.get(suite) if isinstance(suites, dict) else None
    if not isinstance(profile_ids, list) or not profile_ids or any(
        not isinstance(profile_id, str) or not profile_id for profile_id in profile_ids
    ):
        raise ValueError(f"ManiSkill suite {suite!r} must list nonempty profile IDs")
    if len(set(profile_ids)) != len(profile_ids):
        raise ValueError(f"ManiSkill suite {suite!r} contains duplicate profiles")
    return tuple(_resolve(protocol, profile_id) for profile_id in profile_ids)


def self_test_profiles(protocol: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Choose representative integration checks without evaluating every task."""
    profiles = _profiles(protocol)
    profile_ids = protocol.get("self_test_profiles", [next(iter(profiles))])
    if not isinstance(profile_ids, list) or not profile_ids or any(
        not isinstance(profile_id, str) or not profile_id for profile_id in profile_ids
    ) or len(set(profile_ids)) != len(profile_ids):
        raise ValueError("ManiSkill self_test_profiles must list unique profile IDs")
    return tuple(_resolve(protocol, profile_id) for profile_id in profile_ids)
