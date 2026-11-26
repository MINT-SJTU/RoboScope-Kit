# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0
"""Resolve official PPO environment configuration without importing Kit or a simulator."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

SOURCE_REVISION = "baab60ede2e89167c1b7aaed41a9aa8e690a9d1e"
CONFIGURATION_SOURCE = "pinned_upstream_AST_and_official_checkpoint_dataset_path"
OFFICIAL_TASK_IDS = frozenset({
    "PickCube-v1", "RollBall-v1", "PushT-v1", "LiftPegUpright-v1", "PokeCube-v1",
    "PullCube-v1", "PushCube-v1", "StackCube-v1", "TwoRobotPickCube-v1",
    "TwoRobotStackCube-v1", "AnymalC-Reach-v1",
})
_PPO = "examples/baselines/ppo/ppo.py"
_BASE = "mani_skill/envs/sapien_env.py"
_TYPES = "mani_skill/utils/structs/types.py"
_VECTOR = "mani_skill/vector/wrappers/gymnasium.py"
SOURCE_HASHES = {
    _PPO: "d29f25cd4009a400807e682b07947a2c1386802f0a3bce5e3a770d20aec570f3",
    _BASE: "03e765902d42dfd523daaffa00225ad13fab7fecc992063d53b070ff6df79691",
    _TYPES: "7837b3412c6dbbc2328c87b893a9bc6a6427ede608b8ff622dab576ebec5607f",
    _VECTOR: "88b8be602c7841bcc1857b33b2042cb6cb37997581583e023279387dc19f6428",
    "mani_skill/envs/tasks/quadruped/quadruped_reach.py":
        "68fe6b17e3a459a5f16f94988aeaebabc70cfbce6d542ae5ba230a3174e89d99",
    "mani_skill/envs/tasks/tabletop/lift_peg_upright.py":
        "5523403406fd9fdc5672f73bb095959b22ff9b91bfcfb5186b5afbfd53f8840d",
    "mani_skill/envs/tasks/tabletop/pick_cube.py":
        "a935b95d59c4fb34d1cf9ccb00011e733440cabd1c48af1a0a7a781f3911038c",
    "mani_skill/envs/tasks/tabletop/poke_cube.py":
        "baeb2623205fa88a014a9fa1a0142357344d7e8fac6a148d4fe866bda0e8c611",
    "mani_skill/envs/tasks/tabletop/pull_cube.py":
        "8e7f1cebe4bd06b83012d118896f87b5ae57d20f00da61d4e504784d19e646ae",
    "mani_skill/envs/tasks/tabletop/push_cube.py":
        "d64a14a75def07cf20293ff7bdb3e3b52a6982c0b57d9186f4b0b273b03eb049",
    "mani_skill/envs/tasks/tabletop/push_t.py":
        "fb2d06aa6f820af03f30821e4c6eea54081a3b0e4e1d829b88269b93cb6f3d36",
    "mani_skill/envs/tasks/tabletop/roll_ball.py":
        "47a0c297d9fc171e6ce32f5e2c4e0346a29a9d233326a5c57163bfdfcf064061",
    "mani_skill/envs/tasks/tabletop/stack_cube.py":
        "ff76ae608f79ba3ea13334e61ccb72f291660f9461895c70e3c30f8390694440",
    "mani_skill/envs/tasks/tabletop/two_robot_pick_cube.py":
        "a2cb80e811677b6524808a7cb30ec9321ffcdf2fd16b2ededf488d3490c789bd",
    "mani_skill/envs/tasks/tabletop/two_robot_stack_cube.py":
        "95791d4b8cd113d280e23cebf6637f4f2b6adb8a60e560f662561cb2f0e5af82",
}


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def _field(node: ast.ClassDef, name: str) -> Any:
    for value in node.body:
        if (isinstance(value, ast.AnnAssign) and isinstance(value.target, ast.Name)
                and value.target.id == name):
            return ast.literal_eval(value.value)
        if isinstance(value, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in value.targets
        ):
            return ast.literal_eval(value.value)
    raise ValueError(f"Missing pinned upstream field {node.name}.{name}")


def _default(node: ast.ClassDef, name: str) -> Any:
    init = next(value for value in node.body
                if isinstance(value, ast.FunctionDef) and value.name == "__init__")
    pairs = list(zip(init.args.kwonlyargs, init.args.kw_defaults, strict=True))
    if init.args.defaults:
        pairs.extend(zip(init.args.args[-len(init.args.defaults):], init.args.defaults, strict=True))
    return next(ast.literal_eval(value) for arg, value in pairs if arg.arg == name)


def resolve_official_profile(
    upstream_root: Path, task_id: str, checkpoint: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use only pinned upstream source and the official checkpoint's dataset identity."""
    catalog = json.loads(Path(__file__).with_name("official-models.json").read_text(encoding="utf-8"))
    if (catalog["repository"] != "haosulab/ManiSkill_Demonstrations"
            or catalog["revision"] != "3558f72cf99d51d04a6ac4779e88b3da4bbf1628"):
        raise ValueError("Official PPO checkpoint catalog differs from its pinned author source")
    official = catalog["models"].get(task_id)
    if (task_id not in OFFICIAL_TASK_IDS or official is None
            or any(checkpoint.get(key) != official[key] for key in ("dataset_path", "sha256"))
            or any(checkpoint.get(key) is not None for key in (
                "state_projection", "state_prefix_length", "action_clip_bounds",
            ))):
        raise ValueError(f"No independent upstream PPO reference for {task_id!r} and this checkpoint")
    match = re.fullmatch(
        r"demos/([^/]+)/rl/ppo_(pd_ee_delta_pos|pd_ee_delta_pose|pd_joint_delta_pos)_ckpt\.pt",
        official["dataset_path"],
    )
    if match is None or match.group(1) != task_id:
        raise ValueError("Official checkpoint path does not identify this task/controller")
    trees = {}
    for relative, expected in SOURCE_HASHES.items():
        path = upstream_root / relative
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Pinned upstream PPO source differs: {relative}")
        trees[relative] = ast.parse(data, filename=relative)
    args = _class(trees[_PPO], "Args")
    base = _class(trees[_BASE], "BaseEnv")
    sim = _class(trees[_TYPES], "SimConfig")
    env_assignment = next(
        node for node in ast.walk(trees[_PPO]) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "env_kwargs" for target in node.targets)
    )
    environment = {value.arg: ast.literal_eval(value.value) for value in env_assignment.value.keywords}
    registration = None
    for relative, tree in trees.items():
        if "/tasks/" not in relative:
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for deco in node.decorator_list:
                if (isinstance(deco, ast.Call) and isinstance(deco.func, ast.Name)
                        and deco.func.id == "register_env" and deco.args
                        and ast.literal_eval(deco.args[0]) == task_id):
                    registration = (node, deco, relative)
    if registration is None:
        raise ValueError(f"No pinned upstream task registration for {task_id}")
    node, deco, source_file = registration
    robots = _default(node, "robot_uids")
    profile = {
        **environment, "task_id": task_id,
        "robot_uid": list(robots) if isinstance(robots, tuple) else robots,
        "control_mode": match.group(2),
        "max_episode_steps": next(ast.literal_eval(v.value) for v in deco.keywords
                                  if v.arg == "max_episode_steps"),
        "render_backend": _default(base, "render_backend"),
        "enhanced_determinism": _default(base, "enhanced_determinism"),
        "reward_mode": _field(base, "SUPPORTED_REWARD_MODES")[0],
        "reconfiguration_freq": _field(args, "eval_reconfiguration_freq"),
        "ignore_terminations": not _field(args, "eval_partial_reset"),
        # These pinned tasks do not override the upstream SimConfig frequencies.
        "sim_freq": _field(sim, "sim_freq"), "control_freq": _field(sim, "control_freq"),
        "env_kwargs": {}, "sensor_configs": {}, "cameras": {}, "depth_cameras": {},
        "include_root_kinematics": False, "include_joint_names": False,
        "include_cabinet_contact_state": False, "use_default_robot": False,
        "action_chunk_size": 1,
    }
    provenance = {
        "configuration_source": CONFIGURATION_SOURCE,
        "source_revision": SOURCE_REVISION, "source_sha256": dict(SOURCE_HASHES),
        "task_source": source_file, "checkpoint_repository": catalog["repository"],
        "checkpoint_revision": catalog["revision"], "checkpoint_dataset_path": official["dataset_path"],
        "checkpoint_sha256": official["sha256"],
        "upstream_arg_defaults": {key: _field(args, key) for key in (
            "seed", "num_eval_envs", "num_eval_steps", "control_mode",
            "eval_reconfiguration_freq", "eval_partial_reset",
        )},
        "upstream_auto_reset": _default(_class(trees[_VECTOR], "ManiSkillVectorEnv"), "auto_reset"),
        "upstream_reset_schedule": "scalar seeded reset, followed by an unseeded reset before evaluation",
        "scope": "Independent environment configuration only; not original-evaluator paired acceptance",
    }
    return profile, provenance


def require_matching_profile(profile: dict[str, Any], independent: dict[str, Any]) -> None:
    """Reject shared-wrong Kit values before independently constructing an environment."""
    for key, value in independent.items():
        # Kit leaves reward/render defaults implicit; explicit conflicting declarations fail.
        actual = profile.get(key, value) if key in {"reward_mode", "render_mode"} else profile.get(key)
        if actual != value:
            message = f"Kit {profile['profile_id']}.{key} differs from independent upstream configuration"
            raise ValueError(message)
    if "diagnostic_step_limit" in profile:
        raise ValueError("Diagnostic prefixes have no independent full-horizon PPO reference")


def require_vector_groups(protocol: dict[str, Any], requested: int | None, counts: list[int]) -> int:
    """Keep the declared vector group width and its whole-group episode partition."""
    width = protocol.get("vector_envs")
    if (type(width) is not int or width < 1
            or (requested is not None and (type(requested) is not int or requested != width))):
        raise ValueError("Native num-envs must match protocol.vector_envs")
    if any(type(count) is not int or count < 1 or count % width for count in counts):
        raise ValueError("Native episodes must contain whole frozen vector groups")
    return width
