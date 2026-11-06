# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Frozen SimplerEnv evaluation matrix, transcribed from the pinned upstream scripts.

Every variant below is one ``simpler_env/main_inference.py`` invocation of the pinned
SimplerEnv commit's official RT-1 / RT-1-X scripts (``scripts/rt1_*.sh`` and
``scripts/rt1x_bridge.sh``), with the checkpoint and policy arguments removed. The worker
parses ``Variant.argv`` with upstream's own ``simpler_env.evaluation.argparse.get_args``
and expands the episodes in upstream ``maniskill2_evaluator``'s loop order, so robot and
object pose sweeps, overlay images, and environment build keywords are upstream's.

The three suites are separate protocols and are never pooled:

- ``google_robot_visual_matching``: real-image overlay and the four robot/scene recolor
  ``urdf_version`` settings (864 episodes);
- ``google_robot_variant_aggregation``: the background, lighting, distractor, table
  texture, and cabinet variants that ``tools/calc_metrics_evaluation_videos.py`` averages
  (1,722 episodes); the two alternative-camera variants the scripts also run are not in
  the official average and are not in this suite;
- ``widowx_visual_matching``: the four Bridge tasks on WidowX (96 episodes).

A task's official success rate is the unweighted mean of its variants' success rates, as
``calc_metrics_evaluation_videos.py`` computes it (for coke can: per orientation over the
variants, then over the three orientations; every group has equal size, so this is the
mean over the task's variants).

This module must import without the simulator: the kit, the CLI, and the build script
all read it.
"""

from __future__ import annotations

from dataclasses import dataclass

PROTOCOL_VERSION = "simpler-env-v1"
ACTION_REPRESENTATION = "simpler_env.ee-delta-axangle-v1"
ACTION_DIM = 8  # 3 translation + 3 axis-angle rotation + 1 gripper + 1 terminate signal
SUCCESS_CRITERION = "success_at_last_executed_step"

VISUAL_MATCHING = "google_robot_visual_matching"
VARIANT_AGGREGATION = "google_robot_variant_aggregation"
BRIDGE = "widowx_visual_matching"
SUITES = (VISUAL_MATCHING, VARIANT_AGGREGATION, BRIDGE)

GOOGLE_ROBOT = "google_robot_static"
GOOGLE_POLICY_SETUP = "google_robot"
BRIDGE_POLICY_SETUP = "widowx_bridge"

VM_URDF_VERSIONS = (
    "None",
    "recolor_tabletop_visual_matching_1",
    "recolor_tabletop_visual_matching_2",
    "recolor_cabinet_visual_matching_1",
)
# lr_switch: lying horizontally, flipped left-right to match the real evaluation.
COKE_CAN_ORIENTATIONS = (
    ("horizontal", "lr_switch=True"),
    ("vertical", "laid_vertically=True"),
    ("standing", "upright=True"),
)
OPEN_DRAWERS = (
    "OpenTopDrawerCustomInScene-v0",
    "OpenMiddleDrawerCustomInScene-v0",
    "OpenBottomDrawerCustomInScene-v0",
)
CLOSE_DRAWERS = (
    "CloseTopDrawerCustomInScene-v0",
    "CloseMiddleDrawerCustomInScene-v0",
    "CloseBottomDrawerCustomInScene-v0",
)
# The nine visual-matching drawer overlays: (overlay id, robot x, robot y, robot yaw).
DRAWER_OVERLAYS = (
    ("a0", "0.644", "-0.179", "-0.03"),
    ("a1", "0.765", "-0.182", "-0.02"),
    ("a2", "0.889", "-0.203", "-0.06"),
    ("b0", "0.652", "0.009", "0"),
    ("b1", "0.752", "0.009", "0"),
    ("b2", "0.851", "0.035", "0"),
    ("c0", "0.665", "0.224", "0"),
    ("c1", "0.765", "0.222", "-0.025"),
    ("c2", "0.865", "0.222", "-0.025"),
)
APPLE_OVERLAYS = ("a0", "b0", "c0")
INPAINTING = "./ManiSkill2_real2sim/data/real_inpainting"


@dataclass(frozen=True)
class Variant:
    """One upstream ``main_inference.py`` invocation: an environment and its pose sweep."""

    variant_id: str
    suite: str
    task: str
    subtask: str
    factor: str
    script: str
    argv: tuple[str, ...]
    episodes: int
    max_episode_steps: int
    policy_setup: str


@dataclass(frozen=True)
class Task:
    name: str
    suite: str
    variants: tuple[Variant, ...]

    @property
    def episodes(self) -> int:
        return sum(variant.episodes for variant in self.variants)


def _range(start: str, stop: str, count: int) -> tuple[str, ...]:
    return (start, stop, str(count))


def _argv(
    *,
    robot: str,
    control_freq: int,
    sim_freq: int,
    max_steps: int,
    env_name: str,
    scene_name: str,
    robot_x: tuple[str, ...],
    robot_y: tuple[str, ...],
    yaw: tuple[str, ...],
    obj: tuple[str, ...],
    overlay: str | None = None,
    raytracing: bool = False,
    build_kwargs: tuple[str, ...] = (),
    policy_setup: str | None = None,
) -> tuple[str, ...]:
    argv: list[str] = ["--robot", robot]
    if policy_setup is not None:
        argv += ["--policy-setup", policy_setup]
    argv += [
        "--control-freq", str(control_freq),
        "--sim-freq", str(sim_freq),
        "--max-episode-steps", str(max_steps),
        "--env-name", env_name,
        "--scene-name", scene_name,
    ]  # fmt: skip
    if overlay is not None:
        argv += ["--rgb-overlay-path", f"{INPAINTING}/{overlay}"]
    argv += ["--robot-init-x-range", *robot_x, "--robot-init-y-range", *robot_y]
    argv += ["--robot-init-rot-quat-center", "0", "0", "0", "1"]
    argv += ["--robot-init-rot-rpy-range", "0", "0", "1", "0", "0", "1", *yaw]
    argv += list(obj)
    if raytracing:
        argv.append("--enable-raytracing")
    if build_kwargs:
        argv += ["--additional-env-build-kwargs", *build_kwargs]
    return tuple(argv)


def _variant(
    *,
    suite: str,
    task: str,
    subtask: str,
    factor: str,
    name: str,
    script: str,
    episodes: int,
    max_steps: int,
    policy_setup: str = GOOGLE_POLICY_SETUP,
    argv_policy_setup: str | None = None,
    **argv_kwargs,
) -> Variant:
    parts = [task, subtask, name] if subtask else [task, name]
    return Variant(
        variant_id="/".join(parts),
        suite=suite,
        task=task,
        subtask=subtask,
        factor=factor,
        script=script,
        argv=_argv(max_steps=max_steps, policy_setup=argv_policy_setup, **argv_kwargs),
        episodes=episodes,
        max_episode_steps=max_steps,
        policy_setup=policy_setup,
    )


def _urdf_name(urdf_version: str) -> str:
    return "urdf_none" if urdf_version == "None" else f"urdf_{urdf_version}"


# ---------------------------------------------------------------------------- #
# Google Robot: pick coke can (80 steps, 5 x 5 object grid, robot at (0.35, 0.20))
# ---------------------------------------------------------------------------- #

COKE_ROBOT = {"robot_x": _range("0.35", "0.35", 1), "robot_y": _range("0.20", "0.20", 1)}
COKE_OBJ_X = _range("-0.35", "-0.12", 5)
COKE_OBJ_Y = _range("-0.02", "0.42", 5)
COKE_OBJ = ("--obj-init-x-range", *COKE_OBJ_X, "--obj-init-y-range", *COKE_OBJ_Y)
COKE_COMMON = {
    "robot": GOOGLE_ROBOT,
    "control_freq": 3,
    "sim_freq": 513,
    "max_steps": 80,
    "yaw": ("0", "0", "1"),
    "obj": COKE_OBJ,
    "episodes": 25,
    **COKE_ROBOT,
}


def _coke_vm() -> tuple[Variant, ...]:
    return tuple(
        _variant(
            suite=VISUAL_MATCHING,
            task="pick_coke_can",
            subtask=orientation,
            factor="urdf_version",
            name=_urdf_name(urdf),
            script="rt1_pick_coke_can_visual_matching.sh",
            env_name="GraspSingleOpenedCokeCanInScene-v0",
            scene_name="google_pick_coke_can_1_v4",
            overlay="google_coke_can_real_eval_1.png",
            build_kwargs=(option, f"urdf_version={urdf}"),
            **COKE_COMMON,
        )
        for orientation, option in COKE_CAN_ORIENTATIONS
        for urdf in VM_URDF_VERSIONS
    )


# (factor, name, env, scene, extra build kwargs)
COKE_VA_VARIANTS = (
    ("base", "base", "GraspSingleOpenedCokeCanInScene-v0", "google_pick_coke_can_1_v4", ()),
    (
        "table_texture",
        "table_cabinet1",
        "GraspSingleOpenedCokeCanInScene-v0",
        "Baked_sc1_staging_objaverse_cabinet1_h870",
        (),
    ),
    (
        "table_texture",
        "table_cabinet2",
        "GraspSingleOpenedCokeCanInScene-v0",
        "Baked_sc1_staging_objaverse_cabinet2_h870",
        (),
    ),
    (
        "distractor",
        "distractor",
        "GraspSingleOpenedCokeCanDistractorInScene-v0",
        "google_pick_coke_can_1_v4",
        (),
    ),
    (
        "distractor",
        "distractor_more",
        "GraspSingleOpenedCokeCanDistractorInScene-v0",
        "google_pick_coke_can_1_v4",
        ("distractor_config=more",),
    ),
    (
        "background",
        "background_alt1",
        "GraspSingleOpenedCokeCanInScene-v0",
        "google_pick_coke_can_1_v4_alt_background",
        (),
    ),
    (
        "background",
        "background_alt2",
        "GraspSingleOpenedCokeCanInScene-v0",
        "google_pick_coke_can_1_v4_alt_background_2",
        (),
    ),
    (
        "lighting",
        "lighting_darker",
        "GraspSingleOpenedCokeCanInScene-v0",
        "google_pick_coke_can_1_v4",
        ("slightly_darker_lighting=True",),
    ),
    (
        "lighting",
        "lighting_brighter",
        "GraspSingleOpenedCokeCanInScene-v0",
        "google_pick_coke_can_1_v4",
        ("slightly_brighter_lighting=True",),
    ),
)


def _coke_va() -> tuple[Variant, ...]:
    return tuple(
        _variant(
            suite=VARIANT_AGGREGATION,
            task="pick_coke_can",
            subtask=orientation,
            factor=factor,
            name=name,
            script="rt1_pick_coke_can_variant_agg.sh",
            env_name=env_name,
            scene_name=scene_name,
            build_kwargs=(option, *extra),
            **COKE_COMMON,
        )
        for orientation, option in COKE_CAN_ORIENTATIONS
        for factor, name, env_name, scene_name, extra in COKE_VA_VARIANTS
    )


# ---------------------------------------------------------------------------- #
# Google Robot: move near (80 steps, 60 predefined episodes, robot at (0.35, 0.21), yaw -0.09)
# ---------------------------------------------------------------------------- #

MOVE_NEAR_COMMON = {
    "robot": GOOGLE_ROBOT,
    "control_freq": 3,
    "sim_freq": 513,
    "max_steps": 80,
    "robot_x": _range("0.35", "0.35", 1),
    "robot_y": _range("0.21", "0.21", 1),
    "yaw": ("-0.09", "-0.09", "1"),
    "obj": ("--obj-variation-mode", "episode", "--obj-episode-range", "0", "60"),
    "episodes": 60,
}


def _move_near_vm() -> tuple[Variant, ...]:
    return tuple(
        _variant(
            suite=VISUAL_MATCHING,
            task="move_near",
            subtask="",
            factor="urdf_version",
            name=_urdf_name(urdf),
            script="rt1_move_near_visual_matching.sh",
            env_name="MoveNearGoogleBakedTexInScene-v0",
            scene_name="google_pick_coke_can_1_v4",
            overlay="google_move_near_real_eval_1.png",
            build_kwargs=(f"urdf_version={urdf}",),
            **MOVE_NEAR_COMMON,
        )
        for urdf in VM_URDF_VERSIONS
    )


MOVE_NEAR_VA_VARIANTS = (
    ("base", "base", "google_pick_coke_can_1_v4", ()),
    ("distractor", "no_distractor", "google_pick_coke_can_1_v4", ("no_distractor=True",)),
    ("background", "background_alt1", "google_pick_coke_can_1_v4_alt_background", ()),
    ("background", "background_alt2", "google_pick_coke_can_1_v4_alt_background_2", ()),
    ("lighting", "lighting_darker", "google_pick_coke_can_1_v4", ("slightly_darker_lighting=True",)),
    ("lighting", "lighting_brighter", "google_pick_coke_can_1_v4", ("slightly_brighter_lighting=True",)),
    ("table_texture", "table_cabinet1", "Baked_sc1_staging_objaverse_cabinet1_h870", ()),
    ("table_texture", "table_cabinet2", "Baked_sc1_staging_objaverse_cabinet2_h870", ()),
)


def _move_near_va() -> tuple[Variant, ...]:
    return tuple(
        _variant(
            suite=VARIANT_AGGREGATION,
            task="move_near",
            subtask="",
            factor=factor,
            name=name,
            script="rt1_move_near_variant_agg.sh",
            env_name="MoveNearGoogleInScene-v0",
            scene_name=scene_name,
            build_kwargs=extra,
            **MOVE_NEAR_COMMON,
        )
        for factor, name, scene_name, extra in MOVE_NEAR_VA_VARIANTS
    )


# ---------------------------------------------------------------------------- #
# Google Robot: open / close drawer (113 steps, ray tracing)
# ---------------------------------------------------------------------------- #

DRAWER_COMMON = {"robot": GOOGLE_ROBOT, "control_freq": 3, "sim_freq": 513, "max_steps": 113}
ZERO_OBJ = ("--obj-init-x-range", "0", "0", "1", "--obj-init-y-range", "0", "0", "1")
VM_DRAWER_KWARGS = ("station_name=mk_station_recolor", "light_mode=simple", "disable_bad_material=True")


def _drawer_subtask(env_name: str) -> str:
    # "OpenTopDrawerCustomInScene-v0" -> "top"
    for position in ("Top", "Middle", "Bottom"):
        if position in env_name:
            return position.lower()
    raise ValueError(env_name)


def _drawer_vm(task: str, env_names: tuple[str, ...]) -> tuple[Variant, ...]:
    return tuple(
        _variant(
            suite=VISUAL_MATCHING,
            task=task,
            subtask=_drawer_subtask(env_name),
            factor="urdf_version",
            name=f"{_urdf_name(urdf)}/{overlay_id}",
            script="rt1_drawer_visual_matching.sh",
            env_name=env_name,
            scene_name="dummy_drawer",
            robot_x=(x, x, "1"),
            robot_y=(y, y, "1"),
            yaw=(yaw, yaw, "1"),
            obj=ZERO_OBJ,
            overlay=f"open_drawer_{overlay_id}.png",
            raytracing=True,
            build_kwargs=(*VM_DRAWER_KWARGS, f"urdf_version={urdf}"),
            episodes=1,
            **DRAWER_COMMON,
        )
        for urdf in VM_URDF_VERSIONS
        for env_name in env_names
        for overlay_id, x, y, yaw in DRAWER_OVERLAYS
    )


# (factor, name, scene, raytracing flag, build kwargs), as rt1_drawer_variant_agg.sh passes them.
DRAWER_VA_VARIANTS = (
    ("base", "base", "frl_apartment_stage_simple", True, ()),
    ("background", "background_bedroom", "modern_bedroom_no_roof", False, ("shader_dir=rt",)),
    ("background", "background_office", "modern_office_no_roof", False, ("shader_dir=rt",)),
    (
        "lighting",
        "lighting_brighter",
        "frl_apartment_stage_simple",
        False,
        ("shader_dir=rt", "light_mode=brighter"),
    ),
    (
        "lighting",
        "lighting_darker",
        "frl_apartment_stage_simple",
        False,
        ("shader_dir=rt", "light_mode=darker"),
    ),
    (
        "cabinet",
        "cabinet_station2",
        "frl_apartment_stage_simple",
        False,
        ("shader_dir=rt", "station_name=mk_station2"),
    ),
    (
        "cabinet",
        "cabinet_station3",
        "frl_apartment_stage_simple",
        False,
        ("shader_dir=rt", "station_name=mk_station3"),
    ),
)


def _drawer_va(task: str, env_names: tuple[str, ...]) -> tuple[Variant, ...]:
    return tuple(
        _variant(
            suite=VARIANT_AGGREGATION,
            task=task,
            subtask=_drawer_subtask(env_name),
            factor=factor,
            name=name,
            script="rt1_drawer_variant_agg.sh",
            env_name=env_name,
            scene_name=scene_name,
            robot_x=_range("0.65", "0.85", 3),
            robot_y=_range("-0.2", "0.2", 3),
            yaw=("0.0", "0.0", "1"),
            obj=ZERO_OBJ,
            raytracing=raytracing,
            build_kwargs=extra,
            episodes=9,
            **DRAWER_COMMON,
        )
        for factor, name, scene_name, raytracing, extra in DRAWER_VA_VARIANTS
        for env_name in env_names
    )


# ---------------------------------------------------------------------------- #
# Google Robot: place apple in closed top drawer (200 steps, ray tracing, 3 x 3 apple grid)
# ---------------------------------------------------------------------------- #

APPLE_COMMON = {"robot": GOOGLE_ROBOT, "control_freq": 3, "sim_freq": 513, "max_steps": 200}
APPLE_ENV = "PlaceIntoClosedTopDrawerCustomInScene-v0"
APPLE_OBJ = ("--obj-init-x-range", "-0.08", "-0.02", "3", "--obj-init-y-range", "-0.02", "0.08", "3")
APPLE_TASK = "place_apple_in_closed_top_drawer"


def _apple_vm() -> tuple[Variant, ...]:
    overlays = {overlay_id: (x, y, yaw) for overlay_id, x, y, yaw in DRAWER_OVERLAYS}
    return tuple(
        _variant(
            suite=VISUAL_MATCHING,
            task=APPLE_TASK,
            subtask="",
            factor="urdf_version",
            name=f"{_urdf_name(urdf)}/{overlay_id}",
            script="rt1_put_in_drawer_visual_matching.sh",
            env_name=APPLE_ENV,
            scene_name="dummy_drawer",
            robot_x=(overlays[overlay_id][0],) * 2 + ("1",),
            robot_y=(overlays[overlay_id][1],) * 2 + ("1",),
            yaw=(overlays[overlay_id][2],) * 2 + ("1",),
            obj=APPLE_OBJ,
            overlay=f"open_drawer_{overlay_id}.png",
            raytracing=True,
            build_kwargs=(*VM_DRAWER_KWARGS, f"urdf_version={urdf}", "model_ids=baked_apple_v2"),
            episodes=9,
            **APPLE_COMMON,
        )
        for urdf in VM_URDF_VERSIONS
        for overlay_id in APPLE_OVERLAYS
    )


def _apple_va() -> tuple[Variant, ...]:
    return tuple(
        _variant(
            suite=VARIANT_AGGREGATION,
            task=APPLE_TASK,
            subtask="",
            factor=factor,
            name=name,
            script="rt1_put_in_drawer_variant_agg.sh",
            env_name=APPLE_ENV,
            scene_name=scene_name,
            robot_x=_range("0.65", "0.65", 1),
            robot_y=_range("-0.2", "0.2", 3),
            yaw=("0.0", "0.0", "1"),
            obj=APPLE_OBJ,
            raytracing=raytracing,
            build_kwargs=(*extra, "model_ids=apple"),
            episodes=27,
            **APPLE_COMMON,
        )
        for factor, name, scene_name, raytracing, extra in DRAWER_VA_VARIANTS
    )


# ---------------------------------------------------------------------------- #
# WidowX (Bridge): 24 predefined episodes per task, 5 Hz control, 500 Hz simulation
# ---------------------------------------------------------------------------- #

BRIDGE_EPISODES = ("--obj-variation-mode", "episode", "--obj-episode-range", "0", "24")
BRIDGE_TABLE = {
    "robot": "widowx",
    "scene_name": "bridge_table_1_v1",
    "overlay": "bridge_real_eval_1.png",
    "robot_x": ("0.147", "0.147", "1"),
    "robot_y": ("0.028", "0.028", "1"),
    "max_steps": 60,
}
BRIDGE_SINK = {
    "robot": "widowx_sink_camera_setup",
    "scene_name": "bridge_table_1_v2",
    "overlay": "bridge_sink.png",
    "robot_x": ("0.127", "0.127", "1"),
    "robot_y": ("0.06", "0.06", "1"),
    "max_steps": 120,
}
BRIDGE_TASKS = (
    ("put_spoon_on_tablecloth", "PutSpoonOnTableClothInScene-v0", BRIDGE_TABLE),
    ("put_carrot_on_plate", "PutCarrotOnPlateInScene-v0", BRIDGE_TABLE),
    ("stack_green_block_on_yellow_block", "StackGreenCubeOnYellowCubeBakedTexInScene-v0", BRIDGE_TABLE),
    ("put_eggplant_in_basket", "PutEggplantInBasketScene-v0", BRIDGE_SINK),
)


def _bridge() -> tuple[Variant, ...]:
    return tuple(
        _variant(
            suite=BRIDGE,
            task=task,
            subtask="",
            factor="visual_matching",
            name="base",
            script="rt1x_bridge.sh",
            env_name=env_name,
            control_freq=5,
            sim_freq=500,
            yaw=("0", "0", "1"),
            obj=BRIDGE_EPISODES,
            episodes=24,
            policy_setup=BRIDGE_POLICY_SETUP,
            argv_policy_setup=BRIDGE_POLICY_SETUP,
            **setup,
        )
        for task, env_name, setup in BRIDGE_TASKS
    )


def _tasks(suite: str, variants: tuple[Variant, ...]) -> tuple[Task, ...]:
    names: dict[str, None] = dict.fromkeys(variant.task for variant in variants)
    return tuple(
        Task(name, suite, tuple(variant for variant in variants if variant.task == name)) for name in names
    )


SUITE_TASKS: dict[str, tuple[Task, ...]] = {
    VISUAL_MATCHING: _tasks(
        VISUAL_MATCHING,
        (
            *_coke_vm(),
            *_move_near_vm(),
            *_drawer_vm("open_drawer", OPEN_DRAWERS),
            *_drawer_vm("close_drawer", CLOSE_DRAWERS),
            *_apple_vm(),
        ),
    ),
    VARIANT_AGGREGATION: _tasks(
        VARIANT_AGGREGATION,
        (
            *_coke_va(),
            *_move_near_va(),
            *_drawer_va("open_drawer", OPEN_DRAWERS),
            *_drawer_va("close_drawer", CLOSE_DRAWERS),
            *_apple_va(),
        ),
    ),
    BRIDGE: _tasks(BRIDGE, _bridge()),
}
SUITE_EPISODES = {suite: sum(task.episodes for task in tasks) for suite, tasks in SUITE_TASKS.items()}
# Variant ids repeat across suites (e.g. ``move_near/base``), so lookups are per suite.
SUITE_VARIANTS: dict[str, dict[str, Variant]] = {
    suite: {variant.variant_id: variant for task in tasks for variant in task.variants}
    for suite, tasks in SUITE_TASKS.items()
}
# The first episode of each suite's first variant; ``--quick-test`` runs it alone.
QUICK_TEST_VARIANTS = {suite: tasks[0].variants[0].variant_id for suite, tasks in SUITE_TASKS.items()}


def episode_id(variant: Variant, index: int) -> str:
    return f"{variant.variant_id}/{index:03d}"


__all__ = [
    "ACTION_DIM",
    "ACTION_REPRESENTATION",
    "BRIDGE",
    "PROTOCOL_VERSION",
    "QUICK_TEST_VARIANTS",
    "SUCCESS_CRITERION",
    "SUITES",
    "SUITE_EPISODES",
    "SUITE_TASKS",
    "SUITE_VARIANTS",
    "VARIANT_AGGREGATION",
    "VISUAL_MATCHING",
    "Task",
    "Variant",
    "episode_id",
]
