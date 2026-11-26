# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Built-in benchmark contracts exposed to user policy implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from roboscope_kit.contracts import ActionSpec, BenchmarkSpec, FeatureKind, FeatureSpec
from roboscope_kit.image_sizes import CameraSupport, fixed_camera_support


class BenchmarkName(str, Enum):  # noqa: UP042 -- benchmark environments include Python 3.10
    LIBERO = "libero"
    LIBERO_PLUS = "libero_plus"
    LIBERO_PRO = "libero_pro"
    ROBOCASA = "robocasa"
    ROBOCASA365 = "robocasa365"
    ROBOTWIN = "robotwin"
    CALVIN = "calvin"
    METAWORLD = "metaworld"
    ROBODOJO = "robodojo"
    RLBENCH = "rlbench"
    ROBOMIMIC = "robomimic"
    MANISKILL = "maniskill"
    SIMPLER_ENV = "simpler_env"


class BenchmarkDefinition(ABC):
    """Internal provider for one benchmark specification."""

    name: BenchmarkName
    protocol_version: str
    supported_action_representations: tuple[str, ...]
    default_action_representation: str

    @abstractmethod
    def features(self, suite: str) -> tuple[FeatureSpec, ...]: ...

    @abstractmethod
    def action(self, suite: str) -> ActionSpec: ...

    def spec(self, suite: str) -> BenchmarkSpec:
        if not suite:
            raise ValueError("benchmark suite cannot be empty")
        return BenchmarkSpec(
            name=self.name.value,
            suite=suite,
            protocol_version=self.protocol_version,
            features=self.features(suite),
            action=self.action(suite),
            supported_action_representations=self.supported_action_representations,
            default_action_representation=self.default_action_representation,
        )


def _feature(
    name: str,
    kind: FeatureKind,
    description: str,
    *,
    required: bool = True,
    shape: tuple[int | None, ...] | None = None,
    dtype: str | None = None,
) -> FeatureSpec:
    if dtype is None:
        dtype = {
            FeatureKind.IMAGE: "uint8",
            FeatureKind.STATE: "float32",
            FeatureKind.LANGUAGE: "string",
        }[kind]
    return FeatureSpec(
        name=name,
        kind=kind,
        description=description,
        required=required,
        dtype=dtype,
        shape=shape,
    )


class LiberoBenchmarkDefinition(BenchmarkDefinition):
    name = BenchmarkName.LIBERO
    protocol_version = "libero-v1"
    supported_action_representations = ("libero.delta-eef-v1",)
    default_action_representation = "libero.delta-eef-v1"

    def features(self, _suite: str) -> tuple[FeatureSpec, ...]:
        return (
            _feature(
                "images.primary",
                FeatureKind.IMAGE,
                "Agent-view RGB image.",
                shape=(256, 256, 3),
            ),
            _feature(
                "images.wrist",
                FeatureKind.IMAGE,
                "Robot wrist RGB image.",
                shape=(256, 256, 3),
            ),
            _feature(
                "state.proprioception",
                FeatureKind.STATE,
                "EEF position/quaternion, gripper position, and seven joint positions.",
                shape=(16,),
            ),
            _feature("language.instruction", FeatureKind.LANGUAGE, "Task instruction."),
        )

    def action(self, _suite: str) -> ActionSpec:
        return ActionSpec("action.robot", "LIBERO 6-DoF end-effector delta and gripper command.", 7)


class LiberoPlusBenchmarkDefinition(LiberoBenchmarkDefinition):
    name = BenchmarkName.LIBERO_PLUS
    protocol_version = "libero-plus-v1"


class LiberoProBenchmarkDefinition(LiberoBenchmarkDefinition):
    name = BenchmarkName.LIBERO_PRO
    protocol_version = "libero-pro-v1"


class ManiSkillBenchmarkDefinition(BenchmarkDefinition):
    """Task-driven ManiSkill observations and native controller action vectors."""

    name = BenchmarkName.MANISKILL
    protocol_version = "maniskill-v1"
    supported_action_representations = ("maniskill.native-controller-v1",)
    default_action_representation = "maniskill.native-controller-v1"

    def features(self, _suite: str) -> tuple[FeatureSpec, ...]:
        return (
            _feature(
                "state.proprioception", FeatureKind.STATE,
                "Upstream state, or nonvisual agent/extra fields for visual observations.",
                required=False, shape=(None,),
            ),
            _feature(
                "state.root_pose", FeatureKind.STATE,
                "Optional articulation root pose: scene XYZ followed by quaternion WXYZ.",
                required=False, shape=(7,),
            ),
            _feature(
                "state.root_linear_velocity", FeatureKind.STATE,
                "Optional articulation root linear velocity in the scene world frame.",
                required=False, shape=(3,),
            ),
            _feature(
                "state.root_angular_velocity", FeatureKind.STATE,
                "Optional articulation root angular velocity in the scene world frame.",
                required=False, shape=(3,),
            ),
            *(_feature(name, FeatureKind.STATE, description, required=False, shape=(dimension,))
              for name, description, dimension in (
                  ("state.robot_base_pose", "Optional scene-frame robot base-link XYZ/WXYZ pose.", 7),
                  ("state.robot_tcp_pose", "Optional scene-frame measured TCP XYZ/WXYZ pose.", 7),
                  ("state.target_handle_position", "Optional scene-frame cabinet handle position.", 3),
                  ("state.left_finger_pose", "Optional scene-frame left finger XYZ/WXYZ pose.", 7),
                  ("state.right_finger_pose", "Optional scene-frame right finger XYZ/WXYZ pose.", 7),
                  ("state.left_target_contact_force", "Optional left finger/target contact force in N.", 3),
                  ("state.right_target_contact_force", "Optional right finger/target contact force in N.", 3),
              )),
            _feature(
                "state.depth_head", FeatureKind.STATE, "Optional native head-camera depth in millimeters.",
                required=False, shape=(None, None, 1),
            ),
            _feature(
                "state.depth_hand", FeatureKind.STATE, "Optional native hand-camera depth in millimeters.",
                required=False, shape=(None, None, 1),
            ),
            _feature(
                "images.primary", FeatureKind.IMAGE, "Configured primary camera RGB image.",
                required=False, shape=(None, None, 3),
            ),
            _feature(
                "images.wrist", FeatureKind.IMAGE, "Configured wrist camera RGB image.",
                required=False, shape=(None, None, 3),
            ),
            _feature(
                "images.auxiliary", FeatureKind.IMAGE, "Configured auxiliary camera RGB image.",
                required=False, shape=(None, None, 3),
            ),
            _feature(
                "language.instruction", FeatureKind.LANGUAGE, "Task instruction.", required=False,
            ),
        )

    def action(self, _suite: str) -> ActionSpec:
        # Each task validates against its robot/controller action space in the worker.
        return ActionSpec(
            "action.robot", "Unmodified action vector for the per-episode ManiSkill controller.",
            None,
        )


class RoboCasaBenchmarkDefinition(BenchmarkDefinition):
    name = BenchmarkName.ROBOCASA
    protocol_version = "robocasa-v1"
    supported_action_representations = ("robocasa.gr1-joint-v1",)
    default_action_representation = "robocasa.gr1-joint-v1"

    def features(self, _suite: str) -> tuple[FeatureSpec, ...]:
        return (
            _feature("images.primary", FeatureKind.IMAGE, "Primary workspace RGB image."),
            _feature("images.wrist", FeatureKind.IMAGE, "Robot wrist RGB image.", required=False),
            _feature("state.proprioception", FeatureKind.STATE, "RoboCasa embodiment state vector."),
            _feature("language.instruction", FeatureKind.LANGUAGE, "Task instruction."),
        )

    def action(self, _suite: str) -> ActionSpec:
        return ActionSpec("action.robot", "RoboCasa GR1 arms, hands, and waist action vector.", 29)


class RoboCasa365BenchmarkDefinition(BenchmarkDefinition):
    name = BenchmarkName.ROBOCASA365
    protocol_version = "robocasa365-multitask-v1"
    supported_action_representations = ("robocasa365.panda-omron-hybrid-v1",)
    default_action_representation = "robocasa365.panda-omron-hybrid-v1"

    def features(self, _suite: str) -> tuple[FeatureSpec, ...]:
        images = tuple(
            _feature(name, FeatureKind.IMAGE, "Two consecutive physical-step RGB frames.",
                     shape=(2, 256, 256, 3))
            for name in ("images.agentview_left", "images.agentview_right", "images.wrist")
        )
        states = tuple(
            _feature(f"state.{name}", FeatureKind.STATE,
                     "Two consecutive raw PandaOmron state vectors; quaternions use xyzw.",
                     shape=(2, dimension))
            for name, dimension in (
                ("end_effector_position_relative", 3), ("end_effector_rotation_relative", 4),
                ("gripper_qpos", 2), ("base_position", 3), ("base_rotation", 4),
            )
        )
        return (*images, *states,
                _feature("language.instruction", FeatureKind.LANGUAGE, "Upstream episode instruction."))

    def action(self, _suite: str) -> ActionSpec:
        return ActionSpec(
            "action.robot",
            "PandaOmron end-effector delta (6), gripper (1), base/torso (4), hybrid mode (1).",
            12,
        )


class RoboTwinBenchmarkDefinition(BenchmarkDefinition):
    name = BenchmarkName.ROBOTWIN
    protocol_version = "robotwin2-v1"
    supported_action_representations = (
        "robotwin.qpos-v1",
        "robotwin.ee-compact-quat-v1",
    )
    default_action_representation = "robotwin.qpos-v1"

    def features(self, _suite: str) -> tuple[FeatureSpec, ...]:
        return (
            _feature("images.head", FeatureKind.IMAGE, "Head-camera RGB image."),
            _feature("images.left_wrist", FeatureKind.IMAGE, "Left wrist RGB image."),
            _feature("images.right_wrist", FeatureKind.IMAGE, "Right wrist RGB image."),
            _feature("state.joints", FeatureKind.STATE, "Dual-arm joint and gripper state."),
            _feature("language.instruction", FeatureKind.LANGUAGE, "Task instruction."),
        )

    def action(self, _suite: str) -> ActionSpec:
        return ActionSpec("action.robot", "Absolute dual-arm joint and gripper target.", 14)


class RoboDojoBenchmarkDefinition(BenchmarkDefinition):
    name = BenchmarkName.ROBODOJO
    protocol_version = "robodojo-v1"
    supported_action_representations = (
        "robodojo.x5-joint-v1",
        "robodojo.x5-ee-compact-quat-v1",
    )
    default_action_representation = "robodojo.x5-joint-v1"

    def features(self, _suite: str) -> tuple[FeatureSpec, ...]:
        return (
            _feature("images.head", FeatureKind.IMAGE, "Head-camera RGB image."),
            _feature("images.left_wrist", FeatureKind.IMAGE, "Left wrist RGB image."),
            _feature("images.right_wrist", FeatureKind.IMAGE, "Right wrist RGB image."),
            _feature(
                "state.joints", FeatureKind.STATE,
                "Left arm joints and gripper, followed by right arm joints and gripper.",
                shape=(14,),
            ),
            _feature(
                "state.ee_pose", FeatureKind.STATE,
                "Left and right end-effector poses, each [x, y, z, qw, qx, qy, qz].",
                shape=(14,),
            ),
            _feature("language.instruction", FeatureKind.LANGUAGE, "Task instruction."),
        )

    def action(self, _suite: str) -> ActionSpec:
        return ActionSpec(
            "action.robot", "Dual-arm X5 joint or compact end-effector targets and gripper commands.", 14
        )


class CalvinBenchmarkDefinition(BenchmarkDefinition):
    name = BenchmarkName.CALVIN
    protocol_version = "calvin-v1"
    supported_action_representations = (
        "calvin.relative-v1",
        "calvin.absolute-compact-quat-v1",
    )
    default_action_representation = "calvin.relative-v1"

    def features(self, _suite: str) -> tuple[FeatureSpec, ...]:
        return (
            _feature("images.static", FeatureKind.IMAGE, "Static scene RGB image."),
            _feature("images.gripper", FeatureKind.IMAGE, "Gripper RGB image."),
            _feature("state.proprioception", FeatureKind.STATE, "CALVIN robot state vector."),
            _feature("language.instruction", FeatureKind.LANGUAGE, "Task instruction."),
        )

    def action(self, _suite: str) -> ActionSpec:
        return ActionSpec("action.robot", "CALVIN relative end-effector and gripper action.", 7)


METAWORLD_DEFAULT_CAMERA = {
    "feature": "images.primary",
    "camera_name": "corner2",
    "render_width": 480,
    "render_height": 480,
    "camera_position": None,
    "rotation_deg": 180,
    "center_crop_ratio": 1.0,
    "output_width": 480,
    "output_height": 480,
}


class MetaworldBenchmarkDefinition(BenchmarkDefinition):
    name = BenchmarkName.METAWORLD
    protocol_version = "metaworld-v1"
    supported_action_representations = ("metaworld.delta-eef-v1",)
    default_action_representation = "metaworld.delta-eef-v1"

    def features(self, _suite: str, cameras: tuple[dict, ...] | None = None) -> tuple[FeatureSpec, ...]:
        resolved = (METAWORLD_DEFAULT_CAMERA,) if cameras is None else cameras
        features = [str(camera["feature"]) for camera in resolved]
        if not features or len(set(features)) != len(features):
            raise ValueError("metaworld camera features must be unique and non-empty")
        image_features = tuple(
            _feature(
                feature,
                FeatureKind.IMAGE,
                f"MetaWorld RGB image from camera {camera['camera_name']!r}.",
                shape=(int(camera["output_height"]), int(camera["output_width"]), 3),
            )
            for feature, camera in zip(features, resolved, strict=True)
        )
        return (
            *image_features,
            _feature(
                "state.proprioception",
                FeatureKind.STATE,
                "End-effector xyz position and normalized gripper aperture (raw observation[:4]).",
                shape=(4,),
            ),
            _feature("language.instruction", FeatureKind.LANGUAGE, "Official MetaWorld task instruction."),
        )

    def action(self, _suite: str) -> ActionSpec:
        return ActionSpec(
            "action.robot",
            "MetaWorld delta end-effector position (dx, dy, dz) and gripper command in [-1, 1].",
            4,
        )

    def spec(self, suite: str, cameras: tuple[dict, ...] | None = None) -> BenchmarkSpec:
        if not suite:
            raise ValueError("benchmark suite cannot be empty")
        return BenchmarkSpec(
            name=self.name.value,
            suite=suite,
            protocol_version=self.protocol_version,
            features=self.features(suite, cameras),
            action=self.action(suite),
            supported_action_representations=self.supported_action_representations,
            default_action_representation=self.default_action_representation,
        )


class RobomimicBenchmarkDefinition(BenchmarkDefinition):
    """robomimic v0.1 low-dimensional observations and OSC_POSE delta actions.

    Single-arm tasks (Lift, Can, Square, Tool Hang) send the ``robot0`` features and take
    7-dimensional actions; Transport also sends the ``robot1`` features and takes 14.
    """

    name = BenchmarkName.ROBOMIMIC
    protocol_version = "robomimic-v1"
    supported_action_representations = ("robomimic.osc-pose-delta-v1",)
    default_action_representation = "robomimic.osc-pose-delta-v1"

    def features(self, _suite: str) -> tuple[FeatureSpec, ...]:
        robot_features = []
        for index in range(2):
            for key, size, description in (
                ("eef_pos", 3, "end-effector position in the world frame"),
                ("eef_quat", 4, "end-effector orientation quaternion (x, y, z, w)"),
                ("gripper_qpos", 2, "gripper finger joint positions"),
            ):
                robot_features.append(
                    _feature(
                        f"state.robot{index}_{key}",
                        FeatureKind.STATE,
                        f"robosuite robot{index}_{key}: robot {index} {description}"
                        + ("." if index == 0 else " (two-arm Transport only)."),
                        required=index == 0,
                        shape=(size,),
                    )
                )
        return (
            *robot_features,
            _feature(
                "state.object",
                FeatureKind.STATE,
                "robomimic `object` observation (robosuite object-state); its size depends on the task.",
                shape=(None,),
            ),
        )

    def action(self, _suite: str) -> ActionSpec:
        return ActionSpec(
            "action.robot",
            "Per arm, robosuite OSC_POSE delta position (3), delta axis-angle rotation (3) and gripper (1) "
            "in [-1, 1]; Transport concatenates robot0 then robot1 (14).",
        )


# Policy camera resolution per SimplerEnv suite: Google Robot ``overhead_camera`` and WidowX
# ``3rd_view_camera``, as ``get_image_from_maniskill2_obs_dict`` returns them.
SIMPLER_ENV_IMAGE_SHAPES = {
    "google_robot_visual_matching": (512, 640, 3),
    "google_robot_variant_aggregation": (512, 640, 3),
    "widowx_visual_matching": (480, 640, 3),
}


class SimplerEnvBenchmarkDefinition(BenchmarkDefinition):
    """SimplerEnv (SAPIEN / ManiSkill2_real2sim) with its policy camera and current instruction."""

    name = BenchmarkName.SIMPLER_ENV
    protocol_version = "simpler-env-v1"
    supported_action_representations = ("simpler_env.ee-delta-axangle-v1",)
    default_action_representation = "simpler_env.ee-delta-axangle-v1"

    def features(self, suite: str) -> tuple[FeatureSpec, ...]:
        shape = SIMPLER_ENV_IMAGE_SHAPES.get(suite)
        if shape is None:
            raise ValueError(f"unsupported simpler_env suite: {suite}")
        camera = "WidowX 3rd_view_camera" if suite.startswith("widowx") else "Google Robot overhead_camera"
        overlay = (
            "with the real-image background overlay"
            if suite.endswith("visual_matching")
            else "rendered without overlay"
        )
        return (
            _feature(
                "images.primary",
                FeatureKind.IMAGE,
                f"SimplerEnv policy camera RGB ({camera}), {overlay}.",
                shape=shape,
            ),
            _feature(
                "language.instruction",
                FeatureKind.LANGUAGE,
                "The environment's current language instruction; it changes when a long-horizon "
                "task advances to its next subtask.",
            ),
        )

    def action(self, suite: str) -> ActionSpec:
        robot = "WidowX" if suite.startswith("widowx") else "Google Robot"
        return ActionSpec(
            "action.robot",
            f"The {robot} action SimplerEnv's evaluator passes to env.step: end-effector delta "
            "translation (3), delta rotation as an axis-angle vector (3), and the gripper command "
            "(1) in the robot's controller convention, followed by a terminate signal (> 0 ends the "
            "episode, or advances a long-horizon task that is not on its final subtask). float64.",
            8,
            dtype="float64",
        )


RLBENCH_CAMERAS = ("front", "left_shoulder", "right_shoulder", "wrist")


class RLBenchBenchmarkDefinition(BenchmarkDefinition):
    """RLBench with PerAct's four 128x128 cameras, low-dimensional state, and key-frame actions."""

    name = BenchmarkName.RLBENCH
    protocol_version = "rlbench-peract-v2"
    supported_action_representations = ("rlbench.ee-pose-planning-v1",)
    default_action_representation = "rlbench.ee-pose-planning-v1"

    def features(self, _suite: str) -> tuple[FeatureSpec, ...]:
        cameras: list[FeatureSpec] = []
        for camera in RLBENCH_CAMERAS:
            label = camera.replace("_", " ")
            cameras.extend(
                (
                    _feature(
                        f"images.{camera}",
                        FeatureKind.IMAGE,
                        f"RLBench {label} camera RGB.",
                        shape=(128, 128, 3),
                    ),
                    _feature(
                        f"state.{camera}_depth",
                        FeatureKind.STATE,
                        f"RLBench {label} camera depth in metres.",
                        shape=(128, 128),
                    ),
                    _feature(
                        f"state.{camera}_point_cloud",
                        FeatureKind.STATE,
                        f"RLBench {label} camera point cloud: per-pixel world-frame x, y, z in metres.",
                        shape=(128, 128, 3),
                    ),
                    _feature(
                        f"state.{camera}_camera_extrinsics",
                        FeatureKind.STATE,
                        f"RLBench {label} camera-to-world pose as a 4x4 matrix.",
                        shape=(4, 4),
                        dtype="float64",
                    ),
                    _feature(
                        f"state.{camera}_camera_intrinsics",
                        FeatureKind.STATE,
                        f"RLBench {label} camera intrinsic matrix.",
                        shape=(3, 3),
                        dtype="float64",
                    ),
                )
            )
        return (
            *cameras,
            _feature(
                "state.gripper_open",
                FeatureKind.STATE,
                "Gripper open state: 1.0 open, 0.0 closed.",
                shape=(1,),
            ),
            _feature(
                "state.ignore_collisions",
                FeatureKind.STATE,
                "RLBench observation's current-waypoint collision-ignore flag.",
                shape=(1,),
            ),
            _feature(
                "state.gripper_joint_positions",
                FeatureKind.STATE,
                "Panda gripper finger joint positions (unclipped).",
                shape=(2,),
            ),
            _feature(
                "state.gripper_pose",
                FeatureKind.STATE,
                "Gripper tip pose in the world frame: x, y, z, qx, qy, qz, qw.",
                shape=(7,),
            ),
            _feature("state.joint_positions", FeatureKind.STATE, "Panda arm joint positions.", shape=(7,)),
            _feature("state.joint_velocities", FeatureKind.STATE, "Panda arm joint velocities.", shape=(7,)),
            _feature(
                "language.instruction", FeatureKind.LANGUAGE, "The episode's first RLBench task description."
            ),
        )

    def action(self, _suite: str) -> ActionSpec:
        return ActionSpec(
            "action.robot",
            "Key-frame gripper pose in the world frame (x, y, z, qx, qy, qz, qw; a unit quaternion), "
            "gripper open (1) or closed (0), and ignore collisions (1) or not (0) for the motion planner. "
            "float64, as RLBench receives it.",
            9,
            dtype="float64",
        )


_DEFINITIONS: dict[BenchmarkName, BenchmarkDefinition] = {
    definition.name: definition
    for definition in (
        LiberoBenchmarkDefinition(),
        LiberoPlusBenchmarkDefinition(),
        LiberoProBenchmarkDefinition(),


        ManiSkillBenchmarkDefinition(),
        RoboCasaBenchmarkDefinition(),
        RoboCasa365BenchmarkDefinition(),
        RoboTwinBenchmarkDefinition(),
        RoboDojoBenchmarkDefinition(),
        CalvinBenchmarkDefinition(),
        MetaworldBenchmarkDefinition(),
        RLBenchBenchmarkDefinition(),
        RobomimicBenchmarkDefinition(),
        SimplerEnvBenchmarkDefinition(),
    )
}


# LIBERO and LIBERO-Pro render their cameras natively at any size in these bounds; Kit never
# resamples.
LIBERO_CANONICAL_IMAGE_SIZE = (256, 256)
LIBERO_MIN_IMAGE_SIZE = (64, 64)
LIBERO_MAX_IMAGE_SIZE = (1024, 1024)
# robosuite's OSMesa context allocates a fixed 480 x 640 buffer.
LIBERO_OSMESA_MAX_IMAGE_SIZE = (480, 640)
# MetaWorld renders each configured camera natively at any size its camera config accepts; the
# canonical size is the camera's configured output size.
METAWORLD_MIN_CAMERA_DIMENSION = 64
METAWORLD_MAX_CAMERA_DIMENSION = 1024
# RLBench renders each CoppeliaSim vision sensor natively at its configured resolution; its depth and
# point cloud come from the same render. PerAct's protocol uses 128 x 128. At 768 x 768 an observation
# of all four cameras (RGB, depth and point cloud) is about 60 MB, inside the 64 MiB Policy message limit.
RLBENCH_CANONICAL_IMAGE_SIZE = (128, 128)
RLBENCH_MIN_IMAGE_SIZE = (32, 32)
RLBENCH_MAX_IMAGE_SIZE = (768, 768)


# CALVIN's calvin_env cameras (conf/cameras/cameras/{static,gripper}.yaml) render natively at any
# size; the canonical sizes are the upstream configs' and the spec leaves the shapes open.
CALVIN_CANONICAL_IMAGE_SIZES = {"images.static": (200, 200), "images.gripper": (84, 84)}
CALVIN_MIN_IMAGE_SIZE = (64, 64)
CALVIN_MAX_IMAGE_SIZE = (1024, 1024)
# LIBERO-Plus applies its sensor-noise corruptions to the agent view only. Its zoom blur crops and
# rescales a square of the frame's height, and its fog adds a 256 x 256 plasma map, so the agent
# view renders square sizes of at most 256 pixels. The wrist camera has the LIBERO range.
LIBERO_PLUS_PRIMARY_MAX_IMAGE_SIZE = (256, 256)
# ManiSkill's field-of-view cameras render natively at any sensor_configs width and height. Each task
# fixes its own default size (128 x 128 for the tabletop base camera), and one suite may mix tasks, so
# the canonical size stays the task's own and is not declared here. At 512 x 512, one 10-environment
# vector group with all three RGB cameras and both depth channels at that size is a 59.4 MB JSON
# act_batch message, within the 64 MiB protocol limit. Every image feature takes that range except those a
# profile of the suite maps to a Bridge camera (see ``fixed_size_camera_features``).
MANISKILL_MIN_IMAGE_SIZE = (16, 16)
MANISKILL_MAX_IMAGE_SIZE = (512, 512)
# RoboCasa365 renders its three cameras with robosuite's EGL context, whose offscreen buffer grows to the
# requested size. The official Gym wrapper takes one width and one height for all cameras, so the three
# share a size group. One act_batch request carries five slots x three cameras x two frames; at
# 640 x 640 it is 47 MiB as JSON, within the 64 MiB Policy message limit.
ROBOCASA365_CANONICAL_IMAGE_SIZE = (256, 256)
ROBOCASA365_MIN_IMAGE_SIZE = (64, 64)
ROBOCASA365_MAX_IMAGE_SIZE = (640, 640)
ROBOCASA365_IMAGE_FEATURES = ("images.agentview_left", "images.agentview_right", "images.wrist")


def benchmark_camera_support(
    benchmark: BenchmarkSpec,
    defaults: dict[str, Any] | None = None,
    protocol: dict[str, Any] | None = None,
) -> dict[str, CameraSupport]:
    """The sizes each camera of ``benchmark`` can render under the manifest ``defaults``.

    ``protocol`` is the manifest's protocol for the benchmark; ManiSkill reads its suite's task
    profiles from it, and without it every ManiSkill camera renders only its task's own size.
    """

    table = fixed_camera_support(benchmark)
    if benchmark.name in {BenchmarkName.LIBERO.value, BenchmarkName.LIBERO_PRO.value}:
        backend = str((defaults or {}).get("render_backend", "egl"))
        maximum = LIBERO_OSMESA_MAX_IMAGE_SIZE if backend == "osmesa" else LIBERO_MAX_IMAGE_SIZE
        for feature in ("images.primary", "images.wrist"):
            table[feature] = CameraSupport(
                canonical=LIBERO_CANONICAL_IMAGE_SIZE,
                min_size=LIBERO_MIN_IMAGE_SIZE,
                max_size=maximum,
            )
    elif benchmark.name == BenchmarkName.METAWORLD.value:
        for feature, support in table.items():
            table[feature] = CameraSupport(
                canonical=support.canonical,
                min_size=(METAWORLD_MIN_CAMERA_DIMENSION, METAWORLD_MIN_CAMERA_DIMENSION),
                max_size=(METAWORLD_MAX_CAMERA_DIMENSION, METAWORLD_MAX_CAMERA_DIMENSION),
            )
    elif benchmark.name == BenchmarkName.CALVIN.value:
        for feature, canonical in CALVIN_CANONICAL_IMAGE_SIZES.items():
            table[feature] = CameraSupport(
                canonical=canonical, min_size=CALVIN_MIN_IMAGE_SIZE, max_size=CALVIN_MAX_IMAGE_SIZE
            )
    elif benchmark.name == BenchmarkName.RLBENCH.value:
        for camera in RLBENCH_CAMERAS:
            table[f"images.{camera}"] = CameraSupport(
                canonical=RLBENCH_CANONICAL_IMAGE_SIZE,
                min_size=RLBENCH_MIN_IMAGE_SIZE,
                max_size=RLBENCH_MAX_IMAGE_SIZE,
                spatial_features=(f"state.{camera}_depth", f"state.{camera}_point_cloud"),
            )
    elif benchmark.name == BenchmarkName.LIBERO_PLUS.value:
        backend = str((defaults or {}).get("render_backend", "egl"))
        maximum = LIBERO_OSMESA_MAX_IMAGE_SIZE if backend == "osmesa" else LIBERO_MAX_IMAGE_SIZE
        table["images.primary"] = CameraSupport(
            canonical=LIBERO_CANONICAL_IMAGE_SIZE,
            min_size=LIBERO_MIN_IMAGE_SIZE,
            max_size=LIBERO_PLUS_PRIMARY_MAX_IMAGE_SIZE,
            square=True,
        )
        table["images.wrist"] = CameraSupport(
            canonical=LIBERO_CANONICAL_IMAGE_SIZE,
            min_size=LIBERO_MIN_IMAGE_SIZE,
            max_size=maximum,
        )
    elif benchmark.name == BenchmarkName.MANISKILL.value and protocol:
        # Imported here because the ManiSkill runtime imports this module; the profile module reads
        # the manifest protocol without the simulator.
        from roboscope_kit.runtime.benchmarks.maniskill.profile import fixed_size_camera_features

        fixed = fixed_size_camera_features(protocol, benchmark.suite)
        for feature, support in table.items():
            if feature not in fixed:
                table[feature] = CameraSupport(
                    canonical=support.canonical,
                    min_size=MANISKILL_MIN_IMAGE_SIZE,
                    max_size=MANISKILL_MAX_IMAGE_SIZE,
                )
    elif benchmark.name == BenchmarkName.ROBOCASA365.value:
        for feature in ROBOCASA365_IMAGE_FEATURES:
            table[feature] = CameraSupport(
                canonical=ROBOCASA365_CANONICAL_IMAGE_SIZE,
                min_size=ROBOCASA365_MIN_IMAGE_SIZE,
                max_size=ROBOCASA365_MAX_IMAGE_SIZE,
                size_group="robocasa365.cameras",
            )
    return table


def benchmark_rendered_image_features(
    benchmark: BenchmarkSpec,
    protocol: dict[str, Any] | None = None,
) -> frozenset[str] | None:
    """The image features the suite's observations carry, or ``None`` when they carry every one.

    ManiSkill observations carry only the features the suite's task profiles map to a camera. The Kit
    records negotiated sizes for those features only; the others never reach the Policy.
    """

    if benchmark.name == BenchmarkName.MANISKILL.value and protocol:
        from roboscope_kit.runtime.benchmarks.maniskill.profile import rendered_camera_features

        return rendered_camera_features(protocol, benchmark.suite)
    return None


def get_benchmark_spec(name: str | BenchmarkName, suite: str) -> BenchmarkSpec:
    benchmark = BenchmarkName(name)
    return _DEFINITIONS[benchmark].spec(suite)


def benchmark_names() -> tuple[str, ...]:
    return tuple(benchmark.value for benchmark in _DEFINITIONS)


__all__ = [
    "ActionSpec",
    "BenchmarkName",
    "BenchmarkSpec",
    "FeatureKind",
    "FeatureSpec",
    "benchmark_names",
    "get_benchmark_spec",
]
