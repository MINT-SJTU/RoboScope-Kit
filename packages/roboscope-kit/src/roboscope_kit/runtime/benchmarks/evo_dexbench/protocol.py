# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Frozen Evo-DexBench ``evaluation_v1`` workload shared by Kit and the native evaluator."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

PROTOCOL_VERSION = "evo-dexbench-evaluation-v1"
EVO_DEXBENCH_REVISION = "818ac28e42e9199f0cbccce8617cb686519d8abf"
UPSTREAM_PROTOCOL = "evaluation_v1"
CONDITIONS = ("nominal", "spatial", "camera", "lighting", "physical")
EPISODES_PER_CONDITION = 20
EPISODES_PER_TASK = EPISODES_PER_CONDITION * len(CONDITIONS)
NATIVE_ENVS = 4
SIM_FREQ_HZ = 120
CONTROL_FREQ_HZ = 30
CONTROL_MODE = "arm_pd_ee_delta_pose_hand_pd_joint_pos"
OBS_MODE = "rgb"
ACTION_BOUNDS_POLICY = "clip"
OBSERVATION_STEPS = 2
ACTION_REPRESENTATION = "evo-dexbench.semantic-v1"
ACTION_DIM_PER_ROLE = 30
JOINTS_PER_ROLE = 31
ROLES = {"single": ("right",), "dual": ("left", "right")}
CAMERAS = {
    "single": ("context", "interaction", "wrist_right"),
    "dual": ("context", "interaction", "wrist_left", "wrist_right"),
}
REQUIRED_CAMERAS = {
    "single": ("context", "interaction", "wrist_right"),
    "dual": ("context", "wrist_left", "wrist_right"),
}


@dataclass(frozen=True)
class Task:
    task_id: str
    name: str
    horizon: int
    split: str
    family: str

    @property
    def embodiment(self) -> str:
        return "single" if self.task_id.startswith("m") else "dual"

    @property
    def suite(self) -> str:
        return SUITE_BY_EMBODIMENT[self.embodiment]


SUITE_BY_EMBODIMENT = {"single": "single_hand", "dual": "bimanual"}
TASKS = (
    Task("m01_v1", "LampButton", 1500, "core", "button_driven"),
    Task("m02_v1", "WallSwitch", 1000, "core", "button_driven"),
    Task("m03_v1", "PhoneHomeButton", 2000, "core", "button_driven"),
    Task("m04_v1", "ToasterSlider", 1000, "core", "button_driven"),
    Task("m05_v1", "DisplayPowerButton", 1500, "gen", "button_driven"),
    Task("m06_v1", "FaucetHandle", 2000, "core", "rotational_manipulation"),
    Task("m07_v1", "MouseWheel", 2000, "core", "rotational_manipulation"),
    Task("m08_v1", "CoffeeKnob", 2500, "core", "rotational_manipulation"),
    Task("m09_v1", "KettleLidTab", 2500, "core", "rotational_manipulation"),
    Task("m10_v1", "DishwasherKnob", 2500, "gen", "rotational_manipulation"),
    Task("m11_v1", "CupFromCabinet", 3000, "core", "access_and_transfer"),
    Task("m12_v1", "MicrowaveBowl", 3500, "core", "access_and_transfer"),
    Task("m13_v1", "TennisBallTransfer", 3000, "core", "access_and_transfer"),
    Task("m14_v1", "BananaFromDrawer", 4000, "core", "access_and_transfer"),
    Task("m15_v1", "CandleToCabinet", 4000, "gen", "access_and_transfer"),
    Task("b01_v1", "TablewarePlacement", 4000, "core", "structured_configuration"),
    Task("b02_v1", "HammerBlockAssembly", 5000, "core", "structured_configuration"),
    Task("b03_v1", "SandwichAssembly", 4000, "core", "structured_configuration"),
    Task("b04_v1", "RingTowerAssembly", 4000, "core", "structured_configuration"),
    Task("b05_v1", "BoxPacking", 4000, "gen", "structured_configuration"),
    Task("b06_v1", "PenCapping", 3000, "core", "assembly"),
    Task("b07_v1", "CupSleeveFit", 3000, "core", "assembly"),
    Task("b08_v1", "BottleFunnelInsertion", 3000, "core", "assembly"),
    Task("b09_v1", "PlugInsertion", 3000, "core", "assembly"),
    Task("b10_v1", "ScissorsSheathing", 3000, "gen", "assembly"),
    Task("b11_v1", "LaptopClosing", 1500, "core", "articulated_configuration_control"),
    Task("b12_v1", "StaplerPaperMark", 2000, "core", "articulated_configuration_control"),
    Task("b13_v1", "ScissorsOpening", 2000, "core", "articulated_configuration_control"),
    Task("b14_v1", "BoxLidOpening", 1500, "core", "articulated_configuration_control"),
    Task("b15_v1", "GlassesFolding", 2500, "gen", "articulated_configuration_control"),
)
TASKS_BY_ID = {task.task_id: task for task in TASKS}
SUITES = {
    suite: tuple(task.task_id for task in TASKS if task.suite == suite)
    for suite in ("single_hand", "bimanual")
}


def action_dimension(embodiment: str) -> int:
    return ACTION_DIM_PER_ROLE * len(ROLES[embodiment])


def episode_id(task_id: str, episode: int) -> str:
    return f"{task_id}-episode-{episode:03d}"


def plan_sha256(tasks: tuple[str, ...], episodes_per_task: int) -> str:
    plan = {"protocol": PROTOCOL_VERSION, "tasks": list(tasks), "episodes_per_task": episodes_per_task}
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
