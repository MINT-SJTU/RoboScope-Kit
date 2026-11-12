# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Frozen RoboCasa v1.0.1 multi-task task sets and evaluation settings.

The horizons already include the upstream v1.0.1 1.5x increase. Native rollout
labels are identifiers, not seeds passed to ``reset``. Each native environment
slot keeps the RNG stream initialized by its constructor across rollouts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

ROBOCASA_REVISION = "3443cdf3dd20f9ae8ed41adba442059b506891d0"
ROBOSUITE_REVISION = "5ce6643f3092639d08f7b0f90ed1c6a84f50552c"
PROTOCOL_VERSION = "robocasa365-multitask-v1"
ACTION_REPRESENTATION = "robocasa365.panda-omron-hybrid-v1"
SCENE_SPLIT = "pretrain"
EPISODES_PER_TASK = 50
NATIVE_ENVS = 5
CONSTRUCTOR_SEED = 1111111
ROLLOUT_LABEL_BASE = 100000
OBSERVATION_STEPS = 2
ACTION_STEPS = 8
ACTION_DIMENSION = 12
CONTROL_FREQ_HZ = 20


@dataclass(frozen=True)
class Task:
    name: str
    split: str
    horizon: int


_ATOMIC_SEEN = (
    ("CloseBlenderLid", 900),
    ("CloseFridge", 900),
    ("CloseToasterOvenDoor", 450),
    ("CoffeeSetupMug", 600),
    ("NavigateKitchen", 450),
    ("OpenCabinet", 1050),
    ("OpenDrawer", 750),
    ("OpenStandMixerHead", 450),
    ("PickPlaceCounterToCabinet", 750),
    ("PickPlaceCounterToStove", 600),
    ("PickPlaceDrawerToCounter", 750),
    ("PickPlaceSinkToCounter", 900),
    ("PickPlaceToasterToCounter", 600),
    ("SlideDishwasherRack", 450),
    ("TurnOffStove", 750),
    ("TurnOnElectricKettle", 450),
    ("TurnOnMicrowave", 450),
    ("TurnOnSinkFaucet", 600),
)
_COMPOSITE_SEEN = (
    ("DeliverStraw", 2550),
    ("GetToastedBread", 3000),
    ("KettleBoiling", 1500),
    ("LoadDishwasher", 1800),
    ("PackIdenticalLunches", 3900),
    ("PreSoakPan", 2400),
    ("PrepareCoffee", 1800),
    ("RinseSinkBasin", 1350),
    ("ScrubCuttingBoard", 1200),
    ("SearingMeat", 4350),
    ("SetUpCuttingStation", 2400),
    ("StackBowlsCabinet", 2100),
    ("SteamInMicrowave", 2100),
    ("StirVegetables", 2400),
    ("StoreLeftoversInBowl", 2550),
    ("WashLettuce", 1650),
)
_COMPOSITE_UNSEEN = (
    ("ArrangeBreadBasket", 4350),
    ("ArrangeTea", 2250),
    ("BreadSelection", 1950),
    ("CategorizeCondiments", 1650),
    ("CuttingToolSelection", 1200),
    ("GarnishPancake", 2700),
    ("GatherTableware", 2250),
    ("HeatKebabSandwich", 2700),
    ("MakeIceLemonade", 3000),
    ("PanTransfer", 1800),
    ("PortionHotDogs", 2250),
    ("RecycleBottlesByType", 2850),
    ("SeparateFreezerRack", 2400),
    ("WaffleReheat", 4050),
    ("WashFruitColander", 3150),
    ("WeighIngredients", 3000),
)
TASKS = tuple(
    Task(name, split, horizon)
    for split, rows in (
        ("atomic_seen", _ATOMIC_SEEN),
        ("composite_seen", _COMPOSITE_SEEN),
        ("composite_unseen", _COMPOSITE_UNSEEN),
    )
    for name, horizon in rows
)
TASKS_BY_NAME = {task.name: task for task in TASKS}
SUITES = {
    "target50": tuple(task.name for task in TASKS),
    **{
        split: tuple(task.name for task in TASKS if task.split == split)
        for split in ("atomic_seen", "composite_seen", "composite_unseen")
    },
}
IMAGE_KEYS = {
    "images.agentview_left": "video.robot0_agentview_left",
    "images.agentview_right": "video.robot0_agentview_right",
    "images.wrist": "video.robot0_eye_in_hand",
}
STATE_DIMENSIONS = {
    "end_effector_position_relative": 3,
    "end_effector_rotation_relative": 4,
    "gripper_qpos": 2,
    "base_position": 3,
    "base_rotation": 4,
}


def scenario_plan(tasks: tuple[str, ...], episodes: int, native_envs: int) -> dict:
    """Describe constructor streams without mistaking rollout labels for reset seeds."""
    return {
        "protocol": PROTOCOL_VERSION,
        "robocasa_revision": ROBOCASA_REVISION,
        "scene_split": SCENE_SPLIT,
        "native_envs": native_envs,
        "tasks": list(tasks),
        "episodes_per_task": episodes,
        "streams": [
            {
                "slot": slot,
                "constructor_seed": CONSTRUCTOR_SEED + slot,
                "rollout_labels": [
                    ROLLOUT_LABEL_BASE + index for index in range(slot, episodes, native_envs)
                ],
            }
            for slot in range(native_envs)
        ],
    }


def plan_sha256(tasks: tuple[str, ...], episodes: int, native_envs: int) -> str:
    payload = json.dumps(scenario_plan(tasks, episodes, native_envs), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_upstream() -> None:
    """Reject a runtime whose installed registry disagrees with the frozen protocol."""
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    for suite, tasks in SUITES.items():
        if tuple(TASK_SET_REGISTRY[suite]) != tasks:
            raise RuntimeError(f"RoboCasa365 upstream task registry differs for {suite}")
    for task in TASKS:
        if get_task_horizon(task.name) != task.horizon:
            raise RuntimeError(f"RoboCasa365 upstream horizon differs for {task.name}")
