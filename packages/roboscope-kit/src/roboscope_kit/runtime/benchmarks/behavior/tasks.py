# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Frozen task table of the BEHAVIOR Challenge 2026 public-test profile (``challenge_2026``).

The profile is the 2026 BEHAVIOR Challenge evaluation as BEHAVIOR-1K ``v3.9.3-post1`` runs it
(``python -m omnigibson.eval.eval``, ``BatchedEvaluator``): 100 BEHAVIOR activities, not all of
BEHAVIOR-1K's 1,000. Each task is one activity in one scene with the R1Pro robot, and its public
test instances are the stored task-relevant-object states 301-320 of
``2026-challenge-task-instances``; the challenge reports instance indices 0-9 (instances
301-310) with one rollout each. An episode ends when the BDDL goal is met (success), when
OmniGibson's task termination fires, or after ``max_steps`` actions, 1.5x the mean length of the
task's human demonstrations (``EVAL_TIMEOUT_MULTIPLIER`` and ``metadata/task.jsonl``). OmniGibson
checks the ``Timeout`` termination (``episode_steps >= max_steps``) before it counts the step, so a
timed-out episode executes ``max_steps + 1`` actions, and the native result reports that count.

The task IDs are the challenge's (``metadata/B100_task_misc.csv``); the build checks the
dataset's metadata against the digests and step limits below.

This module must import without the simulator: the kit, the CLI, and the build script read it.
"""

from __future__ import annotations

import os
import sys

PROTOCOL_VERSION = "behavior-challenge-2026-v1"
SUITE = "challenge_2026"
SUITES = (SUITE,)
# BEHAVIOR-1K tag v3.9.3-post1, which the organizers use for the 2026 final evaluation.
BEHAVIOR_1K_COMMIT = "bd049de3119acdcdf2334fe9e1ebe060fa20c108"
# omnigibson.eval.utils.eval_utils: TEST_INSTANCE_IDS[:NUM_PUBLIC_TEST_INSTANCES].
PUBLIC_TEST_INSTANCE_IDS = tuple(range(301, 321))
# The challenge evaluates instance indices 0-9 of the public split, one rollout each.
INSTANCES_PER_TASK = 10
ROLLOUT_ID = 0
EVAL_TIMEOUT_MULTIPLIER = 1.5
# omnigibson.eval.utils.eval_utils.DEFAULT_EVAL_SEED, passed to seed_everything() once per process.
EVAL_SEED = 0
# Fixed for repeatable runs; the native runner of the comparison sets the same value.
PYTHON_HASH_SEED = 0
ENV_WRAPPER = "omnigibson.eval.wrappers.RGBDFullResWrapper"
ROBOT = "R1Pro"
ROBOT_NAME = "robot_r1"
# Camera roles in the evaluator's order, the robot sensor of each, and its (H, W).
CAMERAS = (
    ("head", "robot_r1:zed_link:Camera:0", (720, 720)),
    ("left_wrist", "robot_r1:left_realsense_link:Camera:0", (480, 480)),
    ("right_wrist", "robot_r1:right_realsense_link:Camera:0", (480, 480)),
)
PROPRIOCEPTION_DIM = 61
CAMERA_POSES_DIM = 7 * len(CAMERAS)
# base velocity 0:3, torso 3:7, left arm 7:14, left gripper 14, right arm 15:22, right gripper 22.
ACTION_DIM = 23
ACTION_REPRESENTATION = "behavior.r1pro-joint-v1"
CONTROL_HZ = 30
TERMINATIONS = ("success", "task_termination", "max_steps")
QUICK_TEST_TASK = "turning_on_radio"
TASK_METADATA_SHA256 = {
    "metadata/B100_task_misc.csv": "f1533487a09b860150b2ec03ea36d1a0ad5962b0e1292ab28c1fd30ff276f837",
    "metadata/task.jsonl": "7dbb5f7f00a2e06c693dfbb3045e5c28e627b6179e775f1ad93ca4b61f08e6dc",
}

# (task, max_steps) in challenge task-ID order; the task ID is the position.
_TASKS = (
    ("turning_on_radio", 3224),
    ("picking_up_trash", 7901),
    ("putting_away_Halloween_decorations", 20682),
    ("cleaning_up_plates_and_food", 20544),
    ("can_meat", 17770),
    ("setting_mousetraps", 15294),
    ("hiding_Easter_eggs", 11429),
    ("picking_up_toys", 28335),
    ("rearranging_kitchen_furniture", 13414),
    ("putting_up_Christmas_decorations_inside", 20578),
    ("set_up_a_coffee_station_in_your_kitchen", 9399),
    ("putting_dishes_away_after_cleaning", 16430),
    ("preparing_lunch_box", 12367),
    ("loading_the_car", 28839),
    ("carrying_in_groceries", 21412),
    ("bringing_in_wood", 20303),
    ("moving_boxes_to_storage", 21894),
    ("bringing_water", 14157),
    ("tidying_bedroom", 16556),
    ("outfit_a_basic_toolbox", 15956),
    ("sorting_vegetables", 17855),
    ("collecting_childrens_toys", 28779),
    ("putting_shoes_on_rack", 11589),
    ("boxing_books_up_for_storage", 36341),
    ("storing_food", 29803),
    ("clearing_food_from_table_into_fridge", 19602),
    ("assembling_gift_baskets", 39090),
    ("sorting_household_items", 23711),
    ("getting_organized_for_work", 23506),
    ("clean_up_your_desk", 32126),
    ("setting_the_fire", 13677),
    ("clean_boxing_gloves", 12353),
    ("wash_a_baseball_cap", 12524),
    ("wash_dog_toys", 16834),
    ("hanging_pictures", 3581),
    ("attach_a_camera_to_a_tripod", 5867),
    ("clean_a_patio", 18106),
    ("clean_a_trumpet", 7960),
    ("spraying_for_bugs", 9719),
    ("spraying_fruit_trees", 12518),
    ("make_microwave_popcorn", 4856),
    ("cook_cabbage", 21217),
    ("chop_an_onion", 9599),
    ("slicing_vegetables", 22267),
    ("chopping_wood", 16127),
    ("cook_hot_dogs", 13717),
    ("cook_bacon", 11519),
    ("freeze_pies", 18682),
    ("canning_food", 34463),
    ("make_pizza", 28780),
    ("freeze_fruit", 18928),
    ("cook_a_brisket", 11002),
    ("sorting_bottles_cans_and_paper", 15395),
    ("tidying_living_room", 18848),
    ("putting_away_toys", 16945),
    ("re_shelving_library_books", 14292),
    ("make_rose_centerpieces", 6798),
    ("sweeping_garage", 6709),
    ("stacking_wood", 23909),
    ("organizing_art_supplies", 9291),
    ("scrubbing_bathroom_floor", 4730),
    ("bringing_paper_to_recycling", 16821),
    ("halve_an_egg", 9565),
    ("installing_smoke_detectors", 3853),
    ("setting_the_table", 26712),
    ("unloading_the_car", 16985),
    ("turning_out_all_lights_before_sleep", 15009),
    ("boxing_food_after_dinner", 10046),
    ("cleaning_up_branches_and_twigs", 18790),
    ("vacuuming_floors", 3617),
    ("thawing_frozen_food", 12344),
    ("clean_your_rusty_garden_tools", 22736),
    ("cook_a_frozen_pie", 13001),
    ("organizing_school_stuff", 21887),
    ("carrying_out_garden_furniture", 16588),
    ("put_together_a_basic_pruning_kit", 16896),
    ("dispose_of_glass", 13792),
    ("installing_a_modem", 3619),
    ("make_cabinet_doors", 5163),
    ("polishing_shoes", 16106),
    ("clean_up_broken_glass", 13037),
    ("packing_meal_for_delivery", 13289),
    ("store_batteries", 11556),
    ("store_honey", 10150),
    ("tidying_bathroom", 19504),
    ("putting_dirty_dishes_in_sink", 18222),
    ("make_gift_bags_for_baby_showers", 14476),
    ("collecting_aluminum_cans", 15304),
    ("rearrange_your_room", 19282),
    ("installing_a_fax_machine", 6115),
    ("composting_waste", 5457),
    ("store_produce", 14283),
    ("installing_a_scanner", 6387),
    ("clean_a_keyboard", 5814),
    ("dispose_of_batteries", 21642),
    ("cook_brussels_sprouts", 23492),
    ("cook_broccolini", 6926),
    ("setup_a_bar_for_a_cocktail_party", 20128),
    ("laying_tile_floors", 21526),
    ("sorting_books_on_shelf", 11609),
)
TASK_NAMES = tuple(name for name, _ in _TASKS)
TASK_IDS = {name: index for index, name in enumerate(TASK_NAMES)}
MAX_STEPS = dict(_TASKS)


def instance_id(index: int) -> int:
    """The public test instance at challenge instance index ``index``."""

    if not 0 <= index < len(PUBLIC_TEST_INSTANCE_IDS):
        raise ValueError(f"BEHAVIOR public test instance index must be in [0, 20); received {index}")
    return PUBLIC_TEST_INSTANCE_IDS[index]


def episode_id_for(task: str, index: int) -> str:
    return f"{task}-instance-{instance_id(index)}"


def require_python_hash_seed() -> None:
    """Reject a simulator interpreter whose hash seed was not fixed at startup."""

    if os.environ.get("PYTHONHASHSEED") != str(PYTHON_HASH_SEED) or sys.flags.hash_randomization:
        raise RuntimeError(
            f"BEHAVIOR {PROTOCOL_VERSION} requires PYTHONHASHSEED={PYTHON_HASH_SEED} "
            "before the simulator Python process starts"
        )


__all__ = [
    "ACTION_DIM",
    "ACTION_REPRESENTATION",
    "BEHAVIOR_1K_COMMIT",
    "CAMERAS",
    "CAMERA_POSES_DIM",
    "CONTROL_HZ",
    "ENV_WRAPPER",
    "EVAL_SEED",
    "EVAL_TIMEOUT_MULTIPLIER",
    "INSTANCES_PER_TASK",
    "MAX_STEPS",
    "PROPRIOCEPTION_DIM",
    "PROTOCOL_VERSION",
    "PUBLIC_TEST_INSTANCE_IDS",
    "PYTHON_HASH_SEED",
    "QUICK_TEST_TASK",
    "ROBOT",
    "ROBOT_NAME",
    "ROLLOUT_ID",
    "SUITE",
    "SUITES",
    "TASK_IDS",
    "TASK_METADATA_SHA256",
    "TASK_NAMES",
    "TERMINATIONS",
    "episode_id_for",
    "instance_id",
    "require_python_hash_seed",
]
