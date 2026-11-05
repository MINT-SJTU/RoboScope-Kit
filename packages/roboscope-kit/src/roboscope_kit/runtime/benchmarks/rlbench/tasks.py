# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Frozen task table of the RLBench PerAct 18-task reproduction profile (``peract_18``).

The profile is the RLBench multi-task evaluation of PerAct (Shridhar et al., CoRL 2022) as its
pinned ``eval.py`` runs it, not RLBench as a whole: 18 of RLBench's 100+ tasks, each with all of
its variations, and the 25 released test episodes per task. Each test episode is replayed from
the random state and variation stored with it, exactly as PerAct's ``reset_to_demo`` does, so the
scenes are the ones PerAct was tested on. The episode digests below are computed from those
stored states; the build and the formal aggregate reject a test set that differs.

The observation and action settings are PerAct's: four 128x128 OpenGL cameras (front,
left/right shoulder, wrist) with RGB and point clouds, RLBench's ``MoveArmThenGripper`` of
``EndEffectorPoseViaPlanning`` and ``Discrete`` (one action is a key-frame end-effector pose
reached by motion planning, the gripper state, and an ignore-collisions flag), and at most 25
actions per episode.

This module must import without the simulator: the kit, the CLI, and the build script read it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

PROTOCOL_VERSION = "rlbench-peract-v1"
SUITE = "peract_18"
SUITES = (SUITE,)
# PerAct ``framework.eval_episodes`` with ``eval_from_eps_number=0``: test episodes 0..24.
EPISODES_PER_TASK = 25
# PerAct ``rlbench.episode_length``: at most 25 key-frame actions per episode.
EPISODE_LENGTH = 25
CAMERAS = ("front", "left_shoulder", "right_shoulder", "wrist")
IMAGE_SIZE = (128, 128)
ACTION_REPRESENTATION = "rlbench.ee-pose-planning-v1"
# x, y, z, qx, qy, qz, qw, gripper open, ignore collisions
ACTION_DIM = 9
SUCCESS_CRITERION = "terminate_on_first_task_success"
# The errors PerAct's CustomMultiTaskRLBenchEnv.step turns into a failed, terminal step.
PLANNING_ERRORS = ("IKError", "ConfigurationPathError", "InvalidActionError")
TEST_SET_SPLIT = "test"


@dataclass(frozen=True)
class RLBenchTask:
    name: str
    variations: int
    # Digest of the 25 test episodes' (episode, variation, stored random state).
    test_episodes_sha256: str


TASKS: tuple[RLBenchTask, ...] = (
    RLBenchTask(
        "close_jar",
        20,
        "c4d382cf639524be49df8719ebf51487d2470a6b2c28f74e69967954d6e9bb40",
    ),
    RLBenchTask(
        "insert_onto_square_peg",
        20,
        "222edb6348a1ad402c1d90fb6af5ac05238bc4150782182fe549b7b3c3535825",
    ),
    RLBenchTask(
        "light_bulb_in",
        20,
        "57a2ea9385d055642d64a42d17132dba60922f998bca6370fb2e670e1296d5d1",
    ),
    RLBenchTask(
        "meat_off_grill",
        2,
        "3b1b936965c178ef835ac36f943449840064588896e2783b898fcd2f655a517e",
    ),
    RLBenchTask(
        "open_drawer",
        3,
        "49c65733ef2e1d445617e2a68b1e1ec18579a5105e907262b18af56ca81f6ef7",
    ),
    RLBenchTask(
        "place_cups",
        3,
        "c098b5d66f280237410f5fa232e8217820314c695cf391d6231ef576f7d12110",
    ),
    RLBenchTask(
        "place_shape_in_shape_sorter",
        5,
        "06661f7af9fa614af5f9bcc788781836ea4f72522636702f068c189cd780e4e6",
    ),
    RLBenchTask(
        "place_wine_at_rack_location",
        3,
        "9ab1c70810f7e0744c6219ee19acb405d7f73667660ae23e3cfd0130f655eb1a",
    ),
    RLBenchTask(
        "push_buttons",
        50,
        "27c9891165e93b669e0098e8e03e11481ccd7de6c9b198c04f7625ef926a5cf8",
    ),
    RLBenchTask(
        "put_groceries_in_cupboard",
        9,
        "8863a27ac72e501460b31990317697be5ecfaf68e5a3a05c6235be8570820766",
    ),
    RLBenchTask(
        "put_item_in_drawer",
        3,
        "dfe19fb3f5e772f320efcb0a47e7dc6a25888471583cca67cadeffbc1f3d75d1",
    ),
    RLBenchTask(
        "put_money_in_safe",
        3,
        "08841f75d14efc680aa196582f91039338abb810326272e45b54af81cda94fe9",
    ),
    RLBenchTask(
        "reach_and_drag",
        20,
        "9550b0997ea93dd7c3334a8f2e549175222c17b294eecbfe451478f4243bcb70",
    ),
    RLBenchTask(
        "slide_block_to_color_target",
        4,
        "e1fdbd6bb68c962ad92d2ce3234daf36880e4c5d95c4901303a1d2a97483ecf9",
    ),
    RLBenchTask(
        "stack_blocks",
        60,
        "a981ca013a7cf7b7875c6854d6f90616fd20e8b7b412a532d748fba8b0a4655d",
    ),
    RLBenchTask(
        "stack_cups",
        20,
        "d8b20411ea1cc8494a07558416012dfc3b774a97d796155b2f67e7f302b8379d",
    ),
    RLBenchTask(
        "sweep_to_dustpan_of_size",
        2,
        "db49b5d7eaf77be2b4c52b78e81a6ac364828c26c388dbd2f9812958bccf8253",
    ),
    RLBenchTask(
        "turn_tap",
        2,
        "ace210c275887feecac853e7aee5363f84b5826c40685011fefa9cb7fcae9f1b",
    ),
)
TASKS_BY_NAME = {task.name: task for task in TASKS}
TASK_NAMES = tuple(task.name for task in TASKS)
QUICK_TEST_TASK = "open_drawer"


def episode_state_sha256(variation: int, random_state: Any) -> str:
    """Digest of one test episode's variation and stored NumPy ``RandomState`` state tuple."""

    import numpy as np

    name, keys, position, has_gauss, cached_gaussian = random_state
    digest = hashlib.sha256()
    digest.update(
        f"{int(variation)}:{name}:{int(position)}:{int(has_gauss)}:{float(cached_gaussian)!r}:".encode()
    )
    digest.update(np.asarray(keys, dtype="<u4").tobytes())
    return digest.hexdigest()


def test_episodes_sha256(episode_digests: Iterable[tuple[int, str]]) -> str:
    """Per-task digest over ``(episode, episode_state_sha256)`` in episode order."""

    digest = hashlib.sha256()
    for episode, episode_digest in sorted(episode_digests):
        digest.update(f"{episode}:{episode_digest}\n".encode())
    return digest.hexdigest()


__all__ = [
    "ACTION_DIM",
    "ACTION_REPRESENTATION",
    "CAMERAS",
    "EPISODES_PER_TASK",
    "EPISODE_LENGTH",
    "IMAGE_SIZE",
    "PLANNING_ERRORS",
    "PROTOCOL_VERSION",
    "QUICK_TEST_TASK",
    "SUCCESS_CRITERION",
    "SUITE",
    "SUITES",
    "TASKS",
    "TASKS_BY_NAME",
    "TASK_NAMES",
    "TEST_SET_SPLIT",
    "RLBenchTask",
    "episode_state_sha256",
    "test_episodes_sha256",
]
