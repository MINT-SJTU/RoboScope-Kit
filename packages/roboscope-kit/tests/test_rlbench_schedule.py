# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from roboscope_kit.runtime.benchmark import EvaluationRecord, WorkResult
from roboscope_kit.runtime.benchmarks.rlbench.adapter import RLBenchBenchmarkAdapter, RLBenchRuntimeConfig
from roboscope_kit.runtime.benchmarks.rlbench.tasks import (
    EPISODE_LENGTH,
    EPISODES_PER_TASK,
    EVALUATION_ORDER,
    PROTOCOL_VERSION,
    PYTHON_HASH_SEED,
    TASK_NAMES,
)


def config(tasks, episodes):
    path = Path("/unused")
    return RLBenchRuntimeConfig(
        python=path,
        coppeliasim=path,
        test_set=path,
        xvfb=path,
        suite="peract_18",
        tasks=tasks,
        episodes_per_task=episodes,
    )


def test_formal_run_is_one_item_in_evaluation_order():
    items = RLBenchBenchmarkAdapter(config(TASK_NAMES, EPISODES_PER_TASK)).work_items()

    assert len(items) == 1
    segments = items[0].payload["segments"]
    assert [segment["task"] for segment in segments] == list(EVALUATION_ORDER)
    assert all(segment["episodes"] == list(range(EPISODES_PER_TASK)) for segment in segments)
    assert items[0].expected_results == len(TASK_NAMES) * EPISODES_PER_TASK


def test_partial_run_spreads_episodes_over_items():
    items = RLBenchBenchmarkAdapter(config(TASK_NAMES, 2)).work_items()

    assert len(items) == 2 * len(TASK_NAMES)
    assert {(item.payload["task"], tuple(item.payload["episodes"])) for item in items} == {
        (task, (episode,)) for task in TASK_NAMES for episode in range(2)
    }


def formal_result(records):
    return WorkResult(
        item_id="peract_18-evaluation-order", group_id="peract_18", worker_id=0, records=records
    )


def record(task, episode, success=True):
    return EvaluationRecord(
        record_id=f"{task}-episode-{episode:02d}",
        success=success,
        steps=3 if success else EPISODE_LENGTH,
        elapsed_s=1.0,
        metadata={
            "task": task,
            "episode": episode,
            "protocol": PROTOCOL_VERSION,
            "python_hash_seed": PYTHON_HASH_SEED,
            "termination": "success" if success else "timeout",
            "episode_state_sha256": "0" * 64,
            "simulation_steps": 10,
        },
    )


def test_formal_aggregate_reads_tasks_from_records():
    adapter = RLBenchBenchmarkAdapter(config(TASK_NAMES, EPISODES_PER_TASK))
    records = tuple(
        record(task, episode) for task in EVALUATION_ORDER for episode in range(EPISODES_PER_TASK)
    )

    per_task = adapter._collect([formal_result(records)])

    assert {task: len(rows) for task, rows in per_task.items()} == dict.fromkeys(
        TASK_NAMES, EPISODES_PER_TASK
    )
    # The synthetic records carry no real test-episode digests, so the formal check rejects them.
    with pytest.raises(RuntimeError, match="replayed test episodes"):
        adapter.aggregate([formal_result(records)])


def test_suite_item_is_rejected_outside_a_formal_run():
    adapter = RLBenchBenchmarkAdapter(config(TASK_NAMES, 2))

    with pytest.raises(RuntimeError, match="unexpected RLBench task result"):
        adapter._collect([formal_result((record("close_jar", 0),))])
