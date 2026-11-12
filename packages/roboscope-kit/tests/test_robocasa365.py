# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Validate native group accounting without importing the simulator or Policy."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from roboscope_kit import ActionChunk
from roboscope_kit.benchmark_specs import BenchmarkName
from roboscope_kit.runtime.benchmark.factories import BenchmarkRunRequest
from roboscope_kit.runtime.benchmark.manifest import load_benchmark_manifest
from roboscope_kit.runtime.benchmark.orchestrator import aggregate_standard_results
from roboscope_kit.runtime.benchmark.schema import EvaluationRecord, WorkItem, WorkResult
from roboscope_kit.runtime.benchmark.shard_reduce import merge_shard_results
from roboscope_kit.runtime.benchmark.sharding import select_shard_items, shard_info
from roboscope_kit.runtime.benchmarks.robocasa365.adapter import (
    RoboCasa365BenchmarkAdapter,
    RoboCasa365RuntimeConfig,
)
from roboscope_kit.runtime.benchmarks.robocasa365.factory import RoboCasa365BenchmarkFactory
from roboscope_kit.runtime.benchmarks.robocasa365.protocol import (
    CONSTRUCTOR_SEED,
    EPISODES_PER_TASK,
    NATIVE_ENVS,
    PROTOCOL_VERSION,
    ROLLOUT_LABEL_BASE,
    SCENE_SPLIT,
    SUITES,
    TASKS_BY_NAME,
)

TASKS = tuple(SUITES[split][0] for split in ("atomic_seen", "composite_seen", "composite_unseen"))
MANIFEST = Path(__file__).resolve().parents[3] / "benchmarks/robocasa365/benchmark-manifest.json"


def _config(**changes: Any) -> RoboCasa365RuntimeConfig:
    config = RoboCasa365RuntimeConfig(
        python=Path("/usr/bin/python3"), benchmark_root=Path("/sim/robocasa365"),
        suite="target50", tasks=TASKS, episodes_per_task=NATIVE_ENVS,
    )
    return replace(config, **changes)


def _request(*, quick_test: bool = False) -> BenchmarkRunRequest:
    return BenchmarkRunRequest(
        benchmark=BenchmarkName.ROBOCASA365, suite="target50",
        checkpoint_path=Path("/models/checkpoint"), output_dir=Path("/runs/output"),
        episodes=None, quick_test=quick_test,
    )


def _result(item: WorkItem) -> WorkResult:
    task = TASKS_BY_NAME[item.group_id]
    records = tuple(
        EvaluationRecord(
            f"{task.name}-rollout-{ROLLOUT_LABEL_BASE + index}", success=index % 2 == 0,
            steps=task.horizon, elapsed_s=0.25,
            metadata={
                "task": task.name, "split": task.split, "scene_split": SCENE_SPLIT,
                "episode_index": index, "slot": index % NATIVE_ENVS,
                "constructor_seed": CONSTRUCTOR_SEED + index % NATIVE_ENVS,
                "stream_episode": index // NATIVE_ENVS, "termination": "horizon",
                "initial_state_sha256": "a" * 64, "scenario_metadata_sha256": "b" * 64,
                "initial_observation_sha256": "c" * 64,
            },
        )
        for index in range(item.expected_results)
    )
    return WorkResult(item.item_id, item.group_id, 0, records)


@pytest.mark.parametrize("changes, message", [
    ({"episodes_per_task": 4}, "divisible"),
    ({"episodes_per_task": 51}, "episodes must be"),
    ({"native_envs": 2}, "native_envs must be"),
    ({"tasks": (TASKS[0], TASKS[0])}, "unique"),
    ({"suite": "atomic_seen", "tasks": (TASKS[1],)}, "outside"),
    ({"cpu_threads": 0}, "cpu_threads"),
])
def test_config_rejects_invalid_native_workloads(changes: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**changes)


def test_full_suite_requires_every_task_and_official_group_count() -> None:
    config = _config(tasks=SUITES["target50"], episodes_per_task=EPISODES_PER_TASK)
    assert config.full_suite
    assert not replace(config, episodes_per_task=NATIVE_ENVS).full_suite
    assert not replace(config, tasks=TASKS).full_suite
    assert not replace(config, native_envs=1).full_suite
    items = RoboCasa365BenchmarkAdapter(config).work_items()
    assert len(items) == 50
    assert sum(item.expected_results for item in items) == 2500


def test_quick_test_uses_one_complete_native_group() -> None:
    adapter = RoboCasa365BenchmarkFactory().create(
        _request(quick_test=True), load_benchmark_manifest(MANIFEST),
    )
    items = adapter.work_items()
    assert len(items) == 1
    assert items[0].group_id == SUITES["target50"][0]
    assert items[0].expected_results == NATIVE_ENVS
    assert items[0].payload["native_envs"] == NATIVE_ENVS
    assert not adapter.config.full_suite


def test_probe_preserves_groups_and_supplies_every_worker() -> None:
    manifest = load_benchmark_manifest(MANIFEST)
    factory = RoboCasa365BenchmarkFactory()
    small = factory.create_probe(_request(), manifest, concurrency=1)
    large = factory.create_probe(_request(), manifest, concurrency=2)
    assert len(small.work_items()) == 3
    assert len(large.work_items()) == 6
    assert set(small.config.tasks) < set(large.config.tasks)
    assert all(item.expected_results == NATIVE_ENVS for item in large.work_items())
    assert all(item.payload["native_envs"] == NATIVE_ENVS for item in large.work_items())


@pytest.mark.parametrize("changed_field", ["version", "constructor_seed", "task_horizons"])
def test_factory_rejects_changed_frozen_protocol(changed_field: str) -> None:
    manifest = load_benchmark_manifest(MANIFEST)
    if changed_field == "version":
        manifest.protocol_versions[BenchmarkName.ROBOCASA365] = PROTOCOL_VERSION + "-changed"
    elif changed_field == "task_horizons":
        manifest.benchmark_protocols[BenchmarkName.ROBOCASA365]["task_horizons"][TASKS[0]] -= 1
    else:
        manifest.benchmark_protocols[BenchmarkName.ROBOCASA365]["constructor_seed"] += 1
    with pytest.raises(ValueError, match="manifest"):
        RoboCasa365BenchmarkFactory().create(_request(), manifest)


def test_aggregate_reports_complete_task_and_split_counts() -> None:
    adapter = RoboCasa365BenchmarkAdapter(_config())
    output = adapter.aggregate([_result(item) for item in adapter.work_items()])
    assert output["coverage"] == {"expected": 15, "completed": 15, "complete": True}
    assert [(row["task"], row["episodes"], row["successes"]) for row in output["tasks"]] == [
        (task, 5, 3) for task in TASKS
    ]
    assert [row["split"] for row in output["splits"]] == [
        "atomic_seen", "composite_seen", "composite_unseen",
    ]
    assert all(row["episodes"] == 5 and row["success_rate"] == 0.6 for row in output["splits"])
    assert not output["full_suite"] and not output["native_equivalence_verified"]
    partial = adapter.aggregate([_result(adapter.work_items()[0])])
    assert partial["coverage"] == {"expected": 15, "completed": 5, "complete": False}


@pytest.mark.parametrize("case, message", [
    ("duplicate", "duplicate or malformed"),
    ("malformed_id", "duplicate or malformed"),
    ("out_of_range", "outside the manifest"),
    ("short_horizon", "stopped before its horizon"),
    ("false_group_success", "lacks its group success"),
    ("invalid_termination", "invalid termination"),
    ("missing_digest", "lacks valid"),
])
def test_aggregate_rejects_invalid_episode_records(case: str, message: str) -> None:
    adapter = RoboCasa365BenchmarkAdapter(_config())
    result = _result(adapter.work_items()[0])
    record = result.records[0]
    if case == "duplicate":
        records = (record, record)
    else:
        metadata = dict(record.metadata)
        if case == "malformed_id":
            record = replace(record, record_id="wrong-task-rollout-100000")
        elif case == "out_of_range":
            metadata["episode_index"] = NATIVE_ENVS
            identifier = f"{result.group_id}-rollout-{ROLLOUT_LABEL_BASE + NATIVE_ENVS}"
            record = replace(record, record_id=identifier)
        elif case == "short_horizon":
            record = replace(record, steps=record.steps - 1)
        elif case == "false_group_success":
            metadata["termination"] = "native_group_success"
            record = replace(record, success=False)
        elif case == "invalid_termination":
            metadata["termination"] = "unknown"
        else:
            metadata.pop("initial_observation_sha256")
        records = (replace(record, metadata=metadata),)
    with pytest.raises(RuntimeError, match=message):
        adapter.aggregate([replace(result, records=records)])


@pytest.mark.parametrize("field, value", [
    ("task", "another-task"), ("scene_split", "target"), ("slot", 1),
    ("constructor_seed", CONSTRUCTOR_SEED + 1), ("stream_episode", 1),
])
def test_aggregate_rejects_changed_episode_provenance(field: str, value: Any) -> None:
    adapter = RoboCasa365BenchmarkAdapter(_config())
    result = _result(adapter.work_items()[0])
    record = result.records[0]
    record = replace(record, metadata={**record.metadata, field: value})
    with pytest.raises(RuntimeError, match="invalid provenance"):
        adapter.aggregate([replace(result, records=(record,))])


def _shards(adapter: RoboCasa365BenchmarkAdapter, count: int) -> list[dict]:
    items = adapter.work_items()
    expected = sum(item.expected_results for item in items)
    documents = []
    for index in range(count):
        selected = select_shard_items(items, index, count)
        results = [_result(item) for item in selected]
        documents.append({
            "sdk_version": 1,
            "shard": shard_info(index, count, expected, 500, True, {
                "kit_version": "test", "manifest_sha256": "a" * 64,
            }),
            "benchmark": adapter.metadata(),
            "result": aggregate_standard_results(results, sum(item.expected_results for item in selected)),
            "official_result": adapter.aggregate(results),
        })
    return documents


@pytest.mark.parametrize("count", [2, 3])
def test_shard_merge_preserves_complete_official_results(count: int) -> None:
    adapter = RoboCasa365BenchmarkAdapter(_config())
    merged = merge_shard_results(_shards(adapter, count))
    whole = adapter.aggregate([_result(item) for item in adapter.work_items()])
    assert merged["official_result"] == whole
    assert merged["result"]["episodes"] == 15
    assert merged["result"]["successes"] == 9
    assert merged["validation"]["total_records"] == 15


def test_shard_merge_rejects_different_expected_workloads() -> None:
    adapter = RoboCasa365BenchmarkAdapter(_config())
    documents = _shards(adapter, 2)
    documents[1]["official_result"]["coverage"]["expected"] += NATIVE_ENVS
    with pytest.raises(ValueError, match="disagree on the expected workload"):
        merge_shard_results(documents)


@pytest.mark.parametrize("stop_step", [8, 16])
def test_group_success_keeps_slots_stepping_and_uses_first_retained_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stop_step: int,
) -> None:
    from roboscope_kit.runtime.benchmarks.robocasa365 import worker

    for name in ("robocasa", "robocasa.utils", "robocasa.utils.env_utils"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setattr(sys.modules["robocasa.utils.env_utils"], "convert_action", lambda action: action,
                        raising=False)
    monkeypatch.setattr(worker, "build_observation", lambda identifier, step, history: SimpleNamespace(
        episode_id=identifier, step=step,
    ))
    monkeypatch.setattr(worker, "reset_fingerprint", lambda environment, observation: {})
    monkeypatch.setattr(worker, "save_first_batch", lambda *args: None)

    class Environment:
        def __init__(self, slot: int) -> None:
            self.steps = 0
            # Individual successes happen early. The group barrier still waits
            # for all slots' penultimate physical-step success values.
            self.success_steps = {slot + 1, stop_step - 1}
            if stop_step == 16:
                self.success_steps.add(8)

        def reset(self):
            return {}, {}

        def step(self, action):
            assert action.shape == (12,)
            self.steps += 1
            return {}, 0.0, False, False, {"success": self.steps in self.success_steps}

    class Client:
        def __init__(self) -> None:
            self.reset_ids = []
            self.end_ids = []
            self.batch_steps = []

        def episode_group(self):
            return nullcontext()

        def reset(self, context):
            self.reset_ids.append(context.episode_id)

        def act_batch(self, observations):
            self.batch_steps.append([observation.step for observation in observations])
            return tuple(ActionChunk(np.zeros((8, 12), dtype=np.float32)) for _ in observations)

        def end_episode(self, identifier):
            self.end_ids.append(identifier)

    environments = [Environment(slot) for slot in range(NATIVE_ENVS)]
    client = Client()
    args = SimpleNamespace(save_videos=False, output_dir=tmp_path, worker_id=0)
    task = replace(TASKS_BY_NAME[TASKS[0]], horizon=24)
    records, _ = worker.run_group(args, task, environments, 0, client)
    assert [environment.steps for environment in environments] == [stop_step] * NATIVE_ENVS
    assert client.batch_steps == [[step] * NATIVE_ENVS for step in range(0, stop_step, 8)]
    assert client.reset_ids == client.end_ids == [record.record_id for record in records]
    assert all(record.success and record.steps == stop_step for record in records)
    assert all(record.metadata["termination"] == "native_group_success" for record in records)
