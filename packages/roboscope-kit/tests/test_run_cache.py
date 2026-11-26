# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from roboscope_kit.runtime.benchmark.run_cache import (
    RunCache,
    cache_key,
    longest_first,
    task_seconds_from_queues,
)
from roboscope_kit.runtime.benchmark.schema import WorkItem


def _item(ordinal: int, task: str | None, expected: int = 1) -> WorkItem:
    payload = {} if task is None else {"task": task}
    return WorkItem(ordinal, f"item-{ordinal}", f"group-{ordinal}", payload, expected)


def test_longest_first_puts_unmeasured_tasks_first_and_renumbers() -> None:
    items = [_item(0, "short"), _item(1, "long"), _item(2, "unknown"), _item(3, "mid", 2), _item(4, None)]

    ordered = longest_first(items, {"short": 10.0, "long": 100.0, "mid": 40.0})

    assert [item.item_id for item in ordered] == ["item-2", "item-4", "item-1", "item-3", "item-0"]
    assert [item.ordinal for item in ordered] == [0, 1, 2, 3, 4]


def test_task_seconds_are_the_mean_episode_time_per_task(tmp_path: Path) -> None:
    completed = tmp_path / "queue" / "completed"
    completed.mkdir(parents=True)
    records = [
        {"elapsed_s": 10.0, "metadata": {"task": "a"}},
        {"elapsed_s": 30.0, "metadata": {"task": "a"}},
        {"elapsed_s": 5.0, "metadata": {"task": "b"}},
        {"elapsed_s": 0.0, "metadata": {"task": "c"}},
        {"metadata": {"task": "d"}},
    ]
    (completed / "item-000000-x.json").write_text(json.dumps({"records": records}), encoding="utf-8")
    (completed / "item-000001-y.json").write_text("not json", encoding="utf-8")

    assert task_seconds_from_queues([tmp_path / "queue", tmp_path / "missing"]) == {"a": 20.0, "b": 5.0}


def test_run_cache_round_trip_refresh_and_off(tmp_path: Path) -> None:
    fields = {"benchmark": "robotwin", "gpu_models": ["RTX PRO 5000"]}
    first = RunCache("use", fields, root=tmp_path)
    assert first.selection is None and first.task_seconds == {}

    first.store(selection={"k": 48, "g": 8, "mode": "dynamic-probe"}, task_seconds={"a": 20.0})
    second = RunCache("use", fields, root=tmp_path)
    assert second.selection == {"k": 48, "g": 8, "mode": "dynamic-probe"}
    assert second.task_seconds == {"a": 20.0}

    # A cached run stores no new selection and keeps the earlier one; task times merge.
    second.store(selection=None, task_seconds={"b": 5.0})
    assert RunCache("use", fields, root=tmp_path).task_seconds == {"a": 20.0, "b": 5.0}
    assert RunCache("use", fields, root=tmp_path).selection["k"] == 48

    # A run that ran out of memory drops the cached K/g and keeps the task times.
    RunCache("use", fields, root=tmp_path).forget_selection()
    assert RunCache("use", fields, root=tmp_path).selection is None
    assert RunCache("use", fields, root=tmp_path).task_seconds == {"a": 20.0, "b": 5.0}
    first.store(selection={"k": 48, "g": 8, "mode": "dynamic-probe"}, task_seconds={})

    refreshed = RunCache("refresh", fields, root=tmp_path)
    assert refreshed.selection is None and refreshed.task_seconds == {}

    off = RunCache("off", fields, root=tmp_path)
    off.store(selection={"k": 1, "g": 1}, task_seconds={})
    assert RunCache("use", fields, root=tmp_path).selection["k"] == 48

    other = RunCache("use", {**fields, "gpu_models": ["RTX 4090"]}, root=tmp_path)
    assert other.key != first.key and other.selection is None


def test_run_cache_ignores_a_corrupt_or_foreign_entry(tmp_path: Path) -> None:
    fields = {"benchmark": "robotwin"}
    path = tmp_path / f"{cache_key(fields)}.json"
    path.write_text("{", encoding="utf-8")
    assert RunCache("use", fields, root=tmp_path).selection is None
    path.write_text(json.dumps({"schema": 1, "key": "other", "selection": {"k": 2}}), encoding="utf-8")
    assert RunCache("use", fields, root=tmp_path).selection is None


def test_run_cache_rejects_an_unknown_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run cache mode"):
        RunCache("always", {}, root=tmp_path)


def test_run_cache_keeps_one_entry_per_execution_contract(tmp_path: Path) -> None:
    fields = {"benchmark": "libero"}
    canonical = {"benchmark": "libero", "image_sizes": {}}
    small = {"benchmark": "libero", "image_sizes": {"images.primary": [128, 128]}}
    first = RunCache("use", fields, root=tmp_path)
    assert not first.has_sized_entries
    first.bind(canonical)
    first.store(selection={"k": 8, "g": 1, "mode": "dynamic-probe"}, task_seconds={"a": 20.0})
    second = RunCache("use", fields, root=tmp_path)
    second.bind(small)
    assert not second.has_sized_entries and second.selection is None and second.task_seconds == {}
    second.store(selection={"k": 12, "g": 1, "mode": "dynamic-probe"}, task_seconds={"a": 9.0})
    reread = RunCache("use", fields, root=tmp_path)
    assert reread.has_sized_entries
    assert reread.task_seconds_for(canonical) == {"a": 20.0}
    assert reread.task_seconds_for(small) == {"a": 9.0}
    reread.bind(canonical)
    assert reread.selection["k"] == 8

    refreshed = RunCache("refresh", fields, root=tmp_path)
    refreshed.bind(small)
    assert refreshed.selection is None
    refreshed.store(selection=None, task_seconds={"b": 1.0})
    after = RunCache("use", fields, root=tmp_path)
    assert after.task_seconds_for(small) == {"b": 1.0}
    assert after.task_seconds_for(canonical) == {"a": 20.0}


def test_run_cache_reads_entries_written_before_image_size_negotiation(tmp_path: Path) -> None:
    fields = {"benchmark": "libero"}
    legacy = {
        "schema": 1,
        "key": cache_key(fields),
        "key_fields": fields,
        "selection": {"k": 8, "g": 1, "mode": "dynamic-probe"},
        "task_seconds": {"a": 20.0},
    }
    (tmp_path / f"{cache_key(fields)}.json").write_text(json.dumps(legacy), encoding="utf-8")
    cache = RunCache("use", fields, root=tmp_path)
    cache.bind({"benchmark": "libero", "image_sizes": {}, "canonical_observations": True})
    assert cache.selection["k"] == 8 and cache.task_seconds == {"a": 20.0}
    small = {"benchmark": "libero", "image_sizes": {"images.primary": [128, 128]}}
    assert cache.task_seconds_for(small) == {}
