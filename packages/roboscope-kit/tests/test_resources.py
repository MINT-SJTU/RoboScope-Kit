# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from roboscope_kit.runtime.benchmark import resources
from roboscope_kit.runtime.benchmark.resources import (
    CpuStat,
    cpu_quota,
    cpu_window,
    gpu_window,
    read_cpu_stat,
)


@pytest.mark.parametrize(("text", "expected"), [("2000000 100000", 20.0), ("150000 100000", 1.5)])
def test_cpu_quota_reads_cgroup_limit(tmp_path: Path, text: str, expected: float) -> None:
    path = tmp_path / "cpu.max"
    path.write_text(text + "\n", encoding="utf-8")

    assert cpu_quota(path) == expected


@pytest.mark.parametrize("text", [None, "max 100000\n"])
def test_cpu_quota_falls_back_to_cpu_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str | None
) -> None:
    path = tmp_path / "cpu.max"
    if text is not None:
        path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(resources.os, "cpu_count", lambda: 12)

    assert cpu_quota(path) == 12.0


def test_read_cpu_stat_uses_named_fields(tmp_path: Path) -> None:
    path = tmp_path / "cpu.stat"
    path.write_text(
        "usage_usec 2450000\nuser_usec 2200000\nsystem_usec 250000\n"
        "nr_periods 100\nnr_throttled 12\nthrottled_usec 90000\nnr_bursts 0\n",
        encoding="utf-8",
    )

    assert read_cpu_stat(path) == CpuStat(2_450_000, 100, 12, 90_000)
    path.write_text("usage_usec 7\n", encoding="utf-8")
    assert read_cpu_stat(path) == CpuStat(7, 0, 0, 0)


def test_read_cpu_stat_missing_file_uses_this_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def getrusage(who: int) -> SimpleNamespace:
        assert who == resources.resource.RUSAGE_SELF
        return SimpleNamespace(ru_utime=1.25, ru_stime=0.125)

    monkeypatch.setattr(resources.resource, "getrusage", getrusage)
    assert read_cpu_stat(tmp_path / "cpu.stat") == CpuStat(1_375_000, 0, 0, 0)


def test_cpu_window_uses_deltas_and_quota() -> None:
    before = CpuStat(2_000_000, 40, 6, 500_000)
    after = CpuStat(8_000_000, 50, 8, 2_000_000)

    assert cpu_window(before, after, elapsed_s=4.0, quota_cores=1.5) == {
        "cpu_cores": 1.5,
        "throttle_time_ratio": 0.25,
        "throttle_period_ratio": 0.2,
    }


def test_cpu_window_handles_zero_elapsed_time_and_periods() -> None:
    stat = CpuStat(2_000_000, 40, 6, 500_000)

    assert cpu_window(stat, stat, elapsed_s=0.0, quota_cores=1.5) == {
        "cpu_cores": 0.0,
        "throttle_time_ratio": 0.0,
        "throttle_period_ratio": 0.0,
    }


def _gpu(device: str, utilization: float, memory_mib: int) -> dict[str, float | str]:
    return {"device": device, "utilization": utilization, "memory_bytes": float(memory_mib * 1024**2)}


def test_gpu_window_aggregates_only_active_replicas() -> None:
    state = gpu_window(
        [
            [_gpu("0", 0.25, 6), _gpu("GPU-b", 0.5, 12), _gpu("2", 1.0, 99)],
            [],
            [_gpu("0", 0.75, 10), _gpu("GPU-b", 0.0, 2)],
        ],
        replicas=2,
    )

    # The peak total is 18 MiB in one sample, not the sum (22 MiB) of each device's peak.
    assert state == {
        "mean_utilization": 0.375,
        "peak_memory_bytes": 18.0 * 1024**2,
        "gpu_util_avg": {"0": 50.0, "GPU-b": 25.0},
        "gpu_util_max": {"0": 75.0, "GPU-b": 50.0},
        "gpu_mem_mib_max": {"0": 10.0, "GPU-b": 12.0},
    }


@pytest.mark.parametrize("samples", [[], [[], []]])
def test_gpu_window_empty_input(samples: list[list[dict[str, float | str]]]) -> None:
    assert gpu_window(samples, replicas=2) == {
        "mean_utilization": 0.0,
        "peak_memory_bytes": 0.0,
        "gpu_util_avg": {},
        "gpu_util_max": {},
        "gpu_mem_mib_max": {},
    }
