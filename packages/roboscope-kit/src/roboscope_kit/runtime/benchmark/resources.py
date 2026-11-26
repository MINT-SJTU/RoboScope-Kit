# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Host resource measurements shared by all benchmark adapters."""

from __future__ import annotations

import contextlib
import os
import resource
import statistics
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CpuStat:
    usage_usec: int
    nr_periods: int
    nr_throttled: int
    throttled_usec: int


CGROUP = Path("/sys/fs/cgroup")
PROC_SELF_CGROUP = Path("/proc/self/cgroup")


def _meminfo() -> dict[str, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value, *_ = line.replace(":", " ").split()
        values[key] = int(value) * 1024
    return values


def _own_cgroup(root: Path, proc_cgroup: Path) -> Path | None:
    """This process's cgroup v2 directory, or ``None`` without a unified hierarchy.

    Inside a container with its own cgroup namespace the path is ``/``, so this is ``root``.
    On a host, Slurm jobs and systemd scopes or services are cgroups below ``root``.
    """

    try:
        lines = proc_cgroup.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("0::"):
            own = root / line[3:].strip().lstrip("/")
            # A cgroup outside the mounted hierarchy: the mount is this process's own cgroup.
            return own if own.is_dir() else root
    return None


def host_memory_limit(root: Path = CGROUP, proc_cgroup: Path = PROC_SELF_CGROUP) -> int | None:
    """Host memory this process tree may use, else RAM.

    The limit is the smallest ``memory.max`` or ``memory.high`` of this process's cgroup and its
    ancestors, since a parent slice can hold the limit while the leaf reads ``max``.
    """

    limits = []
    cgroup = _own_cgroup(root, proc_cgroup)
    while cgroup is not None:
        for name in ("memory.max", "memory.high"):
            with contextlib.suppress(OSError, ValueError):
                path = cgroup / name
                if path.is_file() and (value := path.read_text(encoding="utf-8").strip()) != "max":
                    limits.append(int(value))
        cgroup = cgroup.parent if cgroup != root and root in cgroup.parents else None
    with contextlib.suppress(OSError, KeyError):
        limits.append(_meminfo()["MemTotal"])
    return min(limits, default=None)


def host_memory_used(root: Path = CGROUP, proc_cgroup: Path = PROC_SELF_CGROUP) -> int | None:
    """Host memory in use without reclaimable page cache: this process's cgroup's, else the machine's."""

    try:
        cgroup = _own_cgroup(root, proc_cgroup)
        if cgroup is not None and (cgroup / "memory.current").is_file():
            stat_text = (cgroup / "memory.stat").read_text(encoding="utf-8")
            stat = dict(line.split() for line in stat_text.splitlines())
            # ``file`` includes ``shmem`` (tmpfs, /dev/shm, shared memory segments), which
            # counts toward the limit and cannot be reclaimed without swap.
            reclaimable = int(stat.get("file", 0)) - int(stat.get("shmem", 0))
            return int((cgroup / "memory.current").read_text(encoding="utf-8")) - reclaimable
        meminfo = _meminfo()
        return meminfo["MemTotal"] - meminfo["MemAvailable"]
    except (OSError, KeyError, ValueError):
        return None


class GpuSampler:
    """Sample GPU utilization and memory, and host memory, until stopped."""

    def __init__(self, interval_s: float = 0.5, gpu_devices: tuple[str, ...] | None = None):
        self._interval_s = interval_s
        self._gpu_devices = set(gpu_devices) if gpu_devices is not None else None
        self._samples: list[list[dict[str, float | str]]] = []
        self.host_memory_samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="gpu-sampler", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> list[list[dict[str, float | str]]]:
        self._stop.set()
        self._thread.join(timeout=5)
        return self._samples

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self._interval_s)

    def _sample_once(self) -> None:
        host = host_memory_used()
        if host is not None:
            self.host_memory_samples.append(host)
        command = [
            "nvidia-smi",
            "--query-gpu=index,uuid,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return
        if result.returncode != 0:
            return
        sample = []
        for line in result.stdout.splitlines():
            index, uuid, utilization, memory_mib = [item.strip() for item in line.split(",")]
            if self._gpu_devices is not None and not {index, uuid} & self._gpu_devices:
                continue
            sample.append(
                {
                    "device": index if self._gpu_devices is None or index in self._gpu_devices else uuid,
                    "utilization": float(utilization) / 100,
                    "memory_bytes": float(memory_mib) * 1024**2,
                }
            )
        if sample:
            self._samples.append(sample)


def read_cpu_stat(path: Path = Path("/sys/fs/cgroup/cpu.stat")) -> CpuStat:
    if not path.is_file():
        usage = resource.getrusage(resource.RUSAGE_SELF)
        usage_usec = int((usage.ru_utime + usage.ru_stime) * 1e6)
        return CpuStat(usage_usec, 0, 0, 0)
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split()
        values[key] = int(value)
    return CpuStat(
        usage_usec=values.get("usage_usec", 0),
        nr_periods=values.get("nr_periods", 0),
        nr_throttled=values.get("nr_throttled", 0),
        throttled_usec=values.get("throttled_usec", 0),
    )


def cpu_quota(path: Path = Path("/sys/fs/cgroup/cpu.max")) -> float:
    if not path.is_file():
        return float(os.cpu_count() or 1)
    quota, period = path.read_text(encoding="utf-8").split()
    if quota == "max":
        return float(os.cpu_count() or 1)
    return int(quota) / int(period)


def cpu_window(before: CpuStat, after: CpuStat, elapsed_s: float, quota_cores: float) -> dict:
    usage_s = (after.usage_usec - before.usage_usec) / 1e6
    throttled_s = (after.throttled_usec - before.throttled_usec) / 1e6
    periods = after.nr_periods - before.nr_periods
    throttled_periods = after.nr_throttled - before.nr_throttled
    entitlement = elapsed_s * quota_cores
    return {
        "cpu_cores": usage_s / max(elapsed_s, 1e-9),
        "throttle_time_ratio": throttled_s / max(entitlement, 1e-9),
        "throttle_period_ratio": throttled_periods / max(periods, 1),
    }


def gpu_window(samples: list[list[dict[str, float | str]]], replicas: int) -> dict[str, object]:
    active_samples = [sample[:replicas] for sample in samples if sample]
    utilizations = [gpu["utilization"] for sample in active_samples for gpu in sample]
    memory_totals = [sum(gpu["memory_bytes"] for gpu in sample) for sample in active_samples]
    utilization_by_gpu: dict[str, list[float]] = {}
    memory_by_gpu: dict[str, list[float]] = {}
    for sample in active_samples:
        for gpu in sample:
            device = str(gpu["device"])
            utilization_by_gpu.setdefault(device, []).append(gpu["utilization"])
            memory_by_gpu.setdefault(device, []).append(gpu["memory_bytes"])
    return {
        "mean_utilization": statistics.fmean(utilizations) if utilizations else 0.0,
        "peak_memory_bytes": max(memory_totals, default=0.0),
        "gpu_util_avg": {
            str(index): 100 * statistics.fmean(values) for index, values in utilization_by_gpu.items()
        },
        "gpu_util_max": {str(index): 100 * max(values) for index, values in utilization_by_gpu.items()},
        "gpu_mem_mib_max": {str(index): max(values) / 1024**2 for index, values in memory_by_gpu.items()},
    }
