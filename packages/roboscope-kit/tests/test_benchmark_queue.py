"""Work-queue claim order and out-of-memory handling for Workers that recycle."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from roboscope_kit.runtime.benchmark import orchestrator
from roboscope_kit.runtime.benchmark.adapters import ProcessSpec
from roboscope_kit.runtime.benchmark.orchestrator import ParallelBenchmarkRuntime, RuntimeConfig
from roboscope_kit.runtime.benchmark.processes import ManagedProcess, ProcessFailureKind
from roboscope_kit.runtime.benchmark.queue import WorkQueue
from roboscope_kit.runtime.benchmark.schema import WorkItem


def _queue(tmp_path: Path, groups: int = 3, items_per_group: int = 2) -> WorkQueue:
    queue = WorkQueue(tmp_path / "queue")
    queue.initialize(
        [
            WorkItem(
                ordinal=group * items_per_group + index,
                item_id=f"g{group}-{index}",
                group_id=f"g{group}",
                payload={},
            )
            for group in range(groups)
            for index in range(items_per_group)
        ]
    )
    return queue


def _claimed_group(queue: WorkQueue, worker_id: int, **kwargs) -> str:
    claimed = queue.claim(worker_id, **kwargs)
    assert claimed is not None
    return claimed.item.group_id


def test_claim_keeps_ordinal_order_without_spreading(tmp_path: Path) -> None:
    queue = _queue(tmp_path)

    claimed = [queue.claim(worker_id) for worker_id in range(6)]

    assert [work.item.ordinal for work in claimed if work is not None] == [0, 1, 2, 3, 4, 5]
    assert queue.claim(6) is None


def test_spreading_workers_start_on_different_groups(tmp_path: Path) -> None:
    queue = _queue(tmp_path)

    groups = [_claimed_group(queue, worker_id, spread_groups=True) for worker_id in range(3)]

    assert sorted(groups) == ["g0", "g1", "g2"]


def test_spreading_worker_keeps_its_preferred_group(tmp_path: Path) -> None:
    queue = _queue(tmp_path, items_per_group=3)
    assert _claimed_group(queue, 0, spread_groups=True) == "g0"

    assert _claimed_group(queue, 0, preferred_group="g0", spread_groups=True) == "g0"


def test_spreading_worker_that_loses_a_claim_chooses_its_group_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = _queue(tmp_path)
    # Worker 1 lists the candidates before worker 0 claims, as when Workers start together.
    stale = queue._pending_candidates(None, True)
    assert _claimed_group(queue, 0, spread_groups=True) == "g0"
    current = queue._pending_candidates
    listings = iter([stale])
    monkeypatch.setattr(
        queue, "_pending_candidates", lambda *args: next(listings, None) or current(*args)
    )

    assert _claimed_group(queue, 1, spread_groups=True) == "g1"


def _managed(tmp_path: Path, log: str) -> ManagedProcess:
    log_path = tmp_path / "worker.log"
    log_path.write_text(log, encoding="utf-8")
    return ManagedProcess(
        process=SimpleNamespace(returncode=1, poll=lambda: 1),
        log_file=io.StringIO(),
        log_path=log_path,
        spec=ProcessSpec(command=("unused",)),
    )


@pytest.mark.parametrize(
    "line",
    [
        "[2026-11-13 10:00:00.000] [SAPIEN] [critical] /physx/gpucommon/src/PxgCudaMemoryAllocator.cpp",
        "[SAPIEN] [critical] /physx/gpucommon/src/PxgCudaMemoryAllocator.cpp:112 allocation failed",
        "RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate`",
        "RuntimeError: CUSOLVER_STATUS_INTERNAL_ERROR, when calling `cusolverDnCreate(handle)`",
    ],
)
def test_gpu_allocation_failures_are_out_of_memory(tmp_path: Path, line: str) -> None:
    log = f"Traceback (most recent call last):\n{line}\nCUDA error: an illegal memory access\n"

    assert _managed(tmp_path, log).failure_kind() == ProcessFailureKind.OOM


@pytest.mark.parametrize(
    "line",
    [
        "[SAPIEN] [warning] PxgCudaMemoryAllocator.cpp is slow",
        "RuntimeError: CUSOLVER_STATUS_INTERNAL_ERROR when calling `cusolverDnSgetrf`",
    ],
)
def test_other_gpu_errors_are_not_out_of_memory(tmp_path: Path, line: str) -> None:
    assert _managed(tmp_path, f"{line}\n").failure_kind() == ProcessFailureKind.OTHER


class _Worker:
    def __init__(self, returncode: int | None) -> None:
        self.process = SimpleNamespace(returncode=returncode, poll=lambda: returncode)

    def stop(self) -> None:
        return None


def test_oom_scale_down_does_not_fail_a_worker_that_recycled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(orchestrator, "OOM_RESTART_DELAY_S", 0.0)
    runtime = ParallelBenchmarkRuntime.__new__(ParallelBenchmarkRuntime)
    runtime.config = RuntimeConfig(output_dir=tmp_path, gpu_devices=("0",), replicas=1, concurrency=3)
    failed, recycled, running = _Worker(1), _Worker(0), _Worker(None)
    runtime.workers = {0: failed, 1: recycled, 2: running}
    runtime.worker_contexts = {worker_id: SimpleNamespace(gpu_slot=0) for worker_id in range(3)}
    runtime.worker_start_order = {0: 0, 1: 1, 2: 2}
    runtime.state_dir = tmp_path / "state"
    runtime.state_dir.mkdir()
    (runtime.state_dir / "worker-001.json").write_text(json.dumps({"state": "recycling"}), encoding="utf-8")
    runtime.queue = SimpleNamespace(requeue_worker=lambda worker_id: 0)
    runtime.failure_events = []
    started: list[int] = []
    runtime._start_worker = started.append
    runtime._worker_failure_payload = lambda worker_id, managed: {"worker_id": worker_id}

    runtime._handle_gpu_oom(0, [(0, failed)], tmp_path / "events.jsonl")

    event = runtime.failure_events[-1]
    assert event["type"] == "worker-oom-scale-down"
    assert event["failed_worker_ids"] == [0]
    assert event["retired_worker_id"] == 2
    assert started == [0]
    assert 1 in runtime.workers
