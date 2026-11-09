# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Reuse what one evaluation learned about a host in the next evaluation on that host.

An entry records the K/g the concurrency search selected and the mean wall time of
an episode of each task. A later evaluation with the same key skips the search and
dispatches the longest tasks first. The key covers everything that changes either:
the kit version and benchmark manifest, the benchmark, suite and episode count, the
checkpoint and Policy command, the GPU models, CPU quota and CUDA MPS.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .queue import atomic_json
from .schema import WorkItem

CACHE_MODES = ("use", "refresh", "off")
CACHE_SCHEMA = 1


def cache_root() -> Path:
    explicit = os.environ.get("ROBOSCOPE_RUN_CACHE_DIR")
    if explicit:
        return Path(explicit)
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "roboscope-kit" / "run-cache"


def gpu_models(gpu_devices: Sequence[str]) -> list[str]:
    """Model name of each GPU, or the device identifiers when nvidia-smi is unavailable."""

    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return [f"device:{device}" for device in gpu_devices]
    names: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            names[parts[0]] = parts[2]
            names[parts[1]] = parts[2]
    return [names.get(device, f"device:{device}") for device in gpu_devices]


def checkpoint_identity(checkpoint: Path) -> dict[str, Any]:
    """Path, size and modification time of the checkpoint (each file of a directory)."""

    files = [checkpoint] if checkpoint.is_file() else sorted(p for p in checkpoint.rglob("*") if p.is_file())
    entries = []
    for path in files:
        stat = path.stat()
        name = str(path.relative_to(checkpoint)) if path != checkpoint else ""
        entries.append([name, stat.st_size, stat.st_mtime_ns])
    return {"path": str(checkpoint.resolve()), "files": entries}


def cache_key(fields: Mapping[str, Any]) -> str:
    encoded = json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]


class RunCache:
    """One cache entry, read at start and updated after the evaluation."""

    def __init__(self, mode: str, key_fields: Mapping[str, Any], root: Path | None = None) -> None:
        if mode not in CACHE_MODES:
            raise ValueError(f"run cache mode must be one of {CACHE_MODES}")
        self.mode = mode
        self.key_fields = dict(key_fields)
        self.key = cache_key(self.key_fields)
        self.path = (root or cache_root()) / f"{self.key}.json"
        self.entry = self._load() if mode == "use" else {}

    def _load(self) -> dict[str, Any]:
        try:
            entry = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(entry, dict) or entry.get("schema") != CACHE_SCHEMA or entry.get("key") != self.key:
            return {}
        return entry

    @property
    def selection(self) -> dict[str, Any] | None:
        selection = self.entry.get("selection")
        return dict(selection) if isinstance(selection, dict) else None

    @property
    def task_seconds(self) -> dict[str, float]:
        return {str(task): float(seconds) for task, seconds in self.entry.get("task_seconds", {}).items()}

    def describe(self) -> dict[str, Any]:
        return {"mode": self.mode, "key": self.key, "path": str(self.path)}

    def store(self, *, selection: Mapping[str, Any] | None, task_seconds: Mapping[str, float]) -> None:
        if self.mode == "off":
            return
        entry = {
            "schema": CACHE_SCHEMA,
            "key": self.key,
            "key_fields": self.key_fields,
            "selection": dict(selection) if selection is not None else self.entry.get("selection"),
            "task_seconds": {**self.task_seconds, **dict(task_seconds)},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(self.path, entry)
        self.entry = entry


def task_seconds_from_queues(queue_dirs: Sequence[Path]) -> dict[str, float]:
    """Mean episode wall time per task from the completed items of finished runs."""

    totals: dict[str, list[float]] = defaultdict(list)
    for queue_dir in queue_dirs:
        for path in sorted((queue_dir / "completed").glob("*.json")):
            try:
                records = json.loads(path.read_text(encoding="utf-8")).get("records", [])
            except (OSError, ValueError):
                continue
            for record in records:
                task = (record.get("metadata") or {}).get("task")
                elapsed = record.get("elapsed_s")
                if task is not None and isinstance(elapsed, (int, float)) and elapsed > 0:
                    totals[str(task)].append(float(elapsed))
    return {task: sum(values) / len(values) for task, values in totals.items()}


def longest_first(items: Sequence[WorkItem], task_seconds: Mapping[str, float]) -> list[WorkItem]:
    """Order items by expected wall time, longest first; tasks without a measurement go first.

    Ordinals are renumbered so the queue hands items out in this order. Items without a
    ``task`` in their payload keep their place among the unmeasured ones.
    """

    from dataclasses import replace

    def cost(item: WorkItem) -> float:
        seconds = task_seconds.get(str(item.payload.get("task")))
        return float("inf") if seconds is None else seconds * item.expected_results

    ordered = sorted(items, key=lambda item: -cost(item))
    return [replace(item, ordinal=index) for index, item in enumerate(ordered)]
