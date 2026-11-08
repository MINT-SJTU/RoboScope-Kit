"""Opt-in diagnostic wall timings, independent of the Policy wire protocol.

Set ROBOSCOPE_TIMING_DIR to an absolute output directory before starting the runtime.
Records contain nested, inclusive durations: do not add parents to children or sum
parallel process durations as wall time. No observation or model payload is recorded.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

DIRECTORY = os.environ.get("ROBOSCOPE_TIMING_DIR", "")
ENABLED = bool(DIRECTORY)
_local = threading.local()
_lock = threading.Lock()


@contextmanager
def capture(kind: str, **metadata: object) -> Iterator[None]:
    if not ENABLED:
        yield
        return
    directory = Path(DIRECTORY)
    if not directory.is_absolute():
        raise ValueError("ROBOSCOPE_TIMING_DIR must be absolute")
    previous = getattr(_local, "record", None)
    record = {"kind": kind, "pid": os.getpid(), **metadata, "spans": {}, "ok": False}
    _local.record = record
    started = time.perf_counter()
    try:
        yield
        record["ok"] = True
    finally:
        record["wall_s"] = time.perf_counter() - started
        _local.record = previous
        directory.mkdir(parents=True, exist_ok=True)
        with _lock, (directory / f"{kind}-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")


def add(name: str, seconds: float) -> None:
    record = getattr(_local, "record", None)
    if record is not None:
        entry = record["spans"].setdefault(name, {"seconds": 0.0, "count": 0})
        entry["seconds"] += seconds
        entry["count"] += 1


@contextmanager
def span(name: str) -> Iterator[None]:
    if not ENABLED:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        add(name, time.perf_counter() - started)
