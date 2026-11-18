# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from pathlib import Path

from roboscope_kit.runtime.benchmark.cuda_mps import CudaMpsSession


def test_control_commands_from_parallel_replica_starts_do_not_overlap(tmp_path: Path) -> None:
    # The real control daemon refuses a second connection while one is open.
    busy = tmp_path / "busy"
    control = tmp_path / "nvidia-cuda-mps-control"
    control.write_text(
        f"#!/bin/sh\nmkdir {busy} 2>/dev/null || {{ echo 'Cannot send command'; exit 1; }}\n"
        f"sleep 0.05\nrmdir {busy}\n",
        encoding="utf-8",
    )
    control.chmod(0o755)
    session = CudaMpsSession(gpu_devices=("0", "1", "2", "3"), log_dir=tmp_path, control=str(control))
    session.pipe_dir = tmp_path
    errors: list[Exception] = []

    def query() -> None:
        try:
            session.clients()
        except Exception as exc:  # noqa: BLE001 - collected for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=query) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
