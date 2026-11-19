# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""RoboTwin render services for ``render_mode: central``: a few per GPU, shared by its Workers."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


class RoboTwinRenderServices:
    """Start, restart and stop the render services that the RoboTwin Workers of one pass connect to.

    A service starts when the first Worker that needs it starts. A Worker that starts after its service
    has exited, for example a Worker restarted because its evaluator lost the service, starts a new one,
    so a restarted Worker never waits on a dead service. Each service also exits on its own when this
    process does.
    """

    def __init__(
        self,
        *,
        python: Path,
        script: Path,
        robotwin_root: Path,
        services_per_gpu: int,
        log_dir: Path,
        ready_timeout_s: float = 300.0,
    ) -> None:
        if services_per_gpu < 1:
            raise ValueError("RoboTwin render_services_per_gpu must be positive")
        self.python = python
        self.script = script
        self.robotwin_root = robotwin_root
        self.services_per_gpu = services_per_gpu
        self.log_dir = log_dir
        self.ready_timeout_s = ready_timeout_s
        # Services are keyed by (GPU slot, service index); the slot is a small integer, so socket and log
        # names stay short and valid for GPU UUIDs and MIG identifiers.
        self.processes: dict[tuple[int, int], subprocess.Popen] = {}
        self.starts: dict[tuple[int, int], int] = {}
        self.assignments: dict[tuple[int, int], int] = {}
        # Unix socket paths are limited to about 100 bytes, so they live in a short private directory.
        self.socket_dir: Path | None = None

    def socket_for(self, gpu_device: str, gpu_slot: int, worker_id: int) -> Path:
        """The socket of this Worker's service; starts the service if it is not running.

        A GPU's Workers are spread over its services in the order they first start, and a restarted
        Worker keeps its service.
        """

        assigned = sum(1 for slot, _ in self.assignments if slot == gpu_slot)
        index = self.assignments.setdefault((gpu_slot, worker_id), assigned % self.services_per_gpu)
        key = (gpu_slot, index)
        process = self.processes.get(key)
        # A service removes its socket while it shuts down, so a running service without one is exiting.
        if process is None or process.poll() is not None or not self._socket(key).exists():
            if process is not None:
                self._stop(process)
            self._start(key, gpu_device)
        return self._socket(key)

    def _socket(self, key: tuple[int, int]) -> Path:
        if self.socket_dir is None:
            self.socket_dir = Path(tempfile.mkdtemp(prefix="roboscope-robotwin-render-"))
        gpu_slot, index = key
        return self.socket_dir / f"gpu{gpu_slot}-{index}.sock"

    def _start(self, key: tuple[int, int], gpu_device: str) -> None:
        socket = self._socket(key)
        socket.unlink(missing_ok=True)
        gpu_slot, index = key
        attempt = self.starts.get(key, 0)
        self.starts[key] = attempt + 1
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"gpu{gpu_slot}-service-{index}-attempt-{attempt:03d}.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                (
                    str(self.python),
                    str(self.script),
                    "--socket", str(socket),
                    "--robotwin-root", str(self.robotwin_root),
                    "--parent-pid", str(os.getpid()),
                ),
                cwd=self.robotwin_root,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu_device, "PYTHONDONTWRITEBYTECODE": "1"},
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                # A terminal SIGINT reaches the Kit, which stops the Workers before the services; the
                # services must not stop first under the Workers.
                start_new_session=True,
            )
        self.processes[key] = process
        deadline = time.monotonic() + self.ready_timeout_s
        while not socket.exists():
            if process.poll() is not None or time.monotonic() > deadline:
                self._stop(process)
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                raise RuntimeError(f"RoboTwin render service on GPU {gpu_device} did not start:\n{tail}")
            time.sleep(0.1)

    @staticmethod
    def _stop(process: subprocess.Popen) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def close(self) -> None:
        processes, self.processes = self.processes, {}
        for process in processes.values():
            self._stop(process)
        if self.socket_dir is not None:
            shutil.rmtree(self.socket_dir, ignore_errors=True)
            self.socket_dir = None


__all__ = ["RoboTwinRenderServices"]
