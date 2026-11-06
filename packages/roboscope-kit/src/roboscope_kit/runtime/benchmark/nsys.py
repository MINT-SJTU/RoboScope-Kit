# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Run a benchmark Worker under Nsight Systems for ``--profile``."""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import subprocess
import time
import uuid
from dataclasses import replace
from pathlib import Path

from roboscope_kit.profiling import DOMAIN, EPISODE_RANGE, PROFILE_ENV

from .adapters import ProcessSpec
from .queue import atomic_json

LOGGER = logging.getLogger(__name__)


class GpuMetricsSession:
    """One device-wide counter collector, separate from Worker/Policy traces.

    Counter ownership is per device, so starting a sampler in every profiled
    process would conflict. No benchmark or CUDA MPS behavior is changed here.
    """

    def __init__(
        self, output_dir: Path, gpu_devices: tuple[str, ...], metric_set: str, frequency_hz: int,
    ) -> None:
        self.directory = (output_dir / "profile").resolve()
        self.report = self.directory / "gpu-metrics.nsys-rep"
        self.database = self.directory / "gpu-metrics.sqlite"
        self.log = self.directory / "gpu-metrics.log"
        self.session = f"roboscope-gpu-{uuid.uuid4().hex}"
        self.env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(gpu_devices)}
        self.tool = shutil.which("nsys")
        self.started = False
        self.attempted = False
        self.metadata = {
            "requested": True, "status": "pending", "available": False,
            "scope": "device-wide; evaluation lifecycle including autotune and cleanup",
            "session": self.session, "devices": list(gpu_devices),
            "device_selector": "cuda-visible", "metric_set": metric_set,
            "frequency_hz": frequency_hz, "tool_version": None,
            "report_path": str(self.report), "sqlite_path": str(self.database),
            "log_path": str(self.log), "sampling_may_affect_timing": False,
            "start": {"status": "not_attempted"}, "stop": {"status": "not_started"},
        }

    def _publish(self) -> None:
        self.metadata.update(report_created=self.report.is_file(), sqlite_created=self.database.is_file())
        atomic_json(self.directory / "gpu-metrics.json", self.metadata)

    def _command(self, stage: str, arguments: list[str], timeout_s: float = 60) -> bool:
        command = [self.tool, *arguments]
        record = {"command": command, "started_unix_s": time.time()}
        try:
            result = subprocess.run(
                command, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, check=False, timeout=timeout_s,
            )
            record.update(returncode=result.returncode, output=result.stdout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            record.update(returncode=None, output=str(exc))
        record["status"] = "succeeded" if record["returncode"] == 0 else "failed"
        record["finished_unix_s"] = time.time()
        self.metadata[stage] = record
        with self.log.open("a", encoding="utf-8") as stream:
            stream.write(f"{stage}: {command!r}\n{record['output']}\n")
        return record["returncode"] == 0

    def _unavailable(self, reason: str) -> None:
        self.metadata.update(status="unavailable", available=False, reason=reason)
        with self.log.open("a", encoding="utf-8") as stream:
            stream.write(f"unavailable: {reason}\n")
        LOGGER.warning("GPU metrics unavailable: %s; diagnostics: %s", reason, self.log)
        self._publish()

    def start(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.tool is None:
            self._unavailable("nsys is not on PATH")
            return
        if not self._command("version", ["--version"]):
            self._unavailable("could not read the Nsight Systems version")
            return
        self.metadata["tool_version"] = self.metadata["version"]["output"].strip()
        self.attempted = True
        if not self._command("start", [
            "start", f"--session-new={self.session}",
            f"--output={self.report.with_suffix('')}",
            "--sample=none", "--cpuctxsw=none",
            "--gpu-metrics-devices=cuda-visible",
            f"--gpu-metrics-set={self.metadata['metric_set']}",
            f"--gpu-metrics-frequency={self.metadata['frequency_hz']}",
        ]):
            self._unavailable(self.metadata["start"]["output"].strip() or "nsys start failed")
            return
        self.started = True
        self.metadata.update(status="collecting", sampling_may_affect_timing=True)
        self._publish()

    def stop(self) -> None:
        if not self.attempted:
            return
        stopped = self._command("stop", ["stop", f"--session={self.session}"], 300) if self.started else False
        # The unique session belongs to this run, including a partially failed start.
        self._command("shutdown", ["shutdown", f"--session={self.session}", "--kill=none"])
        self.started = False
        if not stopped:
            if self.metadata["status"] != "unavailable":
                self._unavailable("nsys stop failed; see the collector log")
            else:
                self._publish()
            return
        if not self.report.is_file() or not self._command("export", [
            "export", "--type=sqlite", f"--output={self.database}", str(self.report),
        ], 300):
            self._unavailable("GPU metrics report or SQLite export is unavailable")
            return
        try:
            with sqlite3.connect(f"{self.database.as_uri()}?mode=ro", uri=True) as connection:
                tables = {row[0] for row in connection.execute(
                    "select name from sqlite_master where type='table'"
                )}
                count = (connection.execute("select count(*) from GPU_METRICS").fetchone()[0]
                         if "GPU_METRICS" in tables else 0)
                diagnostics = ([row[0] for row in connection.execute(
                    "select value from StringIds where lower(value) like '%gpu metrics%' "
                    "or lower(value) like '%permission%' or lower(value) like '%privilege%'"
                )] if "StringIds" in tables else [])
                if "DIAGNOSTIC_EVENT" in tables and any(
                    row[1] == "text" for row in connection.execute("pragma table_info(DIAGNOSTIC_EVENT)")
                ):
                    diagnostics.extend(row[0] for row in connection.execute(
                        "select text from DIAGNOSTIC_EVENT where text is not null"
                    ))
                diagnostics = list(dict.fromkeys(diagnostics))
        except sqlite3.Error as exc:
            self._unavailable(f"cannot inspect GPU metrics data: {exc}")
            return
        self.metadata.update(metric_sample_count=count, diagnostics=diagnostics)
        if not count:
            self._unavailable("no GPU_METRICS samples were exported; counters were not measured")
            return
        partial = any("permission" in message.lower() or "privilege" in message.lower()
                      for message in diagnostics)
        self.metadata.update(status="partial" if partial else "available", available=True)
        if partial:
            LOGGER.warning("GPU metric samples were exported with permission diagnostics; see %s", self.log)
        self._publish()


def require_nsys() -> None:
    if shutil.which("nsys") is None:
        raise ValueError("--profile requires Nsight Systems: nsys is not on PATH")


def profiled_process(
    spec: ProcessSpec, report: Path, *, capture_episode: bool = True, domain: str | None = DOMAIN,
) -> ProcessSpec:
    """Capture a Worker's first episode or a Policy process's complete lifetime."""
    report.parent.mkdir(parents=True, exist_ok=True)
    capture = (
        "--capture-range=nvtx",
        f"--nvtx-capture={EPISODE_RANGE}{'@' + domain if domain else ''}",
        "--capture-range-end=stop",
    ) if capture_episode else ()
    return replace(
        spec,
        command=(
            "nsys", "profile",
            "--output", str(report),
            "--force-overwrite=true",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            *capture,
            "--wait=all",
            "--kill=none",
            *spec.command,
        ),
        env={**spec.env, PROFILE_ENV: "1", "NSYS_NVTX_PROFILER_REGISTER_ONLY": "0"},
    )
