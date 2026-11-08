# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""CPU-only checks of Policy profiling and GPU-metric reporting."""

from __future__ import annotations

import os
import signal
import socket
import sqlite3
import sys
from contextlib import suppress
from pathlib import Path

import pytest

from roboscope_kit.runtime.benchmark import processes
from roboscope_kit.runtime.benchmark.adapters import ProcessSpec
from roboscope_kit.runtime.benchmark.nsys import GpuMetricsSession
from roboscope_kit.runtime.benchmark.processes import InferencePool


class _Adapter:
    def __init__(self, spec_factory) -> None:
        self.spec_factory = spec_factory
        self.released = []

    def baseline_replicas(self, _contexts) -> None:
        pass

    def replica_process(self, context):
        return self.spec_factory(context)

    def initialize_replica(self, _context) -> None:
        pass

    def finalize_replicas(self, _contexts) -> None:
        pass

    def release_replica(self, context) -> None:
        self.released.append(context.replica_id)


def test_missing_metrics_tool_is_reported_without_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("roboscope_kit.runtime.benchmark.nsys.shutil.which", lambda _name: None)
    session = GpuMetricsSession(tmp_path, ("0",), "ad10x-gfxt", 10000)
    session.start()
    session.stop()
    assert session.metadata["status"] == "unavailable"
    assert session.metadata["available"] is False
    assert (tmp_path / "profile/gpu-metrics.json").is_file()


@pytest.mark.parametrize(("sample_count", "expected_status"), [(0, "unavailable"), (2, "available")])
def test_metrics_require_exported_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sample_count: int, expected_status: str,
) -> None:
    monkeypatch.setattr("roboscope_kit.runtime.benchmark.nsys.shutil.which", lambda _name: "/fake/nsys")
    session = GpuMetricsSession(tmp_path, ("0",), "ad10x-gfxt", 10000)

    def fake_command(stage, _arguments, _timeout_s=60):
        session.metadata[stage] = {"returncode": 0, "output": "fake nsys"}
        if stage == "stop":
            session.report.touch()
        if stage == "export":
            with sqlite3.connect(session.database) as db:
                db.execute("CREATE TABLE GPU_METRICS (sample INTEGER)")
                db.executemany("INSERT INTO GPU_METRICS VALUES (?)", [(1,)] * sample_count)
        return True

    monkeypatch.setattr(session, "_command", fake_command)
    session.start()
    session.stop()
    assert session.metadata["status"] == expected_status
    assert session.metadata["available"] is (sample_count > 0)
    assert session.metadata["metric_sample_count"] == sample_count


def test_profile_rejects_tcp_policy_before_starting_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _Adapter(lambda _context: ProcessSpec((sys.executable, "server.py"), ready_port=12345))
    pool = InferencePool(
        adapter=adapter, gpu_devices=("0",), base_port=16694,
        output_dir=tmp_path / "out", ready_timeout_s=1, profile_dir=tmp_path / "profile",
    )

    def unexpected_start(*_args, **_kwargs):
        pytest.fail("a TCP-only Policy must be rejected before its process starts")

    monkeypatch.setattr(processes, "start_process", unexpected_start)
    with pytest.raises(ValueError, match="Unix-domain.*socket"):
        pool.scale(1)
    assert adapter.released == [0]
    assert pool.size == 0


def test_tcp_policy_remains_supported_without_profiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _Adapter(lambda _context: ProcessSpec((sys.executable, "server.py"), ready_port=12345))
    pool = InferencePool(
        adapter=adapter, gpu_devices=("0",), base_port=16694,
        output_dir=tmp_path / "out", ready_timeout_s=1,
    )

    class FakeManaged:
        def __init__(self, spec) -> None:
            self.spec = spec
            self.process = object()

        def stop(self) -> None:
            pass

    monkeypatch.setattr(processes, "start_process", lambda spec, *_args, **_kwargs: FakeManaged(spec))
    monkeypatch.setattr(processes, "wait_port", lambda *_args: None)
    pool.scale(1)
    assert pool.size == 1
    pool.stop()
    assert adapter.released == [0]


@pytest.mark.skipif(not hasattr(socket, "SO_PEERCRED"), reason="requires Unix peer credentials")
def test_profile_targets_unix_policy_group_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate Nsight putting its Policy child in a different process group.
    fake_nsys = tmp_path / "nsys"
    fake_nsys.write_text(
        "#!/usr/bin/env python3\n"
        "import subprocess, sys\n"
        "cmd = sys.argv[sys.argv.index('--kill=none') + 1:]\n"
        "child = subprocess.Popen(cmd, start_new_session=True)\n"
        "sys.exit(child.wait())\n"
    )
    fake_nsys.chmod(0o755)
    server = tmp_path / "server.py"
    server.write_text(
        "import os, socket, sys\n"
        "open(sys.argv[2], 'w').write(str(os.getpid()))\n"
        "sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
        "sock.bind(sys.argv[1]); sock.listen()\n"
        "while True:\n"
        "    conn, _ = sock.accept(); conn.close()\n"
    )
    pid_file = tmp_path / "server.pid"
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    class SocketAdapter(_Adapter):
        def release_replica(self, context) -> None:
            super().release_replica(context)
            if pid_file.exists():
                with suppress(ProcessLookupError):
                    os.kill(int(pid_file.read_text()), signal.SIGTERM)

    class FakeMps:
        def __init__(self) -> None:
            self.attached = None
            self.terminated = None

        def client_env(self, _slot):
            return {}

        def require_attached(self, group) -> None:
            self.attached = group

        def terminate_clients(self, group) -> None:
            self.terminated = group

    adapter = SocketAdapter(lambda context: ProcessSpec(
        (sys.executable, str(server), str(context.socket_path), str(pid_file)),
        ready_socket=context.socket_path,
    ))
    mps = FakeMps()
    pool = InferencePool(
        adapter=adapter, gpu_devices=("0",), base_port=16694,
        output_dir=tmp_path / "out", ready_timeout_s=5,
        cuda_mps=mps, profile_dir=tmp_path / "profile",
    )
    try:
        pool.scale(1)
        managed = pool.processes[0]
        assert managed is not None
        server_pid = int(pid_file.read_text())
        assert managed.process.pid != server_pid
        assert mps.attached == os.getpgid(server_pid) == server_pid
    finally:
        pool.stop()
    assert mps.terminated == server_pid
    assert managed.process.poll() is not None
    with pytest.raises(ProcessLookupError):
        os.kill(server_pid, 0)
