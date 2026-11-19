# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Simulator-free checks for RoboTwin central rendering (``render_mode: central``)."""

from __future__ import annotations

import os
import signal
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import roboscope_kit
from roboscope_kit.runtime.benchmark import orchestrator
from roboscope_kit.runtime.benchmark.adapters import WorkerContext
from roboscope_kit.runtime.benchmark.orchestrator import ParallelBenchmarkRuntime, RuntimeConfig
from roboscope_kit.runtime.benchmark.schema import WorkItem
from roboscope_kit.runtime.benchmarks.robotwin import (
    ROBOTWIN_ALL_TASKS,
    RoboTwinBenchmarkAdapter,
    RoboTwinRuntimeConfig,
)
from roboscope_kit.runtime.benchmarks.robotwin.render_service import RoboTwinRenderServices
from roboscope_kit.runtime.benchmarks.robotwin.worker import (
    RENDER_SERVICE_LOST_PREFIX,
    _gpu_env,
    _run_task_single_process,
)

# Binds the socket like the real service, records its GPU visibility and removes the socket on SIGTERM.
FAKE_SERVICE = """
import argparse, os, signal, socket, sys, time
parser = argparse.ArgumentParser()
parser.add_argument("--socket")
parser.add_argument("--robotwin-root")
parser.add_argument("--parent-pid", type=int)
args = parser.parse_args()
print("CUDA_VISIBLE_DEVICES=" + os.environ["CUDA_VISIBLE_DEVICES"], flush=True)
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(args.socket)
server.listen(1)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
try:
    while True:
        time.sleep(0.05)
finally:
    os.unlink(args.socket)
"""


@pytest.fixture
def render_module(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3]))
    from benchmarks.robotwin import roboscope_robotwin_render

    return roboscope_robotwin_render


def _config(**overrides) -> RoboTwinRuntimeConfig:
    values = {
        "sdk_python": Path("/opt/kit/bin/python"),
        "benchmark_python": Path("/sim/robotwin/bin/python"),
        "robotwin_root": Path("/sim/robotwin"),
        "checkpoint_path": Path("/models/checkpoint"),
        "deploy_policy_path": Path("/sim/robotwin/deploy_policy.yml"),
        "launcher_path": Path("/sim/robotwin/launcher.py"),
        "suite": "demo_clean",
        "tasks": ROBOTWIN_ALL_TASKS[:1],
        "episodes_per_task": 1,
        "episode_shard_size": 1,
        "seed_cache_dir": Path("/sim/robotwin/seed-cache"),
        "render_mode": "central",
        "render_service_path": Path("/sim/robotwin/render.py"),
    }
    values.update(overrides)
    return RoboTwinRuntimeConfig(**values)


def _context(tmp_path: Path, *, representation: str = "robotwin.qpos-v1") -> WorkerContext:
    return WorkerContext(
        worker_id=0,
        gpu_slot=0,
        gpu_device="0",
        replica_id=0,
        replica_port=16694,
        replica_socket=tmp_path / "replica.sock",
        queue_dir=tmp_path / "queue",
        state_dir=tmp_path / "state",
        output_dir=tmp_path / "out",
        execution_contract=roboscope_kit.PolicyExecutionContract(
            benchmark="robotwin",
            action_representation=representation,
        ),
    )


def _services(tmp_path: Path, services_per_gpu: int = 2) -> RoboTwinRenderServices:
    script = tmp_path / "fake_service.py"
    script.write_text(FAKE_SERVICE, encoding="utf-8")
    return RoboTwinRenderServices(
        python=Path(sys.executable),
        script=script,
        robotwin_root=tmp_path,
        services_per_gpu=services_per_gpu,
        log_dir=tmp_path / "logs",
        ready_timeout_s=30.0,
    )


def _wait_until(condition, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline
        time.sleep(0.05)


# --------------------------------------------------------------------------- configuration


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"render_mode": "remote"}, "native or central"),
        ({"render_services_per_gpu": 0}, "render_services_per_gpu"),
        ({"render_service_path": None}, "render_service_path"),
        ({"suite": "demo_randomized"}, "demo_clean"),
        ({"save_videos": True}, "no videos"),
        ({"seed_cache_dir": None}, "seed cache"),
    ),
)
def test_central_config_rejects_unsupported_settings(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**overrides)


def test_native_config_needs_none_of_the_central_settings() -> None:
    config = _config(
        render_mode="native",
        render_service_path=None,
        suite="demo_randomized",
        seed_cache_dir=None,
        save_videos=True,
    )

    assert config.render_mode == "native"


# --------------------------------------------------------------------------- Worker process


def test_central_worker_runs_without_a_gpu_and_with_its_service(tmp_path: Path) -> None:
    services = _services(tmp_path)
    adapter = RoboTwinBenchmarkAdapter(
        _config(
            benchmark_python=Path(sys.executable),
            robotwin_root=tmp_path,
            render_service_path=services.script,
        )
    )
    try:
        spec = adapter.worker_process(_context(tmp_path))
        socket = Path(spec.command[spec.command.index("--render-service") + 1])
        assert socket.exists()
        assert adapter.metadata()["render_mode"] == "central"
    finally:
        adapter.close()

    assert not socket.exists()
    assert _gpu_env(SimpleNamespace(gpu_id="0", render_service=socket)) == {
        "CUDA_VISIBLE_DEVICES": "",
        "ROBOTWIN_RENDER_SERVICE": str(socket),
    }
    assert _gpu_env(SimpleNamespace(gpu_id="3")) == {"CUDA_VISIBLE_DEVICES": "3"}


def test_central_worker_rejects_ee_actions(tmp_path: Path) -> None:
    adapter = RoboTwinBenchmarkAdapter(_config())

    with pytest.raises(ValueError, match="qpos"):
        adapter.worker_process(_context(tmp_path, representation="robotwin.ee-compact-quat-v1"))
    assert adapter.render_services is None


def test_native_worker_starts_no_service(tmp_path: Path) -> None:
    adapter = RoboTwinBenchmarkAdapter(_config(render_mode="native", render_service_path=None))

    spec = adapter.worker_process(_context(tmp_path))

    assert "--render-service" not in spec.command
    assert adapter.render_services is None
    adapter.close()


# --------------------------------------------------------------------------- render services


def test_services_spread_a_gpus_workers_and_keep_a_restarted_workers_service(tmp_path: Path) -> None:
    services = _services(tmp_path)
    gpu = "GPU-01234567-89ab-cdef-0123-456789abcdef/MIG-1"
    try:
        sockets = [services.socket_for(gpu, 0, worker_id) for worker_id in range(3)]
        other_gpu = services.socket_for("1", 1, 3)

        assert [socket.name for socket in sockets] == ["gpu0-0.sock", "gpu0-1.sock", "gpu0-0.sock"]
        assert other_gpu.name == "gpu1-0.sock"
        assert services.socket_for(gpu, 0, 1) == sockets[1]
        assert len(services.processes) == 3
        log = (tmp_path / "logs" / "gpu0-service-0-attempt-000.log").read_text(encoding="utf-8")
        assert f"CUDA_VISIBLE_DEVICES={gpu}" in log
    finally:
        services.close()

    assert services.processes == {}
    assert services.socket_dir is None
    assert not sockets[0].parent.exists()


def test_a_worker_restarted_after_its_service_exited_gets_a_new_service(tmp_path: Path) -> None:
    services = _services(tmp_path)
    try:
        socket = services.socket_for("0", 0, 0)
        first = services.processes[(0, 0)]
        os.kill(first.pid, signal.SIGKILL)
        first.wait(timeout=10)

        assert services.socket_for("0", 0, 0) == socket
        second = services.processes[(0, 0)]
        assert second.pid != first.pid and second.poll() is None
        assert (tmp_path / "logs" / "gpu0-service-0-attempt-001.log").exists()

        # A service that removed its socket is shutting down and is replaced as well.
        socket.unlink()
        services.socket_for("0", 0, 0)
        assert services.processes[(0, 0)].pid != second.pid
        assert second.poll() is not None
    finally:
        services.close()


def test_a_service_that_never_binds_fails_its_worker_start(tmp_path: Path) -> None:
    services = _services(tmp_path)
    services.script.write_text("print('no renderer')\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="did not start:\nno renderer"):
        services.socket_for("0", 0, 0)
    services.close()


# --------------------------------------------------------------------------- lost service


def _lost_service_item() -> WorkItem:
    return WorkItem(
        ordinal=0,
        item_id="demo_clean-adjust_bottle-episodes-0000-0002",
        group_id="demo_clean-adjust_bottle-episodes-0000-0002",
        payload={
            "task": "adjust_bottle",
            "suite": "demo_clean",
            "seed": 0,
            "episode_start": 0,
            "episodes": 2,
            "total_episodes": 2,
        },
        expected_results=2,
    )


def _launcher_args(tmp_path: Path, launcher_source: str) -> SimpleNamespace:
    launcher = tmp_path / "launcher.py"
    launcher.write_text(launcher_source, encoding="utf-8")
    deploy_policy = tmp_path / "deploy_policy.yml"
    deploy_policy.write_text("policy_name: model2robotwin_interface\n", encoding="utf-8")
    return SimpleNamespace(
        worker_id=0,
        gpu_id="0",
        policy_socket=tmp_path / "policy.sock",
        robotwin_dir=tmp_path,
        robotwin_python=Path(sys.executable),
        checkpoint_path=tmp_path / "checkpoint",
        deploy_policy=deploy_policy,
        launcher=launcher,
        queue_dir=tmp_path / "queue",
        output_dir=tmp_path / "out",
        disable_video=True,
        seed_cache_dir=None,
        render_service=tmp_path / "render.sock",
    )


def test_a_lost_render_service_ends_the_worker_instead_of_failing_episodes(
    tmp_path: Path, render_module
) -> None:
    args = _launcher_args(
        tmp_path,
        "import json\n"
        f"print({render_module.SERVICE_LOST_MARKER!r}, json.dumps({{'detail': 'connection closed'}}))\n"
        "raise SystemExit(1)\n",
    )

    with pytest.raises(RuntimeError, match="Render service lost:\n" + RENDER_SERVICE_LOST_PREFIX):
        _run_task_single_process(args, _lost_service_item())


def test_an_ordinary_evaluator_failure_still_records_failed_episodes(tmp_path: Path) -> None:
    args = _launcher_args(tmp_path, "raise SystemExit(1)\n")

    result = _run_task_single_process(args, _lost_service_item())

    assert [record.success for record in result.records] == [False, False]


def test_the_client_reports_a_missing_service_on_stdout(tmp_path: Path, render_module, capsys) -> None:
    with pytest.raises(render_module.RenderServiceLost, match="cannot connect"):
        render_module.RenderClient(str(tmp_path / "missing.sock"), timeout_s=0.2)

    assert capsys.readouterr().out.startswith(RENDER_SERVICE_LOST_PREFIX + " ")


# --------------------------------------------------------------------------- runtime close hook


class _ClosingBenchmark:
    def __init__(self) -> None:
        self.closed = 0

    @staticmethod
    def work_items():
        return [WorkItem(ordinal=0, item_id="item", group_id="item", payload={}, expected_results=1)]

    @staticmethod
    def metadata():
        return {}

    def close(self) -> None:
        self.closed += 1


def test_the_runtime_closes_the_benchmark_when_a_pass_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Sampler:
        def __init__(self, **_kwargs) -> None:
            pass

        def start(self) -> None:
            pass

        @staticmethod
        def stop() -> list:
            return []

    monkeypatch.setattr(orchestrator, "GpuSampler", Sampler)
    runtime = ParallelBenchmarkRuntime.__new__(ParallelBenchmarkRuntime)
    runtime.config = RuntimeConfig(output_dir=tmp_path, gpu_devices=("0",), replicas=1, concurrency=1)
    runtime.benchmark = _ClosingBenchmark()
    runtime.state_dir = tmp_path / "state"
    runtime.queue = SimpleNamespace(initialize=lambda items: None, request_stop=lambda: None)
    runtime.server_pool = SimpleNamespace(
        contexts={},
        scale=lambda replicas, ready_timeout_s=None: None,
        ensure_healthy=lambda ready_timeout_s=None: [],
        maximum_workers_per_replica=lambda: None,
    )
    contract = SimpleNamespace(to_dict=lambda: {})
    runtime.inference = SimpleNamespace(execution_contract=lambda: contract, metadata=lambda: {})
    runtime._select_shard = lambda items: None
    runtime._clear_worker_state = lambda: None
    runtime._inference_metric_offsets = lambda: {}
    runtime._stop_workers = lambda: None
    runtime._stop_owned_server_pool = lambda: None

    def fail_to_start(count: int) -> None:
        raise RuntimeError("worker start failed")

    runtime._start_workers = fail_to_start

    with pytest.raises(RuntimeError, match="worker start failed"):
        runtime.run_workers()
    assert runtime.benchmark.closed == 1


# --------------------------------------------------------------------------- recorded scene data


def test_recorders_keep_construction_writes_and_calls_in_order(render_module) -> None:
    material = render_module.Recorder(base_color=[1, 0, 0, 1])
    material.roughness = 0.5
    material.set_metallic(0.1)
    body = render_module.Recorder()
    body.attach(material)

    assert material._ops == [
        ("init", [], {"base_color": [1, 0, 0, 1]}),
        ("set", "roughness", 0.5),
        ("call", "set_metallic", [0.1], {}),
    ]
    assert material.roughness == 0.5
    assert material.get_metallic() == 0.1
    assert body.render_shapes == [material]


def test_recorded_values_round_trip_through_the_service_decoder(
    render_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Pose:
        def __init__(self, p=(0.0, 0.0, 0.0), q=(1.0, 0.0, 0.0, 0.0)) -> None:
            self.p, self.q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)

    built = []

    class RenderMaterial:
        def __init__(self, **kwargs) -> None:
            self.kwargs, self.calls = kwargs, []
            built.append(self)

        def set_metallic(self, value) -> None:
            self.calls.append(value)

    sapien = types.ModuleType("sapien")
    sapien.Pose = Pose
    sapien.render = SimpleNamespace(RenderMaterial=RenderMaterial)
    monkeypatch.setitem(sys.modules, "sapien", sapien)
    value = {
        "pose": Pose((1, 2, 3), (0, 1, 0, 0)),
        "array": np.arange(3, dtype=np.int32),
        "tuple": (1, "a"),
        "list": [np.float32(0.5), None, True],
    }
    realizer = render_module.Realizer()

    decoded = realizer.decode(render_module.encode(value))

    assert decoded["pose"].p.tolist() == [1, 2, 3] and decoded["pose"].q.tolist() == [0, 1, 0, 0]
    assert decoded["array"].dtype == np.int32 and decoded["array"].tolist() == [0, 1, 2]
    assert decoded["tuple"] == (1, "a") and decoded["list"] == [0.5, None, True]

    # Workers record with subclasses named after the sapien.render classes they replace.
    material = type("RenderMaterial", (render_module.Recorder,), {})(base_color=[1, 0, 0, 1])
    material.set_metallic(0.1)
    node = render_module.encode(material)
    first = realizer.decode(node)
    assert first.kwargs == {"base_color": [1, 0, 0, 1]} and first.calls == [0.1]
    assert realizer.decode(node) is first  # identical materials are built once
    realizer.cache.clear()
    assert realizer.decode(node) is not first
    with pytest.raises(TypeError, match="cannot record"):
        render_module.encode(object())


def test_state_digest_covers_entity_and_camera_poses(render_module) -> None:
    bodies = np.zeros((2, 7), dtype=np.float32)
    cameras = np.ones((3, 7), dtype=np.float32)
    digest = render_module.state_digest(bodies, cameras)

    assert render_module.state_digest(bodies.copy(), cameras.copy()) == digest
    moved = bodies.copy()
    moved[1, 2] = 1e-3
    assert render_module.state_digest(moved, cameras) != digest
    assert render_module.state_digest(bodies, cameras[:2]) != digest
