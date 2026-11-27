# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from roboscope_kit import profiling
from roboscope_kit.runtime.benchmarks.libero import camera as libero_camera
from roboscope_kit.runtime.benchmarks.libero_pro.adapter import (
    LiberoProBenchmarkAdapter,
    LiberoProRuntimeConfig,
)


def _worker(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    # The worker saves videos with imageio, which only the LIBERO simulator environment installs.
    if importlib.util.find_spec("imageio") is None:
        monkeypatch.setitem(sys.modules, "imageio", types.ModuleType("imageio"))
    return importlib.import_module("roboscope_kit.runtime.benchmarks.libero.worker")


def _make_observable(initial_value: int = 0, **overrides: object) -> SimpleNamespace:
    calls = 0

    def sensor(_obs_cache: dict) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.full((4, 4, 3), initial_value + calls, dtype=np.uint8)

    attrs = {
        "_active": True,
        "_sensor": sensor,
        "_corruptor": None,
        "_filter": None,
        "_delay": 0,
        "modality": "image",
    }
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


class _FakeEnv:
    """Records each step's action, delta mode, and simulated camera observable sampling."""

    def __init__(self, observables: dict[str, SimpleNamespace] | None = None) -> None:
        self.controller = SimpleNamespace(use_delta=True)
        obs_map = observables or {
            "agentview_image": _make_observable(10),
            "robot0_eye_in_hand_image": _make_observable(20),
            "robot0_eef_pos": SimpleNamespace(_active=True, modality="robot0_proprio-state"),
        }
        self.env = SimpleNamespace(
            robots=[SimpleNamespace(controller=self.controller)],
            done=False,
            _observables=obs_map,
        )
        self.steps: list[tuple[tuple[float, ...], bool]] = []

    def _sample_obs(self) -> dict[str, np.ndarray]:
        return {
            name: obs._sensor({})
            for name, obs in self.env._observables.items()
            if hasattr(obs, "_sensor")
        }

    def reset(self) -> None:
        pass

    def set_init_state(self, _state: np.ndarray) -> dict[str, np.ndarray]:
        return self._sample_obs()

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, dict]:
        self.steps.append((tuple(float(value) for value in action), self.controller.use_delta))
        return self._sample_obs(), 0.0, False, {}


class _FakePolicy:
    def __init__(self, chunk_size: int = 1) -> None:
        self.chunk_size = chunk_size
        self.actions = np.empty((0, 7), dtype=np.float32)
        self.observed_primary: list[int] = []

    def reset(self, _context: object) -> None:
        self.actions = np.empty((0, 7), dtype=np.float32)

    def step(self, observation: dict, _task_description: str, _step: int) -> np.ndarray:
        if not self.actions.size:
            if "agentview_image" in observation:
                self.observed_primary.append(int(observation["agentview_image"][0, 0, 0]))
            self.actions = np.ones((self.chunk_size, 7), dtype=np.float32)
        action = self.actions[0]
        self.actions = self.actions[1:]
        return action

    def end_episode(self) -> None:
        self.actions = np.empty((0, 7), dtype=np.float32)


@pytest.mark.parametrize("absolute_actions", (False, True))
def test_absolute_actions_switch_the_controller_after_settling(
    monkeypatch: pytest.MonkeyPatch,
    absolute_actions: bool,
) -> None:
    worker = _worker(monkeypatch)
    env = _FakeEnv()
    runtime = worker.TaskRuntime(task_id=0, task_description="task", initial_states=np.zeros((2, 1)), env=env)

    # Two consecutive episodes on one environment, as a worker runs them.
    for episode_index in range(2):
        success, steps, _elapsed_s, _frames = worker.run_episode(
            runtime,
            _FakePolicy(),
            episode_index,
            max_steps=4,
            settle_steps=3,
            save_frames=False,
            absolute_actions=absolute_actions,
        )
        assert (success, steps) == (False, 4)

    dummy = tuple(float(value) for value in worker.LIBERO_DUMMY_ACTION)
    episode = [(dummy, True)] * 3 + [((1.0,) * 7, not absolute_actions)] * 4
    assert env.steps == episode * 2


def test_chunk_camera_gate_skips_intermediate_renders_and_restores_sensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _worker(monkeypatch)
    monkeypatch.setattr(libero_camera, "_verify_robosuite_version", lambda: None)
    env = _FakeEnv()
    originals = {
        name: env.env._observables[name]._sensor
        for name in libero_camera.CAMERA_OBSERVATIONS
    }
    runtime = worker.TaskRuntime(task_id=0, task_description="task", initial_states=np.zeros((1, 1)), env=env)
    policy = _FakePolicy(chunk_size=4)
    counters = {"camera_renders": 0, "camera_renders_skipped": 0}

    success, steps, _elapsed_s, _frames = worker.run_episode(
        runtime,
        policy,
        0,
        max_steps=8,
        settle_steps=2,
        save_frames=False,
        chunk_camera_rendering=True,
        camera_counters=counters,
    )
    assert (success, steps) == (False, 8)
    # 2 settle steps (1 skipped + 1 rendered) + two 4-step chunks (3 skipped + 1 rendered per chunk),
    # across 2 cameras -> (1 + 6) * 2 = 14 skipped, (1 + 2) * 2 = 6 rendered.
    assert counters == {"camera_renders": 6, "camera_renders_skipped": 14}
    # set_init_state (11) + final settle step (12) -> first chunk sees 12; boundary step renders 13.
    assert policy.observed_primary == [12, 13]
    for name, original in originals.items():
        assert env.env._observables[name]._sensor is original


def test_chunk_camera_gate_validates_robosuite_version_and_observable_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(libero_camera, "_VERSION_VERIFIED", False)
    monkeypatch.setattr(libero_camera, "version", lambda _pkg: "1.4.1")
    env = _FakeEnv()
    obs = env.set_init_state(np.zeros(1))
    gate = libero_camera.ChunkCameraGate()
    with pytest.raises(RuntimeError, match="robosuite 1.4.0"), gate.install(env, obs, enabled=True):
        pass

    monkeypatch.setattr(libero_camera, "version", lambda _pkg: "1.4.0")
    no_corr = lambda x: x  # noqa: E731
    no_flt = lambda x: x  # noqa: E731
    no_dly = lambda: 0.0  # noqa: E731
    fake_rs_obs = types.SimpleNamespace(NO_CORRUPTION=no_corr, NO_FILTER=no_flt, NO_DELAY=no_dly)
    monkeypatch.setitem(sys.modules, "robosuite.utils.observables", fake_rs_obs)
    default_rs_env = _FakeEnv(
        {
            "agentview_image": _make_observable(
                10, _corrupter=no_corr, _filter=no_flt, _delayer=no_dly
            ),
            "robot0_eye_in_hand_image": _make_observable(
                20, _corrupter=no_corr, _filter=no_flt, _delayer=no_dly
            ),
        }
    )
    with gate.install(default_rs_env, obs, enabled=True):
        pass

    corrupted_env = _FakeEnv(
        {
            "agentview_image": _make_observable(10, _corrupter=lambda x: x),
            "robot0_eye_in_hand_image": _make_observable(20),
        }
    )
    with (
        pytest.raises(RuntimeError, match="corruptor"),
        gate.install(corrupted_env, obs, enabled=True),
    ):
        pass

    extra_cam_env = _FakeEnv(
        {
            "agentview_image": _make_observable(10),
            "robot0_eye_in_hand_image": _make_observable(20),
            "frontview_image": _make_observable(30),
        }
    )
    with (
        pytest.raises(RuntimeError, match="unexpected active image observable"),
        gate.install(extra_cam_env, obs, enabled=True),
    ):
        pass


def test_libero_pro_adapter_normalizes_task_order_and_runtime_flags(tmp_path: Path) -> None:
    reversed_tasks = list(reversed(range(10)))
    config = LiberoProRuntimeConfig(
        python=tmp_path / "python",
        libero_home=tmp_path / "libero",
        libero_config_path=tmp_path / "config.yaml",
        suite="libero_goal_object",
        num_tasks=10,
        num_trials=2,
        policy_encoding="msgpack",
        chunk_camera_rendering=True,
        compiled_model_cache=True,
        egl_context_reuse=True,
        task_order=reversed_tasks,  # type: ignore[arg-type] -- manifest JSON decodes lists
    )
    assert config.task_order == tuple(reversed_tasks)
    adapter = LiberoProBenchmarkAdapter(config)
    assert adapter.task_ids == reversed_tasks
    meta = adapter.metadata()
    assert meta["policy_encoding"] == "msgpack"
    assert meta["chunk_camera_rendering"] is True
    assert meta["compiled_model_cache"] is True
    assert meta["egl_context_reuse"] is True
    assert meta["task_order"] == reversed_tasks


def test_unified_profiling_stage_timing_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(profiling, "TIMING_DIR", str(tmp_path))
    monkeypatch.setattr(profiling, "TIMING_ENABLED", True)
    with profiling.capture("worker", item_id="item-0"):
        with profiling.span("worker.action_step"):
            pass
        profiling.add_span("custom.cuda", 0.0125)

    files = list(tmp_path.glob("worker-*.jsonl"))
    assert len(files) == 1
    record = json.loads(files[0].read_text(encoding="utf-8").strip())
    assert record["ok"] is True
    assert record["item_id"] == "item-0"
    assert record["spans"]["worker.action_step"]["count"] == 1
    assert record["spans"]["custom.cuda"] == {"seconds": 0.0125, "count": 1}


def test_libero_pro_manifest_strict_and_fast_defaults_skip_autotune(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from roboscope_kit.runtime.benchmark import cli

    repo_root = Path(__file__).resolve().parents[3]
    repo_manifest = repo_root / "benchmarks" / "libero_pro" / "benchmark-manifest.json"
    manifest_data = json.loads(repo_manifest.read_text(encoding="utf-8"))
    manifest_data["runtime"] = {
        "python": sys.executable,
        "benchmark_root": str(tmp_path),
        "config_path": str(tmp_path / "config.yaml"),
    }
    manifest_path = tmp_path / "benchmark-manifest.json"
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps({"schema_version": 2, "command": [sys.executable], "cwd": str(tmp_path)}),
        encoding="utf-8",
    )

    monkeypatch.setattr(cli, "detect_gpu_devices", lambda: ("0",))
    monkeypatch.setattr(
        cli,
        "_autotuned_run",
        lambda *_a, **_kw: pytest.fail("LIBERO-Pro default fixed K/g should not run autotune"),
    )

    captured: list[dict[str, object]] = []

    def fake_run_candidate(**kwargs: object) -> dict[str, object]:
        adapter = kwargs["factory"].create(kwargs["request"], kwargs["manifest"])  # type: ignore[attr-defined]
        meta = adapter.metadata()
        captured.append(
            {
                "k": kwargs["concurrency"],
                "g": kwargs["replicas"],
                "eval_mode": kwargs["eval_mode"],
                "mps": kwargs["cuda_mps"] is not None,
                "meta": meta,
            }
        )
        return {
            "result": {"successes": 1},
            "benchmark": meta,
            "inference": {
                "execution_contract": {
                    "active_fast_optimizations": (
                        ["dynamic-batching"] if kwargs["eval_mode"] == "fast" else []
                    ),
                }
            },
        }

    class FakeMps:
        def __init__(self, **_kwargs: object) -> None:
            self.started = False

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.started = False

        def metadata(self) -> dict[str, object]:
            return {"enabled": True}

    monkeypatch.setattr(cli, "run_candidate", fake_run_candidate)
    monkeypatch.setattr(cli, "CudaMpsSession", FakeMps)

    strict_out = tmp_path / "strict-run"
    base_args = [
        "--benchmark", "libero_pro",
        "--suite", "libero_goal_object",
        "--benchmark-manifest", str(manifest_path),
        "--policy-config", str(policy_path),
        "--checkpoint-source", str(tmp_path),
        "--run-cache", "off",
    ]
    assert cli.main([*base_args, "--output-dir", str(strict_out)]) == 0
    strict_result = json.loads((strict_out / "result.json").read_text(encoding="utf-8"))
    assert strict_result["autotune"] == {"k": 12, "g": 3, "mode": "fixed"}
    assert strict_result["cuda_mps"] == {"enabled": False}
    assert strict_result["eval_mode"] == "strict"
    assert strict_result["active_fast_optimizations"] == []
    assert captured[-1]["meta"]["policy_encoding"] == "json"  # type: ignore[index]
    assert captured[-1]["meta"]["chunk_camera_rendering"] is False  # type: ignore[index]

    fast_out = tmp_path / "fast-run"
    assert cli.main([*base_args, "--eval-mode", "fast", "--output-dir", str(fast_out)]) == 0
    fast_result = json.loads((fast_out / "result.json").read_text(encoding="utf-8"))
    assert fast_result["autotune"] == {"k": 24, "g": 2, "mode": "fixed"}
    assert fast_result["cuda_mps"] == {"enabled": True}
    assert fast_result["eval_mode"] == "fast"
    assert fast_result["active_fast_optimizations"] == [
        "dynamic-batching",
        "msgpack-transport",
        "chunk-camera-rendering",
        "compiled-model-cache",
        "egl-context-reuse",
    ]


def test_starvla_libero_pro_fast_negotiates_224_and_coalesces_batches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import importlib.util

    from roboscope_kit import ANY_SIZE, get_benchmark_spec
    from roboscope_kit.benchmark_specs import benchmark_camera_support
    from roboscope_kit.policy import PolicyCapabilities, negotiate_policy_action_contract
    from roboscope_kit.runtime.benchmark import WorkerContext

    repo_root = Path(__file__).resolve().parents[3]
    starvla_dir = repo_root / "examples" / "starvla"
    monkeypatch.syspath_prepend(str(starvla_dir))
    if importlib.util.find_spec("cv2") is None:
        cv_stub = types.SimpleNamespace(INTER_AREA=3, resize=lambda a, _s, **_k: a)
        monkeypatch.setitem(sys.modules, "cv2", cv_stub)
    if importlib.util.find_spec("PIL") is None:
        pil_img = types.SimpleNamespace(BILINEAR=2, fromarray=lambda a: a)
        monkeypatch.setitem(sys.modules, "PIL", types.SimpleNamespace(Image=pil_img))
        monkeypatch.setitem(sys.modules, "PIL.Image", pil_img)
    if importlib.util.find_spec("yaml") is None:
        monkeypatch.setitem(sys.modules, "yaml", types.SimpleNamespace(safe_load=lambda *_a, **_k: {}))

    spec_obj = importlib.util.spec_from_file_location("starvla_policy", starvla_dir / "starvla_policy.py")
    assert spec_obj is not None and spec_obj.loader is not None
    starvla_policy = importlib.util.module_from_spec(spec_obj)
    monkeypatch.setitem(sys.modules, "starvla_policy", starvla_policy)
    spec_obj.loader.exec_module(starvla_policy)

    class Qwenvl_OFT:  # noqa: N801
        pass

    policy_inst = starvla_policy.StarVLAPolicy()
    cfg = starvla_policy.InferenceConfig(
        dtype="bfloat16",
        unnorm_key="franka",
        inputs=starvla_policy.InputConfig(
            images=("images.primary", "images.wrist"),
            language="language.instruction",
            state=None,
        ),
        kwargs={},
        transforms=(),
        action_dim=7,
        action_horizon=8,
        max_batch_size=1,
    )

    # Strict mode: canonical 256x256 (image_sizes=None), max_batch_size=1, no coalescing.
    assert starvla_policy._resolve_image_sizes("libero_pro", "strict") is None
    assert starvla_policy._resolve_max_batch_size(
        policy_inst, "libero_pro", "strict", Qwenvl_OFT(), cfg
    ) == (1, False)

    # Fast mode: 224x224 negotiated sizes + max_batch_size=4 with coalesce_action_requests=True.
    fast_sizes = starvla_policy._resolve_image_sizes("libero_pro", "fast")
    assert fast_sizes == ((224, 224), ANY_SIZE)
    assert starvla_policy._resolve_image_sizes("calvin", "fast") is None
    assert starvla_policy._resolve_max_batch_size(
        policy_inst, "libero_pro", "fast", Qwenvl_OFT(), cfg
    ) == (4, True)

    # Zero-copy fast path when Worker already delivers 224x224 native frames.
    native_frame = np.zeros((224, 224, 3), dtype=np.uint8)
    assert starvla_policy._resize_image_with_pil(native_frame) is native_frame
    assert starvla_policy._resize_image_with_opencv(native_frame) is native_frame

    # End-to-end contract negotiation and LIBERO-Pro worker CLI propagation.
    bench_spec = get_benchmark_spec("libero_pro", "libero_goal_object")
    support = benchmark_camera_support(bench_spec, {"render_backend": "egl"})
    caps = PolicyCapabilities(
        supported_benchmarks=("libero_pro",),
        required_features=("images.primary", "images.wrist", "language.instruction"),
        supported_suites={"libero_pro": ("libero_goal_object",)},
        max_batch_size=8,
        coalesce_action_requests=True,
        supported_action_chunk_sizes=(8,),
        observation_strides={"libero_pro": 8},
        image_sizes=fast_sizes,
        fast_optimizations=("qwen-oft-logits-to-keep", "qwen-oft-vision-metadata"),
    )
    strict_contract = negotiate_policy_action_contract(
        bench_spec,
        PolicyCapabilities(
            supported_benchmarks=("libero_pro",),
            required_features=("images.primary", "images.wrist", "language.instruction"),
            max_batch_size=8,
            coalesce_action_requests=True,
            supported_action_chunk_sizes=(8,),
        ),
        eval_mode="strict",
        camera_support=support,
    )
    assert strict_contract.canonical_observations is True
    assert strict_contract.active_fast_optimizations == ()

    fast_contract = negotiate_policy_action_contract(
        bench_spec, caps, eval_mode="fast", camera_support=support
    )
    assert fast_contract.image_sizes == {
        "images.primary": (224, 224),
        "images.wrist": (224, 224),
    }
    assert fast_contract.canonical_observations is False
    assert fast_contract.active_fast_optimizations == (
        "dynamic-batching",
        "negotiated-image-sizes",
        "qwen-oft-logits-to-keep",
        "qwen-oft-vision-metadata",
    )

    adapter = LiberoProBenchmarkAdapter(
        LiberoProRuntimeConfig(
            python=tmp_path / "python",
            libero_home=tmp_path / "libero",
            libero_config_path=tmp_path / "config.yaml",
            suite="libero_goal_object",
            num_tasks=10,
            num_trials=1,
            policy_encoding="msgpack",
            chunk_camera_rendering=True,
            compiled_model_cache=True,
            egl_context_reuse=True,
        )
    )
    worker_spec = adapter.worker_process(
        WorkerContext(
            worker_id=0,
            gpu_device="0",
            gpu_slot=0,
            replica_id=0,
            replica_socket=tmp_path / "policy.sock",
            replica_port=None,
            queue_dir=tmp_path / "queue",
            state_dir=tmp_path / "state",
            output_dir=tmp_path / "out",
            execution_contract=fast_contract,
        )
    )
    assert "--primary-size" in worker_spec.command
    idx_p = worker_spec.command.index("--primary-size")
    idx_w = worker_spec.command.index("--wrist-size")
    assert worker_spec.command[idx_p + 1] == "224x224"
    assert worker_spec.command[idx_w + 1] == "224x224"


def test_policy_action_queue_reuses_persistent_client_and_reconnects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from roboscope_kit.policy import ActionChunk, EpisodeContext

    worker = _worker(monkeypatch)
    created_clients: list[object] = []
    fail_next_act = False

    class FakeClient:
        def __init__(self, socket_path: Path) -> None:
            self.socket_path = socket_path
            self.encoding = "msgpack"
            self.connected = False
            self.closed = False
            self.calls: list[str] = []
            created_clients.append(self)

        def connect(self) -> None:
            self.connected = True

        def close(self) -> None:
            self.closed = True

        def reset(self, context: EpisodeContext) -> None:
            self.calls.append(f"reset:{context.episode_id}")

        def act(self, _obs: object) -> ActionChunk:
            nonlocal fail_next_act
            if fail_next_act:
                fail_next_act = False
                raise BrokenPipeError("simulated socket drop")
            self.calls.append("act")
            return ActionChunk(actions=np.ones((2, 7), dtype=np.float32))

        def end_episode(self, episode_id: str) -> None:
            self.calls.append(f"end:{episode_id}")

    monkeypatch.setattr(worker, "PolicyClient", FakeClient)
    obs = {
        "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.zeros(4, dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
        "robot0_joint_pos": np.zeros(7, dtype=np.float32),
    }

    queue = worker.PolicyActionQueue(tmp_path / "policy.sock")
    queue.reset(EpisodeContext(episode_id="ep-0", task_id="0"))
    assert queue.step(obs, "pick bowl", 0).shape == (7,)
    assert queue.step(obs, "pick bowl", 1).shape == (7,)
    queue.end_episode()
    assert len(created_clients) == 1
    assert created_clients[0].calls == ["reset:ep-0", "act", "end:ep-0"]

    # Simulate transient socket disconnect on next episode step -> transparent reconnect.
    queue.reset(EpisodeContext(episode_id="ep-1", task_id="0"))
    fail_next_act = True
    assert queue.step(obs, "pick bowl", 0).shape == (7,)
    assert len(created_clients) == 2
    assert created_clients[0].closed is True
    assert created_clients[1].connected is True

    queue.close()
    assert created_clients[1].closed is True


def test_policy_server_health_is_lock_free_under_active_request(tmp_path: Path) -> None:
    import threading

    from roboscope_kit.benchmark_specs import get_benchmark_spec
    from roboscope_kit.client import PolicyClient
    from roboscope_kit.policy import (
        ActionChunk,
        EpisodeContext,
        Observation,
        Policy,
        PolicyCapabilities,
        SessionContext,
    )
    from roboscope_kit.server import PolicyServer

    entered_act = threading.Event()
    release_act = threading.Event()

    class BlockingPolicy(Policy):
        def load(self, context: SessionContext) -> PolicyCapabilities:
            return PolicyCapabilities(
                supported_benchmarks=(context.benchmark.name,),
                required_features=("language.instruction",),
                supported_action_chunk_sizes=(1,),
            )

        def reset(self, _context: EpisodeContext) -> None:
            return None

        def act(self, _observation: Observation) -> ActionChunk:
            entered_act.set()
            assert release_act.wait(timeout=5.0)
            return ActionChunk(actions=np.zeros((1, 7), dtype=np.float32))

    sock = tmp_path / "health.sock"
    server = PolicyServer(BlockingPolicy(), sock)
    server_thread = threading.Thread(target=server.serve, daemon=True)
    server_thread.start()
    try:
        spec = get_benchmark_spec("libero_pro", "libero_goal_object")
        init_client = PolicyClient(sock, timeout_s=5.0)
        try:
            init_client.wait_until_ready(timeout_s=3.0)
            init_client.load(SessionContext(session_id="s", benchmark=spec, checkpoint_path=str(tmp_path)))
        finally:
            init_client.close()

        act_errors: list[BaseException] = []

        def _run_blocking_act() -> None:
            try:
                with PolicyClient(sock, timeout_s=5.0) as act_client:
                    act_client.act(
                        Observation(
                            episode_id="ep-0",
                            step=0,
                            timestamp_s=0.0,
                            features={
                                "images.primary": np.zeros((256, 256, 3), dtype=np.uint8),
                                "images.wrist": np.zeros((256, 256, 3), dtype=np.uint8),
                                "state.proprioception": np.zeros(16, dtype=np.float32),
                                "language.instruction": "open drawer",
                            },
                        )
                    )
            except BaseException as exc:  # pragma: no cover - surfaced by assertion below
                act_errors.append(exc)

        act_thread = threading.Thread(target=_run_blocking_act, daemon=True)
        act_thread.start()
        assert entered_act.wait(timeout=3.0)

        with PolicyClient(sock, timeout_s=0.5) as health_client:
            assert health_client.health()["state"] == "ready"

        release_act.set()
        act_thread.join(timeout=3.0)
        assert not act_errors
        with PolicyClient(sock, timeout_s=2.0) as shutdown_client:
            shutdown_client.unload()
    finally:
        release_act.set()
        server_thread.join(timeout=3.0)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("chunk_size", (1, 2))
def test_policy_action_queue_preserves_action_dtype_through_env_step(
    monkeypatch: pytest.MonkeyPatch,
    dtype: type[np.floating],
    chunk_size: int,
) -> None:
    worker = _worker(monkeypatch)
    raw = np.array(
        [
            [0.1234567890123456, -0.9876543210987654, 0.3333333333333333, 0.1, -0.2, 0.3, -1.0],
            [-0.4567890123456789, 0.6543210987654321, -0.7777777777777777, -0.1, 0.2, -0.3, 1.0],
        ],
        dtype=dtype,
    )
    chunk_value = raw[0] if chunk_size == 1 else raw[:chunk_size]
    expected_actions = [raw[0], raw[0]] if chunk_size == 1 else [raw[0], raw[1]]

    class _FakeClient:
        def __init__(self, _socket_path: object) -> None:
            pass

        def __enter__(self) -> _FakeClient:
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def reset(self, _context: object) -> None:
            pass

        def act(self, _observation: object) -> SimpleNamespace:
            return SimpleNamespace(actions=chunk_value.copy())

        def end_episode(self, _episode_id: str) -> None:
            pass

    monkeypatch.setattr(worker, "PolicyClient", _FakeClient)

    received: list[np.ndarray] = []

    class _RecordingEnv:
        def __init__(self) -> None:
            self.env = SimpleNamespace(robots=[], done=False)

        def reset(self) -> None:
            pass

        def set_init_state(self, _state: np.ndarray) -> dict:
            return {
                "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
                "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
                "robot0_eef_pos": np.zeros(3, dtype=np.float64),
                "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
                "robot0_gripper_qpos": np.zeros(2, dtype=np.float64),
                "robot0_joint_pos": np.zeros(7, dtype=np.float64),
            }

        def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
            received.append(action.copy())
            return self.set_init_state(np.zeros(1)), 0.0, False, {}

    env = _RecordingEnv()
    runtime = worker.TaskRuntime(task_id=0, task_description="task", initial_states=np.zeros((1, 1)), env=env)
    queue = worker.PolicyActionQueue(socket_path=worker.Path("/unused.sock"))

    success, steps, _elapsed_s, _frames = worker.run_episode(
        runtime,
        queue,
        0,
        max_steps=2,
        settle_steps=1,
        save_frames=False,
    )
    assert (success, steps) == (False, 2)
    policy_actions = received[1:]
    assert len(policy_actions) == 2
    for actual, expected in zip(policy_actions, expected_actions, strict=True):
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()
