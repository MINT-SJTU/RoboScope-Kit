# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Unit-level checks for the unified benchmark adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import roboscope_kit
from roboscope_kit import profiling
from roboscope_kit.benchmark_specs import BenchmarkName
from roboscope_kit.policy import PolicyCapabilities
from roboscope_kit.runtime.benchmark.factories import BenchmarkRunRequest
from roboscope_kit.runtime.benchmark.manifest import BenchmarkManifest
from roboscope_kit.runtime.benchmark.registry import benchmark_factory
from roboscope_kit.runtime.benchmark.schema import EvaluationRecord, WorkResult
from roboscope_kit.runtime.benchmarks.calvin import CalvinBenchmarkFactory
from roboscope_kit.runtime.benchmarks.libero import LiberoBenchmarkFactory
from roboscope_kit.runtime.benchmarks.libero_plus import LiberoPlusBenchmarkFactory
from roboscope_kit.runtime.benchmarks.robotwin import (
    ROBOTWIN_ALL_TASKS,
    RoboTwinBenchmarkAdapter,
    RoboTwinBenchmarkFactory,
    RoboTwinRuntimeConfig,
)

LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
LIBERO_PLUS_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

PUBLIC_SDK_API = [
    "ActionChunk",
    "ActionSpec",
    "ANY_SIZE",
    "BenchmarkName",
    "BenchmarkSpec",
    "EpisodeContext",
    "FeatureKind",
    "FeatureSpec",
    "Observation",
    "Policy",
    "POLICY_CONFIG_SCHEMA_VERSION",
    "PolicyConfigError",
    "PolicyRuntimeConfig",
    "PolicyCapabilities",
    "PolicyExecutionContract",
    "PolicyClient",
    "PolicyRemoteError",
    "PolicyServer",
    "PolicyState",
    "SessionContext",
    "benchmark_names",
    "get_benchmark_spec",
    "load_policy_runtime_config",
    "negotiate_policy_action_contract",
    "run",
    "validate_action_chunk",
    "validate_observation",
    "validate_policy_capabilities",
]


def test_public_sdk_api_matches_supported_exports() -> None:
    assert roboscope_kit.__all__ == PUBLIC_SDK_API
    assert all(hasattr(roboscope_kit, name) for name in PUBLIC_SDK_API)
    assert "runtime" not in PUBLIC_SDK_API
    assert "__version__" not in PUBLIC_SDK_API


def test_policy_episode_group_preserves_lifecycle_and_nested_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    events = []
    monkeypatch.setattr(profiling, "ENABLED", True)
    monkeypatch.setattr(profiling, "episode_start", lambda: events.append("start"))
    monkeypatch.setattr(profiling, "episode_end", lambda: events.append("end"))
    client = roboscope_kit.PolicyClient("/unused-policy-socket")
    monkeypatch.setattr(client, "_request", lambda operation, payload: events.append((operation, payload)))
    first = roboscope_kit.EpisodeContext("slot-0", "task")
    second = roboscope_kit.EpisodeContext("slot-1", "task")
    with client.episode_group():
        client.reset(first)
        with client.episode_group():
            client.reset(second)
            client.end_episode(second.episode_id)
        client.end_episode(first.episode_id)
        assert events == [
            "start", ("reset", {"context": first.to_dict()}), ("reset", {"context": second.to_dict()}),
            ("end_episode", {"episode_id": second.episode_id}),
            ("end_episode", {"episode_id": first.episode_id}),
        ]
    assert events[-1] == "end"
    assert events.count("start") == events.count("end") == 1


def test_policy_episode_group_closes_capture_after_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    events = []
    monkeypatch.setattr(profiling, "ENABLED", True)
    monkeypatch.setattr(profiling, "episode_start", lambda: events.append("start"))
    monkeypatch.setattr(profiling, "episode_end", lambda: events.append("end"))
    client = roboscope_kit.PolicyClient("/unused-policy-socket")
    monkeypatch.setattr(client, "_request", lambda operation, payload: events.append(operation))
    with (
        pytest.raises(RuntimeError, match="inference failed"),
        client.episode_group(),
        client.episode_group(),
    ):
        raise RuntimeError("inference failed")
    assert events == ["start", "end"]
    context = roboscope_kit.EpisodeContext("next-episode", "task")
    client.reset(context)
    client.end_episode(context.episode_id)
    assert events == ["start", "end", "start", "reset", "end_episode", "end"]


def test_policy_episode_group_without_profiling_keeps_normal_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    events = []
    monkeypatch.setattr(profiling, "ENABLED", False)
    monkeypatch.setattr(profiling, "episode_start", lambda: pytest.fail("unexpected capture"))
    monkeypatch.setattr(profiling, "episode_end", lambda: pytest.fail("unexpected capture"))
    client = roboscope_kit.PolicyClient("/unused-policy-socket")
    monkeypatch.setattr(client, "_request", lambda operation, payload: events.append(operation))
    context = roboscope_kit.EpisodeContext("slot-0", "task")
    with client.episode_group():
        client.reset(context)
        client.end_episode(context.episode_id)
    assert events == ["reset", "end_episode"]


def test_legacy_policy_capabilities_use_benchmark_action_default() -> None:
    capabilities = PolicyCapabilities.from_dict(
        {
            "supported_benchmarks": ["robotwin"],
            "required_features": [],
        }
    )
    contract = roboscope_kit.negotiate_policy_action_contract(
        roboscope_kit.get_benchmark_spec(BenchmarkName.ROBOTWIN, "demo_clean"),
        capabilities,
    )

    assert capabilities.action_representations == {}
    assert contract.action_representation == "robotwin.qpos-v1"
    assert contract.action_representation_source == "benchmark_default"


@pytest.mark.parametrize("benchmark_name", (BenchmarkName.LIBERO, BenchmarkName.LIBERO_PLUS))
def test_libero_absolute_action_representation_is_negotiated(benchmark_name: BenchmarkName) -> None:
    benchmark = roboscope_kit.get_benchmark_spec(benchmark_name, "libero_10")
    capabilities = PolicyCapabilities(
        supported_benchmarks=(benchmark_name.value,),
        required_features=(),
        action_representations={benchmark_name.value: "libero.absolute-eef-v1"},
    )
    contract = roboscope_kit.negotiate_policy_action_contract(benchmark, capabilities)

    assert contract.action_representation == "libero.absolute-eef-v1"
    assert contract.action_representation_source == "policy"


def test_explicit_policy_action_representation_is_validated() -> None:
    benchmark = roboscope_kit.get_benchmark_spec(
        BenchmarkName.ROBOTWIN,
        "demo_clean",
    )
    capabilities = PolicyCapabilities(
        supported_benchmarks=("robotwin",),
        required_features=(),
        action_representations={"robotwin": "robotwin.ee-compact-quat-v1"},
    )
    contract = roboscope_kit.negotiate_policy_action_contract(
        benchmark,
        capabilities,
    )

    assert contract.action_representation == "robotwin.ee-compact-quat-v1"
    assert contract.action_representation_source == "policy"

    unknown = PolicyCapabilities(
        supported_benchmarks=("robotwin",),
        required_features=(),
        action_representations={"robotwin": "robotwin.unknown-v1"},
    )
    with pytest.raises(ValueError, match="unsupported by robotwin"):
        roboscope_kit.negotiate_policy_action_contract(benchmark, unknown)


def test_policy_observation_stride_defaults_and_explicit_values() -> None:
    benchmark = roboscope_kit.get_benchmark_spec(
        BenchmarkName.ROBOTWIN,
        "demo_clean",
    )
    legacy = PolicyCapabilities(
        supported_benchmarks=("robotwin",),
        required_features=(),
    )
    legacy_contract = roboscope_kit.negotiate_policy_action_contract(
        benchmark,
        legacy,
    )
    assert legacy_contract.observation_stride == 1
    assert legacy_contract.observation_stride_source == "safe_default"

    explicit = PolicyCapabilities(
        supported_benchmarks=("robotwin",),
        required_features=(),
        observation_strides={"robotwin": 37},
    )
    explicit_contract = roboscope_kit.negotiate_policy_action_contract(
        benchmark,
        explicit,
    )
    assert explicit_contract.observation_stride == 37
    assert explicit_contract.observation_stride_source == "policy"

    with pytest.raises(ValueError, match="observation strides must be positive"):
        PolicyCapabilities(
            supported_benchmarks=("robotwin",),
            required_features=(),
            observation_strides={"robotwin": 0},
        )


def test_policy_replicas_must_negotiate_one_action_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import roboscope_kit.runtime.benchmark.policy_process as policy_process
    from roboscope_kit.runtime.benchmark.adapters import ReplicaContext
    from roboscope_kit.runtime.benchmark.policy_process import PolicyProcessAdapter

    capabilities = iter(
        (
            PolicyCapabilities(
                supported_benchmarks=("robotwin",),
                required_features=(),
                action_representations={"robotwin": "robotwin.qpos-v1"},
            ),
            PolicyCapabilities(
                supported_benchmarks=("robotwin",),
                required_features=(),
                action_representations={"robotwin": "robotwin.ee-compact-quat-v1"},
            ),
        )
    )

    class FakePolicyClient:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            pass

        def load(self, _session, camera_support=None):
            return next(capabilities)

    monkeypatch.setattr(policy_process, "PolicyClient", FakePolicyClient)
    adapter = PolicyProcessAdapter(
        command=("policy",),
        checkpoint_path=tmp_path / "checkpoint",
        benchmark=BenchmarkName.ROBOTWIN,
        suite="demo_clean",
        benchmark_spec=roboscope_kit.get_benchmark_spec(
            BenchmarkName.ROBOTWIN, "demo_clean"
        ),
    )

    def context(replica_id: int) -> ReplicaContext:
        return ReplicaContext(
            replica_id=replica_id,
            gpu_slot=replica_id,
            gpu_device=str(replica_id),
            port=16694 + replica_id,
            socket_path=tmp_path / f"replica-{replica_id}.sock",
            output_dir=tmp_path,
        )

    adapter.initialize_replica(context(0))
    with pytest.raises(ValueError, match="does not match"):
        adapter.initialize_replica(context(1))


def _manifest(
    runtime: dict,
    defaults: dict,
    benchmarks: tuple[BenchmarkName, ...],
    suites: tuple[str, ...],
) -> BenchmarkManifest:
    return BenchmarkManifest(
        path=Path("/sim/benchmark-manifest.json"),
        supported_benchmarks=benchmarks,
        protocol_versions={name: "test-v1" for name in benchmarks},
        supported_suites={name: suites for name in benchmarks},
        runtime=runtime,
        defaults=defaults,
    )


def _libero_manifest(**defaults) -> BenchmarkManifest:
    runtime = {
        "python": "/usr/bin/python3",
        "benchmark_root": "/sim/libero",
        "config_path": "/sim/libero/config.yaml",
    }
    return _manifest(
        runtime,
        defaults,
        (BenchmarkName.LIBERO, BenchmarkName.LIBERO_PLUS),
        LIBERO_SUITES,
    )


def _request(
    benchmark: BenchmarkName,
    suite: str,
    *,
    episodes: int | None = None,
    quick_test: bool = False,
) -> BenchmarkRunRequest:
    return BenchmarkRunRequest(
        benchmark=benchmark,
        suite=suite,
        checkpoint_path=Path("/models/checkpoint"),
        output_dir=Path("/runs/out"),
        episodes=episodes,
        quick_test=quick_test,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_returns_distinct_factories() -> None:
    assert isinstance(benchmark_factory(BenchmarkName.LIBERO), LiberoBenchmarkFactory)
    assert isinstance(benchmark_factory(BenchmarkName.LIBERO_PLUS), LiberoPlusBenchmarkFactory)
    assert isinstance(benchmark_factory(BenchmarkName.CALVIN), CalvinBenchmarkFactory)
    assert (
        benchmark_factory(BenchmarkName.LIBERO).__class__
        is not benchmark_factory(BenchmarkName.LIBERO_PLUS).__class__
    )


# ---------------------------------------------------------------------------
# LIBERO
# ---------------------------------------------------------------------------


def test_libero_factory_full_run_work_items() -> None:
    factory = LiberoBenchmarkFactory()
    adapter = factory.create(
        _request(BenchmarkName.LIBERO, "libero_10"),
        _libero_manifest(episodes=3),
    )
    items = adapter.work_items()
    assert len(items) == 30  # 10 tasks x 3 episodes, chunk_size=1
    assert sum(item.expected_results for item in items) == 30
    assert [item.ordinal for item in items] == list(range(30))
    metadata = adapter.metadata()
    assert metadata["benchmark"] == "libero"
    assert metadata["bound_hint"] == "gpu_infer"
    assert metadata["task_ids"] == list(range(10))


def test_libero_factory_quick_test_single_episode() -> None:
    factory = LiberoBenchmarkFactory()
    adapter = factory.create(
        _request(BenchmarkName.LIBERO, "libero_spatial", quick_test=True),
        _libero_manifest(episodes=50),
    )
    items = adapter.work_items()
    assert sum(item.expected_results for item in items) == 1
    assert adapter.config.save_videos is False


def test_libero_factory_probe_representative_tasks() -> None:
    factory = LiberoBenchmarkFactory()
    adapter = factory.create_probe(
        _request(BenchmarkName.LIBERO, "libero_goal"),
        _libero_manifest(probe_episodes_per_worker=2),
        concurrency=3,
    )
    assert adapter.task_ids == list(range(10))
    items = adapter.work_items()
    assert sum(item.expected_results for item in items) == 10


def test_libero_factory_rejects_unknown_suite() -> None:
    factory = LiberoBenchmarkFactory()
    with pytest.raises(ValueError, match="does not support suite"):
        factory.create(_request(BenchmarkName.LIBERO, "libero_100"), _libero_manifest())


def test_libero_factory_episodes_override() -> None:
    factory = LiberoBenchmarkFactory()
    adapter = factory.create(
        _request(BenchmarkName.LIBERO, "libero_10", episodes=7),
        _libero_manifest(episodes=50),
    )
    assert adapter.config.num_trials == 7


def test_libero_self_test_reports_missing_paths(tmp_path: Path) -> None:
    factory = LiberoBenchmarkFactory()
    manifest = _libero_manifest()
    manifest.runtime.update(
        benchmark_root=str(tmp_path / "missing-libero"),
        config_path=str(tmp_path / "missing-config.yaml"),
    )
    with pytest.raises(RuntimeError, match="LIBERO runtime paths are missing"):
        factory.self_test(manifest, tmp_path)


def test_libero_aggregate_zero_fills_missing_tasks() -> None:
    factory = LiberoBenchmarkFactory()
    adapter = factory.create(
        _request(BenchmarkName.LIBERO, "libero_10"),
        _libero_manifest(episodes=2),
    )
    result = WorkResult(
        item_id="task-0003-episodes-0000-0002",
        group_id="task-0003",
        worker_id=0,
        records=(
            EvaluationRecord(record_id="task-0003-episode-0000", success=True, steps=10, elapsed_s=1.0),
            EvaluationRecord(record_id="task-0003-episode-0001", success=False, steps=20, elapsed_s=2.0),
        ),
        metadata={"task_id": 3, "task_description": "desc", "inference_wait_s": 0.5},
    )
    output = adapter.aggregate([result])
    assert len(output["tasks"]) == 10
    task3 = output["tasks"][3]
    assert task3["episodes"] == 2
    assert task3["successes"] == 1
    assert task3["success_rate"] == 0.5
    empty = output["tasks"][0]
    assert empty["episodes"] == 0
    assert empty["success_rate"] == 0.0
    metrics = output["runtime_metrics"]
    assert metrics["inference_wait_s"] == 0.5
    assert metrics["episode_elapsed_s"] == 3.0


@pytest.mark.parametrize("representation", ("libero.delta-eef-v1", "libero.absolute-eef-v1"))
def test_libero_worker_process_spec(representation: str) -> None:
    from roboscope_kit.runtime.benchmark.adapters import WorkerContext

    factory = LiberoBenchmarkFactory()
    adapter = factory.create(
        _request(BenchmarkName.LIBERO, "libero_10"),
        _libero_manifest(episodes=1),
    )
    context = WorkerContext(
        worker_id=0,
        gpu_slot=1,
        gpu_device="3",
        replica_id=0,
        replica_port=16694,
        replica_socket=Path("/tmp/replica.sock"),
        queue_dir=Path("/tmp/queue"),
        state_dir=Path("/tmp/state"),
        output_dir=Path("/tmp/out"),
        execution_contract=roboscope_kit.PolicyExecutionContract(
            benchmark="libero",
            action_representation=representation,
        ),
    )
    spec = adapter.worker_process(context)
    assert spec.command[1:3] == ("-m", "roboscope_kit.runtime.benchmarks.libero.worker")
    assert spec.command[spec.command.index("--action-representation") + 1] == representation
    assert spec.env["CUDA_VISIBLE_DEVICES"] == "3"
    assert spec.env["MUJOCO_EGL_DEVICE_ID"] == "3"
    assert spec.env["LIBERO_HOME"] == "/sim/libero"
    assert spec.env["LIBERO_CONFIG_PATH"] == "/tmp/state/libero-config"


# ---------------------------------------------------------------------------
# RoboTwin
# ---------------------------------------------------------------------------


def test_robotwin_work_items_interleave_episode_windows() -> None:
    tasks = ROBOTWIN_ALL_TASKS[:2]
    adapter = RoboTwinBenchmarkAdapter(
        RoboTwinRuntimeConfig(
            sdk_python=Path("/opt/kit/bin/python"),
            benchmark_python=Path("/sim/robotwin/bin/python"),
            robotwin_root=Path("/sim/robotwin"),
            checkpoint_path=Path("/models/checkpoint"),
            deploy_policy_path=Path("/sim/robotwin/deploy_policy.yml"),
            launcher_path=Path("/sim/robotwin/launcher.py"),
            suite="demo_randomized",
            tasks=tasks,
            episodes_per_task=25,
            episode_shard_size=10,
        )
    )

    items = adapter.work_items()
    assert [
        (item.payload["task"], item.payload["episode_start"], item.payload["episodes"])
        for item in items
    ] == [
        (tasks[0], 0, 10),
        (tasks[1], 0, 10),
        (tasks[0], 10, 10),
        (tasks[1], 10, 10),
        (tasks[0], 20, 5),
        (tasks[1], 20, 5),
    ]
    assert [item.ordinal for item in items] == list(range(6))
    assert sum(item.expected_results for item in items) == 50
    assert all(item.payload["total_episodes"] == 25 for item in items)
    assert all(item.group_id == item.item_id for item in items)
    assert len({item.group_id for item in items}) == len(items)


@pytest.mark.parametrize(
    ("representation", "action_type"),
    (
        ("robotwin.qpos-v1", "qpos"),
        ("robotwin.ee-compact-quat-v1", "ee"),
    ),
)
def test_robotwin_worker_routes_negotiated_action_representation(
    representation: str,
    action_type: str,
) -> None:
    from roboscope_kit.runtime.benchmark.adapters import WorkerContext

    adapter = RoboTwinBenchmarkAdapter(
        RoboTwinRuntimeConfig(
            sdk_python=Path("/opt/kit/bin/python"),
            benchmark_python=Path("/sim/robotwin/bin/python"),
            robotwin_root=Path("/sim/robotwin"),
            checkpoint_path=Path("/models/checkpoint"),
            deploy_policy_path=Path("/sim/robotwin/deploy_policy.yml"),
            launcher_path=Path("/sim/robotwin/launcher.py"),
            suite="demo_clean",
            tasks=ROBOTWIN_ALL_TASKS[:1],
            episodes_per_task=1,
            episode_shard_size=1,
        )
    )
    context = WorkerContext(
        worker_id=0,
        gpu_slot=0,
        gpu_device="0",
        replica_id=0,
        replica_port=16694,
        replica_socket=Path("/tmp/replica.sock"),
        queue_dir=Path("/tmp/queue"),
        state_dir=Path("/tmp/state"),
        output_dir=Path("/tmp/out"),
        execution_contract=roboscope_kit.PolicyExecutionContract(
            benchmark="robotwin",
            action_representation=representation,
        ),
    )

    assert adapter.worker_process(context).env["ROBOTWIN_ACTION_TYPE"] == action_type


def test_robotwin_refreshes_observations_at_action_chunk_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.syspath_prepend(str(Path(__file__).parents[3]))
    monkeypatch.setitem(sys.modules, "yaml", object())
    from benchmarks.robotwin.roboscope_robotwin_launcher import ModelClient

    client = ModelClient.__new__(ModelClient)
    client.raw_actions = object()
    client.action_chunk_size = 37
    client.observation_stride = 16

    assert [client.needs_observation_refresh(step) for step in (0, 37, 74)] == [
        True,
        True,
        True,
    ]
    assert client.needs_observation_refresh(32) is True
    assert client.needs_observation_refresh(33) is False


def test_robotwin_policy_cadences_share_one_launcher_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.syspath_prepend(str(Path(__file__).parents[3]))
    monkeypatch.setitem(sys.modules, "yaml", object())
    from benchmarks.robotwin.roboscope_robotwin_launcher import ModelClient

    clients = []
    for stride in (1, 16):
        client = ModelClient.__new__(ModelClient)
        client.raw_actions = object()
        client.action_chunk_size = 37
        client.observation_stride = stride
        clients.append(client)

    assert clients[0].needs_observation_refresh(5) is True
    assert clients[1].needs_observation_refresh(5) is False
    assert all(client.needs_observation_refresh(37) for client in clients)


def _robotwin_generation_fixture(tmp_path: Path):
    import pickle
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git is required to test the pinned RoboTwin runtime commit")
    robotwin_root = tmp_path / "RoboTwin"
    task_config = robotwin_root / "task_config" / "demo_clean.yml"
    task_config.parent.mkdir(parents=True)
    task_config.write_text("task: fixture\n", encoding="utf-8")
    subprocess.run(("git", "init", str(robotwin_root)), check=True, capture_output=True)
    subprocess.run(
        ("git", "-C", str(robotwin_root), "config", "user.email", "fixture@example.com"),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(robotwin_root), "config", "user.name", "Fixture"),
        check=True,
    )
    subprocess.run(("git", "-C", str(robotwin_root), "add", "."), check=True)
    subprocess.run(
        ("git", "-C", str(robotwin_root), "commit", "-m", "fixture"),
        check=True,
        capture_output=True,
    )
    commit = subprocess.run(
        ("git", "-C", str(robotwin_root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source = tmp_path / "source" / "demo_clean"
    source.mkdir(parents=True)
    with (source / "adjust_bottle.pkl").open("wb") as handle:
        pickle.dump(
            [
                {"seed": 10, "episode_info": {"info": {"value": 1}}},
                {"seed": 11, "episode_info": {"info": {"value": 2}}},
            ],
            handle,
        )
    scanner = tmp_path / "scanner.py"
    scanner.write_text("# pinned scanner\n", encoding="utf-8")
    protocol = {
        "protocol_id": "fixture-v1",
        "robotwin_runtime_commit": commit,
        "tasks": ["adjust_bottle"],
        "suites": {
            "demo_clean": {
                "task_config_path": "task_config/demo_clean.yml",
                "seed_selection": {
                    "algorithm": "ascending-first-expert-valid-v1",
                    "candidate_start": 10,
                },
                "episodes_per_task": 2,
                "action_horizon": {"maximum": 100},
                "success_criterion": "fixture",
                "aggregation": "fixture",
            }
        },
    }
    return robotwin_root, source.parent, scanner, protocol


def test_robotwin_generation_is_content_addressed_and_detects_tampering(tmp_path: Path) -> None:
    from roboscope_kit.runtime.benchmarks.robotwin.seed_protocol import (
        create_seed_cache_generation,
        validate_seed_cache_generation,
    )

    robotwin_root, source_cache_dir, scanner, protocol = _robotwin_generation_fixture(tmp_path)
    kwargs = {
        "source_cache_dir": source_cache_dir,
        "generation_root": tmp_path / "generations",
        "suite": "demo_clean",
        "tasks": ("adjust_bottle",),
        "protocol": protocol,
        "robotwin_root": robotwin_root,
        "scanner_path": scanner,
    }
    first = create_seed_cache_generation(**kwargs)
    second = create_seed_cache_generation(**kwargs)
    assert first == second
    evidence = validate_seed_cache_generation(
        generation_dir=first,
        suite="demo_clean",
        tasks=("adjust_bottle",),
        episodes=2,
        protocol=protocol,
        robotwin_root=robotwin_root,
        scanner_path=scanner,
    )
    assert evidence["generation_digest"] == first.name
    assert evidence["total_episodes"] == 2

    cache_path = first / "demo_clean" / "adjust_bottle.pkl"
    cache_path.write_bytes(cache_path.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="content differs"):
        validate_seed_cache_generation(
            generation_dir=first,
            suite="demo_clean",
            tasks=("adjust_bottle",),
            episodes=2,
            protocol=protocol,
            robotwin_root=robotwin_root,
            scanner_path=scanner,
        )


def test_robotwin_generation_requires_pinned_clean_runtime(tmp_path: Path) -> None:
    from roboscope_kit.runtime.benchmarks.robotwin.seed_protocol import (
        create_seed_cache_generation,
    )

    robotwin_root, source_cache_dir, scanner, protocol = _robotwin_generation_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="task order differs"):
        create_seed_cache_generation(
            source_cache_dir=source_cache_dir,
            generation_root=tmp_path / "wrong-tasks",
            suite="demo_clean",
            tasks=("other_task",),
            protocol=protocol,
            robotwin_root=robotwin_root,
            scanner_path=scanner,
        )

    mismatched = {**protocol, "robotwin_runtime_commit": "0" * 40}
    with pytest.raises(RuntimeError, match="checkout differs"):
        create_seed_cache_generation(
            source_cache_dir=source_cache_dir,
            generation_root=tmp_path / "mismatch",
            suite="demo_clean",
            tasks=("adjust_bottle",),
            protocol=mismatched,
            robotwin_root=robotwin_root,
            scanner_path=scanner,
        )

    (robotwin_root / "task_config" / "demo_clean.yml").write_text(
        "task: dirty\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="tracked sources differ"):
        create_seed_cache_generation(
            source_cache_dir=source_cache_dir,
            generation_root=tmp_path / "dirty",
            suite="demo_clean",
            tasks=("adjust_bottle",),
            protocol=protocol,
            robotwin_root=robotwin_root,
            scanner_path=scanner,
        )


def _robotwin_adapter_for_generation(
    robotwin_root: Path,
    seed_cache_dir: Path,
    generation_digest: str,
) -> RoboTwinBenchmarkAdapter:
    return RoboTwinBenchmarkAdapter(
        RoboTwinRuntimeConfig(
            sdk_python=Path("/opt/kit/bin/python"),
            benchmark_python=Path("/sim/robotwin/bin/python"),
            robotwin_root=robotwin_root,
            checkpoint_path=Path("/models/checkpoint"),
            deploy_policy_path=Path("/sim/robotwin/deploy_policy.yml"),
            launcher_path=Path("/sim/robotwin/launcher.py"),
            suite="demo_clean",
            tasks=("adjust_bottle",),
            episodes_per_task=2,
            episode_shard_size=1,
            seed_cache_dir=seed_cache_dir,
            protocol_id="fixture-v1",
            seed_generation_digest=generation_digest,
        )
    )


def test_robotwin_formal_gate_validates_v2_generation(
    tmp_path: Path,
) -> None:
    from roboscope_kit.runtime.benchmarks.robotwin.seed_protocol import (
        create_seed_cache_generation,
    )

    robotwin_root, source_cache_dir, scanner, protocol = _robotwin_generation_fixture(tmp_path)
    generation = create_seed_cache_generation(
        source_cache_dir=source_cache_dir,
        generation_root=tmp_path / "generations-v2",
        suite="demo_clean",
        tasks=("adjust_bottle",),
        protocol=protocol,
        robotwin_root=robotwin_root,
        scanner_path=scanner,
    )
    adapter = _robotwin_adapter_for_generation(robotwin_root, generation, generation.name)
    protocol_manifest = BenchmarkManifest(
        path=tmp_path / "benchmark-manifest.json",
        supported_benchmarks=(BenchmarkName.ROBOTWIN,),
        protocol_versions={BenchmarkName.ROBOTWIN: "robotwin2-v1"},
        supported_suites={BenchmarkName.ROBOTWIN: ("demo_clean",)},
        runtime={"seed_scanner_path": str(scanner)},
        defaults={},
        benchmark_protocols={BenchmarkName.ROBOTWIN: protocol},
    )
    RoboTwinBenchmarkFactory._validate_formal_generation(
        adapter,
        protocol_manifest,
        ("adjust_bottle",),
        2,
    )

    mismatched = _robotwin_adapter_for_generation(robotwin_root, generation, "0" * 64)
    with pytest.raises(RuntimeError, match="differs from benchmark manifest"):
        RoboTwinBenchmarkFactory._validate_formal_generation(
            mismatched,
            protocol_manifest,
            ("adjust_bottle",),
            2,
        )


def test_robotwin_paired_verifier_compares_episode_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    kit_root = Path(__file__).parents[3]
    monkeypatch.syspath_prepend(str(kit_root / "benchmarks/robotwin"))
    monkeypatch.setitem(sys.modules, "yaml", object())
    from verify_seed_cache_pair import verify_episode_info_pair

    cached = {"info": {"seed": 7, "poses": [1, 2, 3]}}
    assert len(verify_episode_info_pair(cached, cached)) == 64
    with pytest.raises(RuntimeError, match="cached/native episode_info differ"):
        verify_episode_info_pair(cached, {"info": {"seed": 8, "poses": [1, 2, 3]}})

# ---------------------------------------------------------------------------
# LIBERO+
# ---------------------------------------------------------------------------


def test_libero_plus_factory_full_run_work_items() -> None:
    factory = LiberoPlusBenchmarkFactory()
    adapter = factory.create(
        _request(BenchmarkName.LIBERO_PLUS, "libero_10"),
        _libero_manifest(episodes=1),
    )
    items = adapter.work_items()
    assert len(items) == 2519
    assert sum(item.expected_results for item in items) == 2519
    metadata = adapter.metadata()
    task_ids = list(metadata["task_ids"])
    assert len(task_ids) == 2519
    assert len(set(task_ids)) == len(task_ids)
    assert len({item.item_id for item in items}) == len(items)
    assert len({item.group_id for item in items}) == len(items)
    assert {item.payload["task_id"] for item in items} == set(task_ids)
    assert all(item.expected_results == 1 for item in items)
    assert metadata["benchmark"] == "libero_plus"
    assert metadata["bound_hint"] == "gpu_infer"


def test_libero_plus_rejects_libero_90() -> None:
    factory = LiberoPlusBenchmarkFactory()
    with pytest.raises(ValueError, match="libero_plus does not support suite 'libero_90'"):
        factory.create(_request(BenchmarkName.LIBERO_PLUS, "libero_90"), _libero_manifest())


def test_libero_plus_quick_test_and_probe() -> None:
    from roboscope_kit.runtime.benchmark.adapters import WorkerContext

    factory = LiberoPlusBenchmarkFactory()
    quick = factory.create(
        _request(BenchmarkName.LIBERO_PLUS, "libero_goal", quick_test=True),
        _libero_manifest(episodes=50),
    )
    assert sum(item.expected_results for item in quick.work_items()) == 1
    context = WorkerContext(
        worker_id=0,
        gpu_slot=2,
        gpu_device="2",
        replica_id=0,
        replica_port=16694,
        replica_socket=Path("/tmp/replica.sock"),
        queue_dir=Path("/tmp/queue"),
        state_dir=Path("/tmp/state"),
        output_dir=Path("/tmp/out"),
        execution_contract=roboscope_kit.PolicyExecutionContract(
            benchmark="libero_plus",
            action_representation="libero.delta-eef-v1",
        ),
    )
    worker_spec = quick.worker_process(context)
    worker_env = worker_spec.env
    assert worker_spec.command[1:3] == ("-m", "roboscope_kit.runtime.benchmarks.libero.worker")
    representation_index = worker_spec.command.index("--action-representation") + 1
    assert worker_spec.command[representation_index] == "libero.delta-eef-v1"
    assert "--suite" in worker_spec.command
    assert worker_spec.command[worker_spec.command.index("--suite") + 1] == "libero_goal"
    assert worker_env["LIBERO_CONFIG_PATH"] == "/tmp/state/libero-config"
    assert worker_env["CUDA_VISIBLE_DEVICES"] == "2"
    assert worker_env["MUJOCO_EGL_DEVICE_ID"] == "2"
    probe = factory.create_probe(
        _request(BenchmarkName.LIBERO_PLUS, "libero_goal"),
        _libero_manifest(),
        concurrency=2,
    )
    assert len(probe.task_ids) == 10
    assert sum(item.expected_results for item in probe.work_items()) == 10
    larger = factory.create_probe(
        _request(BenchmarkName.LIBERO_PLUS, "libero_goal"),
        _libero_manifest(),
        concurrency=8,
    )
    assert len(larger.task_ids) == 24
    assert set(probe.task_ids) <= set(larger.task_ids)


def test_nested_task_prefix_matches_the_nested_order() -> None:
    from roboscope_kit.runtime.benchmark.factories import nested_task_order, nested_task_prefix

    for total in (1, 7, 24, 50):
        order = nested_task_order(total)
        for count in range(1, total + 2):
            assert nested_task_prefix(total, count) == tuple(sorted(order[:count]))


def test_libero_plus_aggregate_zero_fill() -> None:
    factory = LiberoPlusBenchmarkFactory()
    adapter = factory.create(
        _request(BenchmarkName.LIBERO_PLUS, "libero_spatial"),
        _libero_manifest(episodes=1),
    )
    output = adapter.aggregate([])
    assert len(output["tasks"]) == 2402
    assert all(row["episodes"] == 0 and row["success_rate"] == 0.0 for row in output["tasks"])


def test_libero_plus_self_test_reports_missing_paths(tmp_path: Path) -> None:
    factory = LiberoPlusBenchmarkFactory()
    manifest = _libero_manifest()
    manifest.runtime.update(
        benchmark_root=str(tmp_path / "missing-libero-plus"),
        config_path=str(tmp_path / "missing-config.yaml"),
    )
    with pytest.raises(RuntimeError, match="LIBERO\\+ runtime paths are missing"):
        factory.self_test(manifest, tmp_path)


# ---------------------------------------------------------------------------
# CALVIN
# ---------------------------------------------------------------------------


def _calvin_manifest(sequences_path: Path, **defaults) -> BenchmarkManifest:
    dataset_path = "/sim/calvin/dataset"
    runtime = {
        "python": "/usr/bin/python3",
        "benchmark_root": "/sim/calvin",
        "dataset_paths": {
            "task_D_D": dataset_path,
            "task_ABC_D": dataset_path,
            "task_ABCD_D": dataset_path,
        },
        "config_path": "/sim/calvin/conf",
        "eval_sequences_path": str(sequences_path),
    }
    return _manifest(
        runtime,
        defaults,
        (BenchmarkName.CALVIN,),
        ("task_D_D",),
    )


def _write_official_sequences(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int = 1000):
    """Write a sequence file and pin the adapter constants to it."""
    import hashlib

    import roboscope_kit.runtime.benchmarks.calvin.adapter as calvin_adapter

    sequences = [[[index] * 4, ["rotate_blue_block"] * 5] for index in range(count)]
    payload = json.dumps(sequences).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    path = tmp_path / "eval_sequences.json"
    path.write_bytes(payload)
    monkeypatch.setattr(calvin_adapter, "OFFICIAL_EVAL_SEQUENCE_COUNT", count)
    monkeypatch.setattr(calvin_adapter, "OFFICIAL_EVAL_SEQUENCES_SHA256", digest)
    return path


def test_calvin_factory_full_run_work_items(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sequences_path = _write_official_sequences(tmp_path, monkeypatch)
    factory = CalvinBenchmarkFactory()
    adapter = factory.create(
        _request(BenchmarkName.CALVIN, "task_D_D"),
        _calvin_manifest(sequences_path, episodes=10),
    )
    items = adapter.work_items()
    assert len(items) == 10
    assert sum(item.expected_results for item in items) == 10
    assert items[0].payload == {"start_sequence": 0, "num_sequences": 1}
    metadata = adapter.metadata()
    assert metadata["benchmark"] == "calvin"
    assert metadata["suite"] == "task_D_D"
    assert metadata["bound_hint"] == "gpu_infer"
    assert metadata["num_sequences"] == 10


def test_calvin_factory_quick_test(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sequences_path = _write_official_sequences(tmp_path, monkeypatch)
    factory = CalvinBenchmarkFactory()
    adapter = factory.create(
        _request(BenchmarkName.CALVIN, "task_D_D", quick_test=True),
        _calvin_manifest(sequences_path, episodes=1000),
    )
    assert adapter.config.num_sequences == 1
    assert len(adapter.work_items()) == 1


def test_calvin_factory_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sequences_path = _write_official_sequences(tmp_path, monkeypatch)
    factory = CalvinBenchmarkFactory()
    adapter = factory.create_probe(
        _request(BenchmarkName.CALVIN, "task_D_D"),
        _calvin_manifest(sequences_path),
        concurrency=4,
    )
    # The default diagonal search measures steady state: three items per worker.
    assert adapter.config.num_sequences == 12
    assert len(adapter.work_items()) == 12
    fixed = factory.create_probe(
        _request(BenchmarkName.CALVIN, "task_D_D"),
        _calvin_manifest(sequences_path, autotune_strategy="two-phase"),
        concurrency=4,
    )
    assert fixed.config.num_sequences == 10


def test_calvin_probe_grows_with_concurrency_and_stays_within_the_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sequences_path = _write_official_sequences(tmp_path, monkeypatch)
    factory = CalvinBenchmarkFactory()
    request = _request(BenchmarkName.CALVIN, "task_D_D")
    manifest = _calvin_manifest(sequences_path)
    assert factory.create_probe(request, manifest, concurrency=16).config.num_sequences == 48
    assert factory.create_probe(request, manifest, concurrency=64).config.num_sequences == 192
    capped = factory.create_probe(
        request, _calvin_manifest(sequences_path, probe_items_per_worker=40), concurrency=64
    )
    assert capped.config.num_sequences == 1000


@pytest.mark.parametrize(
    "representation",
    ("calvin.relative-v1", "calvin.absolute-compact-quat-v1"),
)
def test_calvin_worker_receives_negotiated_action_representation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    representation: str,
) -> None:
    from roboscope_kit.runtime.benchmark.adapters import WorkerContext

    sequences_path = _write_official_sequences(tmp_path, monkeypatch)
    adapter = CalvinBenchmarkFactory().create(
        _request(BenchmarkName.CALVIN, "task_D_D", quick_test=True),
        _calvin_manifest(sequences_path, episodes=1000),
    )
    context = WorkerContext(
        worker_id=0,
        gpu_slot=0,
        gpu_device="0",
        replica_id=0,
        replica_port=16694,
        replica_socket=Path("/tmp/replica.sock"),
        queue_dir=Path("/tmp/queue"),
        state_dir=Path("/tmp/state"),
        output_dir=Path("/tmp/out"),
        execution_contract=roboscope_kit.PolicyExecutionContract(
            benchmark="calvin",
            action_representation=representation,
        ),
    )

    spec = adapter.worker_process(context)
    index = spec.command.index("--action-representation")
    assert spec.command[index + 1] == representation
    assert "CALVIN_ACTION_TYPE" not in spec.env
    assert "--static-size" not in spec.command and "--gripper-size" not in spec.command


def test_calvin_worker_renders_the_negotiated_camera_sizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from roboscope_kit.runtime.benchmark.adapters import WorkerContext
    from roboscope_kit.runtime.benchmarks.calvin import worker as calvin_worker

    contract = _negotiate_image_sizes("calvin", "task_D_D", {"images.static": [(240, 320)]})
    assert contract.image_sizes == {"images.static": (240, 320)}
    sequences_path = _write_official_sequences(tmp_path, monkeypatch)
    adapter = CalvinBenchmarkFactory().create(
        _request(BenchmarkName.CALVIN, "task_D_D", quick_test=True),
        _calvin_manifest(sequences_path, episodes=1000),
    )
    context = WorkerContext(
        worker_id=0,
        gpu_slot=0,
        gpu_device="0",
        replica_id=0,
        replica_port=16694,
        replica_socket=Path("/tmp/replica.sock"),
        queue_dir=Path("/tmp/queue"),
        state_dir=Path("/tmp/state"),
        output_dir=Path("/tmp/out"),
        execution_contract=contract,
    )
    command = adapter.worker_process(context).command
    assert command[command.index("--static-size") + 1] == "240x320"
    assert "--gripper-size" not in command

    class CameraConfig(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    def cameras():
        return CameraConfig(
            cameras=CameraConfig(
                static=CameraConfig(width=200, height=200, aspect=1, fov=10),
                gripper=CameraConfig(width=84, height=84, aspect=1, fov=75),
            )
        )

    env_config = cameras()
    calvin_worker.apply_camera_sizes(env_config, {"static": (240, 320), "gripper": (128, 128)})
    assert env_config.cameras.static == {"width": 320, "height": 240, "aspect": 320 / 240, "fov": 10}
    assert env_config.cameras.gripper == {"width": 128, "height": 128, "aspect": 1.0, "fov": 75}
    with pytest.raises(RuntimeError, match="no tactile camera"):
        calvin_worker.apply_camera_sizes(cameras(), {"tactile": (64, 64)})


def test_calvin_factory_rejects_wrong_suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sequences_path = _write_official_sequences(tmp_path, monkeypatch)
    factory = CalvinBenchmarkFactory()
    with pytest.raises(ValueError, match="calvin does not support suite 'task_X_D'"):
        factory.create(
            _request(BenchmarkName.CALVIN, "task_X_D"),
            _calvin_manifest(sequences_path),
        )


def test_calvin_adapter_rejects_manifest_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "eval_sequences.json"
    path.write_text(json.dumps([[[0] * 4, ["task"] * 5]]), encoding="utf-8")
    factory = CalvinBenchmarkFactory()
    with pytest.raises(ValueError, match="CALVIN manifest contains 1 sequences"):
        factory.create(
            _request(BenchmarkName.CALVIN, "task_D_D"),
            _calvin_manifest(path, episodes=1),
        )


def test_calvin_aggregate_official_scoring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sequences_path = _write_official_sequences(tmp_path, monkeypatch)
    factory = CalvinBenchmarkFactory()
    adapter = factory.create(
        _request(BenchmarkName.CALVIN, "task_D_D"),
        _calvin_manifest(sequences_path, episodes=4),
    )
    lengths = [5, 3, 0, 5]
    result = WorkResult(
        item_id="sequences-0000-0004",
        group_id="sequences-0000",
        worker_id=0,
        records=tuple(
            EvaluationRecord(
                record_id=f"sequence-{index:04d}",
                success=length == 5,
                steps=100,
                elapsed_s=5.0,
                metadata={
                    "sequence_index": index,
                    "chained_success_length": length,
                    "eval_sequence": ["rotate_blue_block"] * 5,
                },
            )
            for index, length in enumerate(lengths)
        ),
        metadata={"elapsed_s": 20.0},
    )
    output = adapter.aggregate([result])
    assert output["num_sequences"] == 4
    assert output["avg_seq_len"] == pytest.approx(13 / 4)
    assert output["success_at_5"] == pytest.approx(0.5)
    assert output["task_success_rate"]["task_1"] == pytest.approx(0.75)
    assert output["task_success_rate"]["task_5"] == pytest.approx(0.5)
    assert output["sequence_success_lengths"] == lengths


def test_calvin_aggregate_rejects_incomplete_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sequences_path = _write_official_sequences(tmp_path, monkeypatch)
    factory = CalvinBenchmarkFactory()
    adapter = factory.create(
        _request(BenchmarkName.CALVIN, "task_D_D"),
        _calvin_manifest(sequences_path, episodes=2),
    )
    result = WorkResult(
        item_id="sequences-0000-0001",
        group_id="sequences-0000",
        worker_id=0,
        records=(
            EvaluationRecord(
                record_id="sequence-0000",
                success=True,
                steps=100,
                elapsed_s=5.0,
                metadata={"sequence_index": 0, "chained_success_length": 5},
            ),
        ),
        metadata={"elapsed_s": 5.0},
    )
    with pytest.raises(RuntimeError, match="do not exactly cover"):
        adapter.aggregate([result])


def test_calvin_self_test_reports_missing_paths(tmp_path: Path) -> None:
    factory = CalvinBenchmarkFactory()
    manifest = _calvin_manifest(tmp_path / "missing.json")
    manifest.runtime.update(
        benchmark_root=str(tmp_path / "missing-calvin"),
        config_path=str(tmp_path / "missing-config"),
        dataset_paths={
            suite: str(tmp_path / "missing-dataset")
            for suite in manifest.runtime["dataset_paths"]
        },
    )
    with pytest.raises(RuntimeError, match="CALVIN runtime paths are missing"):
        factory.self_test(manifest, tmp_path / "out")


def test_robotwin_launcher_loads_the_nvtx_helper_only_in_a_profile_run(monkeypatch) -> None:
    from benchmarks.robotwin import roboscope_robotwin_launcher as launcher

    from roboscope_kit import profiling

    monkeypatch.setenv("ROBOSCOPE_PROFILING_MODULE", profiling.__file__)
    monkeypatch.delenv("ROBOSCOPE_PROFILE", raising=False)
    assert launcher._load_profiling() is None
    # Under --profile but outside nsys there is no injection library, so nothing is recorded.
    monkeypatch.setenv("ROBOSCOPE_PROFILE", "1")
    monkeypatch.delenv("NVTX_INJECTION64_PATH", raising=False)
    assert launcher._load_profiling() is None


def test_robotwin_lean_camera_scene_skips_only_the_unused_cameras() -> None:
    from benchmarks.robotwin import roboscope_robotwin_launcher as launcher

    created = []

    class Scene:
        timestep = 0.004

        def add_camera(self, name, *args, **kwargs):
            created.append(name)
            return name

    lean = launcher._LeanCameraScene(Scene())
    assert lean.add_camera("head_camera", 320, 240) == "head_camera"
    observer = lean.add_camera("observer_camera", 320, 240)
    world = lean.add_camera("world_camera1", 640, 480)
    assert created == ["head_camera"]
    assert lean.timestep == 0.004
    observer.entity.set_pose(None)
    with pytest.raises(RuntimeError, match="world_camera1"):
        world.take_picture()


def _image_size_capabilities(benchmark: str, image_sizes) -> PolicyCapabilities:
    return PolicyCapabilities(
        supported_benchmarks=(benchmark,), required_features=(), image_sizes=image_sizes
    )


def _negotiate_image_sizes(benchmark: str, suite: str, image_sizes, defaults=None):
    from roboscope_kit.benchmark_specs import benchmark_camera_support, get_benchmark_spec

    spec = get_benchmark_spec(benchmark, suite)
    return roboscope_kit.negotiate_policy_action_contract(
        spec,
        _image_size_capabilities(benchmark, image_sizes),
        camera_support=benchmark_camera_support(spec, defaults or {}),
    )


def test_image_sizes_default_to_the_benchmark_size() -> None:
    contract = _negotiate_image_sizes("libero", "libero_10", None)
    assert contract.image_sizes == {} and contract.canonical_observations
    assert contract.to_dict()["image_sizes"] == {} and contract.to_dict()["canonical_observations"] is True


def test_image_sizes_pick_the_first_size_the_benchmark_renders() -> None:
    from roboscope_kit.image_sizes import apply_image_sizes

    preferences = [(2000, 2000), (128, 160), roboscope_kit.ANY_SIZE]
    contract = _negotiate_image_sizes("libero", "libero_10", preferences)
    assert contract.image_sizes == {"images.primary": (128, 160), "images.wrist": (128, 160)}
    assert contract.to_dict()["canonical_observations"] is False
    spec = apply_image_sizes(roboscope_kit.get_benchmark_spec("libero", "libero_10"), contract.image_sizes)
    assert {feature.name: feature.shape for feature in spec.features if feature.kind.value == "image"} == {
        "images.primary": (128, 160, 3),
        "images.wrist": (128, 160, 3),
    }
    wrist_only = _negotiate_image_sizes("libero_pro", "libero_goal_swap", {"images.wrist": [(96, 128)]})
    assert wrist_only.image_sizes == {"images.wrist": (96, 128)}
    fallback = _negotiate_image_sizes("libero", "libero_10", [(2000, 2000), roboscope_kit.ANY_SIZE])
    assert fallback.image_sizes == {}

    calvin = _negotiate_image_sizes("calvin", "task_D_D", [(200, 200), (84, 84), roboscope_kit.ANY_SIZE])
    assert calvin.image_sizes == {"images.gripper": (200, 200)}
    assert _negotiate_image_sizes("calvin", "task_D_D", None).image_sizes == {}
    calvin_spec = apply_image_sizes(
        roboscope_kit.get_benchmark_spec("calvin", "task_D_D"), {"images.static": (96, 128)}
    )
    calvin_shapes = {feature.name: feature.shape for feature in calvin_spec.features}
    assert calvin_shapes["images.static"] == (96, 128, None) and calvin_shapes["images.gripper"] is None


@pytest.mark.parametrize(
    ("benchmark", "suite", "image_sizes", "defaults", "message"),
    [
        ("libero_plus", "libero_10", [(128, 160)], None, "primary: the benchmark renders square"),
        ("libero_plus", "libero_10", {"images.primary": [(320, 320)]}, None, "square sizes 64-256"),
        ("libero_plus", "libero_10", {"images.wrist": [(600, 600)]}, {"render_backend": "osmesa"}, "64-480"),
        ("libero", "libero_10", [(600, 600)], {"render_backend": "osmesa"}, "heights 64-480"),
        ("robodojo", "arx_x5", [(128, 128)], None, "unspecified"),
        ("robotwin", "demo_clean", [(2000, 2000)], {"negotiated_camera_sizes": True}, "heights 64-1024"),
        ("calvin", "task_D_D", [(2048, 2048)], None, "heights 64-1024"),
        ("libero", "libero_10", {"images.agentview": [(128, 128)]}, None, "not libero images"),
    ],
)
def test_image_size_negotiation_names_both_sides_when_it_fails(
    benchmark: str, suite: str, image_sizes, defaults, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _negotiate_image_sizes(benchmark, suite, image_sizes, defaults)


@pytest.mark.parametrize(
    "image_sizes",
    [[], "any", [roboscope_kit.ANY_SIZE, (128, 128)], [(128, 128), (128, 128)], [(True, 128)], [b"AB"],
     [(0, 128)], {1: [(128, 128)]}, {}],
)
def test_image_size_preferences_are_validated(image_sizes) -> None:
    with pytest.raises(ValueError):
        _image_size_capabilities("libero", image_sizes)


def test_libero_plus_renders_square_agent_views_and_any_wrist_size() -> None:
    from roboscope_kit.benchmark_specs import benchmark_camera_support, get_benchmark_spec
    from roboscope_kit.image_sizes import camera_support_from_dict, camera_support_to_dict

    contract = _negotiate_image_sizes(
        "libero_plus", "libero_spatial", {"images.primary": [(128, 128)], "images.wrist": [(128, 160)]}
    )
    assert contract.image_sizes == {"images.primary": (128, 128), "images.wrist": (128, 160)}
    shared = _negotiate_image_sizes("libero_plus", "libero_goal", [(128, 160), (96, 96)])
    assert shared.image_sizes == {"images.primary": (96, 96), "images.wrist": (128, 160)}
    assert _negotiate_image_sizes("libero_plus", "libero_10", None).image_sizes == {}
    assert _negotiate_image_sizes("libero_plus", "libero_10", [(256, 256)]).image_sizes == {}
    table = benchmark_camera_support(get_benchmark_spec("libero_plus", "libero_10"), {})
    payload = camera_support_to_dict(table)
    assert payload["images.primary"]["square"] is True and "square" not in payload["images.wrist"]
    assert camera_support_from_dict(json.loads(json.dumps(payload))) == table
    libero = camera_support_to_dict(benchmark_camera_support(get_benchmark_spec("libero", "libero_10"), {}))
    assert all(set(support) == {"canonical", "min_size", "max_size"} for support in libero.values())
    with pytest.raises(ValueError, match="square must be a boolean"):
        camera_support_from_dict({"images.primary": {**payload["images.primary"], "square": 1}})


def test_image_size_preferences_round_trip() -> None:
    capabilities = _image_size_capabilities("libero", {"images.wrist": [(128, 160), roboscope_kit.ANY_SIZE]})
    payload = capabilities.to_dict()
    assert payload["image_sizes"] == {"images.wrist": [[128, 160], "any"]}
    assert PolicyCapabilities.from_dict(json.loads(json.dumps(payload))) == capabilities
    assert "image_sizes" not in _image_size_capabilities("libero", None).to_dict()


def test_policy_server_validates_observations_at_the_negotiated_size(tmp_path: Path) -> None:
    import numpy as np

    from roboscope_kit.benchmark_specs import benchmark_camera_support
    from roboscope_kit.image_sizes import camera_support_to_dict

    class SmallImagePolicy(roboscope_kit.Policy):
        def load(self, context):
            assert context.benchmark.features[0].shape == (256, 256, 3)
            return PolicyCapabilities(
                supported_benchmarks=("libero",), required_features=(), image_sizes=[(128, 160)]
            )

        def act(self, observation):
            return roboscope_kit.ActionChunk(np.zeros((1, 7), dtype=np.float32))

    spec = roboscope_kit.get_benchmark_spec("libero", "libero_10")
    server = roboscope_kit.PolicyServer(SmallImagePolicy(), tmp_path / "policy.sock")
    context = roboscope_kit.SessionContext(session_id="s", checkpoint_path="/ckpt", benchmark=spec)
    server._dispatch(
        "load",
        {
            "context": context.to_dict(),
            "camera_support": camera_support_to_dict(benchmark_camera_support(spec, {})),
        },
    )
    episode = roboscope_kit.EpisodeContext(episode_id="e", task_id="0")
    server._dispatch("reset", {"context": episode.to_dict()})

    def observation(height: int, width: int) -> dict:
        return roboscope_kit.Observation(
            episode_id="e",
            step=0,
            timestamp_s=0.0,
            features={
                "images.primary": np.zeros((height, width, 3), dtype=np.uint8),
                "images.wrist": np.zeros((height, width, 3), dtype=np.uint8),
                "state.proprioception": np.zeros(16, dtype=np.float32),
                "language.instruction": "task",
            },
        ).to_dict()

    server._dispatch("act", {"observation": observation(128, 160)})
    with pytest.raises(ValueError, match="shape"):
        server._dispatch("act", {"observation": observation(256, 256)})


def test_rlbench_depth_and_point_cloud_follow_the_negotiated_camera_size() -> None:
    from roboscope_kit.benchmark_specs import benchmark_camera_support
    from roboscope_kit.image_sizes import apply_image_sizes, camera_support_from_dict, camera_support_to_dict

    spec = roboscope_kit.get_benchmark_spec("rlbench", "peract_18")
    assert _negotiate_image_sizes("rlbench", "peract_18", None).image_sizes == {}
    assert _negotiate_image_sizes("rlbench", "peract_18", [(128, 128)]).image_sizes == {}
    contract = _negotiate_image_sizes(
        "rlbench", "peract_18", {"images.front": [(2048, 2048), (96, 160)], "images.wrist": [(256, 256)]}
    )
    assert contract.image_sizes == {"images.front": (96, 160), "images.wrist": (256, 256)}
    support = camera_support_from_dict(
        json.loads(json.dumps(camera_support_to_dict(benchmark_camera_support(spec, {}))))
    )
    assert support == benchmark_camera_support(spec, {})
    shapes = {
        feature.name: feature.shape
        for feature in apply_image_sizes(spec, contract.image_sizes, support).features
    }
    assert shapes["images.front"] == (96, 160, 3)
    assert shapes["state.front_depth"] == (96, 160)
    assert shapes["state.front_point_cloud"] == (96, 160, 3)
    assert shapes["state.wrist_point_cloud"] == (256, 256, 3)
    assert shapes["state.left_shoulder_depth"] == (128, 128)
    assert shapes["state.front_camera_intrinsics"] == (3, 3)
    # Benchmarks without dependent features send the camera support they sent before.
    assert "spatial_features" not in camera_support_to_dict(
        benchmark_camera_support(roboscope_kit.get_benchmark_spec("libero", "libero_10"), {})
    )["images.primary"]
    with pytest.raises(ValueError, match="heights 32-768"):
        _negotiate_image_sizes("rlbench", "peract_18", [(16, 16)])
    with pytest.raises(ValueError, match="heights 32-768"):
        _negotiate_image_sizes("rlbench", "peract_18", [(1024, 1024)])

    from roboscope_kit.image_sizes import (
        BASELINE_IMAGE_SIZE_FEATURES,
        IMAGE_SIZE_FEATURES,
        camera_support_for_server,
        camera_support_requirements,
        fixed_camera_support,
    )

    full = benchmark_camera_support(spec, {})
    front = next(feature for feature in spec.features if feature.name == "images.front")
    needed = camera_support_requirements(front, full["images.front"])
    assert needed == {"fixed_shape_ranges", "spatial_features"}
    assert "spatial_features" in IMAGE_SIZE_FEATURES
    assert camera_support_for_server(spec, full, frozenset(IMAGE_SIZE_FEATURES)) == full
    # A Policy server that cannot resize depth and point clouds (0.8.0b77) is offered PerAct's size only.
    baseline = camera_support_for_server(spec, full, BASELINE_IMAGE_SIZE_FEATURES)
    assert baseline == fixed_camera_support(spec)
    capabilities = _image_size_capabilities("rlbench", [(96, 160), roboscope_kit.ANY_SIZE])
    contract = roboscope_kit.negotiate_policy_action_contract(spec, capabilities, camera_support=baseline)
    assert contract.image_sizes == {}


def test_rlbench_worker_command_carries_only_negotiated_sizes(tmp_path: Path) -> None:
    from roboscope_kit.policy import PolicyExecutionContract
    from roboscope_kit.runtime.benchmark.adapters import WorkerContext
    from roboscope_kit.runtime.benchmarks.rlbench.adapter import (
        RLBenchBenchmarkAdapter,
        RLBenchRuntimeConfig,
    )
    from roboscope_kit.runtime.benchmarks.rlbench.tasks import camera_size, camera_sizes

    unused = Path("/unused")
    adapter = RLBenchBenchmarkAdapter(
        RLBenchRuntimeConfig(
            python=unused,
            coppeliasim=unused,
            test_set=unused,
            xvfb=unused,
            suite="peract_18",
            tasks=("open_drawer",),
            episodes_per_task=1,
        )
    )

    def command(image_sizes):
        return adapter.worker_process(
            WorkerContext(
                worker_id=0,
                gpu_slot=0,
                replica_id=0,
                gpu_device="0",
                replica_port=0,
                replica_socket=Path("/tmp/replica.sock"),
                queue_dir=tmp_path / "queue",
                state_dir=tmp_path / "state",
                output_dir=tmp_path,
                execution_contract=PolicyExecutionContract(
                    benchmark="rlbench",
                    action_representation="rlbench.ee-pose-planning-v1",
                    image_sizes=image_sizes,
                ),
            )
        ).command

    assert "--camera-size" not in command({})
    sized = command({"images.wrist": (96, 160), "images.front": (256, 256)})
    values = [sized[index + 1] for index, item in enumerate(sized) if item == "--camera-size"]
    assert values == ["front=256x256", "wrist=96x160"]
    assert camera_sizes(camera_size(value) for value in values) == {"front": (256, 256), "wrist": (96, 160)}
    for bad in ("overhead=128x128", "front=128", "front=0x128"):
        with pytest.raises(argparse.ArgumentTypeError):
            camera_size(bad)
    with pytest.raises(ValueError, match="more than one size"):
        camera_sizes([("front", (128, 128)), ("front", (64, 64))])


@pytest.mark.parametrize(
    ("benchmark", "factory"),
    [(BenchmarkName.LIBERO, LiberoBenchmarkFactory), (BenchmarkName.LIBERO_PLUS, LiberoPlusBenchmarkFactory)],
)
def test_libero_worker_command_carries_only_negotiated_sizes(tmp_path: Path, benchmark, factory) -> None:
    from roboscope_kit.policy import PolicyExecutionContract
    from roboscope_kit.runtime.benchmark.adapters import WorkerContext

    adapter = factory().create(_request(benchmark, "libero_10"), _libero_manifest(episodes=1))

    def command(image_sizes):
        return adapter.worker_process(
            WorkerContext(
                worker_id=0,
                gpu_slot=0,
                replica_id=0,
                gpu_device="0",
                replica_port=0,
                replica_socket=Path("/tmp/replica.sock"),
                queue_dir=tmp_path / "queue",
                state_dir=tmp_path / "state",
                output_dir=tmp_path,
                execution_contract=PolicyExecutionContract(
                    benchmark=benchmark.value,
                    action_representation="libero.delta-eef-v1",
                    image_sizes=image_sizes,
                ),
            )
        ).command

    assert "--primary-size" not in command({}) and "--wrist-size" not in command({})
    sized = command({"images.wrist": (128, 160)})
    assert sized[sized.index("--wrist-size") + 1] == "128x160" and "--primary-size" not in sized
    both = command({"images.primary": (128, 128), "images.wrist": (128, 160)})
    assert both[both.index("--primary-size") + 1] == "128x128"
    assert both[both.index("--wrist-size") + 1] == "128x160"


def _metaworld_contract(cameras, image_sizes):
    from roboscope_kit.benchmark_specs import MetaworldBenchmarkDefinition, benchmark_camera_support

    spec = MetaworldBenchmarkDefinition().spec("mt50", cameras)
    return roboscope_kit.negotiate_policy_action_contract(
        spec,
        _image_size_capabilities("metaworld", image_sizes),
        camera_support=benchmark_camera_support(spec, {}),
    )


def test_metaworld_negotiates_sizes_around_each_configured_camera() -> None:
    from roboscope_kit.runtime.benchmarks.metaworld.config import resolve_evaluation_config

    cameras = tuple(
        resolve_evaluation_config(
            {
                "cameras": [
                    {"feature": "images.primary", "camera_name": "corner2"},
                    {"feature": "images.secondary", "camera_name": "gripperPOV", "output_width": 224,
                     "output_height": 224},
                ]
            }
        )["cameras"]
    )
    assert _metaworld_contract(cameras, None).image_sizes == {}
    preferred = _metaworld_contract(cameras, [(224, 224), (480, 480)])
    assert preferred.image_sizes == {"images.primary": (224, 224)}
    assert _metaworld_contract(cameras, [(96, 160)]).image_sizes == {
        "images.primary": (96, 160),
        "images.secondary": (96, 160),
    }
    with pytest.raises(ValueError, match="heights 64-1024"):
        _metaworld_contract(cameras, [(32, 32)])


def test_metaworld_cameras_render_negotiated_sizes_natively() -> None:
    import numpy as np

    from roboscope_kit.runtime.benchmarks.metaworld.agent import process_image
    from roboscope_kit.runtime.benchmarks.metaworld.config import apply_image_sizes, resolve_evaluation_config

    cameras = tuple(
        resolve_evaluation_config(
            {
                "cameras": [
                    {"feature": "images.primary", "camera_name": "corner2", "center_crop_ratio": 0.8,
                     "output_width": 224, "output_height": 224},
                    {"feature": "images.secondary", "camera_name": "corner", "rotation_deg": 90},
                ]
            }
        )["cameras"]
    )
    assert apply_image_sizes(cameras, {}) == cameras
    sizes = {"images.primary": (96, 160), "images.secondary": (96, 160)}
    primary, secondary = apply_image_sizes(cameras, sizes)
    assert (primary["render_height"], primary["render_width"], primary["center_crop_ratio"]) == (96, 160, 1.0)
    assert (primary["output_height"], primary["output_width"]) == (96, 160)
    assert (secondary["render_height"], secondary["render_width"]) == (160, 96)
    assert (secondary["output_height"], secondary["output_width"]) == (96, 160)
    assert secondary["camera_name"] == "corner" and secondary["rotation_deg"] == 90
    frame = np.arange(160 * 96 * 3, dtype=np.uint8).reshape(160, 96, 3)
    assert np.array_equal(process_image(frame, secondary), np.rot90(frame))
    unchanged = apply_image_sizes(cameras, {"images.secondary": (96, 160)})[0]
    assert unchanged == cameras[0]
    with pytest.raises(ValueError, match="unconfigured cameras"):
        apply_image_sizes(cameras, {"images.tertiary": (96, 160)})


def test_metaworld_worker_command_carries_only_negotiated_sizes(tmp_path: Path) -> None:
    from roboscope_kit.policy import PolicyExecutionContract
    from roboscope_kit.runtime.benchmark.adapters import WorkerContext
    from roboscope_kit.runtime.benchmarks.metaworld.adapter import (
        MetaworldBenchmarkAdapter,
        MetaworldRuntimeConfig,
    )
    from roboscope_kit.runtime.benchmarks.metaworld.config import default_evaluation_config
    from roboscope_kit.runtime.benchmarks.metaworld.worker import _image_size

    adapter = MetaworldBenchmarkAdapter(
        MetaworldRuntimeConfig(
            python=Path("/runtime/python"),
            suite="mt50",
            tasks=("reach-v3",),
            episodes_per_task=1,
            evaluation_config=default_evaluation_config(),
            evaluation_config_path=tmp_path / "evaluation-config.json",
        )
    )

    def command(image_sizes):
        return adapter.worker_process(
            WorkerContext(
                worker_id=0,
                gpu_slot=0,
                replica_id=0,
                gpu_device="0",
                replica_port=0,
                replica_socket=Path("/tmp/replica.sock"),
                queue_dir=tmp_path / "queue",
                state_dir=tmp_path / "state",
                output_dir=tmp_path,
                execution_contract=PolicyExecutionContract(
                    benchmark="metaworld",
                    action_representation="metaworld.delta-eef-v1",
                    image_sizes=image_sizes,
                ),
            )
        ).command

    assert "--image-size" not in command({})
    sized = command({"images.primary": (96, 160)})
    assert sized[sized.index("--image-size") + 1] == "images.primary=96x160"
    assert _image_size("images.primary=96x160") == ("images.primary", (96, 160))
    for value in ("96x160", "images.primary=96", "images.primary=0x160", "=96x160"):
        with pytest.raises(argparse.ArgumentTypeError):
            _image_size(value)


def test_metaworld_offscreen_buffer_grows_to_the_largest_negotiated_camera() -> None:
    from types import SimpleNamespace

    from roboscope_kit.runtime.benchmarks.metaworld.worker import _fit_offscreen_buffer

    visual = SimpleNamespace(offwidth=160, offheight=96)
    model = SimpleNamespace(vis=SimpleNamespace(global_=visual))
    env = SimpleNamespace(unwrapped=SimpleNamespace(model=model))
    cameras = ({"render_width": 160, "render_height": 96}, {"render_width": 120, "render_height": 200})
    _fit_offscreen_buffer(env, cameras)
    assert (visual.offwidth, visual.offheight) == (160, 200)


def test_camera_support_is_limited_to_what_the_policy_server_understands(tmp_path: Path) -> None:
    from dataclasses import replace

    from roboscope_kit.contracts import FeatureKind, FeatureSpec
    from roboscope_kit.image_sizes import (
        BASELINE_IMAGE_SIZE_FEATURES,
        IMAGE_SIZE_FEATURES,
        CameraSupport,
        camera_support_for_server,
        camera_support_requirements,
    )

    spec = roboscope_kit.get_benchmark_spec("libero", "libero_10")
    ranged = CameraSupport(canonical=(256, 256), min_size=(64, 64), max_size=(1024, 1024))
    assert camera_support_requirements(spec.features[0], ranged) == {"fixed_shape_ranges"}
    shapeless = FeatureSpec(name="images.static", kind=FeatureKind.IMAGE, dtype="uint8")
    assert camera_support_requirements(shapeless, CameraSupport(canonical=None)) == frozenset()
    assert camera_support_requirements(shapeless, ranged) == {"open_shape_ranges"}
    with_shapeless = replace(spec, features=(*spec.features, shapeless))
    table = {"images.primary": ranged, "images.static": ranged}
    limited = camera_support_for_server(with_shapeless, table, BASELINE_IMAGE_SIZE_FEATURES)
    assert limited["images.primary"] == ranged
    assert limited["images.static"] == CameraSupport(canonical=(256, 256))
    everything = BASELINE_IMAGE_SIZE_FEATURES | {"open_shape_ranges"}
    assert camera_support_for_server(with_shapeless, table, everything) == table

    server = roboscope_kit.PolicyServer(object(), tmp_path / "policy.sock")
    assert server._dispatch("health", {})["image_size_features"] == list(IMAGE_SIZE_FEATURES)
    assert "open_shape_ranges" in IMAGE_SIZE_FEATURES

    from roboscope_kit.benchmark_specs import benchmark_camera_support

    calvin = roboscope_kit.get_benchmark_spec("calvin", "task_D_D")
    calvin_support = benchmark_camera_support(calvin, {})
    older = camera_support_for_server(calvin, calvin_support, BASELINE_IMAGE_SIZE_FEATURES)
    assert older == {
        "images.static": CameraSupport(canonical=(200, 200)),
        "images.gripper": CameraSupport(canonical=(84, 84)),
    }
    preferences = [(240, 320), roboscope_kit.ANY_SIZE]
    assert roboscope_kit.negotiate_policy_action_contract(
        calvin, _image_size_capabilities("calvin", preferences), camera_support=older
    ).image_sizes == {}
    current = camera_support_for_server(calvin, calvin_support, frozenset(IMAGE_SIZE_FEATURES))
    assert current == calvin_support
    with_encodings = server._dispatch("health", {"encodings": ["json"]})
    assert with_encodings["image_size_features"] == list(IMAGE_SIZE_FEATURES)


def test_libero_plus_square_agent_view_needs_a_server_that_knows_square_sizes() -> None:
    from roboscope_kit.benchmark_specs import benchmark_camera_support
    from roboscope_kit.image_sizes import (
        BASELINE_IMAGE_SIZE_FEATURES,
        IMAGE_SIZE_FEATURES,
        CameraSupport,
        camera_support_for_server,
        camera_support_requirements,
    )

    spec = roboscope_kit.get_benchmark_spec("libero_plus", "libero_spatial")
    table = benchmark_camera_support(spec, {})
    features = {feature.name: feature for feature in spec.features}
    assert camera_support_requirements(features["images.primary"], table["images.primary"]) == {
        "fixed_shape_ranges",
        "square_sizes",
    }
    wrist = camera_support_requirements(features["images.wrist"], table["images.wrist"])
    assert wrist == {"fixed_shape_ranges"}
    assert "square_sizes" in IMAGE_SIZE_FEATURES
    old = camera_support_for_server(spec, table, BASELINE_IMAGE_SIZE_FEATURES)
    assert old["images.primary"] == CameraSupport(canonical=(256, 256))
    assert old["images.wrist"] == table["images.wrist"]
    assert camera_support_for_server(spec, table, frozenset(IMAGE_SIZE_FEATURES)) == table


def _camera_support_tables():
    from roboscope_kit.benchmark_specs import benchmark_camera_support, get_benchmark_spec

    for path in sorted((Path(__file__).parents[3] / "benchmarks").glob("*/benchmark-manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for benchmark, suites in manifest["supported_suites"].items():
            for suite in suites:
                spec = get_benchmark_spec(benchmark, suite)
                for backend in ("egl", "osmesa"):
                    yield spec, benchmark_camera_support(spec, {"render_backend": backend})


def test_size_groups_share_one_size_range_and_accept_the_benchmark_size() -> None:
    tables = list(_camera_support_tables())
    assert any(support.size_group for _, table in tables for support in table.values())
    for spec, table in tables:
        groups: dict[str, list] = {}
        for support in table.values():
            if support.size_group is not None:
                groups.setdefault(support.size_group, []).append(support)
        for group, members in groups.items():
            shape = {(s.canonical, s.min_size, s.max_size, s.square) for s in members}
            assert len(shape) == 1, (spec.name, spec.suite, group, shape)
            canonical = members[0].canonical
            assert canonical is not None and members[0].renders(canonical), (spec.name, spec.suite, group)
        for image_sizes in (None, [roboscope_kit.ANY_SIZE]):
            capabilities = PolicyCapabilities(
                supported_benchmarks=(spec.name,), required_features=(), image_sizes=image_sizes
            )
            assert roboscope_kit.negotiate_policy_action_contract(
                spec, capabilities, camera_support=table
            ).image_sizes == {}


ROBODOJO_PATCHED = {"negotiated_camera_sizes": True}


def test_robodojo_cameras_negotiate_sizes_independently() -> None:
    from roboscope_kit.image_sizes import apply_image_sizes

    assert _negotiate_image_sizes("robodojo", "arx_x5", None, ROBODOJO_PATCHED).image_sizes == {}
    assert _negotiate_image_sizes("robodojo", "arx_x5", [(480, 640)], ROBODOJO_PATCHED).image_sizes == {}
    contract = _negotiate_image_sizes(
        "robodojo", "arx_x5", {"images.head": [(4096, 4096), (240, 424)], "images.left_wrist": [(224, 224)]},
        ROBODOJO_PATCHED,
    )
    assert contract.image_sizes == {"images.head": (240, 424), "images.left_wrist": (224, 224)}
    spec = apply_image_sizes(roboscope_kit.get_benchmark_spec("robodojo", "arx_x5"), contract.image_sizes)
    assert {feature.name: feature.shape for feature in spec.features if feature.kind.value == "image"} == {
        "images.head": (240, 424, None),
        "images.left_wrist": (224, 224, None),
        "images.right_wrist": None,
    }
    with pytest.raises(ValueError, match="heights 64-2048"):
        _negotiate_image_sizes("robodojo", "arx_x5", [(32, 32)], ROBODOJO_PATCHED)
    from roboscope_kit.benchmark_specs import benchmark_camera_support
    from roboscope_kit.image_sizes import (
        BASELINE_IMAGE_SIZE_FEATURES,
        IMAGE_SIZE_FEATURES,
        camera_support_for_server,
    )

    robodojo = roboscope_kit.get_benchmark_spec("robodojo", "arx_x5")
    full = benchmark_camera_support(robodojo, ROBODOJO_PATCHED)
    # A 0.8.0b77 Policy server cannot fix open shapes, so it is offered each camera's own size only.
    old = camera_support_for_server(robodojo, full, BASELINE_IMAGE_SIZE_FEATURES)
    assert all(support.min_size is None for support in old.values())
    assert camera_support_for_server(robodojo, full, frozenset(IMAGE_SIZE_FEATURES)) == full


def test_robodojo_runtime_without_the_camera_patch_offers_its_own_size_only() -> None:
    from roboscope_kit.benchmark_specs import benchmark_camera_support
    from roboscope_kit.image_sizes import fixed_camera_support

    manifest_path = Path(__file__).parents[3] / "benchmarks/robodojo/benchmark-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["defaults"]["negotiated_camera_sizes"] is True
    robodojo = roboscope_kit.get_benchmark_spec("robodojo", "arx_x5")
    # A runtime built before camera-sizes.patch has a manifest without the flag: the Kit offers the
    # same table as before, so the Policy keeps 480 x 640 frames or the run stops before any Worker starts.
    old_defaults = dict(manifest["defaults"])
    del old_defaults["negotiated_camera_sizes"]
    assert benchmark_camera_support(robodojo, old_defaults) == fixed_camera_support(robodojo)
    kept = _negotiate_image_sizes("robodojo", "arx_x5", [(224, 224), roboscope_kit.ANY_SIZE], old_defaults)
    assert kept.image_sizes == {}
    with pytest.raises(ValueError, match="only its own"):
        _negotiate_image_sizes("robodojo", "arx_x5", [(224, 224)], old_defaults)


def test_robodojo_worker_renders_only_negotiated_cameras(tmp_path: Path) -> None:
    from roboscope_kit.policy import PolicyExecutionContract
    from roboscope_kit.runtime.benchmark.adapters import WorkerContext
    from roboscope_kit.runtime.benchmarks.robodojo.adapter import (
        RoboDojoBenchmarkAdapter,
        RoboDojoRuntimeConfig,
    )
    from roboscope_kit.runtime.benchmarks.robodojo.worker import _camera_size, camera_resolutions

    adapter = RoboDojoBenchmarkAdapter(RoboDojoRuntimeConfig(
        python=Path("/env/bin/python"), benchmark_root=tmp_path, suite="arx_x5", tasks=("cover_blocks",),
        episodes={"cover_blocks": 1}, episode_shard_size=25, seed=1, cpu_threads=2,
    ))

    def command(image_sizes):
        return adapter.worker_process(WorkerContext(
            worker_id=0, gpu_slot=0, replica_id=0, gpu_device="0", replica_port=0,
            replica_socket=Path("/tmp/replica.sock"), queue_dir=tmp_path / "queue",
            state_dir=tmp_path / "state", output_dir=tmp_path,
            execution_contract=PolicyExecutionContract(
                benchmark="robodojo", action_representation="robodojo.x5-joint-v1", image_sizes=image_sizes,
            ),
        )).command

    assert "--camera-size" not in command({})
    sized = command({"images.right_wrist": (224, 224), "images.head": (240, 424)})
    assert [sized[index + 1] for index, item in enumerate(sized) if item == "--camera-size"] == [
        "images.head=240x424", "images.right_wrist=224x224",
    ]
    sizes = [_camera_size(value) for value in ("images.head=240x424", "images.right_wrist=224x224")]
    assert camera_resolutions(sizes) == {"cam_head": [424, 240], "cam_right_wrist": [224, 224]}
    for value in ("images.primary=224x224", "images.head=224", "images.head=0x224"):
        with pytest.raises(argparse.ArgumentTypeError):
            _camera_size(value)


ROBOTWIN_SIZED = {"negotiated_camera_sizes": True}


def test_robotwin_cameras_negotiate_any_size_in_range_and_keep_d435_by_default() -> None:
    from roboscope_kit.image_sizes import apply_image_sizes

    cameras = {"images.head", "images.left_wrist", "images.right_wrist"}
    assert _negotiate_image_sizes("robotwin", "demo_clean", None, ROBOTWIN_SIZED).image_sizes == {}
    assert _negotiate_image_sizes("robotwin", "demo_clean", [(240, 320)], ROBOTWIN_SIZED).image_sizes == {}
    contract = _negotiate_image_sizes("robotwin", "demo_randomized", [(224, 224)], ROBOTWIN_SIZED)
    assert contract.image_sizes == {name: (224, 224) for name in cameras}
    head_only = _negotiate_image_sizes(
        "robotwin", "demo_clean", {"images.head": [(180, 320)]}, ROBOTWIN_SIZED
    )
    assert head_only.image_sizes == {"images.head": (180, 320)}
    # RoboTwin leaves its image shapes open, so the negotiated spec checks height and width only.
    spec = roboscope_kit.get_benchmark_spec("robotwin", "demo_clean")
    spec = apply_image_sizes(spec, head_only.image_sizes)
    assert {feature.name: feature.shape for feature in spec.features if feature.name in cameras} == {
        "images.head": (180, 320, None),
        "images.left_wrist": None,
        "images.right_wrist": None,
    }


def _robotwin_sized_command(tmp_path: Path, image_sizes, **config):
    from roboscope_kit.runtime.benchmark.adapters import WorkerContext

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
        **config,
    }
    adapter = RoboTwinBenchmarkAdapter(RoboTwinRuntimeConfig(**values))
    return adapter.worker_process(
        WorkerContext(
            worker_id=0,
            gpu_slot=0,
            gpu_device="0",
            replica_id=0,
            replica_port=0,
            replica_socket=tmp_path / "replica.sock",
            queue_dir=tmp_path / "queue",
            state_dir=tmp_path / "state",
            output_dir=tmp_path / "out",
            execution_contract=roboscope_kit.PolicyExecutionContract(
                benchmark="robotwin",
                action_representation="robotwin.qpos-v1",
                image_sizes=image_sizes,
            ),
        )
    ).command


def test_robotwin_worker_command_carries_only_negotiated_sizes(tmp_path: Path) -> None:
    flags = ("--head-size", "--left-wrist-size", "--right-wrist-size")
    assert not set(flags) & set(_robotwin_sized_command(tmp_path, {}))
    sized = _robotwin_sized_command(tmp_path, {"images.left_wrist": (128, 160)})
    assert sized[sized.index("--left-wrist-size") + 1] == "128x160"
    assert "--head-size" not in sized and "--right-wrist-size" not in sized


def test_robotwin_camera_ranges_need_a_flagged_runtime_and_native_rendering(tmp_path: Path) -> None:
    from roboscope_kit.benchmark_specs import benchmark_camera_support
    from roboscope_kit.image_sizes import CameraSupport

    spec = roboscope_kit.get_benchmark_spec("robotwin", "demo_clean")
    cameras = ("images.head", "images.left_wrist", "images.right_wrist")
    fixed = {name: CameraSupport(canonical=None) for name in cameras}
    # A runtime built before negotiated sizes has no flag, and its launcher renders 240 x 320 only.
    assert benchmark_camera_support(spec, {}) == fixed
    assert benchmark_camera_support(spec, {"negotiated_camera_sizes": "yes"}) == fixed
    assert benchmark_camera_support(spec, {**ROBOTWIN_SIZED, "render_mode": "central"}) == fixed
    ranged = benchmark_camera_support(spec, {**ROBOTWIN_SIZED, "render_mode": "native"})
    assert ranged == benchmark_camera_support(spec, ROBOTWIN_SIZED)
    assert {name: (support.canonical, support.min_size, support.max_size) for name, support in ranged.items()
            if name in cameras} == {name: ((240, 320), (64, 64), (1024, 1024)) for name in cameras}
    # The shipped manifest carries the flag.
    manifest_path = Path(__file__).parents[3] / "benchmarks/robotwin/benchmark-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["defaults"]["negotiated_camera_sizes"] is True
    central = {
        "seed_cache_dir": Path("/sim/robotwin/seed-cache"),
        "render_mode": "central",
        "render_service_path": Path("/sim/robotwin/render.py"),
    }
    with pytest.raises(ValueError, match="task config's camera sizes"):
        _robotwin_sized_command(tmp_path, {"images.head": (224, 224)}, **central)


def test_robotwin_worker_passes_camera_sizes_to_the_launcher_only_when_negotiated(monkeypatch) -> None:
    from roboscope_kit.runtime.benchmarks.robotwin import worker

    base = [
        "worker", "--worker-id", "0", "--gpu-id", "0", "--policy-socket", "/p", "--robotwin-dir", "/r",
        "--robotwin-python", "/py", "--checkpoint-path", "/c", "--deploy-policy", "/d", "--launcher", "/l",
        "--queue-dir", "/q", "--worker-state-dir", "/s", "--output-dir", "/o",
    ]
    monkeypatch.setattr("sys.argv", base)
    assert worker._camera_env(worker.parse_args()) == {}
    monkeypatch.setenv(worker.CAMERA_SIZES_ENV, '{"head_camera": [96, 96]}')
    assert worker.CAMERA_SIZES_ENV not in worker._inherited_env()
    monkeypatch.setattr("sys.argv", [*base, "--head-size", "224x224", "--right-wrist-size", "128x160"])
    env = worker._camera_env(worker.parse_args())
    assert json.loads(env[worker.CAMERA_SIZES_ENV]) == {"head_camera": [224, 224], "right_camera": [128, 160]}
    monkeypatch.setattr("sys.argv", [*base, "--head-size", "224"])
    with pytest.raises(SystemExit):
        worker.parse_args()


def test_robotwin_sized_camera_scene_changes_only_the_named_cameras_sizes(monkeypatch) -> None:
    from benchmarks.robotwin import roboscope_robotwin_launcher as render

    created = []

    class Scene:
        timestep = 0.004

        def add_camera(self, name, *args, **kwargs):
            created.append((name, kwargs))
            return name

    sized = render.SizedCameraScene(Scene(), {"head_camera": (128, 160)})
    sized.add_camera(name="head_camera", width=320, height=240, fovy=0.6, near=0.1, far=100)
    sized.add_camera(name="front_camera", width=320, height=240, fovy=0.6, near=0.1, far=100)
    assert created == [
        ("head_camera", {"width": 160, "height": 128, "fovy": 0.6, "near": 0.1, "far": 100}),
        ("front_camera", {"width": 320, "height": 240, "fovy": 0.6, "near": 0.1, "far": 100}),
    ]
    assert sized.timestep == 0.004

    monkeypatch.delenv(render.CAMERA_SIZES_ENV, raising=False)
    assert render.camera_sizes_from_env() == {}
    monkeypatch.setenv(render.CAMERA_SIZES_ENV, '{"left_camera": [96, 128]}')
    assert render.camera_sizes_from_env() == {"left_camera": (96, 128)}
    monkeypatch.setenv(render.CAMERA_SIZES_ENV, '{"observer_camera": [96, 128]}')
    with pytest.raises(ValueError, match="observer_camera"):
        render.camera_sizes_from_env()


def test_robotwin_launcher_sizes_videos_only_with_a_negotiated_head_camera(monkeypatch) -> None:
    from benchmarks.robotwin import roboscope_robotwin_launcher as launcher

    source = (
        '    if args["eval_video_log"]:\n'
        '        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])\n'
        '                    "-pix_fmt",\n'
    )
    monkeypatch.delenv("ROBOTWIN_CAMERA_SIZES", raising=False)
    assert launcher._inject_camera_sizes(source) == source
    monkeypatch.setenv("ROBOTWIN_CAMERA_SIZES", '{"left_camera": [96, 128]}')
    assert launcher._inject_camera_sizes(source) == source
    monkeypatch.setenv("ROBOTWIN_CAMERA_SIZES", '{"head_camera": [180, 320]}')
    assert '        video_size = "320x180"\n' in launcher._inject_camera_sizes(source)
    assert "pad=" not in launcher._inject_camera_sizes(source)
    monkeypatch.setenv("ROBOTWIN_CAMERA_SIZES", '{"head_camera": [181, 321]}')
    padded = launcher._inject_camera_sizes(source)
    assert '"pad=ceil(iw/2)*2:ceil(ih/2)*2",\n                    "-pix_fmt"' in padded


def test_robotwin_camera_sizes_are_offered_only_to_servers_with_open_shape_ranges() -> None:
    from roboscope_kit.benchmark_specs import benchmark_camera_support
    from roboscope_kit.image_sizes import (
        BASELINE_IMAGE_SIZE_FEATURES,
        IMAGE_SIZE_FEATURES,
        camera_support_for_server,
        negotiate_image_sizes,
    )

    spec = roboscope_kit.get_benchmark_spec("robotwin", "demo_clean")
    table = benchmark_camera_support(spec, ROBOTWIN_SIZED)
    assert "open_shape_ranges" in IMAGE_SIZE_FEATURES
    assert camera_support_for_server(spec, table, frozenset(IMAGE_SIZE_FEATURES)) == table
    # A roboscope-kit 0.8.0b77 server keeps RoboTwin's own size, as 0.8.0b77 does.
    older = camera_support_for_server(spec, table, BASELINE_IMAGE_SIZE_FEATURES)
    assert negotiate_image_sizes(spec, ((224, 224), roboscope_kit.ANY_SIZE), older) == {}
    assert negotiate_image_sizes(spec, ((240, 320),), older) == {}
    with pytest.raises(ValueError, match="renders only 240x320"):
        negotiate_image_sizes(spec, ((224, 224),), older)


def test_metaworld_cameras_resolve_from_the_policy_contract(tmp_path: Path) -> None:
    from roboscope_kit.policy_config import PolicyRuntimeConfig
    from roboscope_kit.runtime.benchmark.cli import _resolve_evaluation_config, parse_args
    from roboscope_kit.runtime.benchmarks.metaworld.config import default_evaluation_config

    def policy(benchmark_config: dict) -> PolicyRuntimeConfig:
        return PolicyRuntimeConfig(
            tmp_path / "roboscope-policy.json", ("python",), tmp_path, benchmark_config
        )

    declared = {
        "metaworld": {
            "cameras": [
                {"feature": "images.primary", "camera_name": "corner", "output_width": 224,
                 "output_height": 224},
            ]
        }
    }
    (camera,) = _resolve_evaluation_config(BenchmarkName.METAWORLD, policy(declared))["cameras"]
    assert (camera["camera_name"], camera["output_height"], camera["output_width"]) == ("corner", 224, 224)
    assert _resolve_evaluation_config(BenchmarkName.METAWORLD, policy({})) == default_evaluation_config()
    assert _resolve_evaluation_config(BenchmarkName.LIBERO, policy(declared)) is None
    with pytest.raises(SystemExit):
        parse_args([
            "--benchmark", "metaworld", "--output-dir", str(tmp_path / "run"),
            "--benchmark-manifest", str(tmp_path / "benchmark-manifest.json"),
            "--evaluation-config", str(tmp_path / "cameras.json"),
        ])


def test_shard_merge_uses_the_camera_config_every_metaworld_shard_recorded(tmp_path: Path) -> None:
    from roboscope_kit.runtime.benchmark.shard_merge import _shard_evaluation_config
    from roboscope_kit.runtime.benchmarks.metaworld.config import default_evaluation_config

    shards = [tmp_path / "shard-0", tmp_path / "shard-1"]
    for shard in shards:
        shard.mkdir()
        (shard / "evaluation-config.json").write_text(json.dumps(default_evaluation_config()))
    assert _shard_evaluation_config(BenchmarkName.METAWORLD, shards) == shards[0] / "evaluation-config.json"
    assert _shard_evaluation_config(BenchmarkName.ROBOTWIN, [tmp_path / "unused"]) is None

    rotated = default_evaluation_config()
    rotated["cameras"][0]["rotation_deg"] = 0
    (shards[1] / "evaluation-config.json").write_text(json.dumps(rotated))
    with pytest.raises(ValueError, match="different camera configurations"):
        _shard_evaluation_config(BenchmarkName.METAWORLD, shards)
    (shards[1] / "evaluation-config.json").unlink()
    with pytest.raises(ValueError, match="evaluation-config.json missing"):
        _shard_evaluation_config(BenchmarkName.METAWORLD, shards)
