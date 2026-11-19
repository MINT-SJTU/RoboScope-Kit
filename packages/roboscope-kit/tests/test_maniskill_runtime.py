"""ManiSkill profile, scoring, shard and Worker-recycling rules, without the simulator."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from roboscope_kit.policy_config import PolicyRuntimeConfig
from roboscope_kit.runtime.benchmark import autotune
from roboscope_kit.runtime.benchmark.orchestrator import (
    ParallelBenchmarkRuntime,
    RuntimeConfig,
    aggregate_standard_results,
)
from roboscope_kit.runtime.benchmark.queue import WorkQueue
from roboscope_kit.runtime.benchmark.schema import EvaluationRecord, WorkItem, WorkResult
from roboscope_kit.runtime.benchmark.shard_reduce import merge_shard_results
from roboscope_kit.runtime.benchmark.shard_validation import validate_shard_accounting
from roboscope_kit.runtime.benchmark.sharding import select_shard_items, shard_info
from roboscope_kit.runtime.benchmark.worker import WorkerLoop
from roboscope_kit.runtime.benchmarks.maniskill.adapter import (
    ManiSkillBenchmarkAdapter,
    ManiSkillRuntimeConfig,
)
from roboscope_kit.runtime.benchmarks.maniskill.metrics import (
    merge_task_results,
    overall_task_results,
    validate_task_row,
)
from roboscope_kit.runtime.benchmarks.maniskill.profile import PROTOCOL_ID, validate_profiles
from roboscope_kit.runtime.benchmarks.maniskill.shard_identity import shard_input_identity


def _protocol() -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "vector_envs": 2,
        "physx_gpu_library_sha256": "a" * 64,
        "episodes_per_task": 4,
        "profiles": {
            "push": {
                "task_id": "PushCube-v1", "obs_mode": "state", "robot_uid": "panda",
                "control_mode": "pd_joint_delta_pos", "max_episode_steps": 50,
            },
            "lift": {
                "task_id": "LiftPegUpright-v1", "obs_mode": "state", "robot_uid": "panda",
                "control_mode": "pd_ee_delta_pose", "max_episode_steps": 40,
                "success_metric": "success_at_end",
            },
            "stand": {
                "task_id": "UnitreeG1Stand-v1", "obs_mode": "state", "robot_uid": "unitree_g1",
                "control_mode": "pd_joint_pos", "max_episode_steps": 30, "success_metric": "return",
            },
        },
        "suites": {"mixed": ["push", "lift", "stand"]},
    }


# ---- profile and configuration validation ---------------------------------------------------


def test_profiles_resolve_protocol_defaults() -> None:
    push, lift, stand = validate_profiles(_protocol(), "mixed")

    assert push["profile_id"] == "push" and push["success_metric"] == "success_once"
    assert push["seeds"] == [0, 1, 2, 3] and push["episodes_per_task"] == 4
    assert push["instruction"] == "PushCube-v1" and push["sim_backend"] == "physx_cuda"
    assert lift["success_metric"] == "success_at_end"
    assert stand["success_metric"] == "return"


def _set(path: tuple[str, ...], value: Any):
    def mutate(protocol: dict[str, Any]) -> None:
        target = protocol
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_set(("protocol_id",), "maniskill-v0"), "protocol_id"),
        (_set(("vector_envs",), 0), "vector_envs"),
        (_set(("physx_gpu_library_sha256",), "A" * 64), "physx_gpu_library_sha256"),
        (_set(("episodes_per_task",), 3), "complete frozen vector groups"),
        (_set(("profiles", "push", "seeds"), [0, 1, 1, 2]), "unique non-negative"),
        (_set(("profiles", "push", "seeds"), [0, 1]), "cover every episode"),
        (_set(("profiles", "push", "success_metric"), "reward"), "success_metric"),
        (_set(("profiles", "push", "ignore_terminations"), False), "full-horizon"),
        (_set(("profiles", "push", "sim_backend"), "physx_cpu"), "physx_cuda"),
        (_set(("profiles", "push", "obs_mode"), "pointcloud"), "observation mode"),
        (_set(("profiles", "push", "control_mode"), ["a", "b"]), "pair each robot"),
        (_set(("profiles", "push", "depth_cameras"), {"state.depth_hand": "hand"}), "native depth"),
        (_set(("profiles", "push", "cameras"), {"images.top": "base"}), "cameras"),
        (_set(("profiles", "push", "diagnostic_step_limit"), 51), "diagnostic_step_limit"),
        (_set(("suites", "mixed"), ["push", "push"]), "duplicate profiles"),
        (_set(("suites", "mixed"), ["push", "missing"]), "missing"),
    ],
)
def test_invalid_profiles_are_rejected(mutate, message: str) -> None:
    protocol = _protocol()
    mutate(protocol)

    with pytest.raises(ValueError, match=message):
        validate_profiles(protocol, "mixed")


# ---- return-only versus binary-success scoring ----------------------------------------------


def _scored(record_id: str, success: bool, metric: str) -> EvaluationRecord:
    return EvaluationRecord(record_id, success, steps=10, elapsed_s=0.5, metadata={"success_metric": metric})


def test_standard_results_keep_return_records_out_of_success_counts() -> None:
    records = (
        _scored("a", True, "success_once"),
        _scored("b", False, "success_at_end"),
        _scored("c", False, "return"),
    )
    result = aggregate_standard_results([WorkResult("item", "group", 0, records)], 3)

    assert result["episodes"] == 3
    assert (result["successes"], result["failures"]) == (1, 1)
    assert result["success_rate"] == 0.5
    assert result["success_evaluated_episodes"] == 2
    assert result["return_evaluated_episodes"] == 1


def test_return_only_results_have_no_success_rate() -> None:
    records = (_scored("a", False, "return"), _scored("b", False, "return"))
    result = aggregate_standard_results([WorkResult("item", "group", 0, records)], 2)

    assert result["success_rate"] is None
    assert (result["successes"], result["failures"]) == (0, 0)


def test_binary_results_keep_their_original_fields() -> None:
    records = (_scored("a", True, "success_once"), _scored("b", False, "success_once"))
    result = aggregate_standard_results([WorkResult("item", "group", 0, records)], 2)

    assert result["success_rate"] == 0.5
    assert "return_evaluated_episodes" not in result


def _row(
    profile_id: str, metric: str, successes: int | None, episodes: int, returns: float,
) -> dict[str, Any]:
    binary = metric != "return"
    return {
        "profile_id": profile_id, "task_id": profile_id, "episodes": episodes,
        "success_metric": metric,
        "successes": successes if binary else None,
        "failures": episodes - successes if binary else None,
        "success_rate": successes / episodes if binary else None,
        "success_once_successes": successes if binary else None,
        "success_once_rate": successes / episodes if binary else None,
        "success_at_end_successes": successes if binary else None,
        "success_at_end_rate": successes / episodes if binary else None,
        "mean_return": returns / episodes, "return_sum": returns, "return_count": episodes,
        "actual_steps": 10 * episodes,
    }


def test_overall_result_excludes_return_rows_from_success() -> None:
    overall = overall_task_results([
        _row("push", "success_once", 3, 4, 2.0), _row("stand", "return", None, 4, -8.0),
    ])

    assert overall["successes"] == 3 and overall["success_rate"] == 0.75
    assert overall["success_evaluated_episodes"] == 4
    assert overall["return_evaluated_episodes"] == 4
    assert overall_task_results([_row("stand", "return", None, 4, -8.0)])["success_rate"] is None


def test_a_return_row_may_not_report_successes() -> None:
    row = _row("stand", "return", None, 4, -8.0)
    row["successes"] = 0

    with pytest.raises(ValueError, match="return-only rows"):
        validate_task_row(row)


def test_merging_sums_samples_instead_of_averaging_rates() -> None:
    def partial(successes: int, episodes: int, returns: float) -> dict[str, Any]:
        rows = [_row("push", "success_once", successes, episodes, returns)]
        return {
            "evaluation_kind": "full_horizon", "tasks": rows, "overall": overall_task_results(rows),
            "runtime_metrics": {
                "actual_policy_steps": 10 * episodes, "inference_wait_s": 1.0, "vector_groups": 1,
            },
        }

    merged = merge_task_results([partial(1, 2, 1.0), partial(0, 6, 3.0)])
    row = merged["tasks"][0]

    assert row["episodes"] == 8 and row["successes"] == 1
    assert row["success_rate"] == 1 / 8  # The mean of the shard rates would be 1/4.
    assert row["mean_return"] == 0.5
    assert merged["formal"] is False


def test_autotune_health_accepts_a_return_only_probe() -> None:
    output = {
        "recovery": {
            "initial_workers_per_gpu": {"0": 2}, "final_workers_per_gpu": {"0": 2},
            "worker_failures": [], "oom_events": 0,
        },
        "result": {
            "episodes": 4, "total_episodes": 4, "successes": 0, "success_rate": None,
            "success_evaluated_episodes": 0, "return_evaluated_episodes": 4,
        },
    }

    health = autotune._concurrency_health(2, output)

    assert health["success_rate"] is None
    assert health["return_evaluated_episodes"] == 4


def _controller(baseline_health: dict[str, Any]) -> autotune.BenchmarkAutotuneController:
    controller = autotune.BenchmarkAutotuneController.__new__(autotune.BenchmarkAutotuneController)
    controller.baseline_key = (1, 1)
    controller.cache = {(1, 1): (None, None, baseline_health)}
    controller.record_success = {}
    controller.limits = SimpleNamespace(maximum_success_rate_drop=0.1)
    return controller


def test_autotune_skips_the_success_check_for_return_only_probes() -> None:
    health: dict[str, Any] = {"success_rate": None}

    _controller({"success_rate": None})._check_correctness(2, 1, health, {})

    assert health["success_check"] == "not_applicable_return_only"


def test_autotune_baseline_counts_only_scored_episodes() -> None:
    baseline = {"success_rate": 0.5, "successes": 2, "completed_episodes": 8, "success_evaluated_episodes": 4}
    health: dict[str, Any] = {"success_rate": 0.5}

    _controller(baseline)._check_correctness(2, 1, health, {})

    assert health["baseline_episodes"] == 4


# ---- shard merge and validation -------------------------------------------------------------


def _adapter() -> ManiSkillBenchmarkAdapter:
    return ManiSkillBenchmarkAdapter(ManiSkillRuntimeConfig(
        python=Path("python"), manifest_path=Path("manifest.json"), runtime_home=Path("home"),
        asset_dir=Path("assets"), suite="mixed", profiles=validate_profiles(_protocol(), "mixed"),
        envs_per_worker=2,
    ))


def _vector_result(adapter: ManiSkillBenchmarkAdapter, item: WorkItem) -> WorkResult:
    profile = next(p for p in adapter.config.profiles if p["profile_id"] == item.group_id)
    metric = profile["success_metric"]
    records = []
    for lane, seed in enumerate(item.payload["seeds"]):
        index = item.payload["episode_start"] + lane
        once, at_end = index % 2 == 0, index % 3 == 0
        metadata = {
            "task_id": profile["task_id"], "profile_id": item.group_id, "suite": "mixed",
            "protocol": PROTOCOL_ID, "episode_index": index, "seed": seed, "vector_lane": lane,
            "success_metric": metric, "terminated": False, "truncated": True,
            # Binary-exact returns, so shard sums equal the whole-run sum bit for bit.
            "episode_return": 0.25 * index - 1.0, "success_once": once, "success_at_end": at_end,
        }
        success = False if metric == "return" else metadata[metric]
        records.append(EvaluationRecord(
            f"{item.group_id}-episode-{index:06d}-seed-{seed}", success,
            steps=profile["max_episode_steps"], elapsed_s=0.5, metadata=metadata,
        ))
    return WorkResult(item.item_id, item.group_id, 0, tuple(records), {"inference_wait_s": 0.25})


BUILD = {
    "kit_version": "0.8.0b67", "manifest_sha256": "a" * 64,
    "checkpoint_sha256": "b" * 64, "policy_sha256": "c" * 64,
}


def _shard_documents(count: int) -> list[dict[str, Any]]:
    adapter = _adapter()
    items = list(adapter.work_items())
    documents = []
    for index in range(count):
        selected = select_shard_items(items, index, count)
        results = [_vector_result(adapter, item) for item in selected]
        standard = aggregate_standard_results(results, sum(item.expected_results for item in selected))
        documents.append({
            "sdk_version": 1,
            "shard": shard_info(
                index, count, sum(item.expected_results for item in items), 500, True, dict(BUILD),
            ),
            "benchmark": adapter.metadata(),
            "result": standard,
            "official_result": adapter.aggregate_partial(results),
        })
    return documents


@pytest.mark.parametrize("count", [1, 2, 3])
def test_merged_shards_equal_the_unsharded_run(count: int) -> None:
    adapter = _adapter()
    everything = [_vector_result(adapter, item) for item in adapter.work_items()]

    merged = merge_shard_results(_shard_documents(count))

    assert merged["official_result"] == adapter.aggregate(everything)
    whole = aggregate_standard_results(everything, 12)
    for key in ("episodes", "successes", "failures", "success_rate",
                "success_evaluated_episodes", "return_evaluated_episodes"):
        assert merged["result"][key] == whole[key], key


def _return_shard(documents: list[dict[str, Any]]) -> dict[str, Any]:
    return next(document for document in documents if "return_evaluated_episodes" in document["result"])


def test_a_shard_without_model_identity_is_rejected() -> None:
    documents = _shard_documents(2)
    for document in documents:
        del document["shard"]["build"]["policy_sha256"]

    with pytest.raises(ValueError, match="policy_sha256"):
        merge_shard_results(documents)


def test_a_shard_with_return_records_must_report_scoring_counts() -> None:
    documents = _shard_documents(3)
    result = _return_shard(documents)["result"]
    del result["success_evaluated_episodes"], result["return_evaluated_episodes"]

    with pytest.raises(ValueError, match="require explicit scoring counts"):
        merge_shard_results(documents)


def test_a_shard_whose_rate_disagrees_with_its_scored_episodes_is_rejected() -> None:
    documents = _shard_documents(1)
    documents[0]["result"]["success_rate"] = documents[0]["result"]["successes"] / 12

    with pytest.raises(ValueError, match="success rate differs"):
        merge_shard_results(documents)


def test_shard_rates_use_the_metric_tolerance() -> None:
    document = _shard_documents(1)[0]
    document["result"]["success_rate"] += 1e-16

    validate_shard_accounting(document)


def test_an_incomplete_profile_is_rejected_on_merge() -> None:
    documents = _shard_documents(1)
    official = documents[0]["official_result"]
    official["tasks"][0]["profile_id"] = "other"

    with pytest.raises(ValueError):
        merge_shard_results(documents)


def test_shard_inputs_are_not_mutated() -> None:
    documents = _shard_documents(2)
    before = copy.deepcopy(documents)

    merge_shard_results(documents)

    assert documents == before


def test_policy_identity_ignores_virtual_environments(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    code = tmp_path / "policy"
    (code / ".venv" / "lib").mkdir(parents=True)
    (code / "env").mkdir()
    (code / "env" / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
    (code / "policy.py").write_text("ACTION = 0\n", encoding="utf-8")
    policy = PolicyRuntimeConfig(tmp_path / "policy.json", ("python", "policy.py"), code, {})
    before = shard_input_identity(checkpoint, policy)

    (code / ".venv" / "lib" / "torch.py").write_text("x = 1\n", encoding="utf-8")
    (code / "env" / "site.py").write_text("x = 1\n", encoding="utf-8")
    assert shard_input_identity(checkpoint, policy) == before

    (code / "policy.py").write_text("ACTION = 1\n", encoding="utf-8")
    assert shard_input_identity(checkpoint, policy)["policy_sha256"] != before["policy_sha256"]


# ---- Worker recycling on a context change ---------------------------------------------------


def _run_worker(
    tmp_path: Path, items: list[tuple[str, int]], recycle: bool,
) -> tuple[list[str], dict[str, Any]]:
    queue = WorkQueue(tmp_path / "queue")
    queue.initialize([
        WorkItem(ordinal=index, item_id=f"item-{index}", group_id=group, payload={}, expected_results=width)
        for index, (group, width) in enumerate(items)
    ])
    (queue.root / "start").touch()
    (queue.root / "stop").touch()
    worker = WorkerLoop(
        worker_id=0, queue_dir=queue.root, state_dir=tmp_path / "state",
        poll_interval_s=0.001, recycle_on_context_change=recycle,
    )
    evaluated: list[str] = []

    def evaluate(item: WorkItem) -> WorkResult:
        evaluated.append(item.item_id)
        records = tuple(
            EvaluationRecord(f"{item.item_id}-{n}", True, steps=1, elapsed_s=0.0)
            for n in range(item.expected_results)
        )
        return WorkResult(item.item_id, item.group_id, 0, records)

    worker.run(evaluate)
    return evaluated, json.loads(worker.state_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "items",
    [
        [("push", 2), ("push", 2), ("lift", 2)],  # Next task.
        [("push", 2), ("push", 2), ("push", 1)],  # Same task, smaller vector width.
    ],
)
def test_worker_recycles_before_evaluating_a_new_context(
    tmp_path: Path, items: list[tuple[str, int]],
) -> None:
    evaluated, state = _run_worker(tmp_path, items, recycle=True)

    assert evaluated == ["item-0", "item-1"]
    assert state["state"] == "recycling" and state["completed_records"] == 4


def test_worker_without_context_recycling_runs_every_item(tmp_path: Path) -> None:
    evaluated, state = _run_worker(tmp_path, [("push", 2), ("lift", 2), ("lift", 1)], recycle=False)

    assert evaluated == ["item-0", "item-1", "item-2"]
    assert state["state"] == "idle"


class _Process:
    def __init__(self, returncode: int | None) -> None:
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode


def test_probe_health_counts_a_recycling_worker_as_live(tmp_path: Path) -> None:
    runtime = ParallelBenchmarkRuntime.__new__(ParallelBenchmarkRuntime)
    runtime.config = RuntimeConfig(output_dir=tmp_path, gpu_devices=("0",), replicas=1, concurrency=3)
    runtime.initial_workers_per_gpu = {0: 3}
    runtime.workers = {
        0: SimpleNamespace(process=_Process(None)),  # Running.
        1: SimpleNamespace(process=_Process(0)),  # Exited for a planned recycle.
        2: SimpleNamespace(process=_Process(1)),  # Crashed.
    }
    runtime.state_dir = tmp_path / "state"
    runtime.state_dir.mkdir()
    for worker_id, state in ((1, "recycling"), (2, "running")):
        (runtime.state_dir / f"worker-{worker_id:03d}.json").write_text(
            json.dumps({"state": state}), encoding="utf-8",
        )
    runtime.queue = SimpleNamespace(observable_results=lambda: [])
    runtime.failure_events = []

    health = runtime._probe_health(12)

    assert health["initial_workers"] == 3
    assert health["final_workers"] == 2
    assert health["expected_episodes"] == 12
