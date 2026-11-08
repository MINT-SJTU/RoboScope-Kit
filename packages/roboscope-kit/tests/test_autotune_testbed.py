# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from roboscope_kit.runtime.benchmark import autotune
from roboscope_kit.runtime.benchmark.autotune import (
    DEFAULT_AUTOTUNE_STRATEGY,
    AutotuneBounds,
    AutotuneLimits,
    BenchmarkAutotuneController,
    ProbeCapacityReached,
    ProbeStopped,
    autotune_strategies,
    resolve_bounds,
)
from roboscope_kit.runtime.benchmark.autotune_testbed import (
    FixedDefault,
    Gap,
    fixed_default,
    load_record,
    main,
    record_runs,
    regression_failures,
    replay_strategy,
    search_mismatches,
    validate_record,
)

WORKLOADS = Path(__file__).parent / "autotune_workloads"
REAL = "robocasa-tabletop-rtx4090-issue80"
FULL_AUTO = "robocasa-tabletop-rtx4090-full-auto"
LIBERO = "libero-10-rtx4090-full-auto"
ROBOTWIN = "robotwin-demo-clean-subset-rtx4090-full-auto"
SYNTHETIC = "synthetic-replica-limits"
UNKNOWN_ON_REAL = (
    "cpu_cores",
    "gpu_utilization",
    "peak_memory_bytes",
    "queue_wait_p95_ms",
    "record_success",
    "steps_per_s",
    "throttle_time_ratio",
    "wall_time_s",
)

# The regression rule, on the default strategy: workload -> expected relative throughput, judged
# with REGRESSION_TOLERANCE. Lowering an expected value is allowed only in a PR that says why, and
# it needs maintainer review.
JUDGED = {
    FULL_AUTO: 0.6484,
    LIBERO: 0.9994,
    ROBOTWIN: 0.9405,
}
# Judged workload -> the recorded real search that ran the autotune code of this branch. Its
# replay must probe the same K/g in the same order, stop for the same reason, select the same K/g,
# and take the same time. A change to the search names here the searches it still reproduces; a
# new workload needs a real search of the current code.
REPRODUCED = {FULL_AUTO: "main 5f79ce2", LIBERO: "main d26a908", ROBOTWIN: "main 44fa1ad"}
# Real workloads the rule does not judge, and why.
NOT_JUDGED = {REAL: "the same workload as FULL_AUTO, without fixed runs or replica probes"}

# (workload, strategy, estimator override) -> what the search must do. A strategy or a workload
# added without an entry here fails test_strategy_on_workload.
EXPECTED = {
    # Known problem, #80: no window at K=16 ends the search, and the replica phase never runs.
    (REAL, "diagonal", None): {
        "selected": (8, 1),
        "best_recorded": (8, 1),
        "relative_throughput": 1.0,
        "probes": 5,
        "tuning_s": None,
        "safe": True,
        "gaps": (),
        "unknown_signals": UNKNOWN_ON_REAL,
    },
    (REAL, "diagonal", "steady"): {
        "selected": (8, 1),
        "best_recorded": (8, 1),
        "relative_throughput": 1.0,
        "probes": 5,
        "tuning_s": None,
        "safe": True,
        "gaps": (),
        "unknown_signals": UNKNOWN_ON_REAL,
    },
    # two-phase's own estimator needs an elapsed rate, which #80 does not give for any K.
    (REAL, "two-phase", None): {
        "selected": None,
        "best_recorded": None,
        "relative_throughput": None,
        "probes": 0,
        "tuning_s": None,
        "safe": None,
        "gaps": (Gap(1, 1, "steps_per_s"),),
        "unknown_signals": (),
    },
    # The GPU at 33% is below the 40% where two-phase ramps replicas.
    (REAL, "two-phase", "steady"): {
        "selected": (8, 1),
        "best_recorded": (8, 1),
        "relative_throughput": 1.0,
        "probes": 5,
        "tuning_s": None,
        "safe": True,
        "gaps": (),
        "unknown_signals": UNKNOWN_ON_REAL,
    },
    # Built by the record command from real runs. Known problem, #80: the search stops at K=8/g=1,
    # while the sweep shows K=12/g=2 as the fastest configuration.
    (FULL_AUTO, "diagonal", None): {
        "selected": (8, 1),
        "best_recorded": (12, 2),
        "relative_throughput": pytest.approx(0.6484, abs=0.0001),
        "probes": 5,
        "tuning_s": pytest.approx(851.97, abs=0.01),
        "end_to_end_s": pytest.approx(6648.8, abs=0.1),
        "end_to_end_vs_best": pytest.approx(1.769, abs=0.001),
        "safe": True,
        "gaps": (),
        "unknown_signals": ("queue_wait_p95_ms",),
    },
    (FULL_AUTO, "diagonal", "steady"): {
        "selected": (8, 1),
        "best_recorded": (12, 2),
        "relative_throughput": pytest.approx(0.6484, abs=0.0001),
        "probes": 5,
        "tuning_s": pytest.approx(851.97, abs=0.01),
        "end_to_end_s": pytest.approx(6648.8, abs=0.1),
        "end_to_end_vs_best": pytest.approx(1.769, abs=0.001),
        "safe": True,
        "gaps": (),
        "unknown_signals": ("queue_wait_p95_ms",),
    },
    # two-phase's next K candidate is 9, which no search probed. The fixed run at K=9/g=1 stands in
    # for that probe, so the search completes, but it ran the full workload and has no probe time.
    (FULL_AUTO, "two-phase", None): {
        "selected": (9, 1),
        "best_recorded": (12, 2),
        "relative_throughput": pytest.approx(0.6199, abs=0.0001),
        "probes": 5,
        "tuning_s": None,
        "safe": True,
        "gaps": (),
        "unknown_signals": ("queue_wait_p95_ms", "wall_time_s"),
        "stand_ins": ((9, 1),),
    },
    (FULL_AUTO, "two-phase", "steady"): {
        "selected": (9, 1),
        "best_recorded": (12, 2),
        "relative_throughput": pytest.approx(0.6199, abs=0.0001),
        "probes": 5,
        "tuning_s": None,
        "safe": True,
        "gaps": (),
        "unknown_signals": ("queue_wait_p95_ms", "wall_time_s"),
        "stand_ins": ((9, 1),),
    },
    # Built by the record command from real runs. The search reaches K=16/g=4, tied with K=12/g=4.
    (LIBERO, "diagonal", None): {
        "selected": (16, 4),
        "best_recorded": (12, 4),
        "relative_throughput": pytest.approx(0.9994, abs=0.0001),
        "probes": 7,
        "tuning_s": pytest.approx(501.35, abs=0.01),
        "end_to_end_s": pytest.approx(1156.0, abs=0.1),
        "end_to_end_vs_best": pytest.approx(1.767, abs=0.001),
        "safe": True,
        "gaps": (),
        "unknown_signals": ("queue_wait_p95_ms",),
    },
    (LIBERO, "diagonal", "steady"): {
        "selected": (16, 4),
        "best_recorded": (12, 4),
        "relative_throughput": pytest.approx(0.9994, abs=0.0001),
        "probes": 7,
        "tuning_s": pytest.approx(501.35, abs=0.01),
        "end_to_end_s": pytest.approx(1156.0, abs=0.1),
        "end_to_end_vs_best": pytest.approx(1.767, abs=0.001),
        "safe": True,
        "gaps": (),
        "unknown_signals": ("queue_wait_p95_ms",),
    },
    # two-phase asks for points no search probed: fixed runs stand in until it runs off the record.
    (LIBERO, "two-phase", None): {
        "selected": (9, 1),
        "best_recorded": (12, 4),
        "relative_throughput": pytest.approx(0.5495, abs=0.0001),
        "probes": 5,
        "tuning_s": None,
        "safe": True,
        "gaps": (Gap(9, 2, "configuration"),),
        "unknown_signals": ("queue_wait_p95_ms", "wall_time_s"),
        "stand_ins": ((9, 1),),
    },
    (LIBERO, "two-phase", "steady"): {
        "selected": (8, 4),
        "best_recorded": (12, 4),
        "relative_throughput": pytest.approx(0.9745, abs=0.0001),
        "probes": 7,
        "tuning_s": None,
        "safe": True,
        "gaps": (Gap(8, 5, "configuration"),),
        "unknown_signals": ("queue_wait_p95_ms", "wall_time_s"),
        "stand_ins": ((9, 1), (8, 4)),
    },
    # Built by the record command from real runs. K=16/g=1 runs out of GPU memory, so the search
    # stops at K=8/g=1 and never tries K=10, the best fixed run.
    (ROBOTWIN, "diagonal", None): {
        "selected": (8, 1),
        "best_recorded": (10, 1),
        "relative_throughput": pytest.approx(0.9405, abs=0.0001),
        "probes": 5,
        "tuning_s": pytest.approx(608.20, abs=0.01),
        "end_to_end_s": pytest.approx(2728.7, abs=0.1),
        "end_to_end_vs_best": pytest.approx(1.368, abs=0.001),
        "safe": True,
        "gaps": (),
        "unknown_signals": ("queue_wait_p95_ms",),
    },
    (ROBOTWIN, "diagonal", "steady"): {
        "selected": (8, 1),
        "best_recorded": (10, 1),
        "relative_throughput": pytest.approx(0.9405, abs=0.0001),
        "probes": 5,
        "tuning_s": pytest.approx(608.20, abs=0.01),
        "end_to_end_s": pytest.approx(2728.7, abs=0.1),
        "end_to_end_vs_best": pytest.approx(1.368, abs=0.001),
        "safe": True,
        "gaps": (),
        "unknown_signals": ("queue_wait_p95_ms",),
    },
    (ROBOTWIN, "two-phase", None): {
        "selected": (9, 1),
        "best_recorded": (10, 1),
        "relative_throughput": pytest.approx(0.9762, abs=0.0001),
        "probes": 5,
        "tuning_s": None,
        "safe": True,
        "gaps": (Gap(9, 2, "configuration"),),
        "unknown_signals": ("queue_wait_p95_ms", "wall_time_s"),
        "stand_ins": ((9, 1),),
    },
    (ROBOTWIN, "two-phase", "steady"): {
        "selected": (8, 1),
        "best_recorded": (10, 1),
        "relative_throughput": pytest.approx(0.9405, abs=0.0001),
        "probes": 6,
        "tuning_s": None,
        "safe": True,
        "gaps": (),
        "unknown_signals": ("queue_wait_p95_ms", "wall_time_s"),
        "stand_ins": ((9, 1), (8, 2)),
    },
    (SYNTHETIC, "diagonal", None): {
        "selected": (4, 1),
        "best_recorded": (4, 1),
        "relative_throughput": 1.0,
        "probes": 4,
        "tuning_s": 400.0,
        "safe": True,
        "gaps": (),
        "unknown_signals": ("record_success",),
    },
    (SYNTHETIC, "diagonal", "steady"): {
        "selected": (4, 1),
        "best_recorded": (4, 1),
        "relative_throughput": 1.0,
        "probes": 4,
        "tuning_s": 400.0,
        "safe": True,
        "gaps": (),
        "unknown_signals": ("record_success",),
    },
    (SYNTHETIC, "two-phase", None): {
        "selected": (8, 2),
        "best_recorded": (8, 2),
        "relative_throughput": 1.0,
        "probes": 6,
        "tuning_s": 580.0,
        "safe": True,
        "gaps": (),
        "unknown_signals": ("record_success",),
    },
    (SYNTHETIC, "two-phase", "steady"): {
        "selected": (4, 1),
        "best_recorded": (4, 1),
        "relative_throughput": 1.0,
        "probes": 4,
        "tuning_s": 400.0,
        "safe": True,
        "gaps": (),
        "unknown_signals": ("record_success",),
    },
}


def _load(name: str) -> dict:
    return load_record(WORKLOADS / f"{name}.json")


def _with_measurements(record: dict, *measurements: dict, drop: tuple[int, int] | None = None) -> dict:
    changed = copy.deepcopy(record)
    changed["measurements"] = [
        item for item in changed["measurements"] if (item["k"], item["g"]) != drop
    ] + list(measurements)
    return changed


@pytest.mark.parametrize("estimator", [None, "steady"])
@pytest.mark.parametrize("strategy", autotune_strategies())
@pytest.mark.parametrize("workload", sorted(path.stem for path in WORKLOADS.glob("*.json")))
def test_strategy_on_workload(workload: str, strategy: str, estimator: str | None) -> None:
    expected = EXPECTED.get((workload, strategy, estimator))
    if expected is None:
        pytest.fail(f"add the expected search of {strategy} on {workload} to EXPECTED")
    score = replay_strategy(_load(workload), strategy, throughput_estimator=estimator)
    actual = {name: getattr(score, name) for name in expected}
    assert actual == expected


@pytest.mark.parametrize("workload", sorted(JUDGED))
def test_judged_workloads_pass_the_regression_rule(workload: str) -> None:
    score = replay_strategy(_load(workload), DEFAULT_AUTOTUNE_STRATEGY)
    assert regression_failures(score, JUDGED[workload]) == ()


def test_every_real_workload_is_judged_or_says_why_not() -> None:
    names = [path.stem for path in WORKLOADS.glob("*.json")]
    real = {name for name in names if _load(name)["provenance"]["kind"] == "real"}
    assert real == JUDGED.keys() | NOT_JUDGED.keys()
    assert not JUDGED.keys() & NOT_JUDGED.keys()


def test_the_regression_rule_allows_noise_and_fails_a_larger_drop() -> None:
    score = replace(replay_strategy(_load(FULL_AUTO), "diagonal"), relative_throughput=0.79)
    assert regression_failures(score, 0.8) == ()
    assert "below" in regression_failures(score, 0.81)[0]
    assert regression_failures(score, 0.81, tolerance=0.1) == ()


def test_the_regression_rule_fails_an_unsafe_or_incomplete_search() -> None:
    score = replay_strategy(_load(FULL_AUTO), "diagonal")
    assert "unsafe" in regression_failures(replace(score, safe=False), 0.5)[0]
    assert "unsafe" in regression_failures(replace(score, safe=None), 0.5)[0]
    assert "incomplete" in regression_failures(replace(score, gaps=(Gap(9, 1, "configuration"),)), 0.5)[0]


def test_end_to_end_needs_the_evaluation_size_and_a_fixed_run_of_the_selection() -> None:
    record = copy.deepcopy(_load(FULL_AUTO))
    del record["evaluation_steps"]
    assert replay_strategy(record, "diagonal").end_to_end_s is None

    assert replay_strategy(_load(SYNTHETIC), "two-phase").end_to_end_s is None


def test_fixed_default_scores_the_worker_cap_with_one_replica_per_gpu() -> None:
    # On RoboCasa the fixed default, K=9/g=1, is slower than any search selection.
    default = fixed_default(_load(FULL_AUTO))
    assert (default.k, default.g, default.safe) == (9, 1, True)
    assert default.relative_throughput == pytest.approx(0.6199, abs=0.0001)
    assert default.evaluation_s == pytest.approx(6062.6, abs=0.1)

    record = _load(REAL)
    assert fixed_default(record) == FixedDefault(64, 1, None, None, None)


def test_real_workload_exists_and_synthetic_ones_are_labeled() -> None:
    kinds = {path.stem: load_record(path)["provenance"]["kind"] for path in WORKLOADS.glob("*.json")}
    assert "real" in kinds.values()
    for name, kind in kinds.items():
        assert (kind == "synthetic") == name.startswith("synthetic-")


def test_issue80_record_holds_only_what_the_issue_reports() -> None:
    record = _load(REAL)
    rows = {(item["k"], item["g"]): item for item in record["measurements"]}
    assert {key: item.get("steady_steps_per_s") for key, item in rows.items()} == {
        (1, 1): 24.8,
        (2, 1): 47.3,
        (4, 1): 91.6,
        (8, 1): 147.7,
        (16, 1): None,
    }
    assert {key: item.get("gpu_utilization") for key, item in rows.items()} == {
        (1, 1): 0.07,
        (2, 1): 0.12,
        (4, 1): 0.24,
        (8, 1): 0.33,
        (16, 1): None,
    }
    assert rows[16, 1]["outcome"] == "no_steady_window"
    allowed = {"k", "g", "source", "outcome", "steady_steps_per_s", "gpu_utilization"}
    assert all(item.keys() <= allowed for item in rows.values())


def _search_index(record: dict, autotune: str) -> int:
    return [search["autotune"] for search in record["observed_searches"]].index(autotune)


def test_replay_reproduces_the_real_search_of_this_autotune_code() -> None:
    assert REPRODUCED.keys() == JUDGED.keys()
    for name, code in REPRODUCED.items():
        record = _load(name)
        assert search_mismatches(record, _search_index(record, code)) == (), name


def test_replay_shows_where_another_autotune_code_searched_differently() -> None:
    """The #81 search went on to K=16/g=2; the code here stops after K=16/g=1, as #80 reports."""

    record = _load(FULL_AUTO)
    mismatches = search_mismatches(record, _search_index(record, "#81 81e3824"))
    assert any(item.startswith("probes [(1, 1), (2, 1), (4, 1), (8, 1), (16, 1)]") for item in mismatches)
    assert "selected (8, 1), the search selected (16, 2)" in mismatches


def test_issue80_ends_the_search_before_the_replica_phase_on_real_data() -> None:
    """Known problem, #80: a K probe without a steady window ends the whole search."""

    for name in (REAL, FULL_AUTO):
        score = replay_strategy(_load(name), "diagonal")
        assert score.selected == (8, 1)
        assert score.stop_reason.startswith("no_steady_window")
        assert score.complete


def test_replay_repeats_a_real_search_and_its_wall_time() -> None:
    """The second repeat of each probe is the real search that stopped at K=8/g=1 in 818 s."""

    score = replay_strategy(_load(FULL_AUTO), "diagonal", repeat=1)
    assert score.selected == (8, 1)
    assert score.tuning_s == pytest.approx(817.85, abs=0.01)


def test_replay_ignores_the_cpu_quota_of_this_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = replay_strategy(_load(REAL), "diagonal")
    monkeypatch.setattr(autotune, "cpu_quota", lambda: 1.0)
    assert replay_strategy(_load(REAL), "diagonal") == expected


def test_replay_is_deterministic_and_selects_repeats_by_index() -> None:
    record = _load(SYNTHETIC)
    first = replay_strategy(record, "two-phase", repeat=0)
    assert replay_strategy(record, "two-phase", repeat=0) == first
    assert first.tuning_s == 580.0
    # K=4/g=1 has two measurements; the second one took 10 s longer.
    assert replay_strategy(record, "two-phase", repeat=1).tuning_s == 590.0
    assert replay_strategy(record, "two-phase", repeat=2) == first


def test_any_registered_strategy_runs_unchanged_and_an_unrecorded_probe_is_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def single_step(controller: BenchmarkAutotuneController) -> dict:
        kept = controller.measure_trial(1, 1, ramp="k")
        try:
            controller.measure_trial(3, 1, ramp="k")
        except ProbeStopped as exc:
            assert exc.stop_reason == "unrecorded_configuration"
        return {"k": kept.k, "g": kept.servers.replicas}

    monkeypatch.setitem(autotune._STRATEGIES, "single-step", single_step)
    score = replay_strategy(_load(SYNTHETIC), "single-step")
    assert score.selected == (1, 1)
    assert score.gaps == (Gap(3, 1, "configuration"),)
    assert not score.complete


def test_the_budget_limits_a_search_through_the_replay_clock() -> None:
    record = copy.deepcopy(_load(SYNTHETIC))
    record["manifest_defaults"]["autotune_total_budget_s"] = 250.0
    score = replay_strategy(record, "diagonal")
    assert "total_budget_exhausted" in score.stop_reason
    assert score.selected == (2, 1)
    assert score.tuning_s == 250.0


def test_a_probe_slower_than_the_candidate_timeout_times_out() -> None:
    record = copy.deepcopy(_load(SYNTHETIC))
    record["manifest_defaults"]["autotune_candidate_timeout_s"] = 105.0
    score = replay_strategy(record, "two-phase")
    assert "candidate_timeout" in score.stop_reason
    assert score.selected == (8, 1)
    assert score.tuning_s == 505.0


def test_a_selection_the_record_shows_failing_is_unsafe() -> None:
    record = _with_measurements(
        _load(SYNTHETIC),
        {"k": 8, "g": 2, "source": "probe", "outcome": "worker_oom", "wall_time_s": 10.0},
    )
    score = replay_strategy(record, "two-phase", repeat=0)
    assert score.selected == (8, 2)
    assert score.safe is False
    assert score.best_recorded != (8, 2)


def test_best_recorded_skips_a_configuration_the_record_shows_failing() -> None:
    record = _with_measurements(
        _load(SYNTHETIC),
        {"k": 4, "g": 2, "source": "probe", "outcome": "worker_failure"},
    )
    score = replay_strategy(record, "two-phase")
    assert score.best_recorded == (8, 2)
    assert score.relative_throughput == 1.0


def test_a_fixed_run_does_not_replace_a_search_probe() -> None:
    """A full run can have a steady window at a K where a probe has none."""

    fixed = {"k": 16, "g": 1, "source": "fixed-run", "outcome": "completed", "steady_steps_per_s": 200.0}
    score = replay_strategy(_with_measurements(_load(REAL), fixed), "diagonal")
    assert "no_steady_window" in score.stop_reason
    assert score.selected == (8, 1)
    assert score.stand_ins == ()


def test_a_score_compares_fixed_runs_only_with_fixed_runs() -> None:
    fixed = {"k": 4, "g": 2, "source": "fixed-run", "outcome": "completed", "steps_per_s": 500.0}
    record = _load(SYNTHETIC)
    record["measurements"] = [item for item in record["measurements"] if item["source"] == "probe"]

    without_selection = replay_strategy(_with_measurements(record, fixed), "two-phase")
    assert without_selection.best_recorded == (8, 2)

    selected = {**fixed, "k": 8, "g": 2, "steps_per_s": 250.0}
    with_selection = replay_strategy(_with_measurements(record, fixed, selected), "two-phase")
    assert with_selection.best_recorded == (4, 2)
    assert with_selection.relative_throughput == 0.5


def test_a_fixed_run_answers_an_unprobed_point_as_a_visible_stand_in() -> None:
    fixed = {"k": 16, "g": 1, "source": "fixed-run", "outcome": "completed", "steady_steps_per_s": 220.0}
    score = replay_strategy(_with_measurements(_load(REAL), fixed, drop=(16, 1)), "diagonal")
    assert (16, 1) in score.stand_ins


def test_recorded_episodes_drive_the_correctness_check() -> None:
    def episodes(passed: int) -> dict[str, bool]:
        return {f"episode-{index}": index < passed for index in range(10)}

    record = copy.deepcopy(_load(SYNTHETIC))
    for item in record["measurements"]:
        if (item["k"], item["g"]) == (1, 1):
            item["record_success"] = episodes(10)
        if (item["k"], item["g"]) == (2, 1):
            item["record_success"] = episodes(5)
    score = replay_strategy(record, "diagonal")
    assert "correctness_regression" in score.stop_reason
    assert score.selected == (1, 1)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda record: record.update(schema_version=0), "schema_version"),
        (lambda record: record["provenance"].update(kind="estimated"), "provenance.kind"),
        (lambda record: record["workload"]["host"].pop("gpu_model"), "gpu_model"),
        (lambda record: record.pop("unmeasured"), "unmeasured"),
        (lambda record: record["unknown_fields"].remove("steps_per_s"), "unknown_fields"),
        (lambda record: record["measurements"][0].update(outcome="fine"), "unknown outcome"),
        (lambda record: record["measurements"][0].pop("steady_steps_per_s"), "steady_steps_per_s"),
        (lambda record: record["measurements"][4].update(steady_steps_per_s=1.0), "steady_steps_per_s"),
        (lambda record: record["provenance"].update(source="/root/rc39/runs/full-auto"), "host path"),
        (lambda record: record["provenance"]["notes"].append("see ~/runs/a"), "host path"),
        (lambda record: record["measurements"][0].update(record_success={"C:\\run\\1": True}), "host path"),
    ],
)
def test_validation_rejects_a_bad_record(change, message: str) -> None:
    record = copy.deepcopy(_load(REAL))
    change(record)
    with pytest.raises(ValueError, match=message):
        validate_record(record)


class _Clock:
    now = 0.0

    def __call__(self) -> float:
        return self.now


def _probe_output(k: int, g: int, root: Path, *, elapsed: float, steady: float | None) -> dict:
    return {
        "runtime": {"output_dir": str(root), "concurrency": k, "replicas": g},
        "inference": {"command": [str(root / "serve.py"), "--token", "secret-token"]},
        "benchmark": {"benchmark": "fake", "suite": "suite", "bound_hint": "cpu"},
        "item_timeline": {"completion_s": {f"item-{index}": 1.0 for index in range(3 * k)}},
        "record_success": {f"episode-{index}": True for index in range(3 * k)},
        "result": {
            "steps_per_s": elapsed,
            **({} if steady is None else {"steady_steps_per_s": steady, "steady_window": {"workers": k}}),
            "episodes": 3 * k,
            "successes": 3 * k,
            "success_rate": 1.0,
            "total_episodes": 3 * k,
        },
        "resources": {
            "cpu": {"cpu_cores": k * 0.5, "throttle_time_ratio": 0.0},
            "gpu": {"mean_utilization": 0.3, "peak_memory_bytes": g * 3 * 1024**3},
        },
        "recovery": {
            "worker_failures": [],
            "oom_events": 0,
            "initial_workers_per_gpu": {"0": k},
            "final_workers_per_gpu": {"0": k},
        },
        "inference_metrics": {"queue_wait_ms": {"p95": 2.0}},
    }


def _write_run(root: Path, bounds: AutotuneBounds, *, oom_at: tuple[int, int] | None = None) -> None:
    """Write what a real search leaves in its output directory, through the real controller."""

    clock = _Clock()
    rates = {
        (1, 1): (8.0, 10.0),
        (2, 1): (16.0, 20.0),
        (4, 1): (18.0, 22.0),
        (4, 2): (50.0, None),
        (6, 3): (50.5, 55.0),
    }

    def measure(k: int, g: int, _limits, _progress):
        clock.now += 10.0
        if (k, g) == oom_at:
            raise ProbeCapacityReached(
                f"worker died under {root}",
                stop_reason="worker_oom",
                health={"requested_workers": k, "oom_events": 1, "detail": str(root)},
            )
        elapsed, steady = rates[k, g]
        output = _probe_output(k, g, root, elapsed=elapsed, steady=steady)
        directory = root / "autotune" / f"candidate-k{k}-g{g}"
        directory.mkdir(parents=True)
        (directory / "result.json").write_text(json.dumps(output), encoding="utf-8")
        return output, 3 * 1024**3

    BenchmarkAutotuneController(
        measure=measure,
        output_dir=root / "autotune",
        bounds=bounds,
        limits=AutotuneLimits(),
        clock=clock,
    ).run()


def _record(tmp_path: Path, bounds: AutotuneBounds) -> dict:
    _write_run(tmp_path / "first", bounds)
    _write_run(tmp_path / "second", bounds, oom_at=(4, 1))
    sweep = tmp_path / "sweep"
    sweep.mkdir()
    output = _probe_output(5, 2, sweep, elapsed=70.0, steady=75.0)
    (sweep / "result.json").write_text(json.dumps(output), encoding="utf-8")
    return record_runs(
        [tmp_path / "first", tmp_path / "second"],
        [sweep],
        source="a test run",
        policy="fake",
        gpu_model="fake",
        kit_version="0",
        date="2026-11-05",
        autotune=["first", "second"],
    )


def _resolved_bounds() -> AutotuneBounds:
    return resolve_bounds(
        1,
        {"vram_per_gpu_bytes": 24 * 1024**3, "maximum_workers": 12, "maximum_replicas": 4},
        cpu_cores=32.0,
    )


def test_record_from_run_outputs_replays_the_same_search(tmp_path: Path) -> None:
    record = _record(tmp_path, _resolved_bounds())
    validate_record(record)
    assert record["provenance"]["kind"] == "real"
    assert record["workload"]["benchmark"] == "fake"
    assert record["bound_hint"] == "cpu"
    assert record["replica_vram_bytes"] == 3 * 1024**3
    assert [search["autotune"] for search in record["observed_searches"]] == ["first", "second"]
    assert record["observed_searches"][0]["selected"] == {"k": 2, "g": 1}
    assert [search_mismatches(record, index) for index in (0, 1)] == [(), ()]
    score = replay_strategy(record, "diagonal")
    assert score.selected == (2, 1)
    assert score.tuning_s == 40.0
    assert score.gaps == ()
    assert score.unknown_signals == ()


def test_record_keeps_how_each_probe_ended_and_marks_repeats(tmp_path: Path) -> None:
    record = _record(tmp_path, _resolved_bounds())
    ended: dict[tuple[int, int], list[tuple[str, str]]] = {}
    for item in record["measurements"]:
        ended.setdefault((item["k"], item["g"]), []).append((item["source"], item["outcome"]))
    assert ended[4, 2] == [("probe", "no_steady_window")]
    assert ended[2, 1] == [("probe", "completed")] * 2
    # The first run's strategy rejected K=4 with a free-text reason, but the probe finished.
    assert ended[4, 1] == [("probe", "completed"), ("probe", "worker_oom")]
    assert ended[5, 2] == [("fixed-run", "completed")]
    fixed = next(item for item in record["measurements"] if item["source"] == "fixed-run")
    assert "wall_time_s" not in fixed
    assert "wall_time_s" in record["unknown_fields"]


def test_record_resolves_to_the_same_bounds(tmp_path: Path) -> None:
    bounds = _resolved_bounds()
    record = _record(tmp_path, bounds)
    host = record["workload"]["host"]
    defaults = {"vram_per_gpu_bytes": host["vram_per_gpu_bytes"], **record["manifest_defaults"]}
    assert resolve_bounds(host["gpu_count"], defaults, cpu_cores=host["cpu_cores"]) == bounds


def test_record_leaves_out_paths_commands_and_messages(tmp_path: Path) -> None:
    text = json.dumps(_record(tmp_path, _resolved_bounds()))
    assert str(tmp_path) not in text
    assert "secret-token" not in text
    assert "worker died" not in text


def test_record_refuses_a_search_that_did_not_complete(tmp_path: Path) -> None:
    run = tmp_path / "run" / "autotune"
    run.mkdir(parents=True)
    (run / "selection.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    with pytest.raises(ValueError, match="no complete search"):
        record_runs(
            [tmp_path / "run"],
            source="a test run",
            policy="fake",
            gpu_model="fake",
            kit_version="0",
            date="2026-11-05",
            autotune=["main"],
        )


def test_score_command_prints_what_the_record_does_not_cover(capsys: pytest.CaptureFixture) -> None:
    assert main(["score", str(WORKLOADS / f"{REAL}.json")]) == 0
    printed = capsys.readouterr().out
    assert "not measured: Every point with g > 1" in printed
    assert "fixed default K=64/g=1, no tuning: no fixed run recorded" in printed
    assert "diagonal [steady]: selected K=8/g=1" in printed
    assert "INCOMPLETE, ran off the record at K=1/g=1 (steps_per_s)" in printed
    assert "tuning time unknown of 1800 s" in printed
    assert "two-phase [elapsed]: selected none" in printed


def test_record_command_writes_a_record_that_loads(tmp_path: Path) -> None:
    _write_run(tmp_path / "run", _resolved_bounds())
    output = tmp_path / "workload.json"
    arguments = ["--source", "a test run", "--policy", "fake", "--gpu-model", "fake"]
    arguments += ["--kit-version", "0", "--date", "2026-11-05", "--output", str(output)]
    assert main(["record", "--run", str(tmp_path / "run"), "--autotune", "main", *arguments]) == 0
    record = load_record(output)
    assert {item["search"] for item in record["measurements"] if item["source"] == "probe"} == {0}
    assert search_mismatches(record, 0) == ()


def test_record_needs_the_autotune_code_of_each_run(tmp_path: Path) -> None:
    _write_run(tmp_path / "run", _resolved_bounds())
    with pytest.raises(ValueError, match="autotune code of each run"):
        record_runs(
            [tmp_path / "run"],
            source="a test run",
            policy="fake",
            gpu_model="fake",
            kit_version="0",
            date="2026-11-05",
            autotune=[],
        )


def test_validate_refuses_a_probe_of_an_unknown_search() -> None:
    record = copy.deepcopy(_load(FULL_AUTO))
    record["measurements"][0]["search"] = 2
    with pytest.raises(ValueError, match="no observed search"):
        validate_record(record)


def test_score_command_says_which_recorded_searches_this_code_reproduces(
    capsys: pytest.CaptureFixture,
) -> None:
    assert main(["score", str(WORKLOADS / f"{FULL_AUTO}.json"), "--strategy", "diagonal"]) == 0
    printed = capsys.readouterr().out
    assert "observed search 1 (main 5f79ce2): reproduced by this code" in printed
    assert "observed search 0 (#81 81e3824): NOT reproduced" in printed
