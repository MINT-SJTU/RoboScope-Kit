# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Offline testbed for autotune strategies: record real probes, replay them, score the search.

A workload record holds what probes at K/g returned on one host. Replay feeds a record to the
unchanged :class:`BenchmarkAutotuneController` through its ``measure`` and ``clock`` seams, so any
registered strategy runs without a GPU, deterministically, in milliseconds.

This is a development tool: an evaluation never imports it. The format and how to add a workload
are in ``docs/autotune-testbed.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .autotune import (
    AutotuneBounds,
    BenchmarkAutotuneController,
    ProbeCapacityReached,
    ProbeCorrectnessRegression,
    ProbeLimits,
    ProbeProgressFn,
    ProbeStopped,
    ProbeTimedOut,
    autotune_strategies,
    resolve_bounds,
    resolve_limits,
)

SCHEMA_VERSION = 1
# What a record says about each search it was built from, so a replay can be checked against it.
OBSERVED_SEARCH_FIELDS = (
    "autotune",
    "strategy",
    "throughput_estimator",
    "path",
    "selected",
    "stop_reason",
    "elapsed_s",
)
KINDS = ("real", "synthetic")
# ``probe`` came from a search. ``fixed-run`` came from ``--concurrency K --replicas G``, which runs
# the full workload, so a search at the same K/g would not have measured the same thing.
SOURCES = ("probe", "fixed-run")
# The probe ran to the end, with or without a steady-state window.
FINISHED_OUTCOMES = ("completed", "no_steady_window")
# The record shows the configuration failing. A selection that lands on one of these is unsafe.
FAILED_OUTCOMES = (
    "candidate_timeout",
    "capacity_reached",
    "correctness_regression",
    "incomplete_probe",
    "policy_replica_failure",
    "worker_failure",
    "worker_loss",
    "worker_oom",
    "worker_shortfall",
)
# The starting tolerance of a workload, until repeated fixed runs give it its own.
NOISE_TOLERANCE = 0.05
# Fixed runs of each configuration that a workload's own tolerance needs, and the standard
# deviations it allows for 1% false alerts on one side.
TOLERANCE_REPEATS = 3
TOLERANCE_Z = 2.576
# How far a replay's search time may be from the recorded search's. The replay adds only probe
# wall times, so the rest is what the real search spent between probes.
SEARCH_TIME_TOLERANCE = 0.02
# A finished measurement should hold all of these; the record must list the ones it does not.
TRACKED_FIELDS = (
    "cpu_cores",
    "gpu_utilization",
    "health",
    "peak_memory_bytes",
    "probe_items",
    "queue_wait_p95_ms",
    "record_success",
    "steady_window",
    "steps_per_s",
    "throttle_time_ratio",
    "wall_time_s",
)
# Signals a replayed probe needs a value for. An unknown one reads as "no limit reached".
REPLAY_SIGNALS = (
    "cpu_cores",
    "gpu_utilization",
    "peak_memory_bytes",
    "queue_wait_p95_ms",
    "record_success",
    "steps_per_s",
    "throttle_time_ratio",
    "wall_time_s",
)
# An absolute POSIX or Windows path, or a home-relative one; URLs do not match.
_HOST_PATH = re.compile(r"(?<![\w./:])/[\w.-]+/|(?<!\w)~/|\b[A-Za-z]:\\")


class UnrecordedProbe(ProbeStopped):
    """The record cannot answer this probe, so the search stops here, visibly."""

    status = "unrecorded"

    def __init__(self, k: int, g: int, missing: str) -> None:
        super().__init__(
            f"the record has no {missing} for K={k}, g={g}",
            stop_reason=f"unrecorded_{missing}",
        )


@dataclass(frozen=True)
class Gap:
    """A probe the record could not answer. ``missing`` is ``configuration`` or a field name."""

    k: int
    g: int
    missing: str


@dataclass(frozen=True)
class Score:
    """How one strategy did on one workload. Read ``relative_throughput`` with ``gaps``."""

    workload: str
    kind: str
    strategy: str
    throughput_estimator: str
    selected: tuple[int, int] | None
    # Best recorded configuration that the record does not show failing, on the rate that
    # ``relative_throughput`` compares. A record that stops at g=1 only has a best g=1 point.
    best_recorded: tuple[int, int] | None
    relative_throughput: float | None
    probes: int
    # Every K/g the strategy asked for, in order, including one the record could not answer.
    path: tuple[tuple[int, int], ...]
    # ``None`` when a probe had no recorded wall time.
    tuning_s: float | None
    budget_s: float
    # Tuning time plus the evaluation at the selection's fixed-run rate, and that total over the
    # evaluation alone at the best recorded rate. ``None`` when the record lacks an input.
    end_to_end_s: float | None
    end_to_end_vs_best: float | None
    # ``None`` when nothing was selected.
    safe: bool | None
    gaps: tuple[Gap, ...]
    # Signals of the replayed probes that the record does not hold.
    unknown_signals: tuple[str, ...]
    # Configurations answered from a fixed run because no search probe was recorded.
    stand_ins: tuple[tuple[int, int], ...]
    stop_reason: str

    @property
    def complete(self) -> bool:
        """Whether the search ran to its own end, without running off the record."""

        return not self.gaps


@dataclass(frozen=True)
class FixedDefault:
    """The trivial baseline: no tuning, ``K`` at the CPU worker cap, one replica per GPU."""

    k: int
    g: int
    # ``None`` when the record has no finished rate for this K/g.
    relative_throughput: float | None
    evaluation_s: float | None
    # ``None`` when the record has no measurement of this K/g, as for a selection.
    safe: bool | None


def validate_record(record: dict[str, Any]) -> None:
    """Raise ``ValueError`` unless ``record`` is well formed and carries no host paths."""

    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"record schema_version must be {SCHEMA_VERSION}")
    provenance = record.get("provenance", {})
    if provenance.get("kind") not in KINDS:
        raise ValueError(f"record provenance.kind must be one of {KINDS}")
    if not provenance.get("source"):
        raise ValueError("record provenance.source is required")
    workload = record.get("workload", {})
    for field in ("benchmark", "suite", "policy", "kit_version", "date", "host"):
        if not workload.get(field):
            raise ValueError(f"record workload.{field} is required")
    for field in ("gpu_model", "gpu_count", "vram_per_gpu_bytes", "cpu_cores"):
        if not workload["host"].get(field):
            raise ValueError(f"record workload.host.{field} is required")
    if not record.get("unmeasured"):
        raise ValueError("record unmeasured must say which K/g were not measured")
    steps = record.get("evaluation_steps", 1)
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("record evaluation_steps must be a positive integer")
    measurements = record.get("measurements")
    if not measurements:
        raise ValueError("record has no measurements")
    searches = record.get("observed_searches", [])
    for index, search in enumerate(searches):
        missing = [field for field in OBSERVED_SEARCH_FIELDS if field not in search]
        if missing:
            raise ValueError(f"observed_searches[{index}] lacks {missing}")
    absent = set()
    for measurement in measurements:
        absent |= _validate_measurement(measurement)
        search = measurement.get("search")
        if search is not None and (measurement["source"] != "probe" or search not in range(len(searches))):
            raise ValueError(
                f"measurement K={measurement['k']}, g={measurement['g']} names search {search!r}, "
                "which is no observed search, or the measurement is no probe"
            )
    listed = set(record.get("unknown_fields", ()))
    if listed != absent:
        raise ValueError(f"record unknown_fields must be {sorted(absent)}, not {sorted(listed)}")
    for text in _strings(record):
        if _HOST_PATH.search(text):
            raise ValueError(f"record contains a host path: {text!r}")


def _validate_measurement(measurement: dict[str, Any]) -> set[str]:
    k, g = measurement.get("k"), measurement.get("g")
    if not isinstance(k, int) or not isinstance(g, int) or k < 1 or g < 1:
        raise ValueError(f"measurement needs positive integer k and g: {measurement}")
    if measurement.get("source") not in SOURCES:
        raise ValueError(f"measurement K={k}, g={g} source must be one of {SOURCES}")
    outcome = measurement.get("outcome")
    if outcome not in (*FINISHED_OUTCOMES, *FAILED_OUTCOMES):
        raise ValueError(f"measurement K={k}, g={g} has unknown outcome {outcome!r}")
    # An absent steady rate must mean "no window", never "not reported": the controller reads it so.
    has_steady = measurement.get("steady_steps_per_s") is not None
    if outcome == "completed" and not has_steady:
        raise ValueError(f"completed measurement K={k}, g={g} needs steady_steps_per_s")
    if outcome == "no_steady_window" and has_steady:
        raise ValueError(f"no_steady_window measurement K={k}, g={g} cannot have steady_steps_per_s")
    if outcome in FAILED_OUTCOMES:
        return set()
    expected = set(TRACKED_FIELDS) - ({"steady_window"} if outcome == "no_steady_window" else set())
    return {field for field in expected if measurement.get(field) is None}


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def load_record(path: Path) -> dict[str, Any]:
    """Read and validate a workload record."""

    record = json.loads(path.read_text(encoding="utf-8"))
    validate_record(record)
    return record


def record_runs(
    runs: Sequence[Path],
    sweeps: Sequence[Path] = (),
    *,
    source: str,
    policy: str,
    gpu_model: str,
    kit_version: str,
    date: str,
    autotune: Sequence[str],
    evaluation_steps: int | None = None,
) -> dict[str, Any]:
    """Build a real workload record from existing evaluation outputs.

    ``runs`` are ``--output-dir`` directories of completed searches; each one adds a measurement
    per probe, so running a search twice records repeats. ``autotune`` names, per run, the
    autotune code it ran, such as ``main 5f79ce2``, so a replay of that code can be checked against
    the run. ``sweeps`` are ``--output-dir`` directories of ``--concurrency K --replicas G``
    evaluations. Only the named fields below are copied, so paths, commands, and error messages
    of the recording host stay out of the record.
    """

    if not runs:
        raise ValueError("a record needs at least one autotune run")
    if len(autotune) != len(runs):
        raise ValueError(f"name the autotune code of each run: {len(runs)} runs, {len(autotune)} names")
    selections = [_read_json(run / "autotune" / "selection.json") for run in runs]
    for run, selection in zip(runs, selections, strict=True):
        if selection.get("status") != "complete":
            raise ValueError(f"{run} holds no complete search: status {selection.get('status')!r}")
    measurements = [
        {**measurement, "search": index}
        for index, (run, selection) in enumerate(zip(runs, selections, strict=True))
        for measurement in _probe_measurements(run, selection)
    ]
    outputs = [_read_json(sweep / "result.json") for sweep in sweeps]
    measurements += [_fixed_measurement(output) for output in outputs]
    selection = selections[0]
    bounds, limits = selection["bounds"], selection["limits"]
    metadata = _first_metadata(runs[0], selection)
    measured = {
        "schema_version": SCHEMA_VERSION,
        "provenance": {"kind": "real", "source": source, "notes": []},
        **({} if evaluation_steps is None else {"evaluation_steps": evaluation_steps}),
        "workload": {
            "benchmark": metadata["benchmark"],
            "suite": metadata["suite"],
            "policy": policy,
            "kit_version": kit_version,
            "date": date,
            "host": {
                "gpu_model": gpu_model,
                "gpu_count": bounds["gpu_count"],
                "vram_per_gpu_bytes": bounds["vram_per_gpu_bytes"],
                "cpu_cores": bounds["cpu_cores"],
            },
        },
        "bound_hint": metadata.get("bound_hint"),
        "replica_vram_bytes": bounds["replica_vram_bytes"],
        # The resolved bounds, written as the manifest values that resolve to them again.
        "manifest_defaults": {
            "worker_cpu_cores": bounds["worker_cpu_cores"],
            "maximum_workers": bounds["max_workers_hard"],
            "maximum_replicas": bounds["max_replicas"],
            "autotune_candidate_timeout_s": limits["candidate_timeout_s"],
            "autotune_total_budget_s": limits["total_budget_s"],
            "autotune_maximum_success_rate_drop": limits["maximum_success_rate_drop"],
        },
        "observed_searches": [
            _observed_search(name, observed) for name, observed in zip(autotune, selections, strict=True)
        ],
        "measurements": measurements,
    }
    measured["unknown_fields"] = sorted(
        {field for item in measurements for field in _validate_measurement(item)}
    )
    measured["unmeasured"] = "Every K/g that no search probed and no fixed run covered."
    validate_record(measured)
    return measured


def _observed_search(autotune: str, selection: dict[str, Any]) -> dict[str, Any]:
    limits = selection["limits"]
    return {
        "autotune": autotune,
        "strategy": limits["strategy"],
        "throughput_estimator": limits["throughput_estimator"],
        "path": [[candidate["k"], candidate["g"]] for candidate in selection["candidates"]],
        "selected": {"k": selection["k"], "g": selection["g"]},
        "stop_reason": stop_summary(selection["stop_reason"]),
        "elapsed_s": selection["elapsed_s"],
    }


_STOP_CODE = re.compile(r"^([a-z_]+): ")


def stop_summary(stop_reason: str) -> str:
    """A stop reason without the messages a probe raised, which can hold host paths.

    Each ``; ``-separated part that starts with a code, as ``worker_oom: ...`` does, is cut to the
    code. A part without one, such as a predicted GPU memory limit, is computed from the bounds and
    the probe results, so a replay gives the same text.
    """

    parts = [part for part in stop_reason.split("; ") if part]
    return "; ".join(
        match.group(1) if (match := _STOP_CODE.match(part)) else part for part in parts
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_metadata(run: Path, selection: dict[str, Any]) -> dict[str, Any]:
    for candidate in selection["candidates"]:
        path = _candidate_result(run, candidate)
        if path.is_file():
            return _read_json(path)["benchmark"]
    raise ValueError(f"{run} has no candidate result.json")


def _candidate_result(run: Path, candidate: dict[str, Any]) -> Path:
    return run / "autotune" / f"candidate-k{candidate['k']}-g{candidate['g']}" / "result.json"


def _probe_measurements(run: Path, selection: dict[str, Any]) -> list[dict[str, Any]]:
    measurements = []
    for candidate in selection["candidates"]:
        outcome = _candidate_outcome(candidate)
        if outcome is None:
            continue
        path = _candidate_result(run, candidate)
        measurement: dict[str, Any] = {
            "k": candidate["k"],
            "g": candidate["g"],
            "source": "probe",
            "outcome": outcome,
            "wall_time_s": candidate["elapsed_s"],
        }
        if "health" in candidate:
            measurement["health"] = {
                key: value
                for key, value in candidate["health"].items()
                if isinstance(value, int | float)
            }
        if outcome in FINISHED_OUTCOMES and not path.is_file():
            raise ValueError(f"{path} is missing for a finished probe")
        if path.is_file():
            measurement.update(_output_fields(_read_json(path)))
        if outcome == "completed" and "steady_steps_per_s" not in measurement:
            measurement["outcome"] = "no_steady_window"
        measurements.append(measurement)
    return measurements


def _candidate_outcome(candidate: dict[str, Any]) -> str | None:
    """How a probe ended, or ``None`` when its result says nothing about the configuration.

    A probe the strategy measured and then did not keep is ``rejected`` with a free-text reason,
    but the probe itself finished. ``total_budget_exhausted`` and ``error`` describe the search
    and the harness, so such a K/g stays unrecorded.
    """

    status = candidate["status"]
    reason = candidate.get("stop_reason")
    if status == "accepted":
        return "completed"
    if status == "timed_out":
        return "candidate_timeout" if reason == "candidate_timeout" else None
    if status != "rejected":
        return None
    return reason if reason in ("no_steady_window", *FAILED_OUTCOMES) else "completed"


def _fixed_measurement(output: dict[str, Any]) -> dict[str, Any]:
    """A fixed run ran the full workload, so its wall time says nothing about a probe's."""

    runtime = output["runtime"]
    fields = _output_fields(output)
    return {
        "k": runtime["concurrency"],
        "g": runtime["replicas"],
        "source": "fixed-run",
        "outcome": "completed" if "steady_steps_per_s" in fields else "no_steady_window",
        **fields,
    }


def _output_fields(output: dict[str, Any]) -> dict[str, Any]:
    result = output["result"]
    resources = output["resources"]
    queue_wait = output.get("inference_metrics", {}).get("queue_wait_ms") or {}
    fields = {
        "steps_per_s": result.get("steps_per_s"),
        "steady_steps_per_s": result.get("steady_steps_per_s"),
        "steady_window": result.get("steady_window"),
        "cpu_cores": resources["cpu"].get("cpu_cores"),
        "throttle_time_ratio": resources["cpu"].get("throttle_time_ratio"),
        "gpu_utilization": resources["gpu"].get("mean_utilization"),
        "peak_memory_bytes": resources["gpu"].get("peak_memory_bytes"),
        "queue_wait_p95_ms": queue_wait.get("p95"),
        "probe_items": len(output.get("item_timeline", {}).get("completion_s", {})) or None,
        "record_success": output.get("record_success") or None,
    }
    return {key: value for key, value in fields.items() if value is not None}


class WorkloadReplay:
    """A ``measure`` and ``clock`` that answer probes from a record.

    Search probes take precedence over fixed runs at a K/g: a fixed run can have a steady window at a K
    where a probe has none, which is the very thing a search must see. ``search`` answers each K/g
    with the probe of that observed search where it has one. Otherwise ``repeat`` selects among
    repeated measurements of a K/g, wrapping around, so a replay is deterministic.
    """

    def __init__(
        self,
        record: dict[str, Any],
        *,
        throughput_estimator: str,
        repeat: int = 0,
        search: int | None = None,
    ) -> None:
        self.record = record
        self.throughput_estimator = throughput_estimator
        self.repeat = repeat
        self.search = search
        self.path: list[tuple[int, int]] = []
        self.now = 0.0
        self.gaps: list[Gap] = []
        self.used: list[dict[str, Any]] = []
        self.stand_ins: list[tuple[int, int]] = []

    def clock(self) -> float:
        return self.now

    def measure(
        self,
        k: int,
        g: int,
        limits: ProbeLimits,
        _progress: ProbeProgressFn,
    ) -> tuple[dict[str, Any], int | None]:
        self.path.append((k, g))
        measurement = self._find(k, g)
        wall_s = measurement.get("wall_time_s", 0.0)
        if wall_s >= limits.timeout_s:
            self.now += limits.timeout_s
            raise ProbeTimedOut(
                f"probe K={k}, g={g} exceeded {limits.timeout_s:.3f}s",
                stop_reason=limits.timeout_reason,
                health=measurement.get("health"),
            )
        self.now += wall_s
        outcome = measurement["outcome"]
        message = f"recorded probe K={k}, g={g} ended with {outcome}"
        health = measurement.get("health", {})
        if outcome == "candidate_timeout":
            raise ProbeTimedOut(message, stop_reason=outcome, health=health)
        if outcome == "correctness_regression":
            raise ProbeCorrectnessRegression(message, health=health)
        if outcome in FAILED_OUTCOMES:
            raise ProbeCapacityReached(message, stop_reason=outcome, health=health)
        return self._output(k, measurement), self.record.get("replica_vram_bytes")

    def _find(self, k: int, g: int) -> dict[str, Any]:
        recorded = [item for item in self.record["measurements"] if (item["k"], item["g"]) == (k, g)]
        if not recorded:
            self.gaps.append(Gap(k, g, "configuration"))
            raise UnrecordedProbe(k, g, "configuration")
        probes = [item for item in recorded if item["source"] == "probe"]
        if not probes:
            self.stand_ins.append((k, g))
        own = [item for item in probes if self.search is not None and item.get("search") == self.search]
        options = own or probes or recorded
        measurement = options[0] if own else options[self.repeat % len(options)]
        finished = measurement["outcome"] in FINISHED_OUTCOMES
        if finished and self.throughput_estimator == "elapsed" and "steps_per_s" not in measurement:
            self.gaps.append(Gap(k, g, "steps_per_s"))
            raise UnrecordedProbe(k, g, "steps_per_s")
        self.used.append(measurement)
        return measurement

    def _output(self, k: int, measurement: dict[str, Any]) -> dict[str, Any]:
        """Rebuild the probe output the controller reads.

        A finished probe had all K Workers, so health follows from its outcome. A signal the
        record does not hold reads as "no limit reached" and is reported in the score; the
        elapsed rate reads as NaN, which :func:`replay_strategy` reports if a decision used it.
        Aggregate success is a constant that cannot regress: correctness is decided by
        ``record_success`` or by a recorded ``correctness_regression``.
        """

        result: dict[str, Any] = {
            "steps_per_s": measurement.get("steps_per_s", math.nan),
            "episodes": 0,
            "total_episodes": 0,
            "successes": 0,
            "success_rate": 1.0,
        }
        if "steady_steps_per_s" in measurement:
            result["steady_steps_per_s"] = measurement["steady_steps_per_s"]
        hint = self.record.get("bound_hint")
        return {
            "benchmark": {} if hint is None else {"bound_hint": hint},
            "result": result,
            "resources": {
                "cpu": {
                    "cpu_cores": measurement.get("cpu_cores", 0.0),
                    "throttle_time_ratio": measurement.get("throttle_time_ratio", 0.0),
                },
                "gpu": {
                    "mean_utilization": measurement.get("gpu_utilization", 0.0),
                    "peak_memory_bytes": measurement.get("peak_memory_bytes", 0),
                },
            },
            "recovery": {
                "worker_failures": [],
                "oom_events": 0,
                "initial_workers_per_gpu": {"0": k},
                "final_workers_per_gpu": {"0": k},
            },
            "inference_metrics": {"queue_wait_ms": {"p95": measurement.get("queue_wait_p95_ms", 0.0)}},
            "record_success": measurement.get("record_success", {}),
        }


def replay_strategy(
    record: dict[str, Any],
    strategy: str,
    *,
    throughput_estimator: str | None = None,
    repeat: int = 0,
    search: int | None = None,
) -> Score:
    """Run ``strategy`` on a record through the real controller and score its search.

    ``throughput_estimator`` overrides the strategy's own, as the manifest's
    ``autotune_throughput_estimator`` does. ``search`` replays on the probes of that observed
    search where it has them. A search the record cannot start returns a score without a
    selection.
    """

    defaults = {**_manifest_defaults(record), "autotune_strategy": strategy}
    if throughput_estimator is not None:
        defaults["autotune_throughput_estimator"] = throughput_estimator
    bounds = _bounds(record, defaults)
    limits = resolve_limits(defaults)
    replay = WorkloadReplay(
        record, throughput_estimator=limits.throughput_estimator, repeat=repeat, search=search
    )
    with tempfile.TemporaryDirectory(prefix="autotune-replay-") as output_dir:
        controller = BenchmarkAutotuneController(
            measure=replay.measure,
            output_dir=Path(output_dir),
            bounds=bounds,
            limits=limits,
            clock=replay.clock,
        )
        try:
            result = controller.run()
        except RuntimeError as exc:
            result = {"stop_reason": str(exc)}
    if controller.elapsed_fallbacks:
        replay.gaps += [
            Gap(*key, "steps_per_s")
            for key, rate in controller.elapsed_throughput.items()
            if math.isnan(rate)
        ]
    return _score(record, strategy, limits.throughput_estimator, limits.total_budget_s, result, replay)


def search_mismatches(record: dict[str, Any], index: int) -> tuple[str, ...]:
    """How a replay of observed search ``index`` differs from what the real search did.

    Empty when the replay asks for the same K/g in the same order, stops for the same reason,
    selects the same K/g, and takes the same time within ``SEARCH_TIME_TOLERANCE``. Only the
    autotune code the search ran, named by its ``autotune`` field, can be expected to match.
    """

    observed = record["observed_searches"][index]
    score = replay_strategy(
        record,
        observed["strategy"],
        throughput_estimator=observed["throughput_estimator"],
        search=index,
    )
    expected_path = tuple((k, g) for k, g in observed["path"])
    expected_selected = (observed["selected"]["k"], observed["selected"]["g"])
    mismatches = []
    if score.path != expected_path:
        mismatches.append(f"probes {list(score.path)}, the search probed {list(expected_path)}")
    if score.selected != expected_selected:
        mismatches.append(f"selected {score.selected}, the search selected {expected_selected}")
    if stop_summary(score.stop_reason) != observed["stop_reason"]:
        mismatches.append(
            f"stopped for {stop_summary(score.stop_reason)!r}, the search for {observed['stop_reason']!r}"
        )
    if score.gaps:
        mismatches.append(f"ran off the record at {list(score.gaps)}")
    elapsed = observed["elapsed_s"]
    if score.tuning_s is None or abs(score.tuning_s - elapsed) > SEARCH_TIME_TOLERANCE * elapsed:
        took = "an unknown time" if score.tuning_s is None else f"{score.tuning_s:.1f} s"
        mismatches.append(f"took {took}, the search {elapsed:.1f} s")
    return tuple(mismatches)


def fixed_default(record: dict[str, Any]) -> FixedDefault:
    """Score the configuration a user gets without any search, on fixed-run rates only."""

    bounds = _bounds(record, _manifest_defaults(record))
    key = (bounds.max_workers, bounds.gpu_count)
    rates = _median_rates(record, "steps_per_s", "fixed-run")
    eligible = [item for item in rates if not _failed(record, *item)]
    relative, evaluation_s = None, None
    if key in rates and eligible:
        relative = rates[key] / max(rates[item] for item in eligible)
        if "evaluation_steps" in record:
            evaluation_s = record["evaluation_steps"] / rates[key]
    measured = any((item["k"], item["g"]) == key for item in record["measurements"])
    return FixedDefault(*key, relative, evaluation_s, not _failed(record, *key) if measured else None)


def repeat_tolerance(
    record: dict[str, Any], selected: tuple[int, int], best: tuple[int, int]
) -> float | None:
    """The tolerance for ``selected`` against ``best``, from their repeated fixed runs.

    Each configuration's spread is the sample standard deviation of its fixed-run elapsed rates
    over their mean. A ratio of the two rates spreads by both, so the tolerance is
    ``TOLERANCE_Z * sqrt(cv_selected**2 + cv_best**2)``, and never less than ``NOISE_TOLERANCE``:
    repeats from one night hold no drift between days. ``None`` unless each has at least
    ``TOLERANCE_REPEATS`` fixed runs; the workload then keeps ``NOISE_TOLERANCE``.
    """

    spreads = []
    for key in {selected, best}:
        rates = [
            item["steps_per_s"]
            for item in record["measurements"]
            if (item["k"], item["g"]) == key and item["source"] == "fixed-run" and "steps_per_s" in item
        ]
        if len(rates) < TOLERANCE_REPEATS:
            return None
        spreads.append(statistics.stdev(rates) / statistics.fmean(rates))
    if len(spreads) == 1:
        spreads *= 2
    return max(NOISE_TOLERANCE, TOLERANCE_Z * math.sqrt(sum(spread**2 for spread in spreads)))


def regression_failures(
    score: Score,
    expected_relative_throughput: float,
    *,
    tolerance: float = NOISE_TOLERANCE,
) -> tuple[str, ...]:
    """Why ``score`` fails the regression rule on its workload; empty when it passes.

    Each workload is judged by itself. An incomplete search cannot be judged at all, so it fails
    until the missing point is measured and added to the record.
    """

    failures = []
    if not score.complete:
        failures.append("incomplete search: measure the missing points and add them to the record")
    if score.safe is not True:
        failures.append("unsafe: nothing selected, or the selection failed in a recorded run")
    floor = expected_relative_throughput * (1 - tolerance)
    if score.relative_throughput is None or score.relative_throughput < floor:
        failures.append(f"relative throughput {score.relative_throughput} is below {floor:.4f}")
    return tuple(failures)


def _manifest_defaults(record: dict[str, Any]) -> dict[str, Any]:
    vram = record["workload"]["host"]["vram_per_gpu_bytes"]
    return {"vram_per_gpu_bytes": vram, **record.get("manifest_defaults", {})}


def _bounds(record: dict[str, Any], defaults: dict[str, Any]) -> AutotuneBounds:
    host = record["workload"]["host"]
    return resolve_bounds(host["gpu_count"], defaults, cpu_cores=float(host["cpu_cores"]))


def _score(
    record: dict[str, Any],
    strategy: str,
    estimator: str,
    budget_s: float,
    result: dict[str, Any],
    replay: WorkloadReplay,
) -> Score:
    selected = (result["k"], result["g"]) if "k" in result else None
    best, relative = _relative_throughput(record, selected, estimator)
    used = replay.used
    workload = record["workload"]
    stop_reason = result.get("stop_reason") or "; ".join(
        reason for reason in (result.get("k_stop_reason"), result.get("g_ramp_reason")) if reason
    )
    tuning_s = replay.now if used and all("wall_time_s" in item for item in used) else None
    end_to_end_s, end_to_end_vs_best = _end_to_end(record, selected, best, tuning_s)
    return Score(
        workload=f"{workload['benchmark']}/{workload['suite']}",
        kind=record["provenance"]["kind"],
        strategy=strategy,
        throughput_estimator=estimator,
        selected=selected,
        best_recorded=best,
        relative_throughput=relative,
        probes=len(used),
        path=tuple(replay.path),
        tuning_s=tuning_s,
        budget_s=budget_s,
        end_to_end_s=end_to_end_s,
        end_to_end_vs_best=end_to_end_vs_best,
        safe=None if selected is None else not _failed(record, *selected),
        gaps=tuple(replay.gaps),
        unknown_signals=tuple(
            field
            for field in REPLAY_SIGNALS
            if any(item["outcome"] in FINISHED_OUTCOMES and field not in item for item in used)
        ),
        stand_ins=tuple(replay.stand_ins),
        stop_reason=stop_reason,
    )


def _failed(record: dict[str, Any], k: int, g: int) -> bool:
    return any(
        item["outcome"] in FAILED_OUTCOMES
        for item in record["measurements"]
        if (item["k"], item["g"]) == (k, g)
    )


def _relative_throughput(
    record: dict[str, Any],
    selected: tuple[int, int] | None,
    estimator: str,
) -> tuple[tuple[int, int] | None, float | None]:
    """Compare the selection with the best non-failing point, on one rate for both.

    Steady and elapsed rates are not comparable, and neither are probes and fixed runs, which
    differ in workload. Fixed runs of the selection come first: they all ran one full workload,
    so their elapsed rates compare every configuration. Otherwise the search's own probe rate is
    used when the selection has it, and the other one if not. Repeats count through their median.
    """

    if selected is None:
        return None, None
    fields = ("steady_steps_per_s", "steps_per_s")
    probe_fields = fields if estimator == "steady" else reversed(fields)
    comparisons = [("fixed-run", "steps_per_s"), *(("probe", field) for field in probe_fields)]
    for source, field in comparisons:
        rates = _median_rates(record, field, source)
        eligible = [key for key in rates if not _failed(record, *key)]
        if selected in rates and eligible:
            best = max(eligible, key=lambda key: rates[key])
            return best, rates[selected] / rates[best]
    return None, None


def _end_to_end(
    record: dict[str, Any],
    selected: tuple[int, int] | None,
    best: tuple[int, int] | None,
    tuning_s: float | None,
) -> tuple[float | None, float | None]:
    """Estimate tuning plus evaluation. Only fixed runs ran a workload like the evaluation."""

    steps = record.get("evaluation_steps")
    rates = _median_rates(record, "steps_per_s", "fixed-run")
    if steps is None or tuning_s is None or selected not in rates or best not in rates:
        return None, None
    end_to_end_s = tuning_s + steps / rates[selected]
    return end_to_end_s, end_to_end_s / (steps / rates[best])


def _median_rates(
    record: dict[str, Any],
    field: str,
    source: str,
) -> dict[tuple[int, int], float]:
    samples: dict[tuple[int, int], list[float]] = {}
    for item in record["measurements"]:
        if field in item and item["source"] == source:
            samples.setdefault((item["k"], item["g"]), []).append(item[field])
    return {key: statistics.median(values) for key, values in samples.items()}


def format_score(score: Score) -> str:
    """One line per score, with every caveat that changes how to read it."""

    def point(value: tuple[int, int] | None) -> str:
        return "none" if value is None else f"K={value[0]}/g={value[1]}"

    relative = "n/a" if score.relative_throughput is None else f"{score.relative_throughput:.3f}"
    tuning = "unknown" if score.tuning_s is None else f"{score.tuning_s:.0f} s"
    safe = {None: "n/a", True: "yes", False: "NO"}[score.safe]
    parts = [
        f"selected {point(score.selected)}",
        f"{relative} of best recorded {point(score.best_recorded)}",
        f"{score.probes} probes",
        f"tuning time {tuning} of {score.budget_s:.0f} s",
    ]
    if score.end_to_end_s is not None:
        parts.append(
            f"end to end {score.end_to_end_s:.0f} s, "
            f"{score.end_to_end_vs_best:.2f}x the best recorded evaluation"
        )
    parts.append(f"safe {safe}")
    if score.complete:
        parts.append("complete")
    else:
        gaps = ", ".join(f"K={gap.k}/g={gap.g} ({gap.missing})" for gap in score.gaps)
        parts.append(f"INCOMPLETE, ran off the record at {gaps}")
    if score.stand_ins:
        parts.append(f"fixed-run stand-ins {', '.join(point(item) for item in score.stand_ins)}")
    if score.unknown_signals:
        parts.append(f"unknown signals {', '.join(score.unknown_signals)}")
    head = f"{score.workload} ({score.kind}) {score.strategy} [{score.throughput_estimator}]"
    return f"{head}: {'; '.join(parts)}"


def format_fixed_default(default: FixedDefault) -> str:
    """One line for the no-tuning baseline of a workload."""

    head = f"fixed default K={default.k}/g={default.g}, no tuning"
    if default.relative_throughput is None:
        return f"{head}: no fixed run recorded"
    parts = [f"{default.relative_throughput:.3f} of best recorded"]
    if default.evaluation_s is not None:
        parts.append(f"evaluation {default.evaluation_s:.0f} s")
    parts.append(f"safe {'yes' if default.safe else 'NO'}")
    return f"{head}: {'; '.join(parts)}"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record", help="build a workload record from evaluation outputs")
    record.add_argument("--run", type=Path, action="append", required=True, help="an autotune --output-dir")
    record.add_argument(
        "--autotune",
        action="append",
        required=True,
        help="per --run, in the same order: the autotune code it ran, such as 'main 5f79ce2'",
    )
    record.add_argument("--sweep", type=Path, action="append", default=[], help="a fixed K/g --output-dir")
    record.add_argument("--source", required=True, help="where the numbers come from, such as an issue")
    record.add_argument("--policy", required=True)
    record.add_argument("--gpu-model", required=True)
    record.add_argument("--kit-version", required=True, help="the roboscope-kit version of the run")
    record.add_argument("--date", required=True, help="the date of the run, YYYY-MM-DD")
    record.add_argument("--evaluation-steps", type=int, help="total steps of the full evaluation")
    record.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("score", help="replay strategies on workload records")
    score.add_argument("records", type=Path, nargs="+")
    score.add_argument("--strategy", action="append", help="default: every registered strategy")
    score.add_argument("--throughput-estimator", choices=("elapsed", "steady"))
    score.add_argument("--repeat", type=int, default=0, help="which repeat of a measurement to replay")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "record":
        record = record_runs(
            args.run,
            args.sweep,
            source=args.source,
            policy=args.policy,
            gpu_model=args.gpu_model,
            kit_version=args.kit_version,
            date=args.date,
            autotune=args.autotune,
            evaluation_steps=args.evaluation_steps,
        )
        args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {len(record['measurements'])} measurements to {args.output}", file=sys.stderr)
        return 0
    for path in args.records:
        record = load_record(path)
        print(f"{path.name}: not measured: {record['unmeasured']}")
        print(format_fixed_default(fixed_default(record)))
        for index, observed in enumerate(record.get("observed_searches", [])):
            mismatches = search_mismatches(record, index)
            verdict = "reproduced" if not mismatches else "NOT reproduced: " + "; ".join(mismatches)
            print(f"observed search {index} ({observed['autotune']}): {verdict} by this code")
        for strategy in args.strategy or autotune_strategies():
            print(
                format_score(
                    replay_strategy(
                        record,
                        strategy,
                        throughput_estimator=args.throughput_estimator,
                        repeat=args.repeat,
                    )
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
