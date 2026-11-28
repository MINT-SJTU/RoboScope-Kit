# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Construct Evo-DexBench evaluation and probe workloads."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from roboscope_kit.runtime.benchmark.autotune import probe_items_per_worker
from roboscope_kit.runtime.benchmark.factories import BenchmarkRunRequest, nested_task_prefix
from roboscope_kit.runtime.benchmark.kit_path import kit_import_root
from roboscope_kit.runtime.benchmark.manifest import BenchmarkManifest

from .adapter import EvoDexBenchBenchmarkAdapter, EvoDexBenchRuntimeConfig, asset_environment
from .protocol import (
    ACTION_BOUNDS_POLICY,
    ACTION_REPRESENTATION,
    CONTROL_FREQ_HZ,
    EPISODES_PER_TASK,
    EVO_DEXBENCH_REVISION,
    NATIVE_ENVS,
    OBS_MODE,
    OBSERVATION_STEPS,
    PROTOCOL_VERSION,
    SIM_FREQ_HZ,
    SUITES,
    TASKS_BY_ID,
)


class EvoDexBenchBenchmarkFactory:
    @staticmethod
    def validate_suite(suite: str) -> None:
        if suite not in SUITES:
            raise ValueError(f"unsupported Evo-DexBench suite {suite!r}; choose from {tuple(SUITES)}")

    def create(
        self, request: BenchmarkRunRequest, manifest: BenchmarkManifest,
    ) -> EvoDexBenchBenchmarkAdapter:
        self.validate_suite(request.suite)
        tasks = SUITES[request.suite]
        if request.quick_test:
            return self._create(request, manifest, tasks[:1], NATIVE_ENVS)
        episodes = request.episodes or int(manifest.defaults.get("episodes", EPISODES_PER_TASK))
        return self._create(request, manifest, tasks, episodes)

    def create_probe(
        self, request: BenchmarkRunRequest, manifest: BenchmarkManifest, concurrency: int,
    ) -> EvoDexBenchBenchmarkAdapter:
        self.validate_suite(request.suite)
        tasks = SUITES[request.suite]
        count = max(
            int(manifest.defaults.get("probe_task_count", 3)),
            probe_items_per_worker(manifest.defaults) * concurrency,
        )
        selected = tuple(tasks[index] for index in nested_task_prefix(len(tasks), min(count, len(tasks))))
        # One wave per task until the suite runs out of tasks, then more waves per task.
        waves = min(-(-count // len(selected)), EPISODES_PER_TASK // NATIVE_ENVS)
        return self._create(request, manifest, selected, waves * NATIVE_ENVS)

    def self_test(self, manifest: BenchmarkManifest, output_dir: Path) -> dict[str, Any]:
        self._validate_manifest(manifest)
        python = manifest.runtime_path("python")
        root = manifest.runtime_path("benchmark_root")
        assets = manifest.runtime_path("asset_root")
        evaluator = root / "dex_benchmark" / "evaluation" / "protocol_parallel.py"
        if not python.is_file() or not evaluator.is_file():
            raise RuntimeError("Evo-DexBench runtime interpreter or pinned source is missing")
        check = ("from roboscope_kit.runtime.benchmarks.evo_dexbench.worker import check_runtime; "
                 "check_runtime()")
        subprocess.run(
            (str(python), "-c", check),
            check=True, cwd=root,
            env={
                **os.environ, "PYTHONPATH": str(kit_import_root()), **asset_environment(assets),
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            },
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "benchmark": "evo_dexbench", "python": str(python), "benchmark_root": str(root),
            "asset_root": str(assets), "suites": list(SUITES), "task_count": len(TASKS_BY_ID),
            "schedules_verified": True, "simulator_reset_render_tested": True,
        }

    @staticmethod
    def _validate_manifest(manifest: BenchmarkManifest) -> None:
        if manifest.protocol_versions.get("evo_dexbench") != PROTOCOL_VERSION:
            raise ValueError(f"Evo-DexBench manifest must declare protocol {PROTOCOL_VERSION}")
        settings = manifest.benchmark_protocols.get("evo_dexbench", {})
        frozen = {
            "evo_dexbench_revision": EVO_DEXBENCH_REVISION, "native_envs": NATIVE_ENVS,
            "sim_freq_hz": SIM_FREQ_HZ, "control_freq_hz": CONTROL_FREQ_HZ, "obs_mode": OBS_MODE,
            "action_bounds_policy": ACTION_BOUNDS_POLICY, "observation_steps": OBSERVATION_STEPS,
            "action_representation": ACTION_REPRESENTATION,
            "episodes_per_task": EPISODES_PER_TASK,
            "task_horizons": {task_id: task.horizon for task_id, task in TASKS_BY_ID.items()},
        }
        for key, expected in frozen.items():
            if settings.get(key) != expected:
                raise ValueError(f"Evo-DexBench manifest {key} must equal {expected!r}")

    def _create(
        self, request: BenchmarkRunRequest, manifest: BenchmarkManifest,
        tasks: tuple[str, ...], episodes: int,
    ) -> EvoDexBenchBenchmarkAdapter:
        self._validate_manifest(manifest)
        return EvoDexBenchBenchmarkAdapter(EvoDexBenchRuntimeConfig(
            python=manifest.runtime_path("python"), benchmark_root=manifest.runtime_path("benchmark_root"),
            asset_root=manifest.runtime_path("asset_root"), suite=request.suite, tasks=tasks,
            episodes_per_task=episodes, cpu_threads=int(manifest.defaults.get("client_cpu_threads", 1)),
        ))
