# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Factory for the RLBench PerAct 18-task adapter."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from roboscope_kit.runtime.benchmark.autotune import probe_items_per_worker
from roboscope_kit.runtime.benchmark.factories import BenchmarkRunRequest
from roboscope_kit.runtime.benchmark.kit_path import kit_import_root
from roboscope_kit.runtime.benchmark.manifest import BenchmarkManifest

from .adapter import RLBenchBenchmarkAdapter, RLBenchRuntimeConfig
from .tasks import EPISODES_PER_TASK, QUICK_TEST_TASK, SUITES, TASK_NAMES


class RLBenchBenchmarkFactory:
    @staticmethod
    def validate_suite(suite: str) -> None:
        if suite not in SUITES:
            supported = ", ".join(SUITES)
            raise ValueError(f"rlbench does not support suite {suite!r}; supported suites: {supported}")

    def create(self, request: BenchmarkRunRequest, manifest: BenchmarkManifest) -> RLBenchBenchmarkAdapter:
        tasks = (QUICK_TEST_TASK,) if request.quick_test else TASK_NAMES
        return self._create(request, manifest, tasks, self._episodes(request, manifest))

    def create_probe(
        self,
        request: BenchmarkRunRequest,
        manifest: BenchmarkManifest,
        concurrency: int,
    ) -> RLBenchBenchmarkAdapter:
        # Every task, a growing prefix of test episodes: each probe repeats the episodes of
        # every smaller probe and mixes the tasks in the formal run's proportions.
        episodes = int(manifest.defaults.get("probe_episodes_per_task", 1))
        items_per_worker = probe_items_per_worker(manifest.defaults)
        if items_per_worker > 0:
            episodes = max(episodes, -(-items_per_worker * concurrency // len(TASK_NAMES)))
        return self._create(request, manifest, TASK_NAMES, min(episodes, EPISODES_PER_TASK))

    def self_test(self, manifest: BenchmarkManifest, output_dir: Path) -> dict[str, Any]:
        paths = self._runtime_paths(manifest)
        for name, path in paths.items():
            if not path.exists():
                raise RuntimeError(f"RLBench runtime {name} is missing: {path}")
        coppeliasim = str(paths["coppeliasim"])
        # Checks the quick-test task's test episodes against the frozen digest, then launches
        # CoppeliaSim headless and renders that task's first test episode; the build checks
        # every task's episodes.
        output_dir.mkdir(parents=True, exist_ok=True)
        report = output_dir / "rlbench-check.json"
        subprocess.run(
            (
                str(paths["python"]),
                "-m",
                "roboscope_kit.runtime.benchmarks.rlbench.check",
                "--test-set",
                str(paths["test_set"]),
                "--xvfb",
                str(paths["xvfb"]),
                "--task",
                QUICK_TEST_TASK,
                "--report",
                str(report),
            ),
            check=True,
            env={
                **os.environ,
                "PYTHONPATH": str(kit_import_root()),
                "COPPELIASIM_ROOT": coppeliasim,
                "LD_LIBRARY_PATH": ":".join((coppeliasim, os.environ.get("LD_LIBRARY_PATH", ""))),
                "QT_QPA_PLATFORM_PLUGIN_PATH": coppeliasim,
                "CUDA_VISIBLE_DEVICES": "",
                "LP_NUM_THREADS": "2",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            stdout=subprocess.DEVNULL,
        )
        return {
            "benchmark": "rlbench",
            "python": str(paths["python"]),
            "suites": list(SUITES),
            "tasks": list(TASK_NAMES),
            "check": json.loads(report.read_text(encoding="utf-8")),
        }

    @staticmethod
    def _runtime_paths(manifest: BenchmarkManifest) -> dict[str, Path]:
        return {name: manifest.runtime_path(name) for name in ("python", "coppeliasim", "test_set", "xvfb")}

    def _create(
        self,
        request: BenchmarkRunRequest,
        manifest: BenchmarkManifest,
        tasks: tuple[str, ...],
        episodes: int,
    ) -> RLBenchBenchmarkAdapter:
        self.validate_suite(request.suite)
        paths = self._runtime_paths(manifest)
        return RLBenchBenchmarkAdapter(
            RLBenchRuntimeConfig(
                python=paths["python"],
                coppeliasim=paths["coppeliasim"],
                test_set=paths["test_set"],
                xvfb=paths["xvfb"],
                suite=request.suite,
                tasks=tasks,
                episodes_per_task=episodes,
                chunk_size=int(manifest.defaults.get("chunk_size", 1)),
                cpu_threads=int(manifest.defaults.get("client_cpu_threads", 1)),
            )
        )

    @staticmethod
    def _episodes(request: BenchmarkRunRequest, manifest: BenchmarkManifest) -> int:
        if request.quick_test:
            return 1
        value = request.episodes or int(manifest.defaults.get("episodes", EPISODES_PER_TASK))
        if not 1 <= value <= EPISODES_PER_TASK:
            raise ValueError(
                f"RLBench episodes per task must be in [1, {EPISODES_PER_TASK}]; received {value}"
            )
        return value


__all__ = ["RLBenchBenchmarkFactory"]
