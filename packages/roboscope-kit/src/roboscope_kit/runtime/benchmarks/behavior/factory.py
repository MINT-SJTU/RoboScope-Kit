# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Factory for the BEHAVIOR Challenge 2026 adapter."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from roboscope_kit.runtime.benchmark.autotune import probe_items_per_worker
from roboscope_kit.runtime.benchmark.factories import BenchmarkRunRequest, nested_task_prefix
from roboscope_kit.runtime.benchmark.kit_path import kit_import_root
from roboscope_kit.runtime.benchmark.manifest import BenchmarkManifest

from .adapter import BehaviorBenchmarkAdapter, BehaviorRuntimeConfig
from .tasks import INSTANCES_PER_TASK, PYTHON_HASH_SEED, QUICK_TEST_TASK, SUITES, TASK_NAMES


class BehaviorBenchmarkFactory:
    @staticmethod
    def validate_suite(suite: str) -> None:
        if suite not in SUITES:
            supported = ", ".join(SUITES)
            raise ValueError(f"behavior does not support suite {suite!r}; supported suites: {supported}")

    def create(self, request: BenchmarkRunRequest, manifest: BenchmarkManifest) -> BehaviorBenchmarkAdapter:
        tasks = (QUICK_TEST_TASK,) if request.quick_test else TASK_NAMES
        return self._create(request, manifest, tasks, self._instances(request, manifest))

    def create_probe(
        self,
        request: BenchmarkRunRequest,
        manifest: BenchmarkManifest,
        concurrency: int,
    ) -> BehaviorBenchmarkAdapter:
        # Instance 0 of a growing, spread-out prefix of the tasks: each probe repeats every episode
        # of a smaller probe.
        items = max(
            int(manifest.defaults.get("probe_tasks", 1)),
            probe_items_per_worker(manifest.defaults) * concurrency,
        )
        tasks = tuple(TASK_NAMES[index] for index in nested_task_prefix(len(TASK_NAMES), items))
        return self._create(request, manifest, tasks, 1)

    def self_test(self, manifest: BenchmarkManifest, output_dir: Path) -> dict[str, Any]:
        paths = self._runtime_paths(manifest)
        for name, path in paths.items():
            if not path.exists():
                raise RuntimeError(f"BEHAVIOR runtime {name} is missing: {path}")
        # Checks the pinned data, then, once per suite (each with its own observation wrapper), loads
        # the quick-test task's scene and its first public test instance headless and maps the first
        # rendered observation to the suite's SDK features.
        output_dir.mkdir(parents=True, exist_ok=True)
        checks = {}
        for suite in SUITES:
            report = output_dir / f"behavior-check-{suite}.json"
            subprocess.run(
                (
                    str(paths["python"]),
                    "-m",
                    "roboscope_kit.runtime.benchmarks.behavior.check",
                    "--task",
                    QUICK_TEST_TASK,
                    "--suite",
                    suite,
                    "--report",
                    str(report),
                ),
                check=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(kit_import_root()),
                    "OMNIGIBSON_DATA_PATH": str(paths["data"]),
                    "OMNI_KIT_ACCEPT_EULA": "YES",
                    "PYTHONHASHSEED": str(PYTHON_HASH_SEED),
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                stdout=subprocess.DEVNULL,
            )
            checks[suite] = json.loads(report.read_text(encoding="utf-8"))
        return {
            "benchmark": "behavior",
            "python": str(paths["python"]),
            "suites": list(SUITES),
            "tasks": len(TASK_NAMES),
            "checks": checks,
        }

    @staticmethod
    def _runtime_paths(manifest: BenchmarkManifest) -> dict[str, Path]:
        return {name: manifest.runtime_path(name) for name in ("python", "data")}

    def _create(
        self,
        request: BenchmarkRunRequest,
        manifest: BenchmarkManifest,
        tasks: tuple[str, ...],
        instances: int,
    ) -> BehaviorBenchmarkAdapter:
        self.validate_suite(request.suite)
        paths = self._runtime_paths(manifest)
        return BehaviorBenchmarkAdapter(
            BehaviorRuntimeConfig(
                python=paths["python"],
                data_path=paths["data"],
                suite=request.suite,
                tasks=tasks,
                instances_per_task=instances,
                write_video=bool(manifest.defaults.get("write_video", False)),
                cpu_threads=int(manifest.defaults.get("client_cpu_threads", 4)),
            )
        )

    @staticmethod
    def _instances(request: BenchmarkRunRequest, manifest: BenchmarkManifest) -> int:
        if request.quick_test:
            return 1
        value = request.episodes or int(manifest.defaults.get("episodes", INSTANCES_PER_TASK))
        if not 1 <= value <= INSTANCES_PER_TASK:
            raise ValueError(
                f"BEHAVIOR episodes per task must be in [1, {INSTANCES_PER_TASK}]; received {value}"
            )
        return value


__all__ = ["BehaviorBenchmarkFactory"]
