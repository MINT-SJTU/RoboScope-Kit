# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Construct isolated RoboCasa365 evaluation and native-group probe workloads."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from roboscope_kit.runtime.benchmark.autotune import probe_items_per_worker
from roboscope_kit.runtime.benchmark.factories import BenchmarkRunRequest, nested_task_prefix
from roboscope_kit.runtime.benchmark.kit_path import kit_import_root
from roboscope_kit.runtime.benchmark.manifest import BenchmarkManifest

from .adapter import RoboCasa365BenchmarkAdapter, RoboCasa365RuntimeConfig
from .protocol import (
    ACTION_REPRESENTATION,
    ACTION_STEPS,
    CONSTRUCTOR_SEED,
    CONTROL_FREQ_HZ,
    EPISODES_PER_TASK,
    NATIVE_ENVS,
    OBSERVATION_STEPS,
    PROTOCOL_VERSION,
    ROBOCASA_REVISION,
    ROBOSUITE_REVISION,
    ROLLOUT_LABEL_BASE,
    SCENE_SPLIT,
    SUITES,
)


class RoboCasa365BenchmarkFactory:
    @staticmethod
    def validate_suite(suite: str) -> None:
        if suite not in SUITES:
            raise ValueError(f"unsupported RoboCasa365 suite {suite!r}; choose from {tuple(SUITES)}")

    def create(
        self, request: BenchmarkRunRequest, manifest: BenchmarkManifest,
    ) -> RoboCasa365BenchmarkAdapter:
        self.validate_suite(request.suite)
        tasks = SUITES[request.suite]
        if request.quick_test:
            return self._create(request, manifest, tasks[:1], 1, 1)
        episodes = request.episodes or int(manifest.defaults.get("episodes", EPISODES_PER_TASK))
        return self._create(request, manifest, tasks, episodes, NATIVE_ENVS)

    def create_probe(
        self, request: BenchmarkRunRequest, manifest: BenchmarkManifest, concurrency: int,
    ) -> RoboCasa365BenchmarkAdapter:
        self.validate_suite(request.suite)
        tasks = SUITES[request.suite]
        count = max(
            int(manifest.defaults.get("probe_task_count", 3)),
            probe_items_per_worker(manifest.defaults) * concurrency,
        )
        selected = tuple(tasks[index] for index in nested_task_prefix(len(tasks), count))
        return self._create(request, manifest, selected, NATIVE_ENVS, NATIVE_ENVS)

    def self_test(self, manifest: BenchmarkManifest, output_dir: Path) -> dict[str, Any]:
        self._validate_manifest(manifest)
        python = manifest.runtime_path("python")
        root = manifest.runtime_path("benchmark_root")
        if not python.is_file() or not (root / "robocasa" / "wrappers" / "gym_wrapper.py").is_file():
            raise RuntimeError("RoboCasa365 runtime interpreter or pinned source is missing")
        subprocess.run(
            (str(python), "-c",
             "from roboscope_kit.runtime.benchmarks.robocasa365.protocol import validate_upstream; "
             "validate_upstream(); import robocasa.wrappers.gym_wrapper; "
             "from robosuite.utils.binding_utils import MjRenderContext; "
             "assert MjRenderContext._roboscope_owning_context_cleanup == 2; "
             "print('RoboCasa365 imports and frozen registry passed')"),
            check=True, cwd=root,
            env={**os.environ, "PYTHONPATH": str(kit_import_root()), "MUJOCO_GL": "egl"},
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "benchmark": "robocasa365", "python": str(python), "benchmark_root": str(root),
            "suites": list(SUITES), "task_count": len(SUITES["target50"]),
            "registry_verified": True, "simulator_episode_tested": False,
            "owning_context_cleanup_version": 2,
        }

    @staticmethod
    def _validate_manifest(manifest: BenchmarkManifest) -> None:
        if manifest.protocol_versions.get("robocasa365") != PROTOCOL_VERSION:
            raise ValueError(f"RoboCasa365 manifest must declare protocol {PROTOCOL_VERSION}")
        settings = manifest.benchmark_protocols.get("robocasa365", {})
        frozen = {
            "robocasa_revision": ROBOCASA_REVISION, "robosuite_revision": ROBOSUITE_REVISION,
            "scene_split": SCENE_SPLIT, "native_envs": NATIVE_ENVS,
            "constructor_seed": CONSTRUCTOR_SEED, "rollout_label_base": ROLLOUT_LABEL_BASE,
            "control_freq_hz": CONTROL_FREQ_HZ, "observation_steps": OBSERVATION_STEPS,
            "action_steps": ACTION_STEPS, "action_representation": ACTION_REPRESENTATION,
        }
        for key, expected in frozen.items():
            if settings.get(key) != expected:
                raise ValueError(f"RoboCasa365 manifest {key} must equal {expected!r}")

    def _create(
        self, request: BenchmarkRunRequest, manifest: BenchmarkManifest,
        tasks: tuple[str, ...], episodes: int, native_envs: int,
    ) -> RoboCasa365BenchmarkAdapter:
        self._validate_manifest(manifest)
        return RoboCasa365BenchmarkAdapter(RoboCasa365RuntimeConfig(
            python=manifest.runtime_path("python"), benchmark_root=manifest.runtime_path("benchmark_root"),
            suite=request.suite, tasks=tasks, episodes_per_task=episodes, native_envs=native_envs,
            cpu_threads=int(manifest.defaults.get("client_cpu_threads", 1)),
            save_videos=bool(manifest.defaults.get("save_videos", False)),
        ))
