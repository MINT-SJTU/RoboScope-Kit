# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Factory for the SimplerEnv adapter."""

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

from .adapter import SimplerEnvBenchmarkAdapter, SimplerEnvRuntimeConfig
from .tasks import QUICK_TEST_VARIANTS, SUITE_EPISODES, SUITE_TASKS, SUITES

VULKAN_ICD_DIRS = ("/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d")
VULKAN_SC_ICD = "nvidia_icd_vksc.json"


def vulkan_preflight(environ: dict[str, str] | None = None) -> None:
    """Reject a host whose Vulkan loader would also load NVIDIA's Vulkan SC ICD.

    With ``nvidia_icd_vksc.json`` installed and no explicit ICD list, every SAPIEN renderer
    teardown (each episode drops its environment) segfaults inside
    ``libnvidia-vksc-core.so`` as the loader unloads it. Naming the regular NVIDIA ICD in
    ``VK_ICD_FILENAMES`` avoids it.
    """

    environ = os.environ if environ is None else environ
    if environ.get("VK_ICD_FILENAMES") or environ.get("VK_DRIVER_FILES"):
        return
    found = [str(Path(directory) / VULKAN_SC_ICD) for directory in VULKAN_ICD_DIRS]
    found = [path for path in found if Path(path).is_file()]
    if found:
        regular = Path(found[0]).with_name("nvidia_icd.json")
        raise RuntimeError(
            f"{found[0]} (NVIDIA Vulkan SC) is installed and crashes SAPIEN's renderer teardown; "
            f"export VK_ICD_FILENAMES={regular} (the regular NVIDIA ICD) before running simpler_env"
        )


def formal_selection(suite: str, cap: int | None = None) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Every variant of the suite (at most ``cap`` episodes each), longest horizons first.

    Long-horizon episodes go to the front of the queue so the tail of a run is short ones.
    """

    variants = [variant for task in SUITE_TASKS[suite] for variant in task.variants]
    variants.sort(key=lambda variant: -variant.max_episode_steps)  # stable: table order within a horizon
    return tuple(
        (variant.variant_id, tuple(range(variant.episodes if cap is None else min(cap, variant.episodes))))
        for variant in variants
    )


def probe_order(suite: str) -> list[tuple[str, int]]:
    """Episodes in an order whose every prefix spans the suite's tasks and variants.

    Tasks take turns; within a task, variants take turns, so a longer probe repeats every
    episode of a shorter one.
    """

    per_task = []
    for task in SUITE_TASKS[suite]:
        order = []
        for index in range(max(variant.episodes for variant in task.variants)):
            order += [(variant.variant_id, index) for variant in task.variants if index < variant.episodes]
        per_task.append(order)
    merged = []
    for position in range(max(len(order) for order in per_task)):
        merged += [order[position] for order in per_task if position < len(order)]
    return merged


def _group(episodes: list[tuple[str, int]]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    grouped: dict[str, list[int]] = {}
    for variant_id, index in episodes:
        grouped.setdefault(variant_id, []).append(index)
    return tuple((variant_id, tuple(indices)) for variant_id, indices in grouped.items())


class SimplerEnvBenchmarkFactory:
    @staticmethod
    def validate_suite(suite: str) -> None:
        if suite not in SUITES:
            raise ValueError(
                f"simpler_env does not support suite {suite!r}; supported suites: {', '.join(SUITES)}"
            )

    def create(self, request: BenchmarkRunRequest, manifest: BenchmarkManifest) -> SimplerEnvBenchmarkAdapter:
        self.validate_suite(request.suite)
        if request.quick_test:
            selection: tuple[tuple[str, tuple[int, ...]], ...] = ((QUICK_TEST_VARIANTS[request.suite], (0,)),)
        else:
            if request.episodes is not None and request.episodes < 1:
                raise ValueError(
                    "simpler_env --episodes caps the episodes of every variant and must be positive"
                )
            selection = formal_selection(request.suite, request.episodes)
        return self._adapter(request, manifest, selection)

    def create_probe(
        self,
        request: BenchmarkRunRequest,
        manifest: BenchmarkManifest,
        concurrency: int,
    ) -> SimplerEnvBenchmarkAdapter:
        self.validate_suite(request.suite)
        count = int(manifest.defaults.get("probe_episodes", len(SUITE_TASKS[request.suite])))
        items_per_worker = probe_items_per_worker(manifest.defaults)
        if items_per_worker > 0:
            count = max(count, items_per_worker * concurrency)
        order = probe_order(request.suite)
        return self._adapter(request, manifest, _group(order[: min(count, len(order))]))

    def self_test(self, manifest: BenchmarkManifest, output_dir: Path) -> dict[str, Any]:
        vulkan_preflight()
        python = manifest.runtime_path("python")
        root = manifest.runtime_path("simpler_env")
        if not python.is_file():
            raise RuntimeError(f"SimplerEnv runtime python is missing: {python}")
        if not (root / "simpler_env" / "main_inference.py").is_file():
            raise RuntimeError(f"SimplerEnv source is missing under {root}")
        output_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            (str(python), "-m", "roboscope_kit.runtime.benchmarks.simpler_env.check"),
            check=True,
            cwd=root,
            env={
                **os.environ,
                "PYTHONPATH": str(kit_import_root()),
                "PYTHONDONTWRITEBYTECODE": "1",
                "DISPLAY": "",
            },
            capture_output=True,
            text=True,
        )
        report_line = [line for line in completed.stdout.splitlines() if line.startswith("{")][-1]
        report = json.loads(report_line)
        (output_dir / "simpler_env-check.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        return {
            "benchmark": "simpler_env",
            "python": str(python),
            "simpler_env": str(root),
            "suites": {suite: SUITE_EPISODES[suite] for suite in SUITES},
            "checked_variants": report["variants"],
        }

    @staticmethod
    def _adapter(
        request: BenchmarkRunRequest,
        manifest: BenchmarkManifest,
        selection: tuple[tuple[str, tuple[int, ...]], ...],
    ) -> SimplerEnvBenchmarkAdapter:
        vulkan_preflight()
        recycle = manifest.defaults.get("recycle_after_records")
        return SimplerEnvBenchmarkAdapter(
            SimplerEnvRuntimeConfig(
                python=manifest.runtime_path("python"),
                simpler_env_root=manifest.runtime_path("simpler_env"),
                suite=request.suite,
                selection=selection,
                cpu_threads=int(manifest.defaults.get("client_cpu_threads", 1)),
                recycle_after_records=None if recycle is None else int(recycle),
            )
        )


__all__ = ["SimplerEnvBenchmarkFactory", "formal_selection", "probe_order", "vulkan_preflight"]
