# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0
"""Factory for manifest-configured ManiSkill task suites."""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

from roboscope_kit.benchmark_specs import BenchmarkName
from roboscope_kit.runtime.benchmark.autotune import probe_items_per_worker
from roboscope_kit.runtime.benchmark.factories import BenchmarkRunRequest
from roboscope_kit.runtime.benchmark.manifest import BenchmarkManifest

from .adapter import ManiSkillBenchmarkAdapter, ManiSkillRuntimeConfig
from .profile import self_test_profiles, validate_profiles


class ManiSkillBenchmarkFactory:
    @staticmethod
    def validate_suite(suite: str) -> None:
        # Manifest validation owns the task list; custom manifests need no SDK edit.
        if not isinstance(suite, str) or not suite:
            raise ValueError("ManiSkill requires a nonempty suite name")

    def create(
        self, request: BenchmarkRunRequest, manifest: BenchmarkManifest,
    ) -> ManiSkillBenchmarkAdapter:
        profiles = self._profiles(request, manifest)
        expected_ids = tuple(profile["profile_id"] for profile in profiles)
        if request.quick_test:
            profiles = ({**profiles[0], "seeds": profiles[0]["seeds"][:1]},)
        else:
            profiles = tuple(self._limit(profile, request.episodes) for profile in profiles)
        return self._adapter(request, manifest, profiles, expected_ids)

    def create_probe(
        self, request: BenchmarkRunRequest, manifest: BenchmarkManifest, concurrency: int,
    ) -> ManiSkillBenchmarkAdapter:
        profiles = self._profiles(request, manifest)
        expected_ids = tuple(profile["profile_id"] for profile in profiles)
        vector_size = int(manifest.defaults.get("envs_per_worker", 10))
        groups = max(len(profiles), concurrency * max(1, probe_items_per_worker(manifest.defaults)))
        count = max(
            int(manifest.defaults.get("probe_episodes_per_task", 1)),
            vector_size * math.ceil(groups / len(profiles)),
        )
        limited = tuple(self._limit(profile, request.episodes) for profile in profiles)
        probes = tuple({**profile, "seeds": profile["seeds"][:count]} for profile in limited)
        return self._adapter(request, manifest, probes, expected_ids)

    def self_test(self, manifest: BenchmarkManifest, output_dir: Path) -> dict[str, Any]:
        self_test_profiles(self._protocol(manifest))
        python = manifest.runtime_path("python")
        if not python.is_file():
            raise FileNotFoundError(f"ManiSkill runtime Python is missing: {python}")
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / "runtime-validation.json"
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment["HOME"] = str(manifest.runtime_path("home"))
        environment["MS_ASSET_DIR"] = str(manifest.runtime_path("asset_dir"))
        result = subprocess.run(
            [
                str(python), "-I", "-m", "roboscope_kit.runtime.benchmarks.maniskill.worker",
                "--benchmark-manifest", str(manifest.path),
                "--self-test", "--self-test-output", str(destination),
            ],
            capture_output=True, text=True, env=environment, timeout=300, check=False,
        )
        log = output_dir / "runtime-validation.log"
        log.write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"ManiSkill simulator self-test failed; see {log}: "
                               + (result.stderr or result.stdout)[-3000:])
        if not destination.is_file():
            raise RuntimeError(f"ManiSkill simulator produced no self-test result; see {log}")
        return json.loads(destination.read_text(encoding="utf-8"))

    @staticmethod
    def _protocol(manifest: BenchmarkManifest) -> dict[str, Any]:
        return manifest.benchmark_protocols.get(BenchmarkName.MANISKILL, {})

    def _profiles(
        self, request: BenchmarkRunRequest, manifest: BenchmarkManifest,
    ) -> tuple[dict[str, Any], ...]:
        self.validate_suite(request.suite)
        manifest.validate_request(BenchmarkName.MANISKILL, request.suite)
        return validate_profiles(self._protocol(manifest), request.suite)

    @staticmethod
    def _limit(profile: dict[str, Any], count: int | None) -> dict[str, Any]:
        count = profile["episodes_per_task"] if count is None else count
        if not 1 <= count <= int(profile["episodes_per_task"]):
            raise ValueError(f"ManiSkill episode count exceeds profile {profile['profile_id']!r}")
        return {**profile, "seeds": profile["seeds"][:count]}

    @staticmethod
    def _adapter(
        request: BenchmarkRunRequest, manifest: BenchmarkManifest,
        profiles: tuple[dict[str, Any], ...], expected_ids: tuple[str, ...],
    ) -> ManiSkillBenchmarkAdapter:
        return ManiSkillBenchmarkAdapter(ManiSkillRuntimeConfig(
            python=manifest.runtime_path("python"), manifest_path=manifest.path,
            runtime_home=manifest.runtime_path("home"), asset_dir=manifest.runtime_path("asset_dir"),
            suite=request.suite, profiles=profiles, expected_profile_ids=expected_ids,
            envs_per_worker=1 if request.quick_test else int(
                manifest.defaults.get("envs_per_worker", 10)
            ),
            policy_batch_size=int(manifest.defaults.get("policy_batch_size", 1)),
            cpu_threads=int(manifest.defaults.get("client_cpu_threads", 1)),
        ))
