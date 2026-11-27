# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Ship the repository's ``benchmarks/`` directory with the package.

``roboscope build`` runs the build scripts from ``roboscope_kit/_benchmarks``. A build from the repository
takes them from ``../../benchmarks``; the sdist carries them as ``benchmarks/`` so a wheel built from the
sdist finds them there.
"""

from __future__ import annotations

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class BenchmarksBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:
        root = Path(self.root)
        candidates = (root / "benchmarks", root.parent.parent / "benchmarks")
        source = next((path for path in candidates if (path / "libero" / "build-runtime.sh").is_file()), None)
        if source is None:
            raise FileNotFoundError(f"benchmarks/ not found next to {root} or in its repository")
        target = "roboscope_kit/_benchmarks" if self.target_name == "wheel" else "benchmarks"
        build_data["force_include"][str(source)] = target
