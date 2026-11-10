# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Official LIBERO-Pro work manifest and SDK-native evaluator process adapter.

LIBERO-Pro is a fork of LIBERO on the same simulator stack, so evaluator processes run the shared
``runtime.benchmarks.libero.worker``. This adapter owns the LIBERO-Pro protocol: each suite is one base
LIBERO suite under one released single-perturbation set, runs at that base suite's horizon, and feeds
the Policy the instruction of the perturbed BDDL file.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from roboscope_kit.protocol import ENCODINGS
from roboscope_kit.runtime.benchmark import ProcessSpec, WorkerContext
from roboscope_kit.runtime.benchmarks.libero.adapter import LiberoBenchmarkAdapter, LiberoRuntimeConfig


@dataclass(frozen=True)
class LiberoProSuite:
    base_suite: str
    perturbation: str
    dimension: str
    max_steps: int


# Policy action steps per episode of each base suite: the LIBERO horizons, which LIBERO-Pro keeps.
_BASE_MAX_STEPS = {
    "libero_goal": 300,
    "libero_spatial": 220,
    "libero_10": 520,
    "libero_object": 280,
}
# Suffix of each released single-perturbation suite and the generalization dimension it tests
# (upstream evaluation_config.yaml perturbation_mapping).
_PERTURBATIONS = {
    "object": "Object",
    "swap": "Position",
    "lan": "Semantic",
    "task": "Task",
}

# The 16 suites whose BDDL and initial-state files LIBERO-Pro released; Environment (``*_env``) and
# combined (``*_temp``) suites have no released files.
LIBERO_PRO_SUITES = {
    f"{base}_{perturbation}": LiberoProSuite(base, perturbation, dimension, max_steps)
    for base, max_steps in _BASE_MAX_STEPS.items()
    for perturbation, dimension in _PERTURBATIONS.items()
}


@dataclass(frozen=True)
class LiberoProRuntimeConfig(LiberoRuntimeConfig):
    policy_encoding: str = "msgpack"
    persistent_policy_connection: bool = False
    chunk_camera_rendering: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.suite not in LIBERO_PRO_SUITES:
            raise ValueError(f"unsupported LIBERO-Pro suite: {self.suite}")
        if self.policy_encoding not in ENCODINGS:
            raise ValueError(f"LIBERO-Pro policy_encoding must be one of {ENCODINGS}")
        if type(self.persistent_policy_connection) is not bool:
            raise ValueError("LIBERO-Pro persistent_policy_connection must be a boolean")
        if type(self.chunk_camera_rendering) is not bool:
            raise ValueError("LIBERO-Pro chunk_camera_rendering must be a boolean")


class LiberoProBenchmarkAdapter(LiberoBenchmarkAdapter):
    """Run the LIBERO-Pro protocol against an SDK Policy process."""

    config: LiberoProRuntimeConfig

    def __init__(self, config: LiberoProRuntimeConfig) -> None:
        super().__init__(config)
        self.suite = LIBERO_PRO_SUITES[config.suite]

    def worker_process(self, context: WorkerContext) -> ProcessSpec:
        spec = super().worker_process(context)
        return dataclasses.replace(
            spec,
            command=(
                *spec.command,
                "--max-steps",
                str(self.suite.max_steps),
                "--instruction-source",
                "bddl",
                *(("--persistent-policy-connection",) if self.config.persistent_policy_connection else ()),
                *(("--chunk-camera-rendering",) if self.config.chunk_camera_rendering else ()),
            ),
            env={**spec.env, "ROBOSCOPE_POLICY_ENCODING": self.config.policy_encoding},
        )

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "benchmark": "libero_pro",
            "base_suite": self.suite.base_suite,
            "perturbation": self.suite.perturbation,
            "dimension": self.suite.dimension,
            "max_steps": self.suite.max_steps,
            "instruction_source": "bddl",
            "policy_encoding": self.config.policy_encoding,
            "persistent_policy_connection": self.config.persistent_policy_connection,
            "chunk_camera_rendering": self.config.chunk_camera_rendering,
        }


__all__ = ["LIBERO_PRO_SUITES", "LiberoProBenchmarkAdapter", "LiberoProRuntimeConfig", "LiberoProSuite"]
