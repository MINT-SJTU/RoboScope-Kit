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
from collections.abc import Sequence
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


def _normalize_task_order(value: Sequence[int] | None, num_tasks: int) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("LIBERO-Pro task_order must be a sequence of all task ids")
    order = tuple(value)
    if any(type(task) is not int for task in order) or sorted(order) != list(range(num_tasks)):
        raise ValueError("LIBERO-Pro task_order must be a permutation of all task ids")
    return order


@dataclass(frozen=True)
class LiberoProRuntimeConfig(LiberoRuntimeConfig):
    policy_encoding: str = "json"
    chunk_camera_rendering: bool = False
    compiled_model_cache: bool = False
    egl_context_reuse: bool = False
    task_order: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.suite not in LIBERO_PRO_SUITES:
            raise ValueError(f"unsupported LIBERO-Pro suite: {self.suite}")
        if self.policy_encoding not in ENCODINGS:
            raise ValueError(f"LIBERO-Pro policy_encoding must be one of {ENCODINGS}")
        for field_name in ("chunk_camera_rendering", "compiled_model_cache", "egl_context_reuse"):
            if type(getattr(self, field_name)) is not bool:
                raise ValueError(f"LIBERO-Pro {field_name} must be a boolean")
        if self.egl_context_reuse and self.render_backend != "egl":
            raise ValueError("LIBERO-Pro egl_context_reuse requires the EGL backend")
        object.__setattr__(self, "task_order", _normalize_task_order(self.task_order, self.num_tasks))


class LiberoProBenchmarkAdapter(LiberoBenchmarkAdapter):
    """Run the LIBERO-Pro protocol against an SDK Policy process."""

    config: LiberoProRuntimeConfig

    def __init__(self, config: LiberoProRuntimeConfig) -> None:
        super().__init__(config)
        if config.task_order is not None:
            selected = set(self.task_ids)
            self.task_ids = [task for task in config.task_order if task in selected]
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
                *(("--chunk-camera-rendering",) if self.config.chunk_camera_rendering else ()),
                *(("--compiled-model-cache",) if self.config.compiled_model_cache else ()),
            ),
            env={
                **spec.env,
                "ROBOSCOPE_POLICY_ENCODING": self.config.policy_encoding,
                "ROBOSCOPE_LIBERO_MODEL_CACHE": "1" if self.config.compiled_model_cache else "0",
                "ROBOSCOPE_LIBERO_EGL_CONTEXT_REUSE": "1" if self.config.egl_context_reuse else "0",
            },
        )

    def metadata(self) -> dict[str, Any]:
        camera_gate_active = self.config.chunk_camera_rendering and not self.config.save_videos
        fast_optimizations = tuple(
            name
            for name, active in (
                ("msgpack-transport", self.config.policy_encoding == "msgpack"),
                ("chunk-camera-rendering", camera_gate_active),
                ("compiled-model-cache", self.config.compiled_model_cache),
                ("egl-context-reuse", self.config.egl_context_reuse),
            )
            if active
        )
        return {
            **super().metadata(),
            "benchmark": "libero_pro",
            "base_suite": self.suite.base_suite,
            "perturbation": self.suite.perturbation,
            "dimension": self.suite.dimension,
            "max_steps": self.suite.max_steps,
            "instruction_source": "bddl",
            "policy_encoding": self.config.policy_encoding,
            "chunk_camera_rendering": self.config.chunk_camera_rendering,
            "compiled_model_cache": self.config.compiled_model_cache,
            "egl_context_reuse": self.config.egl_context_reuse,
            "task_order": list(self.config.task_order) if self.config.task_order is not None else None,
            **({"fast_optimizations": list(fast_optimizations)} if fast_optimizations else {}),
        }


__all__ = ["LIBERO_PRO_SUITES", "LiberoProBenchmarkAdapter", "LiberoProRuntimeConfig", "LiberoProSuite"]
