# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Evo-DexBench DP-C / DP-T checkpoints behind the RoboScope Policy interface.

Checkpoints load through the upstream example's own loader (``dp_policy.server``) and
inputs follow its session and server: two consecutive frames, RGB as CHW float in
[0, 1], state as end-effector pose plus hand joints per role, and the first
``execute_steps`` of each predicted chunk. Each task uses the checkpoint of its task
group, listed in ``roboscope-checkpoints.json`` inside the ``--checkpoint-source``
directory with paths relative to it (see ``roboscope-checkpoints.example.json``).
Sampling noise comes from torch's global generator, unseeded, as in the upstream server.
On a CUDA device the denoising loop replays as a CUDA graph (``graph_sampler.py``),
bit-identical to eager sampling; set ``EVO_DEXBENCH_CUDA_GRAPHS=0`` to sample eagerly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from roboscope_kit import (
    ActionChunk,
    EpisodeContext,
    Observation,
    Policy,
    PolicyCapabilities,
    SessionContext,
)

BENCHMARK = "evo_dexbench"
ACTION_REPRESENTATION = "evo-dexbench.semantic-v1"
CONFIG_NAME = "roboscope-checkpoints.json"
GROUPS = {
    "single1": ("m01_v1", "m02_v1", "m03_v1", "m04_v1", "m05_v1"),
    "single2": ("m06_v1", "m07_v1", "m08_v1", "m09_v1", "m10_v1"),
    "single3": ("m11_v1", "m12_v1", "m13_v1", "m14_v1", "m15_v1"),
    "double1": ("b01_v1", "b02_v1", "b03_v1", "b04_v1", "b05_v1"),
    "double2": ("b06_v1", "b07_v1", "b08_v1", "b09_v1", "b10_v1"),
    "double3": ("b11_v1", "b12_v1", "b13_v1", "b14_v1", "b15_v1"),
}
SUITES = {"single_hand": ("single1", "single2", "single3"), "bimanual": ("double1", "double2", "double3")}


def checkpoint_key(task: str) -> str:
    """The last task of each group is its generalization (Gen) task; the rest use Core."""
    for group, tasks in GROUPS.items():
        if task in tasks:
            return f"{group}_{'gen' if task == tasks[-1] else 'core'}"
    raise ValueError(f"unknown Evo-DexBench task {task!r}")


class EvoDexBenchDiffusionPolicy(Policy):
    def __init__(self) -> None:
        self.paths: dict[str, Path] = {}
        self.models: dict[str, tuple[Any, dict[str, Any]]] = {}
        self.tasks: dict[str, str] = {}
        self.device: Any = None
        self.execute_steps = 8
        self.variant = ""

    def load(self, context: SessionContext) -> PolicyCapabilities:
        import torch
        from dp_policy.upstream import configure_upstream

        configure_upstream()
        root = Path(context.checkpoint_path)
        config = json.loads((root / CONFIG_NAME).read_text())
        self.variant = str(config["variant"])
        if self.variant not in ("dp_c", "dp_t"):
            raise ValueError("variant must be dp_c or dp_t")
        self.execute_steps = int(config.get("execute_steps", 8))
        self.device = torch.device(str(config.get("device", "cuda:0")))
        suite = context.benchmark.suite
        keys = {checkpoint_key(task) for group in SUITES[suite] for task in GROUPS[group]}
        self.paths = {key: root / config["checkpoints"][key] for key in sorted(keys)}
        for key in self.paths:
            self._model(key)
        return PolicyCapabilities(
            supported_benchmarks=(BENCHMARK,),
            required_features=("language.instruction",),
            supported_suites={BENCHMARK: tuple(SUITES)},
            max_batch_size=64,
            supported_action_chunk_sizes=(self.execute_steps,),
            action_representations={BENCHMARK: ACTION_REPRESENTATION},
        )

    def _model(self, key: str) -> tuple[Any, dict[str, Any]]:
        if key not in self.models:
            from dp_policy.server import _load_policy

            policy, metadata = _load_policy(self.paths[key], str(self.device))
            if metadata["variant"] != self.variant:
                raise ValueError(f"{key} is a {metadata['variant']} checkpoint, not {self.variant}")
            if not 1 <= self.execute_steps <= int(metadata["available_action_steps"]):
                raise ValueError(f"execute_steps exceeds the {key} prediction horizon")
            if self.device.type == "cuda" and os.environ.get("EVO_DEXBENCH_CUDA_GRAPHS", "1") != "0":
                import graph_sampler

                graph_sampler.install(policy)
            self.models[key] = (policy, metadata)
        return self.models[key]

    def reset(self, context: EpisodeContext) -> None:
        self.tasks[context.episode_id] = context.task_id

    def act(self, observation: Observation) -> ActionChunk:
        return self.act_batch((observation,))[0]

    def act_batch(self, observations: tuple[Observation, ...]) -> tuple[ActionChunk, ...]:
        import torch

        keys = {checkpoint_key(self.tasks[observation.episode_id]) for observation in observations}
        if len(keys) != 1:
            raise ValueError("an Evo-DexBench batch must come from one task group checkpoint")
        policy, metadata = self._model(keys.pop())
        batch = [model_inputs(observation, metadata) for observation in observations]
        inputs = {
            key: torch.from_numpy(np.stack([row[key] for row in batch])).to(self.device, dtype=torch.float32)
            for key in metadata["observation_keys"]
        }
        with torch.inference_mode():
            actions = policy.predict_action(inputs)["action"].detach().cpu().numpy().astype(np.float32)
        expected = (len(batch), int(metadata["available_action_steps"]), int(metadata["action_dim"]))
        if actions.shape != expected:
            raise ValueError(f"checkpoint returned {actions.shape}, expected {expected}")
        return tuple(ActionChunk(np.ascontiguousarray(row[: self.execute_steps])) for row in actions)

    def end_episode(self, episode_id: str) -> None:
        self.tasks.pop(episode_id, None)

    def unload(self) -> None:
        self.models.clear()


def model_inputs(observation: Observation, metadata: dict[str, Any]) -> dict[str, np.ndarray]:
    """Upstream ``DPAdapter.observation`` applied to each of the two frames."""
    from dex_benchmark.policy_api.observation import compact_runtime_state, rgb_chw_float
    from dex_benchmark.policy_api.policy_features import roles_for_embodiment

    features = observation.features
    roles = roles_for_embodiment(metadata["embodiment"])
    result: dict[str, np.ndarray] = {}
    for key in metadata["observation_keys"]:
        if key == "state":
            frames = []
            for index in range(2):
                manipulators = {
                    role: {
                        "end_effector_pose": features[f"state.{role}.end_effector_pose"][index],
                        "joint_position": features[f"state.{role}.joint_position"][index],
                    }
                    for role in roles
                }
                frames.append(compact_runtime_state(manipulators, roles))
            result[key] = np.stack(frames)
        else:
            camera = key.removesuffix("_rgb")
            images = features[f"images.{camera}"]
            result[key] = np.stack([rgb_chw_float(image, name=key) for image in images])
    return result


def main() -> None:
    from roboscope_kit.server import run

    run(EvoDexBenchDiffusionPolicy())


if __name__ == "__main__":
    main()
