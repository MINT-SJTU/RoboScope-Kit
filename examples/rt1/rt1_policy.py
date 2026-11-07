# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""RT-1-X Policy example for RoboScope Kit's SimplerEnv benchmark.

Serves the released RT-1-X TF-Agents checkpoint on all three ``simpler_env`` suites
(Google Robot visual matching and variant aggregation, WidowX visual matching) through
SimplerEnv's own ``simpler_env.policies.rt1.rt1_model.RT1Inference``, so every
model-specific step is upstream's code:

- image ``tf.image.resize_with_pad`` to 320x256 and the Universal Sentence Encoder
  instruction embedding;
- Google Robot: the small gripper-action filter, axis-angle rotation, raw gripper value;
- WidowX: unnormalizing translation/rotation, RPY to axis-angle, inverting and binarizing
  the gripper;
- the recurrent TF-Agents policy state, reset when the instruction changes (SimplerEnv's
  long-horizon drawer task advances its instruction mid-episode).

One replica serves many concurrent episodes. ``RT1Inference`` keeps one episode's state on
the instance, so the Policy keeps that state per episode and swaps it in around each
upstream ``step``/``reset`` call. Two ``RT1Inference`` instances (``google_robot`` and
``widowx_bridge``) share one loaded SavedModel and encoder.

The returned action is the 7-value vector SimplerEnv's evaluator passes to ``env.step``
(``world_vector``, ``rot_axangle``, ``gripper``) plus ``terminate_episode[0]``, as float64.

Run it with the SimplerEnv runtime's own Python, which has the pinned TensorFlow,
TF-Agents, and SimplerEnv, and put the kit source on ``PYTHONPATH`` (see
``roboscope-policy.json``). ``ROBOSCOPE_RT1_TF_MEMORY_MB`` caps TensorFlow's GPU memory
the way ``main_inference.py --tf-memory-limit`` does (default 3072, upstream's default).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from assets import RT1X, UNIVERSAL_SENTENCE_ENCODER, verify_directory  # noqa: E402

from roboscope_kit.policy import (  # noqa: E402
    ActionChunk,
    EpisodeContext,
    Observation,
    Policy,
    PolicyCapabilities,
    SessionContext,
)

BENCHMARK = "simpler_env"
SUITES = ("google_robot_visual_matching", "google_robot_variant_aggregation", "widowx_visual_matching")
ACTION_REPRESENTATION = "simpler_env.ee-delta-axangle-v1"
POLICY_SETUPS = ("google_robot", "widowx_bridge")
EPISODE_FIELDS = (
    "observation",
    "tfa_time_step",
    "policy_state",
    "task_description",
    "task_description_embedding",
)


@dataclass
class EpisodeState:
    setup: str
    started: bool = False
    fields: dict[str, Any] | None = None


class RT1XPolicy(Policy):
    def __init__(self) -> None:
        self.models: dict[str, Any] = {}
        self.episodes: dict[str, EpisodeState] = {}

    def load(self, context: SessionContext) -> PolicyCapabilities:
        if context.benchmark.name != BENCHMARK or context.benchmark.suite not in SUITES:
            raise ValueError(
                f"this example serves {BENCHMARK} suites {', '.join(SUITES)}; "
                f"got {context.benchmark.name}/{context.benchmark.suite}"
            )
        root = Path(context.checkpoint_path)
        checkpoint = verify_directory(root, RT1X)
        encoder = verify_directory(root, UNIVERSAL_SENTENCE_ENCODER)

        import tensorflow as tf  # noqa: PLC0415

        # main_inference.py's GPU memory cap for its TensorFlow process.
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            limit = int(os.environ.get("ROBOSCOPE_RT1_TF_MEMORY_MB", "3072"))
            tf.config.set_logical_device_configuration(
                gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=limit)]
            )
        import tensorflow_hub as hub  # noqa: PLC0415
        from simpler_env.policies.rt1 import rt1_model  # noqa: PLC0415
        from tf_agents.policies import py_tf_eager_policy  # noqa: PLC0415

        first = rt1_model.RT1Inference(
            saved_model_path=str(checkpoint),
            lang_embed_model_path=str(encoder),
            policy_setup=POLICY_SETUPS[0],
        )
        self.models[POLICY_SETUPS[0]] = first
        # The second setup runs upstream's __init__ too, but reuses the loaded models.
        with (
            mock.patch.object(hub, "load", return_value=first.lang_embed_model),
            mock.patch.object(py_tf_eager_policy, "SavedModelPyTFEagerPolicy", return_value=first.tfa_policy),
        ):
            for setup in POLICY_SETUPS[1:]:
                self.models[setup] = rt1_model.RT1Inference(
                    saved_model_path=str(checkpoint),
                    lang_embed_model_path=str(encoder),
                    policy_setup=setup,
                )
        return PolicyCapabilities(
            supported_benchmarks=(BENCHMARK,),
            supported_suites={BENCHMARK: SUITES},
            required_features=("images.primary", "language.instruction"),
            max_batch_size=1,
            supported_action_chunk_sizes=(1,),
            action_representations={BENCHMARK: ACTION_REPRESENTATION},
        )

    def reset(self, context: EpisodeContext) -> None:
        setup = str(context.metadata.get("policy_setup", ""))
        if setup not in self.models:
            raise ValueError(
                f"unsupported SimplerEnv policy setup {setup!r}; expected one of {POLICY_SETUPS}"
            )
        self.episodes[context.episode_id] = EpisodeState(setup=setup)

    def act(self, observation: Observation) -> ActionChunk:
        state = self.episodes.get(observation.episode_id)
        if state is None:
            raise RuntimeError(f"RT-1 action requested for unknown episode {observation.episode_id}")
        model = self.models[state.setup]
        image = np.asarray(observation.features["images.primary"])
        instruction = str(observation.features["language.instruction"])
        if state.started:
            self._swap_in(model, state)
        else:
            # maniskill2_evaluator calls model.reset(task_description) right after env.reset().
            model.reset(instruction)
            state.started = True
        _raw, action = model.step(image, instruction)
        state.fields = {name: getattr(model, name) for name in EPISODE_FIELDS}
        terminate = float(np.asarray(action["terminate_episode"]).reshape(-1)[0])
        vector = np.concatenate(
            [
                np.asarray(action["world_vector"], dtype=np.float64).reshape(3),
                np.asarray(action["rot_axangle"], dtype=np.float64).reshape(3),
                np.asarray(action["gripper"], dtype=np.float64).reshape(1),
                np.asarray([terminate], dtype=np.float64),
            ]
        )
        return ActionChunk(actions=vector.reshape(1, -1))

    def end_episode(self, episode_id: str) -> None:
        self.episodes.pop(episode_id, None)

    def unload(self) -> None:
        self.episodes.clear()
        self.models.clear()

    @staticmethod
    def _swap_in(model: Any, state: EpisodeState) -> None:
        assert state.fields is not None
        for name, value in state.fields.items():
            setattr(model, name, value)


def main() -> None:
    from roboscope_kit.server import run  # noqa: PLC0415

    run(RT1XPolicy())


if __name__ == "__main__":
    main()
