# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""BEHAVIOR2026VLA Policy example for RoboScope Kit.

Serves ``IliaLarchenko/BEHAVIOR2026VLA``, the Robot Learning Collective's single π0.5-based checkpoint
for all 100 tasks of the 2026 BEHAVIOR Challenge (the learned policy of their 9th-place solution,
arXiv:2611.14432), on the ``behavior`` benchmark's ``challenge_2026_rgb224`` suite: the evaluator's
``DefaultWrapper`` with 224x224 RGB cameras, the observation the model was trained on and that its
authors evaluated it with.

The Policy runs in the solution's own environment (Python 3.11 with JAX and openpi, built by
``build-model-env.sh``) with the kit source on ``PYTHONPATH``, and runs the solution's evaluation
code in process instead of behind its websocket server. Per episode it creates the rollout state
(``B1KPolicyWrapper``) around the one loaded JAX policy and steps it with ``act_batched``, exactly
as ``scripts/serve_b1k.py`` does for the native evaluator's request of one environment, which
carries the native observation keys with a leading batch dimension of one. Everything
model-specific stays in that code: the letter-box resize, the state taken from the 61-dim
proprioception, the task and stage embeddings selected by ``task_id``, 30 predicted actions with 20
denoising steps of which the first 26 are executed over 13 control steps, rolling inpainting,
stage voting and the gripper rule. The model is queried on every 13th step; the other steps
replay its chunk.

The Policy process must start in the solution checkout (``cwd`` in ``roboscope-policy.json``), as
``serve_b1k.py`` does, because the training config resolves its asset directory from there.

``--checkpoint-source`` is the downloaded snapshot directory (``params/`` and ``assets/``).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# serve_b1k.py's JAX memory settings, set before JAX is imported; the environment can override them,
# e.g. XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 for two replicas on one GPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

from roboscope_kit.policy import (  # noqa: E402
    ActionChunk,
    EpisodeContext,
    Observation,
    Policy,
    PolicyCapabilities,
    SessionContext,
)

BENCHMARK = "behavior"
SUITE = "challenge_2026_rgb224"
ACTION_REPRESENTATION = "behavior.r1pro-joint-v1"
ACTION_DIM = 23
# SDK feature -> the native observation key serve_b1k.py reads.
NATIVE_KEYS = {
    "images.head": "robot_r1::robot_r1:zed_link:Camera:0::rgb",
    "images.left_wrist": "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
    "images.right_wrist": "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
    "state.proprioception": "robot_r1::proprio",
    "state.task_id": "task_id",
}
TRAIN_CONFIG = "configs/train_b1k_2026.yaml"
# serve_b1k.py's default number of flow-matching denoising steps.
NUM_STEPS = 20
CHECKPOINT_FILES = ("params/_METADATA", "params/manifest.ocdbt", "assets/b1k_2026/norm_stats.json")


def native_request(observation: Observation) -> dict[str, np.ndarray]:
    """The native evaluator's observation keys serve_b1k.py reads, batched as one environment."""

    return {
        key: np.ascontiguousarray(np.asarray(observation.features[feature])[None])
        for feature, key in NATIVE_KEYS.items()
    }


class Behavior2026VlaPolicy(Policy):
    def __init__(self) -> None:
        self.policy = None
        self.rollout = None
        self.active_episode: str | None = None
        # Per episode: control steps, model calls, and seconds spent in act (model and replay).
        self.timing = {"steps": 0, "model_calls": 0, "act_s": 0.0}

    def load(self, context: SessionContext) -> PolicyCapabilities:
        if context.benchmark.name != BENCHMARK or context.benchmark.suite != SUITE:
            raise ValueError(
                f"this example serves {BENCHMARK}/{SUITE}; "
                f"got {context.benchmark.name}/{context.benchmark.suite}"
            )
        checkpoint = Path(context.checkpoint_path)
        missing = [name for name in CHECKPOINT_FILES if not (checkpoint / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{checkpoint} is not a BEHAVIOR2026VLA snapshot; missing {', '.join(missing)}"
            )
        if not Path(TRAIN_CONFIG).is_file():
            raise FileNotFoundError(
                f"start the Policy in the solution checkout; {TRAIN_CONFIG} is not in {Path.cwd()}"
            )
        from b1k.shared.cuda_env import ensure_packaged_ptxas  # noqa: PLC0415

        ensure_packaged_ptxas()
        from b1k.policies import policy_config  # noqa: PLC0415
        from b1k.training.yaml_train_config import resolve_train_config  # noqa: PLC0415

        config = resolve_train_config(TRAIN_CONFIG)
        self.policy = policy_config.create_trained_policy(
            config, str(checkpoint), sample_kwargs={"num_steps": NUM_STEPS}
        )
        self._warm_up()
        return PolicyCapabilities(
            supported_benchmarks=(BENCHMARK,),
            supported_suites={BENCHMARK: (SUITE,)},
            required_features=tuple(NATIVE_KEYS),
            max_batch_size=1,
            supported_action_chunk_sizes=(1,),
            action_representations={BENCHMARK: ACTION_REPRESENTATION},
        )

    def _warm_up(self) -> None:
        """Compile both model calls of a rollout (the first chunk, then one with an inpainting anchor)
        on a throwaway rollout, so the first ``act`` of an episode is not a JAX compilation; with
        several replicas compiling at once on a busy host it can outlast the worker's request timeout."""

        from b1k.shared.eval_b1k_wrapper import B1KPolicyWrapper, B1KWrapperConfig, act_batched  # noqa: PLC0415

        image = np.zeros((1, 224, 224, 3), dtype=np.uint8)
        request = {key: image for feature, key in NATIVE_KEYS.items() if feature.startswith("images.")}
        request["robot_r1::proprio"] = np.zeros((1, 61), dtype=np.float32)
        request["task_id"] = np.zeros((1, 1), dtype=np.int64)
        rollout = [B1KPolicyWrapper(self.policy, config=B1KWrapperConfig())]
        rollout[0].reset()
        while rollout[0].prediction_count < 2:
            act_batched(rollout, dict(request))
        self.policy.reset()

    def reset(self, context: EpisodeContext) -> None:
        from b1k.shared.eval_b1k_wrapper import B1KPolicyWrapper, B1KWrapperConfig  # noqa: PLC0415

        if self.active_episode is not None and self.active_episode != context.episode_id:
            raise RuntimeError(f"episode {self.active_episode} did not end before {context.episode_id}")
        self.active_episode = context.episode_id
        # A new evaluator connection to serve_b1k.py: fresh rollout state, reset as on its reset message.
        self.rollout = [B1KPolicyWrapper(self.policy, config=B1KWrapperConfig())]
        self.rollout[0].reset()
        self.timing = dict.fromkeys(self.timing, 0)

    def act(self, observation: Observation) -> ActionChunk:
        from b1k.shared.eval_b1k_wrapper import act_batched  # noqa: PLC0415

        if observation.episode_id != self.active_episode:
            raise RuntimeError(f"got a step of {observation.episode_id}, not {self.active_episode}")
        started = time.perf_counter()
        calls = self.rollout[0].prediction_count
        action = act_batched(self.rollout, native_request(observation)).cpu().numpy().astype(np.float32)
        if action.shape != (1, ACTION_DIM):
            raise RuntimeError(f"BEHAVIOR2026VLA action shape is {action.shape}, expected (1, {ACTION_DIM})")
        self.timing["steps"] += 1
        self.timing["model_calls"] += self.rollout[0].prediction_count - calls
        self.timing["act_s"] += time.perf_counter() - started
        return ActionChunk(actions=action)

    def end_episode(self, episode_id: str) -> None:
        if self.active_episode == episode_id:
            timing = {"episode_id": episode_id, **self.timing}
            print(json.dumps({"behavior2026vla_episode_timing": timing}), file=sys.stderr, flush=True)
            self.active_episode = None
            self.rollout = None

    def unload(self) -> None:
        self.policy = None
        self.rollout = None
        # Importing omnigibson (for its evaluation constants) creates a temporary directory.
        omnigibson = sys.modules.get("omnigibson")
        if omnigibson is not None:
            shutil.rmtree(omnigibson.tempdir, ignore_errors=True)


def main() -> None:
    # The solution imports OmniGibson, whose package import installs a SIGINT handler, which Python
    # allows only in the main thread; the kit handles requests in threads, so import it here first,
    # as serve_b1k.py does at its top level.
    import omnigibson  # noqa: F401, PLC0415

    from roboscope_kit.server import run  # noqa: PLC0415

    run(Behavior2026VlaPolicy())


if __name__ == "__main__":
    main()
