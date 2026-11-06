# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""PerAct Policy example for RoboScope Kit's ``rlbench`` benchmark, suite ``peract_18``.

Serves the released PerAct 600k multi-task checkpoint with PerAct's own agent code (the pinned
checkout the RLBench runtime builds, located by ``PERACT_ROOT``). ``--checkpoint-source`` is
the directory ``download_assets.py`` fills; every file is verified against its pinned SHA-256.

Everything model-specific stays here, as PerAct's ``eval.py`` does it:

- the agent is built by PerAct's ``peract_bc.launch_utils.create_agent`` from the checkpoint's
  ``config.yaml`` and loaded with ``load_weights``; CLIP RN50 is read from the pinned local file
  instead of being downloaded;
- each observation becomes the dict YARR's ``RolloutGenerator`` feeds the agent: per camera the
  channel-first RGB and point cloud with the camera's extrinsics and intrinsics, the
  ``low_dim_state`` of PerAct's environment wrapper (gripper open, finger positions clipped to
  [0, 0.04], and the time feature ``(1 - i / (episode_length - 1)) * 2 - 1`` with ``i`` the
  wrapper's step counter as ``eval.py`` reads it, ``max(t - 1, 0)`` before action ``t``), and the CLIP
  tokens of the instruction, each with batch and time dimensions of one;
- the voxelization, the Perceiver, and the arg-max over translation, rotation, gripper, and
  collision classes are PerAct's own ``act``, called with ``deterministic=True``.

PerAct's agent is stateless across steps, so one replica serves concurrent episodes; only the
episode length is kept per episode. Run it with the RLBench runtime's Python, which has the
pinned PerAct dependencies (see ``roboscope-policy.json``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from assets import CHECKPOINT_SEED_DIR, CHECKPOINT_WEIGHT, CLIP_RN50, verify  # noqa: E402

from roboscope_kit.policy import (  # noqa: E402
    ActionChunk,
    EpisodeContext,
    Observation,
    Policy,
    PolicyCapabilities,
    SessionContext,
)

BENCHMARK = "rlbench"
SUITE = "peract_18"
ACTION_REPRESENTATION = "rlbench.ee-pose-planning-v1"
CAMERAS = ("front", "left_shoulder", "right_shoulder", "wrist")
REQUIRED_FEATURES = (
    *(f"images.{camera}" for camera in CAMERAS),
    *(
        f"state.{camera}_{name}"
        for camera in CAMERAS
        for name in ("point_cloud", "camera_extrinsics", "camera_intrinsics")
    ),
    "state.gripper_open",
    "state.gripper_joint_positions",
    "language.instruction",
)
# PerAct's CustomMultiTaskRLBenchEnv clips the finger positions to this range.
GRIPPER_JOINT_RANGE = (0.0, 0.04)


class PerActPolicy(Policy):
    def __init__(self) -> None:
        self.agent = None
        self.device = None
        self.tokenize = None
        self.episode_lengths: dict[str, int] = {}

    def load(self, context: SessionContext) -> PolicyCapabilities:
        if context.benchmark.name != BENCHMARK or context.benchmark.suite != SUITE:
            raise ValueError(
                f"this example serves {BENCHMARK}/{SUITE}; "
                f"got {context.benchmark.name}/{context.benchmark.suite}"
            )
        root = Path(context.checkpoint_path)
        verify(root)
        peract_root = os.environ.get("PERACT_ROOT")
        if not peract_root or not (Path(peract_root) / "agents" / "peract_bc").is_dir():
            raise RuntimeError("PERACT_ROOT must point to the RLBench runtime's PerAct checkout (src/peract)")
        sys.path.insert(0, peract_root)

        import torch  # noqa: PLC0415
        from agents import peract_bc  # noqa: PLC0415
        from clip import tokenize  # noqa: PLC0415  (OpenAI CLIP, as YARR's RLBench env uses)
        from helpers import utils  # noqa: PLC0415
        from helpers.clip.core import clip as peract_clip  # noqa: PLC0415
        from omegaconf import OmegaConf  # noqa: PLC0415

        clip_path = str(root / CLIP_RN50.relative_path)
        # PerAct's QAttention agent calls load_clip("RN50"), which downloads to ~/.cache/clip;
        # hand it the pinned, verified file instead.
        peract_clip._download = lambda url, root=None: clip_path

        seed_dir = root / CHECKPOINT_SEED_DIR
        train_cfg = OmegaConf.load(seed_dir / "config.yaml")
        if list(train_cfg.rlbench.cameras) != list(CAMERAS):
            raise ValueError(
                f"checkpoint cameras {list(train_cfg.rlbench.cameras)} differ from {list(CAMERAS)}"
            )
        # eval.py's device choice: CUDA device 0 with cuDNN benchmarking enabled.
        self.device = utils.get_device(0)
        agent = peract_bc.launch_utils.create_agent(train_cfg)
        agent.build(training=False, device=self.device)
        agent.load_weights(str(seed_dir / "weights" / str(CHECKPOINT_WEIGHT)))
        self.agent = agent
        self.tokenize = tokenize
        self._torch = torch
        return PolicyCapabilities(
            supported_benchmarks=(BENCHMARK,),
            supported_suites={BENCHMARK: (SUITE,)},
            required_features=REQUIRED_FEATURES,
            max_batch_size=1,
            supported_action_chunk_sizes=(1,),
            action_representations={BENCHMARK: ACTION_REPRESENTATION},
        )

    def reset(self, context: EpisodeContext) -> None:
        self.episode_lengths[context.episode_id] = int(context.metadata["episode_length"])
        self.agent.reset()

    def act(self, observation: Observation) -> ActionChunk:
        episode_length = self.episode_lengths.get(observation.episode_id)
        if episode_length is None:
            raise RuntimeError(f"PerAct action requested for unknown episode {observation.episode_id}")
        features = observation.features
        obs: dict[str, np.ndarray] = {}
        for camera in CAMERAS:
            obs[f"{camera}_rgb"] = np.transpose(features[f"images.{camera}"], (2, 0, 1))
            obs[f"{camera}_point_cloud"] = np.transpose(
                features[f"state.{camera}_point_cloud"], (2, 0, 1)
            ).astype(np.float32)
            obs[f"{camera}_camera_extrinsics"] = features[f"state.{camera}_camera_extrinsics"].astype(
                np.float32
            )
            obs[f"{camera}_camera_intrinsics"] = features[f"state.{camera}_camera_intrinsics"].astype(
                np.float32
            )
        # PerAct's wrapper computes the time feature from its step counter before incrementing it
        # (CustomMultiTaskRLBenchEnv.step calls extract_obs, then self._i += 1), so the observation
        # after action t carries t - 1: the agent sees 1.0 at both its first and second action.
        time_index = max(observation.step - 1, 0)
        time_feature = (1.0 - (time_index / float(episode_length - 1))) * 2.0 - 1.0
        obs["low_dim_state"] = np.concatenate(
            [
                features["state.gripper_open"],
                np.clip(features["state.gripper_joint_positions"], *GRIPPER_JOINT_RANGE),
                [time_feature],
            ]
        ).astype(np.float32)
        obs["lang_goal_tokens"] = self.tokenize([features["language.instruction"]])[0].numpy()
        torch = self._torch
        # RolloutGenerator: torch.tensor([history]) with a one-step history -> (1, 1, ...).
        prepped = {key: torch.tensor(np.asarray([[value]]), device=self.device) for key, value in obs.items()}
        # eval.py runs act with autograd enabled; no_grad gives the same values without keeping
        # activations, which roughly halves the replica's GPU memory.
        with torch.no_grad():
            result = self.agent.act(0, prepped, deterministic=True)
        # float64, exactly the array PerAct's eval.py passes to RLBench.
        return ActionChunk(actions=np.asarray(result.action, dtype=np.float64).reshape(1, -1))

    def end_episode(self, episode_id: str) -> None:
        self.episode_lengths.pop(episode_id, None)

    def unload(self) -> None:
        self.episode_lengths.clear()
        self.agent = None


def main() -> None:
    from roboscope_kit.server import run  # noqa: PLC0415

    run(PerActPolicy())


if __name__ == "__main__":
    main()
