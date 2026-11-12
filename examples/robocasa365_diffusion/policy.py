# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Official RoboCasa365 multi-task Diffusion Policy, with native five-slot inference.

The worker provides the two most recent *simulation* observations. The policy does
not substitute observations from successive inference calls for that history.
Each task owns an RNG stream, so independently scheduled task groups cannot consume
one another's diffusion noise. Within a task, batch cardinality and ordering must
match the pinned native evaluator for an action-level comparison.
"""

from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from download_model import file_sha256  # noqa: E402

from roboscope_kit import (  # noqa: E402
    ActionChunk,
    EpisodeContext,
    Observation,
    Policy,
    PolicyCapabilities,
    SessionContext,
)

BENCHMARK = "robocasa365"
SUITES = ("target50", "atomic_seen", "composite_seen", "composite_unseen")
ACTION_REPRESENTATION = "robocasa365.panda-omron-hybrid-v1"
FEATURE_MAP = {
    "robot0_agentview_right_image": "images.agentview_right",
    "robot0_agentview_left_image": "images.agentview_left",
    "robot0_eye_in_hand_image": "images.wrist",
    "robot0_base_to_eef_pos": "state.end_effector_position_relative",
    "robot0_base_to_eef_quat": "state.end_effector_rotation_relative",
    "robot0_gripper_qpos": "state.gripper_qpos",
}
EXPECTED_SHAPES = {
    "robot0_agentview_right_image": (3, 256, 256),
    "robot0_agentview_left_image": (3, 256, 256),
    "robot0_eye_in_hand_image": (3, 256, 256),
    "robot0_base_to_eef_pos": (3,),
    "robot0_base_to_eef_quat": (4,),
    "robot0_gripper_qpos": (2,),
    "lang_emb": (768,),
}


@dataclass
class RNGState:
    cpu: Any
    cuda: Any


@dataclass
class EpisodeState:
    task: str
    instruction: str | None = None
    language_embedding: np.ndarray | None = None


class RoboCasa365DiffusionPolicy(Policy):
    def __init__(self) -> None:
        self.model: Any = None
        self.device: Any = None
        self.encoder: Any = None
        self.tokenizer: Any = None
        self.initial_rng: RNGState | None = None
        self.streams: dict[str, RNGState] = {}
        self.episodes: dict[str, EpisodeState] = {}

    def load(self, context: SessionContext) -> PolicyCapabilities:
        if context.benchmark.name != BENCHMARK or context.benchmark.suite not in SUITES:
            raise ValueError(
                f"this Policy serves {BENCHMARK}/{SUITES}; got {context.benchmark}"
            )
        import dill
        import hydra
        import torch
        from omegaconf import OmegaConf
        from transformers import AutoTokenizer, CLIPTextModelWithProjection

        inputs = json.loads(
            Path(__file__).resolve().with_name("model-inputs.json").read_text()
        )
        path = Path(context.checkpoint_path)
        checkpoint = path / "model.ckpt" if path.is_dir() else path
        root = path if path.is_dir() else path.parent
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"missing official checkpoint: {checkpoint}; run download_model.py"
            )
        if file_sha256(checkpoint) != inputs["checkpoint"]["sha256"]:
            raise ValueError(
                f"{checkpoint} is not the pinned official RoboCasa365 checkpoint"
            )
        clip = root / "clip"
        if not (clip / "model.safetensors").is_file():
            raise FileNotFoundError(
                f"missing pinned CLIP encoder: {clip}; run download_model.py"
            )
        if (
            file_sha256(clip / "model.safetensors")
            != inputs["language_encoder"]["weights_sha256"]
        ):
            raise ValueError(f"{clip} does not contain the pinned CLIP weights")
        # The verified checkpoint is a trusted upstream pickle, including its OmegaConf config.
        with checkpoint.open("rb") as stream:
            payload = torch.load(
                stream, map_location="cpu", pickle_module=dill, weights_only=False
            )
        cfg = payload["cfg"]
        OmegaConf.register_new_resolver(
            "eval",
            lambda expression: eval(expression, {"__builtins__": {}}, {}),
            replace=True,
        )
        expected_target = (
            "diffusion_policy.policy.diffusion_transformer_hybrid_image_policy."
            "DiffusionTransformerHybridImagePolicy"
        )
        if cfg.policy._target_ != expected_target:
            raise ValueError(f"unsupported checkpoint Policy: {cfg.policy._target_}")
        shape_meta = OmegaConf.to_container(cfg.policy.shape_meta, resolve=True)
        actual_shapes = {
            key: tuple(value["shape"]) for key, value in shape_meta["obs"].items()
        }
        if actual_shapes != EXPECTED_SHAPES or tuple(shape_meta["action"]["shape"]) != (
            12,
        ):
            raise ValueError(
                f"unexpected checkpoint observation/action shapes: {shape_meta}"
            )
        if int(cfg.policy.n_obs_steps) != 2 or int(cfg.policy.n_action_steps) != 8:
            raise ValueError(
                "the official Policy requires two observation frames and eight actions"
            )
        if bool(cfg.task.abs_action):
            raise ValueError("the official multi-task Policy uses delta actions")
        self.device = torch.device(
            os.environ.get(
                "ROBOCASA365_POLICY_DEVICE",
                "cuda:0" if torch.cuda.is_available() else "cpu",
            )
        )
        # Match workspace construction's initialization seed. CLIP loads in an isolated
        # scope because the native evaluator loads CLIP in environment subprocesses.
        random.seed(int(cfg.training.seed))
        np.random.seed(int(cfg.training.seed))
        with torch.random.fork_rng(devices=self._rng_devices()):
            torch.manual_seed(int(cfg.training.seed))
            self.model = hydra.utils.instantiate(cfg.policy)
            key = "ema_model" if bool(cfg.training.use_ema) else "model"
            self.model.load_state_dict(payload["state_dicts"][key], strict=True)
            self.model.to(self.device)
            self.model.eval()
            self.initial_rng = self._capture_rng()
        del payload
        with torch.random.fork_rng(devices=self._rng_devices()):
            self.encoder = (
                CLIPTextModelWithProjection.from_pretrained(clip, local_files_only=True)
                .to("cpu")
                .eval()
            )
            self.tokenizer = AutoTokenizer.from_pretrained(clip, local_files_only=True)
        return PolicyCapabilities(
            supported_benchmarks=(BENCHMARK,),
            supported_suites={BENCHMARK: SUITES},
            required_features=(*FEATURE_MAP.values(), "language.instruction"),
            max_batch_size=5,
            supported_action_chunk_sizes=(8,),
            action_representations={BENCHMARK: ACTION_REPRESENTATION},
            observation_strides={BENCHMARK: 8},
        )

    def reset(self, context: EpisodeContext) -> None:
        if self.model is None or self.initial_rng is None:
            raise RuntimeError("load() must run before reset()")
        # BaseImagePolicy.reset is a no-op for this stateless diffusion model.
        # Temporal observations live in the worker and are reset there every episode.
        self.model.reset()
        self.episodes[context.episode_id] = EpisodeState(task=context.task_id)
        new_workload = (
            context.metadata.get("stream_episode") == 0
            and context.metadata.get("slot") == 0
        )
        if new_workload or context.task_id not in self.streams:
            self.streams[context.task_id] = RNGState(
                cpu=self.initial_rng.cpu.clone(),
                cuda=self.initial_rng.cuda.clone()
                if self.initial_rng.cuda is not None
                else None,
            )

    def prepare_observation(self, observation: Observation) -> dict[str, np.ndarray]:
        """Apply the pinned native wrapper's preprocessing to one history window."""
        import torch
        from robomimic.utils.obs_utils import process_frame

        state = self.episodes.get(observation.episode_id)
        if state is None:
            raise RuntimeError(f"unknown episode: {observation.episode_id}")
        instruction = observation.features["language.instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("RoboCasa365 instruction must be a nonempty string")
        if state.instruction is None:
            # Same CPU CLIPTextModelWithProjection and tokenizer calls as LangEncoder.
            with torch.random.fork_rng(devices=self._rng_devices()), torch.no_grad():
                tokens = self.tokenizer(
                    text=instruction,
                    add_special_tokens=True,
                    padding="max_length",
                    return_attention_mask=True,
                    return_tensors="pt",
                ).to("cpu")
                state.language_embedding = self.encoder(**tokens)["text_embeds"][
                    0
                ].numpy()
            state.instruction = instruction
        elif instruction != state.instruction:
            raise ValueError("language instruction changed during an episode")
        result = {}
        for policy_key, feature in FEATURE_MAP.items():
            value = np.asarray(observation.features[feature])
            image = policy_key.endswith("_image")
            expected = (2, 256, 256, 3) if image else (2, *EXPECTED_SHAPES[policy_key])
            if value.shape != expected:
                raise ValueError(
                    f"{feature} has shape {value.shape}; expected {expected}"
                )
            expected_dtype = np.dtype("uint8" if image else "float32")
            if value.dtype != expected_dtype:
                raise ValueError(
                    f"{feature} has dtype {value.dtype}; expected {expected_dtype}"
                )
            result[policy_key] = (
                process_frame(value, channel_dim=3, scale=255.0) if image else value
            )
        result["lang_emb"] = np.repeat(state.language_embedding[None], 2, axis=0)
        return result

    def act(self, observation: Observation) -> ActionChunk:
        return self.act_batch((observation,))[0]

    def act_batch(
        self, observations: tuple[Observation, ...]
    ) -> tuple[ActionChunk, ...]:
        import torch

        if not observations or len(observations) > 5:
            raise ValueError(
                "official Diffusion Policy supports one to five ordered native slots"
            )
        tasks = {
            self.episodes[observation.episode_id].task for observation in observations
        }
        if len(tasks) != 1:
            raise ValueError(
                "a native inference batch must belong to one RoboCasa365 task"
            )
        prepared = [
            self.prepare_observation(observation) for observation in observations
        ]
        tensors = {
            key: torch.from_numpy(np.stack([obs[key] for obs in prepared])).to(
                self.device
            )
            for key in prepared[0]
        }
        task = next(iter(tasks))
        stream = self.streams[task]
        with torch.random.fork_rng(devices=self._rng_devices()), torch.no_grad():
            torch.set_rng_state(stream.cpu)
            if stream.cuda is not None:
                torch.cuda.set_rng_state(stream.cuda, self.device)
            actions = (
                self.model.predict_action(tensors)["action"].detach().cpu().numpy()
            )
            self.streams[task] = self._capture_rng()
        if actions.shape != (len(observations), 8, 12) or not np.all(
            np.isfinite(actions)
        ):
            raise RuntimeError(
                f"invalid official Diffusion Policy output: {actions.shape}"
            )
        return tuple(
            ActionChunk(actions=np.ascontiguousarray(action, dtype=np.float32))
            for action in actions
        )

    def end_episode(self, episode_id: str) -> None:
        self.episodes.pop(episode_id, None)

    def unload(self) -> None:
        self.episodes.clear()
        self.streams.clear()
        self.model = None
        self.encoder = None
        self.tokenizer = None

    def _rng_devices(self) -> list[int]:
        return [self.device.index or 0] if self.device.type == "cuda" else []

    def _capture_rng(self) -> RNGState:
        import torch

        return RNGState(
            cpu=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state(self.device)
            if self.device.type == "cuda"
            else None,
        )


def main() -> None:
    from roboscope_kit.server import run

    run(RoboCasa365DiffusionPolicy())


if __name__ == "__main__":
    main()
