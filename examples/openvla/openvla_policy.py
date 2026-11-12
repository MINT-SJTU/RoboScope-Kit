# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""OpenVLA's four LIBERO checkpoints behind the public Kit Policy lifecycle."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np

from roboscope_kit.policy import (
    ActionChunk,
    EpisodeContext,
    Observation,
    Policy,
    PolicyCapabilities,
    SessionContext,
)

CODE_REPOSITORY = "openvla/openvla-7b"
CODE_REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def prepare_image(canonical: np.ndarray) -> np.ndarray:
    """Match upstream JPEG/Lanczos resize and its 90%-area TensorFlow crop.

    Kit already rotated the primary camera by 180 degrees. Do not rotate it again.
    """
    import tensorflow as tf

    if canonical.dtype != np.uint8 or canonical.shape != (256, 256, 3):
        raise ValueError("OpenVLA expects Kit's uint8 256x256 RGB primary camera")
    with tf.device("/CPU:0"):
        image = tf.image.encode_jpeg(canonical)
        image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8)
        image = tf.image.resize(image, (224, 224), method="lanczos3", antialias=True)
        resized = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
        image = tf.image.convert_image_dtype(tf.convert_to_tensor(resized), tf.float32)
        side = tf.reshape(tf.clip_by_value(tf.sqrt(0.9), 0, 1), (1,))
        offset = (1 - side) / 2
        boxes = tf.stack([offset, offset, offset + side, offset + side], axis=1)
        image = tf.image.crop_and_resize(tf.expand_dims(image, 0), boxes, tf.range(1), (224, 224))[0]
        image = tf.image.convert_image_dtype(tf.clip_by_value(image, 0, 1), tf.uint8, saturate=True)
        return image.numpy()


def environment_action(model_action: np.ndarray) -> np.ndarray:
    action = np.asarray(model_action).copy()
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("OpenVLA must return seven finite action values")
    action[-1] = -np.sign(2 * action[-1] - 1)
    return action.astype(np.float32)


class OpenVLAPolicy(Policy):
    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.episodes: dict[str, EpisodeContext] = {}

    def load(self, context: SessionContext) -> PolicyCapabilities:
        if context.benchmark.name != "libero" or context.benchmark.suite not in SUITES:
            raise ValueError("This Policy supports the four fine-tuned OpenVLA LIBERO suites only")
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
        import torch
        from transformers import (
            AutoImageProcessor,
            AutoModelForVision2Seq,
            AutoTokenizer,
        )
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        if not torch.cuda.is_available():
            raise RuntimeError("OpenVLA BF16/FlashAttention evaluation requires CUDA")
        seed = int(os.environ.get("OPENVLA_SEED", "7"))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.suite = context.benchmark.suite
        checkpoint = Path(context.checkpoint_path)
        statistics = json.loads((checkpoint / "dataset_statistics.json").read_text())
        if self.suite not in statistics:
            raise ValueError(f"Checkpoint has no normalization statistics for {self.suite}")
        local = {"trust_remote_code": True, "code_revision": CODE_REVISION, "local_files_only": True}
        self.model = AutoModelForVision2Seq.from_pretrained(
            str(checkpoint),
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            **local,
        ).to("cuda:0")
        self.model.norm_stats = statistics
        self.model.eval()
        # Construct the processor from explicitly pinned components. ProcessorMixin's
        # nested auto-loaders otherwise lose code_revision in Transformers 4.40.1.
        image_processor = AutoImageProcessor.from_pretrained(str(checkpoint), **local)
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
        processor_class = get_class_from_dynamic_module(
            "processing_prismatic.PrismaticProcessor",
            CODE_REPOSITORY,
            revision=CODE_REVISION,
            local_files_only=True,
        )
        self.processor = processor_class(image_processor=image_processor, tokenizer=tokenizer)
        return PolicyCapabilities(
            supported_benchmarks=("libero",),
            supported_suites={"libero": (self.suite,)},
            required_features=("images.primary", "language.instruction"),
            supported_action_chunk_sizes=(1,),
            action_representations={"libero": "libero.delta-eef-v1"},
        )

    def reset(self, context: EpisodeContext) -> None:
        self.episodes[context.episode_id] = context

    def act(self, observation: Observation) -> ActionChunk:
        import torch
        from PIL import Image

        if observation.episode_id not in self.episodes:
            raise ValueError(f"Unknown episode {observation.episode_id}")
        canonical = np.asarray(observation.features["images.primary"])
        cropped = prepare_image(canonical)
        instruction = str(observation.features["language.instruction"])
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inputs = self.processor(prompt, Image.fromarray(cropped).convert("RGB"))
        inputs = inputs.to("cuda:0", dtype=torch.bfloat16)
        with torch.inference_mode():
            predicted = self.model.predict_action(**inputs, unnorm_key=self.suite, do_sample=False)
        action = environment_action(predicted)
        return ActionChunk(actions=action)

    def end_episode(self, episode_id: str) -> None:
        self.episodes.pop(episode_id, None)

    def unload(self) -> None:
        self.episodes.clear()
        self.model = None
        self.processor = None


if __name__ == "__main__":
    from roboscope_kit.server import run

    run(OpenVLAPolicy())
