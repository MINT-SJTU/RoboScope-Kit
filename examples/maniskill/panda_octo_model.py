"""RDT-author released ManiSkill fine-tuned Octo, using its original EE7 decoding."""

from __future__ import annotations

import json
import os
import sys
from collections import deque
from functools import partial
from pathlib import Path

import numpy as np
from download_panda_octo import PIN_PATH, verify


def load_single_device(checkpoint: Path, step: int):
    """Use the source model definition while restoring its eight-device tensors on one device."""
    import flax
    import jax
    import orbax.checkpoint as ocp
    from octo.model.octo_model import OctoModel
    from octo.model.octo_module import OctoModule
    from octo.utils.spec import ModuleSpec

    config = json.loads((checkpoint / "config.json").read_text())
    example = flax.serialization.msgpack_restore((checkpoint / "example_batch.msgpack").read_bytes())
    statistics = json.loads((checkpoint / "dataset_statistics.json").read_text())
    statistics = jax.tree_map(np.array, statistics, is_leaf=lambda value: not isinstance(value, dict))
    module = OctoModule.create(**config["model"])
    shapes = jax.eval_shape(partial(module.init, train=False), jax.random.PRNGKey(0),
                            example["observation"], example["task"],
                            example["observation"]["timestep_pad_mask"])["params"]
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    restore_args = jax.tree_map(lambda value: ocp.ArrayRestoreArgs(
        restore_type=jax.Array, dtype=value.dtype, global_shape=value.shape, sharding=sharding,
    ), shapes)
    manager = ocp.CheckpointManager(str(checkpoint), ocp.PyTreeCheckpointer())
    params = manager.restore(step, shapes, restore_kwargs={"restore_args": restore_args})
    manager.close()
    text_processor = ModuleSpec.instantiate(config["text_processor"])()
    return OctoModel(module=module, params=params, text_processor=text_processor,
                     example_batch=example, config=config, dataset_statistics=statistics)


class ManiSkillFinetunedOcto:
    """Image/language model with independent episode RNG, history and four-action queues."""

    def __init__(self, checkpoint: str | Path) -> None:
        checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
        pin = json.loads(PIN_PATH.read_text(encoding="utf-8"))
        for name, record in pin["files"].items():
            if not verify(checkpoint / name, record):
                raise ValueError(f"ManiSkill Octo checkpoint differs from its pin: {name}")
        source = checkpoint / "octo-source"
        loaded = sys.modules.get("octo")
        if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(source):
            raise RuntimeError("Run ManiSkill Octo v1.5 separately from the Bridge Octo v1.0 process")
        for package in ("octo", "dlimp"):
            package_root = checkpoint / f"{package}-source"
            expected = pin[f"{package}_source_archive"]["sha256"]
            if (package_root / ".source-sha256").read_text().strip() != expected:
                raise ValueError(f"Unverified {package} source directory")
            sys.path.insert(0, str(package_root))
        # An isolated import overlay adds TFDS without changing the Bridge environment.
        sys.path.insert(0, str(checkpoint / "python-deps"))
        tokenizer = pin["tokenizer"]
        cache = checkpoint / "tokenizer-cache"
        tokenizer_root = cache / "models--t5-base" / "snapshots" / tokenizer["revision"]
        for name, record in tokenizer["files"].items():
            if not verify(tokenizer_root / name, record):
                raise ValueError(f"Offline tokenizer differs from its pin: {name}")
        os.environ.update(HF_HUB_CACHE=str(cache), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          TOKENIZERS_PARALLELISM="false", TFDS_DISABLE_GCS="1")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        flags = os.environ.get("XLA_FLAGS", "")
        if "--xla_gpu_deterministic_ops=true" not in flags:
            os.environ["XLA_FLAGS"] = (flags + " --xla_gpu_deterministic_ops=true").strip()
        import tensorflow as tf

        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.set_visible_devices([], "GPU")
        import jax
        self.tf, self.jax = tf, jax
        self.model = load_single_device(checkpoint, pin["step"])
        tokenizers = self.model.config["model"]["observation_tokenizers"]
        if set(tokenizers) != {"primary", "wrist"}:
            raise ValueError("Expected the released image-only Octo observation tokenizers")
        self.uses_proprio = False
        self.tasks: dict[str, dict] = {}
        self.episodes: dict[str, dict] = {}

        def sample_one(model, observation, task, key):
            observation = jax.tree_map(lambda value: value[None], observation)
            task = jax.tree_map(lambda value: value[None], task)
            # Match eval_sim/eval_octo.py: normalized output goes directly to native EE control.
            return model.sample_actions(observation, task, rng=key)[0]

        self.sample_batch = jax.jit(jax.vmap(sample_one, in_axes=(None, 0, 0, 0)))

    def reset(self, episode_id: str, instruction: str) -> None:
        if instruction not in self.tasks:
            self.tasks[instruction] = self.model.create_tasks(texts=[instruction])
        self.episodes[episode_id] = {
            "instruction": instruction, "images": deque(maxlen=2), "masks": deque(maxlen=2),
            "pending": deque(), "rng": self.jax.random.PRNGKey(0),
        }

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        image = self.tf.convert_to_tensor(image, dtype=self.tf.float32)[None]
        resized = self.tf.image.resize(image, (256, 256), method="lanczos3", antialias=True)
        # Preserve the author's truncating uint8 conversion, including Lanczos overshoot behavior.
        return self.tf.squeeze(resized, axis=0).numpy().astype(np.uint8)

    def act_batch(
        self, episode_ids: list[str], images: list[np.ndarray], proprios=None,
    ) -> list[np.ndarray]:
        """Return raw normalized EE7; proprio is unused because there is no state tokenizer."""
        if len(images) != len(episode_ids) or len(set(episode_ids)) != len(episode_ids):
            raise ValueError("Expected one image per distinct episode")
        pending = []
        for index, (episode_id, image) in enumerate(zip(episode_ids, images, strict=True)):
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError("ManiSkill Octo requires HWC uint8 RGB")
            state = self.episodes[episode_id]
            frame = self.resize_image(image)
            if not state["images"]:
                state["images"].append(frame)
                state["masks"].append(False)
            state["images"].append(frame)
            state["masks"].append(True)
            if not state["pending"]:
                pending.append(index)
        if pending:
            states = [self.episodes[episode_ids[index]] for index in pending]
            keys = []
            for state in states:
                state["rng"], key = self.jax.random.split(state["rng"])
                keys.append(key)
            observations = {
                "image_primary": np.stack([np.stack(state["images"]) for state in states]),
                "timestep_pad_mask": np.stack([np.asarray(state["masks"]) for state in states]),
            }
            tasks = self.jax.tree_map(lambda *values: np.concatenate(values, axis=0),
                *[self.tasks[state["instruction"]] for state in states])
            actions = np.asarray(self.sample_batch(self.model, observations, tasks, np.stack(keys)))
            if actions.shape != (len(states), 4, 7) or not np.isfinite(actions).all():
                raise ValueError(f"Invalid ManiSkill Octo action prediction: {actions.shape}")
            for state, chunk in zip(states, actions, strict=True):
                state["pending"].extend(np.ascontiguousarray(action, dtype=np.float32) for action in chunk)
        return [self.episodes[episode_id]["pending"].popleft() for episode_id in episode_ids]

    def end_episode(self, episode_id: str) -> None:
        self.episodes.pop(episode_id, None)
