"""Pinned Octo-small Bridge inference with independent episode state.

Preprocessing, action unnormalization and temporal ensembling follow the public
SimplerEnv OctoInference widowx_bridge implementation at eddb569. Rotation is
encoded for ManiSkill3's native Euler XYZ controller. Each episode starts an
independent init_rng=0 stream, including its five initial key splits.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from collections import deque
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from download_octo import verify

INSTRUCTIONS = {
    "PutCarrotOnPlateInScene-v1": "put carrot on plate",
    "PutEggplantInBasketScene-v1": "put eggplant into yellow basket",
    "StackGreenCubeOnYellowCubeBakedTexInScene-v1": "stack the green block on the yellow block",
    "PutSpoonOnTableClothInScene-v1": "put the spoon on the towel",
}
CONTROL_MODE = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"


class BridgeOcto:
    def __init__(self, checkpoint: str | Path) -> None:
        checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
        pin = json.loads(Path(__file__).with_name("octo-checkpoint.json").read_text(encoding="utf-8"))
        for name, record in pin["files"].items():
            if not verify(checkpoint / name, record):
                raise ValueError(f"Octo checkpoint does not match its pin: {name}")
        tokenizer = pin["tokenizer"]
        cache = checkpoint / "tokenizer-cache"
        tokenizer_root = cache / "models--t5-base" / "snapshots" / tokenizer["revision"]
        for name, record in tokenizer["files"].items():
            if not verify(tokenizer_root / name, record):
                raise ValueError(f"Offline tokenizer does not match its pin: {name}")
        os.environ["HF_HUB_CACHE"] = str(cache)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        # The pinned XLA otherwise selects GPU algorithms by startup timings.
        # Keep independent SDK/native processes on the same repeatable kernels.
        flags = os.environ.get("XLA_FLAGS", "")
        os.environ["XLA_FLAGS"] = f"{flags} --xla_gpu_deterministic_ops=true".strip()
        import tensorflow as tf

        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.set_visible_devices([], "GPU")
        import jax
        from octo.model.octo_model import OctoModel

        self.jax = jax
        self.tf = tf
        self.model = OctoModel.load_pretrained(str(checkpoint), step=pin["step"])
        stats = self.model.dataset_statistics[pin["dataset_id"]]["action"]
        self.action_mean = np.asarray(stats["mean"])
        self.action_std = np.asarray(stats["std"])
        self.tasks = {}
        self.episodes: dict[str, dict] = {}

        def sample_one(model, observation, task, key):
            def add_batch(value):
                return value[None]

            actions = model.sample_actions(
                jax.tree_map(add_batch, observation), jax.tree_map(add_batch, task), rng=key,
            )
            return actions[0]

        self.sample_batch = jax.jit(jax.vmap(sample_one, in_axes=(None, 0, 0, 0)))

    def reset(self, episode_id: str, task_id: str) -> None:
        instruction = INSTRUCTIONS[task_id]
        if instruction not in self.tasks:
            self.tasks[instruction] = self.model.create_tasks(texts=[instruction])
        key = self.jax.random.PRNGKey(0)
        for _ in range(5):
            key, _ = self.jax.random.split(key)
        self.episodes[episode_id] = {
            "instruction": instruction, "images": deque(maxlen=2),
            "actions": deque(maxlen=4), "rng": key,
        }

    def act_batch(self, episode_ids: list[str], images: list[np.ndarray]) -> list[np.ndarray]:
        from transforms3d.euler import euler2mat, mat2euler

        if not images or len(images) != len(episode_ids):
            raise ValueError("Octo requires one RGB image per episode request")
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError("An Octo batch cannot contain the same episode twice")
        groups: dict[int, list[int]] = {}
        for index, (episode_id, image) in enumerate(zip(episode_ids, images, strict=True)):
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError("Octo requires an HWC uint8 RGB image")
            resized = self.tf.image.resize(image, (256, 256), method="lanczos3", antialias=True)
            resized = self.tf.cast(self.tf.clip_by_value(self.tf.round(resized), 0, 255), self.tf.uint8)
            state = self.episodes[episode_id]
            state["images"].append(resized.numpy())
            groups.setdefault(len(state["images"]), []).append(index)
        result: list[np.ndarray | None] = [None] * len(images)
        for history_size, indices in groups.items():
            states = [self.episodes[episode_ids[index]] for index in indices]
            keys = []
            for state in states:
                state["rng"], key = self.jax.random.split(state["rng"])
                keys.append(key)
            observations = {
                "image_primary": np.stack([np.stack(state["images"]) for state in states]),
                "pad_mask": np.ones((len(states), history_size), dtype=np.float64),
            }
            tasks = self.jax.tree_map(
                lambda *values: np.concatenate(values, axis=0),
                *[self.tasks[state["instruction"]] for state in states],
            )
            predictions = self.sample_batch(self.model, observations, tasks, np.stack(keys))
            if predictions.shape != (len(states), 4, 7):
                raise ValueError(f"Unexpected Octo action shape: {predictions.shape}")
            raw_actions = np.asarray(
                predictions * self.action_std[None, None] + self.action_mean[None, None],
            )
            for index, state, prediction in zip(indices, states, raw_actions, strict=True):
                state["actions"].append(prediction)
                count = len(state["actions"])
                aligned = np.stack([
                    prior[offset]
                    for offset, prior in zip(range(count - 1, -1, -1), state["actions"], strict=True)
                ])
                weights = np.ones(count, dtype=np.float64) / count
                action = np.sum(weights[:, None] * aligned, axis=0)
                # Octo Bridge uses fixed-axis xyz. The pinned MS3 controller
                # multiplies Rx @ Ry @ Rz and consumes intrinsic XYZ angles.
                rotation = euler2mat(*np.asarray(action[3:6], dtype=np.float64), axes="sxyz")
                native_rotation = mat2euler(rotation, axes="rxyz")
                converted = np.concatenate((
                    action[:3], native_rotation, [2.0 * (action[6] > 0.5) - 1.0],
                ))
                if not np.isfinite(converted).all():
                    raise ValueError("Octo produced non-finite controller actions")
                result[index] = np.ascontiguousarray(converted, dtype=np.float32)
        return result

    def end_episode(self, episode_id: str) -> None:
        self.episodes.pop(episode_id, None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if not args.serve:
        parser.error("Use --serve for the independent native evaluator")
    with redirect_stdout(sys.stderr):
        model = BridgeOcto(args.checkpoint)
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        operation = request["op"]
        if operation == "reset":
            for episode in request["episodes"]:
                model.reset(episode["episode_id"], episode["task_id"])
            response = {"reset": True}
        elif operation == "act":
            images = [
                np.frombuffer(base64.b64decode(data), dtype=np.uint8).reshape(shape)
                for data, shape in zip(request["images"], request["shapes"], strict=True)
            ]
            actions = model.act_batch(request["episode_ids"], images)
            response = {"actions": [action.tolist() for action in actions]}
        elif operation == "end":
            for episode_id in request["episode_ids"]:
                model.end_episode(episode_id)
            response = {"ended": True}
        elif operation == "close":
            break
        else:
            raise ValueError(f"Unknown native actor operation: {operation}")
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
