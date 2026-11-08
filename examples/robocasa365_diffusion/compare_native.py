# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Compare a captured real first batch with the pinned upstream loader and wrappers.

This compares inputs and first diffusion actions. A full rollout/native score
comparison is still required before claiming evaluation equivalence.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from policy import FEATURE_MAP, RoboCasa365DiffusionPolicy

from roboscope_kit import EpisodeContext, Observation, SessionContext
from roboscope_kit.benchmark_specs import get_benchmark_spec


def native_process_obs() -> Any:
    """Load the exact upstream method without importing unused simulator modules."""
    import diffusion_policy
    from robomimic.utils.obs_utils import process_frame

    path = (
        Path(next(iter(diffusion_policy.__path__)))
        / "env/robomimic/robomimic_image_wrapper.py"
    )
    source = path.read_text(encoding="utf-8")
    inputs = json.loads(
        Path(__file__).resolve().with_name("model-inputs.json").read_text()
    )
    expected = inputs["diffusion_policy"]["observation_wrapper_sha256"]
    if hashlib.sha256(source.encode("utf-8")).hexdigest() != expected:
        raise ValueError(f"upstream observation wrapper checksum mismatch: {path}")
    cls = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "RobomimicImageWrapper"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "process_obs"
    )
    namespace = {"process_frame": process_frame}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["process_obs"]


def main() -> None:
    import dill
    import hydra
    import torch
    from robomimic.utils.lang_utils import LangEncoder

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.batch, allow_pickle=False) as archive:
        captured = {key: archive[key] for key in archive.files}
    instructions = np.asarray(captured["language.instruction"])
    count = len(instructions)
    task = str(captured["task_name"].item())
    policy = RoboCasa365DiffusionPolicy()
    policy.load(
        SessionContext(
            session_id="native-first-batch-comparison",
            checkpoint_path=str(args.checkpoint),
            benchmark=get_benchmark_spec("robocasa365", "target50"),
        )
    )
    observations = []
    for index in range(count):
        episode_id = f"native-compare-{index}"
        policy.reset(EpisodeContext(episode_id=episode_id, task_id=task))
        observations.append(
            Observation(
                episode_id=episode_id,
                step=0,
                features={
                    **{
                        feature: captured[feature][index]
                        for feature in FEATURE_MAP.values()
                    },
                    "language.instruction": str(instructions[index]),
                },
            )
        )
    adapted = [policy.prepare_observation(observation) for observation in observations]
    checkpoint = (
        args.checkpoint / "model.ckpt" if args.checkpoint.is_dir() else args.checkpoint
    )
    with checkpoint.open("rb") as stream:
        payload = torch.load(
            stream, map_location="cpu", pickle_module=dill, weights_only=False
        )
    cfg = payload["cfg"]
    # Run the actual native workspace loader, including its EMA selection and initialization.
    workspace = hydra.utils.get_class(cfg._target_)(
        cfg, output_dir=str(args.output.parent)
    )
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    model = workspace.ema_model if cfg.training.use_ema else workspace.model
    model.to(policy.device)
    model.eval()
    model.reset()
    native_rng = policy._capture_rng()
    loader_rng_equal = {
        "cpu": bool(torch.equal(native_rng.cpu, policy.initial_rng.cpu)),
        "cuda": (
            bool(torch.equal(native_rng.cuda, policy.initial_rng.cuda))
            if native_rng.cuda is not None
            else policy.initial_rng.cuda is None
        ),
    }
    # Exercise the native wrapper methods while sharing pinned local CLIP weights.
    process_obs = native_process_obs()
    wrapper = SimpleNamespace(shape_meta=cfg.task.shape_meta)
    encoder = LangEncoder.__new__(LangEncoder)
    encoder.device = "cpu"
    encoder.lang_emb_model = policy.encoder
    encoder.tz = policy.tokenizer
    native = []
    for index in range(count):
        raw = {
            cfg.task.shape_meta.obs[key].lerobot_keys[0]: captured[feature][index]
            for key, feature in FEATURE_MAP.items()
        }
        prepared = process_obs(wrapper, raw)
        prepared["lang_emb"] = np.repeat(
            encoder.get_lang_emb(str(instructions[index])).numpy()[None], 2, axis=0
        )
        native.append(prepared)
    feature_equal = {
        key: all(
            np.array_equal(a[key], n[key]) for a, n in zip(adapted, native, strict=True)
        )
        for key in adapted[0]
    }
    tensors = {
        key: torch.from_numpy(np.stack([row[key] for row in native])).to(policy.device)
        for key in native[0]
    }
    # Capture each loader's RNG independently, then restore the native state after
    # input preparation so CLIP cannot mask loader-state differences.
    with torch.random.fork_rng(devices=policy._rng_devices()), torch.no_grad():
        torch.set_rng_state(native_rng.cpu)
        if native_rng.cuda is not None:
            torch.cuda.set_rng_state(native_rng.cuda, policy.device)
        reference_actions = (
            model.predict_action(tensors)["action"].detach().cpu().numpy()
        )
    adapted_actions = np.stack(
        [chunk.actions for chunk in policy.act_batch(tuple(observations))]
    )
    captured_actions = np.asarray(captured["actions"])
    result = {
        "batch_size": count,
        "task": task,
        "native_loader_rng_exact_equal": loader_rng_equal,
        "feature_exact_equal": feature_equal,
        "adapter_native_actions_exact_equal": bool(
            np.array_equal(adapted_actions, reference_actions)
        ),
        "captured_native_actions_exact_equal": bool(
            np.array_equal(captured_actions, reference_actions)
        ),
        "adapter_native_max_abs_difference": float(
            np.max(np.abs(adapted_actions - reference_actions))
        ),
        "captured_native_max_abs_difference": float(
            np.max(np.abs(captured_actions - reference_actions))
        ),
        "scope": "real first-batch preprocessing and action parity; not full rollout equivalence",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if (
        not all(loader_rng_equal.values())
        or not all(feature_equal.values())
        or not result["adapter_native_actions_exact_equal"]
    ):
        raise SystemExit(1)
    if not result["captured_native_actions_exact_equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
