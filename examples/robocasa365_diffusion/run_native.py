# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Run the pinned upstream ``eval_robocasa.py`` for one task with rollout traces.

This calls the official ``eval_task`` at its own task horizon with the native
five-environment runner. It changes no evaluator setting; it only records the
``rollout_trace`` digests that ``compare_rollouts.py`` pairs with a Kit run.

Run it with the inference environment's interpreter (``install-policy.sh`` plus
``native-requirements.txt``). ``--simulator`` is the ``build-runtime.sh`` install
directory: its pinned RoboCasa/robosuite sources and simulator packages are put
ahead of the inference environment, as the upstream evaluator imports both.
``--gym-source`` is an unpacked ``gym==0.21.0`` source tree, the version pinned
by the upstream ``conda_environment.yaml`` (it is pure Python; the sdist does not
build with current setuptools).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

EVALUATOR_SHA256 = "d1b6bb4f939358465d2492c08fe34cd2d69b974ecfc74e0bbe0b566dc6c02492"
RUNNER_SHA256 = "d3cb68ba936ae8716c694fcef8869c17de39b87840f126fc690bcaf48367b31a"
RUNNER_MODULE = "diffusion_policy.env_runner.robomimic_image_runner"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--gym-source", type=Path, required=True)
    parser.add_argument("--diffusion-policy", type=Path, required=True,
                        help="pinned Diffusion Policy source root (contains eval_robocasa.py)")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def prepare_paths(args: argparse.Namespace) -> None:
    """Use the simulator's packages, except its CPU-only torch, ahead of the inference env."""
    packages = str(next((args.simulator / "env/lib").glob("python3.*/site-packages")))
    simulator = [
        str(args.simulator / "src/robocasa"), str(args.simulator / "src/robosuite"),
        str(args.gym_source), packages,
    ]
    # Spawned environments inherit the parent's sys.path; rebuild it in the same order.
    sys.path[:] = [entry for entry in sys.path if entry not in simulator]
    sys.path.insert(0, packages)
    import cv2  # noqa: F401
    import numpy  # noqa: F401

    sys.path.remove(packages)
    import torch  # noqa: F401
    import torchvision  # noqa: F401

    sys.path[:0] = simulator


def load_runner(source_root: Path) -> None:
    """Load the pinned env runner without its unused LeRobot dataset import."""
    path = source_root / "diffusion_policy/env_runner/robomimic_image_runner.py"
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != RUNNER_SHA256:
        raise RuntimeError(f"upstream env runner checksum mismatch: {path}")
    line = b"import robocasa.utils.lerobot_utils as LU\n"
    tree = ast.parse(source)
    if source.count(line) != 1 or any(
        isinstance(node, ast.Name) and node.id == "LU" for node in ast.walk(tree)
    ):
        raise RuntimeError("upstream env runner no longer matches the compatibility edit")
    spec = importlib.util.spec_from_file_location(RUNNER_MODULE, path)
    module = importlib.util.module_from_spec(spec)
    parent_name, _, child_name = RUNNER_MODULE.rpartition(".")
    sys.modules[RUNNER_MODULE] = module
    setattr(importlib.import_module(parent_name), child_name, module)
    exec(compile(source.replace(line, b""), str(path), "exec"), module.__dict__)  # noqa: S102
    original = module.RobomimicImageRunner.__init__

    def initialize(self, *args, **kwargs):
        kwargs["n_test_vis"] = 0  # The Kit comparison runs without videos.
        return original(self, *args, **kwargs)

    module.RobomimicImageRunner.__init__ = initialize


def bootstrap() -> argparse.Namespace:
    """Prepare imports and traces; the runner's spawned environments repeat this."""
    args = parse_args()
    os.environ.update(
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false", WANDB_MODE="disabled",
        MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", PYTHONDONTWRITEBYTECODE="1",
        ROBOCASA365_TRACE_DIR=str(args.output / "trace"), ROBOCASA365_TRACE_ROLE="native",
    )
    prepare_paths(args)
    load_runner(args.diffusion_policy)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import rollout_trace

    rollout_trace.install()
    return args


ARGS = bootstrap()


def main(args: argparse.Namespace) -> None:
    path = args.diffusion_policy / "eval_robocasa.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != EVALUATOR_SHA256:
        raise RuntimeError(f"upstream evaluator checksum mismatch: {path}")
    spec = importlib.util.spec_from_file_location("official_eval_robocasa", path)
    evaluator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = evaluator
    spec.loader.exec_module(evaluator)
    args.output.mkdir(parents=True, exist_ok=True)
    evaluator.eval_task(str(args.checkpoint), str(args.output / "result"), "cuda:0", args.task,
                        args.episodes, 5, "pretrain", overwrite=False)
    import gym

    (args.output / "native-inputs.json").write_text(json.dumps({
        "task": args.task, "episodes": args.episodes,
        "horizon": evaluator.get_task_horizon(args.task),
        "evaluator_sha256": EVALUATOR_SHA256, "env_runner_sha256": RUNNER_SHA256,
        "gym_version": gym.__version__, "simulator": str(args.simulator),
        "checkpoint": str(args.checkpoint),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main(ARGS)
