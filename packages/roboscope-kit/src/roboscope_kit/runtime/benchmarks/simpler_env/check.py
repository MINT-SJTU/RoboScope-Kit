# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Check a SimplerEnv runtime: parse, build, reset, render, and step one episode per check.

Runs in the SimplerEnv runtime environment, from the SimplerEnv checkout (``build-runtime.sh``
and ``--self-test`` call it)::

    python -m roboscope_kit.runtime.benchmarks.simpler_env.check

For the quick-test variant of every suite and one ray-traced drawer variant it expands the
upstream arguments to the frozen episode count, builds the environment through the worker's
own path, resets it, checks the policy image and instruction, and takes one zero action.
Prints one JSON line; exits non-zero on the first failure.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from roboscope_kit.benchmark_specs import SIMPLER_ENV_IMAGE_SHAPES

from .tasks import QUICK_TEST_VARIANTS, SUITE_VARIANTS, VISUAL_MATCHING
from .worker import build_env, expand_episodes, reset_options, upstream_args

RAY_TRACED_CHECK = "open_drawer/top/urdf_none/a0"


def check(suite: str, variant_id: str) -> dict:
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    variant = SUITE_VARIANTS[suite][variant_id]
    args = upstream_args(variant)
    episodes = expand_episodes(args)
    if len(episodes) != variant.episodes:
        raise SystemExit(f"{variant_id} expands to {len(episodes)} episodes, expected {variant.episodes}")
    if args.max_episode_steps != variant.max_episode_steps:
        raise SystemExit(f"{variant_id} parses max_episode_steps {args.max_episode_steps}")
    started = time.perf_counter()
    # Like upstream's evaluator, the environment is dropped, not closed.
    env = build_env(args)
    obs, _ = env.reset(options=reset_options(episodes[0]))
    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=args.obs_camera_name)
    expected = SIMPLER_ENV_IMAGE_SHAPES[suite]
    if image.dtype != np.uint8 or tuple(image.shape) != expected:
        raise SystemExit(
            f"{variant_id} policy image is {image.dtype} {image.shape}, expected uint8 {expected}"
        )
    if float(image.std()) == 0.0:
        raise SystemExit(f"{variant_id} rendered a constant image")
    instruction = env.get_language_instruction()
    if not instruction:
        raise SystemExit(f"{variant_id} has no language instruction")
    _obs, _reward, _done, truncated, info = env.step(np.zeros(7))
    if truncated or info.get("elapsed_steps") != 1:
        raise SystemExit(f"{variant_id} did not take one step: {info}")
    return {
        "suite": suite,
        "variant": variant_id,
        "episodes": len(episodes),
        "instruction": instruction,
        "image_shape": list(image.shape),
        "seconds": round(time.perf_counter() - started, 2),
    }


def main() -> int:
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["DISPLAY"] = ""
    checks = [(suite, variant) for suite, variant in QUICK_TEST_VARIANTS.items()]
    checks.append((VISUAL_MATCHING, RAY_TRACED_CHECK))
    report = {"variants": [check(suite, variant) for suite, variant in checks]}
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
