# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Run PerAct's own ``eval.py`` on the RLBench runtime and log every episode it executes.

    <runtime>/env/bin/python examples/peract/run_native.py --runtime /sim/rlbench \\
        --checkpoint-source /models/peract --output /runs/peract-native [--task close_jar ...]

This is the native entry point the Kit is compared with: the pinned PerAct ``eval.py`` with the
README's test command (all requested tasks in one process, ``eval_envs=1``, episodes 0..24 of
the released test split, ``eval_type=last``, the 600k weights, headless). ``eval.py`` runs as
``__main__``, unmodified, with additions that do not change episode execution:

- CLIP RN50 is read from the pinned local file in ``--checkpoint-source``, as the Kit Policy does,
  instead of being downloaded to ``~/.cache/clip``;
- ``native_hooks.py`` wraps PerAct's ``CustomRLBenchEnv``/``CustomMultiTaskRLBenchEnv``
  ``reset_to_demo`` and ``step`` (and the scene step callback) read-only and records, per episode,
  the test episode, variation, instruction, initial-state distance from the recorded episode,
  every action passed to RLBench, the planning error if any, the outcome, executed actions, and
  simulation steps. YARR evaluates in a ``spawn``-ed child process, so ``eval.py`` is given
  ``native_hooks.HookedIndependentEnvRunner``, an unchanged subclass of YARR's runner whose module
  installs the hooks when the child unpickles it;
- For a one-episode request, the hooks let YARR emit its normal statistics for that episode;
  the pinned accumulator otherwise emits no CSV until it has at least two returns.
- CoppeliaSim runs on a private Xvfb display, like the Kit workers.
- The evaluator restarts with ``PYTHONHASHSEED=0``, matching the Kit simulator workers. The
  released NumPy state does not determine Python set iteration in ``push_buttons``.

Writes ``<output>/native-episodes.jsonl`` (one line per episode), ``<output>/run.json`` (command,
timings), PerAct's own ``eval_data.csv`` under ``<output>/logdir``, and its log.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import runpy
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from assets import CHECKPOINT_SEED_DIR, CHECKPOINT_WEIGHT, CLIP_RN50, verify  # noqa: E402

from roboscope_kit.runtime.benchmarks.rlbench.tasks import (  # noqa: E402
    EVALUATION_ORDER,
    PROTOCOL_VERSION,
    PYTHON_HASH_SEED,
    require_python_hash_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--checkpoint-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task", action="append", choices=EVALUATION_ORDER, help="default: all 18, in README order"
    )
    parser.add_argument("--episodes", type=int, default=25)
    return parser.parse_args()


def reexec_with_runtime_environment(runtime: Path) -> None:
    """The dynamic loader reads LD_LIBRARY_PATH at start-up, so set it and restart once."""

    coppeliasim = str(runtime / "coppeliasim")
    if (
        os.environ.get("ROBOSCOPE_NATIVE_REEXEC") == "1"
        and os.environ.get("PYTHONHASHSEED") == str(PYTHON_HASH_SEED)
        and not sys.flags.hash_randomization
    ):
        return
    environment = {
        **os.environ,
        "ROBOSCOPE_NATIVE_REEXEC": "1",
        "COPPELIASIM_ROOT": coppeliasim,
        "LD_LIBRARY_PATH": ":".join((coppeliasim, os.environ.get("LD_LIBRARY_PATH", ""))),
        "QT_QPA_PLATFORM_PLUGIN_PATH": coppeliasim,
        "PYTHONHASHSEED": str(PYTHON_HASH_SEED),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)


def prepare_logdir(checkpoint_source: Path, output: Path) -> Path:
    """eval.py writes eval_data.csv next to the weights; give it a copy of the checkpoint layout."""

    logdir = output / "logdir"
    seed_dir = logdir / "multi" / "PERACT_BC" / "seed0"
    weights = seed_dir / "weights" / str(CHECKPOINT_WEIGHT)
    weights.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkpoint_source / CHECKPOINT_SEED_DIR / "config.yaml", seed_dir / "config.yaml")
    source = (
        checkpoint_source
        / CHECKPOINT_SEED_DIR
        / "weights"
        / str(CHECKPOINT_WEIGHT)
        / "QAttentionAgent_layer0.pt"
    )
    link = weights / source.name
    if not link.exists():
        link.symlink_to(source.resolve())
    return logdir


def validate_native_output(episodes_path: Path, logdir: Path, tasks: list[str], episodes: int) -> None:
    """Reject a failed/incomplete evaluator child before marking a native run successful."""

    expected = {(task, episode) for task in tasks for episode in range(episodes)}
    records: dict[tuple[str, int], bool] = {}
    try:
        with episodes_path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                task, episode, success = record["task"], record["episode"], record["success"]
                digest = record["episode_state_sha256"]
                if (
                    not isinstance(task, str)
                    or type(episode) is not int
                    or type(success) is not bool
                    or not isinstance(digest, str)
                    or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)
                    or record.get("protocol") != PROTOCOL_VERSION
                    or type(record.get("python_hash_seed")) is not int
                    or record["python_hash_seed"] != PYTHON_HASH_SEED
                ):
                    raise ValueError("native episode has invalid identity, success, or execution contract")
                key = (task, episode)
                if key not in expected or key in records:
                    raise ValueError(f"duplicate or unexpected native episode {key}")
                records[key] = success
        missing = expected - records.keys()
        if missing:
            raise ValueError(f"native evaluation is missing {len(missing)} episodes: {sorted(missing)[:5]}")
        csv_path = logdir / "multi" / "PERACT_BC" / "seed0" / "eval_data.csv"
        with csv_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        matches = [row for row in rows if row.get("step") == str(CHECKPOINT_WEIGHT)]
        if len(matches) != 1:
            raise ValueError(f"expected one native metrics row for checkpoint {CHECKPOINT_WEIGHT}")
        for task in tasks:
            # YARR omits the task suffix when evaluating exactly one task.
            metric = f"eval_envs/return/{task}"
            if len(tasks) == 1 and metric not in matches[0]:
                metric = "eval_envs/return"
            rate = float(matches[0][metric])
            expected_rate = 100.0 * sum(records[(task, episode)] for episode in range(episodes)) / episodes
            if not math.isfinite(rate) or not math.isclose(rate, expected_rate, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(f"native CSV success rate disagrees with episode records for {task}")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"native evaluation did not complete consistently: {exc}") from exc


def main() -> int:
    args = parse_args()
    if not 1 <= args.episodes <= 25:
        raise SystemExit("--episodes must be between 1 and 25")
    if args.task and len(set(args.task)) != len(args.task):
        raise SystemExit("--task must not contain duplicates")
    runtime = args.runtime.resolve()
    reexec_with_runtime_environment(runtime)
    require_python_hash_seed()
    started = time.time()
    checkpoint_source = args.checkpoint_source.resolve()
    verify(checkpoint_source)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    episodes_path = output / "native-episodes.jsonl"
    if episodes_path.exists():
        raise SystemExit(f"{episodes_path} exists; choose a new --output")
    tasks = args.task or list(EVALUATION_ORDER)
    logdir = prepare_logdir(checkpoint_source, output)

    from roboscope_kit.runtime.benchmarks.rlbench.worker import start_xvfb

    start_xvfb(runtime / "bin" / "Xvfb")
    peract_root = runtime / "src" / "peract"
    sys.path.insert(0, str(peract_root))
    os.chdir(peract_root)
    os.environ["ROBOSCOPE_NATIVE_EPISODES"] = str(episodes_path)
    os.environ["ROBOSCOPE_NATIVE_SINGLE_EPISODE"] = "1" if args.episodes == 1 else "0"
    os.environ["ROBOSCOPE_PERACT_CLIP"] = str(checkpoint_source / CLIP_RN50.relative_path)
    import native_hooks
    import yarr.runners.independent_env_runner as runner_module

    runner_module.IndependentEnvRunner = native_hooks.HookedIndependentEnvRunner

    overrides = [
        f"rlbench.tasks=[{','.join(tasks)}]",
        "rlbench.task_name=multi",
        f"rlbench.demo_path={runtime / 'peract-test' / 'test'}",
        "framework.gpu=0",
        f"framework.logdir={logdir}",
        "framework.start_seed=0",
        "framework.eval_envs=1",
        "framework.eval_from_eps_number=0",
        f"framework.eval_episodes={args.episodes}",
        "framework.csv_logging=True",
        "framework.tensorboard_logging=True",
        "framework.eval_type=last",
        "rlbench.headless=True",
        f"hydra.run.dir={output / 'hydra'}",
    ]
    (output / "run.json").write_text(
        json.dumps(
            {
                "eval.py": str(peract_root / "eval.py"),
                "overrides": overrides,
                "protocol": PROTOCOL_VERSION,
                "python_hash_seed": PYTHON_HASH_SEED,
                "started": started,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    sys.argv = [str(peract_root / "eval.py"), *overrides]
    # As `python eval.py ...`: hydra resolves conf/ next to the __main__ script.
    runpy.run_path(str(peract_root / "eval.py"), run_name="__main__")
    validate_native_output(episodes_path, logdir, tasks, args.episodes)
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    run["finished"] = time.time()
    run["elapsed_s"] = run["finished"] - started
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
