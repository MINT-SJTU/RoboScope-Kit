# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Instrumentation of PerAct's ``eval.py``, installed by ``run_native.py``.

YARR forces the ``spawn`` start method, so PerAct evaluates in a fresh child process that
re-imports everything. ``run_native.py`` therefore makes ``eval.py`` construct
:class:`HookedIndependentEnvRunner`, a subclass of YARR's runner with no changes of its own:
unpickling it in the child imports this module, which installs the hooks there.

The hooks wrap PerAct's ``CustomRLBenchEnv``/``CustomMultiTaskRLBenchEnv`` ``reset_to_demo``,
``step`` and the scene step callback, call the originals unchanged, and append one JSON line per
episode to ``$ROBOSCOPE_NATIVE_EPISODES``: the task, test episode, protocol, Python hash seed,
variation, instruction, every
action passed to RLBench, the planning error if any, the outcome, executed actions, and
simulation steps. They also hand PerAct's ``load_clip`` the pinned CLIP RN50 file in
``$ROBOSCOPE_PERACT_CLIP`` instead of downloading it, as the Kit Policy does.

For one-episode evaluations only, an accumulator subclass emits YARR's usual statistics for the
single completed return. The pinned upstream accumulator otherwise emits no CSV for that case.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from helpers import custom_rlbench_env
from helpers.clip.core import clip as peract_clip
from rlbench.task_environment import TaskEnvironment
from yarr.runners import independent_env_runner as runner_module
from yarr.runners.independent_env_runner import IndependentEnvRunner
from yarr.utils.process_str import change_case
from yarr.utils.stat_accumulator import SimpleAccumulator

from roboscope_kit.runtime.benchmarks.rlbench.tasks import (
    PROTOCOL_VERSION,
    PYTHON_HASH_SEED,
    episode_state_sha256,
    require_python_hash_seed,
)
from roboscope_kit.runtime.benchmarks.rlbench.test_set import load_test_episode
from roboscope_kit.runtime.benchmarks.rlbench.worker import (
    LoadedTask,
    max_displacement,
    recorded_positions,
    scene_positions,
)

EPISODE_LENGTH = 25


class SingleEpisodeAccumulator(SimpleAccumulator):
    """Keep YARR's normal statistics when exactly one evaluation episode completed.

    The pinned YARR version suppresses summaries unless at least two returns are available.
    PerAct's native runner otherwise writes no CSV for a one-episode request. This affects only
    reporting: the same accumulated rewards and lengths are summarized with YARR's own code.
    """

    def pop(self):
        train = self._train_acc.pop()
        evaluation = self._eval_acc.pop()
        if not evaluation and len(self._eval_acc._episode_returns) == 1:
            evaluation = self._eval_acc.peak()
            self._eval_acc._reset_data()
        return train + evaluation


class EpisodeLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.current: dict | None = None
        self.simulation_steps = 0
        # The task loaded last, with its object list captured at its first reset (see the Kit
        # worker's LoadedTask), and the object positions right after TaskEnvironment.reset_to_demo.
        self.loaded: LoadedTask | None = None
        self.sampled = None
        self.demo_state_sha256: str | None = None

    def start(self, recorded_state, **episode) -> None:
        objects = self.loaded.objects
        recorded = recorded_positions(objects, recorded_state)
        error, worst = max_displacement(objects, self.sampled, recorded)
        episode["initial_scene_position_error_m"] = error
        episode["initial_scene_worst_object"] = worst
        episode["descriptions_moved_objects_m"] = max_displacement(
            objects, scene_positions(objects), self.sampled
        )[0]
        self.current = {**episode, "actions": [], "started": time.time()}
        self.simulation_steps = 0

    def step(self, action, reward: float, terminal: bool, error: str | None, executed: int) -> None:
        current = self.current
        current["actions"].append([float(value) for value in action])
        if not (terminal or executed == EPISODE_LENGTH):
            return
        success = reward > 0.99
        if error is not None:
            termination = error
        elif success:
            termination = "success"
        elif terminal:
            termination = "fail_condition"
        else:
            termination = "timeout"
        record = {
            "task": current["task"],
            "episode": current["episode"],
            "episode_state_sha256": current["episode_state_sha256"],
            "protocol": PROTOCOL_VERSION,
            "python_hash_seed": PYTHON_HASH_SEED,
            "variation": current["variation"],
            "instruction": current["instruction"],
            "initial_scene_position_error_m": current["initial_scene_position_error_m"],
            "initial_scene_worst_object": current["initial_scene_worst_object"],
            "descriptions_moved_objects_m": current["descriptions_moved_objects_m"],
            "success": success,
            "steps": executed,
            "termination": termination,
            "error": error,
            "simulation_steps": self.simulation_steps,
            "actions": current["actions"],
            "elapsed_s": time.time() - current["started"],
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        self.current = None


def _install(log: EpisodeLog) -> None:
    original_task_reset_to_demo = TaskEnvironment.reset_to_demo

    def task_reset_to_demo(self, demo):
        log.demo_state_sha256 = episode_state_sha256(self._variation_number, demo.random_seed)
        result = original_task_reset_to_demo(self, demo)
        if log.loaded is None or log.loaded.env is not self:
            # PerAct's task swap loads a new TaskEnvironment; this is its first reset.
            log.loaded = LoadedTask(self)
        log.sampled = scene_positions(log.loaded.capture_objects())
        return result

    TaskEnvironment.reset_to_demo = task_reset_to_demo
    for cls in (custom_rlbench_env.CustomRLBenchEnv, custom_rlbench_env.CustomMultiTaskRLBenchEnv):
        original_reset = cls.reset_to_demo
        original_step = cls.step
        original_callback = cls._my_callback

        def reset_to_demo(self, i, *args, _original=original_reset, **kwargs):
            observation = _original(self, i, *args, **kwargs)
            task = change_case(self._task._task.__class__.__name__)
            # get_stored_demos drops the task state PerAct's observation config does not request,
            # so read the recorded first frame the way the Kit worker does.
            test_episode = load_test_episode(Path(self._task._dataset_root), task, int(i))
            if log.demo_state_sha256 != test_episode.state_sha256:
                raise RuntimeError(f"native demo does not match test episode {task}/{i}")
            log.start(
                test_episode.recorded_task_state,
                task=task,
                episode=int(i),
                episode_state_sha256=log.demo_state_sha256,
                variation=int(self._task._variation_number),
                instruction=self._lang_goal,
            )
            return observation

        def step(self, act_result, _original=original_step):
            errors = dict(self._error_type_counts)
            transition = _original(self, act_result)
            error = next(
                (name for name, count in self._error_type_counts.items() if count != errors[name]), None
            )
            log.step(act_result.action, float(transition.reward), bool(transition.terminal), error, self._i)
            return transition

        def my_callback(self, _original=original_callback):
            log.simulation_steps += 1
            return _original(self)

        cls.reset_to_demo = reset_to_demo
        cls.step = step
        cls._my_callback = my_callback


class HookedIndependentEnvRunner(IndependentEnvRunner):
    """YARR's runner, unchanged; importing its module in the child installs the hooks."""


_clip = os.environ.get("ROBOSCOPE_PERACT_CLIP")
if _clip:
    peract_clip._download = lambda url, root=None: _clip
_episodes = os.environ.get("ROBOSCOPE_NATIVE_EPISODES")
if _episodes:
    require_python_hash_seed()
    _install(EpisodeLog(Path(_episodes)))
if os.environ.get("ROBOSCOPE_NATIVE_SINGLE_EPISODE") == "1":
    runner_module.SimpleAccumulator = SingleEpisodeAccumulator
