# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""RoboDojo: the episode loop and its observation stride, layout shards, and the examples that use them."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from roboscope_kit import Observation

REPO_ROOT = Path(__file__).parents[3]


def _load(relative_path: str) -> ModuleType:
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _TaskEnv:
    task_name = "align_blocks"
    env_seeds = (0,)

    def __init__(self, steps: int) -> None:
        self.steps = steps
        self.take_action_cnt = [0]
        self.rendered = 0

    def is_episode_end(self) -> bool:
        return self.take_action_cnt[0] >= self.steps

    def take_action(self, action: dict) -> None:
        self.take_action_cnt[0] += 1

    def get_obs(self) -> dict:
        self.rendered += 1
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        arm = {
            f"{side}_{name}": np.zeros(size, dtype=np.float32)
            for side in ("left", "right")
            for name, size in (("arm_joint_state", 6), ("ee_joint_state", 1), ("ee_pose", 7))
        }
        return {
            "vision": {
                name: {"color": image}
                for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")
            },
            "state": arm,
            "instruction": "",
        }


class _ModelClient:
    episode_id = ""

    def __init__(self, chunk: int) -> None:
        self.chunk = chunk
        self.acted: list[int] = []
        self.observed: list[int] = []

    def begin_episode(self, task: str, layout_id: int) -> str:
        self.episode_id = f"{task}-{layout_id}"
        return self.episode_id

    def end_episode(self) -> None:
        self.episode_id = ""

    def act(self, observation: Observation) -> np.ndarray:
        self.acted.append(observation.step)
        return np.zeros((self.chunk, 14), dtype=np.float32)

    def observe(self, observation: Observation) -> None:
        self.observed.append(observation.step)


def _run_episode(
    monkeypatch: pytest.MonkeyPatch, stride: str | None, chunk: int
) -> tuple[_TaskEnv, _ModelClient]:
    monkeypatch.setenv("ROBOSCOPE_ROBODOJO_ACTION_REPRESENTATION", "robodojo.x5-joint-v1")
    monkeypatch.setenv("ROBOSCOPE_ROBODOJO_STEP_UPDATES", "1")
    if stride is None:
        monkeypatch.delenv("ROBOSCOPE_ROBODOJO_OBSERVATION_STRIDE", raising=False)
    else:
        monkeypatch.setenv("ROBOSCOPE_ROBODOJO_OBSERVATION_STRIDE", stride)
    env, client = _TaskEnv(steps=2 * chunk), _ModelClient(chunk)
    _load("benchmarks/robodojo/roboscope_policy/deploy.py").eval_one_episode(env, client)
    return env, client


def test_default_stride_delivers_every_step(monkeypatch: pytest.MonkeyPatch) -> None:
    env, client = _run_episode(monkeypatch, None, chunk=4)

    assert client.acted == [0, 4]
    assert client.observed == [1, 2, 3, 5, 6, 7]
    assert env.rendered == 8


def test_stride_skips_delivery_but_renders_every_step(monkeypatch: pytest.MonkeyPatch) -> None:
    env, client = _run_episode(monkeypatch, "2", chunk=4)

    assert client.acted == [0, 4]
    assert client.observed == [2, 6]
    assert env.rendered == 8


def test_chunk_sized_stride_leaves_only_act_observations(monkeypatch: pytest.MonkeyPatch) -> None:
    env, client = _run_episode(monkeypatch, "4", chunk=4)

    assert client.acted == [0, 4]
    assert client.observed == []
    assert env.rendered == 8


def test_xpolicylab_stride_rescales_history_interval() -> None:
    interval = _load("examples/xpolicylab/xpolicylab_policy.py")._strided_history_interval

    assert interval({"exec_chunk_size": 10, "history_interval": 20}, 10) == 2


@pytest.mark.parametrize(
    "config",
    [
        {"exec_chunk_size": 10, "history_interval": 25},
        {"exec_chunk_size": 12, "history_interval": 20},
        {"history_interval": 20},
    ],
)
def test_xpolicylab_stride_refuses_a_model_that_reads_skipped_observations(config: dict) -> None:
    interval = _load("examples/xpolicylab/xpolicylab_policy.py")._strided_history_interval

    with pytest.raises(ValueError, match="observation-stride 10"):
        interval(config, 10)


def test_latency_policy_holds_the_current_joints() -> None:
    policy = _load("examples/latency-policy/latency_policy.py").LatencyPolicy(latency_s=0, chunk=3)
    joints = np.arange(14, dtype=np.float32)

    chunk = policy.act(
        Observation(episode_id="e", step=0, features={"state.joints": joints}, timestamp_s=None)
    )

    assert chunk.actions.shape == (3, 14)
    assert np.array_equal(chunk.actions[2], joints)


_FAKE_EVALUATOR = '''
import json, os, sys
from pathlib import Path

calls = Path(os.environ["FAKE_CALLS"])
calls.write_text(calls.read_text() + os.environ["ROBODOJO_LAYOUT_IDS"] + "\\n")
drops = json.loads(os.environ["FAKE_DROPS"])
call = len(calls.read_text().splitlines()) - 1
layouts = [x for x in json.loads(os.environ["ROBODOJO_LAYOUT_IDS"]) if x not in drops[call:call + 1]]
task = sys.argv[sys.argv.index("--task_name") + 1]
out = Path("eval_result/RoboDojo", task, "RoboScopeKit/arx_x5/1_roboscope", os.environ["ROBODOJO_RUN_ID"])
out.mkdir(parents=True)
details = {
    str(i): {"layout_id": x, "success": False, "score": 0.0, "roboscope_steps": 5, "roboscope_elapsed_s": 1.0}
    for i, x in enumerate(layouts)
}
(out / "_result.json").write_text(json.dumps({"eval_time": len(details), "details": details}))
'''


def _run_shard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, drops: list[int], envs: int = 1):
    import argparse
    import sys

    from roboscope_kit.runtime.benchmark import WorkItem
    from roboscope_kit.runtime.benchmarks.robodojo import worker

    root = tmp_path / "RoboDojo"
    (root / "src/eval_client").mkdir(parents=True)
    (root / "src/eval_client/main.py").write_text(_FAKE_EVALUATOR)
    (root / "XPolicyLab/policy/RoboScopeKit").mkdir(parents=True)
    (root / "XPolicyLab/policy/RoboScopeKit/deploy.yml").write_text("eval_batch: true\n")
    calls = tmp_path / "calls"
    calls.write_text("")
    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "YES")
    monkeypatch.setenv("FAKE_CALLS", str(calls))
    monkeypatch.setenv("FAKE_DROPS", json.dumps(drops))
    args = argparse.Namespace(
        worker_id=0, gpu_id="0", policy_socket=tmp_path / "policy.sock",
        action_representation="robodojo.x5-joint-v1", benchmark_python=Path(sys.executable),
        benchmark_root=root, output_dir=tmp_path / "out", async_video=False, unpaced_render=False,
        control_state_reuse=False, physics_threads=None, step_observation_updates=True,
        observation_stride=1, observation_queue_size=0, envs_per_worker=envs,
    )
    item = WorkItem(
        ordinal=0, item_id="arx_x5-align_blocks-layouts-0000-0003", group_id="g",
        payload={"task": "align_blocks", "seed": 1, "start": 0, "count": 3}, expected_results=3,
    )
    return lambda: worker.run_item(args, item), calls


def test_shard_evaluates_abandoned_layouts_again(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run, calls = _run_shard(monkeypatch, tmp_path, drops=[1])

    result = run()

    assert [record.metadata["layout_id"] for record in result.records] == [0, 1, 2]
    assert calls.read_text().splitlines() == ["[0, 1, 2]", "[1]"]
    assert len(result.metadata["result_paths"]) == 2


def test_shard_fails_when_a_layout_keeps_breaking(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run, calls = _run_shard(monkeypatch, tmp_path, drops=[2, 2, 2])

    with pytest.raises(RuntimeError, match=r"incomplete shard after 3 passes; missing layouts \[2\]"):
        run()
    assert calls.read_text().splitlines() == ["[0, 1, 2]", "[2]", "[2]"]


def test_multi_environment_retry_keeps_the_environment_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run, calls = _run_shard(monkeypatch, tmp_path, drops=[1], envs=3)

    result = run()

    assert [record.metadata["layout_id"] for record in result.records] == [0, 1, 2]
    assert calls.read_text().splitlines() == ["[0, 1, 2]", "[1, 0, 2]"]
    assert {record.metadata["evaluator_num_envs"] for record in result.records} == {3}


def test_multi_environment_retry_runs_a_stuck_layout_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run, calls = _run_shard(monkeypatch, tmp_path, drops=[1, 1], envs=3)

    result = run()

    assert calls.read_text().splitlines() == ["[0, 1, 2]", "[1, 0, 2]", "[1]"]
    assert [record.metadata["evaluator_num_envs"] for record in result.records] == [3, 1, 3]
