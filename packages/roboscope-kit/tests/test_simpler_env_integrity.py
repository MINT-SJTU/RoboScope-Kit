# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""SimplerEnv result integrity: formal status, the native runner's reuse, and the native comparison.

None of these needs the simulator. The comparison's upstream argument parsing is replaced by a
small parser of the same flags, and the native runner's runtime Python by a stub that writes
episode records.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from roboscope_kit.runtime.benchmark.schema import EvaluationRecord, WorkResult
from roboscope_kit.runtime.benchmarks.simpler_env.adapter import (
    SimplerEnvBenchmarkAdapter,
    SimplerEnvRuntimeConfig,
)
from roboscope_kit.runtime.benchmarks.simpler_env.tasks import (
    BRIDGE,
    SUITE_EPISODES,
    SUITE_TASKS,
    VISUAL_MATCHING,
    episode_id,
)

EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "rt1"
sys.path.insert(0, str(EXAMPLES))

import compare_native  # noqa: E402
import run_native  # noqa: E402

# ---- adapter: formal status ------------------------------------------------------------------


def _adapter(suite: str) -> SimplerEnvBenchmarkAdapter:
    selection = tuple(
        (variant.variant_id, tuple(range(variant.episodes)))
        for task in SUITE_TASKS[suite]
        for variant in task.variants
    )
    return SimplerEnvBenchmarkAdapter(
        SimplerEnvRuntimeConfig(
            python=Path("/py"), simpler_env_root=Path("/se"), suite=suite, selection=selection
        )
    )


def _results(adapter: SimplerEnvBenchmarkAdapter, limit: int | None = None) -> list[WorkResult]:
    results = []
    for item in adapter.work_items()[:limit]:
        variant = adapter.variants[item.payload["variant"]]
        index = item.payload["indices"][0]
        record = EvaluationRecord(
            record_id=episode_id(variant, index),
            success=index % 2 == 0,
            steps=1,
            elapsed_s=0.1,
            metadata={
                "variant": variant.variant_id,
                "index": index,
                "task": variant.task,
                "terminated_by_policy": True,
                "truncated": False,
                "episode_stats": {"is_src_obj_grasped": index % 3 == 0},
            },
        )
        results.append(WorkResult(item.item_id, item.group_id, 0, (record,)))
    return results


def test_full_result_is_formal() -> None:
    adapter = _adapter(BRIDGE)
    official = adapter.aggregate(_results(adapter))
    assert official["formal"] is True and official["partial"] is False
    assert official["episodes"] == SUITE_EPISODES[BRIDGE]


def test_subset_of_a_formal_selection_is_not_formal() -> None:
    adapter = _adapter(VISUAL_MATCHING)
    official = adapter.aggregate(_results(adapter, limit=1))
    assert adapter.config.formal
    assert official["formal"] is False and official["partial"] is True
    assert (official["episodes"], official["selected_episodes"]) == (1, 864)


def test_shard_aggregate_is_never_formal() -> None:
    adapter = _adapter(BRIDGE)
    official = adapter.aggregate_partial(_results(adapter))
    assert official["formal"] is False and official["partial"] is True


def test_benchmark_metadata_does_not_claim_formal() -> None:
    metadata = _adapter(VISUAL_MATCHING).metadata()
    assert "formal" not in metadata
    assert metadata["formal_selection"] is True


# ---- run_native: output identity and reuse ---------------------------------------------------

ARGV = ["--policy-model", "rt1", "--ckpt-path", "checkpoints/x", "--env-name", "EnvA", "--robot", "r"]
STUB = """#!{python}
import json, os, sys
record = sys.argv[sys.argv.index("--record") + 1]
argv = sys.argv[sys.argv.index("simpler_env/main_inference.py") + 1:]
flag = lambda name, default=None: argv[argv.index(name) + 1] if name in argv else default
count = int(os.environ.get("STUB_EPISODES", "2"))
with open(record, "a") as handle:
    for index in range(count):
        handle.write(json.dumps({{
            "env_name": flag("--env-name"), "policy_model": flag("--policy-model", "rt1"),
            "policy_setup": flag("--policy-setup", "google_robot"),
            "obs_camera_name": flag("--obs-camera-name"), "ckpt_path": flag("--ckpt-path"),
            "index": index,
        }}) + "\\n")
with open(os.environ["STUB_CALLS"], "a") as handle:
    handle.write(record + "\\n")
"""


@pytest.fixture
def native(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    runtime = tmp_path / "runtime"
    python = runtime / "env" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(STUB.format(python=sys.executable), encoding="utf-8")
    python.chmod(0o755)
    output = tmp_path / "out"
    for directory in (output / "episodes", output / "logs"):
        directory.mkdir(parents=True)
    calls = tmp_path / "calls.txt"
    calls.touch()
    monkeypatch.setenv("STUB_CALLS", str(calls))
    args = argparse.Namespace(runtime=runtime, tf_memory_mb=1024, write_videos=False, gpu="0", suite=BRIDGE)
    identity = run_native.bind_output(output, {"suite": BRIDGE, "plan_sha256": "p"})

    def run(argv: list[str] = ARGV, index: int = 0, run_args: argparse.Namespace = args) -> dict:
        entry = {"index": index, "argv": list(argv), "script": "s.sh"}
        return run_native.run_invocation(entry, run_args, tmp_path, output, identity)

    return SimpleNamespace(output=output, run=run, args=args, calls=lambda: calls.read_text().splitlines())


def test_output_directory_rejects_another_identity(native: SimpleNamespace) -> None:
    run_native.bind_output(native.output, {"suite": BRIDGE, "plan_sha256": "p"})
    with pytest.raises(SystemExit, match="different identity"):
        run_native.bind_output(native.output, {"suite": VISUAL_MATCHING, "plan_sha256": "p"})


def test_output_directory_without_identity_but_with_records_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "episodes").mkdir()
    (tmp_path / "episodes" / "0000.jsonl").write_text("{}\n")
    with pytest.raises(SystemExit, match="no run-identity.json"):
        run_native.bind_output(tmp_path, {"suite": BRIDGE})


def test_finished_invocation_is_reused_only_when_it_still_matches(native: SimpleNamespace) -> None:
    first = native.run()
    assert first["returncode"] == 0 and first["episodes"] == 2 and "error" not in first
    assert native.run() == first and len(native.calls()) == 1
    # Another command line under the same index runs again.
    native.run(argv=[*ARGV[:-3], "EnvB", "--robot", "r"])
    assert len(native.calls()) == 2
    # A missing record is not success.
    (native.output / "episodes" / "0000.jsonl").unlink()
    again = native.run(argv=[*ARGV[:-3], "EnvB", "--robot", "r"])
    assert len(native.calls()) == 3 and again["episodes"] == 2
    # Different output settings run again.
    native.run(
        argv=[*ARGV[:-3], "EnvB", "--robot", "r"],
        run_args=argparse.Namespace(**{**vars(native.args), "write_videos": True}),
    )
    assert len(native.calls()) == 4


def test_tampered_record_is_not_reused(native: SimpleNamespace) -> None:
    native.run()
    with (native.output / "episodes" / "0000.jsonl").open("a") as handle:
        handle.write(json.dumps({"env_name": "EnvA"}) + "\n")
    native.run()
    assert len(native.calls()) == 2


def test_exit_zero_without_records_is_a_failure(
    native: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STUB_EPISODES", "0")
    summary = native.run()
    assert summary["error"] == "no episode records"
    assert not (native.output / "episodes" / "0000.done").exists()


def test_records_must_name_their_invocation_configuration() -> None:
    line = json.dumps(
        {"env_name": "EnvA", "policy_model": "rt1", "policy_setup": "google_robot",
         "obs_camera_name": "3rd_view_camera", "ckpt_path": "checkpoints/x"}
    )  # fmt: skip
    assert "obs_camera_name" in run_native.check_records([line], ARGV)
    assert run_native.check_records([line], [*ARGV, "--obs-camera-name", "3rd_view_camera"]) is None


# ---- compare_native: identity and strict acceptance ------------------------------------------

EPISODES_PER_VARIANT = 2


def _fake_parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-model", default="rt1")
    parser.add_argument("--policy-setup", default="google_robot")
    parser.add_argument("--obs-camera-name", default=None)
    parser.add_argument("--env-name")
    parser.add_argument("--scene-name")
    parser.add_argument("--robot")
    parser.add_argument("--control-freq", type=int)
    parser.add_argument("--sim-freq", type=int)
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--rgb-overlay-path", default=None)
    parser.add_argument("--enable-raytracing", action="store_true")
    parser.add_argument("--additional-env-build-kwargs", nargs="*", default=None)
    args, _ = parser.parse_known_args(argv)
    if args.additional_env_build_kwargs is not None:
        args.additional_env_build_kwargs = dict(
            item.split("=", 1) for item in args.additional_env_build_kwargs
        )
    return args


def _fake_episodes(_args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {
            "robot_init_x": 0.1,
            "robot_init_y": 0.2,
            "robot_init_quat": [0.0, 0.0, 0.0, 1.0],
            "obj_episode_id": index,
        }
        for index in range(EPISODES_PER_VARIANT)
    ]


@pytest.fixture
def comparison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.setattr(compare_native, "upstream_args", lambda variant: _fake_parse(list(variant.argv)))
    monkeypatch.setattr(compare_native, "_parse", _fake_parse)
    monkeypatch.setattr(compare_native, "expand_episodes", _fake_episodes)
    variants = [variant for task in SUITE_TASKS[BRIDGE] for variant in task.variants]
    kit_run, native_run = tmp_path / "kit", tmp_path / "native"
    (kit_run / "queue" / "completed").mkdir(parents=True)
    (native_run / "episodes").mkdir(parents=True)
    records = []
    plan = []
    for index, variant in enumerate(variants):
        argv = [*variant.argv, "--policy-model", "rt1", "--ckpt-path", "checkpoints/rt1x"]
        plan.append({"index": index, "argv": argv, "kept": True})
        parsed = _fake_parse(argv)
        lines = []
        for number, episode in enumerate(_fake_episodes(parsed)):
            common = {
                "actions_sha256": "a", "step_action_digests": ["d"], "first_image_sha256": "i",
                "episode_stats": {}, "instructions": ["go"], "truncated": False,
            }  # fmt: skip
            records.append(
                {
                    "record_id": episode_id(variant, number),
                    "success": True,
                    "steps": 3,
                    "metadata": {
                        "variant": variant.variant_id,
                        "task": variant.task,
                        "episode": episode,
                        **common,
                    },
                }
            )
            lines.append(
                {
                    **episode,
                    **common,
                    "env_name": parsed.env_name,
                    "scene_name": parsed.scene_name,
                    "robot": parsed.robot,
                    "control_freq": parsed.control_freq,
                    "sim_freq": parsed.sim_freq,
                    "max_episode_steps": parsed.max_episode_steps,
                    "rgb_overlay_path": parsed.rgb_overlay_path,
                    "enable_raytracing": parsed.enable_raytracing,
                    "additional_env_build_kwargs": parsed.additional_env_build_kwargs or {},
                    "policy_model": parsed.policy_model,
                    "policy_setup": parsed.policy_setup,
                    "obs_camera_name": parsed.obs_camera_name,
                    "success": True,
                    "steps": 3,
                }
            )
        (native_run / "episodes" / f"{index:04d}.jsonl").write_text(
            "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8"
        )
    (kit_run / "queue" / "completed" / "item-0.json").write_text(json.dumps({"records": records}))
    (native_run / "plan.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in plan))

    def run(*extra: str) -> tuple[int, dict]:
        output = tmp_path / "compare.json"
        argv = [
            "compare_native.py",
            "--suite",
            BRIDGE,
            "--kit-run",
            str(kit_run),
            "--native-run",
            str(native_run),
        ]
        monkeypatch.setattr(sys, "argv", [*argv, "--output", str(output), *extra])
        code = compare_native.main()
        return code, json.loads(output.read_text())

    def edit_native(index: int, change) -> None:
        path = native_run / "episodes" / f"{index:04d}.jsonl"
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        path.write_text("".join(json.dumps(line) + "\n" for line in change(lines)))

    def edit_plan(index: int, argv: list[str]) -> None:
        plan[index]["argv"] = argv
        (native_run / "plan.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in plan))

    return SimpleNamespace(
        run=run, edit_native=edit_native, edit_plan=edit_plan, plan=plan, native_run=native_run
    )


def test_complete_comparison_passes_strict(comparison: SimpleNamespace) -> None:
    code, report = comparison.run("--strict")
    assert code == 0 and report["acceptance"]["passed"]
    assert report["coverage"]["paired"] == report["coverage"]["frozen_episodes"] == 4 * EPISODES_PER_VARIANT


def test_missing_native_records_fail_strict_only(comparison: SimpleNamespace) -> None:
    for path in (comparison.native_run / "episodes").glob("*.jsonl"):
        path.unlink()
    code, report = comparison.run()
    assert code == 0 and report["kit_unpaired"] == 4 * EPISODES_PER_VARIANT
    code, report = comparison.run("--strict")
    assert code == 1 and not report["acceptance"]["passed"]


def test_duplicate_native_episode_fails(comparison: SimpleNamespace) -> None:
    comparison.edit_native(0, lambda lines: [*lines, lines[0]])
    code, report = comparison.run()
    assert code == 1 and report["native_duplicate_episodes"] == 1


def test_native_records_outside_the_plan_fail(comparison: SimpleNamespace) -> None:
    (comparison.native_run / "episodes" / "0099.jsonl").write_text(
        (comparison.native_run / "episodes" / "0000.jsonl").read_text()
    )
    code, report = comparison.run()
    assert code == 1 and report["native_unknown_episodes"] == EPISODES_PER_VARIANT


def test_camera_is_part_of_the_protocol_identity(comparison: SimpleNamespace) -> None:
    comparison.edit_plan(0, [*comparison.plan[0]["argv"], "--obs-camera-name", "3rd_view_camera"])
    code, report = comparison.run()
    assert code == 1
    assert {"invocation": 0, "reason": "no Kit variant"} in report["protocol"]["mismatches"]


def test_policy_setup_is_part_of_the_protocol_identity(comparison: SimpleNamespace) -> None:
    argv = list(comparison.plan[0]["argv"])
    argv[argv.index("--policy-setup") + 1] = "google_robot"
    comparison.edit_plan(0, argv)
    code, report = comparison.run()
    assert code == 1 and report["protocol"]["mismatches"]


def test_native_record_contradicting_its_invocation_fails(comparison: SimpleNamespace) -> None:
    comparison.edit_native(1, lambda lines: [{**lines[0], "obs_camera_name": "3rd_view_camera"}, *lines[1:]])
    code, report = comparison.run()
    assert code == 1 and report["native_config_mismatches"] == 1


def test_native_records_without_configuration_fail_strict(comparison: SimpleNamespace) -> None:
    def strip(lines: list[dict]) -> list[dict]:
        return [
            {key: value for key, value in line.items() if key not in compare_native.CONFIG_FIELDS}
            for line in lines
        ]

    comparison.edit_native(2, strip)
    code, report = comparison.run()
    assert code == 0 and report["native_records_without_config"] == EPISODES_PER_VARIANT
    code, _ = comparison.run("--strict")
    assert code == 1


# ---- shard identity and the Vulkan check ------------------------------------------------------


def _factory_adapter(monkeypatch: pytest.MonkeyPatch, suite: str = BRIDGE) -> SimplerEnvBenchmarkAdapter:
    from roboscope_kit.benchmark_specs import BenchmarkName
    from roboscope_kit.runtime.benchmark.factories import BenchmarkRunRequest
    from roboscope_kit.runtime.benchmarks.simpler_env import factory

    def refuse() -> None:
        raise RuntimeError("Vulkan ICDs besides the NVIDIA one are installed")

    monkeypatch.setattr(factory, "vulkan_preflight", refuse)
    manifest = SimpleNamespace(defaults={}, runtime_path=lambda name: Path(f"/host-a/{name}"))
    request = BenchmarkRunRequest(
        benchmark=BenchmarkName.SIMPLER_ENV,
        suite=suite,
        checkpoint_path=Path("/unused"),
        output_dir=Path("/unused"),
        episodes=None,
        quick_test=False,
    )
    return factory.SimplerEnvBenchmarkFactory().create(request, manifest)


def test_adapter_builds_without_the_vulkan_check_but_workers_do_not_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _factory_adapter(monkeypatch)  # what shard_merge does
    assert adapter.metadata()["benchmark"] == "simpler_env"
    with pytest.raises(RuntimeError, match="Vulkan"):
        adapter.worker_process(SimpleNamespace())


def test_benchmark_identity_does_not_depend_on_the_host() -> None:
    adapter = _adapter(BRIDGE)
    other = SimplerEnvBenchmarkAdapter(
        SimplerEnvRuntimeConfig(
            python=Path("/host-b/py"),
            simpler_env_root=Path("/host-b/se"),
            suite=BRIDGE,
            selection=adapter.config.selection,
        )
    )
    assert json.dumps(adapter.metadata(), sort_keys=True) == json.dumps(other.metadata(), sort_keys=True)
    assert "/py" not in json.dumps(adapter.metadata()) and "/se" not in json.dumps(adapter.metadata())
