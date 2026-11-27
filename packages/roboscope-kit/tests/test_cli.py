# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""The ``roboscope`` command: argument resolution, runtime argv, Policy contracts, build and exit codes."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from roboscope_kit import cli
from roboscope_kit.benchmark_specs import BenchmarkName
from roboscope_kit.runtime.benchmark import cli as runtime_cli

NOW = datetime(2026, 11, 27, 9, 30, 5)
PYTHON = os.path.abspath(sys.executable)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolate the runtime registry and ROBOSCOPE_HOME, and run from an empty working directory."""

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.delenv("ROBOSCOPE_HOME", raising=False)
    monkeypatch.delenv("KIT_PYTHON", raising=False)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return tmp_path


def _runtime(root: Path, benchmark: str = "libero", suites=("libero_10",)) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "benchmark-manifest.json").write_text(json.dumps({
        "schema_version": 3,
        "supported_benchmarks": [benchmark],
        "supported_suites": {benchmark: list(suites)},
        "protocol_versions": {benchmark: f"{benchmark}-v1"},
        "runtime": {"python": sys.executable, "benchmark_root": str(root), "config_path": str(root)},
        "defaults": {},
        "benchmark_protocols": {},
    }))
    return root


def _parse(*argv: str):
    return cli.build_parser().parse_args(list(argv))


def _flag(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def test_eval_argv_forwards_every_option_under_its_runtime_name(home):
    runtime = _runtime(home / "sim" / "libero")
    policy = home / "policy.json"
    args = _parse(
        "eval", "libero", "libero_10", "--policy-config", str(policy), "--checkpoint-source", "ckpt",
        "--benchmark-manifest", str(runtime), "--output-dir", "out", "--episodes", "5", "--eval-mode", "fast",
        "--concurrency", "2", "--replicas", "1", "--run-cache", "off", "--cuda-mps", "--profile",
        "--shard-index", "1", "--shard-count", "2", "--shard-episodes", "50",
    )
    invocation = cli.plan_eval(args, NOW)
    argv = invocation.argv
    assert argv[:2] == ["--benchmark", "libero"] and _flag(argv, "--suite") == "libero_10"
    assert _flag(argv, "--benchmark-manifest") == str(runtime / "benchmark-manifest.json")
    assert _flag(argv, "--output-dir") == str(Path.cwd() / "out") == str(invocation.output_dir)
    assert _flag(argv, "--checkpoint-source") == str(Path.cwd() / "ckpt")
    assert _flag(argv, "--policy-config") == str(policy)
    for name, value in (("--episodes", "5"), ("--eval-mode", "fast"), ("--concurrency", "2"),
                        ("--replicas", "1"), ("--run-cache", "off"), ("--shard-index", "1"),
                        ("--shard-count", "2"), ("--shard-episodes", "50")):
        assert _flag(argv, name) == value
    assert {"--cuda-mps", "--profile"} <= set(argv) and "--quick-test" not in argv
    # The assembled list is exactly what the runtime parser accepts.
    parsed = runtime_cli.parse_args(argv)
    assert (parsed.benchmark, parsed.suite, parsed.episodes) == ("libero", "libero_10", 5)
    assert parsed.cuda_mps


def test_eval_leaves_runtime_defaults_to_the_runtime(home):
    _runtime(home / "sim")
    args = _parse("eval", "libero", "libero_10", "--policy-config", "p.json", "--checkpoint-source", "/ckpt",
                  "--benchmark-manifest", str(home / "sim" / "benchmark-manifest.json"))
    argv = cli.plan_eval(args, NOW).argv
    for name in ("--episodes", "--eval-mode", "--run-cache", "--concurrency", "--shard-index"):
        assert name not in argv
    parsed = runtime_cli.parse_args(argv)
    assert parsed.eval_mode == "strict" and parsed.run_cache == "use"


def test_default_output_directory_is_new_and_names_the_run(home):
    _runtime(home / "sim")
    base = ["eval", "libero", "libero_10", "--policy-config", "p.json", "--checkpoint-source", "/ckpt",
            "--benchmark-manifest", str(home / "sim")]
    first = cli.plan_eval(_parse(*base), NOW).output_dir
    assert first == Path.cwd() / "roboscope-runs" / "libero-libero_10-20261127-093005"
    first.mkdir(parents=True)
    assert cli.plan_eval(_parse(*base), NOW).output_dir.name == "libero-libero_10-20261127-093005-2"
    quick = cli.plan_eval(_parse(*base, "--quick-test"), NOW).output_dir
    assert quick.name == "libero-libero_10-quick-20261127-093005"


def test_manifest_comes_from_the_registry_then_roboscope_home(home, monkeypatch):
    args = _parse("eval", "libero", "libero_10", "--policy-config", "p.json", "--checkpoint-source", "/ckpt")
    with pytest.raises(cli.UsageError, match="roboscope build libero"):
        cli.plan_eval(args, NOW)
    monkeypatch.setenv("ROBOSCOPE_HOME", str(home / "rh"))
    _runtime(home / "rh" / "libero")
    assert _flag(cli.plan_eval(args, NOW).argv, "--benchmark-manifest") == str(
        home / "rh" / "libero" / "benchmark-manifest.json")
    registered = _runtime(home / "data" / "libero-runtime")
    cli.register_runtime(BenchmarkName.LIBERO, registered)
    assert _flag(cli.plan_eval(args, NOW).argv, "--benchmark-manifest") == str(
        registered / "benchmark-manifest.json")
    stored = json.loads((home / "config" / "roboscope-kit" / "runtimes.json").read_text())
    assert stored == {"schema_version": 1, "runtimes": {"libero": str(registered)}}


def test_missing_manifest_is_a_usage_error(home, capsys):
    code = cli.main(["eval", "libero", "libero_10", "--policy-config", "p.json", "--checkpoint-source", "/c",
                     "--benchmark-manifest", str(home / "nowhere")])
    assert code == cli.EXIT_USAGE
    assert "benchmark manifest not found" in capsys.readouterr().err


def test_policy_file_becomes_a_content_addressed_policy_config(home):
    script = home / "agent" / "my_policy.py"
    script.parent.mkdir()
    script.write_text("")
    args = _parse("eval", "libero", "libero_10", "--policy", str(script), "--checkpoint-source", "/ckpt")
    path = cli.policy_from_args(args, BenchmarkName.LIBERO)
    text = path.read_text()
    expected = {"schema_version": 2, "command": [PYTHON, str(script)], "cwd": str(script.parent)}
    assert json.loads(text) == expected
    assert path == home / "cache" / "roboscope-kit" / "policy-configs" / (
        hashlib.sha256(text.encode()).hexdigest() + ".json")
    assert cli.policy_from_args(args, BenchmarkName.LIBERO) == path  # same input, same file
    _runtime(home / "sim")
    invocation = cli.plan_eval(_parse("eval", "libero", "libero_10", "--policy", str(script),
                                      "--checkpoint-source", "/ckpt", "--benchmark-manifest",
                                      str(home / "sim")), NOW)
    assert _flag(invocation.argv, "--policy-config") == str(path)
    assert not str(path).startswith(str(invocation.output_dir))


def test_policy_python_without_the_kit_gets_it_injected(home, monkeypatch):
    script = home / "my_policy.py"
    script.write_text("")
    other = home / "env" / "bin" / "python"
    other.parent.mkdir(parents=True)
    other.write_text("")
    other.chmod(0o755)
    monkeypatch.setenv("PYTHONPATH", "/existing")
    args = _parse("eval", "libero", "libero_10", "--policy", str(script), "--policy-python", str(other),
                  "--policy-cwd", str(home), "--checkpoint-source", "/ckpt")
    seen = []
    path = cli.policy_from_args(
        args, BenchmarkName.LIBERO,
        imports_kit=lambda python: seen.append(python) or False, kit_root=lambda: Path("/kit-root"),
    )
    assert seen == [str(other)]
    config = json.loads(path.read_text())
    assert config["command"] == ["/usr/bin/env", "PYTHONPATH=/kit-root:/existing", str(other), str(script)]
    assert config["cwd"] == str(home)
    kept = cli.policy_from_args(args, BenchmarkName.LIBERO, imports_kit=lambda python: True)
    assert json.loads(kept.read_text())["command"] == [str(other), str(script)]


def test_policy_file_is_refused_where_benchmark_config_is_needed(home, capsys):
    script = home / "p.py"
    script.write_text("")
    code = cli.main(["eval", "metaworld", "mt50", "--policy", str(script), "--checkpoint-source", "/c"])
    assert code == cli.EXIT_USAGE
    assert "use --policy-config" in capsys.readouterr().err


@pytest.mark.parametrize("extra, message", [
    ([], "give the Policy"),
    (["--policy-config", "p.json", "--policy-cwd", "/tmp"], "require --policy"),
])
def test_policy_arguments_are_validated(home, capsys, extra, message):
    _runtime(home / "sim")
    code = cli.main(["eval", "libero", "libero_10", "--checkpoint-source", "/c",
                     "--benchmark-manifest", str(home / "sim"), *extra])
    assert code == cli.EXIT_USAGE
    assert message in capsys.readouterr().err


def test_policy_file_and_config_are_exclusive(home, capsys):
    code = cli.main(["eval", "libero", "libero_10", "--checkpoint-source", "/c", "--policy", "a.py",
                     "--policy-config", "b.json"])
    assert code == cli.EXIT_USAGE
    assert "not allowed with argument" in capsys.readouterr().err


def test_unknown_suite_is_refused_before_the_runtime_lookup(home, capsys):
    code = cli.main(["eval", "libero", "libero_99", "--policy-config", "p.json", "--checkpoint-source", "/c"])
    assert code == cli.EXIT_USAGE
    error = capsys.readouterr().err
    assert "libero has no suite 'libero_99'; its suites are libero_spatial, libero_object" in error
    assert "registered" not in error


@pytest.mark.parametrize("argv, message", [
    (["eval", "libero", "libero_10", "--policy-config", "p.json", "--checkpoint-source", "/c"],
     "no libero runtime is registered"),
    (["eval", "libero", "libero_10", "--json"], "--checkpoint-source"),
])
def test_startup_errors_are_one_json_line_with_json(home, capsys, argv, message):
    code = cli.main([*argv, "--json"] if "--json" not in argv else argv)
    assert code == cli.EXIT_USAGE
    summary = json.loads(capsys.readouterr().out)
    assert summary["command"] == "eval" and summary["exit_code"] == cli.EXIT_USAGE
    assert message in summary["error"]


def test_check_argv_runs_the_self_test(home):
    runtime = _runtime(home / "sim")
    invocation = cli.plan_check(_parse("check", "libero", str(runtime), "--output-dir", "report"), NOW)
    assert invocation.argv == [
        "--benchmark", "libero", "--self-test",
        "--benchmark-manifest", str(runtime / "benchmark-manifest.json"),
        "--output-dir", str(Path.cwd() / "report"),
    ]
    assert runtime_cli.parse_args(invocation.argv).self_test
    default = cli.plan_check(_parse("check", "libero", str(runtime)), NOW).output_dir
    assert default.name == "libero-self-test-20261127-093005"


def _fake_runtime(monkeypatch, result: dict | None, code: int = 0, seen: list | None = None):
    def main(argv):
        if seen is not None:
            seen.append(argv)
        output = Path(argv[argv.index("--output-dir") + 1])
        output.mkdir(parents=True, exist_ok=True)
        if result is not None:
            (output / "result.json").write_text(json.dumps(result))
        return code

    monkeypatch.setattr(runtime_cli, "main", main)


def test_eval_prints_one_summary_line(home, monkeypatch, capsys):
    _runtime(home / "sim")
    script = home / "p.py"
    script.write_text("")
    seen = []
    _fake_runtime(monkeypatch, {"result": {"episodes": 10, "successes": 7, "success_rate": 0.7},
                                "phase_timings": {"evaluation_s": 42.0}}, seen=seen)
    code = cli.main(["eval", "libero", "libero_10", "--policy", str(script), "--checkpoint-source", "/c",
                     "--benchmark-manifest", str(home / "sim"), "--output-dir", "out"])
    assert code == cli.EXIT_OK
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    assert "libero/libero_10: 7/10 episodes succeeded (70.0%), evaluation 42 s" in out[0]
    config = json.loads(Path(_flag(seen[0], "--policy-config")).read_text())
    assert config["command"] == [PYTHON, str(script)]


def test_eval_json_summary_and_failure_exit_code(home, monkeypatch, capsys):
    _runtime(home / "sim")
    _fake_runtime(monkeypatch, None, code=1)
    code = cli.main(["eval", "libero", "libero_10", "--policy-config", "p.json", "--checkpoint-source", "/c",
                     "--benchmark-manifest", str(home / "sim"), "--output-dir", "out", "--json"])
    assert code == cli.EXIT_FAILED
    summary = json.loads(capsys.readouterr().out)
    assert summary["exit_code"] == 1 and summary["output_dir"] == str(Path.cwd() / "out")
    assert "episodes" not in summary and summary.get("error") is None


def test_eval_passes_the_signal_exit_code_through(home, monkeypatch, capsys):
    _runtime(home / "sim")

    def terminated(argv):
        raise SystemExit(128 + 15)

    monkeypatch.setattr(runtime_cli, "main", terminated)
    code = cli.main(["eval", "libero", "libero_10", "--policy-config", "p.json", "--checkpoint-source", "/c",
                     "--benchmark-manifest", str(home / "sim"), "--output-dir", "out"])
    assert code == 143
    assert "cannot be resumed" in capsys.readouterr().err


def test_eval_runs_through_the_runtime_with_a_generated_policy_config(home, monkeypatch):
    """The unchanged runtime accepts a generated contract and keys the run cache on its stable text."""

    runtime = _runtime(home / "sim")
    script = home / "p.py"
    script.write_text("")
    keys = []
    monkeypatch.setattr(runtime_cli, "detect_gpu_devices", lambda: ("0",))
    monkeypatch.setattr(runtime_cli, "_autotuned_run", lambda *a, **k: ({"result": {}}, {"k": 1, "g": 1}))
    original = runtime_cli.RunCache

    def recording_cache(mode, fields):
        keys.append(fields.get("policy_config"))
        return original("off", {})

    monkeypatch.setattr(runtime_cli, "RunCache", recording_cache)
    argv = ["eval", "libero", "libero_10", "--policy", str(script), "--checkpoint-source", str(home),
            "--benchmark-manifest", str(runtime)]
    assert cli.main([*argv, "--output-dir", "a"]) == cli.EXIT_OK
    assert cli.main([*argv, "--output-dir", "b"]) == cli.EXIT_OK
    assert keys[0] == keys[1] == cli.policy_config_text([PYTHON, str(script)], home)
    assert {p.name for p in (Path.cwd() / "a").iterdir()} >= {"result.json"}
    assert not list((Path.cwd() / "a").glob("*policy*"))


def test_build_refuses_benchmarks_whose_script_needs_a_checkout(home, capsys):
    assert cli.main(["build", "robodojo", str(home / "rd")]) == cli.EXIT_USAGE
    error = capsys.readouterr().err
    assert "git clone https://github.com/MINT-SJTU/RoboScope-Kit" in error
    assert f"bash RoboScope-Kit/benchmarks/robodojo/build-runtime.sh {home / 'rd'}" in error


def test_build_needs_a_directory_or_roboscope_home(home, capsys):
    assert cli.main(["build", "libero"]) == cli.EXIT_USAGE
    assert "ROBOSCOPE_HOME" in capsys.readouterr().err


def test_build_refuses_a_disk_without_room(home, monkeypatch, capsys):
    monkeypatch.setattr(cli.shutil, "disk_usage", lambda path: SimpleNamespace(free=2 * 10**9))
    assert cli.main(["build", "robotwin", str(home / "rt")]) == cli.EXIT_USAGE
    assert "needs about 45 GB" in capsys.readouterr().err


def test_build_runs_the_script_and_registers_the_runtime(home, monkeypatch):
    monkeypatch.setenv("ROBOSCOPE_HOME", str(home / "rh"))
    monkeypatch.setattr(cli.shutil, "disk_usage", lambda path: SimpleNamespace(free=10**12))
    calls = []

    def run(command, env, check):
        calls.append((command, env["KIT_PYTHON"]))
        return SimpleNamespace(returncode=0)

    args = _parse("build", "libero")
    assert cli.command_build(args, run=run) == cli.EXIT_OK
    command, kit_python = calls[0]
    assert command[0] == "bash" and Path(command[1]).name == "build-runtime.sh"
    assert Path(command[1]).parent.name == "libero" and command[2] == str(home / "rh" / "libero")
    assert kit_python == PYTHON
    assert cli.registered_runtimes() == {"libero": str(home / "rh" / "libero")}
    failed = cli.command_build(args, run=lambda *a, **k: SimpleNamespace(returncode=4))
    assert failed == cli.EXIT_FAILED


def test_every_buildable_benchmark_has_a_script():
    for benchmark in cli.BUILDABLE_BENCHMARKS:
        assert cli.build_script(benchmark).is_file()


def test_merge_finds_the_runtime_and_defaults_the_output(home):
    runtime = _runtime(home / "sim" / "robotwin", "robotwin", ("demo_clean",))
    cli.register_runtime(BenchmarkName.ROBOTWIN, runtime)
    invocation = cli.plan_merge(_parse("merge", "robotwin", "demo_clean", "--shard-dir", "s0",
                                       "--shard-dir", "s1"), NOW)
    assert _flag(invocation.argv, "--benchmark-manifest") == str(runtime / "benchmark-manifest.json")
    expected = Path.cwd() / "roboscope-runs" / "robotwin-demo_clean-merged-20261127-093005"
    assert invocation.output_dir == expected
    assert invocation.argv.count("--shard-dir") == 2


def test_merge_refuses_an_unknown_suite(home, capsys):
    assert cli.main(["merge", "robotwin", "demo_nope", "--shard-dir", "s0"]) == cli.EXIT_USAGE
    assert "robotwin has no suite 'demo_nope'" in capsys.readouterr().err


def test_merge_calls_the_shard_merge(home, monkeypatch, capsys):
    from roboscope_kit.runtime.benchmark import merge

    _runtime(home / "sim", "robotwin", ("demo_clean",))
    seen = []
    monkeypatch.setattr(merge, "main", lambda argv: seen.append(argv) or 0)
    code = cli.main(["merge", "robotwin", "demo_clean", "--shard-dir", "s0",
                     "--benchmark-manifest", str(home / "sim"), "--output-dir", "m"])
    assert code == cli.EXIT_OK and len(seen) == 1
    assert "merged into" in capsys.readouterr().out
    monkeypatch.setattr(merge, "main", lambda argv: 1)
    assert cli.main(["merge", "robotwin", "demo_clean", "--shard-dir", "s0",
                     "--benchmark-manifest", str(home / "sim"), "--output-dir", "m2"]) == cli.EXIT_FAILED


def test_eval_explains_a_missing_driver(home, monkeypatch, capsys):
    _runtime(home / "sim")
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    code = cli.main(["eval", "libero", "libero_10", "--policy-config", "p.json", "--checkpoint-source", "/c",
                     "--benchmark-manifest", str(home / "sim")])
    assert code == cli.EXIT_USAGE
    assert "nvidia-smi was not found" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------------------
# Equivalence: the new commands hand the runtime exactly what the earlier commands passed.


def _same(parsed_new, parsed_old) -> None:
    assert vars(parsed_new) == vars(parsed_old)


def test_eval_parses_like_the_earlier_command(home):
    runtime = _runtime(home / "sim" / "libero")
    manifest = runtime / "benchmark-manifest.json"
    out = home / "runs" / "x"
    options = ["--episodes", "5", "--eval-mode", "fast", "--concurrency", "2", "--replicas", "1",
               "--run-cache", "refresh", "--autotune-strategy", "two-phase", "--cuda-mps", "--profile",
               "--shard-index", "1", "--shard-count", "2", "--shard-episodes", "50"]
    for extra in ([], ["--quick-test"], options):
        old = ["--benchmark", "libero", "--suite", "libero_10", "--checkpoint-source", "/models/c",
               "--benchmark-manifest", str(manifest), "--policy-config", "/p/roboscope-policy.json",
               "--output-dir", str(out), *extra]
        new = cli.plan_eval(_parse("eval", "libero", "libero_10", "--checkpoint-source", "/models/c",
                                   "--benchmark-manifest", str(manifest),
                                   "--policy-config", "/p/roboscope-policy.json",
                                   "--output-dir", str(out), *extra), NOW).argv
        _same(runtime_cli.parse_args(new), runtime_cli.parse_args(old))


@pytest.mark.parametrize("benchmark", [item.value for item in BenchmarkName])
def test_check_parses_like_the_earlier_self_test(home, benchmark):
    runtime = _runtime(home / "sim" / benchmark, benchmark, ("s",))
    manifest = runtime / "benchmark-manifest.json"
    old = ["--benchmark", benchmark, "--self-test", "--benchmark-manifest", str(manifest),
           "--output-dir", str(runtime / "self-test")]
    for location in (runtime, manifest):  # what the build scripts pass, and a manifest path
        new = cli.plan_check(_parse("check", benchmark, str(location), "--output-dir",
                                    str(runtime / "self-test")), NOW).argv
        _same(runtime_cli.parse_args(new), runtime_cli.parse_args(old))


def test_merge_parses_like_the_earlier_command(home):
    from roboscope_kit.runtime.benchmark import merge

    runtime = _runtime(home / "sim" / "robotwin", "robotwin", ("demo_clean",))
    manifest = runtime / "benchmark-manifest.json"
    for extra in ([], ["--episodes", "10"]):
        old = ["--benchmark", "robotwin", "--suite", "demo_clean", *extra,
               "--benchmark-manifest", str(manifest),
               "--shard-dir", "/runs/shard-0", "--shard-dir", "/runs/shard-1", "--output-dir", "/runs/merged"]
        new = cli.plan_merge(_parse("merge", "robotwin", "demo_clean", "--shard-dir", "/runs/shard-0",
                                    "--shard-dir", "/runs/shard-1", "--benchmark-manifest", str(runtime),
                                    "--output-dir", "/runs/merged", *extra), NOW).argv
        _same(merge.parse_args(new), merge.parse_args(old))
