# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""The ``roboscope`` command: evaluate a Policy, check or build a benchmark runtime, merge shards, hosts.

``roboscope`` and ``python -m roboscope_kit`` are the same program. It only assembles arguments: ``eval``
and ``check`` call the unchanged benchmark runtime (:mod:`roboscope_kit.runtime.benchmark.cli`) with the
argument list the earlier ``python -m roboscope_kit.runtime.benchmark`` command took, and ``merge`` calls
the unchanged shard merge with its own arguments. ``eval --hosts`` and ``hosts`` (:mod:`roboscope_kit.hosts`)
run those same commands on registered SSH hosts.

Exit codes: 0 success; 1 the evaluation, check, build or merge ran and failed; 2 the command did not start
because of its arguments or a failed precondition; 128 + N after signal N.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from roboscope_kit.benchmark_specs import BenchmarkName
from roboscope_kit.policy import EVAL_MODES
from roboscope_kit.policy_config import POLICY_CONFIG_SCHEMA_VERSION, load_policy_runtime_config
from roboscope_kit.runtime.benchmark.run_cache import CACHE_MODES

PROG = "roboscope"
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
MANIFEST_NAME = "benchmark-manifest.json"
REPOSITORY_URL = "https://github.com/MINT-SJTU/RoboScope-Kit"

# Benchmarks whose build script reads only its own directory, so the copy shipped in the wheel can run it.
# The values are the free space, in GB, a first build needs: rough upper bounds from the asset sizes the
# scripts state plus their environments.
BUILDABLE_BENCHMARKS: dict[BenchmarkName, int] = {
    BenchmarkName.CALVIN: 10,
    BenchmarkName.LIBERO: 10,
    BenchmarkName.LIBERO_PLUS: 25,
    BenchmarkName.LIBERO_PRO: 10,
    BenchmarkName.ROBOCASA: 20,
    BenchmarkName.ROBOTWIN: 45,
}

# Benchmarks whose evaluation reads ``benchmark_config`` from the Policy config, which ``--policy`` lacks.
POLICY_CONFIG_REQUIRED: frozenset[BenchmarkName] = frozenset({BenchmarkName.METAWORLD})


def _absolute(path: str | Path) -> Path:
    """An absolute, normalized path; symlinks are kept, so a virtual environment's Python stays itself."""

    return Path(os.path.abspath(Path(path).expanduser()))


class UsageError(Exception):
    """The command cannot start; reported with exit code 2."""


class _ArgumentParser(argparse.ArgumentParser):
    """Raise argument errors so ``main`` can also report them as JSON."""

    def error(self, message: str):
        raise _ParseError(self, message)


class _ParseError(Exception):
    def __init__(self, parser: argparse.ArgumentParser, message: str) -> None:
        super().__init__(message)
        self.parser = parser


# ---------------------------------------------------------------------------------------------------------
# Locations


def roboscope_home() -> Path | None:
    value = os.environ.get("ROBOSCOPE_HOME", "").strip()
    return _absolute(value) if value else None


def registry_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME", "").strip() or str(Path.home() / ".config")
    return Path(base) / "roboscope-kit" / "runtimes.json"


def registered_runtimes() -> dict[str, str]:
    path = registry_path()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        raise UsageError(f"cannot read the runtime registry {path}: {exc}") from exc
    runtimes = value.get("runtimes") if isinstance(value, dict) else None
    if not isinstance(runtimes, dict):
        raise UsageError(f"the runtime registry {path} has no runtimes object")
    return {str(name): str(location) for name, location in runtimes.items()}


def register_runtime(benchmark: BenchmarkName, install_dir: Path) -> Path:
    path = registry_path()
    runtimes = registered_runtimes()
    runtimes[benchmark.value] = str(install_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{os.getpid()}")
    staging.write_text(
        json.dumps({"schema_version": 1, "runtimes": dict(sorted(runtimes.items()))}, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(staging, path)
    return path


def manifest_in(location: Path) -> Path:
    """The manifest a ``--benchmark-manifest`` value or runtime directory names."""

    path = _absolute(location)
    if path.is_dir():
        path = path / MANIFEST_NAME
    if not path.is_file():
        raise UsageError(f"benchmark manifest not found: {path}")
    return path


def resolve_manifest(benchmark: BenchmarkName, given: Path | None) -> Path:
    """An explicit manifest or runtime directory, else the registered runtime, else ``$ROBOSCOPE_HOME``."""

    if given is not None:
        return manifest_in(given)
    registered = registered_runtimes().get(benchmark.value)
    if registered is not None:
        return manifest_in(Path(registered))
    home = roboscope_home()
    if home is not None and (home / benchmark.value / MANIFEST_NAME).is_file():
        return home / benchmark.value / MANIFEST_NAME
    raise UsageError(
        f"no {benchmark.value} runtime is registered; build one with "
        f"`{PROG} build {benchmark.value} <install-dir>` or give its runtime directory or "
        "benchmark-manifest.json"
    )


def default_output_dir(benchmark: BenchmarkName, label: str, now: datetime, root: Path) -> Path:
    """A new directory under ``root``: ``<benchmark>-<label>-<YYYYmmdd-HHMMSS>[-<n>]``."""

    stem = f"{benchmark.value}-{label}-{now.strftime('%Y%m%d-%H%M%S')}"
    candidate = root / stem
    index = 1
    while candidate.exists():
        index += 1
        candidate = root / f"{stem}-{index}"
    return candidate


def shipped_benchmark_dir(benchmark: BenchmarkName) -> Path | None:
    """The benchmark's directory shipped in the wheel, or a source checkout's own ``benchmarks/`` copy."""

    package = Path(__file__).resolve().parent
    for root in (package / "_benchmarks", package.parents[3] / "benchmarks"):
        if (root / benchmark.value / "build-runtime.sh").is_file():
            return root / benchmark.value
    return None


def build_script(benchmark: BenchmarkName) -> Path:
    directory = shipped_benchmark_dir(benchmark)
    if directory is not None:
        return directory / "build-runtime.sh"
    raise UsageError(
        f"this roboscope-kit installation carries no build script for {benchmark.value}; reinstall it "
        f"from PyPI or build from a clone of {REPOSITORY_URL}"
    )


def shipped_suites(benchmark: BenchmarkName) -> tuple[str, ...]:
    directory = shipped_benchmark_dir(benchmark)
    if directory is None:
        return ()
    try:
        manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    return tuple(manifest.get("supported_suites", {}).get(benchmark.value, ()))


def validate_suite(benchmark: BenchmarkName, suite: str) -> None:
    """Refuse a suite the benchmark does not define, before any runtime is looked up."""

    if suite in shipped_suites(benchmark):
        return
    from roboscope_kit.runtime.benchmark.registry import benchmark_factory

    try:
        benchmark_factory(benchmark).validate_suite(suite)
    except ValueError as exc:
        suites = shipped_suites(benchmark)
        listed = f"; its suites are {', '.join(suites)}" if suites else ""
        raise UsageError(f"{benchmark.value} has no suite {suite!r}{listed}") from exc


def kit_version() -> str:
    try:
        return version("roboscope-kit")
    except PackageNotFoundError:
        return "unknown"


def gpu_names() -> list[str]:
    """The NVIDIA GPUs ``nvidia-smi`` lists, or none when it is missing or fails."""

    if shutil.which("nvidia-smi") is None:
        return []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def checkpoint_summary(checkpoint: Path) -> dict[str, Any]:
    """Whether the checkpoint exists, its file count and size, and a digest of its file names and sizes."""

    if not checkpoint.exists():
        return {"path": str(checkpoint), "exists": False}
    files = [checkpoint] if checkpoint.is_file() else sorted(p for p in checkpoint.rglob("*") if p.is_file())
    entries = [[str(p.relative_to(checkpoint)) if p != checkpoint else "", p.stat().st_size] for p in files]
    encoded = json.dumps(entries, separators=(",", ":")).encode("utf-8")
    return {
        "path": str(checkpoint),
        "exists": True,
        "files": len(entries),
        "bytes": sum(size for _, size in entries),
        "digest": hashlib.sha256(encoded).hexdigest(),
    }


def output_dir_in_use(output_dir: Path) -> bool:
    """Whether the runtime would refuse ``output_dir``: it holds anything but an error-only retry."""

    if not output_dir.is_dir():
        return output_dir.exists()
    return any(entry.name != "errors.jsonl" for entry in output_dir.iterdir())


# ---------------------------------------------------------------------------------------------------------
# Policy


def python_imports_kit(python: str) -> bool:
    if _absolute(python) == _absolute(sys.executable):
        return True
    try:
        result = subprocess.run(
            [python, "-c", "import roboscope_kit"], capture_output=True, timeout=120, check=False,
        )
    except OSError as exc:
        raise UsageError(f"cannot run --policy-python {python}: {exc}") from exc
    return result.returncode == 0


def policy_configs_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME", "").strip() or str(Path.home() / ".cache")
    return Path(base) / "roboscope-kit" / "policy-configs"


def policy_config_text(command: Sequence[str], cwd: Path) -> str:
    """The canonical JSON of a Policy process contract: equal contracts give equal text."""

    value = {"schema_version": POLICY_CONFIG_SCHEMA_VERSION, "command": list(command), "cwd": str(cwd)}
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def write_policy_config(text: str) -> Path:
    """Store a generated contract outside every output directory, named by its SHA-256."""

    directory = policy_configs_dir()
    path = directory / f"{hashlib.sha256(text.encode('utf-8')).hexdigest()}.json"
    if not path.is_file() or path.read_text(encoding="utf-8") != text:
        directory.mkdir(parents=True, exist_ok=True)
        staging = path.with_name(f".{path.name}.{os.getpid()}")
        staging.write_text(text, encoding="utf-8")
        os.replace(staging, path)
    return path


def policy_from_args(
    args: argparse.Namespace,
    benchmark: BenchmarkName,
    *,
    imports_kit: Callable[[str], bool] = python_imports_kit,
    kit_root: Callable[[], Path] | None = None,
) -> Path:
    """The Policy contract file: ``--policy-config`` itself, or the one ``--policy`` describes."""

    if args.policy is None:
        if args.policy_python is not None or args.policy_cwd is not None:
            raise UsageError("--policy-python and --policy-cwd require --policy")
        if args.policy_config is None:
            raise UsageError("give the Policy with --policy FILE or --policy-config JSON")
        return _absolute(args.policy_config)
    if benchmark in POLICY_CONFIG_REQUIRED:
        raise UsageError(
            f"{benchmark.value} reads benchmark_config from the Policy config, which --policy cannot "
            "provide; use --policy-config"
        )
    policy_file = _absolute(args.policy)
    if not policy_file.is_file():
        raise UsageError(f"--policy file not found: {policy_file}")
    python = str(_absolute(args.policy_python or sys.executable))
    cwd = _absolute(args.policy_cwd) if args.policy_cwd else policy_file.parent
    command = [python, str(policy_file)]
    if not imports_kit(python):
        # The Policy's interpreter lacks the kit: put this installation's kit on its path, the way
        # simulator Workers import it.
        if kit_root is None:
            from roboscope_kit.runtime.benchmark.kit_path import kit_import_root

            kit_root = kit_import_root
        path = str(kit_root())
        inherited = os.environ.get("PYTHONPATH")
        if inherited:
            path = f"{path}{os.pathsep}{inherited}"
        command = ["/usr/bin/env", f"PYTHONPATH={path}", *command]
    path = write_policy_config(policy_config_text(command, cwd))
    try:
        load_policy_runtime_config(path)
    except ValueError as exc:
        raise UsageError(str(exc)) from exc
    return path


# ---------------------------------------------------------------------------------------------------------
# eval


def eval_runtime_argv(
    args: argparse.Namespace, benchmark: BenchmarkName, manifest: Path, output_dir: Path, policy_config: Path,
) -> list[str]:
    """The benchmark runtime's argument list for ``roboscope eval``."""

    argv = [
        "--benchmark", benchmark.value,
        "--suite", args.suite,
        "--benchmark-manifest", str(manifest),
        "--output-dir", str(output_dir),
    ]
    if args.checkpoint_source is not None:
        argv += ["--checkpoint-source", str(_absolute(args.checkpoint_source))]
    argv += ["--policy-config", str(policy_config)]
    for flag, value in (
        ("--episodes", args.episodes),
        ("--eval-mode", args.eval_mode),
        ("--shard-index", args.shard_index),
        ("--shard-count", args.shard_count),
        ("--shard-episodes", args.shard_episodes),
        ("--autotune-strategy", args.autotune_strategy),
        ("--run-cache", args.run_cache),
        ("--concurrency", args.concurrency),
        ("--replicas", args.replicas),
    ):
        if value is not None:
            argv += [flag, str(value)]
    for flag, enabled in (
        ("--quick-test", args.quick_test), ("--cuda-mps", args.cuda_mps), ("--profile", args.profile),
    ):
        if enabled:
            argv.append(flag)
    return argv


@dataclass(frozen=True)
class Invocation:
    """What ``eval`` or ``check`` hands to the benchmark runtime."""

    argv: list[str]
    output_dir: Path


def plan_eval(args: argparse.Namespace, now: datetime | None = None) -> Invocation:
    benchmark = BenchmarkName(args.benchmark)
    validate_suite(benchmark, args.suite)
    policy = policy_from_args(args, benchmark)
    manifest = resolve_manifest(benchmark, args.benchmark_manifest)
    output_dir = (
        _absolute(args.output_dir) if args.output_dir is not None
        else default_output_dir(
            benchmark, args.suite + ("-quick" if args.quick_test else ""),
            now or datetime.now(), Path.cwd() / "roboscope-runs",
        )
    )
    return Invocation(eval_runtime_argv(args, benchmark, manifest, output_dir, policy), output_dir)


def eval_summary(benchmark: str, suite: str, output_dir: Path, exit_code: int, wall_s: float) -> dict:
    summary: dict[str, Any] = {
        "command": "eval",
        "benchmark": benchmark,
        "suite": suite,
        "exit_code": exit_code,
        "output_dir": str(output_dir),
        "wall_s": round(wall_s, 1),
    }
    try:
        result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        result = None
    if exit_code != EXIT_OK:
        try:
            lines = (output_dir / "errors.jsonl").read_text(encoding="utf-8").splitlines()
            summary["error"] = json.loads(lines[-1]).get("message") if lines else None
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
    if isinstance(result, dict):
        standard = result.get("result") or {}
        summary.update({
            "episodes": standard.get("episodes"),
            "successes": standard.get("successes"),
            "success_rate": standard.get("success_rate"),
            "evaluation_s": (result.get("phase_timings") or {}).get("evaluation_s"),
            "autotune": result.get("autotune"),
        })
    return summary


def format_eval_summary(summary: dict) -> str:
    target = f"{summary['benchmark']}/{summary['suite']}"
    if summary["exit_code"] == EXIT_OK and summary.get("episodes") is not None:
        rate = summary.get("success_rate")
        rate_text = f" ({rate:.1%})" if isinstance(rate, (int, float)) else ""
        evaluation_s = summary.get("evaluation_s")
        timing = f", evaluation {evaluation_s:.0f} s" if isinstance(evaluation_s, (int, float)) else ""
        return (
            f"{PROG} eval {target}: {summary['successes']}/{summary['episodes']} episodes succeeded"
            f"{rate_text}{timing}, wall {summary['wall_s']:.0f} s; results in {summary['output_dir']}"
        )
    return (
        f"{PROG} eval {target}: failed with exit code {summary['exit_code']}; see "
        f"{summary['output_dir']}/errors.jsonl. A run cannot be resumed: run the command again with a new "
        "--output-dir (the default creates one) to start over"
    )


def run_runtime(invocation: Invocation) -> int:
    """Run the benchmark runtime in this process and map its outcome to an exit code."""

    from roboscope_kit.runtime.benchmark import cli as runtime_cli

    try:
        return EXIT_OK if runtime_cli.main(invocation.argv) == 0 else EXIT_FAILED
    except SystemExit as exc:  # The runtime turns SIGTERM and SIGHUP into SystemExit(128 + N).
        return exc.code if isinstance(exc.code, int) else EXIT_FAILED
    except KeyboardInterrupt:
        return 128 + 2
    except Exception:
        traceback.print_exc()
        return EXIT_FAILED


def require_nvidia_driver() -> None:
    """Explain a missing driver before the runtime would fail on ``nvidia-smi``."""

    if "CUDA_VISIBLE_DEVICES" not in os.environ and shutil.which("nvidia-smi") is None:
        raise UsageError("nvidia-smi was not found; evaluation needs an NVIDIA GPU with its driver installed")


def dry_run_report(args: argparse.Namespace, invocation: Invocation) -> dict[str, Any]:
    """What ``eval`` would run here, with what a multi-host run compares across hosts."""

    argv = invocation.argv
    manifest = Path(argv[argv.index("--benchmark-manifest") + 1])
    checkpoint = Path(argv[argv.index("--checkpoint-source") + 1])
    return {
        "command": "eval",
        "dry_run": True,
        "benchmark": args.benchmark,
        "suite": args.suite,
        "kit_version": kit_version(),
        "python": str(_absolute(sys.executable)),
        "benchmark_manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "checkpoint": checkpoint_summary(checkpoint),
        "policy_config": argv[argv.index("--policy-config") + 1],
        "output_dir": str(invocation.output_dir),
        "output_dir_in_use": output_dir_in_use(invocation.output_dir),
        "gpus": gpu_names(),
        "runtime_argv": argv,
    }


def command_eval(args: argparse.Namespace) -> int:
    if args.hosts is not None:
        from roboscope_kit import hosts

        return hosts.run(args)
    if args.detach:
        raise UsageError("--detach needs --hosts")
    invocation = plan_eval(args)
    if args.dry_run:
        print(json.dumps(dry_run_report(args, invocation), sort_keys=True, indent=None if args.json else 2))
        return EXIT_OK
    require_nvidia_driver()
    started = time.monotonic()
    exit_code = run_runtime(invocation)
    summary = eval_summary(
        args.benchmark, args.suite, invocation.output_dir, exit_code, time.monotonic() - started,
    )
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    else:
        print(format_eval_summary(summary), file=sys.stderr if exit_code else sys.stdout)
    return exit_code


# ---------------------------------------------------------------------------------------------------------
# merge


def plan_merge(args: argparse.Namespace, now: datetime | None = None) -> Invocation:
    """The shard merge's argument list for ``roboscope merge``."""

    benchmark = BenchmarkName(args.benchmark)
    validate_suite(benchmark, args.suite)
    manifest = resolve_manifest(benchmark, args.benchmark_manifest)
    output_dir = (
        _absolute(args.output_dir) if args.output_dir is not None
        else default_output_dir(
            benchmark, f"{args.suite}-merged", now or datetime.now(), Path.cwd() / "roboscope-runs",
        )
    )
    argv = ["--benchmark", benchmark.value, "--suite", args.suite]
    if args.episodes is not None:
        argv += ["--episodes", str(args.episodes)]
    argv += ["--benchmark-manifest", str(manifest)]
    for shard_dir in args.shard_dir:
        argv += ["--shard-dir", str(_absolute(shard_dir))]
    argv += ["--output-dir", str(output_dir)]
    return Invocation(argv, output_dir)


def command_merge(args: argparse.Namespace) -> int:
    from roboscope_kit.runtime.benchmark import merge

    invocation = plan_merge(args)
    exit_code = EXIT_OK if merge.main(invocation.argv) == 0 else EXIT_FAILED
    if exit_code == EXIT_OK:
        result = invocation.output_dir / "result.json"
        print(f"{PROG} merge {args.benchmark}/{args.suite}: merged into {result}")
    return exit_code


# ---------------------------------------------------------------------------------------------------------
# check


def plan_check(args: argparse.Namespace, now: datetime | None = None) -> Invocation:
    benchmark = BenchmarkName(args.benchmark)
    manifest = resolve_manifest(benchmark, args.runtime)
    output_dir = (
        _absolute(args.output_dir) if args.output_dir is not None
        else default_output_dir(benchmark, "self-test", now or datetime.now(), Path.cwd() / "roboscope-runs")
    )
    argv = [
        "--benchmark", benchmark.value, "--self-test",
        "--benchmark-manifest", str(manifest), "--output-dir", str(output_dir),
    ]
    return Invocation(argv, output_dir)


def command_check(args: argparse.Namespace) -> int:
    invocation = plan_check(args)
    exit_code = run_runtime(invocation)
    if exit_code == EXIT_OK:
        print(f"{PROG} check {args.benchmark}: passed; report in {invocation.output_dir / 'self-test.json'}")
    else:
        print(
            f"{PROG} check {args.benchmark}: failed with exit code {exit_code}; see {invocation.output_dir}",
            file=sys.stderr,
        )
    return exit_code


# ---------------------------------------------------------------------------------------------------------
# build


def install_dir_for(benchmark: BenchmarkName, given: Path | None) -> Path:
    if given is not None:
        return _absolute(given)
    home = roboscope_home()
    if home is None:
        raise UsageError(
            f"give the install directory (`{PROG} build {benchmark.value} <install-dir>`) or set "
            "ROBOSCOPE_HOME; runtimes take several GB to tens of GB and cannot be moved after the build"
        )
    return home / benchmark.value


def unsupported_build_message(benchmark: BenchmarkName, install_dir: str) -> str:
    return (
        f"`{PROG} build` does not build {benchmark.value} yet: its build script still reads files from a "
        "repository checkout. Build it from a clone:\n"
        f"  git clone {REPOSITORY_URL}\n"
        f"  KIT_PYTHON={sys.executable} bash RoboScope-Kit/benchmarks/{benchmark.value}/build-runtime.sh "
        f"{install_dir}"
    )


def check_disk(install_dir: Path, required_gb: int) -> None:
    """Refuse a first build where the install directory's file system lacks ``required_gb`` free."""

    if (install_dir / MANIFEST_NAME).is_file():
        return  # Rebuilding an existing runtime reuses most of what it downloaded.
    probe = install_dir
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < required_gb * 10**9:
        raise UsageError(
            f"{probe} has {free / 10**9:.1f} GB free; building this runtime needs about {required_gb} GB. "
            "Choose an install directory on a larger disk"
        )


def command_build(args: argparse.Namespace, run: Callable[..., Any] = subprocess.run) -> int:
    benchmark = BenchmarkName(args.benchmark)
    if benchmark not in BUILDABLE_BENCHMARKS:
        given = str(install_dir_for(benchmark, args.install_dir)) if args.install_dir else "<install-dir>"
        raise UsageError(unsupported_build_message(benchmark, given))
    install_dir = install_dir_for(benchmark, args.install_dir)
    script = build_script(benchmark)
    check_disk(install_dir, BUILDABLE_BENCHMARKS[benchmark])
    env = {**os.environ, "KIT_PYTHON": os.environ.get("KIT_PYTHON") or str(_absolute(sys.executable))}
    print(f"{PROG} build {benchmark.value}: running {script} {install_dir}", file=sys.stderr)
    result = run(["bash", str(script), str(install_dir)], env=env, check=False)
    if result.returncode != 0:
        print(
            f"{PROG} build {benchmark.value}: the build script exited with {result.returncode}",
            file=sys.stderr,
        )
        return EXIT_FAILED
    registry = register_runtime(benchmark, install_dir)
    print(f"{PROG} build {benchmark.value}: built {install_dir}; registered in {registry}")
    return EXIT_OK


# ---------------------------------------------------------------------------------------------------------
# Parser


def _benchmark_argument(parser: argparse.ArgumentParser, help_text: str, names: Sequence[str] = ()) -> None:
    choices = tuple(names) or tuple(item.value for item in BenchmarkName)
    parser.add_argument(
        "benchmark", choices=choices, metavar="BENCHMARK", help=f"{help_text}: {', '.join(choices)}",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog=PROG,
        description="Evaluate a Policy on RoboScope simulation benchmarks.",
        epilog=f"Documentation: {REPOSITORY_URL}/blob/main/packages/roboscope-kit/README.md",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    evaluate = commands.add_parser(
        "eval", help="evaluate a Policy on a benchmark suite",
        description="Evaluate a Policy on one benchmark suite. The runtime starts the Policy, drives the "
        "simulators, tunes concurrency and scores the results under --output-dir.",
    )
    _benchmark_argument(evaluate, "benchmark to evaluate")
    evaluate.add_argument("suite", metavar="SUITE", help="benchmark suite to evaluate")
    policy = evaluate.add_mutually_exclusive_group()
    policy.add_argument(
        "--policy-config", type=Path, metavar="JSON",
        help="Policy process contract (roboscope-policy.json)",
    )
    policy.add_argument(
        "--policy", type=Path, metavar="FILE",
        help="Policy script to start with --policy-python (default: this Python) in --policy-cwd "
        "(default: the script's directory); not for metaworld",
    )
    evaluate.add_argument("--policy-python", type=Path, metavar="PYTHON", help="interpreter for --policy")
    evaluate.add_argument("--policy-cwd", type=Path, metavar="DIR", help="working directory for --policy")
    evaluate.add_argument(
        "--checkpoint-source", type=Path, required=True, metavar="PATH",
        help="local checkpoint file or directory, passed to Policy.load()",
    )
    evaluate.add_argument(
        "--benchmark-manifest", type=Path, metavar="PATH",
        help="benchmark-manifest.json or the runtime directory holding it "
        "(default: the runtime `roboscope build` registered, then $ROBOSCOPE_HOME/<benchmark>)",
    )
    evaluate.add_argument(
        "--output-dir", type=Path, metavar="DIR",
        help="new or empty run directory (default: ./roboscope-runs/<benchmark>-<suite>-<time>)",
    )
    evaluate.add_argument(
        "--episodes", type=int, metavar="N", help="positive episode count overriding the manifest default",
    )
    evaluate.add_argument(
        "--quick-test", action="store_true",
        help="run a short real evaluation with one Worker and one Server",
    )
    evaluate.add_argument(
        "--eval-mode", choices=EVAL_MODES,
        help="strict (default, protocol-faithful) or fast (opt-in throughput optimizations)",
    )
    evaluate.add_argument(
        "--concurrency", type=int, metavar="K", help="evaluate with K Workers instead of autotuning; "
        "requires --replicas",
    )
    evaluate.add_argument(
        "--replicas", type=int, metavar="G", help="evaluate with G Policy replicas instead of autotuning; "
        "requires --concurrency",
    )
    evaluate.add_argument(
        "--autotune-strategy", metavar="STRATEGY", help="K/g autotune strategy overriding the manifest's",
    )
    evaluate.add_argument(
        "--run-cache", choices=CACHE_MODES, help="use (default), refresh or off: reuse this host's earlier "
        "K/g search and task times",
    )
    evaluate.add_argument("--cuda-mps", action="store_true", help="run Policy replicas under CUDA MPS")
    evaluate.add_argument(
        "--profile", action="store_true", help="capture each Worker's first episode with Nsight Systems; "
        "needs --quick-test or --concurrency and --replicas",
    )
    evaluate.add_argument(
        "--shard-index", type=int, metavar="INDEX", help="run only shard INDEX of a cross-instance run",
    )
    evaluate.add_argument(
        "--shard-count", type=int, metavar="COUNT", help="total shard count (default: from the workload)",
    )
    evaluate.add_argument(
        "--shard-episodes", type=int, metavar="EPISODES", help="episode budget per derived shard",
    )
    evaluate.add_argument(
        "--hosts", metavar="HOSTS", help="run one shard on each of these comma-separated registered hosts or "
        "SSH destinations and merge them; paths name files on the hosts (see `roboscope hosts`)",
    )
    evaluate.add_argument(
        "--detach", action="store_true", help="with --hosts: start the shards and return; run the same "
        "command with the same --output-dir to follow and merge them",
    )
    evaluate.add_argument(
        "--dry-run", action="store_true", help="check the arguments and print what would run, as JSON, "
        "without running it; with --hosts, check every host",
    )
    evaluate.add_argument("--json", action="store_true", help="print the final summary as one JSON line")
    evaluate.set_defaults(handler=command_eval)

    check = commands.add_parser(
        "check", help="check a built benchmark runtime without loading a Policy",
        description="Run the benchmark's runtime checks (its self-test) without loading a Policy.",
    )
    _benchmark_argument(check, "benchmark to check")
    check.add_argument(
        "runtime", type=Path, nargs="?", metavar="RUNTIME",
        help="runtime directory or its benchmark-manifest.json (default: the registered runtime, then "
        "$ROBOSCOPE_HOME/<benchmark>)",
    )
    check.add_argument(
        "--output-dir", type=Path, metavar="DIR",
        help="new or empty directory for the report (default: ./roboscope-runs/<benchmark>-self-test-<time>)",
    )
    check.set_defaults(handler=command_check)

    build = commands.add_parser(
        "build", help="build a benchmark's pinned simulator runtime",
        description="Build a benchmark's pinned simulator runtime with the build script shipped in this "
        "package, then register where it is installed. Supported: "
        + ", ".join(item.value for item in BUILDABLE_BENCHMARKS) + ".",
    )
    _benchmark_argument(build, "benchmark to build")
    build.add_argument(
        "install_dir", type=Path, nargs="?", metavar="INSTALL_DIR",
        help="directory to build into (default: $ROBOSCOPE_HOME/<benchmark>; one of the two is required)",
    )
    build.set_defaults(handler=command_build)

    merge = commands.add_parser(
        "merge", help="merge the shards of a cross-instance evaluation",
        description="Check that the shard runs of one evaluation cover its workload exactly once and merge "
        "their records into one result.json.",
    )
    _benchmark_argument(merge, "benchmark the shards evaluated")
    merge.add_argument("suite", metavar="SUITE", help="suite the shards evaluated")
    merge.add_argument(
        "--shard-dir", type=Path, action="append", required=True, metavar="DIR",
        help="output directory of one shard run; repeat for every shard",
    )
    merge.add_argument(
        "--benchmark-manifest", type=Path, metavar="PATH",
        help="benchmark-manifest.json the shards used, or its runtime directory (default: the runtime "
        "`roboscope build` registered, then $ROBOSCOPE_HOME/<benchmark>)",
    )
    merge.add_argument(
        "--output-dir", type=Path, metavar="DIR",
        help="new or empty directory for the merged result.json "
        "(default: ./roboscope-runs/<benchmark>-<suite>-merged-<time>)",
    )
    merge.add_argument(
        "--episodes", type=int, metavar="N", help="episode count the shard runs used, if not the default",
    )
    merge.set_defaults(handler=command_merge)

    hosts = commands.add_parser(
        "hosts", help="register the SSH hosts `eval --hosts` runs shards on",
        description="List the registered hosts with their kit version, GPUs, runtimes and free space; add or "
        "remove one. `roboscope eval ... --hosts a,b` runs one shard on each.",
    )
    hosts.add_argument("--json", action="store_true", help="print JSON")
    host_actions = hosts.add_subparsers(dest="hosts_action", metavar="ACTION")
    add = host_actions.add_parser(
        "add", help="set up key login to a host, check it and register it",
        description="Set up SSH key login to DESTINATION (asking its password once if needed), check the kit "
        "there and register it as NAME.",
    )
    add.add_argument("name", metavar="NAME", help="name to give the host in --hosts")
    add.add_argument(
        "destination", metavar="DESTINATION",
        help="SSH destination: an ~/.ssh/config host, user@host or ssh://user@host:port",
    )
    add.add_argument(
        "--roboscope", default="roboscope", metavar="COMMAND",
        help="the roboscope command on the host (default: roboscope on its PATH)",
    )
    add.add_argument(
        "--dir", default="roboscope-runs", metavar="DIR",
        help="directory on the host for shard runs; relative to its home (default: roboscope-runs)",
    )
    add.add_argument("--json", action="store_true", help="print JSON")
    remove = host_actions.add_parser("remove", help="remove a registered host")
    remove.add_argument("name", metavar="NAME", help="registered host name")
    probe = host_actions.add_parser(
        "probe", help="describe this machine as a host, as JSON",
        description="Print this machine's kit version, Python, GPUs, CPU quota, built runtimes and free "
        "space; `hosts add` and `hosts` run it on each host.",
    )
    probe.add_argument(
        "--dir", metavar="DIR", help="directory whose free space to report (default: roboscope-runs)",
    )
    probe.add_argument("--json", action="store_true", help="print one JSON line")
    hosts.set_defaults(handler=_command_hosts)
    return parser


def _command_hosts(args: argparse.Namespace) -> int:
    from roboscope_kit import hosts

    return hosts.command_hosts(args)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in arguments
    command = next((item for item in arguments if not item.startswith("-")), None)
    try:
        args = build_parser().parse_args(arguments)
    except _ParseError as exc:
        exc.parser.print_usage(sys.stderr)
        print(f"{exc.parser.prog}: error: {exc}", file=sys.stderr)
        return _usage_failure(command, str(exc), as_json)
    args.argv = arguments
    try:
        return args.handler(args)
    except UsageError as exc:
        print(f"{PROG} {args.command}: error: {exc}", file=sys.stderr)
        return _usage_failure(args.command, str(exc), as_json)


def _usage_failure(command: str | None, message: str, as_json: bool) -> int:
    if as_json:
        print(json.dumps({"command": command, "exit_code": EXIT_USAGE, "error": message}, sort_keys=True))
    return EXIT_USAGE


__all__ = ["main"]
