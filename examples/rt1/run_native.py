# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Run SimplerEnv's official RT-1-X evaluation scripts natively, with per-episode logs.

Usage::

    python examples/rt1/run_native.py --runtime /sim/simpler-env --checkpoint-source /models/rt1 \\
        --suite google_robot_visual_matching --output /runs/native-vm [--jobs 2]

1. **Plan.** The suite's official scripts from the runtime's SimplerEnv checkout are executed
   by bash with ``python`` on ``PATH`` replaced by a shim that only records each
   ``simpler_env/main_inference.py`` command line (``plan.jsonl``). The scripts sweep four
   RT-1 checkpoints; only the RT-1-X invocations are kept. The variant-aggregation scripts
   also run two alternative-camera variants that ``calc_metrics_evaluation_videos.py`` leaves
   out of its average; they are dropped too.
2. **Run.** Each kept invocation runs SimplerEnv's unmodified ``main_inference.py`` through
   ``native_hooks.py`` (read-only logging), from a native root that links the checkout's
   ``simpler_env`` and ``ManiSkill2_real2sim`` and a ``checkpoints/`` directory, so the
   scripts' relative paths resolve unchanged. ``--jobs`` invocations run at once. The
   Universal Sentence Encoder is served from a TF Hub cache entry that points at the
   verified local copy, so ``hub.load`` of upstream's tfhub.dev handle does not download.
   ``--tf-memory-mb`` is appended as ``--tf-memory-limit`` (upstream's default is 3072).

The output directory is bound to one run identity (``run-identity.json``: the suite, the
runtime, the SimplerEnv commit, the pinned checkpoint digests, the TensorFlow memory limit and
the digest of the planned invocations); a re-run with a different identity is refused, so use a
new ``--output`` for another suite or configuration. Within it, a finished invocation is
reused only when its ``.done`` names the same identity, the same command line and output
settings, and its episode JSONL is present, has the recorded SHA-256 and episode count, and
every record names the invocation's environment and policy configuration; otherwise it runs
again. An invocation that exits 0 without episode records is a failure. Runs with the
runtime's Python, from any Python.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from assets import RT1X, UNIVERSAL_SENTENCE_ENCODER, verify_directory  # noqa: E402

HERE = Path(__file__).resolve().parent
SCRIPTS = {
    "google_robot_visual_matching": (
        "rt1_pick_coke_can_visual_matching.sh",
        "rt1_move_near_visual_matching.sh",
        "rt1_drawer_visual_matching.sh",
        "rt1_put_in_drawer_visual_matching.sh",
    ),
    "google_robot_variant_aggregation": (
        "rt1_pick_coke_can_variant_agg.sh",
        "rt1_move_near_variant_agg.sh",
        "rt1_drawer_variant_agg.sh",
        "rt1_put_in_drawer_variant_agg.sh",
    ),
    "widowx_visual_matching": ("rt1x_bridge.sh",),
}
USE_HANDLE = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
EXCLUDED_ENV_MARKER = "AltGoogleCamera"
SHIM = """#!/usr/bin/env bash
# Records the command line instead of running it.
printf '%s\\0' "$@" >> "$ROBOSCOPE_NATIVE_PLAN"
printf '\\n' >> "$ROBOSCOPE_NATIVE_PLAN"
"""


def link(path: Path, target: Path) -> None:
    if path.is_symlink() or path.exists():
        if path.resolve() == target.resolve():
            return
        path.unlink()
    path.symlink_to(target)


def prepare_root(output: Path, source: Path, checkpoints: Path) -> Path:
    root = output / "root"
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    link(root / "simpler_env", source / "simpler_env")
    link(root / "ManiSkill2_real2sim", source / "ManiSkill2_real2sim")
    link(root / "scripts", source / "scripts")
    link(root / "checkpoints" / RT1X.directory, verify_directory(checkpoints, RT1X))
    hub_cache = output / "tfhub-cache"
    hub_cache.mkdir(exist_ok=True)
    link(
        hub_cache / hashlib.sha1(USE_HANDLE.encode("utf8")).hexdigest(),  # noqa: S324 -- TF Hub's cache key
        verify_directory(checkpoints, UNIVERSAL_SENTENCE_ENCODER),
    )
    return root


def plan(suite: str, root: Path, output: Path) -> list[dict]:
    """Every ``main_inference.py`` invocation of the suite's scripts, ``kept`` marking RT-1-X ones."""

    shim_dir = output / "plan-shim"
    shim_dir.mkdir(exist_ok=True)
    shim = shim_dir / "python"
    shim.write_text(SHIM, encoding="utf-8")
    shim.chmod(0o755)
    raw = output / "plan-raw.txt"
    raw.unlink(missing_ok=True)
    for script in SCRIPTS[suite]:
        subprocess.run(
            ("bash", f"scripts/{script}"),
            cwd=root,
            check=True,
            env={**os.environ, "PATH": f"{shim_dir}:{os.environ['PATH']}", "ROBOSCOPE_NATIVE_PLAN": str(raw)},
            stdout=subprocess.DEVNULL,
        )
        with raw.open("a", encoding="utf-8") as handle:
            handle.write(f"\0#script {script}\n")
    invocations: list[dict] = []
    for line in raw.read_text(encoding="utf-8").splitlines():
        fields = [field for field in line.split("\0") if field]
        if fields and fields[0].startswith("#script "):
            # The marker is written after a script's invocations; tag them.
            for entry in invocations:
                entry.setdefault("script", fields[0].removeprefix("#script "))
            continue
        if not fields or fields[0] != "simpler_env/main_inference.py":
            continue
        argv = fields[1:]
        ckpt = argv[argv.index("--ckpt-path") + 1].rstrip("/")
        env_name = argv[argv.index("--env-name") + 1]
        invocations.append(
            {
                "index": len(invocations),
                "argv": argv,
                "rt1x": Path(ckpt).name == RT1X.directory,
                "excluded_camera_variant": EXCLUDED_ENV_MARKER in env_name,
            }
        )
    for entry in invocations:
        entry["kept"] = entry["rt1x"] and not entry["excluded_camera_variant"]
    return invocations


def write_plan(output: Path, invocations: list[dict]) -> None:
    with (output / "plan.jsonl").open("w", encoding="utf-8") as handle:
        for entry in invocations:
            handle.write(json.dumps(entry) + "\n")


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf8")).hexdigest()


def run_identity(args: argparse.Namespace, source: Path, invocations: list[dict]) -> dict:
    """Everything that decides what an invocation index of this output directory means and runs."""

    commit = subprocess.run(
        ("git", "-C", str(source), "rev-parse", "HEAD"), capture_output=True, text=True, check=False
    )
    return {
        "format": 2,
        "suite": args.suite,
        "runtime": str(args.runtime),
        "simpler_env_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "checkpoints": {archive.directory: archive.files for archive in (RT1X, UNIVERSAL_SENTENCE_ENCODER)},
        "tf_memory_mb": args.tf_memory_mb,
        "plan_sha256": _digest([[entry["index"], entry["argv"], entry["kept"]] for entry in invocations]),
    }


def bind_output(output: Path, identity: dict) -> str:
    """Record ``identity`` for a fresh output directory, or refuse one recorded for another run."""

    path = output / "run-identity.json"
    if path.exists():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        if recorded != identity:
            changed = sorted(key for key in {*recorded, *identity} if recorded.get(key) != identity.get(key))
            raise SystemExit(
                f"{output} holds a native run with a different identity ({', '.join(changed)}); "
                "use a new --output directory"
            )
    else:
        if any((output / "episodes").iterdir()):
            raise SystemExit(f"{output}/episodes has records but no run-identity.json; use a new --output")
        path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _digest(identity)


def _flag(argv: list[str], name: str, default: str | None = None) -> str | None:
    return argv[argv.index(name) + 1] if name in argv else default


def check_records(lines: list[str], argv: list[str]) -> str | None:
    """Why the episode records do not belong to ``argv``'s invocation, or None when they do."""

    if not lines:
        return "no episode records"
    expected = {
        "env_name": _flag(argv, "--env-name"),
        "policy_model": _flag(argv, "--policy-model", "rt1"),
        "policy_setup": _flag(argv, "--policy-setup", "google_robot"),
        "obs_camera_name": _flag(argv, "--obs-camera-name"),
        "ckpt_path": _flag(argv, "--ckpt-path"),
    }
    for number, line in enumerate(lines, 1):
        record = json.loads(line)
        for field, value in expected.items():
            if field not in record:
                return f"record {number} has no {field}"
            if record[field] != value:
                return f"record {number} has {field}={record[field]!r}, expected {value!r}"
    return None


def reusable(done: Path, record: Path, expected: dict) -> str | None:
    """Why a finished invocation cannot be reused, or None when it can."""

    try:
        summary = json.loads(done.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"unreadable {done.name}: {exc}"
    for key, value in expected.items():
        if summary.get(key) != value:
            return f"{key} differs"
    if not record.is_file():
        return f"{record.name} is missing"
    data = record.read_bytes()
    if hashlib.sha256(data).hexdigest() != summary.get("record_sha256"):
        return f"{record.name} changed since the invocation finished"
    lines = data.decode("utf-8").splitlines()
    if len(lines) != summary.get("episodes"):
        return f"{record.name} has {len(lines)} records, expected {summary.get('episodes')}"
    return check_records(lines, expected["argv"])


def run_invocation(entry: dict, args: argparse.Namespace, root: Path, output: Path, identity: str) -> dict:
    episodes = output / "episodes"
    logs = output / "logs"
    name = f"{entry['index']:04d}"
    record = episodes / f"{name}.jsonl"
    done = episodes / f"{name}.done"
    argv = [
        *entry["argv"],
        "--tf-memory-limit",
        str(args.tf_memory_mb),
        "--logging-dir",
        str(output / "results"),
    ]
    expected = {
        "identity_sha256": identity,
        "index": entry["index"],
        "argv": argv,
        "write_videos": bool(args.write_videos),
    }
    if done.exists():
        reason = reusable(done, record, expected)
        if reason is None:
            return json.loads(done.read_text(encoding="utf-8"))
        print(f"invocation {name}: not reusing the finished run ({reason}); running it again", flush=True)
        done.unlink()
    record.unlink(missing_ok=True)
    command = [
        str(args.runtime / "env" / "bin" / "python"),
        str(HERE / "native_hooks.py"),
        "--record",
        str(record),
    ]
    if args.write_videos:
        command.append("--write-videos")
    command += ["--", "simpler_env/main_inference.py", *argv]
    started = time.time()
    with (logs / f"{name}.log").open("w", encoding="utf-8") as log:
        log.write(shlex.join(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=root,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": args.gpu,
                "TFHUB_CACHE_DIR": str(output / "tfhub-cache"),
                "TF_CPP_MIN_LOG_LEVEL": "2",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    data = record.read_bytes() if record.exists() else b""
    lines = data.decode("utf-8").splitlines()
    problem = check_records(lines, argv) if completed.returncode == 0 else None
    summary = {
        **expected,
        "script": entry.get("script"),
        "returncode": completed.returncode,
        "episodes": len(lines),
        "record_sha256": hashlib.sha256(data).hexdigest(),
        "started_unix": started,
        "elapsed_s": time.time() - started,
    }
    if problem is not None:
        summary["error"] = problem
        print(f"invocation {name}: {problem}", flush=True)
    if completed.returncode == 0 and problem is None:
        done.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runtime", type=Path, required=True, help="SimplerEnv runtime install dir")
    parser.add_argument("--checkpoint-source", type=Path, required=True, help="download_assets.py output")
    parser.add_argument("--suite", choices=sorted(SCRIPTS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--tf-memory-mb", type=int, default=3072)
    parser.add_argument("--write-videos", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--invocation", type=int, action="append", help="run only these plan indexes (repeatable)"
    )
    parser.add_argument(
        "--script", action="append", help="run only this official script's invocations (repeatable)"
    )
    args = parser.parse_args()
    args.runtime = args.runtime.resolve()
    output = args.output.resolve()
    for directory in (output, output / "episodes", output / "logs"):
        directory.mkdir(parents=True, exist_ok=True)
    source = args.runtime / "src" / "SimplerEnv"
    root = prepare_root(output, source, args.checkpoint_source.resolve())
    invocations = plan(args.suite, root, output)
    identity = bind_output(output, run_identity(args, source, invocations))
    write_plan(output, invocations)
    kept = [entry for entry in invocations if entry["kept"]]
    print(f"{len(kept)} RT-1-X invocations planned from {', '.join(SCRIPTS[args.suite])}", flush=True)
    if args.plan_only:
        return 0
    if args.invocation:
        kept = [entry for entry in kept if entry["index"] in set(args.invocation)]
    if args.script:
        kept = [entry for entry in kept if entry.get("script") in set(args.script)]
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        summaries = list(pool.map(lambda entry: run_invocation(entry, args, root, output, identity), kept))
    failed = [summary for summary in summaries if summary["returncode"] != 0 or "error" in summary]
    report = {
        "suite": args.suite,
        "invocations": len(kept),
        "failed_invocations": [summary["index"] for summary in failed],
        "episodes": sum(summary["episodes"] for summary in summaries),
        "jobs": args.jobs,
        "wall_s": time.time() - started,
        "invocation_elapsed_s": sum(summary["elapsed_s"] for summary in summaries),
    }
    (output / "native-run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
