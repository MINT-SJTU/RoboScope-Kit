# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Registered SSH hosts (``roboscope hosts``) and multi-host evaluation (``roboscope eval ... --hosts``).

``roboscope hosts add NAME DESTINATION`` sets up key login to an SSH destination, checks the kit there and
records the host in ``$XDG_CONFIG_HOME/roboscope-kit/hosts.json``.

``roboscope eval ... --hosts a,b,c`` runs shard *i* of ``len(hosts)`` on host *i* with the same ``roboscope
eval`` arguments, detached from the SSH session. This process checks every host first with ``roboscope eval
--dry-run``, starts the shards, polls them, and once every shard has succeeded copies the records the merge
reads onto the first host, runs ``roboscope merge`` there and copies the merged ``result.json`` into the
local output directory. The benchmark runtime and the shard merge run unchanged on the hosts.

The local output directory keeps ``hosts.json``: the plan and each shard's host, remote directory and
attempts. Running the same command with the same ``--output-dir`` again follows the shards that are still
running, starts a new attempt for each shard that failed, and merges once all have succeeded.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from roboscope_kit import cli

STATE_NAME = "hosts.json"
STATE_SCHEMA_VERSION = 1
REGISTRY_SCHEMA_VERSION = 1
DEFAULT_ROBOSCOPE = "roboscope"
DEFAULT_RUNS_DIR = "roboscope-runs"
SSH_COMMAND: tuple[str, ...] = (
    "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
    "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6",
)
POLL_SECONDS = 30.0
REPORT_SECONDS = 300.0
STOP_GRACE_SECONDS = 120.0
CHECK_TIMEOUT_S = 600.0
COMMAND_TIMEOUT_S = 120.0
TRANSFER_TIMEOUT_S = 4 * 3600.0
HOST_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,62}")
# What `roboscope merge` reads from a shard directory; evaluation-config.json exists for MetaWorld only.
MERGE_INPUTS = ("manifest.json", "result.json", "evaluation-config.json", "queue/completed")

# eval options every host receives unchanged, as (flag, attribute).
FORWARDED_VALUES: tuple[tuple[str, str], ...] = (
    ("--policy-config", "policy_config"),
    ("--policy", "policy"),
    ("--policy-python", "policy_python"),
    ("--policy-cwd", "policy_cwd"),
    ("--checkpoint-source", "checkpoint_source"),
    ("--benchmark-manifest", "benchmark_manifest"),
    ("--episodes", "episodes"),
    ("--eval-mode", "eval_mode"),
    ("--concurrency", "concurrency"),
    ("--replicas", "replicas"),
    ("--autotune-strategy", "autotune_strategy"),
    ("--run-cache", "run_cache"),
)
FORWARDED_FLAGS: tuple[tuple[str, str], ...] = (("--cuda-mps", "cuda_mps"), ("--profile", "profile"))
# eval options this command sets per host or keeps for itself.
CONTROLLER_OPTIONS = frozenset({
    "benchmark", "suite", "output_dir", "json", "hosts", "detach", "dry_run",
    "shard_index", "shard_count", "shard_episodes", "quick_test",
})

# Runs on a host with the Python its `roboscope` uses; reports each attempt of that host as JSON.
STATUS_SCRIPT = r"""
import json, os, sys
root, names = sys.argv[1], json.loads(sys.argv[2])
report = {}
for name in names:
    state = {"exit": None, "alive": False, "done": None, "total": None}
    base = os.path.join(root, name)
    try:
        with open(base + ".exit") as handle:
            state["exit"] = int(handle.read().strip())
    except (OSError, ValueError):
        try:
            with open(base + ".pid") as handle:
                os.kill(int(handle.read().strip()), 0)
            state["alive"] = True
        except PermissionError:
            state["alive"] = True
        except (OSError, ValueError):
            pass
    try:
        with open(os.path.join(base, "manifest.json")) as handle:
            state["total"] = len(json.load(handle)["items"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        state["done"] = sum(
            1 for entry in os.listdir(os.path.join(base, "queue", "completed"))
            if entry.startswith("item-") and entry.endswith(".json")
        )
    except OSError:
        pass
    report[name] = state
print(json.dumps(report))
"""


class HostError(Exception):
    """A host could not be reached or refused a command."""


@dataclass(frozen=True)
class Host:
    """Runs POSIX shell scripts on one SSH destination, without a password prompt."""

    destination: str

    def argv(self, script: str) -> list[str]:
        return [*SSH_COMMAND, self.destination, "sh -c " + shlex.quote(script)]

    def run(self, script: str, timeout: float = COMMAND_TIMEOUT_S) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                self.argv(script), stdin=subprocess.DEVNULL, capture_output=True, text=True,
                timeout=timeout, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise HostError(f"{self.destination}: no answer within {timeout:.0f} s") from exc

    def check(self, script: str, timeout: float = COMMAND_TIMEOUT_S) -> str:
        """Run ``script`` and return its output; raise ``HostError`` unless it exits 0."""

        result = self.run(script, timeout)
        if result.returncode != 0:
            raise HostError(describe_failure(self.destination, result))
        return result.stdout


def describe_failure(destination: str, result: subprocess.CompletedProcess[str]) -> str:
    detail = (result.stderr or result.stdout).strip().splitlines()
    last = detail[-1] if detail else "no output"
    if result.returncode == 255:
        return (
            f"{destination}: ssh failed ({last}). Hosts need SSH without a password prompt; "
            f"`{cli.PROG} hosts add` sets that up"
        )
    return f"{destination}: exited {result.returncode}: {last}"


def _q(value: object) -> str:
    return shlex.quote(str(value))


def _command_text(command: Sequence[str]) -> str:
    return " ".join(_q(part) for part in command)


def _last_json(text: str) -> Any:
    for line in reversed(text.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


# ---------------------------------------------------------------------------------------------------------
# Registry


def hosts_registry_path() -> Path:
    return cli.registry_path().with_name("hosts.json")


def registered_hosts() -> dict[str, dict[str, Any]]:
    path = hosts_registry_path()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        raise cli.UsageError(f"cannot read the host registry {path}: {exc}") from exc
    hosts = value.get("hosts") if isinstance(value, dict) else None
    if not isinstance(hosts, dict):
        raise cli.UsageError(f"the host registry {path} has no hosts object")
    return hosts


def write_registry(hosts: dict[str, dict[str, Any]]) -> Path:
    path = hosts_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{os.getpid()}")
    staging.write_text(
        json.dumps(
            {"schema_version": REGISTRY_SCHEMA_VERSION, "hosts": dict(sorted(hosts.items()))}, indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    os.replace(staging, path)
    return path


def host_entry(name: str, destination: str, roboscope: str, runs_dir: str) -> dict[str, Any]:
    if not destination or destination.startswith("-") or any(char.isspace() for char in destination):
        raise cli.UsageError(f"{destination!r} is not an SSH destination")
    command = shlex.split(roboscope)
    if not command:
        raise cli.UsageError("--roboscope is empty")
    if runs_dir in {"", "~"}:
        raise cli.UsageError("--dir must name a directory, not the remote home itself")
    return {
        "name": name,
        "destination": destination,
        "roboscope": command,
        "dir": runs_dir[2:] if runs_dir.startswith("~/") else runs_dir,
    }


def registered_entry(name: str, value: dict[str, Any]) -> dict[str, Any]:
    return host_entry(name, value["destination"], _command_text(value["roboscope"]), value["dir"])


def resolve_hosts(value: str) -> list[dict[str, Any]]:
    """``--hosts``: registered names, else SSH destinations with the default settings."""

    items = [item.strip() for item in value.split(",")]
    if not items or any(not item for item in items):
        raise cli.UsageError(
            "--hosts takes host names or SSH destinations separated by commas, such as gpu1,gpu2"
        )
    registry = registered_hosts()
    resolved = []
    for item in items:
        if item in registry:
            entry = registry[item]
            resolved.append(registered_entry(item, entry))
        else:
            resolved.append(host_entry(item, item, DEFAULT_ROBOSCOPE, DEFAULT_RUNS_DIR))
    return resolved


def copy_id_argv(destination: str) -> list[str]:
    """``ssh-copy-id`` for ``destination``; it takes the port as ``-p``, not inside an ``ssh://`` URI."""

    match = re.fullmatch(r"ssh://(?P<target>[^/:]+(?:@[^/:]+)?)(?::(?P<port>\d+))?/?", destination)
    if match is None:
        return ["ssh-copy-id", destination]
    return ["ssh-copy-id", *(["-p", match["port"]] if match["port"] else []), match["target"]]


def ensure_key_login(destination: str, interactive: bool) -> bool:
    """Log in without a password, installing this machine's public key first if needed.

    Returns whether a key was installed.
    """

    result = Host(destination).run("true", timeout=60)
    if result.returncode == 0:
        return False
    if result.returncode != 255:
        raise cli.UsageError(describe_failure(destination, result))
    if not interactive:
        raise cli.UsageError(
            f"{destination}: SSH needs a password and there is no terminal to type it in; run "
            f"`{cli.PROG} hosts add` from a terminal or install a key with ssh-copy-id"
        )
    if shutil.which("ssh-copy-id") is None:
        raise cli.UsageError("ssh-copy-id was not found; install OpenSSH's client tools")
    keys = Path.home() / ".ssh"
    if not any((keys / f"{name}.pub").is_file() for name in ("id_ed25519", "id_ecdsa", "id_rsa")):
        keys.mkdir(mode=0o700, exist_ok=True)
        print(f"{cli.PROG} hosts: creating the SSH key {keys / 'id_ed25519'}", file=sys.stderr)
        subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(keys / "id_ed25519")], check=True,
        )
    print(f"{cli.PROG} hosts: installing this machine's SSH key on {destination}; type its password once",
          file=sys.stderr)
    if subprocess.run(copy_id_argv(destination), check=False).returncode != 0:
        raise cli.UsageError(f"{destination}: ssh-copy-id failed")
    again = Host(destination).run("true", timeout=60)
    if again.returncode != 0:
        raise cli.UsageError(describe_failure(destination, again))
    return True


def probe_report(runs_dir: str) -> dict[str, Any]:
    """This machine as a host: kit, Python, GPUs, CPU quota, built runtimes and free space for runs."""

    from roboscope_kit.runtime.benchmark.resources import cpu_quota

    target = cli._absolute(runs_dir)
    existing = target
    while not existing.exists():
        existing = existing.parent
    try:
        runtimes = cli.registered_runtimes()
    except cli.UsageError:
        runtimes = {}
    return {
        "hostname": socket.gethostname(),
        "kit_version": cli.kit_version(),
        "python": str(cli._absolute(sys.executable)),
        "gpus": cli.gpu_names(),
        "cpus": cpu_quota(),
        "runtimes": runtimes,
        "dir": str(target),
        "disk_free_bytes": shutil.disk_usage(existing).free,
    }


def probe(entry: dict[str, Any]) -> dict[str, Any]:
    """``roboscope hosts probe`` on the host ``entry`` names."""

    command = [*entry["roboscope"], "hosts", "probe", "--dir", entry["dir"], "--json"]
    result = Host(entry["destination"]).run("exec " + _command_text(command), timeout=COMMAND_TIMEOUT_S)
    report = _last_json(result.stdout)
    if result.returncode == 127:
        raise HostError(
            f"{entry['destination']}: {_command_text(entry['roboscope'])} was not found; install "
            "roboscope-kit there, or register the host with --roboscope <path to its roboscope>"
        )
    if result.returncode != 0 or not isinstance(report, dict):
        raise HostError(describe_failure(entry["destination"], result))
    return report


def describe_probe(name: str, entry: dict[str, Any], report: dict[str, Any] | HostError) -> str:
    if isinstance(report, HostError):
        return f"{name}  {entry['destination']}  unreachable: {report}"
    local = cli.kit_version()
    kit = report.get("kit_version")
    kit_text = f"kit {kit}" + ("" if kit == local else f" (this machine has {local})")
    gpus = report.get("gpus") or []
    gpu_text = f"{len(gpus)} GPU" + ("" if len(gpus) == 1 else "s") + (f" ({gpus[0]})" if gpus else "")
    runtimes = ", ".join(sorted(report.get("runtimes") or {})) or "none"
    free = report.get("disk_free_bytes")
    free_text = f"{free / 10**9:.0f} GB free in {report.get('dir')}" if isinstance(free, int) else ""
    return (
        f"{name}  {entry['destination']}  {kit_text}, {gpu_text}, {report.get('cpus', 0):g} CPUs, "
        f"runtimes: {runtimes}; {free_text}"
    )


def command_hosts(args: argparse.Namespace) -> int:
    action = args.hosts_action or "list"
    if action == "probe":
        report = probe_report(args.dir or DEFAULT_RUNS_DIR)
        print(json.dumps(report, sort_keys=True, indent=None if args.json else 2))
        return cli.EXIT_OK
    if action == "remove":
        hosts = registered_hosts()
        if args.name not in hosts:
            raise cli.UsageError(f"no host named {args.name!r} is registered")
        del hosts[args.name]
        write_registry(hosts)
        print(f"{cli.PROG} hosts: removed {args.name}")
        return cli.EXIT_OK
    if action == "add":
        if HOST_NAME.fullmatch(args.name) is None:
            raise cli.UsageError(f"{args.name!r} is not a host name: use letters, digits, '.', '-' and '_'")
        entry = host_entry(args.name, args.destination, args.roboscope, args.dir)
        ensure_key_login(entry["destination"], interactive=sys.stdin.isatty())
        try:
            report = probe(entry)
        except HostError as exc:
            raise cli.UsageError(str(exc)) from exc
        hosts = registered_hosts()
        hosts[args.name] = {key: entry[key] for key in ("destination", "roboscope", "dir")}
        registry = write_registry(hosts)
        if args.json:
            print(json.dumps({"name": args.name, **hosts[args.name], "probe": report}, sort_keys=True))
        else:
            print(describe_probe(args.name, entry, report))
            print(f"{cli.PROG} hosts: registered {args.name} in {registry}")
        return cli.EXIT_OK
    hosts = registered_hosts()
    entries = {name: registered_entry(name, value) for name, value in hosts.items()}

    def one(entry: dict[str, Any]) -> dict[str, Any] | HostError:
        try:
            return probe(entry)
        except HostError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=min(16, len(entries)) or 1) as pool:
        reports = dict(zip(entries, pool.map(one, entries.values()), strict=True))
    if args.json:
        print(json.dumps({
            name: {**hosts[name], "probe": None if isinstance(report, HostError) else report,
                   **({"error": str(report)} if isinstance(report, HostError) else {})}
            for name, report in reports.items()
        }, sort_keys=True))
    elif not entries:
        print(f"no hosts registered; add one with `{cli.PROG} hosts add NAME DESTINATION`")
    else:
        for name, report in reports.items():
            print(describe_probe(name, entries[name], report))
    return cli.EXIT_OK


# ---------------------------------------------------------------------------------------------------------
# Plan


def forwarded_options(args: argparse.Namespace) -> list[str]:
    """The eval options every host receives, as given here: paths name files on the hosts."""

    argv: list[str] = []
    for flag, attribute in FORWARDED_VALUES:
        value = getattr(args, attribute)
        if value is not None:
            argv += [flag, str(value)]
    for flag, attribute in FORWARDED_FLAGS:
        if getattr(args, attribute):
            argv.append(flag)
    return argv


def make_plan(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    for flag, attribute in (
        ("--shard-index", "shard_index"),
        ("--shard-count", "shard_count"),
        ("--shard-episodes", "shard_episodes"),
    ):
        if getattr(args, attribute) is not None:
            raise cli.UsageError(f"--hosts assigns one shard to each host; drop {flag}")
    if args.quick_test:
        raise cli.UsageError("a quick test cannot be sharded; run it without --hosts")
    cli.validate_suite(cli.BenchmarkName(args.benchmark), args.suite)
    return {
        "benchmark": args.benchmark,
        "suite": args.suite,
        "options": forwarded_options(args),
        "hosts": resolve_hosts(args.hosts),
        "run_name": output_dir.name,
    }


def attempt_name(index: int, attempt: int) -> str:
    return f"shard-{index}" if attempt == 0 else f"shard-{index}-retry{attempt}"


def eval_command(plan: dict[str, Any], index: int, output_dir: str, *extra: str) -> list[str]:
    return [
        *plan["hosts"][index]["roboscope"], "eval", plan["benchmark"], plan["suite"], *plan["options"],
        "--shard-index", str(index), "--shard-count", str(len(plan["hosts"])),
        "--output-dir", output_dir, *extra,
    ]


# ---------------------------------------------------------------------------------------------------------
# Preflight


def dry_run(plan: dict[str, Any], index: int, name: str, remote_dir: str) -> dict[str, Any]:
    """``roboscope eval --dry-run`` on shard ``index``'s host: its versions, files and output directory."""

    entry = plan["hosts"][index]
    host = Host(entry["destination"])
    command = eval_command(plan, index, f"{remote_dir}/{name}", "--dry-run", "--json")
    result = host.run("exec " + _command_text(command), CHECK_TIMEOUT_S)
    report = _last_json(result.stdout)
    if result.returncode == 127:
        raise HostError(
            f"{entry['name']}: {_command_text(entry['roboscope'])} was not found; install roboscope-kit "
            "there, or register the host with --roboscope <path to its roboscope>"
        )
    if result.returncode != 0 or not isinstance(report, dict):
        if isinstance(report, dict) and report.get("error"):
            raise HostError(f"{entry['name']}: {report['error']}")
        raise HostError(describe_failure(entry["name"], result))
    return report


def check_reports(names: Sequence[str], reports: Sequence[dict[str, Any]]) -> list[str]:
    """Problems that keep these hosts from running one evaluation together."""

    problems = []
    local = cli.kit_version()
    for name, report in zip(names, reports, strict=True):
        if report.get("kit_version") != local:
            problems.append(f"{name}: roboscope-kit {report.get('kit_version')}, this machine has {local}")
        if not report.get("gpus"):
            problems.append(f"{name}: no NVIDIA GPU found")
        checkpoint = report.get("checkpoint") or {}
        if not checkpoint.get("exists"):
            problems.append(f"{name}: checkpoint not found: {checkpoint.get('path')}")
        if report.get("output_dir_in_use"):
            problems.append(f"{name}: {report.get('output_dir')} already holds run artifacts")
    manifests = [report.get("manifest_sha256") for report in reports]
    checkpoints = [(report.get("checkpoint") or {}).get("digest") for report in reports]
    for label, values in (("benchmark manifests", manifests), ("checkpoint files", checkpoints)):
        if len(set(values)) > 1:
            pairs = zip(names, values, strict=True)
            listing = ", ".join(f"{name} {str(value)[:12]}" for name, value in pairs)
            problems.append(f"the hosts have different {label}: {listing}")
    return problems


def preflight(plan: dict[str, Any], targets: Sequence[tuple[int, str, str]]) -> dict[int, dict[str, Any]]:
    """Dry-run every ``(index, attempt name, remote dir)`` in parallel and refuse inconsistent hosts."""

    def one(target: tuple[int, str, str]) -> dict[str, Any] | HostError:
        try:
            return dry_run(plan, *target)
        except HostError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=min(16, len(targets)) or 1) as pool:
        outcomes = list(pool.map(one, targets))
    errors = [str(item) for item in outcomes if isinstance(item, HostError)]
    if errors:
        raise cli.UsageError("cannot start on every host:\n  " + "\n  ".join(errors))
    reports = {target[0]: item for target, item in zip(targets, outcomes, strict=True)}
    names = [plan["hosts"][index]["name"] for index, _, _ in targets]
    problems = check_reports(names, [reports[index] for index, _, _ in targets])
    if problems:
        raise cli.UsageError("the hosts cannot run this evaluation together:\n  " + "\n  ".join(problems))
    return reports


# ---------------------------------------------------------------------------------------------------------
# Remote shards


def launch_script(remote_dir: str, name: str, command: Sequence[str]) -> str:
    """Start ``command`` in a new session under ``remote_dir``; print its PID.

    ``<name>.out`` and ``<name>.log`` take its output, ``<name>.pid`` its PID and ``<name>.exit`` its exit
    status once it ends; an attempt name is never reused.
    """

    runner = '"$@" > "$0.out" 2> "$0.log" < /dev/null & echo $! > "$0.pid"; wait $!; echo $? > "$0.exit"'
    pid_file = _q(f"{name}.pid")
    return (
        f"mkdir -p {_q(remote_dir)} && cd {_q(remote_dir)} || exit 3\n"
        f"if [ -e {pid_file} ]; then echo 'attempt {name} was already started' >&2; exit 3; fi\n"
        f"setsid sh -c {_q(runner)} {_q(name)} {_command_text(command)} < /dev/null > /dev/null 2>&1 &\n"
        f"i=0; while [ ! -s {pid_file} ] && [ $i -lt 100 ]; do sleep 0.1; i=$((i + 1)); done\n"
        f"cat {pid_file}"
    )


def start_shard(plan: dict[str, Any], shard: dict[str, Any]) -> None:
    name = shard["attempts"][-1]
    command = eval_command(plan, shard["index"], f"{shard['remote_dir']}/{name}", "--json")
    Host(shard["destination"]).check(launch_script(shard["remote_dir"], name, command))
    shard["state"] = "running"


def status_script(python: str, remote_dir: str, names: Sequence[str]) -> str:
    return _command_text((python, "-I", "-c", STATUS_SCRIPT, remote_dir, json.dumps(list(names))))


def poll_destination(destination: str, shards: Sequence[dict[str, Any]]) -> dict[int, dict[str, Any]] | None:
    """Each shard's current attempt on one destination, or ``None`` when it cannot be reached."""

    report: dict[int, dict[str, Any]] = {}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for shard in shards:
        groups.setdefault((shard["python"], shard["remote_dir"]), []).append(shard)
    try:
        for (python, remote_dir), members in groups.items():
            names = [shard["attempts"][-1] for shard in members]
            states = _last_json(Host(destination).check(status_script(python, remote_dir, names)))
            for shard, name in zip(members, names, strict=True):
                report[shard["index"]] = states[name]
    except (HostError, KeyError, TypeError):
        return None
    return report


def apply_status(shard: dict[str, Any], status: dict[str, Any] | None) -> str:
    """Update a running ``shard`` from one poll; return how to show it."""

    if shard["state"] == "running" and status is not None:
        if status["exit"] is not None:
            shard["state"] = "succeeded" if status["exit"] == 0 else "failed"
            shard["exit"] = status["exit"]
        elif not status["alive"]:
            shard["state"] = "lost"
    if shard["state"] == "succeeded":
        return "succeeded"
    if shard["state"] == "failed":
        return f"failed (exit {shard['exit']})"
    if shard["state"] == "lost":
        return "lost (its process ended without an exit status)"
    if status is None:
        return "unreachable"
    if status["total"] is None:
        return "starting"
    return f"{status['done'] or 0}/{status['total']} items"


def poll(shards: Sequence[dict[str, Any]]) -> dict[int, dict[str, Any] | None]:
    """Poll every running shard, one SSH call per destination, in parallel."""

    by_destination: dict[str, list[dict[str, Any]]] = {}
    for shard in shards:
        if shard["state"] == "running":
            by_destination.setdefault(shard["destination"], []).append(shard)
    with ThreadPoolExecutor(max_workers=min(16, len(by_destination)) or 1) as pool:
        polled = list(pool.map(lambda item: poll_destination(*item), by_destination.items()))
    statuses: dict[int, dict[str, Any] | None] = {}
    for (_, members), report in zip(by_destination.items(), polled, strict=True):
        for shard in members:
            statuses[shard["index"]] = None if report is None else report.get(shard["index"])
    return statuses


def stop_shards(shards: Sequence[dict[str, Any]]) -> None:
    """Send SIGTERM to every running attempt; the runtime then stops its Workers and Policy Servers."""

    for shard in shards:
        if shard["state"] != "running":
            continue
        name = shard["attempts"][-1]
        with suppress(HostError):
            Host(shard["destination"]).run(
                f"cd {_q(shard['remote_dir'])} && [ ! -e {_q(name + '.exit')} ] && "
                f"kill -TERM \"$(cat {_q(name + '.pid')})\""
            )


def shard_error(shard: dict[str, Any]) -> str | None:
    """The last error the attempt reported: its ``--json`` summary, else the end of its log."""

    name = shard["attempts"][-1]
    try:
        result = Host(shard["destination"]).run(
            f"cd {_q(shard['remote_dir'])} && tail -n 1 {_q(name + '.out')}; tail -n 3 {_q(name + '.log')}"
        )
    except HostError:
        return None
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("error"):
            return str(value["error"])
    return lines[-1] if lines else None


# ---------------------------------------------------------------------------------------------------------
# Merge


def pack_script(remote_dir: str, name: str) -> str:
    inputs = " ".join(_q(f"{name}/{item}") for item in MERGE_INPUTS)
    return (
        f"cd {_q(remote_dir)} || exit 3\nset --\n"
        f"for f in {inputs}; do if [ -e \"$f\" ]; then set -- \"$@\" \"$f\"; fi; done\n"
        "exec tar czf - \"$@\""
    )


def unpack_script(root: str, name: str) -> str:
    return f"rm -rf -- {_q(f'{root}/{name}')} && mkdir -p {_q(root)} && exec tar xzf - -C {_q(root)}"


def copy_shard(source: dict[str, Any], target_destination: str, target_root: str) -> str:
    """Stream the merge inputs of ``source``'s attempt through this machine to ``target_root``."""

    name = source["attempts"][-1]
    sender = subprocess.Popen(
        Host(source["destination"]).argv(pack_script(source["remote_dir"], name)),
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert sender.stdout is not None
    receiver = subprocess.Popen(
        Host(target_destination).argv(unpack_script(target_root, name)),
        stdin=sender.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    sender.stdout.close()
    try:
        _, receive_error = receiver.communicate(timeout=TRANSFER_TIMEOUT_S)
        _, send_error = sender.communicate(timeout=60)
    except subprocess.TimeoutExpired as exc:
        sender.kill()
        receiver.kill()
        raise HostError(
            f"copying shard {source['index']} did not finish in {TRANSFER_TIMEOUT_S:.0f} s"
        ) from exc
    if sender.returncode != 0 or receiver.returncode != 0:
        detail = (send_error or receive_error or b"").decode(errors="replace").strip().splitlines()
        raise HostError(
            f"copying shard {source['index']} from {source['host']} failed"
            + (f": {detail[-1]}" if detail else "")
        )
    return f"{target_root}/{name}"


def merge_on_first_host(plan: dict[str, Any], state: dict[str, Any]) -> str:
    """Collect every shard's records on the first host, run the shard merge there, return its result."""

    first = state["shards"][0]
    merged_dir = f"{first['remote_dir']}/merged"
    host = Host(first["destination"])
    existing = host.run(f"cat {_q(merged_dir + '/result.json')}")
    if existing.returncode == 0 and existing.stdout.strip():
        return existing.stdout
    inputs_root = f"{first['remote_dir']}/merge-inputs"
    shard_dirs = []
    for shard in state["shards"]:
        if shard["destination"] == first["destination"] and shard["remote_dir"] == first["remote_dir"]:
            shard_dirs.append(f"{shard['remote_dir']}/{shard['attempts'][-1]}")
        else:
            _log(f"copying shard {shard['index']}'s records from {shard['host']} to {first['host']}")
            shard_dirs.append(copy_shard(shard, first["destination"], inputs_root))
    command = [*plan["hosts"][0]["roboscope"], "merge", plan["benchmark"], plan["suite"]]
    for directory in shard_dirs:
        command += ["--shard-dir", directory]
    options = plan["options"]
    for flag in ("--benchmark-manifest", "--episodes"):
        if flag in options:
            command += [flag, options[options.index(flag) + 1]]
    command += ["--output-dir", merged_dir]
    _log(f"merging the shards on {first['host']}")
    host.check(f"cd {_q(first['remote_dir'])} && exec {_command_text(command)}", TRANSFER_TIMEOUT_S)
    return host.check(f"cat {_q(merged_dir + '/result.json')}")


# ---------------------------------------------------------------------------------------------------------
# State


def write_state(output_dir: Path, state: dict[str, Any]) -> None:
    path = output_dir / STATE_NAME
    staging = path.with_name(f".{path.name}.{os.getpid()}")
    staging.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(staging, path)


def load_state(output_dir: Path, plan: dict[str, Any]) -> dict[str, Any] | None:
    """The state of an earlier run of this plan in ``output_dir``, or ``None`` for a new directory."""

    path = output_dir / STATE_NAME
    if not path.is_file():
        if output_dir.exists() and any(entry.name != ".hosts.lock" for entry in output_dir.iterdir()):
            raise cli.UsageError(
                f"{output_dir} is not empty and holds no {STATE_NAME}; choose a new --output-dir"
            )
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise cli.UsageError(f"cannot read {path}: {exc}") from exc
    if state.get("schema_version") != STATE_SCHEMA_VERSION or state.get("plan") != plan:
        raise cli.UsageError(
            f"{output_dir} belongs to a different multi-host run; give the same benchmark, suite, options "
            "and --hosts to follow it, or choose a new --output-dir"
        )
    return state


@contextmanager
def exclusive(output_dir: Path) -> Iterator[None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / ".hosts.lock", "a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise cli.UsageError(f"another {cli.PROG} eval is already following {output_dir}") from exc
        yield


# ---------------------------------------------------------------------------------------------------------
# Command


class _Stop(Exception):
    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


class _Detach(Exception):
    pass


@contextmanager
def _signals() -> Iterator[None]:
    """Ctrl-C and SIGTERM stop the shards; SIGHUP (the terminal went away) leaves them running."""

    def stop(signum, frame):
        raise _Stop(signum)

    def detach(signum, frame):
        raise _Detach()

    previous = {
        signal.SIGINT: signal.signal(signal.SIGINT, stop),
        signal.SIGTERM: signal.signal(signal.SIGTERM, stop),
        signal.SIGHUP: signal.signal(signal.SIGHUP, detach),
    }
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def follow_command(args: argparse.Namespace, output_dir: Path) -> str:
    """The command that follows this run again: the original arguments with this ``--output-dir``."""

    argv = [item for item in getattr(args, "argv", None) or [] if item != "--detach"]
    if args.output_dir is None:
        argv += ["--output-dir", str(output_dir)]
    return _command_text([cli.PROG, *argv])


def _log(message: str) -> None:
    print(f"{datetime.now().strftime('%H:%M:%S')} {cli.PROG} eval: {message}", file=sys.stderr, flush=True)


def wait_for_shards(state: dict[str, Any], output_dir: Path, sleep=time.sleep) -> None:
    last_line = None
    last_report = 0.0
    while True:
        statuses = poll(state["shards"])
        parts = [
            f"shard {shard['index']} {shard['host']}: {apply_status(shard, statuses.get(shard['index']))}"
            for shard in state["shards"]
        ]
        write_state(output_dir, state)
        line = " | ".join(parts)
        if line != last_line or time.monotonic() - last_report >= REPORT_SECONDS:
            _log(line)
            last_line, last_report = line, time.monotonic()
        if all(shard["state"] != "running" for shard in state["shards"]):
            return
        sleep(POLL_SECONDS)


def summary(plan: dict[str, Any], state: dict[str, Any], output_dir: Path, exit_code: int) -> dict[str, Any]:
    wall_s = time.time() - state["started_at"]
    value = cli.eval_summary(plan["benchmark"], plan["suite"], output_dir, exit_code, wall_s)
    value.pop("error", None)
    value["hosts"] = [
        {
            "index": shard["index"], "host": shard["host"], "state": shard["state"],
            "remote_dir": f"{shard['remote_dir']}/{shard['attempts'][-1]}",
            **({"error": shard["error"]} if shard.get("error") else {}),
        }
        for shard in state["shards"]
    ]
    return value


def run(args: argparse.Namespace, now: datetime | None = None) -> int:
    """``roboscope eval ... --hosts``."""

    benchmark = cli.BenchmarkName(args.benchmark)
    output_dir = (
        cli._absolute(args.output_dir) if args.output_dir is not None
        else cli.default_output_dir(
            benchmark, f"{args.suite}-hosts", now or datetime.now(), Path.cwd() / "roboscope-runs",
        )
    )
    plan = make_plan(args, output_dir)
    first_targets = [
        (index, attempt_name(index, 0), f"{entry['dir']}/{plan['run_name']}")
        for index, entry in enumerate(plan["hosts"])
    ]
    if args.dry_run:
        reports = preflight(plan, first_targets)
        print(json.dumps(
            {"command": "eval", "dry_run": True, "hosts": [reports[index] for index in sorted(reports)]},
            sort_keys=True, indent=None if args.json else 2,
        ))
        return cli.EXIT_OK
    with exclusive(output_dir):
        state = load_state(output_dir, plan)
        if state is None:
            reports = preflight(plan, first_targets)
            state = {
                "schema_version": STATE_SCHEMA_VERSION,
                "plan": plan,
                "started_at": time.time(),
                "shards": [
                    {
                        "index": index,
                        "host": entry["name"],
                        "destination": entry["destination"],
                        "remote_dir": str(Path(reports[index]["output_dir"]).parent),
                        "python": reports[index]["python"],
                        "attempts": [attempt_name(index, 0)],
                        "state": "new",
                    }
                    for index, entry in enumerate(plan["hosts"])
                ],
            }
            write_state(output_dir, state)
        return _run(args, plan, state, output_dir)


def _run(args: argparse.Namespace, plan: dict[str, Any], state: dict[str, Any], output_dir: Path) -> int:
    retry = [shard for shard in state["shards"] if shard["state"] in {"failed", "lost"}]
    if retry:
        for shard in retry:
            shard["attempts"].append(attempt_name(shard["index"], len(shard["attempts"])))
            shard["state"] = "new"
            shard.pop("exit", None)
            shard.pop("error", None)
        preflight(plan, [(shard["index"], shard["attempts"][-1], shard["remote_dir"]) for shard in retry])
        write_state(output_dir, state)
    with _signals():
        try:
            for shard in state["shards"]:
                if shard["state"] == "new":
                    start_shard(plan, shard)
                    write_state(output_dir, state)
                    _log(f"started shard {shard['index']} on {shard['host']} in "
                         f"{shard['remote_dir']}/{shard['attempts'][-1]}")
            if args.detach:
                print(f"{cli.PROG} eval: the shards run on; follow and merge them with:\n  "
                      f"{follow_command(args, output_dir)}")
                return cli.EXIT_OK
            wait_for_shards(state, output_dir)
        except _Detach:
            write_state(output_dir, state)
            return 128 + signal.SIGHUP
        except _Stop as stop:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            _log("stopping the shards")
            stop_shards(state["shards"])
            deadline = time.monotonic() + STOP_GRACE_SECONDS
            def running() -> bool:
                return any(shard["state"] == "running" for shard in state["shards"])

            while time.monotonic() < deadline and running():
                time.sleep(min(5.0, POLL_SECONDS))
                statuses = poll(state["shards"])
                for shard in state["shards"]:
                    if shard["state"] == "running":
                        apply_status(shard, statuses.get(shard["index"]))
            write_state(output_dir, state)
            return 128 + stop.signum
        except HostError as exc:
            write_state(output_dir, state)
            raise cli.UsageError(str(exc)) from exc
    failed = [shard for shard in state["shards"] if shard["state"] != "succeeded"]
    if failed:
        for shard in failed:
            shard["error"] = shard_error(shard)
            log = f"{shard['remote_dir']}/{shard['attempts'][-1]}.log"
            _log(f"shard {shard['index']} on {shard['host']} {shard['state']}: "
                 f"{shard['error'] or 'no error message'}; log {log}")
        write_state(output_dir, state)
        return _finish(args, plan, state, output_dir, cli.EXIT_FAILED,
                       f"to start new attempts for the failed shards and merge, run:\n  "
                       f"{follow_command(args, output_dir)}")
    try:
        merged = merge_on_first_host(plan, state)
    except HostError as exc:
        _log(f"merge failed: {exc}")
        return _finish(args, plan, state, output_dir, cli.EXIT_FAILED,
                       f"to merge again, run:\n  {follow_command(args, output_dir)}")
    (output_dir / "result.json").write_text(merged, encoding="utf-8")
    state["merged"] = f"{state['shards'][0]['host']}:{state['shards'][0]['remote_dir']}/merged"
    write_state(output_dir, state)
    return _finish(args, plan, state, output_dir, cli.EXIT_OK, None)


def _finish(
    args: argparse.Namespace, plan: dict[str, Any], state: dict[str, Any], output_dir: Path, exit_code: int,
    advice: str | None,
) -> int:
    value = summary(plan, state, output_dir, exit_code)
    if args.json:
        print(json.dumps(value, sort_keys=True))
        if advice:
            print(f"{cli.PROG} eval: {advice}", file=sys.stderr)
        return exit_code
    target = f"{plan['benchmark']}/{plan['suite']}"
    count = len(state["shards"])
    if exit_code == cli.EXIT_OK and value.get("episodes") is not None:
        rate = value.get("success_rate")
        rate_text = f" ({rate:.1%})" if isinstance(rate, (int, float)) else ""
        print(f"{cli.PROG} eval {target} on {count} hosts: {value['successes']}/{value['episodes']} episodes "
              f"succeeded{rate_text}, wall {value['wall_s']:.0f} s; merged result in "
              f"{output_dir / 'result.json'}")
    else:
        failed = sum(shard["state"] != "succeeded" for shard in state["shards"])
        reason = f"{failed} of {count} shards did not succeed" if failed else "the merge failed"
        print(f"{cli.PROG} eval {target} on {count} hosts: {reason}; {advice}", file=sys.stderr)
    return exit_code


__all__ = ["command_hosts", "run"]
