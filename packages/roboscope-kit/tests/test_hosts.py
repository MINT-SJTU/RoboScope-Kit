# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""``roboscope hosts`` and ``roboscope eval --hosts`` against a local stand-in for SSH.

The stand-in runs each script with ``sh`` in a home directory of its own per destination, so the launch,
status, stop, copy and merge scripts run for real; a stand-in ``roboscope`` answers ``eval --dry-run`` and
``hosts probe`` with the real command and writes small shard and merge results for the rest.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from pathlib import Path

import pytest

import roboscope_kit
from roboscope_kit import cli, hosts

FAKE_SSH = """
import os, subprocess, sys
destination, command = sys.argv[1], " ".join(sys.argv[2:])
if destination.startswith("down"):
    print(f"ssh: connect to host {destination} port 22: Connection refused", file=sys.stderr)
    sys.exit(255)
home = os.path.join(os.environ["FAKE_SSH_ROOT"], destination)
os.makedirs(home, exist_ok=True)
os.chdir(home)
sys.exit(subprocess.run(["sh", "-c", command], env={**os.environ, "HOME": home}).returncode)
"""

FAKE_ROBOSCOPE = """
import json, os, signal, sys, time
sys.path.insert(0, {kit_root!r})
from roboscope_kit import cli

cli.gpu_names = lambda: ["Fake GPU"]
argv = sys.argv[1:]
if "--dry-run" in argv or argv[:2] == ["hosts", "probe"]:
    sys.exit(cli.main(argv))

def value(flag):
    return argv[argv.index(flag) + 1]

if argv[0] == "eval":
    index, output = int(value("--shard-index")), value("--output-dir")
    os.makedirs(os.path.join(output, "queue", "completed"))
    with open(os.path.join(output, "manifest.json"), "w") as handle:
        json.dump({{"items": [{{"item_id": f"{{index}}-{{n}}"}} for n in range(3)]}}, handle)
    time.sleep(float(os.environ.get("FAKE_EVAL_SECONDS", "0.3")))
    fail_file = os.environ.get("FAKE_FAIL_FILE")
    failing = open(fail_file).read().split() if fail_file and os.path.exists(fail_file) else []
    if str(index) in failing:
        with open(fail_file, "w") as handle:
            handle.write(" ".join(item for item in failing if item != str(index)))
        print(json.dumps({{"command": "eval", "exit_code": 1, "error": f"fake failure in shard {{index}}"}}))
        sys.exit(1)
    for n in range(3):
        with open(os.path.join(output, "queue", "completed", f"item-{{n}}.json"), "w") as handle:
            json.dump({{"item_id": f"{{index}}-{{n}}"}}, handle)
    with open(os.path.join(output, "result.json"), "w") as handle:
        json.dump({{"result": {{"state": "complete", "episodes": 3, "successes": index + 1}}}}, handle)
    print(json.dumps({{"command": "eval", "exit_code": 0}}))
    sys.exit(0)
if argv[0] == "merge":
    shard_dirs = [argv[i + 1] for i, item in enumerate(argv) if item == "--shard-dir"]
    output = value("--output-dir")
    results = [json.load(open(os.path.join(d, "result.json")))["result"] for d in shard_dirs]
    completed = [len(os.listdir(os.path.join(d, "queue", "completed"))) for d in shard_dirs]
    episodes, successes = sum(r["episodes"] for r in results), sum(r["successes"] for r in results)
    os.makedirs(output, exist_ok=True)
    with open(os.path.join(output, "result.json"), "w") as handle:
        json.dump({{
            "merged": True, "shard_dirs": shard_dirs, "completed": completed, "argv": argv,
            "result": {{"state": "complete", "episodes": episodes, "successes": successes,
                        "success_rate": successes / episodes}},
        }}, handle)
    sys.exit(0)
sys.exit(f"unexpected arguments {{argv}}")
"""


@pytest.fixture
def lab(tmp_path, monkeypatch):
    """Two reachable stand-in hosts ``a`` and ``b`` registered with the stand-in roboscope."""

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("FAKE_SSH_ROOT", str(tmp_path / "hosts"))
    monkeypatch.delenv("ROBOSCOPE_HOME", raising=False)
    ssh = tmp_path / "fake_ssh.py"
    ssh.write_text(textwrap.dedent(FAKE_SSH))
    roboscope = tmp_path / "fake_roboscope.py"
    kit_root = str(Path(roboscope_kit.__file__).resolve().parents[1])
    roboscope.write_text(textwrap.dedent(FAKE_ROBOSCOPE.format(kit_root=kit_root)))
    monkeypatch.setattr(hosts, "SSH_COMMAND", (sys.executable, str(ssh)))
    monkeypatch.setattr(hosts, "POLL_SECONDS", 0.05)
    monkeypatch.setattr(hosts, "STOP_GRACE_SECONDS", 10.0)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "benchmark-manifest.json").write_text(json.dumps({
        "schema_version": 3, "supported_benchmarks": ["libero"],
        "supported_suites": {"libero": ["libero_10"]},
        "protocol_versions": {"libero": "libero-v1"}, "defaults": {}, "benchmark_protocols": {},
        "runtime": {"python": sys.executable, "benchmark_root": str(runtime), "config_path": str(runtime)},
    }))
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "model.bin").write_bytes(b"weights")
    policy = tmp_path / "roboscope-policy.json"
    policy.write_text("{}")
    entry = {"roboscope": [sys.executable, str(roboscope)], "dir": "runs"}
    hosts.write_registry({"a": {"destination": "a", **entry}, "b": {"destination": "b", **entry}})
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return tmp_path


def _eval(lab: Path, *extra: str, hosts_value: str = "a,b", output: str = "out") -> list[str]:
    return [
        "eval", "libero", "libero_10", "--policy-config", str(lab / "roboscope-policy.json"),
        "--checkpoint-source", str(lab / "ckpt"), "--benchmark-manifest", str(lab / "runtime"),
        "--hosts", hosts_value, "--output-dir", str(lab / output), *extra,
    ]


def _state(lab: Path, output: str = "out") -> dict:
    return json.loads((lab / output / hosts.STATE_NAME).read_text())


def test_every_eval_option_is_forwarded_or_set_per_host():
    parser = cli.build_parser()
    evaluate = parser._subparsers._group_actions[0].choices["eval"]
    dests = {action.dest for action in evaluate._actions} - {"help"}
    forwarded = {attribute for _, attribute in hosts.FORWARDED_VALUES + hosts.FORWARDED_FLAGS}
    assert dests == forwarded | hosts.CONTROLLER_OPTIONS
    assert not forwarded & hosts.CONTROLLER_OPTIONS


def test_hosts_resolve_registered_names_then_ssh_destinations(lab):
    resolved = hosts.resolve_hosts("a,root@gpu9")
    assert resolved[0]["destination"] == "a" and resolved[0]["dir"] == "runs"
    assert resolved[1] == {
        "name": "root@gpu9", "destination": "root@gpu9", "roboscope": ["roboscope"], "dir": "roboscope-runs",
    }
    for value in ("a,,b", "-oProxyCommand=x"):
        with pytest.raises(cli.UsageError):
            hosts.resolve_hosts(value)


def test_copy_id_takes_the_port_out_of_an_ssh_uri():
    assert hosts.copy_id_argv("ssh://root@connect.example.com:15078") == [
        "ssh-copy-id", "-p", "15078", "root@connect.example.com",
    ]
    assert hosts.copy_id_argv("gpu1") == ["ssh-copy-id", "gpu1"]


@pytest.mark.parametrize("extra", [("--shard-index", "0"), ("--shard-count", "2"), ("--quick-test",)])
def test_hosts_assign_the_shards_themselves(lab, capsys, extra):
    assert cli.main(_eval(lab, *extra)) == cli.EXIT_USAGE
    err = capsys.readouterr().err
    assert ("quick test cannot be sharded" if extra == ("--quick-test",) else "drop --shard") in err


def test_single_host_dry_run_reports_what_hosts_compare(lab, capsys, monkeypatch):
    monkeypatch.setattr(cli, "gpu_names", lambda: ["Fake GPU"])
    argv = _eval(lab, "--dry-run", "--json")
    argv = argv[: argv.index("--hosts")] + argv[argv.index("--hosts") + 2:]
    assert cli.main(argv) == cli.EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"] and report["kit_version"] == cli.kit_version()
    assert report["checkpoint"]["exists"] and report["checkpoint"]["files"] == 1
    assert len(report["manifest_sha256"]) == 64 and report["output_dir_in_use"] is False
    assert not (lab / "out").exists()


def test_eval_on_two_hosts_runs_each_shard_and_merges_on_the_first(lab, capsys):
    assert cli.main(_eval(lab)) == cli.EXIT_OK
    captured = capsys.readouterr()
    assert "libero/libero_10 on 2 hosts: 3/6 episodes succeeded (50.0%)" in captured.out
    merged = json.loads((lab / "out" / "result.json").read_text())
    first = lab / "hosts" / "a" / "runs" / "out"
    assert merged["shard_dirs"] == [str(first / "shard-0"), str(first / "merge-inputs" / "shard-1")]
    assert merged["completed"] == [3, 3]
    assert "--benchmark-manifest" in merged["argv"]
    state = _state(lab)
    assert [shard["state"] for shard in state["shards"]] == ["succeeded", "succeeded"]
    assert state["shards"][1]["remote_dir"] == str(lab / "hosts" / "b" / "runs" / "out")
    launched = (lab / "hosts" / "b" / "runs" / "out" / "shard-1.out").read_text()
    assert json.loads(launched)["exit_code"] == 0
    assert "shard 1 b: succeeded" in captured.err


def test_unreachable_host_stops_the_run_before_any_shard_starts(lab, capsys):
    assert cli.main(_eval(lab, hosts_value="a,down1")) == cli.EXIT_USAGE
    err = capsys.readouterr().err
    assert "down1" in err and "roboscope hosts add" in err
    assert not list((lab / "hosts").rglob("*.pid"))


def test_hosts_with_different_checkpoints_are_refused(lab, capsys):
    for name, content in (("a", b"one"), ("b", b"other")):
        checkpoint = lab / "hosts" / name / "ckpt"
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.bin").write_bytes(content * (1 if name == "a" else 2))
    argv = _eval(lab)
    argv[argv.index("--checkpoint-source") + 1] = "ckpt"  # relative: each host's own home
    assert cli.main(argv) == cli.EXIT_USAGE
    assert "different checkpoint files" in capsys.readouterr().err


def test_a_failed_shard_gets_a_new_attempt_when_the_command_runs_again(lab, capsys, monkeypatch):
    fail = lab / "fail"
    fail.write_text("1")
    monkeypatch.setenv("FAKE_FAIL_FILE", str(fail))
    assert cli.main(_eval(lab)) == cli.EXIT_FAILED
    err = capsys.readouterr().err
    assert "fake failure in shard 1" in err and "--output-dir" in err
    assert [shard["state"] for shard in _state(lab)["shards"]] == ["succeeded", "failed"]
    assert not (lab / "out" / "result.json").exists()
    assert cli.main(_eval(lab)) == cli.EXIT_OK
    state = _state(lab)
    assert state["shards"][1]["attempts"] == ["shard-1", "shard-1-retry1"]
    assert state["shards"][0]["attempts"] == ["shard-0"]
    merged = json.loads((lab / "out" / "result.json").read_text())
    assert merged["shard_dirs"][1].endswith("merge-inputs/shard-1-retry1")


def test_detach_returns_and_the_same_command_follows_and_merges(lab, capsys):
    assert cli.main(_eval(lab, "--detach")) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "follow and merge them with" in out and "--detach" not in out.splitlines()[-1]
    assert all(shard["state"] == "running" for shard in _state(lab)["shards"])
    assert cli.main(_eval(lab)) == cli.EXIT_OK
    assert json.loads((lab / "out" / "result.json").read_text())["result"]["episodes"] == 6


def test_a_different_command_cannot_follow_an_existing_run(lab, capsys):
    assert cli.main(_eval(lab, "--detach")) == cli.EXIT_OK
    assert cli.main(_eval(lab, "--episodes", "2")) == cli.EXIT_USAGE
    assert "different multi-host run" in capsys.readouterr().err


def test_stop_sends_sigterm_and_the_shard_ends_failed(lab, monkeypatch):
    monkeypatch.setenv("FAKE_EVAL_SECONDS", "60")
    assert cli.main(_eval(lab, "--detach")) == cli.EXIT_OK
    state = _state(lab)
    hosts.stop_shards(state["shards"])
    for _ in range(200):
        statuses = hosts.poll(state["shards"])
        for shard in state["shards"]:
            hosts.apply_status(shard, statuses.get(shard["index"]))
        if all(shard["state"] != "running" for shard in state["shards"]):
            break
        time.sleep(0.05)
    assert [shard.get("exit") for shard in state["shards"]] == [143, 143]


def test_an_attempt_name_is_never_reused(lab):
    script = hosts.launch_script("runs/x", "shard-0", ["true"])
    host = hosts.Host("a")
    assert host.run(script).returncode == 0
    second = host.run(script)
    assert second.returncode == 3 and "already started" in second.stderr


def test_hosts_add_probes_and_registers_and_list_shows_each_host(lab, capsys):
    roboscope = hosts.registered_hosts()["a"]["roboscope"]
    command = hosts._command_text(roboscope)
    assert cli.main(["hosts", "add", "c", "c", "--roboscope", command, "--dir", "~/r"]) == 0
    out = capsys.readouterr().out
    assert "c  c  kit " in out and "1 GPU (Fake GPU)" in out and "registered c" in out
    assert hosts.registered_hosts()["c"] == {"destination": "c", "roboscope": roboscope, "dir": "r"}
    hosts.write_registry({**hosts.registered_hosts(), "gone": {"destination": "down2", **{
        key: hosts.registered_hosts()["a"][key] for key in ("roboscope", "dir")}}})
    assert cli.main(["hosts"]) == 0
    listing = capsys.readouterr().out
    assert "gone  down2  unreachable" in listing and listing.count("Fake GPU") == 3
    assert cli.main(["hosts", "remove", "gone"]) == 0
    assert "gone" not in hosts.registered_hosts()


def test_hosts_add_needs_a_terminal_for_a_password(lab, monkeypatch, capsys):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    assert cli.main(["hosts", "add", "x", "down3"]) == cli.EXIT_USAGE
    assert "no terminal" in capsys.readouterr().err
    assert "x" not in hosts.registered_hosts()


def test_probe_reports_this_machine(lab, capsys, monkeypatch):
    monkeypatch.setattr(cli, "gpu_names", lambda: [])
    assert cli.main(["hosts", "probe", "--dir", "nowhere/yet", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["kit_version"] == cli.kit_version() and report["gpus"] == []
    assert report["dir"] == os.path.join(os.getcwd(), "nowhere", "yet") and report["disk_free_bytes"] > 0
