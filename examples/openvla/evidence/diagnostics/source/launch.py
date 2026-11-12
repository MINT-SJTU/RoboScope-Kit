"""Run eight selected historical disagreements through both unchanged rollout paths."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "source"
NATIVE = Path("/root/autodl-tmp/openvla-native-runtime")
KIT = Path("/root/autodl-tmp/openvla-kit-20261109/runtime")
MODEL = Path("/root/autodl-fs/Kit-Models/openvla")
PLAN = {"spatial": {"7": [0], "1": [3]}, "object": {"2": [4], "3": [0]},
        "goal": {"4": [1], "0": [2]}, "10": {"2": [0], "1": [2]}}

def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)

base_env = {**os.environ, "HF_HOME": "/root/autodl-tmp/openvla-libero-quick-20261109/hf",
    "OPENVLA_SEED": "7", "PYTHONHASHSEED": "7", "CUDA_VISIBLE_DEVICES": "0",
    "TF_FORCE_GPU_ALLOW_GROWTH": "true", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
manifest = json.loads((KIT / "benchmark-manifest.json").read_text())
for key in manifest["runtime"]:
    manifest["runtime"][key] = str(KIT / manifest["runtime"][key])
manifest["defaults"].update(seed=7, save_videos=False, chunk_size=1, ready_timeout_s=1800)
write(ROOT / "kit-manifest.json", manifest)
write(ROOT / "policy.json", {"schema_version": 2, "command": [str(KIT / "env/bin/python"), str(SOURCE / "kit_policy.py")], "cwd": str(SOURCE)})
write(ROOT / "experiment.json", {"scope": "diagnostic, not formal acceptance or throughput evidence",
    "selection": PLAN, "historical_kit_source": "b432786", "kit_concurrency": 2, "kit_replicas": 2, "execution_order": "all Native suites, then all Kit suites", "kit_build": json.loads((KIT / "build-manifest.json").read_text()),
    "native_build": json.loads((NATIVE / "build-manifest.json").read_text()),
    "hardware": subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total", "--format=csv"], text=True),
    "cpu": subprocess.check_output(["lscpu"], text=True),
    "sources": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in SOURCE.glob("*.py")},
    "deviations": ["selected task/state identities only", "per-step lossless traces", "Kit work-item selection wrapper", "historical runtime and Policy, not rebased PR head"],
    "environment": {k: v for k, v in base_env.items() if k in ["HF_HOME", "OPENVLA_SEED", "PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM"]}})
status = {"state": "running", "completed": [], "kit_concurrency": 2, "kit_replicas": 2}
# Adopt the already running Spatial Native job without loading it again.
while not (ROOT / "spatial-native/summary.json").exists():
    try:
        os.kill(20216, 0)
    except ProcessLookupError:
        raise RuntimeError("Existing Native job exited without a summary")
    time.sleep(5)
existing = json.loads((ROOT / "spatial-native/summary.json").read_text())
if not existing["execution_valid"]:
    raise RuntimeError("Existing Native diagnostic failed; inspect its logs")
status["completed"].append("spatial-native")
try:
    for arm in ["native", "kit"]:
        for suite, selection in PLAN.items():
            if f"{suite}-{arm}" in status["completed"]:
                continue
            output = ROOT / f"{suite}-{arm}"
            env = {**base_env, "DIAGNOSTIC_SELECTION": json.dumps(selection), "OPENVLA_TRACE_ROOT": str(output / "policy-trace")}
            if arm == "native":
                cmd = [str(NATIVE / "env/bin/python"), str(SOURCE / "run.py"), "--runtime", str(NATIVE),
                       "--checkpoint", str(MODEL / suite), "--checkpoint-inventory", f"/root/autodl-tmp/openvla-libero-quick-20261109/inventories/{suite}.json",
                       "--suite", f"libero_{suite}", "--output-dir", str(output), "--episodes-per-task", "5",
                       "--seed", "7", "--env-seed", "7", "--video", "off", "--profile-name", "matched-serial", "--diagnostic-trace"]
                for task in selection:
                    cmd.extend(["--task-id", task])
            else:
                cmd = [str(KIT / "env/bin/python"), str(SOURCE / "kit_entry.py"), "--benchmark", "libero", "--suite", f"libero_{suite}",
                       "--checkpoint-source", str(MODEL / suite), "--benchmark-manifest", str(ROOT / "kit-manifest.json"),
                       "--policy-config", str(ROOT / "policy.json"), "--episodes", "5", "--concurrency", "2", "--replicas", "2", "--output-dir", str(output)]
            write(ROOT / f"{suite}-{arm}-command.json", cmd)
            status.update(current=f"{suite}-{arm}", updated_unix_s=time.time())
            write(ROOT / "status.json", status)
            with (ROOT / f"{suite}-{arm}.log").open("x") as log:
                subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            status["completed"].append(f"{suite}-{arm}")
            write(ROOT / "status.json", status)
    status.update(state="complete", updated_unix_s=time.time())
except BaseException as e:
    status.update(state="failed", error=repr(e), updated_unix_s=time.time())
    raise
finally:
    write(ROOT / "status.json", status)
