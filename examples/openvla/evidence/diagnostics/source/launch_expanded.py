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
MODEL = Path("/root/autodl-tmp/openvla-diagnostic-model-cache-20261111")
PLAN = json.loads((ROOT / "selection.json").read_text())

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
write(ROOT / "expanded-experiment.json", {"scope": "diagnostic, not formal acceptance or throughput evidence",
    "selection": PLAN, "historical_kit_source": "b432786", "kit_concurrency": 2, "kit_replicas": 2, "execution_order": "all Native suites, then all Kit suites", "kit_build": json.loads((KIT / "build-manifest.json").read_text()),
    "native_build": json.loads((NATIVE / "build-manifest.json").read_text()),
    "hardware": subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total", "--format=csv"], text=True),
    "cpu": subprocess.check_output(["lscpu"], text=True),
    "sources": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in SOURCE.glob("*.py")},
    "deviations": ["all 36 Native versus Kit 2/2 disagreements; selected task/state identities only", "per-step lossless traces", "Kit work-item selection wrapper", "historical runtime and Policy, not rebased PR head"],
    "environment": {k: v for k, v in base_env.items() if k in ["HF_HOME", "OPENVLA_SEED", "PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM"]}})

status = {"state": "running", "completed": ["spatial-native"], "kit_concurrency": 2, "kit_replicas": 2, "expected_pairs": 36}
write(ROOT / "status.json", {**status, "current": "object-native (adopted)"})
while not (ROOT / "object-native/summary.json").exists():
    try: os.kill(22085, 0)
    except ProcessLookupError: raise RuntimeError("Adopted Native job exited without summary")
    time.sleep(5)
if not json.loads((ROOT / "object-native/summary.json").read_text())["execution_valid"]:
    raise RuntimeError("Adopted Object Native job failed")
status["completed"].append("object-native")
previous = {"spatial": {"7": [0], "1": [3]}, "object": {"2": [4], "3": [0]}}
jobs=[]
for suite, selection in PLAN.items():
    remaining={task:[i for i in indices if i not in previous.get(suite,{}).get(task,[])] for task,indices in selection.items()}
    remaining={task:indices for task,indices in remaining.items() if indices}
    jobs.append((suite,"native",remaining,f"{suite}-native"+("-extra" if suite in previous else "")))
for suite, selection in PLAN.items():jobs.append((suite,"kit",selection,f"{suite}-kit"))
try:
    for suite, arm, selection, name in jobs:
        output=ROOT/name
        while not (MODEL/suite/"VERIFIED.json").exists():
            status.update(current="waiting for verified checkpoint: "+suite);write(ROOT/"status.json",status)
            time.sleep(5)
        env={**base_env,"DIAGNOSTIC_SELECTION":json.dumps(selection),"OPENVLA_TRACE_ROOT":str(output/"policy-trace")}
        if arm=="native":
            cmd=[str(NATIVE/"env/bin/python"),str(SOURCE/"run.py"),"--runtime",str(NATIVE),"--checkpoint",str(MODEL/suite),
                 "--checkpoint-inventory",f"/root/autodl-tmp/openvla-libero-quick-20261109/inventories/{suite}.json",
                 "--suite",f"libero_{suite}","--output-dir",str(output),"--episodes-per-task","5","--seed","7","--env-seed","7",
                 "--video","off","--profile-name","matched-serial","--diagnostic-trace"]
            for task in sorted(selection,key=int):cmd.extend(["--task-id",task])
        else:
            cmd=[str(KIT/"env/bin/python"),str(SOURCE/"kit_entry.py"),"--benchmark","libero","--suite",f"libero_{suite}",
                 "--checkpoint-source",str(MODEL/suite),"--benchmark-manifest",str(ROOT/"kit-manifest.json"),
                 "--policy-config",str(ROOT/"policy.json"),"--episodes","5","--concurrency","2","--replicas","2","--output-dir",str(output)]
        write(ROOT/f"{name}-command.json",cmd)
        write(ROOT/f"{name}-selection.json",selection)
        status.update(current=name,updated_unix_s=time.time());write(ROOT/"status.json",status)
        with (ROOT/f"{name}.log").open("x") as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        status["completed"].append(name);write(ROOT/"status.json",status)
    status.update(state="complete",updated_unix_s=time.time())
except BaseException as e:
    status.update(state="failed",error=repr(e),updated_unix_s=time.time());raise
finally:write(ROOT/"status.json",status)
