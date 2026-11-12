"""Select diagnostic work items without changing installed SDK rollout code."""
import dataclasses
import hashlib
import sys
import json
import os
import runpy
from pathlib import Path
from roboscope_kit.runtime.benchmarks.libero.adapter import LiberoBenchmarkAdapter

original = LiberoBenchmarkAdapter.work_items
selection = json.loads(os.environ["DIAGNOSTIC_SELECTION"])

def selected(self):
    items = [item for item in original(self)
             if item.payload["task_id"] in map(int, selection)
             and item.payload["start_episode"] in selection[str(item.payload["task_id"])]]
    assert len(items) == sum(len(v) for v in selection.values())
    return [dataclasses.replace(item, ordinal=i) for i, item in enumerate(items)]

LiberoBenchmarkAdapter.work_items = selected
original_worker_process = LiberoBenchmarkAdapter.worker_process

def observed_worker_process(self, context):
    spec = original_worker_process(self, context)
    command = list(spec.command)
    assert command[1:3] == ["-m", "roboscope_kit.runtime.benchmarks.libero.worker"]
    command[1:3] = [str(Path(__file__).with_name("worker_entry.py"))]
    env = {**spec.env, "DIAGNOSTIC_WORKER_TRACE": str(Path(os.environ["OPENVLA_TRACE_ROOT"]).parent / "worker-trace")}
    return dataclasses.replace(spec, command=tuple(command), env=env)

LiberoBenchmarkAdapter.worker_process = observed_worker_process
if "--help" not in sys.argv:
    output = Path(os.environ["OPENVLA_TRACE_ROOT"]).parent
    provenance = {"argv": sys.argv, "selection": selection, "python": sys.version,
        "sources": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob("*.py")},
        "observer": "External Worker wrapper delegates rollout and records actual environment actions"}
    (output.parent / (output.name + "-diagnostic-provenance.json")).write_text(json.dumps(provenance, indent=2) + "\n")
runpy.run_module("roboscope_kit.runtime.benchmark", run_name="__main__")
