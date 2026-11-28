"""RoboInF v1.0.0 clear suite factory and per-case accounting."""

import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from statistics import mean

from roboscope_kit.runtime.benchmark import ProcessSpec, WorkItem
from roboscope_kit.runtime.benchmark.kit_path import kit_import_root


def cases(manifest):
    data = manifest.runtime_path("data")
    raw = (data / "evaluation/test-cases.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != "9a88101906acce3ecc55011121bd012e85047ba4a9b75f9e7a670309d7d0951b":
        raise ValueError("RoboInF test cases do not match the release")
    return list(map(json.loads, raw.splitlines()))


class Adapter:
    def __init__(self, manifest, rows, suite):
        self.manifest = manifest
        self.rows = rows
        self.suite = suite

    def work_items(self):
        return [WorkItem(i, r["case_id"], r["task"], {}, 1) for i, r in enumerate(self.rows)]

    def metadata(self):
        return {
            "adapter": type(self).__name__,
            "benchmark": "roboinf",
            "suite": self.suite,
            "protocol": "roboinf-test-v1",
            "formal": len(self.rows) == 861,
        }

    def worker_process(self, context):
        judge = os.environ.get("ROBOSCOPE_ROBOINF_JUDGE")
        method = os.environ.get("ROBOSCOPE_ROBOINF_METHOD", "").strip()
        if not method:
            raise ValueError("Set ROBOSCOPE_ROBOINF_METHOD to the policy identifier")
        if not judge or not os.access(judge, os.X_OK):
            raise ValueError("Configure an authorized ROBOSCOPE_ROBOINF_JUDGE executable")
        m = self.manifest
        args = {
            "python": m.runtime_path("python"),
            "sim": m.runtime_path("launcher"),
            "robotwin": m.runtime_path("robotwin"),
            "data": m.runtime_path("data"),
            "policy-socket": context.replica_socket,
            "worker-id": context.worker_id,
            "state-dir": context.state_dir,
            "output": context.output_dir,
            "queue-dir": context.queue_dir,
            "method": method,
            "domain": self.suite,
            "action-spec": context.execution_contract.action_representation.removeprefix("roboinf."),
            "judge": judge,
        }
        return ProcessSpec(
            command=(
                sys.executable,
                "-m",
                "roboscope_kit.runtime.benchmarks.roboinf.worker",
                *(arg for k, v in args.items() for arg in ("--" + k, str(v))),
            ),
            env={
                "CUDA_VISIBLE_DEVICES": str(context.gpu_device),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": os.pathsep.join([str(kit_import_root()), str(m.runtime_path("source"))]),
            },
        )

    def aggregate(self, results):
        expected = {r["case_id"]: r["task"] for r in self.rows}
        seen = set()
        by_task = defaultdict(list)
        for result in results:
            for r in result.records:
                task = r.metadata["task"]
                if (
                    expected.get(r.record_id) != task
                    or r.record_id in seen
                    or r.metadata["domain"] != self.suite
                    or r.metadata["case_id"] != r.record_id
                    or type(r.success) is not bool
                    or r.success != r.metadata["task_success"]
                    or type(r.metadata["instruction_success"]) is not bool
                ):
                    raise ValueError("Invalid or repeated RoboInF case")
                seen.add(r.record_id)
                by_task[task].append((r.success, r.success and r.metadata["instruction_success"]))
        if seen != expected.keys():
            raise ValueError("Incomplete RoboInF workload")
        tasks = {
            k: {"cases": len(v), "TR": mean(p[0] for p in v), "JR": mean(p[1] for p in v)}
            for k, v in sorted(by_task.items())
        }
        return {
            "tasks": tasks,
            "formal": len(seen) == 861,
            **{"macro_" + k: mean(r[k] for r in tasks.values()) for k in ("TR", "JR")},
        }


class Factory:
    @staticmethod
    def validate_suite(suite):
        if suite not in ("test", "test_random"):
            raise ValueError("Expected test or test_random")

    def create(self, request, manifest):
        self.validate_suite(request.suite)
        rows = cases(manifest)
        if request.episodes is not None or request.quick_test:
            n = 1 if request.quick_test else request.episodes
            if not 1 <= n <= len(rows):
                raise ValueError("Invalid case limit")
            rows = rows[:n]
        return Adapter(manifest, rows, request.suite)

    def create_probe(self, request, manifest, concurrency):
        self.validate_suite(request.suite)
        return Adapter(manifest, cases(manifest)[:concurrency], request.suite)

    def self_test(self, manifest, output_dir):
        row = cases(manifest)[0]
        args = [
            str(manifest.runtime_path("python")),
            str(manifest.runtime_path("launcher")),
            "--self-test",
            "--case",
            row["case_id"],
            "--data",
            str(manifest.runtime_path("data")),
            "--robotwin",
            str(manifest.runtime_path("robotwin")),
        ]
        for domain in ("test", "test_random"):
            subprocess.run(
                [*args, "--domain", domain],
                check=True,
                env={**os.environ, "PYTHONPATH": str(manifest.runtime_path("source"))},
            )
        return {
            "benchmark": "roboinf",
            "reset_and_render": {d: row["case_id"] for d in ("test", "test_random")},
        }
