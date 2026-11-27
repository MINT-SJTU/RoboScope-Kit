"""Dispatch official clear cases and obtain judgments from the evaluator owner."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from roboscope_kit.client import PolicyClient
from roboscope_kit.runtime.benchmark import EvaluationRecord, WorkerLoop, WorkResult

from .rollout import run_case


def main():
    p = argparse.ArgumentParser()
    for name in (
        "python",
        "sim",
        "data",
        "robotwin",
        "policy-socket",
        "queue-dir",
        "state-dir",
        "output",
        "judge",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--worker-id", type=int, required=True)
    p.add_argument("--method", required=True)
    args = p.parse_args()
    predictions = args.output / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    with PolicyClient(args.policy_socket) as client:

        def evaluate(item):
            started = time.monotonic()
            cid, task = item.item_id, item.group_id
            output = predictions / f"{cid}.jsonl"
            steps = run_case(
                client,
                [
                    str(args.python),
                    str(args.sim),
                    "--case",
                    cid,
                    "--data",
                    str(args.data),
                    "--robotwin",
                    str(args.robotwin),
                    "--method",
                    args.method,
                    "--output",
                    str(output),
                ],
                task,
            )
            digest = hashlib.sha256(output.read_bytes()).hexdigest()
            decision = json.loads(subprocess.check_output([str(args.judge), str(output)]))
            if (
                decision["case_id"] != cid
                or decision["domain"] != "test"
                or decision["prediction_sha256"] != digest
                or any(type(decision[k]) is not bool for k in ("task_success", "instruction_success"))
            ):
                raise ValueError(f"Invalid checker receipt for {cid}")
            tr = decision["task_success"]
            record = EvaluationRecord(cid, tr, steps, time.monotonic() - started, {"task": task, **decision})
            return WorkResult(cid, task, args.worker_id, (record,))

        WorkerLoop(
            worker_id=args.worker_id, queue_dir=args.queue_dir, state_dir=args.state_dir, poll_interval_s=0.05
        ).run(evaluate)


if __name__ == "__main__":
    main()
