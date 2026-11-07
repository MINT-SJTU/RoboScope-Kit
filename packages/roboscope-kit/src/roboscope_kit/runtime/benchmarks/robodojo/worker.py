# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Run a RoboDojo layout shard in the pinned simulator environment."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import uuid
from pathlib import Path

from roboscope_kit.runtime.benchmark import EvaluationRecord, WorkerLoop, WorkItem, WorkResult

LAYOUT_PASSES = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--policy-socket", type=Path, required=True)
    parser.add_argument(
        "--action-representation", required=True,
        choices=("robodojo.x5-joint-v1", "robodojo.x5-ee-compact-quat-v1"),
    )
    parser.add_argument("--benchmark-python", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--worker-state-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--async-video", action="store_true")
    parser.add_argument("--unpaced-render", action="store_true")
    parser.add_argument("--control-state-reuse", action="store_true")
    parser.add_argument("--physics-threads", type=int, choices=(1, 4, 8))
    parser.add_argument("--envs-per-worker", type=int, default=1)
    parser.add_argument("--step-observation-updates", action="store_true")
    parser.add_argument("--interactive-policy", action="store_true")
    parser.add_argument("--observation-stride", type=int, default=1)
    parser.add_argument("--observation-queue-size", type=int, choices=range(0, 11), default=0)
    args = parser.parse_args()
    if args.observation_queue_size and not args.step_observation_updates:
        parser.error("observation overlap requires step observation updates")
    if args.interactive_policy and (args.envs_per_worker != 1 or args.observation_queue_size):
        parser.error("interactive policies require one environment and no observation overlap")
    if args.envs_per_worker < 1:
        parser.error("envs per worker must be positive")
    if args.observation_stride < 1:
        parser.error("observation stride must be positive")
    return args


def _run_evaluator(command: list[str], root: Path, env: dict, log, log_path: Path) -> None:
    # RoboDojo's resume manifest is keyed by run_id; preserve it across bounded
    # restarts for the upstream PhysX fatal-restart exit codes.
    for attempt in range(4):
        completed = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode == 0:
            return
        if completed.returncode not in (99, 134, 139, -6, -11) or attempt == 3:
            log.flush()
            with log_path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                handle.seek(max(0, handle.tell() - 8192))
                tail = handle.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"RoboDojo evaluator exited {completed.returncode}; see {log_path}\n{tail}")


def _layout_rows(result_path: Path, layouts: list[int]) -> dict[int, dict]:
    """Rows of one evaluator pass by layout; each must be one of the requested layouts, at most once."""

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    details = payload.get("details")
    if not isinstance(details, dict) or payload.get("eval_time") != len(details):
        raise RuntimeError(f"RoboDojo returned an inconsistent result; see {result_path}")
    rows = list(details.values())
    ids = [row.get("layout_id") if isinstance(row, dict) else None for row in rows]
    if len(set(ids)) != len(ids) or not set(ids) <= set(layouts):
        raise RuntimeError(
            f"RoboDojo returned incorrect layout IDs; expected a subset of {layouts}, got {ids}"
        )
    return {int(row["layout_id"]): row for row in rows}


def run_item(args: argparse.Namespace, item: WorkItem) -> WorkResult:
    task = str(item.payload["task"])
    seed = int(item.payload["seed"])
    start, count = int(item.payload["start"]), int(item.payload["count"])
    if start < 0 or count < 1 or count != item.expected_results:
        raise ValueError("invalid RoboDojo layout shard")
    if os.environ.get("OMNI_KIT_ACCEPT_EULA", "").upper() not in {"Y", "YES"}:
        raise RuntimeError(
            "RoboDojo uses NVIDIA Isaac Sim. Read its license at "
            "https://docs.isaacsim.omniverse.nvidia.com/latest/common/"
            "NVIDIA_Omniverse_License_Agreement.html and set OMNI_KIT_ACCEPT_EULA=YES to accept it."
        )

    root = args.benchmark_root.resolve()
    if args.envs_per_worker > 1:
        deploy = root / "XPolicyLab/policy/RoboScopeKit/deploy.yml"
        if "eval_batch: true" not in deploy.read_text(encoding="utf-8").splitlines():
            # RoboDojo silently forces one environment without it; rebuild with build-runtime.sh.
            raise RuntimeError(f"envs_per_worker > 1 requires eval_batch: true in {deploy}")
    logs = args.output_dir / "native-logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"worker-{args.worker_id:03d}-{item.item_id}.log"
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": args.gpu_id,
        "PYTHONPATH": os.pathsep.join((str(root), str(root / "XPolicyLab"))),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "ROBOSCOPE_INFERENCE_SOCKET": str(args.policy_socket.resolve()),
        # Camera frames go every control step; MessagePack avoids JSON/base64 encoding them.
        "ROBOSCOPE_POLICY_ENCODING": "msgpack",
        "ROBOSCOPE_ROBODOJO_ACTION_REPRESENTATION": args.action_representation,
        "ROBOSCOPE_ROBODOJO_STEP_UPDATES": "1" if args.step_observation_updates else "0",
        "ROBOSCOPE_ROBODOJO_INTERACTIVE": "1" if getattr(args, "interactive_policy", False) else "0",
        "ROBOSCOPE_ROBODOJO_OBSERVATION_QUEUE_SIZE": str(args.observation_queue_size),
        "ROBOSCOPE_ROBODOJO_OBSERVATION_STRIDE": str(args.observation_stride),
        # The caller has explicitly accepted the same EULA above.
        "ACCEPT_EULA": "Y",
    }
    for name in (
        "ROBODOJO_ASYNC_VIDEO", "ROBODOJO_UNPACED_RENDER", "ROBODOJO_CONTROL_STATE_REUSE",
        "ROBODOJO_PHYSICS_THREADS",
    ):
        env.pop(name, None)
    if args.async_video:
        env["ROBODOJO_ASYNC_VIDEO"] = "1"
    if args.unpaced_render:
        env["ROBODOJO_UNPACED_RENDER"] = "1"
    if args.control_state_reuse:
        env["ROBODOJO_CONTROL_STATE_REUSE"] = "1"
    if args.physics_threads is not None:
        env["ROBODOJO_PHYSICS_THREADS"] = str(args.physics_threads)
    command = [
        str(args.benchmark_python), "-u", str(root / "src/eval_client/main.py"),
        "--task_name", task, "--env_cfg_type", "arx_x5", "--num_envs", "1",
        "--enable_cameras", "--kit_args",
        "--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera",
        "--device_id", "0", "--policy_name", "RoboScopeKit", "--port", "1",
        "--protocol", "ws", "--policy_server_url", "ws://localhost:1",
        "--additional_info", "roboscope", "--seed", str(seed), "--host", "localhost", "--headless",
    ]
    # RoboDojo abandons a layout whose physics breaks and, with no spare seeds in a fixed shard,
    # returns without it. Evaluate the missing layouts again, so the shard keeps its layout set.
    expected = list(range(start, start + count))
    evaluator_envs = min(args.envs_per_worker, count)
    rows: dict[int, dict] = {}
    result_paths = []
    # Append, so a Worker that retries this item keeps the failed attempt's log.
    with log_path.open("a", encoding="utf-8") as log:
        for _ in range(LAYOUT_PASSES):
            missing = [layout for layout in expected if layout not in rows]
            if len(missing) < evaluator_envs:
                raise RuntimeError(
                    f"RoboDojo cannot retry missing layouts {missing} without reducing environments "
                    f"from {evaluator_envs} to {len(missing)}; see {result_paths}"
                )
            run_id = f"roboscope-{uuid.uuid4().hex}"
            result_path = (
                root / "eval_result/RoboDojo" / task / "RoboScopeKit/arx_x5" / f"{seed}_roboscope" / run_id
                / "_result.json"
            )
            env.update({
                "ROBODOJO_RUN_ID": run_id,
                "ROBODOJO_LAYOUT_IDS": json.dumps(missing),
                "EVAL_NUM": str(len(missing)),
            })
            command[command.index("--num_envs") + 1] = str(evaluator_envs)
            _run_evaluator(command, root, env, log, log_path)
            if not result_path.is_file():
                raise RuntimeError(f"RoboDojo did not write results; see {log_path}")
            result_paths.append(str(result_path))
            rows.update(_layout_rows(result_path, missing))
            if len(rows) == count:
                break
        else:
            raise RuntimeError(
                f"RoboDojo returned an incomplete shard after {LAYOUT_PASSES} passes; "
                f"missing layouts {[layout for layout in expected if layout not in rows]}; see {result_paths}"
            )
    records = []
    for layout, row in sorted(rows.items()):
        success, steps = row.get("success"), row.get("roboscope_steps")
        score, elapsed = row.get("score"), row.get("roboscope_elapsed_s")
        if (type(success) is not bool or type(steps) is not int or steps < 0
                or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1
                or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0):
            raise RuntimeError(f"invalid RoboDojo result for {task} layout {layout}: {row!r}")
        records.append(EvaluationRecord(
            record_id=f"{task}-{seed}-{layout:04d}",
            success=success,
            steps=steps,
            elapsed_s=float(elapsed),
            metadata={"task": task, "layout_id": layout, "score": float(score),
                      "evaluator_num_envs": evaluator_envs},
        ))
    return WorkResult(
        item_id=item.item_id,
        group_id=item.group_id,
        worker_id=args.worker_id,
        records=tuple(records),
        metadata={"task": task, "result_paths": result_paths, "log_path": str(log_path),
                  "evaluator_num_envs": evaluator_envs},
    )


def main() -> None:
    args = parse_args()
    WorkerLoop(
        worker_id=args.worker_id,
        queue_dir=args.queue_dir,
        state_dir=args.worker_state_dir,
    ).run(lambda item: run_item(args, item))


if __name__ == "__main__":
    main()
