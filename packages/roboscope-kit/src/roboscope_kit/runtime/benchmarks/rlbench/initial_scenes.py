# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Check every test episode's single seeded reset against its recorded first frame, object by object.

Runs in the RLBench runtime environment (``build-runtime.sh`` calls it)::

    python -m roboscope_kit.runtime.benchmarks.rlbench.initial_scenes --test-set <root> \\
        --xvfb <Xvfb> --pins benchmarks/rlbench/peract-initial-scenes.json [--jobs N] \\
        [--report <file>] [--write-pins]

For each of the 18 tasks x 25 test episodes it loads the task afresh, resets it once to the
episode with the worker's own code (``worker.reset_scene``), and compares every task object's
position, by object handle, with the episode's recorded ``task_low_dim_state``. The recorded
vector has no object names: it is laid out over the task's object list in the order the scene
listed it when the frame was recorded. The check requires that the vector's length is exactly
the layout of a fresh load (``worker.recorded_positions``). For an episode more than
``TOLERANCE_M`` off it tries whether the vector was laid out after a grasped object was released
in the same loaded task (``explain_by_release``), which lists that object later, and classifies
the episode (``_known_difference``).

Each task runs in its own CoppeliaSim process, episodes 0 to 24 in order, at most ``--jobs`` at a
time: where objects settle into contact, a reset's outcome depends on what the process simulated
before, and this makes the result independent of ``--jobs``.

The episodes more than ``TOLERANCE_M`` off are compared with the frozen list in ``--pins``. The
check fails on a layout mismatch, on an object order its model of CoppeliaSim cannot reproduce,
on an episode off that the list does not hold, on a listed recorded order that is now within the
tolerance or explained by releasing other objects, or on a pin file whose digest does not match
its list. Where contact settling is involved it reports instead of failing: an episode of a task
with listed settling episodes that settles past the tolerance, a listed settling episode that
comes out within it, and a changed explanation of an episode that also settles. Objects that
settle into contact are not guaranteed to land at the same place on other hardware.
``--write-pins`` writes the measured list instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from .tasks import EPISODES_PER_TASK, PROTOCOL_VERSION, TASK_NAMES

# An object more than 1 cm from its recorded position: the episode does not start from its test scene.
TOLERANCE_M = 0.01


def pinned_sha256(episodes: list[dict]) -> str:
    """Digest of a known-difference list: ``task:episode:cause:released objects``, in order.

    The measured distances are not part of it: objects settling into contact do not land at the
    same place in every run.
    """

    digest = hashlib.sha256()
    keys = sorted(
        (row["task"], int(row["episode"]), row["cause"], ",".join(row.get("released") or ()))
        for row in episodes
    )
    for task, episode, cause, released in keys:
        digest.update(f"{task}:{episode}:{cause}:{released}\n".encode())
    return digest.hexdigest()


def _known_difference(row: dict) -> dict:
    """Classify an episode more than ``TOLERANCE_M`` off.

    ``recorded_order``: laid out over the order left by releasing ``released``, the recorded
    state is within the tolerance (``error_after_release_m``). ``recorded_order_and_settling``:
    that order explains part of the difference and objects are still more than the tolerance
    off. ``settling``: no release order explains it. ``unexplained_order``: the model of
    CoppeliaSim's object order did not reproduce the task's own list, so nothing was tried.
    """

    entry = {key: row[key] for key in ("task", "episode", "object")}
    entry["max_error_m"] = round(row["max_error_m"], 4)
    released = row["released"]
    if released is None:
        entry["cause"] = "unexplained_order"
    elif released["error_m"] < row["max_error_m"] - TOLERANCE_M:
        entry["cause"] = (
            "recorded_order" if released["error_m"] <= TOLERANCE_M else "recorded_order_and_settling"
        )
        entry["released"] = released["moved"]
        entry["error_after_release_m"] = round(released["error_m"], 4)
    else:
        entry["cause"] = "settling"
    return entry


def _children(objects, base) -> dict[int, list[int]]:
    """Each object's direct children, by handle, in CoppeliaSim's order."""

    return {
        node.get_handle(): [
            child.get_handle()
            for child in node.get_objects_in_tree(exclude_base=True, first_generation_only=True)
        ]
        for node in (base, *(obj for obj, _ in objects))
    }


def _tree_order(children: dict[int, list[int]], root: int) -> list[int]:
    """The order ``simGetObjectsInTree`` lists a tree in: breadth first, children in their order."""

    order, queue = [], list(children[root])
    while queue:
        node = queue.pop(0)
        order.append(node)
        queue.extend(children[node])
    return order


def _released(children: dict[int, list[int]], moved: tuple[int, ...]) -> dict[int, list[int]]:
    """The tree after ``Gripper.release`` re-parented each of ``moved``: last among its siblings."""

    children = {node: list(kids) for node, kids in children.items()}
    for handle in moved:
        parent = next(node for node, kids in children.items() if handle in kids)
        children[parent].remove(handle)
        children[parent].append(handle)
    return children


def explain_by_release(task, objects, sampled: np.ndarray, recorded_state) -> dict | None:
    """Whether the recorded state was laid out after one or two grasped objects were released.

    Re-parenting a released object puts it after its siblings (``Gripper.release``), so a task
    state recorded after a grasp in the same loaded task lists that object, and everything below
    it, later. This tries every graspable object and every ordered pair, lays the recorded state
    out over the resulting order, and returns the order with the smallest error, or ``None`` if
    the model of CoppeliaSim's order does not reproduce the task's own object list.
    """

    from itertools import permutations

    from .worker import max_displacement, recorded_positions

    base = task.get_base()
    children = _children(objects, base)
    by_handle = {obj.get_handle(): (obj, kind) for obj, kind in objects}
    if _tree_order(children, base.get_handle()) != list(by_handle):
        return None
    graspable = [obj.get_handle() for obj in task.get_graspable_objects() if obj.get_handle() in by_handle]
    position = {handle: index for index, handle in enumerate(by_handle)}
    best = None
    for count in (1, 2):
        for moved in permutations(graspable, count):
            order = [
                by_handle[handle] for handle in _tree_order(_released(children, moved), base.get_handle())
            ]
            try:
                recorded = recorded_positions(order, recorded_state)
            except ValueError:
                continue
            actual = sampled[[position[obj.get_handle()] for obj, _ in order]]
            error, worst = max_displacement(order, actual, recorded)
            if best is None or error < best["error_m"]:
                best = {
                    "moved": [by_handle[h][0].get_name() for h in moved],
                    "error_m": error,
                    "object": worst,
                }
        if best is not None and best["error_m"] <= TOLERANCE_M:
            break
    return best


def measure(test_set: Path, xvfb: Path, tasks: list[str]) -> list[dict]:
    from .test_set import load_test_episode
    from .worker import LoadedTask, create_environment, max_displacement, reset_scene, start_xvfb, task_class

    start_xvfb(xvfb)
    environment = create_environment(test_set)
    rows = []
    try:
        for task in tasks:
            for index in range(EPISODES_PER_TASK):
                episode = load_test_episode(test_set, task, index)
                started = time.perf_counter()
                loaded = LoadedTask(environment.get_task(task_class(task)))
                row: dict = {"task": task, "episode": index}
                try:
                    _, objects, sampled, recorded = reset_scene(loaded, episode)
                except ValueError as exc:
                    row["layout_error"] = str(exc)
                else:
                    error, worst = max_displacement(objects, sampled, recorded)
                    row.update(objects=len(objects), max_error_m=error, object=worst)
                    if error > TOLERANCE_M:
                        row["released"] = explain_by_release(
                            loaded.env._task, objects, sampled, episode.recorded_task_state
                        )
                row["load_and_reset_s"] = round(time.perf_counter() - started, 3)
                rows.append(row)
    finally:
        environment.shutdown()
    return rows


def _measure_per_task(args: argparse.Namespace, tasks: list[str]) -> list[dict]:
    """Measure each task in its own process, at most ``--jobs`` at a time.

    Where objects settle into contact, the reset's outcome depends on what the CoppeliaSim
    process simulated before; a fresh process per task, playing episodes 0 to 24 in order, makes
    the result independent of ``--jobs`` and of the other tasks.
    """

    with tempfile.TemporaryDirectory(prefix="rlbench-initial-scenes-") as scratch:
        pending = list(tasks)
        running: list[tuple[Path, subprocess.Popen]] = []
        rows = []
        try:
            while pending or running:
                while pending and len(running) < args.jobs:
                    task = pending.pop(0)
                    output = Path(scratch) / f"{task}.json"
                    command = [sys.executable, "-m", __spec__.name, "--test-set", str(args.test_set)]
                    command += ["--xvfb", str(args.xvfb), "--rows", str(output), "--task", task]
                    with output.with_suffix(".log").open("wb") as log:
                        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=log)
                    running.append((output, process))
                finished = [item for item in running if item[1].poll() is not None]
                if not finished:
                    time.sleep(0.2)
                    continue
                for output, process in finished:
                    running.remove((output, process))
                    if process.returncode != 0:
                        log = output.with_suffix(".log").read_text(encoding="utf-8", errors="replace")
                        raise RuntimeError(
                            f"initial-scene check of {output.stem} exited with {process.returncode}:\n"
                            + log[-4000:]
                        )
                    rows.extend(json.loads(output.read_text(encoding="utf-8")))
        finally:
            for _, process in running:
                process.kill()
    order = {task: position for position, task in enumerate(TASK_NAMES)}
    return sorted(rows, key=lambda row: (order[row["task"]], row["episode"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test-set", type=Path, required=True)
    parser.add_argument("--xvfb", type=Path, required=True)
    parser.add_argument("--pins", type=Path, help="the frozen known-difference list")
    parser.add_argument("--task", action="append", choices=TASK_NAMES)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--report", type=Path, help="also write the JSON report here")
    parser.add_argument("--write-pins", action="store_true", help="write the measured list to --pins")
    parser.add_argument("--rows", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    tasks = args.task or list(TASK_NAMES)
    if args.rows is not None:
        args.rows.write_text(json.dumps(measure(args.test_set, args.xvfb, tasks)), encoding="utf-8")
        return 0
    if args.pins is None or args.jobs < 1:
        parser.error("--pins is required and --jobs must be positive")

    started = time.perf_counter()
    rows = _measure_per_task(args, tasks)
    elapsed = time.perf_counter() - started
    layout_errors = [row for row in rows if "layout_error" in row]
    differing = [_known_difference(row) for row in rows if row.get("max_error_m", 0.0) > TOLERANCE_M]
    report: dict = {
        "protocol": PROTOCOL_VERSION,
        "tolerance_m": TOLERANCE_M,
        "episodes": len(rows),
        "elapsed_s": round(elapsed, 1),
        "layout_errors": layout_errors,
        "differing": differing,
        "rows": rows,
    }
    failures = [f"{row['task']} episode {row['episode']}: {row['layout_error']}" for row in layout_errors]
    failures += [
        f"{row['task']} episode {row['episode']}: CoppeliaSim's object order is not the modelled one"
        for row in differing
        if row["cause"] == "unexplained_order"
    ]

    if args.write_pins:
        if failures or args.task:
            parser.error("--write-pins needs all tasks and a check without failures")
        pins = {
            "protocol": PROTOCOL_VERSION,
            "tolerance_m": TOLERANCE_M,
            "episodes_sha256": pinned_sha256(differing),
            "episodes": differing,
        }
        args.pins.write_text(json.dumps(pins, indent=2) + "\n", encoding="utf-8")
    else:
        pins = json.loads(args.pins.read_text(encoding="utf-8"))
        if pins["protocol"] != PROTOCOL_VERSION or pins["tolerance_m"] != TOLERANCE_M:
            failures.append(f"{args.pins} is for {pins['protocol']} at {pins['tolerance_m']} m")
        if pinned_sha256(pins["episodes"]) != pins["episodes_sha256"]:
            failures.append(f"{args.pins} does not match its episodes_sha256")
        pinned = {(row["task"], int(row["episode"])): row for row in pins["episodes"]}
        measured = {(row["task"], row["episode"]): row for row in differing}
        settling_tasks = {row["task"] for row in pins["episodes"] if "settling" in row["cause"]}
        report["unexpected"], report["settling_outside_list"] = [], []
        for key in sorted(set(measured) - set(pinned)):
            # Contact settling depends on the simulator's arithmetic; in a task known to settle,
            # an episode settling past the tolerance is reported, not failed.
            if key[0] in settling_tasks and "settling" in measured[key]["cause"]:
                report["settling_outside_list"].append(key)
            else:
                report["unexpected"].append(key)
                failures.append(
                    f"{key[0]} episode {key[1]} is more than {TOLERANCE_M} m off and not in {args.pins}"
                )
        report["pinned_within_tolerance"] = []
        for key in sorted(key for key in set(pinned) - set(measured) if key[0] in tasks):
            if pinned[key]["cause"] == "recorded_order":
                failures.append(
                    f"{key[0]} episode {key[1]}: pinned as recorded in another order, now within tolerance"
                )
            else:
                report["pinned_within_tolerance"].append(key)
        # Which objects were released before a frame was recorded is a property of the test set.
        for key in sorted(set(pinned) & set(measured)):
            causes = {pinned[key]["cause"], measured[key]["cause"]}
            if pinned[key].get("released") != measured[key].get("released"):
                message = (
                    f"{key[0]} episode {key[1]}: recorded order explained by releasing "
                    f"{measured[key].get('released')}, pinned {pinned[key].get('released')}"
                )
                if causes == {"recorded_order"}:
                    failures.append(message)
                else:
                    report.setdefault("settling_changed_explanation", []).append(message)
    report["failures"] = failures

    text = json.dumps(report, indent=2)
    if args.report is not None:
        args.report.write_text(text + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "rows"}
    print(json.dumps(summary, indent=2))
    if failures:
        print("RLBench initial-scene check failed:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
