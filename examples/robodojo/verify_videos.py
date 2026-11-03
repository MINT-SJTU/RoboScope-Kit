"""Validate every saved camera video after timing, including decoding each frame."""

import argparse
import json
import subprocess
from pathlib import Path


def verify_result(result, video_root):
    expected, frames = set(), 0
    for index, row in result["details"].items():
        tag = "success" if row["success"] else "fail"
        paths = set(video_root.glob(f"episode_{int(index):07d}_*_{tag}.mp4"))
        if len(paths) != 3:
            raise ValueError(f"Expected three camera videos: episode {index}, {paths}")
        for path in sorted(paths):
            info = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames",
                "-of", "json", str(path),
            ], text=True))
            stream, = info["streams"]
            if (stream["width"], stream["height"], stream["avg_frame_rate"], int(stream["nb_read_frames"])) != (
                640, 480, "25/1", row["executed_steps"] + 1,
            ):
                raise ValueError(f"Unexpected video format or frame count: {path}: {stream}")
            frames += int(stream["nb_read_frames"])
        expected.update(paths)
    actual = set(video_root.glob("*.mp4"))
    if actual != expected:
        raise ValueError(f"Unexpected or missing videos: {actual.symmetric_difference(expected)}")
    return {"episodes": len(result["details"]), "videos": len(expected), "decoded_frames": frames}


def verify_run(directory):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if not summary.get("complete"):
        raise ValueError(f"Incomplete run: {directory}")
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    rows = list(result["details"].values())
    expected = {int(row["layout_id"]): row for row in rows}
    if (len(expected) != len(rows) or len(rows) != summary["requested_episodes"]
            or sorted(expected) != list(range(len(rows)))):
        raise ValueError("Incomplete or duplicate fixed-layout outcomes")
    if "worker_results" not in summary:
        return {"run_id": summary["run_id"], **verify_result(result, Path(summary["result_path"]).parent)}
    workers = summary["worker_results"]
    if len(workers) != summary["k"]:
        raise ValueError("Worker outputs do not match declared concurrency")
    seen, records, paths = {}, [], set()
    for worker in workers:
        path = Path(worker["result_path"]).resolve()
        if path in paths:
            raise ValueError("Duplicate worker output path")
        paths.add(path)
        shard = json.loads(path.read_text(encoding="utf-8"))
        shard_rows = list(shard["details"].values())
        ids = [int(row["layout_id"]) for row in shard_rows]
        if len(set(ids)) != len(ids) or sorted(ids) != sorted(worker["layout_ids"]):
            raise ValueError("Missing or duplicated worker layouts")
        for row in shard_rows:
            key = int(row["layout_id"])
            if key in seen or expected.get(key) != row:
                raise ValueError("Worker results differ from the aggregate")
            seen[key] = row
        records.append(verify_result(shard, path.parent))
    if seen != expected:
        raise ValueError("Worker output union differs from the declared workload")
    return {"run_id": summary["run_id"], **{
        key: sum(record[key] for record in records) for key in ("episodes", "videos", "decoded_frames")
    }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="completed run evidence, or a completed series directory")
    args = parser.parse_args()
    series_path = args.directory / "series.json"
    if series_path.is_file():
        series = json.loads(series_path.read_text(encoding="utf-8"))
        if not series["complete"] or series["completed_episodes"] != series["requested_episodes"]:
            raise ValueError("Only a complete series can pass full-output validation")
        directories = [Path(row["summary_path"]).parent for row in series["runs"]]
        if len(set(directories)) != series["rounds"]:
            raise ValueError("Missing or repeated round evidence")
    else:
        directories = [args.directory]
    records = [verify_run(directory) for directory in directories]
    print(json.dumps({"complete": True, "runs": records}, indent=2))


if __name__ == "__main__":
    main()
