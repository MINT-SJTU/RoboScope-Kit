"""Download immutable NEWT checkpoints and write a verified task-indexed bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--tasks", nargs="+", help="Native task IDs; defaults to all pinned tasks")
    args = parser.parse_args()
    catalog = json.loads(Path(__file__).with_name("newt-checkpoints.json").read_text("utf-8"))
    tasks = args.tasks or list(catalog["models"])
    unknown = set(tasks) - set(catalog["models"])
    if unknown:
        parser.error(f"No pinned NEWT model for: {', '.join(sorted(unknown))}")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    for task in tasks:
        record = catalog["models"][task]
        path = output / record["path"]
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]:
            print(f"Verified cached {task}", flush=True)
        else:
            url = f"https://huggingface.co/{catalog['repository']}/resolve/{catalog['revision']}/{record['path']}"
            print(f"Downloading {task}: {record['bytes'] / 1e6:.2f} MB", flush=True)
            started = time.monotonic()
            progress_at = started
            temporary = path.with_suffix(".pt.part")
            with urllib.request.urlopen(url, timeout=120) as source, temporary.open("wb") as target:
                while block := source.read(1024 * 1024):
                    target.write(block)
                    now = time.monotonic()
                    if now - progress_at >= 15:
                        rate = target.tell() / max(now - started, 0.001)
                        remaining = max(record["bytes"] - target.tell(), 0) / max(rate, 1)
                        print(f"{task}: {target.tell() / 1e6:.1f} MB, ETA {remaining:.0f} s", flush=True)
                        progress_at = now
            if temporary.stat().st_size != record["bytes"]:
                raise RuntimeError(f"{task} checkpoint size differs from its pin")
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != record["sha256"]:
                raise RuntimeError(f"{task} checkpoint SHA-256 differs from its pin")
            temporary.replace(path)
            print(f"Verified {task} in {time.monotonic() - started:.2f} s", flush=True)
        records[task] = dict(record)
    bundle = {**catalog, "models": records}
    (output / "models.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    print(f"Checkpoint bundle: {output / 'models.json'}", flush=True)


if __name__ == "__main__":
    main()
