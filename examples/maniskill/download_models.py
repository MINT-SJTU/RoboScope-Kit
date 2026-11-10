"""Download selected official PPO checkpoints and write a models.json bundle.

Uses immutable Hugging Face dataset revisions and verifies every checkpoint's
published size and SHA-256. No demonstrations or training data are downloaded.
"""

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
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--tasks", nargs="+", help="Task IDs; defaults to the three baseline models")
    selection.add_argument("--all", action="store_true", help="Download every pinned official state model")
    args = parser.parse_args()
    catalog = json.loads(Path(__file__).with_name("official-models.json").read_text(encoding="utf-8"))
    tasks = list(catalog["models"]) if args.all else args.tasks or catalog["default_tasks"]
    unknown = set(tasks) - set(catalog["models"])
    if unknown:
        parser.error(f"No pinned example checkpoint for: {', '.join(sorted(unknown))}")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    for task in tasks:
        record = catalog["models"][task]
        path = output / record["path"]
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]:
            print(f"Verified cached {task}: {path}", flush=True)
        else:
            url = (
                f"https://huggingface.co/datasets/{catalog['repository']}/resolve/"
                f"{catalog['revision']}/{record['dataset_path']}"
            )
            print(f"Downloading {task}: {record['bytes'] / 1e6:.2f} MB", flush=True)
            started = time.perf_counter()
            temporary = path.with_suffix(".pt.part")
            with urllib.request.urlopen(url, timeout=120) as source, temporary.open("wb") as target:
                while block := source.read(1024 * 1024):
                    target.write(block)
            if temporary.stat().st_size != record["bytes"]:
                raise RuntimeError(f"Checkpoint size differs from pinned metadata: {task}")
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != record["sha256"]:
                raise RuntimeError(f"Checkpoint SHA-256 differs from pinned metadata: {task}")
            temporary.replace(path)
            print(f"Verified {task} in {time.perf_counter() - started:.2f} s", flush=True)
        records[task] = dict(record)
    bundle = {
        "schema_version": 1,
        "repository": catalog["repository"],
        "revision": catalog["revision"],
        "models": records,
    }
    (output / "models.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    print(f"Checkpoint bundle: {output / 'models.json'}", flush=True)


if __name__ == "__main__":
    main()
