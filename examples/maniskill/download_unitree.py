"""Download pinned official Unitree TorchScript actors and a model bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    catalog = json.loads(Path(__file__).with_name("unitree-checkpoints.json").read_text("utf-8"))
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for task, record in catalog["models"].items():
        path = output / record["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            url = (
                f"https://raw.githubusercontent.com/{catalog['repository']}/"
                f"{catalog['source_revision']}/{record['source_path']}"
            )
            temporary = path.with_suffix(".pt.part")
            with urllib.request.urlopen(url, timeout=60) as source:
                temporary.write_bytes(source.read())
            if temporary.stat().st_size != record["bytes"]:
                raise RuntimeError(f"{task} checkpoint size differs from its pin")
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != record["sha256"]:
                raise RuntimeError(f"{task} checkpoint SHA-256 differs from its pin")
            temporary.replace(path)
        print(f"Verified {task}: {path}", flush=True)
    (output / "models.json").write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
