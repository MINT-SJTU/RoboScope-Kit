"""Download the immutable public Go2 locomotion transfer checkpoint."""

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
    metadata = json.loads(Path(__file__).with_name("go2-checkpoint.json").read_text("utf-8"))
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / metadata["path"]
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        url = f"https://huggingface.co/{metadata['repository']}/resolve/{metadata['revision']}/{metadata['path']}"
        temporary = path.with_suffix(".pt.part")
        with urllib.request.urlopen(url, timeout=60) as source:
            temporary.write_bytes(source.read())
        if temporary.stat().st_size != metadata["bytes"]:
            raise RuntimeError("Go2 checkpoint size differs from its pin")
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != metadata["sha256"]:
            raise RuntimeError("Go2 checkpoint SHA-256 differs from its pin")
        temporary.replace(path)
    print(f"Verified Go2 checkpoint: {path}", flush=True)


if __name__ == "__main__":
    main()
