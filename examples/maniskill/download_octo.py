"""Download the pinned public Octo-small model and its offline T5 tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path


def verify(path: Path, record: dict) -> bool:
    if not path.is_file() or path.stat().st_size != record["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == record["sha256"]


def download(root: Path, repository: str, revision: str, files: dict) -> None:
    for name, record in files.items():
        target = root / name
        if verify(target, record):
            print(f"Verified {name}", flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".part")
        url = f"https://huggingface.co/{repository}/resolve/{revision}/{name}"
        print(f"Downloading {name} ({record['bytes'] / 1e6:.2f} MB)", flush=True)
        with urllib.request.urlopen(url, timeout=120) as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        if not verify(temporary, record):
            raise RuntimeError(f"Downloaded file differs from its pinned bytes or SHA-256: {name}")
        temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    pin_path = Path(__file__).with_name("octo-checkpoint.json")
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    download(output, pin["repository"], pin["revision"], pin["files"])
    tokenizer = pin["tokenizer"]
    cache = output / "tokenizer-cache" / ("models--" + tokenizer["cache_alias"].replace("/", "--"))
    download(
        cache / "snapshots" / tokenizer["revision"], tokenizer["repository"],
        tokenizer["revision"], tokenizer["files"],
    )
    (cache / "refs").mkdir(exist_ok=True)
    (cache / "refs" / "main").write_text(tokenizer["revision"], encoding="utf-8")
    (output / "octo-checkpoint.json").write_bytes(pin_path.read_bytes())
    print(f"Pinned Octo checkpoint and offline tokenizer: {output}", flush=True)


if __name__ == "__main__":
    main()
