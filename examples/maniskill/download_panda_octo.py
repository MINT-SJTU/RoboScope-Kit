"""Download the pinned ManiSkill fine-tuned Octo baseline and isolated v1.5 source."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tarfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

PIN_PATH = Path(__file__).with_name("panda-octo-checkpoint.json")


def verify(path: Path, record: dict) -> bool:
    if not path.is_file() or path.stat().st_size != record["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == record["sha256"]


def download(root: Path, pin: dict, tokenizer_source: Path | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    progress = {"bytes": 0, "last_report": started}
    lock = threading.Lock()

    def fetch(name: str, record: dict, url: str) -> None:
        target = root / name
        if verify(target, record):
            print(f"Verified {name}", flush=True)
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        print(f"Downloading {name}: {record['bytes'] / 1e6:.2f} MB", flush=True)
        with urllib.request.urlopen(url, timeout=120) as source, partial.open("wb") as output:
            while block := source.read(1024 * 1024):
                output.write(block)
                with lock:
                    progress["bytes"] += len(block)
                    now = time.monotonic()
                    if now - progress["last_report"] >= 20:
                        elapsed = now - started
                        print(f"Downloaded {progress['bytes'] / 1e6:.1f} MB in {elapsed:.0f}s "
                              f"({progress['bytes'] / max(elapsed, .001) / 1e6:.2f} MB/s)", flush=True)
                        progress["last_report"] = now
        if not verify(partial, record):
            raise ValueError(f"Downloaded bytes or SHA-256 differ: {name}")
        partial.replace(target)
        print(f"Completed {name}", flush=True)

    base = f"https://huggingface.co/{pin['repository']}/resolve/{pin['revision']}/{pin['subdirectory']}"
    print(f"Checkpoint download: {sum(x['bytes'] for x in pin['files'].values()) / 1e6:.1f} MB", flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(fetch, name, record, f"{base}/{name}")
                   for name, record in pin["files"].items()]
        for future in futures:
            future.result()
    for package in ("octo", "dlimp"):
        archive = pin[f"{package}_source_archive"]
        archive_name = f"{package}-source.tar.gz"
        fetch(archive_name, archive, archive["url"])
        source_root = root / f"{package}-source"
        source_root.mkdir(exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO((root / archive_name).read_bytes()), mode="r:gz") as bundle:
            for member in bundle.getmembers():
                parts = PurePosixPath(member.name).parts[1:]
                if not parts or member.isdir():
                    continue
                if ".." in parts or not member.isfile():
                    raise ValueError("Unexpected source archive member")
                target = source_root.joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        (source_root / ".source-sha256").write_text(archive["sha256"] + "\n", encoding="utf-8")
    tokenizer = pin["tokenizer"]
    cache_name = "models--" + tokenizer["cache_alias"].replace("/", "--")
    relative = Path("tokenizer-cache") / cache_name / "snapshots" / tokenizer["revision"]
    for name, record in tokenizer["files"].items():
        if tokenizer_source is not None:
            candidate = tokenizer_source / cache_name / "snapshots" / tokenizer["revision"] / name
            if verify(candidate, record):
                target = root / relative / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(candidate, target)
                continue
        fetch((relative / name).as_posix(), record,
              f"https://huggingface.co/{tokenizer['repository']}/resolve/{tokenizer['revision']}/{name}")
    refs = root / "tokenizer-cache" / cache_name / "refs"
    refs.mkdir(exist_ok=True)
    (refs / "main").write_text(tokenizer["revision"], encoding="utf-8")
    (root / PIN_PATH.name).write_text(json.dumps(pin, indent=2) + "\n", encoding="utf-8")
    print(f"Verified checkpoint and source in {root}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--tokenizer-source", type=Path, help="Reuse an already pinned tokenizer cache")
    arguments = parser.parse_args()
    download(arguments.output_dir.resolve(), json.loads(PIN_PATH.read_text(encoding="utf-8")),
             arguments.tokenizer_source)
