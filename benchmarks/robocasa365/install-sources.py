#!/usr/bin/env python3
"""Install hash-verified canonical source snapshots without requiring Git transport."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

RECORD_NAME = ".roboscope-source.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_inputs(path: Path) -> dict:
    inputs = json.loads(path.read_text(encoding="utf-8"))
    if inputs.get("schema_version") != 1 or set(inputs.get("sources", {})) != {"robocasa", "robosuite"}:
        raise ValueError("Expected RoboCasa365 source input schema 1 with both simulator sources")
    for name, spec in inputs["sources"].items():
        if not re.fullmatch(r"[0-9a-f]{40}", spec.get("revision", "")):
            raise ValueError(f"Invalid source revision for {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", spec.get("sha256") or ""):
            raise ValueError(f"Missing reviewed source archive SHA256 for {name}")
        if not isinstance(spec.get("bytes"), int) or spec["bytes"] <= 0:
            raise ValueError(f"Missing source archive byte count for {name}")
        if Path(spec["archive"]).name != spec["archive"] or "/" in spec["prefix"]:
            raise ValueError(f"Invalid archive filename or prefix for {name}")
    return inputs


def verify_source(destination: Path, spec: dict) -> None:
    marker = destination / RECORD_NAME
    if not marker.is_file():
        raise RuntimeError(
            f"Unverified existing source directory: {destination}; move it aside before building"
        )
    record = json.loads(marker.read_text(encoding="utf-8"))
    if record.get("source") != spec:
        raise RuntimeError(f"Source pin or archive differs for {destination}")
    patch_inputs = json.loads(Path(__file__).with_name("runtime-patches.json").read_text(encoding="utf-8"))
    if patch_inputs.get("schema_version") != 1:
        raise ValueError("Expected EGL cleanup patch schema 1")
    reviewed = {
        patch["path"]: patch for patch in patch_inputs["patches"]
        if patch["source"] == destination.name and patch["revision"] == spec["revision"]
    }
    if reviewed.keys() - record["files"].keys():
        raise RuntimeError(
            f"Reviewed patch targets are absent from the original source record: {destination}"
        )
    for relative, digest in record["files"].items():
        path = destination / relative
        allowed = {digest}
        if relative in reviewed:
            patch = reviewed[relative]
            if digest != patch["original_sha256"]:
                raise RuntimeError(f"Reviewed patch original hash differs from its source archive: {path}")
            allowed.add(patch["patched_sha256"])
        if not path.is_file() or path.is_symlink() or sha256(path) not in allowed:
            raise RuntimeError(f"Pinned source file changed or disappeared: {path}")
    # Editable builds may add egg-info / bytecode. Official asset ZIPs also
    # contain texture-generation utility scripts; their separate ZIP lock owns
    # additions below models/assets. Original archive files remain checked above.
    for directory, children, files in os.walk(destination):
        if Path(directory).relative_to(destination).as_posix() == "robocasa/models/assets":
            children[:] = []
            continue
        children[:] = [item for item in children if item not in {"__pycache__", ".git"}]
        for filename in files:
            if filename.endswith((".py", ".pyi", ".so", ".pyd")):
                relative = (Path(directory) / filename).relative_to(destination).as_posix()
                if relative not in record["files"]:
                    raise RuntimeError(f"Unexpected code file in pinned source: {relative}")
    print(f"Verified unchanged source snapshot: {destination}", flush=True)


def obtain_archive(spec: dict, cache: Path) -> Path:
    archive = cache / spec["archive"]
    if not archive.exists():
        partial = archive.with_suffix(archive.suffix + ".partial")
        print(f"Downloading pinned source: {spec['url']}", flush=True)
        try:
            request = urllib.request.Request(spec["url"], headers={"User-Agent": "RoboScope-Kit"})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
            if partial.stat().st_size != spec["bytes"] or sha256(partial) != spec["sha256"]:
                raise RuntimeError(f"Downloaded source archive failed integrity verification: {partial}")
            partial.replace(archive)
        finally:
            partial.unlink(missing_ok=True)
    if archive.stat().st_size != spec["bytes"] or sha256(archive) != spec["sha256"]:
        raise RuntimeError(f"Cached source archive failed integrity verification: {archive}")
    return archive


def install_source(archive: Path, destination: Path, spec: dict) -> None:
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-extract-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary)
        with tarfile.open(archive, "r:gz") as bundle:
            file_names = []
            seen = set()
            for member in bundle.getmembers():
                relative = PurePosixPath(member.name)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not relative.parts
                    or relative.parts[0] != spec["prefix"]
                    or "\\" in member.name
                    or not (member.isdir() or member.isfile())
                    or relative.as_posix() in seen
                ):
                    raise RuntimeError(f"Unexpected source archive entry: {member.name}")
                seen.add(relative.as_posix())
                if member.isfile():
                    file_names.append(PurePosixPath(*relative.parts[1:]).as_posix())
            bundle.extractall(staging, filter="data")
        extracted = staging / spec["prefix"]
        record = {"source": spec, "files": {name: sha256(extracted / name) for name in file_names}}
        (extracted / RECORD_NAME).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        extracted.rename(destination)
    print(f"Installed verified source snapshot: {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path(__file__).with_name("source-inputs.json"))
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    inputs = read_inputs(args.inputs)
    if args.check_inputs:
        for name, spec in inputs["sources"].items():
            destination = args.source_root / name
            if destination.is_symlink():
                raise RuntimeError(f"Source preflight refuses a redirected directory: {destination}")
            if destination.exists():
                verify_source(destination, spec)
        print("Source archive pins and existing source snapshots verified")
        return
    args.source_root.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in inputs["sources"].items():
        destination = args.source_root / name
        if destination.exists():
            verify_source(destination, spec)
        else:
            install_source(obtain_archive(spec, args.cache_dir), destination, spec)


if __name__ == "__main__":
    main()
