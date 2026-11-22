#!/usr/bin/env python3
"""Download, verify and extract the fixed RoboCasa365 assets.

Normal operation rejects an incomplete checksum lock before downloading. For
maintainers bootstrapping a new upstream asset revision, --download-only with
--record-checksums FILE records candidate hashes without extracting anything or
marking the runtime ready. Review and commit those hashes before a normal build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def obtain_archive(name: str, spec: dict, cache: Path) -> tuple[Path, str]:
    archive = cache / f"{name}.zip"
    expected_bytes = spec["bytes"]
    expected_digest = spec["sha256"]
    if archive.exists() and archive.stat().st_size != expected_bytes:
        raise RuntimeError(f"Cached archive has wrong size: {archive}; remove it and retry")
    if not archive.exists():
        partial = archive.with_suffix(".zip.partial")
        print(f"Downloading {name} ({expected_bytes:,} bytes)", flush=True)
        # Box rejects HEAD even when a GET works; do not preflight with HEAD.
        request = urllib.request.Request(spec["url"], headers={"User-Agent": "RoboScope-Kit"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
            if partial.stat().st_size != expected_bytes:
                raise RuntimeError(f"Downloaded size mismatch for {name}: {partial.stat().st_size}")
            partial.replace(archive)
        finally:
            partial.unlink(missing_ok=True)
    digest = sha256(archive)
    if expected_digest is not None and digest != expected_digest:
        raise RuntimeError(f"SHA256 mismatch for {archive}: expected {expected_digest}, got {digest}")
    print(f"Verified {name}: {digest}", flush=True)
    return archive, digest


def extract_archive(archive: Path, destination: Path, root: str, ignored_entries: list[str]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / root / ".roboscope-asset-sha256"
    digest = sha256(archive)
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == digest:
        print(f"Reusing verified extracted assets: {destination / root}", flush=True)
        return
    # Validate the archive layout before writing any entries. The fixed ZIPs have
    # one runtime directory plus explicitly pinned attribution metadata. Reject
    # traversal and links even for ignored entries; never extract that metadata.
    with zipfile.ZipFile(archive) as bundle:
        members = []
        ignored = set(ignored_entries)
        seen_ignored = set()
        for entry in bundle.infolist():
            relative = PurePosixPath(entry.filename)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not relative.parts
                or "\\" in entry.filename
                or stat.S_ISLNK(entry.external_attr >> 16)
            ):
                raise RuntimeError(f"Unexpected ZIP entry in {archive}: {entry.filename}")
            if entry.filename in ignored:
                if entry.is_dir() or stat.S_IFMT(entry.external_attr >> 16) not in (0, stat.S_IFREG):
                    raise RuntimeError(f"Ignored metadata must be a regular file: {entry.filename}")
                seen_ignored.add(entry.filename)
                continue
            if relative.parts[0] != root:
                raise RuntimeError(f"Unexpected ZIP entry in {archive}: {entry.filename}")
            members.append(entry)
        if seen_ignored != ignored:
            raise RuntimeError(
                f"Pinned metadata entries are missing from {archive}: {ignored - seen_ignored}"
            )
        print(f"Extracting {archive.name} to {destination}", flush=True)
        bundle.extractall(destination, members=members)
    marker.write_text(digest + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--inputs", type=Path, default=Path(__file__).with_name("asset-inputs.json"))
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--record-checksums", type=Path)
    args = parser.parse_args()
    lock = json.loads(args.inputs.read_text(encoding="utf-8"))
    if lock["schema_version"] != 1:
        raise ValueError("Unsupported asset input schema")
    bootstrap = args.download_only and args.record_checksums is not None
    if args.record_checksums is not None and not args.download_only:
        parser.error("--record-checksums requires --download-only; candidate assets cannot be extracted")
    if args.record_checksums is not None and args.record_checksums.resolve() == args.inputs.resolve():
        parser.error("Record candidate checksums to a separate file; do not overwrite the reviewed lock")
    for name, spec in lock["archives"].items():
        if not re.fullmatch(r"[a-z_]+", name) or not isinstance(spec["bytes"], int) or spec["bytes"] <= 0:
            raise ValueError(f"Invalid archive input: {name}")
        ignored = spec.get("ignored_entries", [])
        if not isinstance(ignored, list) or any(
            not isinstance(entry, str)
            or len(PurePosixPath(entry).parts) != 1
            or "\\" in entry
            or not entry.endswith(".md")
            for entry in ignored
        ):
            raise ValueError(f"Invalid explicit metadata exclusions for {name}")
        digest = spec.get("sha256")
        if digest is None and bootstrap:
            continue
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(
                f"Missing reviewed SHA256 for {name}; complete asset-inputs.json before building"
            )
    asset_root = args.source_root / "robocasa" / "models" / "assets"
    if not asset_root.is_dir():
        raise FileNotFoundError(f"Pinned RoboCasa source is missing: {asset_root}")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in lock["archives"].items():
        archive, digest = obtain_archive(name, spec, args.cache_dir)
        spec["sha256"] = digest
        if not args.download_only:
            extract_archive(
                archive, asset_root / spec["destination"], spec["root"], spec.get("ignored_entries", [])
            )
    if args.record_checksums is not None:
        args.record_checksums.parent.mkdir(parents=True, exist_ok=True)
        args.record_checksums.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
        print(f"Candidate checksum record only: {args.record_checksums}; assets have not been extracted")
    if not args.download_only:
        record = args.source_root / "roboscope-assets-provenance.json"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=record.parent, delete=False) as output:
            json.dump(lock, output, indent=2)
            output.write("\n")
            temporary = Path(output.name)
        os.replace(temporary, record)
        print(f"Verified all six asset archives: {record}")


if __name__ == "__main__":
    main()
