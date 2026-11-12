# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Install checksum-pinned public model source archives without invoking Git."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

SOURCE_NAMES = ("diffusion_policy", "robomimic")
RECORD_NAME = ".roboscope-source.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pinned_sources() -> dict[str, dict[str, str]]:
    inputs = json.loads(
        Path(__file__).resolve().with_name("model-inputs.json").read_text()
    )
    sources = {}
    for name in SOURCE_NAMES:
        source = inputs[name]
        revision = source["revision"]
        digest = source.get("archive_sha256", "")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"{name} requires a complete source commit pin")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{name} requires a committed source archive SHA-256")
        expected_url = (
            f"https://codeload.github.com/robocasa-benchmark/{name}/tar.gz/{revision}"
        )
        if source.get("archive_url") != expected_url:
            raise ValueError(f"{name} requires its canonical pinned codeload URL")
        sources[name] = {
            "repository": source["repository"],
            "revision": revision,
            "archive_url": expected_url,
            "archive_sha256": digest,
        }
    return sources


def cached_archive(name: str, source: dict[str, str], cache: Path) -> Path:
    archive = cache / f"{name}-{source['revision']}.tar.gz"
    if not archive.is_file():
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{name}-", suffix=".download", dir=cache, delete=False
            ) as out:
                temporary = Path(out.name)
                with urllib.request.urlopen(
                    source["archive_url"], timeout=60
                ) as response:
                    shutil.copyfileobj(response, out)
            if file_sha256(temporary) != source["archive_sha256"]:
                raise ValueError(f"downloaded source checksum mismatch: {name}")
            temporary.replace(archive)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    if file_sha256(archive) != source["archive_sha256"]:
        raise ValueError(f"cached source checksum mismatch: {archive}")
    return archive


def archive_members(
    archive: tarfile.TarFile, name: str, revision: str
) -> list[tuple[Any, Path]]:
    members = []
    expected_root = f"{name}-{revision}"
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or "\\" in member.name:
            raise ValueError(f"unsafe source archive path: {member.name}")
        if not path.parts or path.parts[0] != expected_root:
            raise ValueError(f"unexpected source archive root: {member.name}")
        if not member.isdir() and not member.isfile():
            raise ValueError(
                f"source archive contains a link or special file: {member.name}"
            )
        if len(path.parts) == 1:
            continue
        members.append((member, Path(*path.parts[1:])))
    return members


def verify_source(
    target: Path, archive: tarfile.TarFile, members: list[tuple[Any, Path]]
) -> None:
    for member, relative in members:
        destination = target / relative
        if any(
            path.is_symlink()
            for path in (destination, *destination.parents)
            if path != target.parent
        ):
            raise ValueError(f"source checkout contains a symlink: {destination}")
        if member.isdir():
            if not destination.is_dir():
                raise ValueError(f"source directory is missing: {destination}")
            continue
        original = archive.extractfile(member)
        if original is None or not destination.is_file():
            raise ValueError(f"source file is missing: {destination}")
        with original:
            expected_digest = hashlib.file_digest(original, "sha256").hexdigest()
        if file_sha256(destination) != expected_digest:
            raise ValueError(
                f"source file differs from its pinned archive: {destination}"
            )


def install_source(
    name: str, source: dict[str, str], archive_path: Path, install: Path
) -> None:
    target = install / name
    record = {"schema_version": 1, **source}
    if target.is_symlink():
        raise ValueError(f"source destination must be a regular directory: {target}")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive_members(archive, name, source["revision"])
        if target.exists():
            provenance = target / RECORD_NAME
            if not provenance.is_file() or json.loads(provenance.read_text()) != record:
                raise ValueError(
                    f"existing source lacks matching archive provenance: {target}"
                )
            verify_source(target, archive, members)
            print(f"Verified existing pinned source: {target}")
            return
        staging = Path(tempfile.mkdtemp(prefix=f".source-{name}-", dir=install))
        try:
            for member, relative in members:
                destination = staging / relative
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                original = archive.extractfile(member)
                if original is None:
                    raise ValueError(
                        f"source archive file cannot be read: {member.name}"
                    )
                with original, destination.open("wb") as out:
                    shutil.copyfileobj(original, out)
                destination.chmod(member.mode & 0o777)
            verify_source(staging, archive, members)
            (staging / RECORD_NAME).write_text(json.dumps(record, indent=2) + "\n")
            staging.replace(target)
            print(f"Installed pinned source: {target}")
        finally:
            # Delete only this helper's private staging directory inside <install-dir>.
            resolved = staging.resolve()
            if resolved.parent != install.resolve() or not resolved.name.startswith(
                f".source-{name}-"
            ):
                raise ValueError(f"unsafe source staging cleanup path: {resolved}")
            if staging.exists():
                shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("install_dir", type=Path)
    parser.add_argument("--archives-dir", type=Path)
    args = parser.parse_args()
    sources = pinned_sources()
    install = args.install_dir.resolve()
    cache = (args.archives_dir or install / "source-archives").resolve()
    install.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        archive = cached_archive(name, source, cache)
        install_source(name, source, archive, install)


if __name__ == "__main__":
    main()
