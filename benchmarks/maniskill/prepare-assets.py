#!/usr/bin/env python3
"""Plan or install only the assets required by selected ManiSkill suites.

MS_ASSET_DIR is selected before importing the pinned upstream registry. Downloads
use the upstream asset sources and checksum verification without changing them.
The default baseline suite uses bundled robots and procedural objects only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import socket
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def tree_digest(root: Path) -> dict[str, Any]:
    entries = []
    size = 0
    paths = [root] if root.is_file() else sorted(root.rglob("*"))
    for path in paths:
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            size += path.stat().st_size
            relative = path.name if path == root else path.relative_to(root).as_posix()
            entries.append((relative, digest))
    payload = json.dumps(entries, separators=(",", ":")).encode()
    result = {
        "files": len(entries), "bytes": size,
        "tree_sha256": hashlib.sha256(payload).hexdigest(),
    }
    if root.is_file():
        result["file_sha256"] = entries[0][1]
    return result


def asset_present(path: Path) -> bool:
    return (path.is_file() and path.stat().st_size > 0) or (
        path.is_dir() and any(path.iterdir())
    )


@contextmanager
def archive_member_source(source, upstream, override, archive, record):
    """Recover an unchanged upstream file from a checksum-pinned author bundle."""
    member_name = override.get("archive_member")
    if member_name is None:
        yield source
        return
    with tempfile.TemporaryDirectory(prefix="maniskill-asset-") as temporary:
        directory = Path(temporary)
        bundle = archive if archive is not None else directory / "bundle.tar.gz"
        if not bundle.is_file():
            bundle.parent.mkdir(parents=True, exist_ok=True)
            last_report = [0.0]

            def progress(blocks, block_size, total):
                now = time.monotonic()
                if now - last_report[0] >= 15:
                    print(f"Asset mirror download: {blocks * block_size}/{total} bytes", flush=True)
                    last_report[0] = now

            urllib.request.urlretrieve(override["url"], bundle, reporthook=progress)
        with bundle.open("rb") as stream:
            bundle_digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if bundle_digest != override["sha256"]:
            raise ValueError(f"Asset mirror bundle checksum differs: {bundle}")
        filename = Path(urllib.parse.urlsplit(upstream.url).path).name
        recovered = directory / filename
        with tarfile.open(bundle) as container:
            member = container.getmember(member_name)
            if not member.isfile() or member.size != override["member_bytes"]:
                raise ValueError("Asset mirror member does not match its pinned size/type")
            with container.extractfile(member) as reader, recovered.open("wb") as writer:
                shutil.copyfileobj(reader, writer)
        with recovered.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != override["member_sha256"] or digest != upstream.checksum:
            raise ValueError("Recovered asset differs from the original upstream checksum")
        record.update({
            "mirror_archive_sha256": bundle_digest, "mirror_archive_bytes": bundle.stat().st_size,
            "mirror_member": member_name, "member_sha256": digest,
            "member_bytes": recovered.stat().st_size, "original_archive_unchanged": True,
        })
        original = copy.copy(upstream)
        original.url = recovered.as_uri()
        original.filename = upstream.filename or filename
        yield original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument(
        "--archive-dir", type=Path,
        help="Reuse <asset UID with / replaced by __> plus the source filename suffixes",
    )
    parser.add_argument("--suite", action="append", default=[])
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--estimate", action="store_true", help="Query archive sizes with HTTP HEAD")
    parser.add_argument("--list-tasks", action="store_true", help="List the pinned upstream registry")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.asset_dir = args.asset_dir.expanduser().resolve()
    os.environ["MS_ASSET_DIR"] = str(args.asset_dir)

    import mani_skill.envs  # noqa: F401 -- populate pinned task and asset registries
    from mani_skill.utils import assets, download_asset
    from mani_skill.utils.registration import REGISTERED_ENVS

    if args.list_tasks:
        print(json.dumps({
            uid: {"max_episode_steps": spec.max_episode_steps,
                  "asset_download_ids": spec.asset_download_ids}
            for uid, spec in sorted(REGISTERED_ENVS.items())
        }, indent=2))
        return
    manifest = json.loads(args.benchmark_manifest.read_text(encoding="utf-8"))
    protocol = manifest["benchmark_protocols"]["maniskill"]
    profiles = protocol["profiles"]
    selected: set[str] = set()
    for suite in args.suite or ([] if args.task_id else ["baseline_state_v1"]):
        selected.update(protocol["suites"][suite])
    for task_id in args.task_id:
        matches = {key for key, value in profiles.items() if value["task_id"] == task_id}
        if not matches:
            raise ValueError(f"Task is not declared in this manifest: {task_id}")
        selected.update(matches)
    source_ids: set[str] = set()
    for key in sorted(selected):
        profile = profiles[key]
        task_id = profile["task_id"]
        if task_id not in REGISTERED_ENVS:
            raise ValueError(f"Task is not registered in the pinned ManiSkill source: {task_id}")
        requirements = [
            *(REGISTERED_ENVS[task_id].asset_download_ids or []),
            *profile.get("asset_download_ids", []),
        ]
        for uid in requirements:
            if uid in assets.DATA_SOURCES:
                source_ids.add(uid)
            else:
                source_ids.update(assets.expand_data_group_into_individual_data_source_ids(uid))
    records = []
    started = time.perf_counter()
    if args.download:
        args.asset_dir.mkdir(parents=True, exist_ok=True)
    for uid in sorted(source_ids):
        upstream_source = assets.DATA_SOURCES[uid]
        source = copy.copy(upstream_source)
        override = protocol.get("asset_sources", {}).get(uid, {})
        if override:
            source.url = override["url"]
            source.checksum = override["sha256"]
        target = Path(source.output_dir) / source.target_path
        record: dict[str, Any] = {
            "uid": uid, "url": source.url,
            "upstream_url": upstream_source.url, "upstream_sha256": upstream_source.checksum,
            "expected_archive_sha256": source.checksum,
            "source_revision": override.get("revision"),
            "expected_archive_bytes": override.get("bytes"),
            "target": str(target), "present": asset_present(target),
            "archive_cache_used": False,
        }
        if args.estimate and not record["present"] and source.url:
            try:
                request = urllib.request.Request(source.url, method="HEAD")
                with urllib.request.urlopen(request, timeout=20) as response:
                    length = response.headers.get("Content-Length")
                    record["download_bytes"] = int(length) if length else None
            except (OSError, ValueError) as error:
                record["size_query_error"] = str(error)
        if args.download and not record["present"]:
            download_source = source
            source_filename = (
                Path(urllib.parse.urlsplit(source.url).path).name if source.url else None
            )
            archive = (
                args.archive_dir / (
                    uid.replace("/", "__") + "".join(Path(source_filename).suffixes)
                ) if args.archive_dir is not None and source_filename else None
            )
            if archive is not None and archive.is_file():
                digest = hashlib.sha256(archive.read_bytes()).hexdigest()
                if source.checksum is not None and digest != source.checksum:
                    raise ValueError(f"Cached archive checksum differs from the pinned source: {archive}")
                download_source = copy.copy(source)
                download_source.url = archive.resolve().as_uri()
                download_source.filename = source.filename or source_filename
                record.update({
                    "archive_cache_used": True, "archive_path": str(archive.resolve()),
                    "archive_sha256": digest, "archive_bytes": archive.stat().st_size,
                })
            print(f"Installing ManiSkill asset {uid}: {download_source.url}", flush=True)
            previous_timeout = socket.getdefaulttimeout()
            try:
                socket.setdefaulttimeout(60)
                with archive_member_source(
                    download_source, upstream_source, override, archive, record,
                ) as resolved_source:
                    download_asset.download(resolved_source, non_interactive=True)
            except (OSError, ValueError) as error:
                record.update({"availability": "download_failed", "download_error": str(error)})
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps({
                        "source_commit": protocol["maniskill_source_commit"],
                        "manifest_sha256": hashlib.sha256(
                            args.benchmark_manifest.read_bytes()).hexdigest(),
                        "asset_dir": str(args.asset_dir), "profiles": sorted(selected),
                        "download_requested": True, "assets": [*records, record],
                        "completed": False, "elapsed_s": time.perf_counter() - started,
                    }, indent=2) + "\n", encoding="utf-8")
                raise RuntimeError(
                    f"Asset {uid!r} download failed from {source.url}. "
                    "Place a verified archive in --archive-dir "
                    "(MANISKILL_ASSET_ARCHIVE_DIR for build-runtime.sh) and rerun. "
                    f"Original error: {error}"
                ) from error
            finally:
                socket.setdefaulttimeout(previous_timeout)
            record["present"] = asset_present(target)
            if not record["present"]:
                raise RuntimeError(f"Asset download did not populate {target}")
        if args.download and record["present"]:
            record.update(tree_digest(target))
        record["availability"] = "installed" if record["present"] else "not_installed"
        records.append(record)
    result = {
        "source_commit": protocol["maniskill_source_commit"],
        "manifest_sha256": hashlib.sha256(args.benchmark_manifest.read_bytes()).hexdigest(),
        "asset_dir": str(args.asset_dir), "profiles": sorted(selected),
        "download_requested": args.download, "assets": records,
        "completed": True,
        "known_missing_download_bytes": sum(
            row.get("download_bytes") or 0 for row in records if not row["present"]
        ),
        "unknown_missing_archive_sizes": sum(
            not row["present"] and row.get("download_bytes") is None for row in records
        ),
        "elapsed_s": time.perf_counter() - started,
        "note": "Bundled assets are pinned by the source archive; external sources retain upstream terms.",
    }
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
