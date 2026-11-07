#!/usr/bin/env python3
"""Apply the two reviewed RoboCasa365 EGL cleanup patches to pinned sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

REQUIRED_PATCHES = {
    "owning-egl-context-cleanup": "robosuite/utils/binding_utils.py",
    "preserve-egl-thread-state": "robosuite/renderers/context/egl_context.py",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or not path.parts:
        raise ValueError(f"Invalid reviewed patch path: {value!r}")
    return path


def read_patches(path: Path) -> list[dict]:
    inputs = json.loads(path.read_text(encoding="utf-8"))
    if set(inputs) != {"schema_version", "patches"} or inputs["schema_version"] != 1:
        raise ValueError("Expected EGL cleanup patch schema 1")
    specs = inputs["patches"]
    if len(specs) != 2 or {spec["id"]: spec["path"] for spec in specs} != REQUIRED_PATCHES:
        raise ValueError("Both coupled EGL cleanup patches are required, without other runtime changes")
    sources = json.loads(path.with_name("source-inputs.json").read_text(encoding="utf-8"))["sources"]
    for spec in specs:
        if spec["source"] != "robosuite" or spec["revision"] != sources["robosuite"]["revision"]:
            raise ValueError("EGL cleanup patch source differs from the source lock")
        for key in ("patch_sha256", "original_sha256", "patched_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", spec.get(key, "")):
                raise ValueError(f"Missing reviewed {key} for {spec['id']}")
        relative_path(spec["path"])
        patch = path.parent.joinpath(*relative_path(spec["patch"]).parts)
        if patch.is_symlink() or sha256(patch) != spec["patch_sha256"]:
            raise RuntimeError(f"Reviewed EGL patch integrity check failed: {patch}")
    return specs


def preflight(root: Path, specs: list[dict]) -> None:
    record = root / "roboscope-runtime-patches.json"
    if record.is_symlink() or (
        record.exists() and json.loads(record.read_text(encoding="utf-8")) != {
            "schema_version": 1, "patches": specs,
        }
    ):
        raise RuntimeError("Existing runtime patch provenance differs; use a fresh prefix")
    for spec in specs:
        source = root / spec["source"]
        target = source.joinpath(*relative_path(spec["path"]).parts)
        if source.is_symlink() or target.is_symlink() or not target.resolve().is_relative_to(root):
            raise RuntimeError(f"EGL cleanup refuses a redirected source path: {target}")
        if target.exists() and sha256(target) not in {spec["original_sha256"], spec["patched_sha256"]}:
            raise RuntimeError(f"EGL cleanup refuses an unknown source file: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=Path(__file__).with_name("runtime-patches.json"))
    parser.add_argument("--check-only", action="store_true", help="Read-only integrity/provenance preflight")
    args = parser.parse_args()
    inputs = args.inputs.resolve()
    specs = read_patches(inputs)
    root = args.source_root.resolve()
    preflight(root, specs)
    if args.check_only:
        print("Reviewed EGL cleanup preflight passed", flush=True)
        return
    for spec in specs:
        source = root / spec["source"]
        target = source.joinpath(*relative_path(spec["path"]).parts)
        if sha256(target) == spec["patched_sha256"]:
            print(f"Verified existing EGL cleanup patch: {spec['path']}", flush=True)
            continue
        patch = inputs.parent.joinpath(*relative_path(spec["patch"]).parts)
        command = ("git", "-c", "core.autocrlf=false", "-c", "core.eol=lf", "apply", "--whitespace=error-all")
        subprocess.run((*command, "--check", str(patch)), cwd=source, check=True)
        subprocess.run((*command, str(patch)), cwd=source, check=True)
        if sha256(target) != spec["patched_sha256"]:
            raise RuntimeError(f"EGL cleanup produced an unexpected source file: {target}")
        print(f"Applied reviewed EGL cleanup patch: {spec['path']}", flush=True)
    record = root / "roboscope-runtime-patches.json"
    temporary = record.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"schema_version": 1, "patches": specs}, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(record)


if __name__ == "__main__":
    main()
