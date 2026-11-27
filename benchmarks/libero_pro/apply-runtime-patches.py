# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Apply checksum-pinned simulator source patches; called only by build-runtime.sh."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from importlib.metadata import distribution, version
from pathlib import Path
from typing import Any


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _apply_patch(pins_dir: Path, pin: dict[str, Any]) -> None:
    package = distribution(pin["distribution"])
    if package.version != pin["version"] or version("mujoco") != pin["mujoco_version"]:
        raise SystemExit("simulator patches require the pinned robosuite and MuJoCo versions")
    patch = pins_dir / pin["patch"]
    if _sha256(patch.read_bytes()) != pin["sha256"]:
        raise SystemExit(f"LIBERO-Pro simulator patch checksum mismatch: {patch}")
    target = Path(package.locate_file(pin["path"]))
    actual = _sha256(target.read_bytes())
    if actual == pin["patched_sha256"]:
        return
    if actual != pin["original_sha256"]:
        raise SystemExit(
            f"installed source differs from both pinned original and patched source: {pin['path']}"
        )
    with tempfile.TemporaryDirectory(prefix="roboscope-libero-patch-") as temporary:
        staged = Path(temporary) / pin["path"]
        staged.parent.mkdir(parents=True, exist_ok=True)
        # The PyPI robosuite 1.4.0 wheel ships CRLF line endings; the checked-in patch uses LF.
        staged.write_text(target.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=temporary, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=temporary, check=True)
        patched = staged.read_bytes()
    if _sha256(patched) != pin["patched_sha256"]:
        raise SystemExit(f"patched robosuite source checksum mismatch: {pin['path']}")
    # Atomic replacement keeps package-manager hardlinks and other runtimes intact.
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        replacement = Path(stream.name)
        stream.write(patched)
    try:
        replacement.chmod(target.stat().st_mode & 0o777)
        os.replace(replacement, target)
    finally:
        replacement.unlink(missing_ok=True)


def main() -> None:
    pins_path, report_path = map(Path, sys.argv[1:])
    manifest = json.loads(pins_path.read_text(encoding="utf-8"))
    for pin in manifest["patches"]:
        _apply_patch(pins_path.parent, pin)
        print(f"LIBERO-Pro simulator patch verified: {pin['path']}")
    report_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
