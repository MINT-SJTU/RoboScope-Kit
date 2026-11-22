# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Download the pinned PerAct 600k checkpoint and CLIP RN50 into one directory and verify them.

    python examples/peract/download_assets.py /models/peract

The directory is then the Policy's ``--checkpoint-source``. Every file is checked against the
SHA-256 pinned in ``assets.py``; an existing verified file is not downloaded again.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from assets import (  # noqa: E402
    CHECKPOINT_FILES,
    CHECKPOINT_URL,
    CHECKPOINT_ZIP_BYTES,
    CHECKPOINT_ZIP_SHA256,
    CLIP_RN50,
    CLIP_RN50_URL,
    file_sha256,
    verify,
)


def download(url: str, target: Path, sha256: str) -> None:
    if target.is_file() and file_sha256(target) == sha256:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as stream:
        while block := response.read(1 << 20):
            stream.write(block)
    actual = file_sha256(partial)
    if actual != sha256:
        partial.unlink()
        raise ValueError(f"{url} has SHA-256 {actual}, expected {sha256}")
    partial.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if not all((root / pinned.relative_path).is_file() for pinned in CHECKPOINT_FILES):
        with tempfile.TemporaryDirectory(dir=root.parent if root.parent.exists() else None) as scratch:
            archive = Path(scratch) / "peract_600k.zip"
            download(CHECKPOINT_URL, archive, CHECKPOINT_ZIP_SHA256)
            if archive.stat().st_size != CHECKPOINT_ZIP_BYTES:
                raise ValueError(f"{CHECKPOINT_URL} is not {CHECKPOINT_ZIP_BYTES} bytes")
            with zipfile.ZipFile(archive) as bundle:
                for pinned in CHECKPOINT_FILES:
                    member = pinned.relative_path.as_posix()
                    target = root / pinned.relative_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(member) as source, target.open("wb") as stream:
                        while block := source.read(1 << 20):
                            stream.write(block)
    download(CLIP_RN50_URL, root / CLIP_RN50.relative_path, CLIP_RN50.sha256)
    verify(root)
    print(f"PerAct assets verified in {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
