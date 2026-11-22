# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Download and verify the RT-1-X checkpoint and the Universal Sentence Encoder.

Usage::

    python examples/rt1/download_assets.py /models/rt1 [--archives-dir DIR]

Writes ``<root>/rt_1_x_tf_trained_for_002272480_step/`` and
``<root>/universal-sentence-encoder-large-5/``, the directory the Policy takes as
``--checkpoint-source``. ``--archives-dir`` reuses archives that are already downloaded;
every archive is checked against its pinned size and SHA-256 before it is extracted, and
every extracted file the Policy loads is checked again. Needs only the standard library.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from assets import ARCHIVES, RT1X, Archive, file_sha256, verify_directory  # noqa: E402


def fetch(archive: Archive, downloads: Path, archives_dir: Path | None) -> Path:
    if archives_dir is not None and (archives_dir / archive.filename).is_file():
        path = archives_dir / archive.filename
    else:
        path = downloads / archive.filename
        if not path.is_file():
            partial = path.with_suffix(path.suffix + ".part")
            print(f"downloading {archive.name} ({archive.size / 1e6:.0f} MB) from {archive.url}", flush=True)
            with urllib.request.urlopen(archive.url) as response, partial.open("wb") as handle:  # noqa: S310
                shutil.copyfileobj(response, handle, length=1 << 22)
            partial.replace(path)
    if path.stat().st_size != archive.size:
        raise ValueError(f"{path} has {path.stat().st_size} bytes, expected {archive.size}")
    actual = file_sha256(path)
    if actual != archive.sha256:
        raise ValueError(f"{path} has SHA-256 {actual}, expected {archive.sha256}")
    return path


def extract(archive: Archive, path: Path, root: Path) -> None:
    target = root / archive.directory
    if target.exists():
        shutil.rmtree(target)
    with tempfile.TemporaryDirectory(dir=root) as scratch:
        scratch_path = Path(scratch)
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as bundle:
                bundle.extractall(scratch_path)
            # The zip holds one top-level directory named like the checkpoint.
            (scratch_path / archive.directory).rename(target)
        else:
            with tarfile.open(path, "r:gz") as bundle:
                bundle.extractall(scratch_path / archive.directory, filter="data")
            (scratch_path / archive.directory).rename(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="directory to hold the extracted model files")
    parser.add_argument("--archives-dir", type=Path, help="reuse archives already downloaded here")
    args = parser.parse_args()
    root = args.root.resolve()
    downloads = root / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    for archive in ARCHIVES:
        try:
            verify_directory(root, archive)
            print(f"{archive.name}: already extracted and verified")
            continue
        except (FileNotFoundError, ValueError):
            pass
        path = fetch(archive, downloads, args.archives_dir)
        extract(archive, path, root)
        verify_directory(root, archive)
        print(f"{archive.name}: verified {path.name} and extracted {root / archive.directory}")
    print(f"--checkpoint-source {root}  (RT-1-X in {root / RT1X.directory})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
