# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Pinned PerAct model assets: the released 600k checkpoint and the CLIP RN50 language encoder."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

# PerAct v1.0.0 release, "PerAct - 2048 Latents" (ID seed0): 18 tasks, 100 demos per task,
# 600k training iterations. Apache-2.0, like the PerAct source.
CHECKPOINT_URL = "https://github.com/peract/peract/releases/download/v1.0.0/peract_600k.zip"
CHECKPOINT_ZIP_SHA256 = "bf068b7b4cc81425824d631b528da050df9de582e9b4ea0b1df7fbc3868765cd"
CHECKPOINT_ZIP_BYTES = 123_791_087
# Inside the zip, and inside --checkpoint-source after extraction.
CHECKPOINT_SEED_DIR = Path("ckpts/multi/PERACT_BC/seed0")
CHECKPOINT_WEIGHT = 600000


@dataclass(frozen=True)
class PinnedFile:
    relative_path: Path
    sha256: str


CHECKPOINT_FILES = (
    PinnedFile(
        CHECKPOINT_SEED_DIR / "config.yaml",
        "9cb5b9291ed7bb7993c295b84a448e3062540b950a9e8787f4af2298aafd8b2e",
    ),
    PinnedFile(
        CHECKPOINT_SEED_DIR / "weights" / str(CHECKPOINT_WEIGHT) / "QAttentionAgent_layer0.pt",
        "765872a8df50a46c5f1b5bb1c3f69f033210829912324601c08c3f05869169e2",
    ),
)

# OpenAI CLIP RN50, the URL PerAct's helpers/clip/core/clip.py downloads (MIT). The URL path
# carries the file's SHA-256, which CLIP itself also checks.
CLIP_RN50_URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt"
)
CLIP_RN50 = PinnedFile(
    Path("clip/RN50.pt"), "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root: Path) -> None:
    """Raise unless ``root`` holds every pinned checkpoint file and CLIP RN50 with its SHA-256."""

    for pinned in (*CHECKPOINT_FILES, CLIP_RN50):
        path = root / pinned.relative_path
        if not path.is_file():
            raise FileNotFoundError(f"missing PerAct asset {path}; run download_assets.py")
        actual = file_sha256(path)
        if actual != pinned.sha256:
            raise ValueError(f"{path} has SHA-256 {actual}, expected {pinned.sha256}")
