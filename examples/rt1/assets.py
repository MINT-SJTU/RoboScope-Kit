# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Pinned RT-1-X checkpoint and language encoder for the SimplerEnv RT-1 example.

Both are what SimplerEnv's ``RT1Inference`` loads: the RT-1-X TF-Agents SavedModel that its
README downloads from the Open X-Embodiment bucket, and the Universal Sentence Encoder
(large, v5) that its default ``lang_embed_model_path`` points at on TF Hub. Every file the
Policy loads is checked against the SHA-256 below before TensorFlow reads it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Archive:
    name: str
    url: str
    filename: str
    size: int
    sha256: str
    directory: str
    files: dict[str, str]


RT1X = Archive(
    name="RT-1-X (rt_1_x_tf_trained_for_002272480_step)",
    url=(
        "https://storage.googleapis.com/gdm-robotics-open-x-embodiment/"
        "open_x_embodiment_and_rt_x_oss/rt_1_x_tf_trained_for_002272480_step.zip"
    ),
    filename="rt_1_x_tf_trained_for_002272480_step.zip",
    size=774_610_722,
    sha256="293fffb1331150d81a5676d4bfebe8334fc59750f247b880d0411b27f9b67ef0",
    directory="rt_1_x_tf_trained_for_002272480_step",
    files={
        "saved_model.pb": "ee418cab500da16a982fdddcc83a49357e5a48d5c44af1c30df8f52653f573f0",
        "policy_specs.pbtxt": "1193dce8ed8b0f962053bb42c90c25244ae5727697a4bd9ce586c287d87d37a0",
        "variables/variables.index": "8bffa4f5a44837b40ef20bafd6e80328cc9afc0005fb25ce2dbc8c79b44d992e",
        "variables/variables.data-00000-of-00001": (
            "2bbce09cb3d43ddef4dc7ae56f03755a777fa4e19350f6b6917df5157c0cc99d"
        ),
    },
)

UNIVERSAL_SENTENCE_ENCODER = Archive(
    name="Universal Sentence Encoder large v5",
    url="https://tfhub.dev/google/universal-sentence-encoder-large/5?tf-hub-format=compressed",
    filename="universal-sentence-encoder-large-5.tar.gz",
    size=550_093_654,
    sha256="2f5edf452025745e79822069fe8027ddbc682a06ddd8b2bf7b68bd48ab349b2e",
    directory="universal-sentence-encoder-large-5",
    files={
        "saved_model.pb": "2c81f1ed0cec0abb9fea0a48ec079f90342a79ae6b65a629cbb245d4f4dea461",
        "variables/variables.index": "3bebb13df81f2a406bc7cd2c212769707272eeae447f12b33e924b90587dd868",
        "variables/variables.data-00000-of-00001": (
            "233ff0b03aa6309672bc82e331dc10f04989fdd02968413502e7178ff0326f32"
        ),
    },
)

ARCHIVES = (RT1X, UNIVERSAL_SENTENCE_ENCODER)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_directory(root: Path, archive: Archive) -> Path:
    """Check every pinned file of ``archive`` under ``root`` and return its directory."""

    directory = root / archive.directory
    for relative, expected in archive.files.items():
        path = directory / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}; run examples/rt1/download_assets.py {root}")
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"{path} has SHA-256 {actual}, expected {expected}")
    return directory


__all__ = ["ARCHIVES", "RT1X", "UNIVERSAL_SENTENCE_ENCODER", "Archive", "file_sha256", "verify_directory"]
