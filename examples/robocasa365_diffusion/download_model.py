# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Download the published multi-task Diffusion Policy and its pinned CLIP encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    from huggingface_hub import hf_hub_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    inputs = json.loads(
        Path(__file__).resolve().with_name("model-inputs.json").read_text()
    )
    args.destination.mkdir(parents=True, exist_ok=True)
    checkpoint = inputs["checkpoint"]
    destination = args.destination / checkpoint["local_filename"]
    if not destination.exists():
        downloaded = hf_hub_download(
            repo_id=checkpoint["repository"],
            filename=checkpoint["filename"],
            revision=checkpoint["revision"],
        )
        temporary = destination.with_suffix(".download")
        shutil.copyfile(downloaded, temporary)
        temporary.replace(destination)
    if destination.stat().st_size != checkpoint["size_bytes"]:
        raise ValueError(f"checkpoint size mismatch: {destination}")
    if file_sha256(destination) != checkpoint["sha256"]:
        raise ValueError(f"checkpoint SHA-256 mismatch: {destination}")
    encoder = inputs["language_encoder"]
    for filename in encoder["files"]:
        hf_hub_download(
            repo_id=encoder["repository"],
            filename=filename,
            revision=encoder["revision"],
            local_dir=str(args.destination / "clip"),
        )
    if (
        file_sha256(args.destination / "clip" / "model.safetensors")
        != encoder["weights_sha256"]
    ):
        raise ValueError("CLIP weights SHA-256 mismatch")
    (args.destination / "model-inputs.json").write_text(
        json.dumps(inputs, indent=2) + "\n"
    )
    print(f"Verified official Diffusion Policy and CLIP weights in {args.destination}")


if __name__ == "__main__":
    main()
