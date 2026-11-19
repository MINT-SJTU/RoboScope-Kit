# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Content identities for ManiSkill shard inputs, without importing a model or simulator."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path

from roboscope_kit.policy_config import PolicyRuntimeConfig


def _file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# Directories that hold installed packages, build output or tool caches, not Policy code.
_SKIPPED_CODE_DIRS = {
    ".git", "__pycache__", ".venv", "venv", ".tox", ".nox", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", "node_modules", "site-packages", "build", "dist",
}


def _python_sources(root: Path) -> Iterator[Path]:
    """Walk ``root`` without descending into virtual environments or build trees."""
    for directory, names, files in os.walk(root):
        parent = Path(directory)
        names[:] = [
            name for name in names
            if name not in _SKIPPED_CODE_DIRS and not (parent / name / "pyvenv.cfg").is_file()
        ]
        for name in files:
            path = parent / name
            if name.endswith(".py") and path.is_file():
                yield path


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def shard_input_identity(checkpoint: Path, policy: PolicyRuntimeConfig) -> dict[str, str]:
    """Hash immutable checkpoint content and declared Policy code once before a sharded run.

    Paths are relative to their input roots so equivalent installations may live
    at different absolute paths. Python bytecode and Git internals are not model
    inputs, and Policy code excludes virtual environments and build trees under
    ``policy.cwd``. Keep mutable runtime caches outside the checkpoint directory.
    """
    checkpoint = checkpoint.resolve(strict=True)
    if checkpoint.is_file():
        files = [(checkpoint.name, _file_digest(checkpoint))]
    else:
        files = [
            (path.relative_to(checkpoint).as_posix(), _file_digest(path))
            for path in sorted(checkpoint.rglob("*"))
            if path.is_file() and not {".git", "__pycache__"}.intersection(
                path.relative_to(checkpoint).parts
            ) and path.suffix not in {".pyc", ".pyo"}
        ]
    if not files:
        raise ValueError("ManiSkill shard checkpoint contains no files")
    code = [
        (path.relative_to(policy.cwd).as_posix(), _file_digest(path))
        for path in sorted(_python_sources(policy.cwd))
    ]
    arguments: list[object] = [Path(policy.command[0]).name]
    for argument in policy.command[1:]:
        path = Path(argument)
        if not path.is_absolute():
            path = policy.cwd / path
        arguments.append(
            {"file": path.name, "sha256": _file_digest(path)} if path.is_file() else argument
        )
    return {
        "checkpoint_sha256": _digest(files),
        "policy_sha256": _digest({
            "command": arguments, "python_sources": code,
            "benchmark_config": policy.benchmark_config,
        }),
    }
