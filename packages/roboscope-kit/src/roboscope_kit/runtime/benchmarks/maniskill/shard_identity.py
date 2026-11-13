# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Content identities for ManiSkill shard inputs, without importing a model or simulator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from roboscope_kit.policy_config import PolicyRuntimeConfig


def _file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def shard_input_identity(checkpoint: Path, policy: PolicyRuntimeConfig) -> dict[str, str]:
    """Hash immutable checkpoint content and declared Policy code once before a sharded run.

    Paths are relative to their input roots so equivalent installations may live
    at different absolute paths. Python bytecode and Git internals are not model
    inputs. Keep mutable runtime caches outside the checkpoint directory.
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
        for path in sorted(policy.cwd.rglob("*.py"))
        if path.is_file() and not {".git", "__pycache__"}.intersection(
            path.relative_to(policy.cwd).parts
        )
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
