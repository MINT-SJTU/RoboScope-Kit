"""Strict loader for ManiSkill PPO state checkpoint bundles.

Architecture matches examples/baselines/ppo/ppo_fast.py at ManiSkill revision
baab60ede2e89167c1b7aaed41a9aa8e690a9d1e. This is an RL-trained policy,
not a fine-tuned VLA. Evaluation uses the deterministic actor mean.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

CHECKPOINT_SHA256 = "0b17b5ed9690ccf83111d0f09af8d1599f69ee7ba0077e1aac48814ac78ce99c"


class PPOAgent(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int) -> None:
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(observation_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 1),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(observation_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, action_dim),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))


@dataclass(frozen=True)
class LoadedPPO:
    agent: PPOAgent
    observation_dim: int
    action_dim: int
    checkpoint: Path
    sha256: str
    metadata: dict[str, Any]


def checkpoint_records(source: str | Path) -> dict[str, dict[str, Any]]:
    """Resolve explicit task-to-model mappings without loading Torch weights."""
    selected = Path(source).resolve(strict=True)
    if selected.is_dir():
        manifest = selected / "models.json"
        if manifest.is_file():
            selected = manifest
        else:
            files = sorted(selected.glob("*.pt"))
            if len(files) != 1:
                raise ValueError("A multi-model checkpoint directory must contain models.json")
            selected = files[0]
    if selected.suffix == ".pt":
        # A standalone file is associated with a task only by a known pinned digest.
        digest = hashlib.sha256(selected.read_bytes()).hexdigest()
        official = json.loads(
            Path(__file__).with_name("official-models.json").read_text(encoding="utf-8")
        )["models"]
        for task_id, record in official.items():
            if record["sha256"] == digest:
                return {task_id: {**record, "checkpoint": selected}}
        raise ValueError("Unknown standalone checkpoint; provide an explicit models.json task mapping")
    manifest = json.loads(selected.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("models"):
        raise ValueError("Expected schema_version=1 and a nonempty models mapping")
    records = {}
    for task_id, record in manifest["models"].items():
        if not isinstance(record, dict) or "path" not in record or "sha256" not in record:
            raise ValueError(f"Model {task_id} requires path and sha256")
        records[task_id] = {
            **record,
            "checkpoint": (selected.parent / record["path"]).resolve(strict=True),
        }
    return records


def load_agent(
    source: str | Path, device: str = "cuda", *, expected_sha256: str,
) -> tuple[PPOAgent, int, int]:
    checkpoint = Path(source).resolve(strict=True)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"Checkpoint differs from the declared SHA-256: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not all(isinstance(x, torch.Tensor) for x in state.values()):
        raise TypeError("The official checkpoint must be a plain tensor state dict")
    observation_dim = state["actor_mean.0.weight"].shape[1]
    action_dim = state["actor_mean.6.weight"].shape[0]
    agent = PPOAgent(observation_dim, action_dim)
    agent.load_state_dict(state, strict=True)
    agent = agent.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    return agent, observation_dim, action_dim


def load_model_bundle(source: str | Path, device: str = "cuda") -> dict[str, LoadedPPO]:
    models = {}
    for task_id, record in checkpoint_records(source).items():
        agent, observation_dim, action_dim = load_agent(
            record["checkpoint"], device, expected_sha256=record["sha256"],
        )
        models[task_id] = LoadedPPO(
            agent, observation_dim, action_dim, record["checkpoint"], record["sha256"], record,
        )
    return models
