"""NEWT's deterministic state policy prior, without its training dependencies.

Architecture and inference match nicklashansen/newt at
1d3fc058b81ddf8d36a5457c29ea407dd374c1b8, with mpc=False and eval_mode=True.
All world-model tensors are loaded strictly; only the encoder and actor execute.
The original implementation is MIT licensed, Copyright (c) Nicklas Hansen.
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
from torch.nn import functional as F


class SimNorm(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        shape = value.shape
        return F.softmax(value.reshape(*shape[:-1], -1, 8), dim=-1).reshape(shape)


class NormedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, *, act: nn.Module | None = None):
        super().__init__(in_features, out_features)
        self.ln = nn.LayerNorm(out_features)
        self.act = nn.Mish(inplace=False) if act is None else act

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.act(self.ln(super().forward(value)))


def mlp(in_features: int, hidden: list[int], out_features: int, act: nn.Module | None = None):
    dimensions = [in_features, *hidden, out_features]
    layers: list[nn.Module] = [
        NormedLinear(dimensions[index], dimensions[index + 1])
        for index in range(len(dimensions) - 2)
    ]
    layers.append(
        nn.Linear(dimensions[-2], dimensions[-1])
        if act is None else NormedLinear(dimensions[-2], dimensions[-1], act=act)
    )
    return nn.Sequential(*layers)


class QEnsemble(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._Qs = nn.ModuleList([mlp(528, [512, 512], 101) for _ in range(5)])


class QOnlineTargetEnsemble(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.online = QEnsemble()
        self.target = QEnsemble()


class NewtPrior(nn.Module):
    """Single-task model-size B state network with the upstream tensor names."""

    def __init__(self) -> None:
        super().__init__()
        # The released single-task checkpoints retain ten identical environment masks.
        self.register_buffer("_action_masks", torch.zeros(10, 16))
        self._encoder = nn.ModuleDict({"state": mlp(128, [256], 512, SimNorm())})
        self._dynamics = mlp(528, [512, 512], 512, SimNorm())
        self._reward = mlp(528, [512, 512], 101)
        self._pi = mlp(512, [512, 512], 32)
        self._Qs = QOnlineTargetEnsemble()
        self.register_buffer("log_std_min", torch.tensor(-10.0))
        self.register_buffer("log_std_dif", torch.tensor(12.0))

    def forward(self, padded_state: torch.Tensor) -> torch.Tensor:
        latent = self._encoder["state"](padded_state)
        mean, _ = self._pi(latent).chunk(2, dim=-1)
        # WorldModel.pi first masks the unsquashed mean, then applies tanh.
        # The Gaussian sample/log-probability do not affect eval_mode=True.
        return torch.tanh(mean * self._action_masks[0])


@dataclass(frozen=True)
class LoadedNewt:
    model: NewtPrior
    observation_dim: int
    action_dim: int
    metadata: dict[str, Any]


def checkpoint_records(source: str | Path) -> dict[str, dict[str, Any]]:
    selected = Path(source).expanduser().resolve(strict=True)
    if selected.is_dir():
        selected = selected / "models.json"
    catalog = json.loads(Path(__file__).with_name("newt-checkpoints.json").read_text("utf-8"))
    if selected.suffix == ".pt":
        digest = hashlib.sha256(selected.read_bytes()).hexdigest()
        records = {
            task: {**record, "checkpoint": selected}
            for task, record in catalog["models"].items() if record["sha256"] == digest
        }
        if len(records) != 1:
            raise ValueError("Unknown NEWT checkpoint; use a pinned checkpoint from the catalog")
        return records
    bundle = json.loads(selected.read_text("utf-8"))
    if bundle.get("schema_version") != 1 or not bundle.get("models"):
        raise ValueError("Expected a schema_version=1 bundle with a nonempty models mapping")
    records = {}
    for task, record in bundle["models"].items():
        expected = catalog["models"].get(task)
        if expected is None or any(record.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{task} does not match the pinned NEWT model contract")
        records[task] = {
            **record, "checkpoint": (selected.parent / record["path"]).resolve(strict=True),
        }
    return records


def load_model_bundle(source: str | Path, device: str = "cuda") -> dict[str, LoadedNewt]:
    models = {}
    for task, record in checkpoint_records(source).items():
        checkpoint = record["checkpoint"]
        if checkpoint.stat().st_size != record["bytes"]:
            raise ValueError(f"{task} checkpoint size differs from the pinned artifact")
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"{task} checkpoint SHA-256 differs from the pinned artifact")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if isinstance(state, Mapping) and "model" in state:
            state = state["model"]
        if not isinstance(state, Mapping) or not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise TypeError("Expected a NEWT model tensor state dict")
        if all(key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}
        model = NewtPrior()
        model.load_state_dict(state, strict=True)
        expected_mask = torch.zeros(10, 16)
        expected_mask[:, :record["action_dimension"]] = 1
        if not torch.equal(model._action_masks, expected_mask):
            raise ValueError(f"{task} checkpoint action mask differs from its native controller")
        model = model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        models[task] = LoadedNewt(
            model, record["observation_dimension"], record["action_dimension"], record,
        )
    return models
