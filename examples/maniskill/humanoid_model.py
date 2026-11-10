"""Pinned TD-MPC2 Humanoid policy priors, preserving every released tensor.

Inference follows nicklashansen/tdmpc2 at
332f19e4aba4bffe2dc736f45ea7f7bdc585903a with mpc=False and eval_mode=True.
The original implementation is MIT licensed, Copyright (c) 2023 Nicklas Hansen.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from newt_model import SimNorm, mlp
from torch import nn


class EnsembleParameters(nn.Module):
    """Retain the original checkpoint's stacked online or target critic parameters."""

    def __init__(self) -> None:
        super().__init__()
        shapes = [(5, 512, 533), *((5, 512),) * 3,
                  (5, 512, 512), *((5, 512),) * 3, (5, 101, 512), (5, 101)]
        self.params = nn.ParameterList([nn.Parameter(torch.zeros(shape)) for shape in shapes])


class HumanoidPrior(nn.Module):
    """The complete original 60-tensor, 67-state/21-action world model layout."""

    def __init__(self) -> None:
        super().__init__()
        self._encoder = nn.ModuleDict({"state": mlp(67, [256], 512, SimNorm())})
        self._dynamics = mlp(533, [512, 512], 512, SimNorm())
        self._reward = mlp(533, [512, 512], 101)
        self._pi = mlp(512, [512, 512], 42)
        self._Qs = EnsembleParameters()
        self._target_Qs = EnsembleParameters()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        latent = self._encoder["state"](state)
        mean, _ = self._pi(latent).chunk(2, dim=-1)
        return torch.tanh(mean)


@dataclass(frozen=True)
class LoadedHumanoid:
    actor: HumanoidPrior
    metadata: dict[str, Any]


def catalog() -> dict[str, Any]:
    return json.loads(Path(__file__).with_name("humanoid-checkpoints.json").read_text("utf-8"))


def load_models(source: str | Path, device: str = "cuda") -> dict[str, LoadedHumanoid]:
    source = Path(source).expanduser().resolve(strict=True)
    pins = catalog()["models"]
    if source.is_dir():
        source /= "models.json"
    if source.suffix == ".pt":
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        selected = {task: record for task, record in pins.items() if record["sha256"] == digest}
        paths = {task: source for task in selected}
    else:
        bundle = json.loads(source.read_text("utf-8"))
        if bundle.get("schema_version") != 1 or not bundle.get("models"):
            raise ValueError("Expected a nonempty schema_version=1 Humanoid model bundle")
        selected = bundle["models"]
        paths = {task: source.parent / record["path"] for task, record in selected.items()}
    if not selected:
        raise ValueError("No pinned TD-MPC2 Humanoid checkpoint selected")
    models = {}
    for task, record in selected.items():
        if pins.get(task) != record:
            raise ValueError(f"{task} differs from the pinned Humanoid model contract")
        path = paths[task]
        data = path.read_bytes()
        if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"{task} checkpoint size or digest differs from the published artifact")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state = payload["model"]
        if (not isinstance(state, Mapping) or len(state) != 60
                or not all(isinstance(value, torch.Tensor) for value in state.values())):
            raise TypeError("Expected the original 60-tensor TD-MPC2 checkpoint")
        if payload["metadata"]["task"] != record["model_task"]:
            raise ValueError(f"{task} source task metadata differs from its pin")
        actor = HumanoidPrior()
        actor.load_state_dict(state, strict=True)
        actor = actor.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        models[task] = LoadedHumanoid(actor, record)
    return models


def download_models(destination: str | Path) -> None:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    pins = catalog()
    for record in pins["models"].values():
        target = destination / record["path"]
        if target.exists():
            data = target.read_bytes()
        else:
            url = (f"https://huggingface.co/{pins['repository']}/resolve/{pins['revision']}/"
                   f"dmcontrol/{record['path']}")
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read()
        if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"Invalid published Humanoid checkpoint: {target.name}")
        target.write_bytes(data)
    (destination / "models.json").write_text(
        json.dumps({"schema_version": 1, "models": pins["models"]}, indent=2) + "\n", "utf-8",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download three pinned TD-MPC2 Humanoid policies")
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    download_models(arguments.destination)
