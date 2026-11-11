"""Pinned TD-MPC2 LiftCube actor prior, retaining all original world-model tensors.

Source: nicklashansen/tdmpc2 at 7ec6bc83a82a5188ca3faddc59aea83f430ab570.
The original implementation is MIT licensed, Copyright (c) 2023 Nicklas Hansen.
Inference explicitly selects mpc=False, eval_mode=True; it does not reproduce
the authors' default stochastic MPC evaluation.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import torch
from newt_model import SimNorm, mlp
from torch import nn


class LiftCubeEnsemble(nn.Module):
    def __init__(self, configuration: dict) -> None:
        super().__init__()
        count, hidden = configuration["num_q"], configuration["mlp_dim"]
        shapes = [(count, hidden, configuration["latent_dim"] + 4), *((count, hidden),) * 3,
                  (count, hidden, hidden), *((count, hidden),) * 3,
                  (count, configuration["num_bins"], hidden), (count, configuration["num_bins"])]
        self.params = nn.ParameterList([nn.Parameter(torch.zeros(shape)) for shape in shapes])


class LiftCubePrior(nn.Module):
    """Complete original 60-tensor, 42-state/4-action single-task world model."""

    def __init__(self, configuration: dict) -> None:
        super().__init__()
        latent, hidden = configuration["latent_dim"], configuration["mlp_dim"]
        self._encoder = nn.ModuleDict({"state": mlp(42, [configuration["enc_dim"]], latent, SimNorm())})
        self._dynamics = mlp(latent + 4, [hidden, hidden], latent, SimNorm())
        self._reward = mlp(latent + 4, [hidden, hidden], configuration["num_bins"])
        self._pi = mlp(latent, [hidden, hidden], 8)
        self._Qs = LiftCubeEnsemble(configuration)
        self._target_Qs = LiftCubeEnsemble(configuration)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        latent = self._encoder["state"](state)
        mean, _ = self._pi(latent).chunk(2, dim=-1)
        return torch.tanh(mean)


def catalog() -> dict:
    return json.loads(Path(__file__).with_name("liftcube-checkpoint.json").read_text("utf-8"))


def load_actor(source: str | Path, device: str = "cuda") -> LiftCubePrior:
    pin = catalog()
    path = Path(source).expanduser().resolve(strict=True)
    if path.is_dir():
        path /= pin["path"]
    data = path.read_bytes()
    if len(data) != pin["bytes"] or hashlib.sha256(data).hexdigest() != pin["sha256"]:
        raise ValueError("LiftCube checkpoint size or SHA-256 differs from its author release")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if any(checkpoint["metadata"].get(key) != value for key, value in pin["configuration"].items()):
        raise ValueError("LiftCube checkpoint metadata differs from the pinned configuration")
    if len(checkpoint["model"]) != 60:
        raise ValueError("Expected all original 60 LiftCube world-model tensors")
    actor = LiftCubePrior(pin["configuration"])
    actor.load_state_dict(checkpoint["model"], strict=True)
    return actor.to(device=device, dtype=torch.float32).eval().requires_grad_(False)


def download_model(destination: str | Path) -> Path:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    pin = catalog()
    target = destination / pin["path"]
    if target.exists():
        data = target.read_bytes()
    else:
        with urllib.request.urlopen(pin["url"], timeout=120) as response:
            data = response.read()
    if len(data) != pin["bytes"] or hashlib.sha256(data).hexdigest() != pin["sha256"]:
        raise ValueError("Downloaded LiftCube model differs from the pinned author artifact")
    target.write_bytes(data)
    return target


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download the pinned TD-MPC2 LiftCube prior")
    parser.add_argument("destination", type=Path)
    download_model(parser.parse_args().destination)
