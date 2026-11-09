"""Pinned HORA-RMA Allegro student actor and observation normalization.

Adapted from hora/algo/models/models.py at HORA-RMA revision
b1f009128f9adad3b07bc8aff91d7916feed09fc. Copyright (c) 2022 Haozhi Qi.
The upstream MIT permission notice is reproduced in hora_policy.md.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import torch
from torch import nn

SOURCE_REVISION = "b1f009128f9adad3b07bc8aff91d7916feed09fc"
ARCHIVE_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=1AKecNsQZ56TCyJU49DU06GxnQRbeawMu&export=download&authuser=0&confirm=t"
)
ARCHIVE_SHA256 = "beed1cfdc3095899e3e03687d1ebee80539174bf49b8686041e5aface85678f0"
CHECKPOINT_SHA256 = "b8e16f5d64c3af111205c4b088b9888acaa4d7fc91e2a42c97b6c544ab87d2aa"
CHECKPOINT_NAME = "hora-stage2.pth"


class MLP(nn.Module):
    def __init__(self, units: tuple[int, ...], input_size: int) -> None:
        super().__init__()
        layers = []
        for output_size in units:
            layers.extend((nn.Linear(input_size, output_size), nn.ELU()))
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.mlp(value)


class ProprioAdaptTConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.channel_transform = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(),
        )
        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(32, 32, 9, stride=2), nn.ReLU(),
            nn.Conv1d(32, 32, 5), nn.ReLU(),
            nn.Conv1d(32, 32, 5), nn.ReLU(),
        )
        self.low_dim_proj = nn.Linear(96, 8)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.channel_transform(value).permute(0, 2, 1)
        return self.low_dim_proj(self.temporal_aggregation(value).flatten(1))


class HORAActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Retain the unused teacher/critic parameters for strict checkpoint loading.
        self.env_mlp = MLP((256, 128, 8), 9)
        self.adapt_tconv = ProprioAdaptTConv()
        self.actor_mlp = MLP((512, 256, 128), 104)
        self.value = nn.Linear(128, 1)
        self.mu = nn.Linear(128, 16)
        self.sigma = nn.Parameter(torch.zeros(16))

    def forward(self, observation: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        adaptation = torch.tanh(self.adapt_tconv(history))
        return self.mu(self.actor_mlp(torch.cat((observation, adaptation), dim=-1)))


class RunningMeanStd(nn.Module):
    """Frozen inference formula from the source normalizer, including its clamp."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(shape, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = (value - self.running_mean.float()) / torch.sqrt(self.running_var.float() + 1e-5)
        return value.clamp(-5.0, 5.0)


class HORAModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = HORAActor()
        self.observation_normalizer = RunningMeanStd((96,))
        self.history_normalizer = RunningMeanStd((30, 32))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        observation = self.observation_normalizer(history[:, -3:].reshape(-1, 96))
        return self.actor(observation, self.history_normalizer(history))


def load_hora(source: str | Path, device: str = "cuda") -> HORAModel:
    checkpoint = Path(source).resolve(strict=True)
    if checkpoint.is_dir():
        checkpoint /= CHECKPOINT_NAME
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("Expected the pinned public HORA stage2 student checkpoint")
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = HORAModel()
    model.actor.load_state_dict(weights["model"], strict=True)
    model.observation_normalizer.load_state_dict(weights["running_mean_std"], strict=True)
    model.history_normalizer.load_state_dict(weights["sa_mean_std"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def download_hora(destination: str | Path) -> Path:
    """Download 7.1 MB and retain only the 1.15 MB student, excluding training logs."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / CHECKPOINT_NAME
    if checkpoint.exists():
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
            raise ValueError(f"Existing checkpoint differs from the pinned digest: {checkpoint}")
        return checkpoint
    with urllib.request.urlopen(ARCHIVE_URL, timeout=120) as response:
        archive = response.read()
    if hashlib.sha256(archive).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Downloaded HORA archive differs from the pinned digest")
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        weights = bundle.read("stage2_nn/last.pth")
    if hashlib.sha256(weights).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("HORA student differs from the pinned digest")
    checkpoint.write_bytes(weights)
    return checkpoint


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download the pinned HORA-RMA Allegro student")
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    print(download_hora(arguments.destination))
