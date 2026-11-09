"""Official MS-HAB PPO architecture and immutable Fetch Open checkpoints.

The architecture follows arth-shukla/mshab at
 e9ff3d23496d38e4431c8d913e147ffa007f7f72 (package metadata: MIT).
Released checkpoint weights are CC-BY-4.0; see FETCH_MSHAB.md for attribution.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


class NatureCNN(nn.Module):
    """Preserve the author's two depth encoders and unactivated state projection."""

    def __init__(self) -> None:
        super().__init__()
        extractors = {}
        for key in ("fetch_head_depth", "fetch_hand_depth"):
            cnn = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(), nn.Flatten(),
            )
            fc = nn.Sequential(nn.Linear(64 * 12 * 12, 256), nn.ReLU())
            extractors[key] = nn.Sequential(cnn, fc)
        extractors["state"] = nn.Linear(42, 256)
        self.extractors = nn.ModuleDict(extractors)
        self.out_features = 768

    def forward(self, observations: dict[str, object]) -> torch.Tensor:
        encoded = []
        for key, extractor in self.extractors.items():
            if key == "state":
                encoded.append(extractor(observations["state"]))
            else:
                with torch.no_grad():
                    pixels = 1 - torch.tanh(observations["pixels"][key].float() / 1000)
                    batch, frames, channels, height, width = pixels.shape
                    pixels = pixels.reshape(batch, frames * channels, height, width)
                encoded.append(extractor(pixels))
        return torch.cat(encoded, dim=1)


class MshabPPO(nn.Module):
    """Retain the complete released actor, critic and learned log standard deviation."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_net = NatureCNN()
        self.critic = nn.Sequential(nn.Linear(768, 512), nn.ReLU(inplace=True), nn.Linear(512, 1))
        self.actor_mean = nn.Sequential(nn.Linear(768, 512), nn.ReLU(inplace=True), nn.Linear(512, 13))
        self.actor_logstd = nn.Parameter(torch.zeros(1, 13))

    def forward(self, observations: dict[str, object]) -> torch.Tensor:
        return self.actor_mean(self.feature_net(observations))


@dataclass(frozen=True)
class LoadedFetch:
    actor: MshabPPO
    metadata: dict


def catalog() -> dict:
    return json.loads(Path(__file__).with_name("fetch-mshab-checkpoints.json").read_text("utf-8"))


def load_models(source: str | Path, device: str = "cuda") -> dict[str, LoadedFetch]:
    source = Path(source).expanduser().resolve(strict=True)
    if source.is_dir():
        source /= "models.json"
    bundle = json.loads(source.read_text("utf-8"))
    if bundle.get("schema_version") != 1 or not bundle.get("models"):
        raise ValueError("Expected a nonempty schema_version=1 MS-HAB model bundle")
    pins = catalog()["models"]
    models = {}
    for task, record in bundle["models"].items():
        if pins.get(task) != record:
            raise ValueError(f"{task} differs from the pinned official MS-HAB model contract")
        path = source.parent / record["path"]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != record["bytes"] or digest != record["sha256"]:
            raise ValueError(f"{task} checkpoint size or SHA-256 differs from the published artifact")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        actor = MshabPPO()
        actor.load_state_dict(payload["agent"], strict=True)
        actor = actor.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        models[task] = LoadedFetch(actor, record)
    return models
