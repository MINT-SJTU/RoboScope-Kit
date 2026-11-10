"""Pinned real-robot TriFinger experts with their consumed physical state inputs."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import torch
from torch import nn

SOURCE_REVISION = "e4841da99127a19e319bd06b71d2becd52067e38"
MODEL_SHA256 = {
    "push": "e469bb3690eee61deb073e3fe63a1609253b495ca15060d9a06b406ffa7212b7",
    "lift": "680c06886538e8bd609200aef1663b5dfdca9c41d2cd736ab408bb3ecaa05d49",
}


class TriFingerExpert(nn.Module):
    """Execute the original modules without constructing unused source observations."""

    def __init__(self, source: torch.jit.ScriptModule) -> None:
        super().__init__()
        self.actor_mlp = source.actor_mlp
        self.mu = source.mu
        self.running_mean_std = source.running_mean_std
        constants = source.code_with_constants[1].const_mapping
        for index in range(9):
            self.register_buffer(f"c{index}", constants[f"c{index}"].clone())

    def forward(self, physical_state: torch.Tensor) -> torch.Tensor:
        # The original simulator clips physical observations to these same bounds.
        physical_state = torch.maximum(self.c0, torch.minimum(physical_state, self.c0 + self.c1))
        calibrated = ((physical_state - self.c0) / self.c1) * self.c2 - self.c3
        normalized = self.running_mean_std(calibrated)
        action = self.mu(self.actor_mlp(normalized)).clamp(-1.0, 1.0)
        half_range = (self.c4 - self.c5) / self.c2
        center = (self.c4 + self.c5) / self.c2
        action = action * half_range + center
        return ((action + self.c3) * self.c6) * self.c7 + self.c8


def load_expert(checkpoint: str | Path, kind: str, device: str = "cpu") -> TriFingerExpert:
    checkpoint = Path(checkpoint).resolve(strict=True)
    if checkpoint.is_dir():
        checkpoint /= f"{kind}.pt"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != MODEL_SHA256[kind]:
        raise ValueError(f"Expected the pinned public TriFinger {kind} expert")
    source = torch.jit.load(str(checkpoint), map_location="cpu").eval()
    return TriFingerExpert(source).to(device).eval().requires_grad_(False)


def download_experts(destination: str | Path) -> None:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for kind, expected_digest in MODEL_SHA256.items():
        checkpoint = destination / f"{kind}.pt"
        if checkpoint.exists():
            data = checkpoint.read_bytes()
        else:
            url = (
                "https://raw.githubusercontent.com/rr-learning/trifinger-rl-example/"
                f"{SOURCE_REVISION}/trifinger_rl_example/policies/{kind}.pt"
            )
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read()
        if hashlib.sha256(data).hexdigest() != expected_digest:
            raise ValueError(f"TriFinger {kind} checkpoint differs from the pinned digest")
        checkpoint.write_bytes(data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download the pinned public TriFinger experts")
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    download_experts(arguments.destination)
    print(arguments.destination)
