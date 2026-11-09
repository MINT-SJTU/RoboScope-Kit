"""Pretrained ROBEL DClaw screw actor, without legacy TensorFlow dependencies."""

from __future__ import annotations

import collections
import hashlib
import io
import pickle
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SOURCE_REVISION = "039466811957f8d046213eddaa7d0e764971030d"
CHECKPOINT_NAME = "DClawScrewRandom-v0-policy.pkl"
CHECKPOINT_SHA256 = "7c536de59cfe1a75e7e16e2a8e446df6d2ea9ff59bc0dd12b3b8332693ebadf5"
CHECKPOINT_URL = (
    f"https://raw.githubusercontent.com/vikashplus/robel_dev/{SOURCE_REVISION}/"
    f"robel/scripts/data/softlearning/{CHECKPOINT_NAME}"
)


class _SerializedRecord:
    """Preserve configuration metadata without importing or instantiating TF objects."""

    def __init__(self, *arguments: object) -> None:
        self.arguments = arguments

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)


class _WeightUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> object:
        if (module, name) in {
            ("softlearning.policies.gaussian_policy", "FeedforwardGaussianPolicy"),
            ("tensorflow.python.framework.tensor_shape", "TensorShape"),
            ("tensorflow.python.framework.tensor_shape", "Dimension"),
        }:
            return _SerializedRecord
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (module, name) == ("numpy.core.multiarray", "_reconstruct"):
            # This pinned historical pickle uses NumPy's ndarray reconstruction entry point.
            from numpy.core.multiarray import _reconstruct

            return _reconstruct
        if module == "numpy" and name in {"ndarray", "dtype"}:
            return getattr(np, name)
        raise pickle.UnpicklingError(f"Unexpected checkpoint object: {module}.{name}")


@dataclass(frozen=True)
class ROBELActor:
    weights: tuple[np.ndarray, ...]

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        first, first_bias, second, second_bias, output, output_bias = self.weights
        hidden = np.maximum(observation @ first + first_bias, np.float32(0))
        hidden = np.maximum(hidden @ second + second_bias, np.float32(0))
        # GaussianPolicy splits [mean, log_scale]; deterministic evaluation uses tanh(mean).
        return np.tanh((hidden @ output + output_bias)[:, :9])


def load_actor(source: str | Path) -> ROBELActor:
    checkpoint = Path(source).resolve(strict=True)
    if checkpoint.is_dir():
        checkpoint /= CHECKPOINT_NAME
    serialized = checkpoint.read_bytes()
    if hashlib.sha256(serialized).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("Expected the pinned public ROBEL DClawScrewRandom actor")
    payload = _WeightUnpickler(io.BytesIO(serialized)).load()
    configuration = vars(payload["policy"])
    if configuration["__args"] != ((256, 256), "relu", "linear"):
        raise ValueError("Unexpected ROBEL actor architecture")
    if configuration["__kwargs"]["preprocessors"] != {"obs": None}:
        raise ValueError("Unexpected ROBEL observation preprocessing")
    weights = payload["weights"]
    expected_shapes = ((21, 256), (256,), (256, 256), (256,), (256, 18), (18,))
    if len(weights) != len(expected_shapes):
        raise ValueError("Unexpected ROBEL weight count")
    for value, shape, duplicate in zip(
        weights, expected_shapes, configuration["pickled_weights"], strict=True,
    ):
        if (not isinstance(value, np.ndarray) or value.shape != shape
                or value.dtype != np.float32 or not np.isfinite(value).all()
                or not np.array_equal(value, duplicate)):
            raise ValueError("ROBEL weights differ from their declared tensor metadata")
    return ROBELActor(tuple(weights))


def download_actor(destination: str | Path) -> Path:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / CHECKPOINT_NAME
    if checkpoint.exists():
        data = checkpoint.read_bytes()
    else:
        with urllib.request.urlopen(CHECKPOINT_URL, timeout=120) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("Downloaded ROBEL checkpoint differs from its pinned digest")
    checkpoint.write_bytes(data)
    return checkpoint


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download the pretrained ROBEL DClaw screw actor")
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    print(download_actor(arguments.destination))
