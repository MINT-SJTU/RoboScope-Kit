"""Config-driven StarVLA Policy for RoboScope Kit."""

from __future__ import annotations

import gc
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from roboscope_kit import timing
from roboscope_kit.policy import (
    ActionChunk,
    EpisodeContext,
    Observation,
    Policy,
    PolicyCapabilities,
    SessionContext,
)
from roboscope_kit.server import run
from starvla_base_assets import ensure_starvla_base_model
from starvla_full_checkpoint import (
    CHECKPOINT_SUFFIXES,
    LoadedStarVLA,
    load_full_checkpoint,
)
from starvla_timing import ModelTiming

if TYPE_CHECKING:
    from deployment.model_server.policy_norm_processor import PolicyNormProcessor

DTYPES = frozenset({"bfloat16", "float16", "float32"})
OFFLINE_ENVIRONMENT = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
# Pinned StarVLA training and native evaluation clients both use 224x224 inputs.
STARVLA_IMAGE_SIZE = (224, 224)
ImagePreprocessor = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class InputConfig:
    images: tuple[str, ...]
    language: str
    state: str | None

    @property
    def required_features(self) -> tuple[str, ...]:
        names = (*self.images, self.language, *((self.state,) if self.state is not None else ()))
        if len(names) != len(set(names)):
            raise ValueError("StarVLA inference input feature names must be unique")
        return names


@dataclass(frozen=True)
class ActionTransform:
    operation: str
    index: int
    parameters: tuple[float, ...]

    def apply(self, actions: np.ndarray) -> None:
        if self.operation == "threshold":
            threshold, below, above = self.parameters
            actions[..., self.index] = np.where(actions[..., self.index] > threshold, above, below)
            return
        scale, offset = self.parameters
        actions[..., self.index] = actions[..., self.index] * scale + offset


@dataclass(frozen=True)
class InferenceConfig:
    dtype: str
    unnorm_key: str | None
    inputs: InputConfig
    kwargs: dict[str, Any]
    transforms: tuple[ActionTransform, ...]
    action_dim: int
    action_horizon: int


@dataclass(frozen=True)
class _PolicyRuntime:
    framework: torch.nn.Module
    norm_processor: PolicyNormProcessor
    config: InferenceConfig
    image_preprocessor: ImagePreprocessor
    predict: Callable[..., Mapping[str, Any]]
    model_timing: ModelTiming | None
    max_batch_size: int


class StarVLAPolicy(Policy):
    def __init__(self) -> None:
        self._runtime: _PolicyRuntime | None = None
        self.original_cwd: Path | None = None
        self.original_environment: dict[str, str | None] = {}

    def load(self, context: SessionContext) -> PolicyCapabilities:
        batch_text = os.environ.get("ROBOSCOPE_STARVLA_MAX_BATCH_SIZE", "1")
        if not batch_text.isdecimal() or int(batch_text) < 1:
            raise ValueError("ROBOSCOPE_STARVLA_MAX_BATCH_SIZE must be a positive integer")
        max_batch_size = int(batch_text)
        image_preprocessor = _image_preprocessor(context.benchmark.name)
        checkpoint, run_dir = resolve_starvla_run_snapshot(context.checkpoint_path)
        ensure_starvla_base_model(run_dir)
        self._enter_offline_run(run_dir)
        try:
            loaded = load_full_checkpoint(checkpoint)
            config = parse_inference_config(loaded.config)
            if max_batch_size > 1 and (
                type(self) is not StarVLAPolicy or context.benchmark.name != "libero_pro"
                or type(loaded.framework).__name__ != "Qwenvl_OFT"
            ):
                raise ValueError("native batch > 1 is currently supported only for LIBERO-Pro QwenOFT")
            self._runtime = self._initialize_runtime(
                checkpoint, loaded, config, image_preprocessor, max_batch_size,
            )
        except Exception:
            self.unload()
            raise
        return PolicyCapabilities(
            supported_benchmarks=(context.benchmark.name,),
            supported_suites={context.benchmark.name: (context.benchmark.suite,)},
            required_features=config.inputs.required_features,
            max_batch_size=max_batch_size,
            supported_action_chunk_sizes=(config.action_horizon,),
            action_representations={
                context.benchmark.name: context.benchmark.default_action_representation
            },
            # A new chunk is requested every action_horizon steps; the
            # observations in between are never sent to the Policy.
            observation_strides={context.benchmark.name: config.action_horizon},
        )

    def _require_runtime(self) -> _PolicyRuntime:
        if self._runtime is None:
            raise RuntimeError("StarVLA Policy is not loaded")
        return self._runtime

    def _enter_offline_run(self, run_dir: Path) -> None:
        if self.original_cwd is not None:
            raise RuntimeError("StarVLA Policy is already loaded")
        self.original_cwd = Path.cwd()
        self.original_environment = {name: os.environ.get(name) for name in OFFLINE_ENVIRONMENT}
        os.chdir(run_dir)
        for name in OFFLINE_ENVIRONMENT:
            os.environ[name] = "1"

    def _initialize_runtime(
        self,
        checkpoint: Path,
        loaded: LoadedStarVLA,
        config: InferenceConfig,
        image_preprocessor: ImagePreprocessor,
        max_batch_size: int,
    ) -> _PolicyRuntime:
        from deployment.model_server.policy_norm_processor import PolicyNormProcessor

        framework = loaded.framework.to(dtype=getattr(torch, config.dtype), device="cuda").eval()
        norm_processor = PolicyNormProcessor(str(checkpoint), unnorm_key=config.unnorm_key)
        predict = torch.inference_mode()(framework.predict_action)
        model_timing = ModelTiming(framework) if timing.ENABLED else None
        if model_timing is not None:
            predict = model_timing.wrap_prediction(predict)
        return _PolicyRuntime(
            framework, norm_processor, config, image_preprocessor, predict, model_timing, max_batch_size,
        )

    def reset(self, _context: EpisodeContext) -> None:
        return None

    def act(self, observation: Observation) -> ActionChunk:
        return self.act_batch((observation,))[0]

    def act_batch(self, observations: tuple[Observation, ...]) -> tuple[ActionChunk, ...]:
        runtime = self._require_runtime()
        batch_size = len(observations)
        if not 1 <= batch_size <= runtime.max_batch_size:
            raise ValueError(
                f"StarVLA batch size {batch_size} must be between 1 and {runtime.max_batch_size}"
            )
        with timing.span("policy.example_build"):
            examples = [
                _build_example(
                    item.features, runtime.config.inputs, runtime.norm_processor, runtime.image_preprocessor,
                )
                for item in observations
            ]
        with timing.span("policy.predict_total"):
            output = runtime.predict(examples=examples, **runtime.config.kwargs)
        with timing.span("policy.postprocess"):
            return _postprocess_batch(output, batch_size, runtime)

    def unload(self) -> None:
        runtime, self._runtime = self._runtime, None
        if runtime is not None and runtime.model_timing is not None:
            runtime.model_timing.close()
        del runtime
        if self.original_cwd is not None:
            os.chdir(self.original_cwd)
        self.original_cwd = None
        for name, value in self.original_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self.original_environment = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _postprocess_batch(
    output: Mapping[str, Any], batch_size: int, runtime: _PolicyRuntime,
) -> tuple[ActionChunk, ...]:
    if not isinstance(output, Mapping) or "normalized_actions" not in output:
        raise TypeError("StarVLA predict_action() must return normalized_actions")
    normalized = np.asarray(output["normalized_actions"])
    expected = (batch_size, runtime.config.action_horizon, runtime.config.action_dim)
    if normalized.shape != expected:
        raise ValueError(f"StarVLA returned normalized action shape {normalized.shape}; expected {expected}")
    # The pinned processor accepts only (T, D). Keep its per-chunk semantics, then
    # perform conversion, transforms and validation once across the whole batch.
    actions = np.array(
        [runtime.norm_processor.unapply_actions(row) for row in normalized],
        dtype=np.float32, copy=True, order="C",
    )
    if actions.shape != expected:
        raise ValueError(f"StarVLA returned unnormalized action shape {actions.shape}; expected {expected}")
    for transform in runtime.config.transforms:
        transform.apply(actions)
    if not np.isfinite(actions).all():
        raise ValueError("StarVLA returned non-finite actions")
    return tuple(ActionChunk(row) for row in actions)


def resolve_starvla_run_snapshot(source_path: str | Path) -> tuple[Path, Path]:
    source = Path(source_path).resolve(strict=True)
    if source.is_file():
        return _validate_checkpoint(source)
    if (source / "config.yaml").is_file():
        return _validate_run_directory(source)
    candidates = [path.parent for path in source.rglob("config.yaml") if path.is_file()]
    complete = [run_dir for run_dir in candidates if (run_dir / "dataset_statistics.json").is_file()]
    if len(complete) != 1:
        raise ValueError(f"expected one StarVLA run snapshot under {source}, found {len(complete)}")
    return _validate_run_directory(complete[0])


def _validate_run_directory(run_dir: Path) -> tuple[Path, Path]:
    statistics = run_dir / "dataset_statistics.json"
    checkpoint_dir = run_dir / "checkpoints"
    if not statistics.is_file() or not checkpoint_dir.is_dir():
        raise ValueError(f"incomplete StarVLA run snapshot: {run_dir}")
    checkpoints = sorted(
        path.resolve()
        for path in checkpoint_dir.iterdir()
        if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES
    )
    if len(checkpoints) != 1:
        raise ValueError(f"StarVLA run snapshot must contain one selected checkpoint: {run_dir}")
    return checkpoints[0], run_dir.resolve()


def _validate_checkpoint(checkpoint: Path) -> tuple[Path, Path]:
    checkpoint = checkpoint.resolve()
    if checkpoint.suffix.lower() not in CHECKPOINT_SUFFIXES or checkpoint.parent.name != "checkpoints":
        raise ValueError("StarVLA checkpoint must be a .pt or .safetensors file inside checkpoints/")
    selected, run_dir = _validate_run_directory(checkpoint.parents[1])
    if selected != checkpoint:
        raise ValueError("StarVLA run snapshot selected a different checkpoint")
    return selected, run_dir


def parse_inference_config(config: Mapping[str, Any]) -> InferenceConfig:
    inference = _mapping(config, "inference")
    _require_keys(
        inference,
        required={"dtype", "inputs"},
        optional={"unnorm_key", "kwargs", "action_transforms"},
        field="inference",
    )
    dtype = _required_string(inference, "dtype")
    if dtype not in DTYPES:
        raise ValueError(f"unsupported StarVLA inference dtype: {dtype}")
    inputs = _parse_inputs(_mapping(inference, "inputs"))
    action_model = _mapping(_mapping(config, "framework"), "action_model")
    action_dim = _positive_integer(action_model, "action_dim")
    action_horizon = _positive_integer(action_model, "action_horizon")
    transforms = _parse_action_transforms(inference.get("action_transforms", ()), action_dim)
    kwargs = _optional_mapping(inference, "kwargs")
    if any(not isinstance(key, str) for key in kwargs):
        raise ValueError("StarVLA inference.kwargs keys must be strings")
    if "examples" in kwargs:
        raise ValueError("StarVLA inference.kwargs cannot override examples")
    return InferenceConfig(
        dtype=dtype,
        unnorm_key=_optional_string(inference, "unnorm_key"),
        inputs=inputs,
        kwargs=kwargs,
        transforms=transforms,
        action_dim=action_dim,
        action_horizon=action_horizon,
    )


def _parse_inputs(config: Mapping[str, Any]) -> InputConfig:
    _require_keys(
        config,
        required={"images", "language"},
        optional={"state"},
        field="inference.inputs",
    )
    images = _string_list(config, "images")
    language = _required_string(config, "language")
    inputs = InputConfig(
        images=images,
        language=language,
        state=_optional_string(config, "state"),
    )
    _ = inputs.required_features
    return inputs


def _parse_action_transforms(value: Any, action_dim: int) -> tuple[ActionTransform, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("StarVLA inference.action_transforms must be a list")
    return tuple(_parse_action_transform(item, action_dim) for item in value)


def _parse_action_transform(value: Any, action_dim: int) -> ActionTransform:
    if not isinstance(value, Mapping):
        raise ValueError("each StarVLA action transform must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError("StarVLA action transform keys must be strings")
    operation = _required_string(value, "operation")
    index = _integer(value, "index")
    if index < 0 or index >= action_dim:
        raise ValueError(f"StarVLA action transform index is out of range: {index}")
    parameter_names = {
        "threshold": ("threshold", "below", "above"),
        "affine": ("scale", "offset"),
    }.get(operation)
    if parameter_names is None:
        raise ValueError(f"unsupported StarVLA action transform operation: {operation}")
    allowed = {"operation", "index", *parameter_names}
    unknown = set(value) - allowed
    missing = allowed - set(value)
    if unknown or missing:
        raise ValueError(
            f"invalid {operation} action transform fields; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    parameters = tuple(_finite_number(value, name) for name in parameter_names)
    return ActionTransform(operation=operation, index=index, parameters=parameters)


def _build_example(
    features: Mapping[str, Any],
    config: InputConfig,
    norm_processor: Any,
    image_preprocessor: ImagePreprocessor,
) -> dict[str, Any]:
    example: dict[str, Any] = {
        "image": [image_preprocessor(_array_feature(features, name)) for name in config.images],
        "lang": _string_feature(features, config.language),
    }
    if config.state is not None:
        example["state"] = _prepare_state(
            _array_feature(features, config.state),
            norm_processor,
        )
    return example


def _image_preprocessor(benchmark_name: str) -> ImagePreprocessor:
    preprocessors = {
        "robotwin": _resize_image_with_opencv,
        "robocasa": _resize_image_with_opencv,
        "libero_plus": _resize_image_with_opencv,
        "libero": _resize_image_with_pil,
        "libero_pro": _resize_image_with_pil,
        "calvin": _resize_calvin_image,
        "metaworld": _resize_image_with_opencv,
    }
    try:
        return preprocessors[benchmark_name]
    except KeyError:
        raise ValueError(f"StarVLA Policy does not support benchmark {benchmark_name!r}") from None


def _resize_image_with_opencv(image: np.ndarray) -> np.ndarray:
    height, width = STARVLA_IMAGE_SIZE
    return np.ascontiguousarray(cv.resize(image, (width, height), interpolation=cv.INTER_AREA))


def _resize_image_with_pil(image: np.ndarray) -> np.ndarray:
    if image.shape[:2] == STARVLA_IMAGE_SIZE:
        return image
    height, width = STARVLA_IMAGE_SIZE
    return np.ascontiguousarray(Image.fromarray(image).resize((width, height), Image.BILINEAR))


def _resize_calvin_image(image: np.ndarray) -> np.ndarray:
    from deployment.model_server.tools import image_tools

    height, width = STARVLA_IMAGE_SIZE
    resized = image_tools.resize_with_pad(image, height, width)
    return np.ascontiguousarray(image_tools.convert_to_uint8(resized))


def _prepare_state(state: np.ndarray, norm_processor: Any) -> np.ndarray:
    state = np.asarray(state)
    if state.ndim == 1:
        state = state[None, :]
    if state.ndim != 2 or state.shape[0] != 1:
        raise ValueError(f"StarVLA state input must have shape (D,) or (1, D); received {state.shape}")

    state_keys = tuple(norm_processor.state_keys)
    state_key_dims = getattr(norm_processor, "_state_key_dims", None)
    if not state_keys or not isinstance(state_key_dims, Mapping):
        raise RuntimeError("StarVLA normalization processor does not expose state feature dimensions")

    ordered_dims = tuple(_state_key_dimension(state_key_dims, key) for key in state_keys)
    expected_dim = sum(ordered_dims)
    if state.shape[1] != expected_dim:
        raise ValueError(
            f"StarVLA raw state dimension is {state.shape[1]}; expected {expected_dim} "
            f"for configured state keys {state_keys}"
        )

    values: dict[str, np.ndarray] = {}
    cursor = 0
    for key, dimension in zip(state_keys, ordered_dims, strict=True):
        values[key] = np.ascontiguousarray(state[:, cursor : cursor + dimension])
        cursor += dimension
    transformed = norm_processor.transform.apply(values)
    if "state" in transformed:
        prepared = _numpy_value(transformed["state"])
    else:
        missing = [key for key in state_keys if key not in transformed]
        if missing:
            raise RuntimeError(f"StarVLA state transform removed configured keys: {missing}")
        prepared = np.concatenate([_numpy_value(transformed[key]) for key in state_keys], axis=-1)
    if prepared.ndim != 2 or prepared.shape[0] != 1:
        raise ValueError(f"StarVLA transformed state must have shape (1, D); received {prepared.shape}")
    return np.ascontiguousarray(prepared)


def _state_key_dimension(state_key_dims: Mapping[str, Any], key: str) -> int:
    dimension = state_key_dims.get(key)
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise RuntimeError(f"StarVLA state feature {key!r} has invalid dimension {dimension!r}")
    return dimension


def _numpy_value(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _mapping(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"StarVLA config field {key} must be a mapping")
    return dict(value)


def _require_keys(
    config: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str],
    field: str,
) -> None:
    if any(not isinstance(key, str) for key in config):
        raise ValueError(f"StarVLA config field {field} keys must be strings")
    missing = required - set(config)
    unknown = set(config) - required - optional
    if missing or unknown:
        raise ValueError(
            f"invalid StarVLA config field {field}; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _optional_mapping(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"StarVLA config field {key} must be a mapping when provided")
    return dict(value)


def _string_list(config: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = config.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(f"StarVLA config field {key} must be a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"StarVLA config field {key} must contain non-empty strings")
    return tuple(item.strip() for item in value)


def _required_string(config: Mapping[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"StarVLA config field {key} must be a non-empty string")
    return value.strip()


def _optional_string(config: Mapping[str, Any], key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    return _required_string(config, key)


def _integer(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"StarVLA config field {key} must be an integer")
    return value


def _positive_integer(config: Mapping[str, Any], key: str) -> int:
    value = _integer(config, key)
    if value < 1:
        raise ValueError(f"StarVLA config field {key} must be positive")
    return value


def _finite_number(config: Mapping[str, Any], key: str) -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"StarVLA config field {key} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"StarVLA config field {key} must be finite")
    return number


def _array_feature(features: Mapping[str, Any], name: str) -> np.ndarray:
    value = features.get(name)
    if not isinstance(value, np.ndarray):
        raise TypeError(f"StarVLA input feature {name!r} must be a numpy array")
    return np.ascontiguousarray(value)


def _string_feature(features: Mapping[str, Any], name: str) -> str:
    value = features.get(name)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"StarVLA input feature {name!r} must be a non-empty string")
    return value


def main() -> None:
    run(StarVLAPolicy())


if __name__ == "__main__":
    main()
