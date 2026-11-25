# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Negotiation of camera image sizes between a benchmark and a Policy.

A benchmark describes, per image feature, the size its official protocol uses (``canonical``) and the
sizes its simulator can render natively. A Policy lists the sizes it accepts in order of preference,
optionally ending with :data:`ANY_SIZE`. The runtime picks one size per feature; it never resamples.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Union

from roboscope_kit.contracts import BenchmarkSpec, FeatureKind, FeatureSpec

ANY_SIZE = "any"

ImageSize = tuple[int, int]
SizePreference = Union[ImageSize, str]  # noqa: UP007 -- benchmark environments include Python 3.10
SizePreferences = tuple[SizePreference, ...]


def _image_size(value: Any, where: str) -> ImageSize:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 2
        or any(type(dimension) is not int or dimension < 1 for dimension in value)
    ):
        raise ValueError(f"{where} must be a (height, width) pair of positive integers; got {value!r}")
    return (value[0], value[1])


def normalize_size_preferences(value: Any, where: str) -> SizePreferences:
    """Validate one ordered preference list: sizes, optionally ending with ``ANY_SIZE``."""

    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence) or not value:
        raise ValueError(f"{where} must be a non-empty list of (height, width) sizes")
    preferences: list[SizePreference] = []
    for index, item in enumerate(value):
        if item == ANY_SIZE:
            if index != len(value) - 1:
                raise ValueError(f"{where}: {ANY_SIZE!r} may only be the last preference")
            preferences.append(ANY_SIZE)
            continue
        size = _image_size(item, f"{where}[{index}]")
        if size in preferences:
            raise ValueError(f"{where}: size {size[0]}x{size[1]} is listed twice")
        preferences.append(size)
    return tuple(preferences)


def normalize_image_sizes(value: Any) -> SizePreferences | dict[str, SizePreferences] | None:
    """Normalize ``PolicyCapabilities.image_sizes``: a list for every camera or a per-feature dict."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        if not value:
            raise ValueError("image_sizes must not be an empty mapping")
        for feature in value:
            if not isinstance(feature, str) or not feature:
                raise ValueError(f"image_sizes keys must be feature names; got {feature!r}")
        return {
            feature: normalize_size_preferences(preferences, f"image_sizes[{feature!r}]")
            for feature, preferences in value.items()
        }
    return normalize_size_preferences(value, "image_sizes")


def image_sizes_to_dict(value: SizePreferences | dict[str, SizePreferences]) -> Any:
    def encode(preferences: SizePreferences) -> list[Any]:
        return [item if item == ANY_SIZE else list(item) for item in preferences]

    if isinstance(value, dict):
        return {feature: encode(preferences) for feature, preferences in value.items()}
    return encode(value)


@dataclass(frozen=True)
class CameraSupport:
    """Sizes one benchmark camera can provide.

    ``canonical`` is the size of the official protocol, or ``None`` when the benchmark does not fix
    it. ``min_size``/``max_size`` bound the sizes the simulator renders natively; when they are
    ``None`` the camera renders only its canonical size.
    """

    canonical: ImageSize | None
    min_size: ImageSize | None = None
    max_size: ImageSize | None = None

    def __post_init__(self) -> None:
        if (self.min_size is None) != (self.max_size is None):
            raise ValueError("camera support needs both min_size and max_size, or neither")
        if self.min_size is not None and self.max_size is not None and (
            self.min_size[0] > self.max_size[0] or self.min_size[1] > self.max_size[1]
        ):
            raise ValueError("camera support min_size must not exceed max_size")

    def renders(self, size: ImageSize) -> bool:
        if self.min_size is None or self.max_size is None:
            return size == self.canonical
        return (
            self.min_size[0] <= size[0] <= self.max_size[0]
            and self.min_size[1] <= size[1] <= self.max_size[1]
        )

    def describe(self) -> str:
        if self.min_size is None or self.max_size is None:
            if self.canonical is None:
                return "only its own (unspecified) size"
            return f"only {self.canonical[0]}x{self.canonical[1]}"
        return (
            f"heights {self.min_size[0]}-{self.max_size[0]} and widths "
            f"{self.min_size[1]}-{self.max_size[1]}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical": list(self.canonical) if self.canonical is not None else None,
            "min_size": list(self.min_size) if self.min_size is not None else None,
            "max_size": list(self.max_size) if self.max_size is not None else None,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CameraSupport:
        if not isinstance(value, Mapping):
            raise ValueError(f"camera support must be an object; got {value!r}")

        def size(key: str) -> ImageSize | None:
            item = value.get(key)
            return _image_size(item, f"camera support {key}") if item is not None else None

        return cls(canonical=size("canonical"), min_size=size("min_size"), max_size=size("max_size"))


CameraSupportTable = dict[str, CameraSupport]


def spatial_size(feature: FeatureSpec) -> ImageSize | None:
    """The (height, width) of an image feature's declared shape, or ``None`` when it is dynamic."""

    shape = feature.shape
    if shape is None or len(shape) < 3:
        return None
    height, width = shape[-3], shape[-2]
    if height is None or width is None:
        return None
    return (height, width)


def fixed_camera_support(benchmark: BenchmarkSpec) -> CameraSupportTable:
    """Every image feature renders only the size its spec declares."""

    return {
        feature.name: CameraSupport(canonical=spatial_size(feature))
        for feature in benchmark.features
        if feature.kind is FeatureKind.IMAGE
    }


def camera_support_to_dict(table: Mapping[str, CameraSupport]) -> dict[str, Any]:
    return {feature: support.to_dict() for feature, support in table.items()}


def camera_support_from_dict(value: Mapping[str, Any]) -> CameraSupportTable:
    if not isinstance(value, Mapping):
        raise ValueError(f"camera support table must be an object; got {value!r}")
    return {str(feature): CameraSupport.from_dict(support) for feature, support in value.items()}


def negotiate_image_sizes(
    benchmark: BenchmarkSpec,
    requested: SizePreferences | dict[str, SizePreferences] | None,
    camera_support: Mapping[str, CameraSupport] | None = None,
) -> dict[str, ImageSize]:
    """Resolve one size per image feature; return only the features that differ from canonical."""

    image_features = [feature for feature in benchmark.features if feature.kind is FeatureKind.IMAGE]
    names = {feature.name for feature in image_features}
    if isinstance(requested, dict):
        unknown = sorted(set(requested) - names)
        if unknown:
            raise ValueError(
                f"image_sizes names features that are not {benchmark.name} images: {', '.join(unknown)}"
            )
    support_table = fixed_camera_support(benchmark)
    if camera_support is not None:
        support_table.update({name: camera_support[name] for name in names if name in camera_support})
    resolved: dict[str, ImageSize] = {}
    for feature in image_features:
        if requested is None:
            preferences: SizePreferences = (ANY_SIZE,)
        elif isinstance(requested, dict):
            preferences = requested.get(feature.name, (ANY_SIZE,))
        else:
            preferences = requested
        support = support_table[feature.name]
        chosen: ImageSize | None = None
        accepted = False
        for preference in preferences:
            if preference == ANY_SIZE:
                chosen, accepted = support.canonical, True
                break
            if support.renders(preference):  # type: ignore[arg-type]
                chosen, accepted = preference, True  # type: ignore[assignment]
                break
        if not accepted:
            offered = ", ".join(
                ANY_SIZE if item == ANY_SIZE else f"{item[0]}x{item[1]}" for item in preferences
            )
            raise ValueError(
                f"{benchmark.name} {feature.name}: the benchmark renders {support.describe()}; "
                f"the Policy accepts [{offered}]"
            )
        if chosen is not None and chosen != support.canonical:
            resolved[feature.name] = chosen
    return resolved


def apply_image_sizes(benchmark: BenchmarkSpec, image_sizes: Mapping[str, ImageSize]) -> BenchmarkSpec:
    """Return the spec whose image features carry the negotiated spatial sizes."""

    if not image_sizes:
        return benchmark
    features = []
    for feature in benchmark.features:
        size = image_sizes.get(feature.name)
        if size is None:
            features.append(feature)
            continue
        if feature.shape is None or len(feature.shape) < 3:
            raise ValueError(f"{benchmark.name} {feature.name} has no spatial shape to resize")
        shape = (*feature.shape[:-3], size[0], size[1], feature.shape[-1])
        features.append(replace(feature, shape=shape))
    return replace(benchmark, features=tuple(features))
