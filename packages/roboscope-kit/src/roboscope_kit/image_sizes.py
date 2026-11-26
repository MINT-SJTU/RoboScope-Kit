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

# Image-size features a Policy server understands, reported in its ``health`` reply. A server that
# reports none (roboscope-kit 0.8.0b77) understands ``fixed_shape_ranges``: size ranges for features
# whose spec fixes a spatial shape. Add a name here whenever camera support gains a meaning an older
# server would misread, and teach ``camera_support_requirements`` about it.
# ``open_shape_ranges``: size ranges for features whose spec leaves the shape open; the feature
# takes the negotiated ``(height, width)`` (CALVIN, ManiSkill).
# ``spatial_features``: a camera's other features (depth, point cloud) take its negotiated size
# (``CameraSupport.spatial_features``, RLBench).
# ``square_sizes``: a size range limited to square sizes (``CameraSupport.square``, LIBERO-Plus).
# ``shared_sizes``: cameras of one ``CameraSupport.size_group`` render one size, negotiated jointly
# (RoboCasa365).
IMAGE_SIZE_FEATURES: tuple[str, ...] = (
    "fixed_shape_ranges",
    "open_shape_ranges",
    "spatial_features",
    "square_sizes",
    "shared_sizes",
)
BASELINE_IMAGE_SIZE_FEATURES = frozenset({"fixed_shape_ranges"})

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
    ``None`` the camera renders only its canonical size. ``spatial_features`` names the other
    observation features rendered by this camera, such as depth or a point cloud, whose first two
    dimensions are the camera's (height, width) and follow its negotiated size. ``square``
    limits the camera to sizes whose height equals their width. Cameras with the same
    ``size_group`` render one size, which the negotiation picks for all of them together; they must
    have equal ``canonical``, ``min_size``, ``max_size`` and ``square``, with the canonical size inside
    the range, so that a Policy without ``image_sizes`` or with ``[ANY_SIZE]`` always negotiates.
    """

    canonical: ImageSize | None
    min_size: ImageSize | None = None
    max_size: ImageSize | None = None
    spatial_features: tuple[str, ...] = ()
    square: bool = False
    size_group: str | None = None

    def __post_init__(self) -> None:
        if (self.min_size is None) != (self.max_size is None):
            raise ValueError("camera support needs both min_size and max_size, or neither")
        if self.min_size is not None and self.max_size is not None and (
            self.min_size[0] > self.max_size[0] or self.min_size[1] > self.max_size[1]
        ):
            raise ValueError("camera support min_size must not exceed max_size")
        if not isinstance(self.spatial_features, tuple) or any(
            not isinstance(feature, str) or not feature for feature in self.spatial_features
        ):
            raise ValueError("camera support spatial_features must be a tuple of feature names")
        if self.size_group is not None and (not isinstance(self.size_group, str) or not self.size_group):
            raise ValueError("camera support size_group must be a non-empty string")

    def renders(self, size: ImageSize) -> bool:
        if self.min_size is None or self.max_size is None:
            return size == self.canonical
        if self.square and size[0] != size[1]:
            return False
        return (
            self.min_size[0] <= size[0] <= self.max_size[0]
            and self.min_size[1] <= size[1] <= self.max_size[1]
        )

    def describe(self) -> str:
        if self.min_size is None or self.max_size is None:
            if self.canonical is None:
                return "only its own (unspecified) size"
            return f"only {self.canonical[0]}x{self.canonical[1]}"
        if self.square:
            return f"square sizes {self.min_size[0]}-{self.max_size[0]}"
        return (
            f"heights {self.min_size[0]}-{self.max_size[0]} and widths "
            f"{self.min_size[1]}-{self.max_size[1]}"
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "canonical": list(self.canonical) if self.canonical is not None else None,
            "min_size": list(self.min_size) if self.min_size is not None else None,
            "max_size": list(self.max_size) if self.max_size is not None else None,
        }
        if self.spatial_features:
            payload["spatial_features"] = list(self.spatial_features)
        if self.square:
            payload["square"] = True
        if self.size_group is not None:
            payload["size_group"] = self.size_group
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CameraSupport:
        if not isinstance(value, Mapping):
            raise ValueError(f"camera support must be an object; got {value!r}")

        def size(key: str) -> ImageSize | None:
            item = value.get(key)
            return _image_size(item, f"camera support {key}") if item is not None else None

        spatial_features = value.get("spatial_features", [])
        if isinstance(spatial_features, (str, bytes, bytearray)) or not isinstance(
            spatial_features, Sequence
        ):
            raise ValueError(f"camera support spatial_features must be a list; got {spatial_features!r}")
        square = value.get("square", False)
        if type(square) is not bool:
            raise ValueError(f"camera support square must be a boolean; got {square!r}")
        size_group = value.get("size_group")
        if size_group is not None and (not isinstance(size_group, str) or not size_group):
            raise ValueError(f"camera support size_group must be a non-empty string; got {size_group!r}")
        return cls(
            canonical=size("canonical"),
            min_size=size("min_size"),
            max_size=size("max_size"),
            spatial_features=tuple(spatial_features),
            square=square,
            size_group=size_group,
        )


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


def camera_support_requirements(feature: FeatureSpec, support: CameraSupport) -> frozenset[str]:
    """The image-size features a Policy server needs to apply ``support`` correctly."""

    if support.min_size is None or support.max_size is None:
        return frozenset()
    # A size range for a feature whose spec fixes its shape, or leaves it open.
    needed = {"fixed_shape_ranges" if spatial_size(feature) is not None else "open_shape_ranges"}
    if support.spatial_features:
        needed.add("spatial_features")
    if support.square:
        # Older servers drop the flag and would accept any shape.
        needed.add("square_sizes")
    if support.size_group is not None:
        # Older servers negotiate each camera alone and could pick sizes the group cannot render.
        needed.add("shared_sizes")
    return frozenset(needed)


def camera_support_for_server(
    benchmark: BenchmarkSpec,
    camera_support: Mapping[str, CameraSupport],
    server_features: frozenset[str] | set[str],
) -> CameraSupportTable:
    """Keep the entries a server with ``server_features`` understands; others offer their canonical size."""

    withheld = withheld_camera_support(benchmark, camera_support, server_features)
    return {
        name: CameraSupport(canonical=support.canonical) if name in withheld else support
        for name, support in camera_support.items()
    }


def withheld_camera_support(
    benchmark: BenchmarkSpec,
    camera_support: Mapping[str, CameraSupport],
    server_features: frozenset[str] | set[str],
) -> dict[str, frozenset[str]]:
    """Each camera whose sizes a server with ``server_features`` cannot apply, with the features it lacks."""

    features = {feature.name: feature for feature in benchmark.features}
    withheld: dict[str, frozenset[str]] = {}
    for name, support in camera_support.items():
        feature = features.get(name)
        if feature is not None:
            missing = camera_support_requirements(feature, support) - server_features
            if missing:
                withheld[name] = frozenset(missing)
    return withheld


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
    def preferences_of(name: str) -> SizePreferences:
        if requested is None:
            return (ANY_SIZE,)
        if isinstance(requested, dict):
            return requested.get(name, (ANY_SIZE,))
        return requested

    groups: dict[str, list[str]] = {}
    for feature in image_features:
        group = support_table[feature.name].size_group
        if group is not None:
            groups.setdefault(group, []).append(feature.name)
    # A mapping names some cameras; a flat list names every camera.
    named = set(requested) if isinstance(requested, dict) else names
    shared = {
        name: size
        for group, members in groups.items()
        for name, size in _negotiate_group(
            benchmark.name,
            group,
            members,
            {name: preferences_of(name) for name in members},
            _group_support(benchmark.name, group, members, support_table),
            named,
        ).items()
    }
    resolved: dict[str, ImageSize] = {}
    for feature in image_features:
        preferences = preferences_of(feature.name)
        support = support_table[feature.name]
        if support.size_group is not None:
            if shared[feature.name] != support.canonical:
                resolved[feature.name] = shared[feature.name]
            continue
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


def _describe_preferences(preferences: SizePreferences) -> str:
    return ", ".join(ANY_SIZE if item == ANY_SIZE else f"{item[0]}x{item[1]}" for item in preferences)


def _group_support(
    benchmark: str, group: str, members: list[str], support_table: Mapping[str, CameraSupport]
) -> CameraSupport:
    """The sizes every camera of ``group`` shares; the group's cameras must declare the same ones."""

    shapes = {
        (support.canonical, support.min_size, support.max_size, support.square)
        for support in (support_table[name] for name in members)
    }
    if len(shapes) != 1:
        raise ValueError(
            f"{benchmark} size group {group!r} cameras {', '.join(members)} must share canonical, "
            "min_size, max_size and square"
        )
    support = support_table[members[0]]
    if support.canonical is not None and not support.renders(support.canonical):
        raise ValueError(f"{benchmark} size group {group!r} must render its canonical size")
    return support


def _negotiate_group(
    benchmark: str,
    group: str,
    members: list[str],
    preferences: Mapping[str, SizePreferences],
    support: CameraSupport,
    named: set[str],
) -> dict[str, ImageSize | None]:
    """Pick the one size every camera of ``group`` accepts and ``support`` renders.

    Candidates are each member's preferences in feature order, ``ANY_SIZE`` standing for the group's
    canonical size. A member accepts a size it lists, or the canonical size when its list ends with
    ``ANY_SIZE``. The first candidate every member accepts and the group renders wins. ``named`` holds
    the cameras the Policy lists sizes for; the others accept only the canonical size.
    """

    def accepts(name: str, size: ImageSize | None) -> bool:
        listed = preferences[name]
        if size == support.canonical:
            return size in listed or listed[-1] == ANY_SIZE
        return size is not None and size in listed and support.renders(size)

    candidates: list[ImageSize | None] = []
    for name in members:
        for preference in preferences[name]:
            size = support.canonical if preference == ANY_SIZE else preference
            if size not in candidates:
                candidates.append(size)  # type: ignore[arg-type]
    for size in candidates:
        if all(accepts(name, size) for name in members):
            return dict.fromkeys(members, size)
    own = "its own size" if support.canonical is None else f"{support.canonical[0]}x{support.canonical[1]}"
    offered = "; ".join(
        f"{name} [{_describe_preferences(preferences[name])}]" if name in named
        else f"{name} not listed ({own} only)"
        for name in members
    )
    raise ValueError(
        f"{benchmark} cameras {', '.join(members)} (size group {group!r}) render one shared size: "
        f"{support.describe()}; no size is accepted by every camera: the Policy accepts {offered}"
    )


def apply_image_sizes(
    benchmark: BenchmarkSpec,
    image_sizes: Mapping[str, ImageSize],
    camera_support: Mapping[str, CameraSupport] | None = None,
) -> BenchmarkSpec:
    """Return the spec whose image features carry the negotiated spatial sizes.

    A camera's ``spatial_features`` in ``camera_support`` take its size as their first two
    dimensions.
    """

    if not image_sizes:
        return benchmark
    followers: dict[str, ImageSize] = {}
    for camera, size in image_sizes.items():
        support = (camera_support or {}).get(camera)
        for name in support.spatial_features if support is not None else ():
            followers[name] = size
    features = []
    for feature in benchmark.features:
        size = image_sizes.get(feature.name)
        if size is not None:
            if feature.shape is None:
                # A spec that leaves the image shape open still fixes the negotiated spatial size.
                shape: tuple[int | None, ...] = (size[0], size[1], None)
            elif len(feature.shape) < 3:
                raise ValueError(f"{benchmark.name} {feature.name} has no spatial shape to resize")
            else:
                shape = (*feature.shape[:-3], size[0], size[1], feature.shape[-1])
            features.append(replace(feature, shape=shape))
        elif feature.name in followers:
            if feature.shape is None or len(feature.shape) < 2:
                raise ValueError(f"{benchmark.name} {feature.name} has no spatial shape to resize")
            size = followers[feature.name]
            features.append(replace(feature, shape=(size[0], size[1], *feature.shape[2:])))
        else:
            features.append(feature)
    return replace(benchmark, features=tuple(features))
