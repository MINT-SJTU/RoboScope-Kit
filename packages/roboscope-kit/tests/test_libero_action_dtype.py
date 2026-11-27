# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for LIBERO float32/float64 action contract and transport preservation."""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path

import numpy as np
import pytest
from examples.openvla.openvla_policy import environment_action

from roboscope_kit import (
    ActionChunk,
    ActionSpec,
    BenchmarkName,
    EpisodeContext,
    Observation,
    Policy,
    PolicyCapabilities,
    PolicyClient,
    PolicyServer,
    PolicyState,
    SessionContext,
    get_benchmark_spec,
    validate_action_chunk,
)

LIBERO_BENCHMARKS = (
    (BenchmarkName.LIBERO, "libero_goal"),
    (BenchmarkName.LIBERO_PLUS, "libero_goal"),
    (BenchmarkName.LIBERO_PRO, "libero_goal"),
)


def test_action_spec_additional_dtypes_validation_and_serialization() -> None:
    default_spec = ActionSpec("action.robot", "desc", 7)
    assert default_spec.dtype == "float32"
    assert default_spec.additional_dtypes == ()
    assert "additional_dtypes" not in default_spec.to_dict()
    assert ActionSpec.from_dict(default_spec.to_dict()) == default_spec
    assert ActionSpec.from_dict({**default_spec.to_dict(), "additional_dtypes": None}) == default_spec

    dual_spec = ActionSpec("action.robot", "desc", 7, additional_dtypes=("float64",))
    payload = dual_spec.to_dict()
    assert payload["additional_dtypes"] == ["float64"]
    assert ActionSpec.from_dict(payload) == dual_spec

    with pytest.raises(ValueError, match="additional action dtypes cannot be empty"):
        ActionSpec("action.robot", "desc", 7, additional_dtypes=("",))

    with pytest.raises(ValueError, match="action dtypes must be unique"):
        ActionSpec("action.robot", "desc", 7, additional_dtypes=("float32",))

    with pytest.raises(ValueError, match="action dtypes must be unique"):
        ActionSpec("action.robot", "desc", 7, additional_dtypes=("float64", "float64"))


@pytest.mark.parametrize(("benchmark_name", "suite"), LIBERO_BENCHMARKS)
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("shape", ((7,), (4, 7)))
def test_validate_action_chunk_accepts_float32_and_float64_for_libero_family(
    benchmark_name: BenchmarkName,
    suite: str,
    dtype: type[np.floating],
    shape: tuple[int, ...],
) -> None:
    spec = get_benchmark_spec(benchmark_name, suite)
    validate_action_chunk(ActionChunk(actions=np.zeros(shape, dtype=dtype)), spec)


@pytest.mark.parametrize(("benchmark_name", "suite"), LIBERO_BENCHMARKS)
@pytest.mark.parametrize("bad_dtype", (np.float16, np.int32))
def test_validate_action_chunk_rejects_other_dtypes_for_libero_family(
    benchmark_name: BenchmarkName,
    suite: str,
    bad_dtype: type[np.number],
) -> None:
    spec = get_benchmark_spec(benchmark_name, suite)
    with pytest.raises(TypeError, match="float32 or float64"):
        validate_action_chunk(ActionChunk(actions=np.zeros(7, dtype=bad_dtype)), spec)


@pytest.mark.parametrize(
    ("benchmark_name", "suite", "dim"),
    (
        (BenchmarkName.ROBOCASA, "pick_place", 29),
        (BenchmarkName.ROBOTWIN, "demo_clean", 14),
        (BenchmarkName.CALVIN, "ABCD_D", 7),
    ),
)
def test_validate_action_chunk_keeps_other_benchmarks_float32_only(
    benchmark_name: BenchmarkName,
    suite: str,
    dim: int,
) -> None:
    spec = get_benchmark_spec(benchmark_name, suite)
    validate_action_chunk(ActionChunk(actions=np.zeros(dim, dtype=np.float32)), spec)
    with pytest.raises(TypeError, match="dtype float32"):
        validate_action_chunk(ActionChunk(actions=np.zeros(dim, dtype=np.float64)), spec)


@pytest.mark.parametrize(("benchmark_name", "suite"), LIBERO_BENCHMARKS)
@pytest.mark.parametrize("encoding", ("json", "msgpack"))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("chunk_size", (1, 4))
def test_policy_transport_preserves_libero_action_dtype_and_bytes(
    benchmark_name: BenchmarkName,
    suite: str,
    encoding: str,
    dtype: type[np.floating],
    chunk_size: int,
) -> None:
    base = np.array(
        [0.1234567890123456, -0.9876543210987654, 0.3333333333333333, 0.1, -0.2, 0.3, -1.0],
        dtype=dtype,
    )
    expected = base if chunk_size == 1 else np.stack([base + idx * dtype(1e-7) for idx in range(chunk_size)])

    class _FixedActionPolicy(Policy):
        def load(self, _context: SessionContext) -> PolicyCapabilities:
            return PolicyCapabilities(
                supported_benchmarks=(benchmark_name.value,),
                required_features=("language.instruction",),
                supported_action_chunk_sizes=(chunk_size,),
            )

        def act(self, _observation: Observation) -> ActionChunk:
            return ActionChunk(actions=expected.copy())

    with tempfile.TemporaryDirectory(prefix="rsk-") as sock_dir:
        socket_path = Path(sock_dir) / "p.sock"
        server = PolicyServer(_FixedActionPolicy(), socket_path)
        thread = threading.Thread(target=server.serve, daemon=True)
        thread.start()

        try:
            spec = get_benchmark_spec(benchmark_name, suite)
            client = PolicyClient(socket_path, encoding=encoding)
            try:
                client.wait_until_ready(timeout_s=5.0)
                client.load(SessionContext(session_id="s1", checkpoint_path="/unused", benchmark=spec))
                client.reset(EpisodeContext(episode_id="ep-0", task_id="0"))
                observation = Observation(
                    episode_id="ep-0",
                    step=0,
                    features={
                        "images.primary": np.zeros((256, 256, 3), dtype=np.uint8),
                        "images.wrist": np.zeros((256, 256, 3), dtype=np.uint8),
                        "state.proprioception": np.zeros(16, dtype=np.float32),
                        "language.instruction": "pick up the bowl",
                    },
                )
                returned = client.act(observation)
                assert returned.actions.dtype == expected.dtype
                assert returned.actions.shape == expected.shape
                assert returned.actions.tobytes() == expected.tobytes()
                client.end_episode("ep-0")
                client.unload()
            finally:
                client.close()
        finally:
            server.state = PolicyState.CLOSED
            thread.join(timeout=5.0)


def test_openvla_environment_action_preserves_float64_precision() -> None:
    raw = np.array(
        [0.1234567890123456, -0.9876543210987654, 0.3333333333333333, 0.1, -0.2, 0.3, 0.8],
        dtype=np.float64,
    )
    converted = environment_action(raw)
    assert converted.dtype == np.float64
    assert converted[:6].tobytes() == raw[:6].tobytes()
    assert converted[6] == -1.0
    assert converted[:6].tobytes() != raw[:6].astype(np.float32).astype(np.float64).tobytes()
