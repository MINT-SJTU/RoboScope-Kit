# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import socket

import numpy as np
import pytest

from roboscope_kit.protocol import Encoding, receive_frame, send_message


@pytest.mark.parametrize("encoding", ["json", "msgpack"])
def test_policy_frame_preserves_array_shape_and_dtype(encoding: Encoding) -> None:
    arrays = {
        "zero_dim": np.array(3, dtype=np.int16),
        "one_element": np.array([3], dtype=np.int16),
        "strided": np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2],
    }
    sender, receiver = socket.socketpair()
    with sender, receiver:
        send_message(sender, {**arrays, "numpy_scalar": np.float32(1.25)}, encoding)
        frame = receive_frame(receiver)

    assert frame is not None
    message, received_encoding = frame
    assert received_encoding == encoding
    for name, expected in arrays.items():
        actual = message[name]
        assert isinstance(actual, np.ndarray)
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert actual.flags.writeable
        np.testing.assert_array_equal(actual, expected)
    assert type(message["numpy_scalar"]) is float
    assert message["numpy_scalar"] == 1.25
