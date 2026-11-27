# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import socket
import struct

import msgpack
import numpy as np
import pytest

from roboscope_kit.protocol import (
    MAX_MESSAGE_BYTES,
    PROTOCOL_VERSION,
    ProtocolError,
    receive_frame,
    send_message,
    validate_version,
)


def _connected_pair() -> tuple[socket.socket, socket.socket]:
    return socket.socketpair()


def _send_raw(sender: socket.socket, payload: bytes, *, msgpack_frame: bool = False) -> None:
    prefix = b"RSKM" if msgpack_frame else b""
    sender.sendall(prefix + struct.pack(">I", len(payload)) + payload)


@pytest.mark.parametrize("encoding", ["json", "msgpack"])
def test_round_trip_preserves_arrays_bytes_scalars_and_tuples(encoding: str) -> None:
    source = np.arange(12, dtype=np.uint8).reshape(3, 4)
    non_contiguous = source[:, ::2]
    message = {
        "nested": {"items": [1, {"ok": True}]},
        "arrays": {
            "uint8": source,
            "float32": np.array([1.5, -2.25], dtype=np.float32),
            "int64": np.array([2**40, -(2**40)], dtype=np.int64),
            "bool": np.array([True, False], dtype=np.bool_),
            "empty": np.empty((0, 3), dtype=np.float32),
            "view": non_contiguous,
        },
        "bytes": b"\x00\xffnot-utf8\x80",
        "scalars": {"int": np.int64(7), "float": np.float32(1.5), "bool": np.bool_(True)},
        "tuple": (1, "two"),
    }

    sender, receiver = _connected_pair()
    try:
        send_message(sender, message, encoding=encoding)
        received, detected = receive_frame(receiver)
    finally:
        sender.close()
        receiver.close()

    assert detected == encoding
    assert received is not None
    assert received["nested"] == {"items": [1, {"ok": True}]}
    assert received["bytes"] == message["bytes"]
    assert received["tuple"] == [1, "two"]
    assert received["scalars"] == {"int": 7, "float": 1.5, "bool": True}
    for name, expected in message["arrays"].items():
        actual = received["arrays"][name]
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)
        assert actual.flags.writeable


def test_peer_close_before_any_frame_returns_none() -> None:
    sender, receiver = _connected_pair()
    try:
        sender.close()
        assert receive_frame(receiver) is None
    finally:
        receiver.close()


@pytest.mark.parametrize(
    "prefix",
    [b"\x00\x00", b"RSKM\x00"],
    ids=["inside-json-header", "inside-msgpack-header"],
)
def test_peer_close_inside_header_raises(prefix: bytes) -> None:
    sender, receiver = _connected_pair()
    try:
        sender.sendall(prefix)
        sender.close()
        with pytest.raises(ProtocolError, match="inside a message header"):
            receive_frame(receiver)
    finally:
        receiver.close()


def test_peer_close_inside_body_raises() -> None:
    sender, receiver = _connected_pair()
    try:
        sender.sendall(struct.pack(">I", 5) + b"ab")
        sender.close()
        with pytest.raises(ProtocolError, match="before message body"):
            receive_frame(receiver)
    finally:
        receiver.close()


def test_oversize_header_raises() -> None:
    sender, receiver = _connected_pair()
    try:
        sender.sendall(struct.pack(">I", MAX_MESSAGE_BYTES + 1))
        with pytest.raises(ProtocolError, match="maximum is"):
            receive_frame(receiver)
    finally:
        sender.close()
        receiver.close()


@pytest.mark.parametrize(
    ("dtype", "shape", "data"),
    [
        ("<u1", [4], b"\x01\x02"),
        ("<u1", [-1], b"\x01"),
        ("|O", [1], b""),
    ],
    ids=["length-mismatch", "negative-dimension", "object-dtype"],
)
@pytest.mark.parametrize("msgpack_frame", [False, True], ids=["json", "msgpack"])
def test_malformed_arrays_raise(
    dtype: str, shape: list[int], data: bytes, msgpack_frame: bool
) -> None:
    array = {"__roboscope_type__": "ndarray", "dtype": dtype, "shape": shape}
    if msgpack_frame:
        payload = msgpack.packb({**array, "data": data}, use_bin_type=True)
    else:
        payload = (
            '{"__roboscope_type__":"ndarray","dtype":'
            + repr(dtype).replace("'", '"')
            + ',"shape":'
            + str(shape).replace(" ", "")
            + ',"data":"'
            + base64.b64encode(data).decode("ascii")
            + '"}'
        ).encode("utf-8")

    sender, receiver = _connected_pair()
    try:
        _send_raw(sender, payload, msgpack_frame=msgpack_frame)
        with pytest.raises(ProtocolError):
            receive_frame(receiver)
    finally:
        sender.close()
        receiver.close()


def test_json_rejects_non_string_keys_on_send() -> None:
    sender, receiver = _connected_pair()
    try:
        with pytest.raises(TypeError, match="keys must be strings"):
            send_message(sender, {1: "value"}, encoding="json")
    finally:
        sender.close()
        receiver.close()


def test_messagepack_non_string_key_raises_on_receive() -> None:
    sender, receiver = _connected_pair()
    try:
        send_message(sender, {1: "value"}, encoding="msgpack")
        with pytest.raises(ProtocolError):
            receive_frame(receiver)
    finally:
        sender.close()
        receiver.close()


def test_validate_version_accepts_only_current_version() -> None:
    validate_version({"version": PROTOCOL_VERSION})
    with pytest.raises(ProtocolError, match="unsupported protocol version"):
        validate_version({"version": PROTOCOL_VERSION + 1})
