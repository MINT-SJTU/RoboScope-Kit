# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""RoboTwin's standalone PolicySocketClient negotiates MessagePack and falls back to JSON."""

from __future__ import annotations

import socket
import sys
import tempfile
import threading
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from roboscope_kit import Policy, PolicyCapabilities, PolicyServer, PolicyState, SessionContext
from roboscope_kit.protocol import receive_frame, send_message


@pytest.fixture
def launcher(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3]))
    monkeypatch.setitem(sys.modules, "yaml", object())
    monkeypatch.delenv("ROBOSCOPE_POLICY_ENCODING", raising=False)
    from benchmarks.robotwin import roboscope_robotwin_launcher

    return roboscope_robotwin_launcher


class _Recorder:
    """Wraps a socket and records the first bytes of every frame the client sends."""

    def __init__(self, connection: socket.socket) -> None:
        self.connection = connection
        self.sent: list[bytes] = []

    def sendall(self, data: bytes) -> None:
        self.sent.append(bytes(data[:4]))
        self.connection.sendall(data)

    def recv(self, size: int) -> bytes:
        return self.connection.recv(size)

    def close(self) -> None:
        self.connection.close()


def _client(launcher: ModuleType, connection: socket.socket) -> tuple[object, _Recorder]:
    client = launcher.PolicySocketClient.__new__(launcher.PolicySocketClient)
    recorder = _Recorder(connection)
    client.connection = recorder
    client._msgpack = None
    client.encoding = None
    return client, recorder


def _echo_server(connection: socket.socket, advertise: bool) -> threading.Thread:
    """Speaks the kit protocol; answers ``health`` like PolicyServer or like a server without MessagePack."""

    def serve() -> None:
        while True:
            received = receive_frame(connection)
            if received is None:
                return
            request, encoding = received
            payload = request["payload"]
            if request["operation"] == "health":
                result = {"state": "created"}
                if advertise and "encodings" in payload:
                    result["encodings"] = ["json", "msgpack"]
            else:
                result = {"seen_encoding": encoding, "payload": payload}
            send_message(
                connection,
                {"version": 1, "request_id": request["request_id"], "ok": True, "result": result},
                encoding,
            )

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return thread


def _stop(near: socket.socket, far: socket.socket, server: threading.Thread) -> None:
    near.close()  # the echo server sees EOF and returns
    server.join(timeout=5.0)
    far.close()


def _payload() -> dict[str, object]:
    return {
        "images": np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)[:, :, ::2],
        "state": np.array([1.5, -2.25], dtype=np.float32),
        "ids": np.array([2**40, -(2**40)], dtype=np.int64),
        "empty": np.empty((0, 3), dtype=np.float64),
        "scalar": np.float32(0.5),
        "text": "pick up the cup",
    }


def _assert_round_trip(result: dict[str, object], sent: dict[str, object]) -> None:
    echoed = result["payload"]
    assert echoed["text"] == "pick up the cup"
    assert echoed["scalar"] == 0.5
    for name in ("images", "state", "ids", "empty"):
        actual, expected = echoed[name], sent[name]
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)


def test_client_uses_messagepack_after_the_server_lists_it(launcher: ModuleType) -> None:
    near, far = socket.socketpair()
    client, recorder = _client(launcher, near)
    server = _echo_server(far, advertise=True)
    try:
        sent = _payload()
        result = client.request("act", sent)
    finally:
        _stop(near, far, server)

    assert client.encoding == "msgpack"
    assert result["seen_encoding"] == "msgpack"
    _assert_round_trip(result, sent)
    # The health probe is JSON (it cannot know yet); the act request is the RSKM frame.
    assert recorder.sent[0] != b"RSKM"
    assert recorder.sent[1] == b"RSKM"


def test_client_falls_back_to_json_when_the_server_does_not_list_messagepack(launcher: ModuleType) -> None:
    near, far = socket.socketpair()
    client, recorder = _client(launcher, near)
    server = _echo_server(far, advertise=False)
    try:
        sent = _payload()
        result = client.request("act", sent)
    finally:
        _stop(near, far, server)

    assert client.encoding == "json"
    assert result["seen_encoding"] == "json"
    _assert_round_trip(result, sent)
    assert b"RSKM" not in recorder.sent


def test_json_preference_sends_no_probe(launcher: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROBOSCOPE_POLICY_ENCODING", "json")
    near, far = socket.socketpair()
    client, recorder = _client(launcher, near)
    server = _echo_server(far, advertise=True)
    try:
        result = client.request("act", _payload())
    finally:
        _stop(near, far, server)

    assert client.encoding == "json"
    assert result["seen_encoding"] == "json"
    assert len(recorder.sent) == 1


def test_missing_msgpack_package_keeps_json(launcher: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher, "_load_msgpack", lambda: None)
    near, far = socket.socketpair()
    client, recorder = _client(launcher, near)
    server = _echo_server(far, advertise=True)
    try:
        result = client.request("act", _payload())
    finally:
        _stop(near, far, server)

    assert client.encoding == "json"
    assert result["seen_encoding"] == "json"
    assert len(recorder.sent) == 1


def test_unknown_encoding_preference_is_rejected(
    launcher: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ROBOSCOPE_POLICY_ENCODING", "protobuf")
    near, far = socket.socketpair()
    client, _ = _client(launcher, near)
    server = _echo_server(far, advertise=True)
    try:
        with pytest.raises(ValueError, match="json or msgpack"):
            client.request("health", {})
    finally:
        _stop(near, far, server)


def test_client_negotiates_messagepack_with_a_real_policy_server(launcher: ModuleType) -> None:
    class _Idle(Policy):
        def load(self, _context: SessionContext) -> PolicyCapabilities:
            raise NotImplementedError

        def act(self, _observation: object) -> object:
            raise NotImplementedError

    with tempfile.TemporaryDirectory(prefix="rsk-") as sock_dir:
        socket_path = Path(sock_dir) / "p.sock"
        server = PolicyServer(_Idle(), socket_path)
        thread = threading.Thread(target=server.serve, daemon=True)
        thread.start()
        client = launcher.PolicySocketClient(str(socket_path))
        try:
            result = client.request("health", {})
            assert client.encoding == "msgpack"
            assert result["state"] == "created"
        finally:
            client.close()
            server.state = PolicyState.CLOSED
            thread.join(timeout=5.0)
