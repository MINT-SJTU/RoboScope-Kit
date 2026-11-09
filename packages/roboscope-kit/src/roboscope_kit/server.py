# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Unix socket server that exposes a user Policy to the simulation runtime."""

from __future__ import annotations

import json
import os
import socket
import threading
import traceback
import uuid
from collections.abc import Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from roboscope_kit import timing
from roboscope_kit.policy import (
    ActionChunk,
    EpisodeContext,
    Observation,
    Policy,
    PolicyCapabilities,
    SessionContext,
    validate_action_chunk,
    validate_observation,
    validate_policy_capabilities,
)
from roboscope_kit.protocol import (
    ENCODINGS,
    PROTOCOL_VERSION,
    Encoding,
    ProtocolError,
    receive_frame,
    send_message,
    validate_version,
)

DEFAULT_SOCKET_PATH = "/tmp/roboscope-inference.sock"


class PolicyState(str, Enum):  # noqa: UP042 -- benchmark environments include Python 3.10
    CREATED = "created"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    UNLOADING = "unloading"
    CLOSED = "closed"


class PolicyServer:
    def __init__(self, policy: Policy, socket_path: str | Path) -> None:
        self.policy = policy
        self.socket_path = Path(socket_path)
        self.state = PolicyState.CREATED
        self.context: SessionContext | None = None
        self.capabilities: PolicyCapabilities | None = None
        self._request_lock = threading.Lock()
        self._batch_lock = threading.Lock()
        self._batch_queue: Queue[tuple[dict[str, Any], Future] | None] = Queue()
        self._batch_thread: threading.Thread | None = None
        self._batch_stopping = False
        directory = os.environ.get("ROBOSCOPE_BATCH_METRICS_DIR")
        self._batch_metrics_path = Path(directory) if directory else None
        if self._batch_metrics_path is not None:
            if not self._batch_metrics_path.is_absolute():
                raise ValueError("ROBOSCOPE_BATCH_METRICS_DIR must be absolute")
            self._batch_metrics_path.mkdir(parents=True, exist_ok=True)
            self._batch_metrics_path /= f"batch-{os.getpid()}.jsonl"

    def serve(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                listener.bind(str(self.socket_path))
                self.socket_path.chmod(0o600)
                listener.listen(64)
                listener.settimeout(0.2)
                while self.state is not PolicyState.CLOSED:
                    try:
                        connection, _ = listener.accept()
                    except TimeoutError:
                        continue
                    threading.Thread(
                        target=self._serve_client,
                        args=(connection,),
                        daemon=True,
                    ).start()
        finally:
            with self._request_lock:
                self._close_policy()
            if self._batch_thread is not None:
                self._batch_thread.join()
            self.socket_path.unlink(missing_ok=True)

    def _serve_client(self, connection: socket.socket) -> None:
        with connection:
            self._serve_connection(connection)

    def _serve_connection(self, connection: socket.socket) -> None:
        while self.state is not PolicyState.CLOSED:
            request: dict[str, Any] | None = None
            encoding: Encoding = "json"
            try:
                received = receive_frame(connection)
                if received is None:
                    return
                request, encoding = received
            except Exception as exc:
                send_message(connection, self._error_response(exc, request_id=""), encoding)
                continue
            with timing.capture("server", operation=request.get("operation"),
                                request_id=request.get("request_id")):
                try:
                    if (request.get("operation") == "act" and self.capabilities is not None
                            and self.capabilities.max_batch_size > 1):
                        with timing.span("server.batch_wait"):
                            response = self._submit_action(request)
                    else:
                        with timing.span("server.lock_wait"):
                            self._request_lock.acquire()
                        try:
                            with timing.span("server.dispatch"):
                                response = self._handle(request)
                        finally:
                            self._request_lock.release()
                except Exception as exc:
                    response = self._error_response(exc, request_id=str(request.get("request_id") or ""))
                with timing.span("server.send"):
                    send_message(connection, response, encoding)

    def _submit_action(self, request: dict[str, Any]) -> dict[str, Any]:
        future: Future = Future()
        with self._batch_lock:
            queued = not self._batch_stopping
            if queued:
                self._batch_queue.put((request, future))
                if self._batch_thread is None:
                    self._batch_thread = threading.Thread(target=self._batch_actions, daemon=True)
                    self._batch_thread.start()
        if queued:
            return future.result()
        with self._request_lock:
            return self._handle(request)

    def _batch_actions(self) -> None:
        deferred = None
        while True:
            first = deferred if deferred is not None else self._batch_queue.get()
            deferred = None
            if first is None:
                return
            pending = [first]
            stopping = False
            episodes: set[str] = set()
            maximum = self.capabilities.max_batch_size if self.capabilities is not None else 1
            while len(pending) <= maximum:
                request = pending[-1][0]
                payload = request.get("payload")
                observation = payload.get("observation") if isinstance(payload, dict) else None
                episode = observation.get("episode_id") if isinstance(observation, dict) else None
                if not isinstance(episode, str) or not episode or episode in episodes:
                    if len(pending) > 1:
                        deferred = pending.pop()
                    break
                episodes.add(episode)
                if len(pending) == maximum:
                    break
                try:
                    item = self._batch_queue.get_nowait()
                except Empty:
                    break
                if item is None:
                    stopping = True
                    break
                pending.append(item)
            with self._request_lock:
                try:
                    self._handle_action_batch(pending)
                except BaseException as exc:
                    if self.state not in {PolicyState.CLOSED, PolicyState.UNLOADING}:
                        self.state = PolicyState.FAILED
                    for request, future in pending:
                        if not future.done():
                            future.set_result(self._error_response(
                                exc, request_id=str(request.get("request_id") or ""), operation="act",
                            ))
            if stopping:
                return

    def _handle_action_batch(self, pending: list[tuple[dict[str, Any], Future]]) -> None:
        valid = []
        for request, future in pending:
            try:
                validate_version(request)
            except ProtocolError:
                future.set_result(self._handle(request))
            else:
                valid.append((request, future))
        if len(valid) == 1:
            request, future = valid[0]
            future.set_result(self._handle(request))
        elif valid:
            request = {
                **valid[0][0], "operation": "act_batch",
                "payload": {"observations": [item[0]["payload"]["observation"] for item in valid]},
            }
            response = self._handle(request)
            for index, (original, future) in enumerate(valid):
                request_id = str(original.get("request_id") or uuid.uuid4())
                if response["ok"]:
                    result = self._success_response(
                        request_id, {"action_chunk": response["result"]["action_chunks"][index]},
                    )
                else:
                    result = {**response, "request_id": request_id,
                              "error": {**response["error"], "operation": "act"}}
                future.set_result(result)

    @contextmanager
    def _measure_batch(self, size: int) -> Iterator[None]:
        if self._batch_metrics_path is None:
            yield
            return
        ok = False
        with timing.collect() as durations:
            try:
                with timing.span("server.batch"):
                    yield
                ok = True
            finally:
                with self._batch_metrics_path.open("a") as stream:
                    stream.write(json.dumps({
                        "size": size, "policy_wall_s": durations["server.batch"], "ok": ok,
                        "prediction_wall_s": durations.get("policy.predict_total"),
                    }) + "\n")

    def _handle(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = str(request.get("request_id") or uuid.uuid4())
        operation = str(request.get("operation") or "")
        try:
            validate_version(request)
            result = self._dispatch(operation, request.get("payload") or {})
            return self._success_response(request_id, result)
        except Exception as exc:
            if (
                operation in {"load", "reset", "act", "act_batch", "observe", "end_episode"}
                and not isinstance(exc, ProtocolError)
            ):
                self.state = PolicyState.FAILED
            return self._error_response(exc, request_id=request_id, operation=operation)

    def _dispatch(self, operation: str, payload: dict[str, Any]) -> Any:
        if operation == "health":
            # A client offering encodings learns which ones this server can receive.
            if "encodings" in payload:
                return {"state": self.state.value, "encodings": list(ENCODINGS)}
            return {"state": self.state.value}
        if operation == "load":
            self._require_state(operation, PolicyState.CREATED)
            self.state = PolicyState.LOADING
            context = SessionContext.from_dict(payload["context"])
            capabilities = self.policy.load(context)
            if not isinstance(capabilities, PolicyCapabilities):
                raise TypeError("Policy.load() must return PolicyCapabilities")
            validate_policy_capabilities(context.benchmark, capabilities)
            if capabilities.step_observation_updates and type(self.policy).observe is Policy.observe:
                raise TypeError("Policy must implement observe() when it declares step observation updates")
            self.context = context
            self.capabilities = capabilities
            self.state = PolicyState.READY
            return {"state": self.state.value, "capabilities": capabilities.to_dict()}
        if operation == "reset":
            self._require_state(operation, PolicyState.READY)
            self.policy.reset(EpisodeContext.from_dict(payload["context"]))
            return {"state": self.state.value}
        if operation == "act":
            self._require_state(operation, PolicyState.READY)
            context, capabilities = self._loaded_contract()
            observation = Observation.from_dict(payload["observation"])
            validate_observation(observation, context.benchmark, capabilities)
            with self._measure_batch(1), timing.span("server.policy_act"):
                action = self.policy.act(observation)
            if not isinstance(action, ActionChunk):
                raise TypeError("Policy.act() must return ActionChunk")
            validate_action_chunk(action, context.benchmark)
            self._validate_action_chunk_size(action, capabilities)
            return {"action_chunk": action.to_dict()}
        if operation == "observe":
            self._require_state(operation, PolicyState.READY)
            context, capabilities = self._loaded_contract()
            if context.benchmark.name not in capabilities.step_observation_updates:
                raise ProtocolError("policy did not declare step observation updates for this benchmark")
            observation = Observation.from_dict(payload["observation"])
            validate_observation(observation, context.benchmark, capabilities)
            self.policy.observe(observation)
            return {"state": self.state.value}
        if operation == "act_batch":
            self._require_state(operation, PolicyState.READY)
            context, capabilities = self._loaded_contract()
            observations = tuple(Observation.from_dict(item) for item in payload["observations"])
            if not observations:
                raise ValueError("observation batch cannot be empty")
            if len(observations) > capabilities.max_batch_size:
                raise ValueError(
                    f"observation batch size {len(observations)} exceeds policy maximum "
                    f"{capabilities.max_batch_size}"
                )
            for observation in observations:
                validate_observation(observation, context.benchmark, capabilities)
            with (
                self._measure_batch(len(observations)),
                timing.capture("batch", size=len(observations)),
                timing.span("server.policy_batch"),
            ):
                actions = self.policy.act_batch(observations)
            if len(actions) != len(observations):
                raise ValueError("Policy.act_batch() must return one ActionChunk per observation")
            for action in actions:
                if not isinstance(action, ActionChunk):
                    raise TypeError("Policy.act_batch() must return ActionChunk values")
                validate_action_chunk(action, context.benchmark)
                self._validate_action_chunk_size(action, capabilities)
            return {"action_chunks": [action.to_dict() for action in actions]}
        if operation == "end_episode":
            self._require_state(operation, PolicyState.READY)
            episode_id = str(payload.get("episode_id") or "")
            if not episode_id:
                raise ValueError("episode_id cannot be empty")
            self.policy.end_episode(episode_id)
            return {"state": self.state.value}
        if operation == "unload":
            self._require_state(operation, PolicyState.CREATED, PolicyState.READY, PolicyState.FAILED)
            self._close_policy()
            return {"state": self.state.value}
        raise ProtocolError(f"unknown operation {operation!r}")

    def _close_policy(self) -> None:
        with self._batch_lock:
            if not self._batch_stopping:
                self._batch_stopping = True
                if self._batch_thread is not None:
                    self._batch_queue.put(None)
        if self.state in {PolicyState.UNLOADING, PolicyState.CLOSED}:
            return
        should_unload = self.state is not PolicyState.CREATED
        self.state = PolicyState.UNLOADING
        try:
            if should_unload:
                self.policy.unload()
        finally:
            self.state = PolicyState.CLOSED

    def _require_state(self, operation: str, *allowed: PolicyState) -> None:
        if self.state not in allowed:
            expected = ", ".join(state.value for state in allowed)
            raise ProtocolError(
                f"operation {operation!r} is invalid in state {self.state.value!r}; expected {expected}"
            )

    def _loaded_contract(self) -> tuple[SessionContext, PolicyCapabilities]:
        if self.context is None or self.capabilities is None:
            raise RuntimeError("policy contract is unavailable before load")
        return self.context, self.capabilities

    @staticmethod
    def _validate_action_chunk_size(
        action: ActionChunk,
        capabilities: PolicyCapabilities,
    ) -> None:
        chunk_size = 1 if action.actions.ndim == 1 else action.actions.shape[0]
        if chunk_size not in capabilities.supported_action_chunk_sizes:
            raise ValueError(
                f"policy returned action chunk size {chunk_size}; supported sizes are "
                f"{capabilities.supported_action_chunk_sizes}"
            )

    def _success_response(self, request_id: str, result: Any) -> dict[str, Any]:
        return {
            "version": PROTOCOL_VERSION,
            "request_id": request_id,
            "ok": True,
            "result": result,
        }

    def _error_response(
        self,
        exc: Exception,
        *,
        request_id: str,
        operation: str = "protocol",
    ) -> dict[str, Any]:
        return {
            "version": PROTOCOL_VERSION,
            "request_id": request_id,
            "ok": False,
            "error": {
                "operation": operation,
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                "state": self.state.value,
            },
        }


def run(policy: Policy) -> None:
    """Run a policy at the socket path injected by the RoboScope runtime."""

    socket_path = os.environ.get("ROBOSCOPE_INFERENCE_SOCKET", DEFAULT_SOCKET_PATH)
    PolicyServer(policy, socket_path).serve()
