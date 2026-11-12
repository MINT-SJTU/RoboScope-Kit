# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Execute one ordered group of RoboCasa365 simulator slots."""

from __future__ import annotations

import ctypes
import multiprocessing
import os
import pickle
import signal
import socket
import struct
import time
import traceback
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .protocol import OBSERVATION_STEPS, Task

SLOT_TIMEOUT_S = 300
STOP_TIMEOUT_S = 5
MAX_MESSAGE_BYTES = 256 * 1024 * 1024


class _SlotChannel:
    """Frame slot messages with deadlines covering headers and complete bodies."""

    def __init__(self, transport: socket.socket) -> None:
        self._transport = transport

    @staticmethod
    def _remaining(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("slot message deadline expired")
        return remaining

    def send(self, value: Any, deadline: float | None = None) -> None:
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        if len(payload) > MAX_MESSAGE_BYTES:
            raise ValueError("RoboCasa365 slot message exceeds the transport limit")
        for data in (struct.pack("!Q", len(payload)), payload):
            self._transport.settimeout(self._remaining(deadline))
            self._transport.sendall(data)

    def _read(self, size: int, deadline: float | None) -> bytes:
        data = bytearray()
        while len(data) < size:
            self._transport.settimeout(self._remaining(deadline))
            part = self._transport.recv(min(size - len(data), 1024 * 1024))
            if not part:
                raise EOFError("slot closed its message channel")
            data.extend(part)
        return bytes(data)

    def recv(self, deadline: float | None = None) -> Any:
        size, = struct.unpack("!Q", self._read(8, deadline))
        if size > MAX_MESSAGE_BYTES:
            raise ValueError("RoboCasa365 slot message exceeds the transport limit")
        value = pickle.loads(self._read(size, deadline))
        self._remaining(deadline)
        return value

    def close(self) -> None:
        self._transport.close()

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        self.close()


@dataclass
class SlotState:
    history: deque
    initial: dict[str, Any]
    steps: int = 0
    success: bool = False
    done: bool = False
    recent_success: deque = field(default_factory=lambda: deque(maxlen=OBSERVATION_STEPS))


class NativeSlot:
    def __init__(self, task: Task, index: int) -> None:
        from .worker import create_environment

        self.task = task
        self.environment = create_environment(task, index)
        try:
            context = self.environment.unwrapped.env.sim._render_context_offscreen
            if getattr(context, "_roboscope_owning_context_cleanup", None) != 2:
                raise RuntimeError("rebuild RoboCasa365 runtime with the owning-context cleanup patches")
        except Exception:
            self.environment.close()
            raise
        self.state: SlotState | None = None

    def reset(self, identifier: str) -> SlotState:
        from .worker import build_observation, reset_fingerprint

        frame, _ = self.environment.reset()
        history = deque((frame, frame), maxlen=OBSERVATION_STEPS)
        initial = reset_fingerprint(self.environment, build_observation(identifier, 0, history))
        self.state = SlotState(history, initial)
        return self.state

    def advance(self, actions: np.ndarray, writer=None) -> tuple[SlotState, float, float]:
        from robocasa.utils.env_utils import convert_action

        from .worker import environment_range

        assert self.state is not None
        state = self.state
        step_s = video_s = 0.0
        for action in actions:
            if state.done:
                break
            started = time.perf_counter()
            with environment_range("environment.step"):
                frame, _, terminated, truncated, info = self.environment.step(convert_action(action))
            step_s += time.perf_counter() - started
            state.history.append(frame)
            state.steps += 1
            success = bool(info["success"])
            state.recent_success.append(success)
            state.success |= success
            state.done = bool(terminated or truncated or state.steps >= self.task.horizon)
            if writer is not None and state.steps % 2 == 1:
                started = time.perf_counter()
                writer.append_data(frame["video.robot0_agentview_right"])
                video_s += time.perf_counter() - started
        return state, step_s, video_s

    def close(self) -> None:
        self.environment.close()


class LocalSlots:
    def __init__(self, task: Task, count: int) -> None:
        self._cleanup = ExitStack()
        self._slots = []
        try:
            for index in range(count):
                slot = NativeSlot(task, index)
                self._cleanup.callback(slot.close)
                self._slots.append(slot)
        except BaseException:
            self.close()
            raise

    def reset(self, identifiers: list[str]) -> list[SlotState]:
        return [slot.reset(identifier) for slot, identifier in zip(self._slots, identifiers, strict=True)]

    def advance(self, actions: list[np.ndarray], writers: list) -> tuple[list[SlotState], float, float]:
        replies = [slot.advance(chunk, writer) for slot, chunk, writer in zip(
            self._slots, actions, writers, strict=True
        )]
        return [state for state, _, _ in replies], sum(step for _, step, _ in replies), sum(
            video for _, _, video in replies
        )

    def close(self) -> None:
        self._cleanup.close()


def _run_slot(connection: _SlotChannel, task: Task, index: int, parent_pid: int) -> None:
    # The SDK Worker can die before it gets a chance to close its simulator children.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG on Linux.
        raise OSError(ctypes.get_errno(), "could not set slot parent-death signal")
    if os.getppid() != parent_pid:
        return
    slot = None
    with connection:
        try:
            slot = NativeSlot(task, index)
            connection.send((True, None))
            while True:
                operation, value = connection.recv()
                if operation == "close":
                    return
                result = slot.reset(value) if operation == "reset" else slot.advance(value)
                connection.send((True, result))
        except (EOFError, BrokenPipeError):
            pass
        except Exception:
            connection.send((False, traceback.format_exc()))
        finally:
            if slot is not None:
                slot.close()


class SpawnedSlots:
    def __init__(self, task: Task, count: int) -> None:
        context = multiprocessing.get_context("spawn")
        self._connections = []
        self._processes = []
        try:
            for index in range(count):
                parent_socket, child_socket = socket.socketpair()
                parent, child = _SlotChannel(parent_socket), _SlotChannel(child_socket)
                self._connections.append(parent)
                with child:
                    process = context.Process(
                        target=_run_slot, args=(child, task, index, os.getpid()), daemon=True,
                    )
                    process.start()
                self._processes.append(process)
            self._receive("create")
        except BaseException as error:
            self._close_after_error(error)
            raise

    def _receive(self, operation: str, deadline: float | None = None) -> list:
        if deadline is None:
            deadline = time.monotonic() + SLOT_TIMEOUT_S
        results = []
        for index, connection in enumerate(self._connections):
            try:
                ok, value = connection.recv(deadline)
            except TimeoutError as error:
                raise TimeoutError(f"RoboCasa365 slot {index} timed out during {operation}") from error
            except (EOFError, OSError) as error:
                raise RuntimeError(f"RoboCasa365 slot {index} exited during {operation}") from error
            if not ok:
                raise RuntimeError(f"RoboCasa365 slot {index} failed during {operation}\n{value}")
            results.append(value)
        return results

    def _call(self, operation: str, values: list) -> list:
        deadline = time.monotonic() + SLOT_TIMEOUT_S
        try:
            for index, (connection, value) in enumerate(zip(self._connections, values, strict=True)):
                try:
                    connection.send((operation, value), deadline)
                except TimeoutError as error:
                    raise TimeoutError(f"RoboCasa365 slot {index} timed out sending {operation}") from error
                except OSError as error:
                    raise RuntimeError(f"RoboCasa365 slot {index} exited during {operation}") from error
            return self._receive(operation, deadline)
        except BaseException as error:
            self._close_after_error(error)
            raise

    def _close_after_error(self, error: BaseException) -> None:
        try:
            self.close()
        except Exception as cleanup_error:
            error.add_note(f"RoboCasa365 slot cleanup also failed: {cleanup_error}")

    def reset(self, identifiers: list[str]) -> list[SlotState]:
        return self._call("reset", identifiers)

    def advance(self, actions: list[np.ndarray], writers: list) -> tuple[list[SlotState], float, float]:
        assert not any(writers)
        started = time.perf_counter()
        replies = self._call("advance", actions)
        return [state for state, _, _ in replies], time.perf_counter() - started, 0.0

    def close(self) -> None:
        deadline = time.monotonic() + STOP_TIMEOUT_S
        for connection in self._connections:
            try:
                connection.send(("close", None), deadline)
            except OSError:
                pass
            finally:
                connection.close()
        for process in self._processes:
            process.join(max(0.0, deadline - time.monotonic()))
        for process in self._processes:
            if process.is_alive():
                process.kill()
        deadline = time.monotonic() + STOP_TIMEOUT_S
        for process in self._processes:
            process.join(max(0.0, deadline - time.monotonic()))
        if any(process.is_alive() for process in self._processes):
            raise RuntimeError("RoboCasa365 simulator child did not exit after SIGKILL")
        for process in self._processes:
            process.close()
        self._connections.clear()
        self._processes.clear()
