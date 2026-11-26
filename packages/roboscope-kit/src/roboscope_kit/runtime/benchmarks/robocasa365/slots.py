# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Execute the five RoboCasa365 simulator slots of a Worker, serially or on threads."""

from __future__ import annotations

import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .protocol import OBSERVATION_STEPS, Task


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
    def __init__(self, task: Task, count: int, slot_factory=NativeSlot) -> None:
        self._cleanup = ExitStack()
        self._slots = []
        try:
            for index in range(count):
                slot = slot_factory(task, index)
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


class ThreadedSlots:
    """Keep each slot on its own thread and step the slots concurrently.

    MuJoCo releases the GIL while stepping, so the slots' physics overlaps in
    one process at the memory of serial slots. A slot is
    created, reset, stepped and closed only on its thread, where its EGL context
    is current. Creation and reset run one slot at a time in slot order, as in
    the serial path; only action chunks run concurrently.
    """

    def __init__(self, task: Task, count: int, slot_factory=NativeSlot) -> None:
        self._threads = [
            ThreadPoolExecutor(1, thread_name_prefix=f"robocasa365-slot-{index}") for index in range(count)
        ]
        self._slots = []
        try:
            for index, thread in enumerate(self._threads):
                self._slots.append(thread.submit(slot_factory, task, index).result())
        except BaseException:
            self.close()
            raise

    def reset(self, identifiers: list[str]) -> list[SlotState]:
        return [
            thread.submit(slot.reset, identifier).result()
            for thread, slot, identifier in zip(self._threads, self._slots, identifiers, strict=True)
        ]

    def advance(self, actions: list[np.ndarray], writers: list) -> tuple[list[SlotState], float, float]:
        assert not any(writers)
        started = time.perf_counter()
        futures = [
            thread.submit(slot.advance, chunk)
            for thread, slot, chunk in zip(self._threads, self._slots, actions, strict=True)
        ]
        replies = [future.result() for future in futures]
        return [state for state, _, _ in replies], time.perf_counter() - started, 0.0

    def close(self) -> None:
        errors = []
        for thread, slot in zip(self._threads, self._slots, strict=False):  # Fewer slots if creation failed.
            try:
                thread.submit(slot.close).result()
            except Exception as error:
                errors.append(error)
        for thread in self._threads:
            thread.shutdown()
        self._slots.clear()
        self._threads.clear()
        if errors:
            raise errors[0]
