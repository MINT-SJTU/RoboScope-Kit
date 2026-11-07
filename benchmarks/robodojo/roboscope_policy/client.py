# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Connect RoboDojo's evaluation lifecycle to a RoboScope Policy replica."""

from __future__ import annotations

import os
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from roboscope_kit import ActionChunk, EpisodeContext, PolicyClient


class WsModelClient:
    """RoboDojo client interface backed by the kit's local Policy protocol.

    Each RoboDojo environment slot runs its own Kit episode. With one environment
    (the default) there is at most one open episode; a multi-environment batch
    opens one episode per environment and ends each one independently.
    """

    def __init__(self, **_kwargs):
        self.socket_path = os.environ["ROBOSCOPE_INFERENCE_SOCKET"]
        self.episode_id = ""
        self.episodes: list[str] = []
        self.queue_size = int(os.environ.get("ROBOSCOPE_ROBODOJO_OBSERVATION_QUEUE_SIZE", "0"))
        if not 0 <= self.queue_size <= 10:
            raise ValueError("RoboDojo observation queue size must be between 0 and 10")
        self.pending = deque()
        self.executor = ThreadPoolExecutor(max_workers=1) if self.queue_size else None

    def call(self, func_name=None, obs=None, **_kwargs):
        if func_name != "reset" or obs is not None:
            raise ValueError(f"unsupported RoboDojo model operation: {func_name!r}")
        self.end_episode()

    def begin_episode(self, task, layout_id):
        self.end_episode()
        episode_id = self.open_episode(task, layout_id)
        self.episode_id = episode_id
        return episode_id

    def open_episode(self, task, layout_id):
        """Start one more concurrent episode, for a multi-environment batch."""
        episode_id = f"robodojo-{uuid.uuid4()}"
        metadata = {"layout_id": layout_id}
        if run_id := os.environ.get("ROBODOJO_RUN_ID"):
            metadata["robodojo_run_id"] = run_id
        with PolicyClient(self.socket_path, timeout_s=900) as client:
            client.reset(EpisodeContext(
                episode_id=episode_id,
                task_id=task,
                metadata=metadata,
            ))
        self.episodes.append(episode_id)
        return episode_id

    def act(self, observation):
        self._flush()
        with PolicyClient(self.socket_path, timeout_s=900) as client:
            answer = client.act(observation)
        return answer.actions if isinstance(answer, ActionChunk) else answer

    def poll(self, episode_id):
        with PolicyClient(self.socket_path, timeout_s=900) as client:
            answer = client.poll(episode_id)
        return answer.actions if isinstance(answer, ActionChunk) else answer

    def episode_outcome(self, episode_id, outcome):
        self._flush()
        with PolicyClient(self.socket_path, timeout_s=900) as client:
            client.episode_outcome(episode_id, outcome)

    def _observe_after(self, previous, observation):
        if previous is not None:
            previous.result()
        with PolicyClient(self.socket_path, timeout_s=900) as client:
            client.observe(observation)

    def observe(self, observation):
        if self.executor is None:
            self._observe_after(None, observation)
            return
        while len(self.pending) >= self.queue_size:
            self.pending.popleft().result()
        previous = self.pending[-1] if self.pending else None
        self.pending.append(self.executor.submit(self._observe_after, previous, observation))

    def _flush(self):
        while self.pending:
            self.pending.popleft().result()

    def end_episode(self, episode_id=None):
        """End one episode, or every open episode when no ID is given."""
        if episode_id is None:
            targets = list(self.episodes)
        elif episode_id in self.episodes:
            targets = [episode_id]
        else:
            return
        if not targets:
            return
        for target in targets:
            self.episodes.remove(target)
        if self.episode_id in targets:
            self.episode_id = ""
        try:
            self._flush()
        except BaseException as primary:
            try:
                self._end(targets)
            except BaseException as cleanup:
                raise primary from cleanup
            raise
        self._end(targets)

    def _end(self, episode_ids):
        failure = None
        for episode_id in episode_ids:
            try:
                with PolicyClient(self.socket_path, timeout_s=900) as client:
                    client.end_episode(episode_id)
            except BaseException as exc:
                failure = failure or exc
        if failure is not None:
            raise failure

    def close(self):
        try:
            self.end_episode()
        finally:
            if self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=True)
