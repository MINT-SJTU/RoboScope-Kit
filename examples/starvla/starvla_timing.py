# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Opt-in diagnostic stage hooks for pinned QwenOFT without modifying StarVLA sources."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any

import torch

from roboscope_kit import profiling


class ModelTiming:
    def __init__(self, framework: Any) -> None:
        self.originals: list[tuple[Any, str, bool, Any]] = []
        self.pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        interface = getattr(framework, "qwen_vl_interface", None)
        if interface is None:
            return
        self._wrap(interface, "build_qwenvl_inputs", "policy.input_build", cuda=False)
        self._wrap(interface, "forward", "policy.vlm", cuda=True)
        self._wrap(framework, "_gather_action_token_embeddings", "policy.gather", cuda=True)
        self._wrap(framework.action_model, "predict_action", "policy.action_head", cuda=True)

    def _wrap(self, owner: Any, attribute: str, name: str, *, cuda: bool) -> None:
        original = getattr(owner, attribute)
        self.originals.append((owner, attribute, attribute in owner.__dict__, original))
        wall_name = f"{name}.wall"

        @wraps(original)
        def measured(*args: Any, **kwargs: Any) -> Any:
            start = end = None
            if cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            try:
                with profiling.span(wall_name):
                    return original(*args, **kwargs)
            finally:
                if start is not None and end is not None:
                    end.record()
                    self.pending.append((name, start, end))

        setattr(owner, attribute, measured)

    def wrap_prediction(self, predict: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(predict)
        def measured(*args: Any, **kwargs: Any) -> Any:
            with self.request():
                return predict(*args, **kwargs)

        return measured

    @contextmanager
    def request(self) -> Iterator[None]:
        self.pending.clear()
        try:
            yield
        finally:
            # QwenOFT returns a CPU ndarray after synchronizing its output copy.
            # Events are read only afterwards without inserting per-stage syncs.
            with profiling.span("policy.cuda_event_read"):
                for name, start, end in self.pending:
                    end.synchronize()
                    profiling.add_span(f"{name}.cuda_interval", start.elapsed_time(end) / 1000.0)
            self.pending.clear()

    def close(self) -> None:
        for owner, attribute, had_override, original in reversed(self.originals):
            if had_override:
                setattr(owner, attribute, original)
            else:
                delattr(owner, attribute)
        self.originals.clear()
