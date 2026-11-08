"""Diagnostic hooks for pinned QwenOFT; leave installed StarVLA sources untouched."""

from contextlib import contextmanager
from functools import wraps

from roboscope_kit import timing


class ModelTiming:
    def __init__(self, framework):
        self.originals = []
        self.pending = []
        interface = getattr(framework, "qwen_vl_interface", None)
        if interface is None:
            return
        self._wrap(interface, "build_qwenvl_inputs", "policy.input_build", cuda=False)
        self._wrap(interface, "forward", "policy.vlm", cuda=True)
        self._wrap(framework, "_gather_action_token_embeddings", "policy.gather", cuda=True)
        self._wrap(framework.action_model, "predict_action", "policy.action_head", cuda=True)

    def _wrap(self, owner, attribute, name, *, cuda):
        original = getattr(owner, attribute)
        self.originals.append((owner, attribute, attribute in owner.__dict__, original))

        @wraps(original)
        def measured(*args, **kwargs):
            import torch

            start = end = None
            if cuda:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
            try:
                with timing.span(name + ".wall"):
                    return original(*args, **kwargs)
            finally:
                if end is not None:
                    end.record()
                    self.pending.append((name, start, end))

        setattr(owner, attribute, measured)

    @contextmanager
    def request(self):
        self.pending.clear()
        try:
            yield
        finally:
            # QwenOFT returns a CPU ndarray, already waiting for its output copy.
            # Events are read only afterwards, not synchronized at every stage.
            # Their intervals include GPU scheduling/host gaps, not just kernel work.
            with timing.span("policy.cuda_event_read"):
                for name, start, end in self.pending:
                    end.synchronize()
                    timing.add(name + ".cuda_interval", start.elapsed_time(end) / 1000)
            self.pending.clear()

    def close(self):
        for owner, attribute, had_override, original in reversed(self.originals):
            if had_override:
                setattr(owner, attribute, original)
            else:
                delattr(owner, attribute)
        self.originals.clear()
