# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Request-local CPU token metadata for the checksum-pinned QwenOFT inference path."""

from __future__ import annotations

from typing import Any

import torch
from starvla_vision_metadata import (
    TRANSFORMERS_QWEN25_VL_SHA256,
    verify_module_source_sha256,
)

STARVLA_QWEN_OFT_SHA256 = "039117281f3598cc08056a9f7dfc57e2f1b156ce396623f06d55b98d4b55913b"
_REQUIRED_PROCESSOR_KEYS = ("input_ids", "attention_mask", "image_grid_thw")


class TokenMetadata:
    """Run upstream action-token selection and RoPE indexing on CPU inputs.

    Install after ``VisionMetadata`` and close before it when both are enabled.
    No metadata is reused across requests, and no activations or actions are cached.
    """

    def __init__(self, framework: Any, *, gather: bool = True, rope: bool = True) -> None:
        from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as upstream

        verify_module_source_sha256(
            upstream,
            TRANSFORMERS_QWEN25_VL_SHA256,
            "CPU token metadata (Transformers 4.57.0)",
        )
        verify_module_source_sha256(
            type(framework),
            STARVLA_QWEN_OFT_SHA256,
            "CPU token metadata (StarVLA QwenOFT)",
        )
        self.framework = framework
        self.interface = framework.qwen_vl_interface
        self.model = self.interface.model.model
        self.processor = self.interface.processor
        self.build = self.interface.build_qwenvl_inputs
        self.gather = framework._gather_action_token_embeddings
        self.rope = self.model.get_rope_index
        self.use_gather = gather
        self.use_rope = rope
        self.cpu: dict[str, torch.Tensor] | None = None
        self.device: dict[str, torch.Tensor] | None = None
        self.indices: torch.Tensor | None = None
        self.positions: tuple[torch.Tensor, ...] | None = None
        self.interface.processor = self
        self.interface.build_qwenvl_inputs = self.build_inputs
        if gather:
            framework._gather_action_token_embeddings = self.gather_embeddings
        if rope:
            self.model.get_rope_index = self.rope_index

    def __getattr__(self, name: str) -> Any:
        return getattr(self.processor, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result = self.processor(*args, **kwargs)
        if result.get("video_grid_thw") is not None or any(
            result.get(name) is None or result[name].device.type != "cpu"
            for name in _REQUIRED_PROCESSOR_KEYS
        ):
            raise RuntimeError("CPU token metadata requires CPU image-only processor inputs")
        self.cpu = {name: result[name].clone() for name in _REQUIRED_PROCESSOR_KEYS}
        ids = self.cpu["input_ids"]
        if self.use_gather:
            # Applying the original gather to positions preserves its validation,
            # multi-ID handling, and last-chunk ordering without duplicating that logic.
            positions = torch.arange(ids.shape[1]).view(1, -1, 1).expand(ids.shape[0], -1, 1)
            self.indices = self.gather(
                positions,
                ids,
                action_token_id=self.framework.action_token_id,
            )
        if self.use_rope:
            self.positions = self.rope(**self.cpu)
        return result

    def build_inputs(self, *args: Any, **kwargs: Any) -> Any:
        self.cpu = None
        self.device = None
        self.indices = None
        self.positions = None
        result = self.build(*args, **kwargs)
        if self.cpu is None:
            raise RuntimeError("Processor did not provide CPU token metadata")
        self.device = {name: result[name] for name in self.cpu}
        device = result["input_ids"].device
        if self.use_gather and self.indices is not None:
            self.indices = self.indices.to(device)
        if self.use_rope and self.positions is not None:
            self.positions = tuple(value.to(device) for value in self.positions)
        return result

    def gather_embeddings(
        self,
        last_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        action_token_id: Any = None,
    ) -> torch.Tensor:
        if (
            self.device is None
            or self.indices is None
            or input_ids is not self.device["input_ids"]
            or action_token_id != self.framework.action_token_id
            or last_hidden.shape[:2] != input_ids.shape
        ):
            raise RuntimeError("Action indices must belong to the current processor request")
        return last_hidden.gather(1, self.indices.expand(-1, -1, last_hidden.shape[-1]))

    def rope_index(
        self,
        input_ids: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        second_per_grid_ts: Any = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if (
            self.device is None
            or self.positions is None
            or input_ids is not self.device["input_ids"]
            or image_grid_thw is not self.device["image_grid_thw"]
            or attention_mask is not self.device["attention_mask"]
            or video_grid_thw is not None
            or second_per_grid_ts is not None
        ):
            raise RuntimeError("RoPE metadata must belong to the current image-only request")
        return self.positions

    def close(self) -> None:
        self.interface.processor = self.processor
        self.interface.build_qwenvl_inputs = self.build
        if self.use_gather:
            self.framework._gather_action_token_embeddings = self.gather
        if self.use_rope:
            self.model.get_rope_index = self.rope
        self.cpu = None
        self.device = None
        self.indices = None
        self.positions = None
