# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Opt-in CPU vision metadata and grouped window attention for pinned Qwen2.5-VL SDPA.

The vision forward below is adapted from Transformers 4.57.0 (Apache-2.0,
Copyright HuggingFace Inc.). Only metadata placement, static grid-plan reuse, and
equal-length window batching change; all learned operations, attention partitions,
and their ordering follow the pinned source.
"""

from __future__ import annotations

import hashlib
import inspect
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

TRANSFORMERS_QWEN25_VL_SHA256 = "f5051523300145f66de080765c06bd8080350c15692c096d2c38e73fe9d03099"
_MAX_GRID_PLAN_CACHE = 16


@dataclass(frozen=True)
class _GridPlan:
    window_index: torch.Tensor
    reverse_indices: torch.Tensor
    cu_window_seqlens: torch.Tensor
    cu_seqlens: torch.Tensor
    position_embeddings: tuple[torch.Tensor, torch.Tensor]
    window_shape: tuple[int, int] | None
    full_shape: tuple[int, int] | None


@lru_cache(maxsize=4)
def verify_module_source_sha256(module_or_type: Any, expected_sha256: str, label: str) -> None:
    """Verify once per process that an imported module or class matches its pinned source."""
    source = inspect.getsourcefile(module_or_type)
    if source is None:
        raise RuntimeError(f"{label} source file is unavailable for verification")
    actual = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(f"{label} requires the verified pinned source ({expected_sha256[:12]})")


def _uniform_window_shape(cu_seqlens: torch.Tensor) -> tuple[int, int] | None:
    """Return ``(num_windows, window_len)`` when 2+ windows share one positive length."""
    window_count = cu_seqlens.numel() - 1
    if window_count < 2:
        return None
    lengths = cu_seqlens.diff()
    window_len = int(lengths[0])
    if window_len <= 0 or not bool((lengths == window_len).all()):
        return None
    return window_count, window_len


def _build_grid_plan(
    visual: Any,
    grid_thw: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> _GridPlan:
    rotary_pos_emb = visual.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = visual.get_window_index(grid_thw)
    seq_dtype = grid_thw.dtype if torch.jit.is_tracing() else torch.int32
    target_device = grid_thw.device if torch.jit.is_tracing() else device
    cu_window_tensor = torch.tensor(cu_window_seqlens, device=target_device, dtype=seq_dtype)
    cu_window_tensor = torch.unique_consecutive(cu_window_tensor)

    merged_groups = seq_len // visual.spatial_merge_unit
    rotary_pos_emb = rotary_pos_emb.reshape(merged_groups, visual.spatial_merge_unit, -1)[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (
        emb.cos().to(device=target_device, non_blocking=True),
        emb.sin().to(device=target_device, non_blocking=True),
    )

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=seq_dtype,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).to(device=target_device, non_blocking=True)
    return _GridPlan(
        window_index=window_index.to(device=target_device, dtype=torch.long, non_blocking=True),
        reverse_indices=torch.argsort(window_index).to(
            device=target_device, dtype=torch.long, non_blocking=True
        ),
        cu_window_seqlens=cu_window_tensor,
        cu_seqlens=cu_seqlens,
        position_embeddings=position_embeddings,
        window_shape=_uniform_window_shape(cu_window_tensor),
        full_shape=_uniform_window_shape(cu_seqlens),
    )


class VisionMetadata:
    """Retain each request's CPU image grid before BatchFeature uploads it to CUDA."""

    def __init__(self, interface: Any, *, group_windows: bool = False) -> None:
        from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as upstream

        verify_module_source_sha256(
            upstream,
            TRANSFORMERS_QWEN25_VL_SHA256,
            "CPU vision metadata (Transformers 4.57.0)",
        )
        self.visual = interface.model.model.visual
        if self.visual.config._attn_implementation != "sdpa":
            raise RuntimeError("CPU vision metadata requires SDPA")
        self.interface = interface
        self.processor = interface.processor
        self.build = interface.build_qwenvl_inputs
        self.forward = self.visual.forward
        self.cpu_grid: torch.Tensor | None = None
        self.device_grid: torch.Tensor | None = None
        self._window_shapes: dict[int, tuple[int, int] | None] = {}
        self._grid_plans: OrderedDict[tuple[Any, ...], _GridPlan] = OrderedDict()
        self.attention_forwards: list[tuple[Any, Callable[..., Any]]] = []
        if group_windows:
            sdpa_fn = upstream.ALL_ATTENTION_FUNCTIONS["sdpa"]
            rotary_fn = upstream.apply_rotary_pos_emb_vision
            for block in self.visual.blocks:
                attention = block.attn
                original_forward = attention.forward
                self.attention_forwards.append((attention, original_forward))
                attention.forward = partial(
                    self._grouped_window_attention,
                    attention,
                    original_forward,
                    sdpa_fn,
                    rotary_fn,
                )
        interface.processor = self
        interface.build_qwenvl_inputs = self.build_inputs
        self.visual.forward = self.run_vision

    def __getattr__(self, name: str) -> Any:
        return getattr(self.processor, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result = self.processor(*args, **kwargs)
        grid = result.get("image_grid_thw")
        if grid is None or grid.device.type != "cpu" or result.get("video_grid_thw") is not None:
            raise RuntimeError("CPU vision metadata requires CPU image grids and no video")
        self.cpu_grid = grid.clone()
        return result

    def build_inputs(self, *args: Any, **kwargs: Any) -> Any:
        self.cpu_grid = None
        self.device_grid = None
        result = self.build(*args, **kwargs)
        if self.cpu_grid is None:
            raise RuntimeError("Processor did not provide CPU vision metadata")
        self.device_grid = result["image_grid_thw"]
        return result

    def _get_grid_plan(
        self,
        grid_thw: torch.Tensor,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> _GridPlan:
        if self.visual.training or torch.jit.is_tracing():
            return _build_grid_plan(self.visual, grid_thw, seq_len, device)
        key = (tuple(grid_thw.reshape(-1).tolist()), seq_len, device, dtype)
        plan = self._grid_plans.get(key)
        if plan is not None:
            self._grid_plans.move_to_end(key)
            return plan
        plan = _build_grid_plan(self.visual, grid_thw, seq_len, device)
        self._grid_plans[key] = plan
        if len(self._grid_plans) > _MAX_GRID_PLAN_CACHE:
            self._grid_plans.popitem(last=False)
        return plan

    def run_vision(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if grid_thw is not self.device_grid or self.cpu_grid is None:
            raise RuntimeError("Vision metadata must belong to the current request")
        hidden_states = self.visual.patch_embed(hidden_states)
        seq_len, _ = hidden_states.size()
        plan = self._get_grid_plan(self.cpu_grid, seq_len, hidden_states.device, hidden_states.dtype)
        merged_groups = seq_len // self.visual.spatial_merge_unit
        hidden_states = hidden_states.reshape(merged_groups, self.visual.spatial_merge_unit, -1)[
            plan.window_index, :, :
        ].reshape(seq_len, -1)

        if self.attention_forwards:
            self._window_shapes[id(plan.cu_window_seqlens)] = plan.window_shape
            self._window_shapes[id(plan.cu_seqlens)] = plan.full_shape
        try:
            for layer_num, blk in enumerate(self.visual.blocks):
                if layer_num in self.visual.fullatt_block_indexes:
                    cu_seqlens_now = plan.cu_seqlens
                else:
                    cu_seqlens_now = plan.cu_window_seqlens
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens_now,
                    position_embeddings=plan.position_embeddings,
                    **kwargs,
                )
        finally:
            self._window_shapes.clear()

        hidden_states = self.visual.merger(hidden_states)
        return hidden_states[plan.reverse_indices, :]

    def _grouped_window_attention(
        self,
        attention: Any,
        original: Callable[..., Any],
        sdpa_fn: Callable[..., Any],
        rotary_fn: Callable[..., Any],
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Batch independent vision windows of identical length into a single SDPA call."""
        window_shape = self._window_shapes.get(id(cu_seqlens))
        if window_shape is None and id(cu_seqlens) not in self._window_shapes:
            window_shape = _uniform_window_shape(cu_seqlens)
        if attention.training or window_shape is None or position_embeddings is None:
            return original(
                hidden_states,
                cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        window_count, window_len = window_shape
        tokens = hidden_states.shape[0]
        query, key, value = (
            attention.qkv(hidden_states)
            .reshape(tokens, 3, attention.num_heads, -1)
            .unbind(1)
        )
        query, key = rotary_fn(query, key, *position_embeddings)
        # [window, head, token, dim] preserves upstream sequence and window ordering.
        query, key, value = (
            tensor.reshape(window_count, window_len, attention.num_heads, -1).transpose(1, 2)
            for tensor in (query, key, value)
        )
        output, _ = sdpa_fn(
            attention,
            query,
            key,
            value,
            attention_mask=None,
            scaling=attention.scaling,
            dropout=0.0,
            is_causal=False,
            **kwargs,
        )
        return attention.proj(output.reshape(tokens, -1))

    def close(self) -> None:
        self.interface.processor = self.processor
        self.interface.build_qwenvl_inputs = self.build
        self.visual.forward = self.forward
        for attention, forward in self.attention_forwards:
            attention.forward = forward
        self.attention_forwards.clear()
        self._window_shapes.clear()
        self._grid_plans.clear()
        self.cpu_grid = None
        self.device_grid = None
