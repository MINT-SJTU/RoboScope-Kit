# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Zero-sync GPU patchify, prompt token cache, and shared-pool CUDA Graphs for LIBERO-Pro QwenOFT."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from starvla_token_metadata import STARVLA_QWEN_OFT_SHA256
from starvla_vision_metadata import (
    TRANSFORMERS_QWEN25_VL_SHA256,
    VisionMetadata,
    verify_module_source_sha256,
)

# 224 = 7 * 32 keeps B=4 at M = 896 = 7 * 128 (exact Ada BF16 GEMM & FlashAttention tile multiple),
# covering 159/160 LIBERO-Pro tasks; the single 225-token task falls back cleanly to _predict_eager.
STATIC_SEQ_LEN = 224
IMAGE_SIZE = (224, 224)
IMAGES_PER_SAMPLE = 2
TOKENS_PER_IMAGE = 64
PATCHES_PER_IMAGE = 256
IMAGE_TOKENS_PER_SAMPLE = IMAGES_PER_SAMPLE * TOKENS_PER_IMAGE  # 128
PATCH_DIM = 3 * 2 * 14 * 14  # 1176
MAX_PROMPT_CACHE = 256


@dataclass(frozen=True)
class _CachedPrompt:
    raw_ids: torch.Tensor
    # Precomputed right-padded static-shape (STATIC_SEQ_LEN) slices when raw_ids.numel() <= STATIC_SEQ_LEN.
    # Because Qwen2.5-VL decoder attention is strictly causal (`is_causal=True` with `attention_mask=None`),
    # real tokens at `0..seq_len-1` never attend to right-padded positions `seq_len..STATIC_SEQ_LEN-1`.
    static_ids: torch.Tensor | None
    static_pos_ids: torch.Tensor | None
    static_gather_idx: torch.Tensor | None
    static_image_pos: torch.Tensor | None


@dataclass(frozen=True)
class _GraphSlot:
    graph: torch.cuda.CUDAGraph
    static_images: torch.Tensor
    static_ids: torch.Tensor
    static_pos_ids: torch.Tensor
    static_gather_idx: torch.Tensor
    static_image_pos: torch.Tensor
    static_actions: torch.Tensor
    host_images: torch.Tensor
    host_ids: torch.Tensor
    host_pos_ids: torch.Tensor
    host_gather_idx: torch.Tensor
    host_image_pos: torch.Tensor
    host_actions: torch.Tensor


class QwenOFTFastRunner:
    """Accelerated `--eval-mode fast` predictor for checksum-pinned LIBERO-Pro QwenOFT.

    Eliminates per-step CPU PIL/BICUBIC/float32 image normalization, repeated Jinja2/tokenizer
    prompt formatting, `DynamicCache` KV allocations, `output_hidden_states=True` retention,
    and unused `lm_head` vocabulary GEMMs, while replaying a shared-pool end-to-end CUDA Graph
    across batch sizes `1..max_batch_size` whenever sequence length fits `STATIC_SEQ_LEN` (224).
    """

    def __init__(
        self,
        framework: Any,
        vision_metadata: VisionMetadata,
        *,
        max_batch_size: int,
        enable_cuda_graph: bool = True,
    ) -> None:
        from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as upstream

        verify_module_source_sha256(
            upstream,
            TRANSFORMERS_QWEN25_VL_SHA256,
            "QwenOFTFastRunner (Transformers 4.57.0)",
        )
        verify_module_source_sha256(
            type(framework),
            STARVLA_QWEN_OFT_SHA256,
            "QwenOFTFastRunner (StarVLA QwenOFT)",
        )
        self.framework = framework
        self.interface = framework.qwen_vl_interface
        self.vision_metadata = vision_metadata
        self.max_batch_size = max_batch_size

        iproc = self.interface.processor.image_processor
        if (
            getattr(iproc, "patch_size", None) != 14
            or getattr(iproc, "temporal_patch_size", None) != 2
            or getattr(iproc, "merge_size", None) != 2
            or not getattr(iproc, "do_rescale", True)
            or not getattr(iproc, "do_normalize", True)
        ):
            raise RuntimeError("Unexpected Qwen2.5-VL image processor configuration for fast patchify")

        vla_cfg = getattr(getattr(framework.config, "datasets", None), "vla_data", None)
        obs_img_size = getattr(vla_cfg, "obs_image_size", None)
        if obs_img_size not in (None, 224, (224, 224), [224, 224]):
            raise RuntimeError(f"Unsupported QwenOFT obs_image_size for 224x224 fast path: {obs_img_size!r}")

        self.device = next(self.interface.model.parameters()).device
        self.mean_gpu = torch.tensor(
            iproc.image_mean, device=self.device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self.std_gpu = torch.tensor(
            iproc.image_std, device=self.device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self.rescale_factor = float(iproc.rescale_factor)

        self.pad_token_id = int(self.interface.processor.tokenizer.pad_token_id)
        self.image_token_id = int(self.interface.model.config.image_token_id)
        self.action_token_id = int(self.framework.action_token_id)
        if self.pad_token_id in (self.image_token_id, self.action_token_id):
            raise RuntimeError("Right-padded causal fast path requires distinct pad/image/action token IDs")
        self.embed_tokens = self.interface.model.model.get_input_embeddings()
        self.language_model = (
            self.interface.model.model.language_model
            if hasattr(self.interface.model.model, "language_model")
            else self.interface.model.model
        )
        self._get_rope_index = self.interface.model.model.get_rope_index
        self._gather_fn = framework._gather_action_token_embeddings
        self._raw_build_inputs = self.vision_metadata.build

        action_tokens = framework.action_token * framework.chunk_len
        self.prompt_suffix = (
            f" Please predict the next {framework.chunk_len} robot actions: "
            f"<action>{action_tokens}<action>."
        )
        self._dummy_pil_pair = [
            Image.fromarray(np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8)),
            Image.fromarray(np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8)),
        ]
        self._static_pos_row = torch.arange(STATIC_SEQ_LEN, dtype=torch.long).view(1, STATIC_SEQ_LEN, 1)
        self._grid_cpu_by_b: dict[int, torch.Tensor] = {
            b: torch.tensor([[1, 16, 16]] * (IMAGES_PER_SAMPLE * b), dtype=torch.long)
            for b in range(1, max_batch_size + 1)
        }
        self._prompt_cache: OrderedDict[str, _CachedPrompt] = OrderedDict()
        self._graphs: dict[int, _GraphSlot] = {}

        self._verify_patchify_equivalence(iproc)
        self._verify_causal_right_pad_equivalence()
        if enable_cuda_graph:
            self._capture_graphs()

    def _verify_patchify_equivalence(self, iproc: Any) -> None:
        rng = np.random.default_rng(42)
        test_np = rng.integers(0, 256, (2, *IMAGE_SIZE, 3), dtype=np.uint8)
        pil_list = [Image.fromarray(test_np[0]), Image.fromarray(test_np[1])]
        ref = iproc(images=pil_list, return_tensors="pt")["pixel_values"].to(
            device=self.device, dtype=torch.bfloat16
        )
        with torch.inference_mode():
            actual = self._gpu_patchify_224(
                torch.from_numpy(test_np).to(device=self.device)
            ).to(dtype=torch.bfloat16)
        if not torch.equal(ref, actual) and (ref.float() - actual.float()).abs().max().item() > 1e-2:
            raise RuntimeError("GPU 224x224 patchify diverged from Qwen2VLImageProcessor")

    def _verify_causal_right_pad_equivalence(self) -> None:
        prompt = self._get_prompt("open the middle layer of the drawer")
        if (
            prompt.static_ids is None
            or prompt.static_pos_ids is None
            or prompt.static_gather_idx is None
            or prompt.static_image_pos is None
        ):
            return
        raw_ids = prompt.raw_ids.unsqueeze(0)
        raw_mask = torch.ones_like(raw_ids)
        raw_pos, _ = self._get_rope_index(
            input_ids=raw_ids,
            image_grid_thw=self._grid_cpu_by_b[1],
            attention_mask=raw_mask,
        )
        raw_gather = self._gather_fn(
            torch.arange(raw_ids.shape[1], dtype=torch.long).view(1, -1, 1),
            raw_ids,
            action_token_id=self.action_token_id,
        )
        rng = np.random.default_rng(7)
        dummy_imgs = torch.from_numpy(
            rng.integers(0, 256, (IMAGES_PER_SAMPLE, *IMAGE_SIZE, 3), dtype=np.uint8)
        ).to(self.device)
        grid_cpu = self._grid_cpu_by_b[1]
        self.vision_metadata.cpu_grid = grid_cpu
        self.vision_metadata.device_grid = grid_cpu
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            ie = self.vision_metadata.run_vision(
                self._gpu_patchify_224(dummy_imgs), self.vision_metadata.device_grid
            )
            img_pos = prompt.static_image_pos.unsqueeze(0).to(self.device)
            ref_emb = self.embed_tokens(raw_ids.to(self.device)).clone()
            ref_emb.scatter_(
                1,
                img_pos.expand(-1, -1, ref_emb.shape[-1]),
                ie.reshape(1, IMAGE_TOKENS_PER_SAMPLE, -1).to(ref_emb.dtype),
            )
            ref_out = self.language_model(
                input_ids=None,
                position_ids=raw_pos.to(self.device),
                attention_mask=None,
                past_key_values=None,
                inputs_embeds=ref_emb,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state
            ref_q = ref_out.gather(
                1, raw_gather.to(self.device).expand(-1, -1, ref_out.shape[-1])
            )

            pad_emb = self.embed_tokens(prompt.static_ids.unsqueeze(0).to(self.device)).clone()
            pad_emb.scatter_(
                1,
                img_pos.expand(-1, -1, pad_emb.shape[-1]),
                ie.reshape(1, IMAGE_TOKENS_PER_SAMPLE, -1).to(pad_emb.dtype),
            )
            pad_out = self.language_model(
                input_ids=None,
                position_ids=prompt.static_pos_ids.unsqueeze(1).to(self.device),
                attention_mask=None,
                past_key_values=None,
                inputs_embeds=pad_emb,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state
            pad_q = pad_out.gather(
                1,
                prompt.static_gather_idx.unsqueeze(0).to(self.device).expand(-1, -1, pad_out.shape[-1]),
            )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float32):
            ref_act = self.framework.action_model.predict_action(ref_q)
            pad_act = self.framework.action_model.predict_action(pad_q)
        max_diff = (ref_act.float() - pad_act.float()).abs().max().item()
        if max_diff > 5e-2:
            raise RuntimeError(
                f"Right-padded causal SDPA diverged from unpadded reference (max_diff={max_diff:.4g})"
            )
        eager_np = dummy_imgs.detach().cpu().numpy()
        eager_out = self._predict_eager(
            [{"image": [eager_np[0], eager_np[1]], "lang": "open the middle layer of the drawer"}],
            [prompt],
        )["normalized_actions"]
        if eager_out.shape != (1, self.framework.chunk_len, ref_act.shape[-1]):
            raise RuntimeError(f"Unexpected eager fallback output shape: {eager_out.shape}")

    def _gpu_patchify_224(self, uint8_nhwc: torch.Tensor) -> torch.Tensor:
        n_img = uint8_nhwc.shape[0]
        x = uint8_nhwc.permute(0, 3, 1, 2).to(torch.float32)
        x = (x * self.rescale_factor - self.mean_gpu) / self.std_gpu
        x = x.unsqueeze(1).expand(-1, 2, -1, -1, -1)
        x = x.reshape(n_img, 2, 3, 8, 2, 14, 8, 2, 14)
        x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).contiguous()
        return x.reshape(n_img * PATCHES_PER_IMAGE, PATCH_DIM)

    def _get_prompt(self, instruction: str) -> _CachedPrompt:
        cached = self._prompt_cache.get(instruction)
        if cached is not None:
            self._prompt_cache.move_to_end(instruction)
            return cached

        full_text = instruction + self.prompt_suffix
        built = self._raw_build_inputs(
            images=[self._dummy_pil_pair],
            instructions=[full_text],
        )
        raw_ids = built["input_ids"][0].detach().cpu().to(dtype=torch.long)
        seq_len = int(raw_ids.numel())
        n_img_tokens = int((raw_ids == self.image_token_id).sum().item())

        if seq_len <= STATIC_SEQ_LEN and n_img_tokens == IMAGE_TOKENS_PER_SAMPLE:
            static_ids = torch.full((1, STATIC_SEQ_LEN), self.pad_token_id, dtype=torch.long)
            static_mask = torch.zeros((1, STATIC_SEQ_LEN), dtype=torch.long)
            static_ids[0, :seq_len] = raw_ids
            static_mask[0, :seq_len] = 1
            pos_ids, _ = self._get_rope_index(
                input_ids=static_ids,
                image_grid_thw=self._grid_cpu_by_b[1],
                attention_mask=static_mask,
            )
            gather_idx = self._gather_fn(
                self._static_pos_row,
                static_ids,
                action_token_id=self.action_token_id,
            )
            image_pos = (
                (static_ids == self.image_token_id)
                .nonzero(as_tuple=False)[:, 1]
                .reshape(1, IMAGE_TOKENS_PER_SAMPLE, 1)
            )
            entry = _CachedPrompt(
                raw_ids=raw_ids,
                static_ids=static_ids[0].contiguous(),
                static_pos_ids=pos_ids[:, 0, :].contiguous(),
                static_gather_idx=gather_idx[0].contiguous(),
                static_image_pos=image_pos[0].contiguous(),
            )
        else:
            entry = _CachedPrompt(
                raw_ids=raw_ids,
                static_ids=None,
                static_pos_ids=None,
                static_gather_idx=None,
                static_image_pos=None,
            )

        self._prompt_cache[instruction] = entry
        if len(self._prompt_cache) > MAX_PROMPT_CACHE:
            self._prompt_cache.popitem(last=False)
        return entry

    def _capture_graphs(self) -> None:
        warmup_prompt = self._get_prompt("open the middle layer of the drawer")
        if warmup_prompt.static_ids is None:
            return
        assert warmup_prompt.static_pos_ids is not None
        assert warmup_prompt.static_gather_idx is not None
        assert warmup_prompt.static_image_pos is not None
        pool = torch.cuda.graph_pool_handle()
        capture_stream = torch.cuda.Stream(device=self.device)

        for b in range(1, self.max_batch_size + 1):
            grid_cpu = self._grid_cpu_by_b[b]
            self.vision_metadata.cpu_grid = grid_cpu
            self.vision_metadata.device_grid = grid_cpu
            self.vision_metadata._get_grid_plan(
                grid_cpu,
                IMAGES_PER_SAMPLE * PATCHES_PER_IMAGE * b,
                self.device,
                torch.bfloat16,
            )

            st_imgs = torch.zeros(
                (IMAGES_PER_SAMPLE * b, *IMAGE_SIZE, 3), device=self.device, dtype=torch.uint8
            )
            st_ids = warmup_prompt.static_ids.unsqueeze(0).expand(b, -1).contiguous().to(self.device)
            st_pos = (
                warmup_prompt.static_pos_ids.unsqueeze(1).expand(-1, b, -1).contiguous().to(self.device)
            )
            st_gather = (
                warmup_prompt.static_gather_idx.unsqueeze(0).expand(b, -1, -1).contiguous().to(self.device)
            )
            st_img_pos = (
                warmup_prompt.static_image_pos.unsqueeze(0).expand(b, -1, -1).contiguous().to(self.device)
            )

            def _step(
                imgs: torch.Tensor = st_imgs,
                ids: torch.Tensor = st_ids,
                pos: torch.Tensor = st_pos,
                g_idx: torch.Tensor = st_gather,
                i_pos: torch.Tensor = st_img_pos,
                batch_n: int = b,
            ) -> torch.Tensor:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    pv = self._gpu_patchify_224(imgs)
                    ie = self.vision_metadata.run_vision(pv, self.vision_metadata.device_grid)
                    emb = self.embed_tokens(ids).clone()
                    emb.scatter_(
                        1,
                        i_pos.expand(-1, -1, emb.shape[-1]),
                        ie.reshape(batch_n, IMAGE_TOKENS_PER_SAMPLE, -1).to(emb.dtype),
                    )
                    out = self.language_model(
                        input_ids=None,
                        position_ids=pos,
                        attention_mask=None,
                        past_key_values=None,
                        inputs_embeds=emb,
                        use_cache=False,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                    last_hidden = out.last_hidden_state
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float32):
                    queries = last_hidden.gather(1, g_idx.expand(-1, -1, last_hidden.shape[-1]))
                    return self.framework.action_model.predict_action(queries)

            capture_stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(capture_stream):
                for _ in range(3):
                    _ = _step()
            torch.cuda.current_stream(self.device).wait_stream(capture_stream)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                st_actions = _step()

            self._graphs[b] = _GraphSlot(
                graph=graph,
                static_images=st_imgs,
                static_ids=st_ids,
                static_pos_ids=st_pos,
                static_gather_idx=st_gather,
                static_image_pos=st_img_pos,
                static_actions=st_actions,
                host_images=torch.empty_like(st_imgs, device="cpu", pin_memory=True),
                host_ids=torch.empty_like(st_ids, device="cpu", pin_memory=True),
                host_pos_ids=torch.empty_like(st_pos, device="cpu", pin_memory=True),
                host_gather_idx=torch.empty_like(st_gather, device="cpu", pin_memory=True),
                host_image_pos=torch.empty_like(st_img_pos, device="cpu", pin_memory=True),
                host_actions=torch.empty_like(st_actions, device="cpu", pin_memory=True),
            )
        torch.cuda.synchronize(self.device)

    def predict(self, examples: Sequence[Mapping[str, Any]], **_kwargs: Any) -> dict[str, np.ndarray]:
        batch_n = len(examples)
        prompts = [self._get_prompt(str(ex["lang"])) for ex in examples]
        slot = self._graphs.get(batch_n)

        if (
            slot is not None
            and all(
                p.static_ids is not None
                and len(ex["image"]) == IMAGES_PER_SAMPLE
                and all(img.shape == (*IMAGE_SIZE, 3) and img.dtype == np.uint8 for img in ex["image"])
                for p, ex in zip(prompts, examples, strict=True)
            )
        ):
            grid_cpu = self._grid_cpu_by_b[batch_n]
            self.vision_metadata.cpu_grid = grid_cpu
            self.vision_metadata.device_grid = grid_cpu

            host_imgs_np = slot.host_images.numpy()
            for idx, (ex, prompt) in enumerate(zip(examples, prompts, strict=True)):
                imgs = ex["image"]
                host_imgs_np[2 * idx] = imgs[0]
                host_imgs_np[2 * idx + 1] = imgs[1]
                assert prompt.static_ids is not None
                assert prompt.static_pos_ids is not None
                assert prompt.static_gather_idx is not None
                assert prompt.static_image_pos is not None
                slot.host_ids[idx].copy_(prompt.static_ids)
                slot.host_pos_ids[:, idx, :].copy_(prompt.static_pos_ids)
                slot.host_gather_idx[idx].copy_(prompt.static_gather_idx)
                slot.host_image_pos[idx].copy_(prompt.static_image_pos)

            stream = torch.cuda.current_stream(self.device)
            slot.static_images.copy_(slot.host_images, non_blocking=True)
            slot.static_ids.copy_(slot.host_ids, non_blocking=True)
            slot.static_pos_ids.copy_(slot.host_pos_ids, non_blocking=True)
            slot.static_gather_idx.copy_(slot.host_gather_idx, non_blocking=True)
            slot.static_image_pos.copy_(slot.host_image_pos, non_blocking=True)
            slot.graph.replay()
            slot.host_actions.copy_(slot.static_actions, non_blocking=True)
            stream.synchronize()
            return {"normalized_actions": slot.host_actions.numpy().copy()}

        return self._predict_eager(examples, prompts)

    def _predict_eager(
        self,
        examples: Sequence[Mapping[str, Any]],
        prompts: Sequence[_CachedPrompt],
    ) -> dict[str, np.ndarray]:
        batch_n = len(examples)
        max_len = max(int(p.raw_ids.numel()) for p in prompts)
        ids_cpu = torch.full((batch_n, max_len), self.pad_token_id, dtype=torch.long)
        mask_cpu = torch.zeros((batch_n, max_len), dtype=torch.long)
        for idx, prompt in enumerate(prompts):
            length = int(prompt.raw_ids.numel())
            ids_cpu[idx, :length] = prompt.raw_ids
            mask_cpu[idx, :length] = 1

        grid_cpu = self._grid_cpu_by_b.get(batch_n)
        if grid_cpu is None:
            grid_cpu = torch.tensor(
                [[1, 16, 16]] * (IMAGES_PER_SAMPLE * batch_n), dtype=torch.long
            )
        pos_cpu, _ = self._get_rope_index(
            input_ids=ids_cpu,
            image_grid_thw=grid_cpu,
            attention_mask=mask_cpu,
        )
        pos_grid = torch.arange(max_len, dtype=torch.long).view(1, -1, 1).expand(batch_n, -1, 1)
        gather_cpu = self._gather_fn(
            pos_grid,
            ids_cpu,
            action_token_id=self.action_token_id,
        )

        imgs_np = np.empty((IMAGES_PER_SAMPLE * batch_n, *IMAGE_SIZE, 3), dtype=np.uint8)
        for idx, ex in enumerate(examples):
            imgs = ex["image"]
            imgs_np[2 * idx] = imgs[0]
            imgs_np[2 * idx + 1] = imgs[1]

        imgs_gpu = torch.from_numpy(imgs_np).to(device=self.device, non_blocking=True)
        ids_gpu = ids_cpu.to(device=self.device, non_blocking=True)
        pos_gpu = pos_cpu.to(device=self.device, non_blocking=True)
        gather_gpu = gather_cpu.to(device=self.device, non_blocking=True)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            pv = self._gpu_patchify_224(imgs_gpu)
            self.vision_metadata.cpu_grid = grid_cpu
            self.vision_metadata.device_grid = grid_cpu
            ie = self.vision_metadata.run_vision(pv, self.vision_metadata.device_grid)
            emb = self.embed_tokens(ids_gpu).clone()
            image_mask = (ids_gpu == self.image_token_id).unsqueeze(-1).expand_as(emb)
            emb = emb.masked_scatter(image_mask, ie.to(emb.dtype))
            out = self.language_model(
                input_ids=None,
                position_ids=pos_gpu,
                attention_mask=None,
                past_key_values=None,
                inputs_embeds=emb,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            last_hidden = out.last_hidden_state

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float32):
            queries = last_hidden.gather(1, gather_gpu.expand(-1, -1, last_hidden.shape[-1]))
            actions = self.framework.action_model.predict_action(queries)
        return {"normalized_actions": actions.detach().cpu().numpy()}

    def close(self) -> None:
        self._graphs.clear()
        self._prompt_cache.clear()
