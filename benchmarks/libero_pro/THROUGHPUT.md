# LIBERO-Pro throughput notes

This document summarizes the conservative `strict` baseline (`main`) and `--eval-mode fast`
optimizations for [issue #21](https://github.com/MINT-SJTU/RoboScope-Kit/issues/21) under the frozen
`libero-pro-v1` protocol (16 single-perturbation suites × 10 tasks × 50 initial states = 8,000
episodes, seed 7, 10 settle steps, perturbed BDDL instructions, StarVLA
`Qwen2.5-VL-OFT-LIBERO-4in1@a60fff4` in `bf16`).

## Comparison setup: conservative `strict` baseline vs `--eval-mode fast`

Both modes ship pre-selected fixed `(K, g)` defaults in [`benchmark-manifest.json`](benchmark-manifest.json)
and execute directly without an Autotune search phase:

- **Baseline (`--eval-mode strict`, default):** Matches conservative `main` behavior at fixed
  `K=12, g=3` — Autotune off, CUDA MPS off, single-request execution (`coalesce_action_requests=false`),
  JSON framing, canonical `256x256` RGB rendering every step (`canonical_observations=true`), Policy
  `OMP_NUM_THREADS=1`.
- **Candidate (`--eval-mode fast`):** Runs at fixed `K=24, g=2` — Autotune off, CUDA MPS on,
  MessagePack framing with persistent per-Worker `PolicyClient` sockets and lock-free server health probes,
  negotiated native `224x224` camera rendering (`image_sizes=[(224, 224), ANY_SIZE]`, skipping PIL CPU resize
  and marking `canonical_observations=false`), `chunk_camera_rendering=true` (gating both intermediate action
  chunk steps and the first `settle_steps - 1` unconsumed settle renders), `compiled_model_cache=true`,
  `egl_context_reuse=true`, cross-episode dynamic queue coalescing (`max_batch_size=4`,
  `coalesce_action_requests=true`), and `QwenOFTFastRunner` (pinned-memory GPU `224x224` patchify, per-prompt
  token/3D-RoPE/gather/scatter LRU cache, direct `language_model` execution with right-padded causal
  `attention_mask=None` FlashAttention-2 SDPA without `lm_head` or `DynamicCache` allocations, static
  GPU-resident vision `_GridPlan` cache with grouped equal-length SDPA windows, and shared-pool zero-sync
  `CUDAGraph` replay across `B=1..4` at tile-aligned `STATIC_SEQ_LEN=224`).

Formal evaluation throughput is `8000 / sum(evaluation_s)` across all 16 suites, reported alongside
actual `sum(steps) / sum(evaluation_s)` and end-to-end CLI wall time.

## Measured reference results on dedicated RTX 4090 (48 GiB VRAM)

### 8,000-episode full-matrix baselines and fast-path progression

| Configuration (all 16 suites, 8,000 episodes) | `K/g` | MPS | Batch cap | Summed formal `evaluation_s` | Episodes/s | Steps/s | Speedup vs conservative `main` |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Historical #19 fixed baseline (`0.8.0b29`) | `2/1` (13), `8/2` (3) | Off | 1 | 20,853.71 | 0.3836 | 95.89 | 0.48× |
| **Conservative `main` strict baseline (`a7f756c`, `0.8.0b66`)** | **`12/3` fixed** | **Off** | **1** | **10,060.22** | **0.7952** | **198.55** | **1.00×** |
| Earlier fast-path run (MPS + msgpack + camera gate + `batch_cap=2`, `0.8.0b38`) | per-suite | On | 2 | 6,725.75 | 1.1895 | 297.04 | 1.50× |
| Hotpath 16-suite run (+ model cache + CPU metadata, `0.8.0b72`, pre-grouped-windows) | per-suite | On | 1 | — | — | 329.14 | 1.66× (steps/s) |
| **Final `--eval-mode fast` full 16-suite run (`0.8.0b92`, 2026-11-27)** | **`24/2` fixed** | **On** | **4** | **4,945.12** | **1.6178** | **404.81** | **2.034×** |

Under conservative `main` (`strict`, MPS off, `batch_cap=1`), a 128-point `(K, g)` grid sweep on
`libero_goal_object` (`K=1..32, g=1..4`) confirmed `K=9..12, g=3..4` as the throughput plateau
(`172.8–179.4 steps/s`), with `g=3` achieving the lowest end-to-end CLI wall time (`78.7–82.6 s` vs
`88.4–90.2 s` at `g=4`) and zero OOMs across all 8,000 episodes.

### Per-suite 8,000-episode comparison: conservative `main` (`strict`) vs `--eval-mode fast` (`0.8.0b92`)

All 16 suites ran end-to-end on one dedicated RTX 4090 (48 GiB VRAM, 20-core CPU quota, 72 GiB RAM) at default
fixed `K=24, g=2` (`0` OOM events, `0` Worker failures, `8,000 / 8,000` unique terminal episode records with
no gaps or retries). Summed across the 16 standalone CLI runs, `runtime_prepare_s` totaled `33.68 s` (`2.10 s`
mean), 2-replica `policy_load_s` (including `B=1..4` CUDA Graph capture) totaled `443.49 s` (`27.72 s` mean),
`autotune_s` totaled `0.006 s`, formal `evaluation_s` totaled `4,945.12 s` (**2.034×** vs `10,060.22 s`),
result writing and MPS teardown totaled `81.44 s` (`5.09 s` mean), end-to-end `cli_wall_s` totaled
`5,503.73 s`, active-phase GPU utilization averaged `97.2%–99.5%`, peak VRAM ranged from `25,405` to
`35,980 MiB` (`24.8–35.1 GiB`), and overall success rate was `3,760 / 8,000 = 47.00%` (vs
`3,802 / 8,000 = 47.53%` under conservative `strict`):

| Suite | `strict` SR (`/500`) | `fast` SR (`/500`) | `strict` `evaluation_s` | `fast` `evaluation_s` | `fast` `steps/s` | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `libero_goal_object` | 348 (69.6%) | 322 (64.4%) | 423.08 | 222.88 | 385.61 | **1.90×** |
| `libero_goal_swap` | 5 (1.0%) | 3 (0.6%) | 753.33 | 362.03 | 414.00 | **2.08×** |
| `libero_goal_lan` | 447 (89.4%) | 455 (91.0%) | 331.52 | 171.90 | 354.17 | **1.93×** |
| `libero_goal_task` | 46 (9.2%) | 49 (9.8%) | 698.30 | 335.75 | 409.89 | **2.08×** |
| `libero_spatial_object` | 480 (96.0%) | 475 (95.0%) | 286.83 | 143.67 | 371.00 | **2.00×** |
| `libero_spatial_swap` | 0 (0.0%) | 0 (0.0%) | 584.09 | 285.06 | 385.88 | **2.05×** |
| `libero_spatial_lan` | 473 (94.6%) | 477 (95.4%) | 302.18 | 160.46 | 326.11 | **1.88×** |
| `libero_spatial_task` | 277 (55.4%) | 270 (54.0%) | 424.20 | 219.02 | 361.13 | **1.94×** |
| `libero_10_object` | 238 (47.6%) | 236 (47.2%) | 987.23 | 480.26 | 427.17 | **2.06×** |
| `libero_10_swap` | 0 (0.0%) | 0 (0.0%) | 1,252.74 | 599.80 | 433.48 | **2.09×** |
| `libero_10_lan` | 454 (90.8%) | 454 (90.8%) | 681.51 | 336.95 | 402.58 | **2.02×** |
| `libero_10_task` | 51 (10.2%) | 41 (8.2%) | 1,224.89 | 587.14 | 433.99 | **2.09×** |
| `libero_object_object` | 452 (90.4%) | 446 (89.2%) | 380.53 | 192.65 | 388.12 | **1.98×** |
| `libero_object_swap` | 0 (0.0%) | 0 (0.0%) | 702.73 | 335.84 | 416.86 | **2.09×** |
| `libero_object_lan` | 483 (96.6%) | 483 (96.6%) | 364.72 | 188.73 | 366.69 | **1.93×** |
| `libero_object_task` | 48 (9.6%) | 49 (9.8%) | 662.35 | 322.99 | 412.36 | **2.05×** |
| **Total (8,000 episodes)** | **3,802 (47.53%)** | **3,760 (47.00%)** | **10,060.22** | **4,945.12** | **404.81** | **2.034×** |

### `--eval-mode fast` component ablations (`libero_goal_object` / `libero_goal_swap`)

- **MessagePack + chunk camera gate (`K=12/g=3`, MPS off):** `423.08 s` → `410.11 s` (+5.1% steps/s).
- **CUDA MPS + compiled MuJoCo model cache + CPU vision metadata (`K=12/g=4`):** `281.26 s` → `272.61 s`
  (`295.69` → `304.83 steps/s`, DtoH copies `160` → `18` per prediction).
- **CPU token/RoPE metadata + idle EGL context reuse (`K=12/g=4`):** `272.54 s` → `267.48 s`
  (`304.22` → `309.61 steps/s`, DtoH copies `18` → `4`, `512` EGL context reuse hits, `31,536` replay
  arrays byte-identical).
- **Grouped equal-length vision attention windows (`K=16/g=4`, `batch_cap=1`, `256x256`):** `270.78 s` →
  `258.62 s` (`301.98` → `312.49 steps/s`, vision SDPA calls `232` → `32` per prediction).
- **Static `_GridPlan` cache + zero-copy protocol (`K=16/g=4`, `batch_cap=1`, `256x256`, 9-suite / 4,500-episode
  pass on 2026-11-27):** `250.48 s` (`327.50 steps/s`, **1.69×**) on `libero_goal_object`, `430.11 s`
  (`347.95 steps/s`, **1.75×**) on `libero_goal_swap`, and `2,837.62 s` vs `4,790.76 s` (**1.69×**,
  `329.05 steps/s`) across the first 9 suites.
- **Negotiated native `224x224` cameras + dynamic batching (`K=16/g=2` → `K=24/g=3`, eager QwenOFT, 2026-11-27):**
  `240.27 s` (`347.22 steps/s`, **1.76×**) on `libero_goal_object` and `399.86 s` (`374.51 steps/s`, **1.88×**)
  on `libero_goal_swap` at `K=16/g=2` (**1.84×** combined); `231.45 s` (`368.58 steps/s`, **1.83×**) at
  `K=24/g=3` (`38.8 GiB` VRAM).
- **`QwenOFTFastRunner` (GPU `224x224` patchify + prompt cache + direct causal SDPA backbone + zero-sync
  `CUDAGraph` at `STATIC_SEQ_LEN=224`) + persistent Worker sockets + settle render gating (`K=24/g=2`,
  `max_batch_size=4`, 2026-11-27):** Single-replica `Policy.act_batch` latency dropped from `41.90 ms` to
  `23.14 ms` at `B=1` (`345.8 steps/s`) and from `83.80 ms` to `66.51 ms` at `B=4` (`481.1 steps/s`). Closed-loop
  `K=24/g=2` reached `222.88 s` (**1.90×**) on `libero_goal_object`, `362.03 s` (**2.08×**, `414.00 steps/s`)
  on `libero_goal_swap`, and **4,945.12 s** vs **10,060.22 s** (**2.034×**, `404.81 steps/s`, `29.2–35.1 GiB`
  VRAM) across all 16 suites (`8,000` episodes).

## One-command reproduction (`strict` vs `fast`)

1. Build a clean simulator runtime (`benchmarks/libero_pro/build-runtime.sh /sim/libero-pro`), prepare
   the StarVLA snapshot (`examples/starvla/prepare_snapshot.py libero_pro /models/starvla-libero-pro`),
   and set `OMP_NUM_THREADS=1` in the evaluation environment.
2. Run conservative `strict` baseline (`K=12, g=3`, MPS off, `256x256`, unbatched, no Autotune) and
   `--eval-mode fast` (`K=24, g=2`, MPS on, `224x224` native rendering, `max_batch_size=4` coalescing,
   `QwenOFTFastRunner` CUDA Graph, no Autotune) with the exact same manifest and Policy config:

   ```bash
   # Conservative strict baseline (defaults to K=12, g=3, MPS off, 256x256, single-request)
   python -m roboscope_kit.runtime.benchmark \
     --benchmark libero_pro --suite libero_goal_object \
     --checkpoint-source /models/starvla-libero-pro \
     --benchmark-manifest /sim/libero-pro/benchmark-manifest.json \
     --policy-config /configs/starvla-policy.json \
     --output-dir /runs/libero-pro-strict/libero_goal_object

   # Fast mode (defaults to K=24, g=2, MPS on, 224x224 native cameras, batch<=4 coalescing, QwenOFT CUDA Graph)
   python -m roboscope_kit.runtime.benchmark \
     --benchmark libero_pro --suite libero_goal_object \
     --checkpoint-source /models/starvla-libero-pro \
     --benchmark-manifest /sim/libero-pro/benchmark-manifest.json \
     --policy-config /configs/starvla-policy.json \
     --eval-mode fast \
     --output-dir /runs/libero-pro-fast/libero_goal_object
   ```

   On 24 GiB GPUs, pass `--concurrency 16 --replicas 1` in `--eval-mode fast` (or `--concurrency 8 --replicas 2`
   in `strict` mode).
