# LIBERO-Pro runtime

Build the pinned Linux x86_64 simulator runtime with `build-runtime.sh <install-dir>`. The builder
verifies the `Zxy-MLlab/LIBERO-PRO` commit, all 320 perturbed BDDL and initial-state SHA-256 digests,
the CPython 3.11.18 conda lock, the checksum-pinned `robosuite==1.4.0` patches in
[`runtime-patches.json`](runtime-patches.json), and the Kit `--self-test`.

See the [LIBERO evaluation protocol](../../packages/roboscope-kit/LIBERO_EVALUATION_PROTOCOL.md#11-libero-pro)
for the frozen 16-suite `libero-pro-v1` scope, [`examples/starvla`](../../examples/starvla/README.md)
for the Qwen2.5-VL OFT Policy setup, and [`THROUGHPUT.md`](THROUGHPUT.md) for the conservative `strict`
baseline versus `--eval-mode fast` comparison and measurements.

## `strict` vs `fast` mode defaults (no Autotune)

[`benchmark-manifest.json`](benchmark-manifest.json) ships fixed `(K, g)` defaults for both evaluation
modes so runs execute directly without an Autotune probe phase (pass `--concurrency K --replicas G` to
override, or `--autotune-strategy diagonal` to run dynamic search):

| Setting | `--eval-mode strict` (default) | `--eval-mode fast` |
| --- | --- | --- |
| Default `(K, g)` | `K=12, g=3` (no Autotune) | `K=24, g=2` (no Autotune; 12 Workers/replica for `B=1..4` dynamic batching) |
| CUDA MPS | Off (`--cuda-mps` overrides) | On (`--no-cuda-mps` overrides) |
| Policy framing (`policy_encoding`) | `"json"` | `"msgpack"` (automatic JSON fallback; persistent per-Worker `PolicyClient` socket) |
| Camera rendering (`image_sizes`) | Canonical `256x256` + PIL resize (`canonical_observations: true`) | Negotiated native `224x224` (`image_sizes=[(224, 224), ANY_SIZE]`, skips CPU resize, `-23.4%` pixels/IPC, `canonical_observations: false`) |
| `chunk_camera_rendering` | `false` | `true` (skips intermediate RGB renders inside action chunks and the first `settle_steps - 1` settle steps; verified against `robosuite 1.4.0`; disabled when `save_videos: true`) |
| `compiled_model_cache` | `false` | `true` (2-entry pristine MuJoCo XML model cache per Worker) |
| `egl_context_reuse` | `false` | `true` (parks one idle EGL context per Worker thread/device across task switches) |
| `task_order` | `null` (native `0..9`) | `null` (optional permutation of `0..9` in manifest) |
| StarVLA QwenOFT batching | `max_batch_size=1`, `coalesce=false` (served individually) | `max_batch_size=4`, `coalesce_action_requests=true` (`ROBOSCOPE_STARVLA_MAX_BATCH_SIZE` overrides) |
| StarVLA QwenOFT model hotpath | Standard upstream forward | `QwenOFTFastRunner`: GPU `224x224` patchify, prompt/3D-RoPE/gather LRU cache, direct causal SDPA `language_model` execution, grouped equal-length vision SDPA windows, and shared-pool `CUDAGraph` (`B=1..4`, `STATIC_SEQ_LEN=224`; `=0` env vars disable individual stages) |

Set `ROBOSCOPE_TIMING_DIR=/abs/path` to collect nested Worker, Policy Server, and QwenOFT JSONL stage
spans via `roboscope_kit.profiling`, and summarize them with
`python benchmarks/libero_pro/analyze_timings.py <dir>`.

## Profiling

On 2026-11-16 to 2026-11-18, `libero_goal_object` was profiled on a dedicated RTX 4090 (49,140 MiB
VRAM, 20-core CPU quota, 72 GiB RAM) with StarVLA `Qwen2.5-VL-OFT-LIBERO-4in1` (`bf16`, SDPA, chunk 8,
`OMP_NUM_THREADS=1`) using Nsight Systems 2026.5.1 (`cuda-sw,nvtx,opengl`) and PyTorch operator/copy
attribution:

- **Model DtoH readbacks and stream syncs:** Each fixed-input prediction originally triggered 160
  device-to-host copies (96 from per-block Q/K/V `cu_seqlens` `.tolist()` reads, 46 from grid/window
  scalars, 13 from `get_rope_index`, 1 from action-token count validation, and 4 others). Keeping
  vision metadata on CPU reduced copies from 160 to 18 per prediction; adding CPU token/RoPE metadata
  reduced them to 4 (20/20 archived observations matched byte-for-byte).
- **Vision SDPA kernel launches:** Grouping equal-length Qwen2.5-VL vision windows into batched SDPA
  calls reduced local-window SDPA launches from 224 to 28 and full-image SDPA launches from 8 to 4 per
  prediction (+3.48% step throughput in a paired 500-episode `libero_goal_object` run).
- **Simulator rendering and resets:** With chunk camera gating active, Worker RGB sensor calls averaged
  1.9–2.2 ms on chunk boundaries and skipped 7/8 of intermediate action-step RGB renders. OpenGL host
  time during task switches remained concentrated in `glGenerateMipmap`, `glReadPixels`, and
  `eglMakeCurrent`, motivating the bounded compiled-model cache and idle EGL context reuse.
