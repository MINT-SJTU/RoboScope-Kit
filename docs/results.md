# Evaluation Results

Detailed results for the headline numbers in the [main README](../README.md#results), organized by benchmark. All
evaluation times exclude initialization, autotuning or offline tuning, and cleanup unless noted otherwise.

## LIBERO-Long

Evaluated on `libero_10` (10 tasks × 50 initial states, seed 7, 500 episodes) on one 48 GiB RTX 4090 with
[`StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1`](https://huggingface.co/StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1). From the
[technical report](https://arxiv.org/abs/2611.03850) (Table 3):

| System | Success | Δ success vs. StarVLA [95% CI] | Agreement | Eval time | Throughput | Speedup vs. StarVLA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| StarVLA official | 90.2% | reference | reference | 2,990 s (49.8 min) | 45.8 steps/s | 1.00× |
| RLinf (tuned) | 90.3% | +0.1 pp [-2.6, +3.3] | 88.1% | 1,882 s (31.4 min) | 74.2 steps/s | 1.60× |
| vla-eval (tuned) | 89.8% | -0.4 pp [-1.6, +0.7] | 91.7% | 1,016 s (16.9 min) | 133.5 steps/s | 2.93× |
| **RoboScope Kit** (`K12/g3`) | **89.9%** | -0.3 pp [-0.8, 0.0] | **97.9%** | **671 s (11.2 min)** | **204.6 steps/s** | **4.45×** (2.78× RLinf, 1.52× vla-eval) |
| **RoboScope Kit** (`K16/g4`, CUDA MPS) | **89.9%** | -0.3 pp [-1.3, +0.8] | **96.5%** | **485 s (8.1 min)** | **282.5 steps/s** | **6.16×** (3.85× RLinf, 2.10× vla-eval) |

Run records: [`autotune-diag-w1-20261101`](https://github.com/suica/evostudio-kit-tech-report/tree/main/data/results/autotune-diag-w1-20261101)
(and H1 autotune search in [`autotune-paper-20261031`](https://github.com/suica/evostudio-kit-tech-report/tree/main/data/results/autotune-paper-20261031)).

### OpenVLA across four LIBERO suites (200-episode subset)

Ten tasks × initial states 0–4 per suite on one RTX 4090 at `K2/g2`
([experiment report](../examples/openvla/EXPERIMENT_REPORT.md)):

| Suite | Native success | Kit `2/2` success | Native eval (min) | Kit eval (min) |
| --- | ---: | ---: | ---: | ---: |
| Spatial | 43/50 | 40/50 | 22.96 | 19.22 |
| Object | 42/50 | 43/50 | 28.66 | 24.19 |
| Goal | 33/50 | 38/50 | 29.11 | 22.66 |
| Long | 28/50 | 30/50 | 64.20 | 57.62 |
| **Total** | **146/200** | **151/200** | **144.93** | **123.68** |

## RoboTwin Easy

Evaluated on `demo_clean` with
[`StarVLA/Qwen3-VL-OFT-Robotwin2`](https://huggingface.co/StarVLA/Qwen3-VL-OFT-Robotwin2) (chunk length 16).

### 1× RTX 4090 (500 episodes, 50 tasks × 10)

| System | Configuration | Successes | Same outcome | Same step | Eval time | End-to-end | Throughput | Slowdown vs. Kit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| StarVLA official | 1 env | 232/500 | 500/500 | 500/500 | 25,238 s (7.0 h) | 27,165 s | 8.4 steps/s | 13.2× |
| vla-eval | 9 shards, batch 1 | 232/500 | 500/500 | 500/500 | 14,072 s (3.9 h) | 14,085 s | 15.0 steps/s | 7.3× |
| RLinf | 8 envs | 230/500 | 434/500 | 287/500 | 4,237 s (70.6 min) | 4,286 s | 49.9 steps/s | 2.2× |
| **RoboScope Kit** | `K8/g1` | **232/500** | reference | reference | **1,916 s (31.9 min)** | **1,931 s** | **110.2 steps/s** | **1.0×** |

Run records: [`robotwin-baselines-20261031`](https://github.com/suica/evostudio-kit-tech-report/tree/main/data/results/robotwin-baselines-20261031).

### 4× RTX PRO 5000 (500 and 5,000 episodes)

Against four parallel official StarVLA evaluators on the same four GPUs. Every RoboScope Kit launch matches the
official evaluation on every success outcome and terminal step:

| Workload | System | Runs | Successes | Eval time | vs. official |
| --- | --- | ---: | ---: | ---: | ---: |
| 500 episodes (50 × 10) | StarVLA official | 3 | 230/500 | 7,076 s (mean) | 1.0× |
| 500 episodes (50 × 10) | **RoboScope Kit** (`K60/g8`) | 1 | **230/500** | **706 s (11.8 min)** | **10.0×** |
| 5,000 episodes (50 × 100) | StarVLA official | 1 | 2,313/5,000 | 63,036 s (17.5 h) | 1.0× |
| 5,000 episodes (50 × 100) | **RoboScope Kit** (`K60/g8`) | 1 | **2,313/5,000** | **3,985 s (66.4 min)** | **15.8×** |

Run records: [`robotwin-4gpu-pro5000-20261109`](https://github.com/suica/evostudio-kit-tech-report/tree/main/data/results/robotwin-4gpu-pro5000-20261109),
[`robotwin-4gpu-pro5000-5000-20261111`](https://github.com/suica/evostudio-kit-tech-report/tree/main/data/results/robotwin-4gpu-pro5000-5000-20261111).

## RoboDojo

126 episodes (layouts 0–2 of all 42 `arx_x5` tasks) on one 48 GiB RTX 4090 with
[`Simate_beta`](https://huggingface.co/SIPAILab/sipai-robodojo-eval-ae) (`K2/g1`, autotuned). Paired by task and
layout, 6 episodes succeed only in Kit and 6 only upstream (McNemar *p* = 1.0). From
[`examples/xpolicylab/README.md`](../examples/xpolicylab/README.md#results):

| Metric | RoboScope Kit | Upstream native |
| --- | ---: | ---: |
| Successes | 26/126 (20.6%) | 26/126 (20.6%) |
| Mean score | 0.278 | 0.271 |
| Evaluation time | **4.4 h (15,770 s)** | 6.6 h (23,753 s) |
| Total time (incl. 30 min autotuning) | 4.9 h (17,576 s) | 6.6 h (23,753 s) |

Run records: [`robodojo-simate-acceptance-20261106`](https://github.com/suica/evostudio-kit-tech-report/tree/main/data/results/robodojo-simate-acceptance-20261106).

## RoboCasa365

250 episodes (`target50` × 5, 452,250 steps at official horizons) on one 48 GiB RTX 4090 with the official
[`Diffusion Policy`](https://huggingface.co/robocasa/robocasa365_checkpoints) (`pretrain_human300`, FP32, 100 DDPM
steps) at `K4/g2` with CUDA MPS, three runs per arm. All six runs pass the native comparison checks. From
[`benchmarks/robocasa365/README.md`](../benchmarks/robocasa365/README.md#throughput):

| Metric | Kit baseline | Kit optimized (threaded slots + CUDA graph) | Speedup |
| --- | ---: | ---: | ---: |
| Median eval time | 9,810 s (163.5 min) | **4,850 s (80.8 min)** | **2.02×** |
| Best eval time | 9,703 s (161.7 min) | **4,637 s (77.3 min)** | **2.09×** |
| Throughput (median) | 46.1 steps/s | **93.2 steps/s** | **2.02×** |

Run records: [`robocasa365-throughput-20261121`](https://github.com/QLL-CCDD/RoboScope-Kit/releases/tag/robocasa365-throughput-20261121).

## ManiSkill

1,100 episodes across the 11 official PPO tasks (of the 55 inventory tasks; full integration acceptance is incomplete)
on one 48 GiB RTX 4090 with
[`Official PPO`](https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations/tree/3558f72cf99d51d04a6ac4779e88b3da4bbf1628)
checkpoints, matching all native records. From
[`benchmarks/maniskill/VALIDATION.md`](../benchmarks/maniskill/VALIDATION.md#throughput-by-worker-count):

| Workers | Eval before task-spread (s) | Eval after task-spread (s) | Speedup vs. 1 Worker (426.2 s) | Peak VRAM after (GiB) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 426.2 ± 3.1 | 425.1 | 1.00× | 4.7 |
| 4 | 245.1 ± 1.6 | 196.9 ± 2.4 | 2.16× | 16.5–16.8 |
| 8 | 226.6 ± 3.3 | **182.7 ± 0.5** | **2.33×** | 32.5–32.6 |
| 12 | 192.5 ± 3.9 | 184.1 ± 0.4 | 2.32× | 47.3–47.4 |

## Other validated benchmarks

All remaining benchmarks in the repository have native-comparison or acceptance validation without cross-system
speedup measurements:

- **LIBERO-Plus / LIBERO-Pro:** [LIBERO protocol](../packages/roboscope-kit/LIBERO_EVALUATION_PROTOCOL.md)
- **RoboCasa GR-1 Tabletop:** [Benchmark manifest](../benchmarks/robocasa/benchmark-manifest.json)
- **CALVIN:** [CALVIN protocol](../packages/roboscope-kit/CALVIN_EVALUATION_PROTOCOL.md)
- **MetaWorld:** [MetaWorld protocol](../packages/roboscope-kit/METAWORLD_EVALUATION_PROTOCOL.md)
- **RLBench:** [PerAct comparison](../examples/peract/README.md)
- **robomimic:** [robomimic comparison](../examples/robomimic/README.md)
- **SimplerEnv:** [RT-1-X comparison](../examples/rt1/README.md)
