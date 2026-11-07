# Native RTX 4090: RoboTwin MPS and hardware counters

**Keep MPS off for the measured native K2/G1 workload.** Two unprofiled repetitions per arm give
**−0.602% pooled formal throughput** with identical episode results. K1's apparent gain falls from
+12.593% in R2 to +0.707% after reversing order in R3; its pooled +6.703% is dominated by the first slow OFF run.
This 2026-11-07 follow-up adds hardware counters to the [48 GiB investigation](profiling-mps.md); platforms are not pooled.

## Environment and matched workloads

The native host has an **i9-14900K**, 32 GB RAM and **RTX 4090, 24564 MiB**, driver **580.178.04**,
Ubuntu 22.04.5/kernel 6.8.0-138, and Nsight Systems 2026.6.1. Kit is **0.8.0b31**, PR98 head
`36c950ae9b54d8f82d38e088e355e26dc6d79ef1`, based on main `44fa1ad`.
The generic SDK `--profile-gpu-metrics` collector works here; no global counter-permission, driver/module or reboot change was made. Samples are device-wide, not per-process shares.

| Experiment | Environment / workload | Purpose |
| --- | --- | --- |
| Primary R2/R3 | Official simulator; first 10 frozen `adjust_bottle` episodes, shard 1, instruction limit 10; K1/G1 and K2/G1 off/on | Uninstrumented speed/correctness |
| Serial R2 | Same 10 episodes, K1/G1 off; wall timing and `ad10x` | Disjoint wall budget |
| Replacement R4 | Same 10 episodes, fixed K2/G1 off/on; separate `ad10x-gfxt`/`ad10x` passes at 500 Hz | Valid hardware/host pairs |
| Representative R2 | Ten representative tasks × 2 episodes, shard 2; requested K2/G1 | Capacity diagnostics after recovery |
| Static R3/R4/R5 | Existing simulator; initial `adjust_bottle` scene, seed 1000000; one/two continuously observing scenes | Renderer counters/CPU stacks |

Formal simulator: **Python 3.8.20, Torch 2.4.1+cu121, SAPIEN 3.0.0b1, mplib 0.2.1, NumPy 1.24.4** and pinned
cuRobo (distribution 0.0.0), built by the unmodified official builder. Control/Policy Python is 3.11.15; Policy uses Torch 2.6.0+cu124, FlashAttention 2.7.4.post1 and Transformers 4.57.0.
Static controls use the existing Python 3.10.20/Torch 2.4.1/NumPy 1.26.4/cuRobo 0.7.8 environment;
SAPIEN/mplib versions match, but its non-Git source is identified by saved file hashes.

Formal arms share affinity `0-11,14-21` (12/13 offline), one OMP/MKL/OpenBLAS thread, dedicated caches, the native NVIDIA ICD and identical checkpoint/config. The state-omitted native Policy recipe is fixed;
changing that recipe is not SDK speedup. The stock factory validates the complete frozen generation first.
Policy chunks contain up to 16 actions, unlike the static continuous-observation loop. Four 320×240 RGB cameras
use RT 32 spp, path depth 8 and OIDN: head, front and both wrists; Policy consumes three. Video, saved data,
depth and point clouds are off. These research subsets are not the official 5000-episode evaluation.

## Speed, correctness and resources

R2 uses OFF→ON; R3 reverses ON→OFF, K2 before K1. Each arm has two 10-episode executions.
Formal is SDK `evaluation_s`, including Worker/scene startup and excluding loading/cleanup; E2E is command lifetime.
Admission waiting is excluded. Rates pool completed episodes/steps over formal time.

| K/G | MPS | Formal mean±sample SD, s | E2E mean±sample SD, s | Episodes/h | Steps/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 1/1 | off | 59.317±5.422 | 67.306±5.457 | 606.9 | 18.325 |
| 1/1 | on | 55.591±0.703 | 63.768±0.915 | 647.6 | 19.554 |
| 2/1 | off | 32.754±0.145 | 40.674±0.144 | 1099.1 | 33.186 |
| 2/1 | on | 32.953±0.425 | 41.043±0.561 | 1092.5 | 32.986 |

K2 effects are **−1.193% / −0.000294%** in R2/R3. K1's first OFF formal time is 63.151 s versus 55.483 s
next repeat; n=2 sample SD is descriptive, not a confidence interval. G1 observes exactly **one MPS Policy
client**, even at K2: MPS applies to Policy CUDA, not simulator Vulkan graphics.
All eight primary runs retain K without OOM/recovery/restart/timeout. All ten `(id, task, seed, success, steps)`
tuples match: **10/10 successes, 1087 steps per run**. All four R4 diagnostics match these terminal tuples too.

| K/G | MPS | GPU busy mean% | Device VRAM mean/peak, MiB | CPU mean cores | Tree RSS mean/peak, GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| 1/1 | off | 10.734 | 17372.2/17948 | 0.978 | 6.121/6.583 |
| 1/1 | on | 14.894 | 17366.5/17982 | 0.985 | 5.947/6.227 |
| 2/1 | off | 22.111 | 19614.8/21449 | 1.816 | 8.007/9.030 |
| 2/1 | on | 20.700 | 19680.7/21483 | 1.805 | 8.195/9.275 |

Means are formal-duration weighted over approximately 1 Hz samples; peaks can miss spikes. Device memory
includes a resident background CUDA service (about 4184 MiB), display and driver; low background CPU/socket
activity does not establish zero GPU work. RSS can double count shared pages. ON adds 34 MiB sampled peak memory;
GPU busy is not instruction throughput or compute saturation.

## Serial wall budget and inclusive latency

The separate K1/OFF diagnostic has **55.496 s formal** and **48.021 s post-setup active** time.
Active starts at actual `setup_demo` completion and ends at the pre-close terminal snapshot, including reset,
instructions and first observations. Interval subtraction makes these buckets disjoint.

| Wall bucket | Formal seconds / share | Active seconds / share |
| --- | ---: | ---: |
| Episode setup | 4.247 / 7.654% | 0 / 0% |
| Observation/render/readback/assembly | 14.452 / 26.041% | 14.452 / 30.095% |
| Policy RPC round trip | 3.618 / 6.519% | 3.615 / 7.529% |
| Action-side simulation/control/wait, excluding nested observations | 29.900 / 53.878% | 29.900 / 62.264% |
| Episode close | <0.0001 / <0.001% | 0 / 0% |
| Other/uninstrumented | 3.279 / 5.909% | 0.054 / 0.112% |

Inclusive action totals 30.361557 s contain 0.461636 s of nested observations; subtraction leaves 29.899921 s.
The active budget is principally action-side simulation/control/wait, then observation/render/readback, then
RPC. Action includes driver/scheduler waiting and is not pure physics; get_obs is not pure GPU rendering.
Residual wall time is not labelled idle. These are serial wall classifications, not GPU execution fractions.

The following diagnostic host values are **mean/p95 milliseconds**; each case has 72 Policy acts, 82 observations and 1087 control actions. K2 columns are the valid R4 GFX pair, separate from primary speed runs.

| Span | Serial K1 off | K2 off, GFX | K2 on, GFX |
| --- | ---: | ---: | ---: |
| Policy act | 45.766/46.318 | 50.845/76.763 | 52.733/97.563 |
| Act queue wait | 0.002/0.002 | 7.921/33.708 | 8.560/36.897 |
| Act service | 45.999/46.591 | 51.086/77.008 | 52.969/97.819 |
| Act RPC round trip | 50.179/52.815 | 63.711/96.715 | 66.055/106.382 |
| Observation/render | 176.240/184.873 | 201.544/241.160 | 200.667/192.876 |
| Simulation action, inclusive | 27.932/44.689 | 29.290/47.457 | 28.643/45.696 |

Observation medians are 49.830/50.296/50.288 ms; initialization tails raise means. RPC minus aggregate
queue/service means is 4.178/4.704/4.526 ms, an unattributed residual, not paired wire latency.
Act is nested within service/RPC. K2 Policy-act/get_obs intersections are 0.620/0.848 s: overlapping process
totals cannot form an additive global wall pie or an Amdahl estimate.

## Valid K2 hardware-counter pairs

All four R4 runs retain K2 without recovery. GFX formal windows are 33.859/33.654 s off/on; general windows
are 33.257/32.851 s. Counters use formal phase/monotonic alignment, with UTC cross-checks within 268 ns.
Means are weighted over covered intervals; p95 describes observed samples. Sets are separate passes.

| Exported metric | Off mean/p95,% | On mean/p95,% | Coverage off/on,% |
| --- | ---: | ---: | ---: |
| SM Throughput (`ad10x-gfxt`) | 2.481/18 | 2.504/18 | 99.6001/100 |
| RTCORE Throughput (`ad10x-gfxt`) | 1.389/14 | 1.396/14 | 99.6001/100 |
| VRAM Throughput (`ad10x-gfxt`) | 3.418/27.9 | 3.442/28 | 99.6001/100 |
| SMs Active (`ad10x`) | 12.924/86 | 13.221/87 | 100/100 |
| SM Issue (`ad10x`) | 2.547/18 | 2.595/19 | 100/100 |
| Tensor Active (`ad10x`) | 1.737/18 | 1.809/19 | 100/100 |
| DRAM Read Bandwidth (`ad10x`) | 2.631/21 | 2.701/24 | 100/100 |
| DRAM Write Bandwidth (`ad10x`) | 1.157/5 | 1.212/5 | 100/100 |

GFX off retains one overflow warning; GFX on and both general passes have none formally. Assigning every
missing OFF interval 100% RT yields a full-window RT mean upper bound **1.784%** (lower 1.384%); ON is
fully covered at 1.396%. Both observed maxima are 33%, with no observed time above 90%. This rejects sustained
whole-window RT saturation at the sampled scale; shorter bursts and the actual limiting stage remain distinct questions.
The [NVIDIA metric guide](https://docs.nvidia.com/nsight-systems/UserGuide/#gpu-metric-sampling) defines
these exported percentages: they are not core counts, TFLOPS, rays/s or measured GB/s. SMs Active counts
cycles with a warp in flight, distinct from instruction throughput. Device counters combine Policy/rendering/background work.

## Continuous static-render controls

The legacy controls observe one fixed scene for 20 seconds after separately recorded setup/first observations.
Their four-camera RT32/OIDN loop omits Policy/control actions. Unprofiled aggregate observations/s fall
**24.955→19.102 (−23.45%)** with two renderers, so counter collection alone does not create the penalty.

| Static metric | One renderer | Two renderers |
| --- | ---: | ---: |
| RT throughput mean/p95,% | 10.44/22 | 8.10/21 |
| SM throughput mean/p95,% | 14.64/28 | 11.26/27 |
| VRAM throughput mean/p95,% | 3.57/8 | 2.94/8 |
| GFX coverage,% | 98.295 | 99.012 |
| Whole-window RT mean bounds,% | 10.26–11.97 | 8.02–9.01 |
| Sampled device busy mean,% /renderer VRAM, MiB | 83.35/3274 | 94.55/6548 |
| SMs Active / Tensor Active mean,% (general set) | 61.713/0.700 | 55.097/0.532 |
| DRAM read/write mean,% (general set) | 1.428/2.652 | 1.481/2.077 |
| Process-tree CPU mean cores (GFX pass) | 0.982 | 1.944 |

The general-set rows use separate complete 1000 Hz captures; GFX rows use 500 Hz and the stated partial coverage.

Each static GFX pass retains four overflows; even assigning gaps 100% does not imply sustained RT saturation.
Clean R5 CPU stacks resolve `takePicture → RTRenderer::render → DenoiserOidn::denoise → oidnExecuteFilter
→ UNetFilter::execute → CUDAEngine::wait → cudaStreamSynchronize → cuStreamSynchronize`.
Driver leaves are 85.84/89.05% of on-CPU samples; synchronization appears in 88.73/91.70% of deeper-backtrace
samples (one in four IP samples), with 0.990/1.970 scheduled CPU cores. Unresolved internal driver symbols
prevent labelling every driver sample pure polling or every wait OIDN compute; queued CUDA/Vulkan dependencies
can contribute. Low RT/SM/VRAM figures alone do not identify the limiting stage; this loop cannot replace the rollout wall budget.

## Capacity, exploration and reproduction

All four representative20 profiles OOM during SAPIEN camera-buffer allocation and recover **K2→1**.
Final 20 de-duplicated tuples agree: **9 successes, 11 native-horizon failures, 7272 steps**; three captures
retain 21 terminal snapshots from retries. These are capacity evidence, not fixed-K2 hardware/MPS pairs.
The earlier 48 GiB 8/20 successes and 7415 steps are not reproduced; equal weight hashes do not establish
cross-environment rollout equivalence. Preserved R1 Git-ownership aborts are excluded from MPS runtime-fault counts.

G2 is not attempted on this 24 GB host: K2/G1 peaks at 21483 MiB and another Policy needs about 10212 MiB.
On the earlier 48 GiB host, 100-episode K6/G2 confirmation gave only **+0.051623%** formal throughput,
adding 10.056 GiB mean VRAM and worsening E2E throughput 2.847%. Keep the existing bounded autotune budget;
this follow-up tests neither native autotune nor four-GPU scaling.

Pins: RoboTwin base `c3ddfa8b97d5519efa828b075999bd0006778e5e` → runtime `0fe7d15a86ae19d9411a52520683039cfc81d9ff`;
cuRobo `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`; StarVLA `7e4a47360093d79dc42b883885c04fbe790c5b60`;
model revision `e9cdfc38a0db968bf3a865bcb542d3d455952226`. Checkpoint SHA256:
`8ccee8ddb33fa5df83a9a5e2b7d4f737db93f19a051b23f738af8a484fe2d7e5`; native config SHA256:
`1c0cf1828c4cff02e6f8073ac0ec54a7423f7b1f249df057be6287c8ab3fe8a5`.

Rebuild through official `build-runtime.sh <install-dir>` and three verified asset archives; the 50-task self-test
and both seed generations pass. Evidence preserves exact source/config hashes and frozen external scripts:
`run-quick-only.sh` (1/1 success, 87 steps, one MPS client, 557115 counter samples), `run-campaign-r2.sh`,
`run-confirmation-r3.sh`, `run-adjust-counters-r4.sh`. These use normal CLI orchestration; wall timers do not
synchronize CUDA per step. Export reports with `nsys export --type=sqlite --output=report.sqlite report.nsys-rep`.
The collector starts/stops/exports successfully; later “No active sessions” cleanup is not failed collection.
QuickTest takes **12.615 s SDK formal /30.892 s E2E**; stock `result.elapsed_s` is a different 13.516 s boundary.

After installing the measured SDK in the controller and Policy environments, use the recorded paths/configuration
and an empty output directory. The native counter collector needs administrator permissions. The runner clears
`SUDO_*` and sets process-local Git trust for only the verified runtime directory; no global Git setting changes.

```bash
bash benchmarks/robotwin/build-runtime.sh /workspace/sim/robotwin
export EVIDENCE=/workspace/robotwin-baremetal-20261107 # extracted data archive
export ROBOSCOPE_CAMPAIGN_DIR=/workspace/reproduce-native
for script in run-quick-only.sh run-measurements-r2.sh run-confirmation-r3.sh run-adjust-counters-r4.sh; do
  sudo --preserve-env=ROBOSCOPE_CAMPAIGN_DIR bash "$EVIDENCE/scripts/native/$script"
done
```

`run-measurements-r2.sh` also reproduces the separately reported representative-scene capacity diagnostics.

[Evidence release](https://github.com/QLL-CCDD/RoboScope-Kit/releases/tag/robotwin-native-mps-evidence-20261107)
and [SHA256 manifest](https://github.com/QLL-CCDD/RoboScope-Kit/releases/download/robotwin-native-mps-evidence-20261107/SHA256SUMS).
Each archive contains a per-file inventory. SQLite exports are regenerated from the original reports;
weights, assets, caches, environments and connection credentials are excluded.

| Archive | Contents | SHA256 |
| --- | --- | --- |
| [Data and scripts](https://github.com/QLL-CCDD/RoboScope-Kit/releases/download/robotwin-native-mps-evidence-20261107/robotwin-baremetal-20261107-data.tar.gz) | Results, terminal records, errors, telemetry, host timers, analysis and reproduction scripts | `38b3fde4ab76b38e0357a6149dc6f8e471c79dd059e8e4410ddfab6ee4a8ca98` |
| [Static renderer reports](https://github.com/QLL-CCDD/RoboScope-Kit/releases/download/robotwin-native-mps-evidence-20261107/robotwin-baremetal-20261107-static-nsys.tar.gz) | Eight original hardware/CPU `.nsys-rep` reports | `803cbbcb2486a02ffcde0526cee4a9a1aeabea9d044ce084cb19840761ed57ea` |
| [Native evaluation reports](https://github.com/QLL-CCDD/RoboScope-Kit/releases/download/robotwin-native-mps-evidence-20261107/robotwin-baremetal-20261107-native-nsys.tar.gz) | Valid matched captures and separately named failed/recovered diagnostics | `006f7ad9b89bce27d45eaf9358f40568a43f36c50b07166fe29993c66ca3ea83` |
