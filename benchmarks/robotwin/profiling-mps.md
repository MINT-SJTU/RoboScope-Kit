# RoboTwin profiling repair and CUDA MPS measurements

Measured 2026-11-06; [Issue #79](https://github.com/MINT-SJTU/RoboScope-Kit/issues/79), related performance work [#33](https://github.com/MINT-SJTU/RoboScope-Kit/issues/33).

**Use MPS off for the measured single-GPU K8/G1 setup.** Two 500-episode runs per arm change pooled throughput by **−0.0293%** with identical saved terminal results. Profiling is repaired so the same CLI can expose Worker, Policy and queue activity. Hardware counters were permission-blocked in this container; the [native RTX 4090 follow-up](profiling-native-mps.md) measures RT/SM/Tensor/VRAM throughput separately.

## Change and experiment boundaries

RoboTwin's independent socket client had no capture trigger; Policy processes were outside the Worker capture. Nsight also separates the real server's process group from its launcher. The repair adds equivalent bridge markers, Policy queue/service/inference capture, real socket-peer PID identification, and normal-exit trace flushing before MPS cleanup. A single optional device-wide counter collector records actual sample availability. These paths use shared SDK runtime code plus the RoboTwin bridge; no Policy algorithm or default autotune algorithm is changed.

| Stage | Measured runtime | Workload / repetitions | Purpose |
| --- | --- | --- | --- |
| Main MPS baseline | b25, `5f79ce2` | 50 tasks × 10 episodes; two runs per arm | Unprofiled throughput, resources and terminal correctness |
| Built-in profiling repair | b26 local integration, final source `604ab77` | 10 representative tasks × 2 episodes; one run per condition | Worker/Policy capture and profiling overhead |
| Fixed K/G and capacity follow-ups | Same b26/`604ab77` | 50× 2 or 10× 2; one run per cell | Separate replica count, MPS and capacity effects |
| Complete external GPU diagnostics | Same b26/`604ab77`, Nsight 2026.6.1 | `adjust_bottle`× 10; K1/K4×MPS off/on | Correlated CUDA/Vulkan work, host latency and queueing |
| Current submission | b31 on main `44fa1ad` | Existing checks and real CLI QuickTest | Preserve upstream MessagePack and LIBERO-Pro while integrating repairs |

Historical local counter candidates labeled b27 predate the published MessagePack b27; they are not the released b27. Performance results retain their measured versions. The current submission is b31.

Hardware: one RTX 4090 reporting **49,140 MiB**, driver 580.105.08, compute mode `Default`; cgroup 20 CPU cores / 96 GiB RAM, CPU affinity 0–19. Policy: Python 3.11, Torch 2.6.0+cu124/runtime 12.4, FlashAttention 2.7.4.post1; simulator: Python 3.8, SAPIEN 3.0b1, Torch 2.4.1+cu121/runtime 12.1; nvcc 12.1. Driver CUDA 13.0 is the advertised interface ceiling.

StarVLA source `7e4a47360093d79dc42b883885c04fbe790c5b60`; checkpoint `StarVLA/Qwen3-VL-OFT-Robotwin2@e9cdfc38a0db968bf3a865bcb542d3d455952226`, weight SHA256 `8ccee8ddb33fa5df83a9a5e2b7d4f737db93f19a051b23f738af8a484fe2d7e5`. Suite `demo_clean`, bf16, stride 16, video off, two-episode work items except one-episode clean traces. The native-input snapshot omits the demo's state input to match the pinned native client; that Policy recipe change is not counted as runtime speedup.

Frozen generation `7b6622f7e8dc2272b53ea30bb5fef9acede91a6d5270b5487c10232a4250781d`; complete generation is checked before selecting research subsets. Official evaluation is 5000 episodes. Instruction limit 2 applies to 20/100-episode experiments and 10 to 500 episodes; language pools differ, so rates are compared only within a matched workload. **Formal** means Kit `evaluation_s`: excludes Policy loading/tuning/cleanup, includes Worker/scene startup. **E2E** is command lifetime. Device means use approximately 1 Hz sampling; peaks are sampled peaks, CPU is container cgroup core use.

## Repeated 500-episode MPS baseline

K8/G1, b25, two runs per arm. All four runs have identical 500-record seed/success/terminal-step maps.

| Metric | MPS off | MPS on |
| --- | ---: | ---: |
| Formal mean ± sample SD, s (n=2) | 1814.891802 ±2.090578 | 1815.424001 ±0.330082 |
| Episodes/h / steps/s | 991.794661 /114.553385 | 991.503913 /114.519804 |
| E2E r1 with search / r2 fixed, s | 2456.031 /1829.141 | 2417.065 /1831.362 |
| Autotune r1, s | 636.999 | 585.339 |
| GPU busy mean % | 64.854645 | 65.549998 |
| VRAM mean / sampled peak, GiB | 40.290616 /45.422852 | 40.537904 /45.749023 |
| CPU mean / tree sampled peak, cores | 6.883973 /8.129288 | 6.896924 /8.109039 |
| Maximum recorded single-PID GPU peak, MiB | 10474 (Policy) | 10402 (Policy) |
| Successful / unsuccessful episodes per run | 234/266 | 234/266 |
| Environment steps per run | 207902 | 207902 |
| Formal errors / OOM / wall-timeouts | 0/0/0 | 0/0/0 |
| RT/SM/Tensor/DRAM throughput | unavailable | unavailable |

Pooled MPS throughput changes by **−0.029315%**; the two paired effects are +0.064978% and −0.123584%. R1 includes autotune; R2 fixes K8/G1. Per-PID GPU memory follows driver attribution, including MPS server accounting; C/G duplicate rows are coalesced rather than summed. The MPS-on R1 restores a Policy before formal evaluation in 13.566 s; daemon/server PIDs remain unchanged.

The published four-GPU reference reports 231/500 successes and 211220 steps, versus 234/500 and 207902 here. Historical per-episode prompts/configuration are insufficient to resolve that difference; matching the current MPS arms does not reproduce the published correctness baseline.

## Built-in profiling repair

K4/G2, b26 integration, 20 episodes per condition, one run each. All conditions have 8 successes, 7415 steps and identical 20-record terminal maps; no runtime error, OOM or wall-timeout.

| MPS | Profile | Formal / E2E, s | Episodes/h /steps/s | GPU busy, % | VRAM mean / peak, GiB | CPU cores |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Off | Off | 117.685/193.091 | 611.805/63.007 | 30.475 | 32.301/34.371 | 3.829 |
| Off | On | 124.742/185.350 | 577.190/59.443 | 33.714 | 32.548/34.443 | 3.973 |
| On | Off | 116.301/143.037 | 619.085/63.757 | 28.238 | 32.526/34.403 | 3.846 |
| On | On | 126.336/186.945 | 569.907/58.692 | 28.222 | 32.562/34.474 | 3.923 |

Profiling adds **6.00% / 8.63%** formal time off/on in these single pairs. Cold-load differences explain why E2E does not follow that ordering. Each profiled run exports four Worker and two Policy reports. Policy lifetime totals are 1551921 kernels, 80725 copies and 87181 memsets, including loading/exit. Correcting shutdown order recovered 10959 previously dropped MPS trace events; the final Nsight 2023 captures have no incomplete-CUPTI diagnostic.

## Fixed K/G grid: 100 episodes per cell

b26, one execution per cell. The six healthy cells each return 49 successes and 39225 steps, with complete matched terminal records and no runtime faults.

| Requested K/G | MPS | Formal / E2E, s | Episodes/h /steps/s | GPU busy, % | VRAM mean / peak, GiB | CPU cores | Successes / steps |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4/1 | off | 554.909/569.309 | 648.755/70.687 | 36.503 | 25.559/27.772 | 3.929 | 49/39225 |
| 4/1 | on | 548.957/563.540 | 655.789/71.454 | 36.958 | 25.550/27.552 | 3.948 | 49/39225 |
| 4/2 | off | 557.790/585.227 | 645.404/70.322 | 37.506 | 35.653/37.842 | 3.969 | 49/39225 |
| 4/2 | on | 546.038/573.472 | 659.295/71.836 | 39.532 | 35.630/37.875 | 3.993 | 49/39225 |
| 8/1 | off | 352.698/367.803 | 1020.703/111.214 | 61.782 | 37.866/40.955 | 6.615 | 49/39225 |
| 8/1 | on | 357.506/373.422 | 1006.976/109.718 | 62.468 | 37.936/41.205 | 6.601 | 49/39225 |
| 8/2→K7 | off | 378.669/406.512 | N/A(recovered) | 62.256 | 44.716/47.361 | 6.318 | 49/39241 |
| 8/2→K7 | on | 378.076/407.779 | N/A(recovered) | 56.982 | 44.481/47.359 | 6.299 | 49/39169 |

At K4, G2/G1 changes throughput by −0.5165% off / +0.5347% on while adding 10.094 / 10.080 GiB mean VRAM. Fixed-K/G MPS effects are +1.0842% at K4/G1, +2.1523% at K4/G2 and −1.3448% at K8/G1; each is one pair.

Both requested K8/G2 runs fail camera buffer allocation and recover K8→7: Worker6 restarts, Worker7 retires and two items are requeued. Peaks are 48498 / 48496 MiB. All 100 IDs finish, but `dump_bin_bigbin` episode0001, seed1006010, succeeds after 353 / 281 steps versus 337 in healthy G1 runs. The recovered runs retain actual times but have no valid fixed-K throughput comparison.

## Short K2/K6 exploration: 20 episodes per cell

b26, 10 representative tasks× 2 episodes, one run per cell. All ten runs have 8 successes and 7415 steps, with 17 matched comparisons and no runtime error/OOM/timeout/recovery. Paired entries below are **off/on**.

| K/G | Formal s, off/on | E2E s, off/on | Episodes/h off/on | Steps/s off/on | VRAM mean→peak GiB, off; on | CPUcores off/on | MPS throughput change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2/1 | 208.525/206.273 | 222.896/220.378 | 345.282/349.051 | 35.559/35.947 | 17.091→17.895;17.124→17.930 | 2.086/2.064 | +1.091% |
| 2/2 | 207.849/207.299 | 234.336/234.247 | 346.405/347.324 | 35.675/35.770 | 27.116→27.972;27.153→28.004 | 2.076/2.063 | +0.265% |
| 6/1 | 123.224/123.696 | 137.739/139.009 | 584.303/582.073 | 60.175/59.945 | 27.869→31.011;27.984→31.042 | 4.005/4.007 | -0.382% |
| 6/2 | 119.464/121.115 | 146.652/148.914 | 602.690/594.476 | 62.069/61.223 | 37.831→40.984;37.804→41.019 | 4.154/4.074 | -1.363% |
| 8/1 | 124.640/126.411 | 139.770/141.568 | 577.664/569.570 | 59.491/58.658 | 33.276→37.728;33.159→37.765 | 4.214/4.183 | -1.401% |

At K2, G2/G1 changes formal throughput by +0.325% / −0.495%; at K6, +3.147% / +2.131%. K6/G2 off is fastest formally (119.464 s), whereas K6/G1 off is fastest E2E (137.739 s): the second model raises Policy loading from 12.427 to 24.539 s. MPS-on runs observe one actual Policy client at G1 and two at G2.

## Confirmation and integer capacity frontier: 100 episodes, MPS off

b26, one execution per point. All three preserve requested K, return 49 successes /39225 steps and match each other's100 terminal tuples and the historical b26 K8/G1 reference, without runtime faults.

| K/G | Formal / E2E, s | Episodes/h /steps/s | GPU busy, % | VRAM mean / peak, GiB | CPU cores | Power mean / peak, W | Final queue tail, s / % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6/1 | 414.535/429.256 | 868.443/94.624 | 52.544 | 31.784/34.359 | 5.320 | 198.620/247.50 | 27.764/6.698% |
| 6/2 | 414.321/441.835 | 868.892/94.673 | 51.878 | 41.840/44.374 | 5.453 | 197.598/267.29 | 30.052/7.253% |
| 7/2 | 374.934/402.639 | 960.169/104.618 | 57.678 | 44.671/47.369 | 6.207 | 211.176/266.93 | 21.779/5.809% |

K6/G2 gives only **+0.051623%** formal throughput, **−2.846998%** E2E throughput and **+10.056 GiB** mean VRAM versus G1. Its short-run gain does not persist. K7/G2 retains K7 with peak48506 MiB, 634 MiB below device capacity. Its throughput is5.930564% below the historical K8/G1 reference; the95% continuation gate was not met, so no new K8 anchor or K7-on run follows.

The final single-item queue-publication tail is30.22% of short K6/G2 time versus 7.25% at 100 episodes. Short K8/G1 and K6/G2 tail lengths differ by 4.448 s of their5.176 s formal difference. These are intervals between the final two result-file publication mtimes, not GPU execution time or exclusive task durations.

## Additional device telemetry

Formal means from the short matrix; power cells are mean/peak W. Off/on ordering is unchanged. The 100-episode power/CPU data appear above.

| Short K/G | GPU busy, % | Power mean/peak W, off; on | SM clock MHz, off/on | Mean temperature °C, off/on | Memory activity, % | PCIe RX/TX KB/s, off; on |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2/1 | 16.08/16.62 | 115.8/152.85;116.4/154.48 | 2700.84/2714.47 | 46.06/45.03 | 3.56/3.65 | 29831/20082;40521/19742 |
| 2/2 | 17.25/17.69 | 116.0/152.33;116.2/157.31 | 2703.44/2699.18 | 46.06/45.28 | 3.93/3.68 | 66155/27159;45564/20114 |
| 6/1 | 32.72/32.83 | 151.5/233.65;149.4/237.27 | 2735.94/2729.78 | 49.38/48.56 | 7.41/6.41 | 160590/44851;165404/57156 |
| 6/2 | 35.60/29.58 | 153.4/236.43;150.4/242.55 | 2732.82/2734.27 | 48.78/48.27 | 7.64/6.82 | 111877/41262;57145/40283 |
| 8/1 | 35.14/31.56 | 148.0/255.51;148.3/251.26 | 2731.38/2735.51 | 46.53/48.34 | 6.84/5.87 | 64895/35433;108580/43807 |

Memory clocks remain10501 MHz; short-run SM means are2699–2736 MHz and temperatures peak at 60°C. `sw_power_cap` is sampled active for0.965% of K2/G2-on and1.652% of K6/G2-on time; no active reason is sampled in the100-episode confirmations. Memory activity is read/write-active time, PCIe KB/s is bus traffic, and clocks are frequencies; none is RT/SM throughput or DRAM GB/s.

## Policy, rendering and request waiting

The b25 timing-only runs each use20 episodes,469 Policy acts and7415 control actions. Values below are mean host milliseconds unless labeled; spans are inclusive and can overlap.

| K/G, MPS | Policy act | Lock wait | Request mean / median / p95 | Observation / render / readback | Control action |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4/2 off | 78.066 | 7.214 | 93.700/75.417/165.764 | 138.549 | 43.245 |
| 4/2 on | 72.152 | 4.546 | 85.260/71.766/142.486 | 140.582 | 43.624 |
| 8/1 off | 88.408 | 80.190 | 177.931/142.021/407.470 | 231.991 | 44.099 |
| 8/1 on | 87.698 | 78.209 | 175.332/153.150/369.981 | 233.327 | 43.915 |

The separate b26 clean captures use `adjust_bottle`× 10, shard1, four ray-traced cameras,32 SPP/depth8 and OIDN. Every capture has10 successes,1087 actions,72 Policy acts and82 observations, with identical terminal results.

| K/G, MPS | Formal s | SIM GPU union, s | Policy CUDA union, s | All captured GPU union, s | Observation median host / Vulkan GPU, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1/1 off | 97.802654 | 3.776467 | 2.245897 | 6.022364 | 54.375/42.045 |
| 1/1 on | 100.377496 | 3.765752 | 2.252511 | 6.018263 | 54.845/41.761 |
| 4/1 off | 41.517755 | 4.020196 | 2.224240 | 6.100894 | 60.927/43.834 |
| 4/1 on | 40.893630 | 3.984432 | 2.215165 | 6.111034 | 61.008/45.580 |

Each clean trace has72×3309 Policy kernels and complete observed CPU/GPU Vulkan correlations. SIM includes Vulkan and0.15–0.16 s of CUDA OIDN/CUTLASS. Timestamp unions overlap and are not hardware utilization. The labels remain `vkQueueSubmit`, without isolated TraceRays duration.

K1/off control calls average48.119 ms/action; Policy host act averages79.694 ms/chunk versus 30.469 ms/chunk of CUDA kernel union. With15.10 actions/chunk, control calls total about 726.5 ms/chunk. K4/off request mean142.704 ms comprises95.020 ms service,39.223 ms lock wait and8.461 ms other serialization/transport/scheduling. This supports a mixed control/observation/queue bottleneck, not a demonstrated RT saturation claim. Old469-act manual traces have a partly missing final act and remain lower bounds.


## Current submission validation: b31

Rebased onto main `44fa1ad`, preserving MessagePack, LIBERO-Pro and the existing protocol. All **187 existing pytest tests**, Ruff and the Python support-policy check pass. The real QuickTest uses the current SDK in both controller and Policy process; their 80 source-file hashes match. The prior officially built simulator bridge is byte-identical to this source.

| Current b31 CLI check | Result |
| --- | --- |
| `--self-test` | 50 task runtime paths verified |
| `--quick-test --profile --cuda-mps --profile-gpu-metrics ad10x-gfxt` | 1/1 success, 87 steps; one actual MPS client |
| Worker capture | One episode, 87 action spans, seven observations; expected labels present |
| Policy capture | Six act spans, each with 3309 CUDA kernels; expected labels present |
| Device counters | `unavailable`; actual permission denial, no counter report/SQLite |
| Nsight diagnostic | Actual Policy process reports possible missing NVTX events; per-act CUDA coverage verified |
| Cleanup | GPU 0 MiB / 0%; no owned evaluation/MPS process or Nsight session |

This QuickTest uses RoboTwin's JSON bridge; MessagePack compatibility is covered by existing software tests. The NVTX warning is retained alongside successful SDK-label and per-act kernel checks.

## Autotune decision and Issue #79 status

Keep the default search and budget. Diagonal accepts K8/G1, then K16/G1 OOM stops the G phase (`g_trials=[]`); it does not exhaust fixed-K alternatives. The added K2/K6/K7 points are valid configurations, but the broader K6 confirmation gives only+0.0516% throughput for+10.056 GiB mean VRAM. K7/G2 misses the historical K8/G1 95% continuation gate; no new K8 anchor or K7-on run was triggered. Existing fixed-point controls and opt-in strategies cover targeted follow-up.

The extra ten short cells cost 1765.509 s E2E, K6 confirmation871.090 s and K7 check402.639 s: **3039.239 s (50m39 s)** total. This includes independent model loads, not the marginal cost of a production autotune probe, which reuses its pool. No measured broad-workload win justifies expanding default exploration.

| Issue requirement | Result |
| --- | --- |
| Single-GPU attachment, restart and shutdown | Verified actual Policy MPS clients, replica replacement/reattachment and owned cleanup |
| Fixed K/G MPS comparison | Complete on one GPU; valid pairs and recovered OOM attempts reported above |
| Paired autotune | Both diagonal/steady select K8/G1; costs636.999 s off/585.339 s on; both K16/G1 OOM |
| Four-GPU lifecycle and500-episode repeats | Open: only one GPU allocated; published K16/G4 was not reproduced |
| Timing boundary | Measured Kit phase includes Worker/scene startup; a fully initialization-excluded window is not separately measured |
| Recommendation and profiling | Findings recorded here; retain MPS off for this host/configuration |
| RT/SM/Tensor/DRAM throughput | Container probes unavailable; the [native follow-up](profiling-native-mps.md) obtains valid hardware samples and matched fixed-K controls on a 24-GB RTX 4090 |

Main K8/G1 has one Policy MPS client; its server serializes requests. The private MPS session does not attach the simulator's Vulkan renderer or CUDA OIDN denoiser. Rendering is material inside an observation, but control/chunk cadence, initialization and queueing also contribute. Hardware unit saturation has not been established. The four clean traces have complete observed act/kernel and CPU/GPU Vulkan correlations; earlier tail-truncated traces remain separate diagnostics.

## Reproduction and evidence

For the submitted runtime, install the package, update the bridge through the supported build entry point, then run self-test and profiling with the same Policy configuration:

```bash
# Install this candidate in both the controller and the user Policy environment.
python -m pip install ./packages/roboscope-kit
bash benchmarks/robotwin/build-runtime.sh /workspace/sim/robotwin
export PATH=/workspace/tools/nsys-2026.6.1/opt/nvidia/nsight-systems-cli/2026.6.1/target-linux-x64:$PATH
python -m roboscope_kit.runtime.benchmark --benchmark robotwin --suite demo_clean \
  --benchmark-manifest /workspace/sim/robotwin/benchmark-manifest.json \
  --checkpoint-source /workspace/models/starvla-robotwin-native \
  --policy-config /workspace/robotwin-prep/roboscope-policy.json \
  --output-dir /workspace/reproduce-self-test --self-test
# Use the same arguments with a new output directory and these flags:
# --quick-test --profile --cuda-mps --profile-gpu-metrics ad10x-gfxt
```

For the historical research subsets, use the pinned runtime and external runners in the attachments. Change only K/G/MPS within each pair; choose an empty output directory:

```bash
tar -xzf raw-results-search.tar.gz -C /workspace
export ROBOSCOPE_CAMPAIGN_DIR=/workspace/reproduce-short
export ROBOSCOPE_SEARCH_TASK_COUNT=10
export ROBOSCOPE_SEARCH_POINTS='6:2:off 6:2:on'
bash /workspace/robotwin-mps-search-20261106/run-campaign.sh
# Task-count50 gives100episodes; the primary500episode runner uses subset-episodes10.
python /workspace/robotwin-mps-search-20261106/summarize_search.py \
  /workspace/reproduce-short --expected 20
```

[Download the evidence release](https://github.com/QLL-CCDD/RoboScope-Kit/releases/tag/robotwin-mps-evidence-20261106) and its [SHA256 manifest](https://github.com/QLL-CCDD/RoboScope-Kit/releases/download/robotwin-mps-evidence-20261106/SHA256SUMS). These attachments are outside Git. Each packet contains actual commands, configurations, per-episode records and logs; the trace packets also contain SQLite exports. Weights/assets are excluded.

| Packet | Data | SHA256 |
| --- | --- | --- |
| [profiling-integration-raw-results.tar.gz](https://github.com/QLL-CCDD/RoboScope-Kit/releases/download/robotwin-mps-evidence-20261106/profiling-integration-raw-results.tar.gz) | Profiling repair, paired runs and report-content audits | `6b926cc3161251b1d38a4188c7a2682a85130e7f219231969a0713344a8541f6` |
| [mps-primary-raw-results.tar.gz](https://github.com/QLL-CCDD/RoboScope-Kit/releases/download/robotwin-mps-evidence-20261106/mps-primary-raw-results.tar.gz) | b25 screening, four500-episode runs and early diagnostics | `4adcc09ea84c2250027eca5a503d477d2bd055d0a7110533c2e843a525ec88fb` |
| [raw-results-followup.tar.gz](https://github.com/QLL-CCDD/RoboScope-Kit/releases/download/robotwin-mps-evidence-20261106/raw-results-followup.tar.gz) | b26 grid, OOM recovery, four clean CUDA/Vulkan captures | `f70447666446cde8b644e172d526c7be43d966a0df0e835e4f522994148e5c81` |
| [raw-results-search.tar.gz](https://github.com/QLL-CCDD/RoboScope-Kit/releases/download/robotwin-mps-evidence-20261106/raw-results-search.tar.gz) | Ten20episode cells, K6/K7 hundred-episode checks, extra telemetry | `d6b351a706765d8c6130cbb8827f021fb309927fff5e6264537673c5ae1ed458` |
| [validation-evidence.tar.gz](https://github.com/QLL-CCDD/RoboScope-Kit/releases/download/robotwin-mps-evidence-20261106/validation-evidence.tar.gz) | Current b31 QuickTest, reports, source identity, PID audit and counter failure | `3f8047a353d18c448b2ffaa51a58161e7c2ed1d935379d5b51ddf7c71254d2da` |
