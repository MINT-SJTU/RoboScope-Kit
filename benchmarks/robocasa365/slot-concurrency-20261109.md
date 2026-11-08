# RoboCasa365 slot concurrency pilot, 2026-11-09

The adapter can overlap work within its existing five-slot simulation group.
Each slot owns a spawned simulator with its original constructor RNG stream.
The Worker collects all slots at the existing action-chunk boundary and sends
one ordered Policy batch. This implementation shares RoboCasa365 native slot
state and success logic; it does not introduce a generic public N-slot API.
Autotune chooses Workers/Policy replicas, while this option removes serialization
inside each Worker. CUDA MPS manages CUDA clients. These act at different layers;
the comparison enables MPS on both sides.

Baseline source `04cdd858a548aba13ef63a21f06f6301a58a0a2d` (b34) and parallel source
`b529000ff70ff9591436d54fcc0191ded34aa0f8` (b35) ran sequentially on one RTX 4090
with 49,140 MiB of VRAM, driver 580.105.08, an Intel Xeon Gold 6459C,
a 12-core CPU quota and a 96 GiB container limit. All arms used CUDA MPS,
the official human300 Diffusion Policy in FP32 with 100 denoising steps,
the same runtime and seeds 1111111–1111115, full RGB and videos off.
The diagnostic caps six tasks at 128 steps, with five episodes each.
A fixed K=4/G=2 comparison isolates the slot setting. Separate diagonal
searches use the same K≤4/G≤2 bounds, elapsed estimator, 500 s candidate timeout
and 1,500 s total budget. Both searches selected K=4/G=2.
The harness records probe and formal tasks and preserves the normal Kit
policy, queue, evaluator and result aggregation. Each autotune arm is one run;
the fixed pair is repeated below. The cap belongs only to this diagnostic
harness; the committed official task horizons remain unchanged.

## Speed and resources

| Arm | K/G | Evaluation (s) | Autotune (s) | Whole command (s) | Episodes/h | Steps/s | Successes | Steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-serial | 4/2 | 175.10 | 0.01 | 199.62 | 616.8 | 21.93 | 3/30 | 3,840 |
| fixed-parallel | 4/2 | 92.26 | 0.05 | 117.24 | 1170.6 | 41.62 | 3/30 | 3,840 |
| tuned-serial | 4/2 | 173.50 | 528.73 | 706.45 | 622.5 | 22.13 | 3/30 | 3,840 |
| tuned-parallel | 4/2 | 83.62 | 228.41 | 317.68 | 1291.5 | 45.92 | 3/30 | 3,840 |

Formal time is the Kit `evaluation_s` stage, including Worker environment
creation and reset. End-to-end time also includes initialization, Policy
setup, any search and cleanup. The first fixed-pair formal throughput ratio
is **1.898×** and the independently tuned ratio is
**2.075×**. Their end-to-end ratios are
1.703× and 2.224×, respectively.
The ≥2× issue target is for the official full workload. This six-task cap
does not establish that target. RGB deferral is a separate experiment.

### Repeated fixed K=4/G=2 measurements

| Trial | Serial evaluation (s) | Parallel evaluation (s) | Ratio | Serial / parallel episodes/h | Serial / parallel steps/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 175.10 | 92.26 | 1.898× | 616.8 / 1170.6 | 21.93 / 41.62 |
| 2 | 173.18 | 82.73 | 2.093× | 623.6 / 1305.4 | 22.17 / 46.42 |
| 3 | 172.08 | 88.12 | 1.953× | 627.6 / 1225.6 | 22.32 / 43.58 |

Serial median **173.18 s** (range 172.08–175.10); parallel median **88.12 s** (range 82.73–92.26). The principal fixed-K/G
result is the ratio of those medians: **1.965×**.
Individual paired ratios range from 1.898× to 2.093×.
Every arm completed the same 30 episodes and 3,840 physical steps.

| Arm | GPU busy mean / P95 / peak | VRAM mean / peak (MiB) | Memory busy mean | Power mean (W) | CPU cores mean | MPS clients |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-serial | 17.15% / 55% / 65% | 16,595 / 24,831 | 2.02% | 84.2 | 3.15 | 2 |
| fixed-parallel | 33.23% / 78% / 100% | 21,907 / 39,429 | 4.11% | 106.8 | 7.04 | 2 |
| tuned-serial | 18.67% / 56% / 65% | 16,759 / 24,853 | 2.20% | 88.0 | 3.15 | 2 |
| tuned-parallel | 33.13% / 80% / 89% | 21,766 / 39,451 | 4.29% | 110.3 | 7.44 | 2 |

GPU samples are device-wide, every 250 ms during formal evaluation.
They include both rendering and inference activity. Raw samples also retain
SM/memory clocks and temperature. CPU values are cgroup-accounted mean cores.
Peak GPU busy rate alone cannot establish that the evaluation is GPU-bound.

Host memory was sampled separately at one-second intervals for the final
fixed repetition. These are cgroup values for the container, including
file cache and other container processes; they are not process RSS and
cannot retrospectively describe the four initial arms.

| Fixed trial 3 | Samples | Anonymous mean / peak (GiB) | Cgroup current mean / peak (GiB) | File cache mean / peak (GiB) |
| --- | ---: | ---: | ---: | ---: |
| fixed-serial | 168 | 38.03 / 43.63 | 47.31 / 52.94 | 8.46 / 8.48 |
| fixed-parallel | 84 | 52.22 / 84.37 | 61.60 / 93.83 | 8.50 / 8.52 |

The parallel arm reached 93.83 GiB cgroup current memory under the 96 GiB
container limit. It completed without an OOM, but this measured memory cost
keeps the mode opt-in; worker limits require RAM headroom on the target host.

### Timing boundaries and recurring work

| Arm | Checkpoint preparation (s) | Runtime preparation (s) | Policy load stage (s) | Autotune (s) | Formal evaluation (s) | Other command time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-serial | 0.018 | 0.837 | 20.193 | 0.008 | 175.102 | 3.464 |
| fixed-parallel | 0.063 | 0.883 | 20.887 | 0.051 | 92.258 | 3.101 |
| tuned-serial | 0.030 | 0.854 | 0.021 | 528.726 | 173.499 | 3.320 |
| tuned-parallel | 0.012 | 1.355 | 0.099 | 228.408 | 83.624 | 4.178 |

Checkpoint preparation resolves the already-downloaded real checkpoint;
it excludes earlier asset/model downloads and the runtime build. Autotune
includes Policy startup and probe work and retains usable replicas. The
later `policy_load_s` stage in a tuned run therefore reuses loaded replicas.
Other command time includes interpreter/CLI work and finalization; result
serialization and cleanup are not independently isolated by these stages.
Videos are disabled in every arm and Worker video-write time is zero.

| Arm | Worker creation (s) | Reset (s) | Native steps/barriers (s) | Policy RPC (s) | Cleanup (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed-serial | 146.41 | 164.41 | 146.94 | 80.32 | 2.94 |
| fixed-parallel | 74.73 | 51.12 | 40.16 | 110.82 | 7.23 |
| tuned-serial | 148.04 | 162.78 | 147.10 | 77.87 | 2.99 |
| tuned-parallel | 69.46 | 50.44 | 38.81 | 94.22 | 6.88 |

Worker intervals overlap and must not be added to obtain wall time.
Serial step time sums simulator calls. Parallel step time includes the
chunk barrier and returning slot data over IPC. In the first fixed pair,
that recurring interval fell from 146.94 to 40.16 s (3.66×). Policy RPC wait rose
from 80.32 to 110.82 s with faster
simulator supply and more concurrency. RPC includes transport and queue wait;
it does not isolate Policy GPU execution. Construction and reset also overlap, but
the recurring step improvement persists beyond startup. Five simulators
give an ideal 5× bound for that serialized component alone. Holding RPC
time constant gives about 2.07× for the measured step-plus-RPC
interval sums under ideal five-way stepping. This is a component estimate,
not an upper bound or prediction for full-evaluation wall time: contention,
queues and overlapping Workers matter.

## Correctness and failure checks

| Pair | Paired records | Success differences | Step differences | Termination differences | Initial fingerprint differences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed K/G, trial 1 | 30 | 0 | 0 | 0 | 0 |
| Independent autotune | 30 | 0 | 0 | 0 | 0 |
| Fixed K/G, trial 2 | 30 | 0 | 0 | 0 | 0 |
| Fixed K/G, trial 3 | 30 | 0 | 0 | 0 | 0 |

The comparator checks complete, unique task and slot coverage, actual slot
execution, initial state, scenario and raw initial observation hashes, success,
step counts and termination. The timed runs do not record full trajectories.

| Task | Fixed serial successes by slot | Fixed parallel | Tuned serial | Tuned parallel | Steps per slot, all arms |
| --- | --- | --- | --- | --- | --- |
| `TurnOnElectricKettle` | `00100` | `00100` | `00100` | `00100` | 128/128/128/128/128 |
| `CloseToasterOvenDoor` | `00000` | `00000` | `00000` | `00000` | 128/128/128/128/128 |
| `TurnOnMicrowave` | `00000` | `00000` | `00000` | `00000` | 128/128/128/128/128 |
| `NavigateKitchen` | `00000` | `00000` | `00000` | `00000` | 128/128/128/128/128 |
| `OpenStandMixerHead` | `00011` | `00011` | `00011` | `00011` | 128/128/128/128/128 |
| `SlideDishwasherRack` | `00000` | `00000` | `00000` | `00000` | 128/128/128/128/128 |

The five binary digits follow constructor-seed slot order, with 1 meaning
success. Every arm scored 3/30; all three fixed repetitions had these same
per-slot results. Repeated records are retained per task and slot as well.
These short prefixes do not establish
native-horizon success-rate equivalence; the separate native comparison
also documents observation and trajectory variation.

| Arm | Worker failures | Worker recycles | OOM |
| --- | ---: | ---: | ---: |
| fixed-serial | 0 | 0 | 0 |
| fixed-parallel | 0 | 0 | 0 |
| tuned-serial | 0 | 0 | 0 |
| tuned-parallel | 0 | 0 | 0 |
| trial-2-fixed-serial | 0 | 0 | 0 |
| trial-2-fixed-parallel | 0 | 0 | 0 |
| trial-3-fixed-serial | 0 | 0 | 0 |
| trial-3-fixed-parallel | 0 | 0 | 0 |

The real capped five-slot smoke passed. A Linux parent-death check confirmed
that the child exited after its parent was forcibly killed. Both reviewed
source trees passed 298 pytest checks and Ruff. The fresh pinned runtime
passed its double-reset/render self-test. Standard serial and parallel Quick
Tests each completed five CloseBlenderLid episodes at the official 900-step
horizon (4,500 steps, 0/5 successes), without execution errors.
The full 2,500-episode A/B, repeated autotune and full-horizon multi-task
success-rate comparison remain open.

Earlier harness attempts are retained separately in the archive. Initial
serial timing attempts imported the controller environment's NumPy through
an overly broad `PYTHONPATH` and failed; the corrected driver uses SDK-only
overlays. An initial smoke resolved the simulator Python symlink and thereby
lost its virtual environment; the driver now preserves that interpreter path.
Those failed attempts are not included in these completed-run speed ratios.

## Reproduction

Extract `entry.py`, `run.py`, `repeat_fixed.py` and `summarize.py` from the evidence archive.
Point the driver at the two frozen checkouts, SDK overlays, installed Policy,
and checksum-verified simulator runtime on the same Linux GPU host. The archive
source proof maps the measured executable files to the reviewed branches;
later documentation commits do not replace the recorded measurement IDs.

```bash
W=/root/autodl-tmp/robocasa365-20261107
S=/autodl-fs/data/robocasa365/acceptance-20261107
C="$W/control-reviewed-native/bin/python"
OUT="$S/reviewed-slots-fixed-autotune128-repeat"
"$C" -B "$W/pr100-slot-study/run.py" --root "$OUT" \
  --baseline "$W/reviewed-native-src" \
  --baseline-overlay "$W/reviewed-native-overlay" \
  --baseline-commit 04cdd858a548aba13ef63a21f06f6301a58a0a2d \
  --candidate "$W/reviewed-slots-src" --candidate-overlay "$W/reviewed-slots-overlay" \
  --candidate-commit b529000ff70ff9591436d54fcc0191ded34aa0f8 --runtime "$W/runtime-reviewed-native" --cap 128
"$C" -B "$W/pr100-slot-study/summarize.py" "$OUT"
```

[Raw results, GPU samples and harness](https://github.com/QLL-CCDD/RoboScope-Kit/releases/tag/robocasa365-review-evidence-20261109) · [Native comparison](native-comparison-20261109.md) · [Issue #97](https://github.com/MINT-SJTU/RoboScope-Kit/issues/97)
