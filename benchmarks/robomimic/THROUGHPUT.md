# robomimic throughput, 2026-11-08

This report addresses issue #32 on top of the #26 integration (merged as #84). It reproduces the baseline
on the measurement machine and distinguishes configuration gains, code gains, and native correctness.
The runtime/Policy changes were measured as candidate v7; the delivery version is 0.8.0b43.
The [evidence index](EVIDENCE.md) links the raw archive, checksums, source snapshots, commands,
native comparisons, build logs, resource samples, and rejected experiments.

## Frozen workload and hardware

- Base Kit commit: `9174d1b6fb557c0285deb4b3d7d28cc66e2f4ba4`.
- One RTX 4090 reporting 49,140 MiB, driver 595.71.05; Ubuntu 22.04.3; container quota of 20 CPU cores and
  96 GiB RAM. The host's 208 logical CPUs are not the available CPU budget. No extra GPU was added.
- robomimic `7c66e7a41b5d9dcc905b1a68346bfee1b49b79c9`, robosuite
  `c848ca848020d0c4ccdd10c5056bd06f2a195ba2`, MuJoCo 2.3.7, torch 2.4.1, NumPy 1.26.4,
  Python 3.11.15, uv 0.11.14. All five official checkpoints were SHA-256 verified.
- `ph_low_dim`: Lift, Can, Square, Transport, Tool Hang, 50 episodes each, seeds 0-49. Horizons remain
  400/400/400/700/700, control frequency 20 Hz, action chunk 1, FP32, no cameras/video. Model, weights,
  normalization, sampling, per-episode RNN/RNG state, observations, actions, termination and scoring are
  unchanged.
- Every accepted full run executes 250 episodes and 81,289 steps. Successes are 50/50, 50/50, 37/50,
  29/50 and 10/50 respectively: 176 successes, 74 task failures. Task failures are counted outcomes;
  none of these accepted runs has a Worker failure, retry or OOM.
- The original simulator lock missed MessagePack after the shared client began importing it. Both sides
  were rebuilt with `msgpack==1.2.4`; this dependency repair is not counted as a speed optimization.

## Timing and comparison

Formal time is `result.json.phase_timings.evaluation_s`: queue start through completion of all work items.
It includes first environment construction, reset, rollout, and per-item result writes. Worker process
readiness and model loading precede this interval. Throughput is 250 divided by formal time; steps/s is
81,289 divided by the same time. No trajectory reduction contributes to the reported gains.

The external harness measures process start through exit and samples whole-container CPU/memory and
GPU usage every second. Worker timing sums overlap and must not be added as serial wall-clock phases.
Both sides use new output directories and fresh Worker/Policy processes, one CPU thread per process,
the same local immutable assets/checkpoints, warmed download caches, and no autotune probes. The cache
starts empty in every new Worker; its fill cost is included. `diagonal` remains the default autotune
strategy, but this comparison explicitly uses fixed K/G. It does not measure an autotuning improvement.

### Repeated complete evaluations

| Run | K/G | MPS | Formal s | End-to-end s | Steps/s |
| --- | --- | --- | ---: | ---: | ---: |
| baseline-full-01 | 16/1 | off | 287.471738 | 292.709307 | 282.77 |
| baseline-full-02 | 16/1 | off | 341.152072 | 346.945508 | 238.28 |
| candidate-v7-k20g8-mps-01 | 20/8 | on | 142.950367 | 168.412230 | 568.65 |
| candidate-v7-k20g8-mps-02 | 20/8 | on | 138.663132 | 163.583030 | 586.23 |
| candidate-v7-k20g8-mps-03 | 20/8 | on | 137.007911 | 163.174295 | 593.32 |
| final-b43-k20g8-mps-01 (fresh-build acceptance) | 20/8 | on | 139.750959 | 163.977132 | 581.67 |

Baseline formal mean +/- sample SD is **314.312 +/- 37.958 s** (n=2); candidate is
**139.540 +/- 3.067 s** (n=3). The pooled formal throughput ratio is **2.252x**. The baseline varied
substantially, so a conservative comparison also uses its fastest run: 287.472 / 139.540 = **2.060x**;
even the slowest candidate against that baseline is **2.011x**. This is a small set of repeats on one
machine, not a claim that every future run or hardware configuration exceeds 2x.

Mean process end-to-end time is 319.827 s versus 165.057 s, **1.938x**, reported separately from formal
throughput. Mean candidate formal throughput is 1.792 episodes/s and 582.55 steps/s.
The final fresh-build acceptance completed the same 250 episodes in **139.751 s**, **2.057x**
against baseline-full-01, with no record mismatches. It is separate from the three-run v7 summary.

### Configuration and code ablations

| Change/configuration | Formal s | Interpretation |
| --- | ---: | --- |
| Original code, K16/G4, MPS off | 195.549650 | More Policy replicas remove much of the original queue wait |
| Original code, K20/G8, MPS on | 154.924153 | Configuration-only 1.856x versus the original K16/G1 run |
| no_grad only, K20/G8, MPS on | 153.960426 | Small single-run difference; not an independent reliable gain |
| no_grad + longer tasks first (v6), K20/G8, MPS on | 148.152542 | Preserved fallback, 1.940x versus original run |
| v6, K32/G8, MPS on | 158.594870 | Oversubscription did not improve throughput |
| Original code, K32/G8, MPS on | 167.609283 | Matched original-code K32 control |
| v6, K20/G8, MessagePack, MPS on | 148.693906 | No observed transport benefit for these small observations |
| v6, K24/G8, MessagePack, MPS on | 148.866381 | No observed gain over K20 |
| v6, K20/G8, MessagePack, MPS off | 155.218255 | Single matched MPS-off diagnostic |
| v7, K20/G8, MPS on, cache off | 149.043636 | Same candidate code/instrumentation, cache disabled |
| v7, K20/G8, MPS on, cache on, mean of three | 139.540470 | Cache-on mean versus cache-off single: 1.068x |

At matched K20/G8/MPS, original code versus the optimized mean is **1.110x**, not 2x. The >=2x headline
includes changing the documented initial fixed configuration; it must not be presented as a 2x
source-only gain at fixed K/G. Most gains come from parallel configuration. The matched ablation controls
have only one run each and should be read with the repeated-run spread above.

### Startup, preparation and resources

The first repaired baseline clean build took 173 s; the v7 clean build took 160 s, including cached
package/source installation, all initial-state checks, and self-test. Download preparation is a one-time
shared prerequisite: official files were staged through a hash-verified wheelhouse/source cache after
network failures; those failures and retries are retained. They are excluded from both formal timers.
The final delivery build took 446 s (22:26:45-22:34:11 +08:00), including initial-state checks
and self-test. Build duration is variable and is excluded from the throughput comparison. There is
no controlled cold-network-download comparison; raw preparation attempts are retained as provenance.

For v7's first full run, checkpoint preparation was 0.000157 s, the fixed-configuration selection path
0.000285 s, Policy loading 20.821592 s, and Worker readiness 1.619611 s. All first-use scene work remains
inside the 142.950367 s formal interval. The remainder of the 168.412230 s process time includes CLI
startup, shutdown and final output. On the final build, the corresponding times were:

| Phase | Baseline-full-01 s | Final-b43 s | Timing boundary |
| --- | ---: | ---: | --- |
| Checkpoint preparation | 0.000181 | 0.000165 | Locate staged, verified checkpoint files |
| Fixed configuration selection | 0.000084 | 0.000275 | No autotuning probes |
| Cold Policy/model loading | 2.519917 | 20.071678 | One versus eight new Policy replicas |
| Worker process readiness | 1.413246 | 1.420482 | Fresh Worker startup before queue start |
| Formal interval | 287.471738 | 139.750959 | First scene initialization through completed work items |
| Process start through exit | 292.709307 | 163.977132 | Includes final writing and shutdown |

Final-build Worker instrumentation recorded 135.706 s of initialization, 551.447 s of reset,
0.828 s of Policy reset, 2.229 s of observation preparation, 1,751.756 s of simulation,
283.250 s of Policy round trips, and 0.170 s of Policy end calls. These are **overlapping Worker sums**,
not additive wall-clock phases. The baseline had 1,717.823 s of Policy round trips; its existing timers
combine reset and other item setup, so no retrospective isolated reset number is inferred.
The retained five-Worker baseline cProfile diagnostic does separately identify seven environment
creations (14.191 cumulative s), ten episode resets (31.006 s), 3,273 observation preparations
(0.113 s), and 35 Worker queue/state JSON writes (0.013674 s). It covers ten diagnostic episodes and
overlapped the native diagnostic; these overlapping profile sums are not full-run phase totals and are
excluded from the performance comparison.

A separate real 250-episode diagnostic timed existing serialization/write/rename calls with an external
wrapper. Its 250 completion JSON writes totaled **0.071243 s**, both final result JSON writes
**0.003014 s**, 169 metrics snapshots **0.097672 s**, and other queue/state/progress JSON writes
**0.421975 s**. It matched every episode and is excluded from speedup statistics because it adds
instrumentation. Worker sums overlap; the writer does not fsync, so this measures application write
latency, not durable-media flush time. Raw measurements and the wrapper are in the archive.

The baseline's mean sampled CPU use was 9.40 cores, versus 16.69 for v7. Both had the same 20-core quota.
Peak measured GPU memory was 854 MiB versus 4,597 MiB. Whole-container memory includes caches and
background services; it is not the Policy's private RSS. Resource samples and cgroup memory events are
retained for every run. GPU busy percentages are not used as proof of a bottleneck.

## Changes and correctness

The queue starts Transport and Tool Hang before the shorter-horizon tasks and retains group affinity.
The Policy disables backward-graph recording around the unmodified upstream FP32 inference/sampling
calls. It still saves and restores each episode's torch RNG and recurrent state.

The builder applies a SHA-256-pinned source patch to exactly the pinned robosuite file, checks the output
hash and atomically replaces the file so uv's hard-link cache is not modified. The two-entry cache stores
pristine compiled models, keyed by exact XML and working directory, and returns a MuJoCo-owned copy for
each simulation. The assets and process working directory are fixed. Before XML serialization it restores
MuJoCo's last-compiled XML context when necessary; MuJoCo still serializes the live model's values.
It does not change any controller or simulation step. `ROBOSCOPE_ROBOMIMIC_MODEL_CACHE=0` disables reuse.

Every accepted run is compared by record ID, success, actual steps and complete episode metadata,
including initial-state hash. No IDs are missing, extra or duplicated. The independent upstream
`run_trained_agent.py` baseline comparison matched **250/250** independent seeds in success and steps;
all accepted optimized runs also match those baseline records. Five upstream continuous 50-rollout
batches were additionally run, but use a different RNG-stream protocol and are not paired with the
independent-seed workload. Their combined elapsed time is not a speedup denominator. No advantage over
an equivalently optimized native evaluator is claimed.
Per-step action/state traces were not retained, so these comparisons establish the stated initial-state,
terminal-outcome and executed-step agreement; they do not establish bitwise identity of every trajectory.

Rejected variants remain visible: a one-entry model cache took 296.560412 s at K16/G1; skipping
controller forward changed 35 episodes; forwardSkip reuse changed 38 episodes and produced 172 successes.
These simulator changes are absent from the delivered patch. A faster run with changed trajectories is
not an accepted optimization.

## Reproduce

Use distinct runtime and run directories. The checkpoint files are listed in the example's pinned
model-zoo table. Prepare a Policy JSON using the built runtime's Python and this source checkout.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0
export KIT_PYTHON=/path/to/kit-python
export PYTHONPATH="$PWD/packages/roboscope-kit/src"
export ROBOSCOPE_ROBOMIMIC_MODEL_CACHE=1
bash benchmarks/robomimic/build-runtime.sh /new/runtime
"$KIT_PYTHON" -m roboscope_kit.runtime.benchmark --benchmark robomimic --suite ph_low_dim \
  --checkpoint-source /path/to/checkpoints --benchmark-manifest /new/runtime/benchmark-manifest.json \
  --policy-config /path/to/policy.json --concurrency 20 --replicas 8 --cuda-mps \
  --output-dir /new/full-run
```

For the cache ablation, run `export ROBOSCOPE_ROBOMIMIC_MODEL_CACHE=0` in the same shell and repeat the
evaluation in a new output directory, retaining all other arguments and the same built runtime. The
switch must reach the evaluation process; an assignment scoped only to the build command does not
configure a subsequent evaluation. The builder always installs the same hash-verified patch.
The original-code comparison uses the base commit with only the MessagePack dependency repair; commands
and source inventories for both sides are stored with the evidence. Existing pytest, Ruff, Python support,
lockfile, bash syntax and ShellCheck results, the final fresh build, and final complete evaluation are
recorded in the delivery evidence index. No new tests were added.
Final delivery validation passed all 264 existing pytest cases, Ruff (including the Policy example),
Python support policy, uv lock consistency, bash syntax, ShellCheck, all 250 initial-state digests,
the builder's self-test, and the complete real-model acceptance run.

## Prior work informing the experiments

- [#37](https://github.com/MINT-SJTU/RoboScope-Kit/issues/37): complete workload and equivalent-resource
  acceptance, fixed K/G, separate end-to-end cost.
- [RoboCasa #39](https://github.com/MINT-SJTU/RoboScope-Kit/pull/39): configuration ablations and per-stage
  timers. Discarded-image deferral does not apply to this low-dimensional, chunk-one workload.
- [CALVIN #22](https://github.com/MINT-SJTU/RoboScope-Kit/pull/22): native batching candidate; not adopted
  because BC-RNN state and sampling must remain per episode, and its current Kit real-model acceptance
  is incomplete.
- [RoboDojo #93](https://github.com/MINT-SJTU/RoboScope-Kit/pull/93): existing MessagePack transport,
  measured here without changing arrays; not enabled by default because no benefit was observed.
- [RoboTwin #98](https://github.com/MINT-SJTU/RoboScope-Kit/pull/98): matched MPS comparisons and distinct
  Worker/Policy profiling; MPS is not assumed universally beneficial.
- [MuJoCo 2.3.7 Python documentation](https://mujoco.readthedocs.io/en/2.3.7/python.html): the
  process-global last-compiled XML context, which the cache must preserve for serialization.
