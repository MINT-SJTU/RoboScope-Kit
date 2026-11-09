# robomimic review validation, 2026-11-09

This report responds to PR #110's acceptance-comparison and simulator-cache review.
The historical MPS-off/MPS-on ratios in [THROUGHPUT.md](THROUGHPUT.md) do not establish issue #32's
matched-switch >=2x criterion. The new matched repeats give **1.745x**, so **#32 remains open**.
The historical source, measurements, rejected variants and v6 fallback are preserved.

## Matched hardware and workload

The resumed Linux container has a **12-core CPU quota**, 96 GiB RAM quota, and one RTX 4090 reporting
49,140 MiB, driver 580.105.08. The previous measurements used a 20-core quota and driver 595.71.05.
No timing from the previous hardware allocation is a denominator for the new results.
The cgroup limits, GPU UUID, driver and checkpoint hashes are retained with the evidence.

All new Kit repeats use MPS **off**, one CPU thread per process, no autotune, fresh Worker/Policy
processes, identical staged immutable assets and all five verified PH low-dimensional BC-RNN checkpoints.
Every run executes 50 independent seeds (0-49) on each of Lift, Can, Square, Transport and Tool Hang.
FP32, observation/action semantics, normalization, RNN/RNG handling, control frequency, chunk, horizons,
termination and scoring remain fixed; no cameras or video are enabled. The process-local model cache
starts empty. The cache implementation is the source optimization under measurement, not a prewarmed
process granted only to one side.

The original source is `9174d1b6fb557c0285deb4b3d7d28cc66e2f4ba4` with the previously documented
MessagePack dependency repair. The candidate is the same executable source as PR head `02dd0eb`;
all 200 source/pin/configuration files checked against the measured `final-b43` tree match. Markdown
and Git checkout/diff whitespace attributes are excluded from that executable correspondence check.
The original and patched simulator binding SHA-256 values also match the previous fresh builds.

## Repeated Kit measurements

Formal time is unchanged: `result.json.phase_timings.evaluation_s`, from queue evaluation start until
all work completes. It includes first environment creation, resets, online rollouts and per-item writes.
Model loading and Worker readiness precede this interval. The external harness separately measures
process start through exit and samples whole-container CPU/RAM and GPU usage every second.

The predeclared sequence interleaves three repeats of each configuration. No experiment overlaps
another. All repeats are included; no fastest-run selection is used. Values after +/- are sample SDs.

| Source / fixed K/G (MPS off) | Formal repeats, s | Formal mean +/- SD, s | End-to-end mean, s |
| --- | --- | ---: | ---: |
| Original, 16/1 | 354.179360, 354.446800, 386.214368 | 364.946843 +/- 18.418703 | 371.420320 |
| Original, 20/8 | 225.555819, 220.316283, 221.139883 | 222.337328 +/- 2.817551 | 243.072360 |
| Final b43, 20/8 | 209.193615, 210.327531, 208.040099 | 209.187082 +/- 1.143730 | 230.401180 |

Combined configuration and code gain is **364.946843 / 209.187082 = 1.744596x**. Configuration alone
gives **1.641411x**; at the same K20/G8, the code gain is **222.337328 / 209.187082 = 1.062864x**.
Process end-to-end improvement is **1.612059x**. These measurements demonstrate an improvement but do
**not** meet #32's >=2x target. The variation of the K16/G1 baseline is retained explicitly.

Each of the nine complete runs has **250 unique episodes, 81,289 executed steps and 176 successes**:
Lift 50, Can 50, Square 37, Transport 29, Tool Hang 10. Every ID, success, step count and initial-state
metadata record matches the original correctness baseline, with no gaps, duplicates, extra records or
pairing mismatches. The 74 unsuccessful episodes are counted failures of the policy, not missing work.

## Per-step cache equivalence

An external diagnostic calls the actual `worker.run_episode` and `RobomimicModelZooPolicy`, with an
in-process client shim in all arms to isolate simulator behavior. It uses the same final Worker/Policy
source with three runtimes/settings: the original pinned robosuite binding, the patched binding with
`ROBOSCOPE_ROBOMIMIC_MODEL_CACHE=0`, and the patched binding with the switch set to `1`. Diagnostic
timings are excluded from all performance results.

Each arm runs **260 episodes** in ten persistent processes: all 250 task/seed pairs plus ten same-seed
repeats. Each process cycles through all five tasks in blocks of two seeds, retaining its environments.
Process zero subsequently returns to every task in reverse order and repeats seed zero twice. This
exercises both repeated resets and serialization after another task's model has been compiled.

Every episode retains sampled initial state, raw observations after `reset_to`, the float32 observations
sent to the Policy, every action, next observation, reward/done, and MuJoCo state after reset and every
step: time, qpos/qvel, actuator state/control, qacc/qacc_warmstart, applied forces, mocap and userdata.
Selected mutable model arrays are copied after reset. Initial XML is also retained verbatim.

The comparator requires matching shapes and dtypes, rejects non-finite values, and declares
**atol = rtol = 1e-12**. It additionally checks byte equality. First divergence is ordered by process
shard, execution ordinal, step and observation/action/state phase.

| Pair | Episodes | Arrays compared | Byte-identical arrays | Maximum absolute error | First divergence |
| --- | ---: | ---: | ---: | ---: | --- |
| Original / patched cache off | 260 | 18,356 | 18,356 | 0 | None |
| Original / patched cache on | 260 | 18,356 | 18,356 | 0 | None |
| Patched cache off / cache on | 260 | 18,356 | 18,356 | 0 | None |

All within-arm same-seed repeats pass. Each arm's full workload still has 81,289 steps; all terminal
records and initial-state metadata match the original 250-record baseline, including the 74 failures.
The cache-on arm records **314 hits, 516 misses and 63 serialization calls requiring restoration of
the last-compiled XML context**, out of 260 serialization calls. Thus the diagnostic exercises the
cache and context-restoration paths. Cache-off has zero hits/misses and an empty cache. The raw XML
contains installation-specific absolute asset paths and is not included in the byte-identical numeric
array claim. These results establish agreement for the retained workload and states, not a proof for
arbitrary tasks, assets or simulator versions.

## Independently timed native evaluator

The external native driver imports the unchanged pinned upstream
`robomimic.scripts.run_trained_agent.rollout`. Its function-source SHA-256 is
`81f64a8ff42af969c9581b57c436453fc3fd3933e22df9fec6d3d449ef85b7ae` in all 60 worker instances.
The driver uses 20 persistent processes, cached checkpoint models and retained task environments,
longer tasks first, FP32, `no_grad`, cache on, MPS off and one CPU thread per process. Each seed is
independently reset for NumPy/Torch, matching the frozen workload. It does not start a new interpreter
for each seed or append additional continuous-seed batches. Upstream trajectory construction,
observation copying, `get_state`/XML serialization and termination remain in the rollout function;
there is no video or dataset output.

The native formal timer begins after every worker has loaded its models, immediately before first
dispatch, and ends after the 250 per-episode files are written and the final worker completion arrives.
First environment creation, resets and rollout work are included. Model/import readiness and full
process start-to-exit are separate. Lightweight stage timers wrap calls in the external driver; their
overhead is included. Native has **20 direct inference replicas**, compared with Kit's eight Policy
replicas at K20/G8, under the same 12-core/96-GiB/one-GPU cap. Native is not forced through a new IPC
layer to share eight models. This is a measured persistent native configuration, not a claim that all
possible native configurations have been exhausted.

| Run suffix (prefix `review110-`) | Formal s | End-to-end s | Model/import readiness s | Episodes/s | Steps/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| native-k20-off-01 | 240.270071 | 249.656595 | 8.568055 | 1.040496 | 338.323452 |
| native-k20-off-02 | 239.376071 | 246.928543 | 6.779045 | 1.044382 | 339.586992 |
| native-k20-off-03 | 238.206507 | 246.394575 | 7.364826 | 1.049510 | 341.254321 |

The following pooled rates use total completed work divided by total formal time over three repeats.
Every repeat in every row completes 250 unique episodes, 81,289 steps and 176 successes with no record
pairing mismatch. The native repeats ran after the Kit repeats, without overlapping experiments;
these small repeat sets do not bound all shared-host variation.

| Implementation / fixed K/G | Formal mean +/- sample SD, s | End-to-end mean +/- sample SD, s | Episodes/s | Steps/s |
| --- | ---: | ---: | ---: | ---: |
| Original Kit, 16/1 | 364.946843 +/- 18.418703 | 371.420320 +/- 17.138811 | 0.685031 | 222.742028 |
| Original Kit, 20/8 | 222.337328 +/- 2.817551 | 243.072360 +/- 2.878472 | 1.124418 | 365.611122 |
| Final Kit, 20/8 | 209.187082 +/- 1.143730 | 230.401180 +/- 1.916228 | 1.195102 | 388.594742 |
| Native, 20 direct workers | 239.284217 +/- 1.034844 | 247.659904 +/- 1.749675 | 1.044783 | 339.717350 |

Kit's formal rate is **1.143877x** this native configuration; its end-to-end rate is **1.074907x**.
These ratios are separate from the **1.744596x** Kit-internal gain and do not satisfy the 2x target.

## Startup, stages and resources

Both implementations reuse the same locally staged, hash-verified assets and checkpoints; no new
download preparation is included in these timers. Fixed K/G skips autotune. Original Kit model-load
times are 6.455/2.096/2.156 s at G1 and 16.223/16.103/16.235 s at G8. Final G8 times are
18.256/16.085/15.849 s; Worker readiness is 1.818-1.823 s. Native process/model readiness is tabulated
above; individual worker import and model-load times are retained in `summary.json.ready` and overlap.

For example, the first native run has overlapping worker sums of 169.527 s for environment creation,
459.222 s for reset, 508.879 s for reset_to, 435.529 s for get_state, 2,069.438 s for simulator steps
and 0.054079 s for per-episode JSON writes. Reset and get_state can be nested in other stages; these
sums must not be added as sequential wall-clock phases. Kit's per-worker stages, Policy waits and
per-run phase timers remain in the raw results. The separate historical output-writing diagnostic
and preparation/build costs remain in [THROUGHPUT.md](THROUGHPUT.md), explicitly outside this repeat set.

| Configuration | Mean CPU cores over the three runs | Peak container memory, GiB | Peak GPU memory, MiB |
| --- | ---: | ---: | ---: |
| Original Kit K16/G1 | 8.038 | 14.487 | 853 |
| Original Kit K20/G8 | 10.880 | 22.594 | 4,598 |
| Final Kit K20/G8 | 11.009 | 29.657 | 4,598 |
| Native K20 | 11.820 | 35.994 | 11,794 |

Samples cover process start-to-exit at one-second intervals. Container memory includes file caches
and background services, not just private model RSS. CPU values are means of per-run means; memory
values are maxima over all three runs. All twelve runs exit zero, with no OOM/memory-limit events
or lost episodes. Policy task failures are retained. GPU utilization alone is not used to infer a
bottleneck. The changed CPU quota and driver preclude attributing the day-to-day timing difference
to code, or quantifying the effect of either hardware change in isolation.

## Reproduction and delivery

The [evidence index](EVIDENCE.md#review-revision-2026-11-09) provides the complete archive, small summary
bundle, checksums and exact scripts. `matched-repeats.sh` records the nine-run order and resource guards;
`diagnostics.sh` records all trace/native invocations. All generated evidence stays outside the repository.
The scripts record the measurement host layout under `/root/issue32` and checkpoint path
`/root/benchmark/checkpoint`; adjust these explicit paths and choose new output directories when
reproducing. The original correctness records used for pairing are retained in the historical archive.

Build each source snapshot only through its own `benchmarks/robomimic/build-runtime.sh`, into an empty
runtime directory, with `KIT_PYTHON` set to the installed Kit control interpreter. The original source
snapshot includes only the documented dependency repair; the final snapshot includes the pinned cache
patch. Create each Policy configuration using that source and runtime, as in the archived `policy-*.json`.
The final MPS-off invocation is:

```bash
export KIT_PYTHON=/path/to/kit-python
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export ROBOSCOPE_ROBOMIMIC_MODEL_CACHE=1
export PYTHONPATH="$PWD/packages/roboscope-kit/src"
bash benchmarks/robomimic/build-runtime.sh /new/final-runtime
"$KIT_PYTHON" -m roboscope_kit.runtime.benchmark --benchmark robomimic --suite ph_low_dim \
  --checkpoint-source /path/to/checkpoints \
  --benchmark-manifest /new/final-runtime/benchmark-manifest.json \
  --policy-config /path/to/final-policy.json --concurrency 20 --replicas 8 \
  --output-dir /new/final-full-run
```

MPS is off on both sides: omit `--cuda-mps` and verify no MPS server is active. Run the original snapshot
at K16/G1 and K20/G8 with its own runtime/Policy configuration; repeat each arm three times in separate
output directories. The cache switch must be exported to the evaluator, not scoped only to the builder.
Use the archived external `trace.py`, `compare-traces.py` and `native.py` for the diagnostic/native runs;
invoke each with the corresponding built runtime's `env/bin/python`. The summary bundle includes the
exact argument lists and scripts without requiring the full trace download.

This revision changes documentation only. Executable source and simulator pins are unchanged from
the previously completed clean b43 build, whose log verifies all 250 initial-state digests. The version,
lock and dated changelog remain **0.8.0b43**. Remote Linux checks were repeated: **264 existing pytest
cases**, Ruff (including the example), Python support policy, uv lock consistency, bash syntax and
ShellCheck all passed. A new output-directory `--self-test` passed; its lightweight state check reports
Lift, while complete five-task coverage comes from the real evaluations and prior builder checks.
No new repository tests or manual installed-simulator edits were added. These local-to-the-server
checks are separate from GitHub's compatibility matrix, which still requires its own successful run.
