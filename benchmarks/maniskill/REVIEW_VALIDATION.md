# ManiSkill issue 25 review validation

Validated on 2026-11-09 (Asia/Shanghai). The candidate starts from upstream main `9174d1b` and
uses Kit `0.8.0b44`, ManiSkill `baab60ede2e89167c1b7aaed41a9aa8e690a9d1e`, SAPIEN `3.0.0b1`,
CPython `3.11.15`, Torch `2.6.0` / CUDA `12.4` and NumPy `1.26.4`. Hardware is one RTX 4090
with 49,140 MiB, driver `580.105.08`, and an Intel Xeon Platinum 8470Q host. The container CPU
quota is `2000000/100000`, equivalent to 20 cores. The pinned builder and [protocol](PROTOCOL.md)
record environment and asset provenance. [The inventory](TASK_INVENTORY.md) defines the complete task scope.

## Source and scope

The per-profile tables and lifecycle sums below belong to build `runtime-r1` and batch `models-r2`.
This is the first correct eleven-profile batch after controller-profile and native-reference fixes.
Build `runtime-r2` adds missing-asset preflight and Bridge image dtype handling. Its complete PPO
suite, process-isolated native check and parallel routing check are recorded in the next section.
The historical per-profile measurements retain their original source label.

Each profile runs independently for 100 seeds (0–99), fixed K1/G1 and N10 GPU-vectorized groups,
with autotune, MPS, video and profiling disabled. Every vector group performs the full seeded
reset and native horizon. The unchanged official PPO actors use deterministic FP32 means.
The native reference constructs upstream Gym environments and wrappers independently of the SDK worker.

## Final runtime PPO suite and routing

The `runtime-r2` SDK command runs all eleven profiles through `official_ppo_state_v1` in one
evaluation at fixed K1/G1 and N10, with the same 100 seeds per task and full horizons. It completes
1,100 episodes and 88,000 control steps, with 1,010 success-once episodes, 621 success-at-end
episodes and 90 unsuccessful task episodes. There are no execution failures, missing records or
duplicate records. Every result matches the independent native task references in `models-r2`,
including the initial-state digest, first action, episode return, both success metrics and step count.

The comparison uses `results/native1100-merged.json`, which retains the paths and SHA-256 values
of the eleven previously validated 100-episode native runs. These references were collected with
`runtime-r1`. They are reused as correctness references for the final SDK source. The single-process
multi-task native attempt described below did not produce a complete reference.

| Metric, one `runtime-r2` SDK suite run | Value |
| --- | ---: |
| Complete command time, s | 418.953 |
| Formal evaluation time, s | 412.541 |
| Formal episodes/h | 9,599.1 |
| Formal control steps/s | 213.31 |
| Complete command episodes/h | 9,452.1 |
| Complete command control steps/s | 210.05 |
| GPU mean busy, complete command % | 31.60 |
| Device VRAM sampled peak, GiB | 5.240 |
| Completed / expected episodes | 1,100 / 1,100 |
| Success once / success at end | 1,010 / 621 |
| Actual control steps | 88,000 |

The retained native launcher runs each task in its own process. A current-source 110-episode check
(ten episodes per task at N10, 8,800 control steps) passes with this process isolation. A separate
K2/G2 SDK run completes the same 110-episode workload with 101 success-once episodes, and its
records match that native reference exactly. This routing check uses the first ten seeds per task
and retains each task's full horizon. Its reduced episode count is recorded as diagnostic coverage.

The current-source native 110-episode run also matches the corresponding first ten seeds of all
eleven original 100-episode native references exactly. This independently repeats the native
initial-state, first-action, return, success and step-count checks. The result is retained in
`results/native-repeat110-comparison.json`.

## Native process lifetime investigation

The first single-process native command for all eleven tasks failed while constructing the final
AnymalC-Reach environment, when the pinned simulator queried net contact forces. The preceding
ten tasks had run, but the command exited with CUDA out-of-memory and produced no complete
native result. Its sampled device VRAM peak was approximately 5.1 GB, so that observation alone
does not establish exhausted device memory or the cause of the failure.

A reduced probe reproduced the failure after two one-step groups for each of the preceding ten
tasks. Clearing retained tensor locals, dropping the environment reference and collecting garbage
made the old environment and scene weak references expire, but AnymalC-Reach still failed. This
negative result does not establish a Python reference leak as the cause. The cleanup experiment
is excluded from the submitted implementation.

The native launcher now follows the upstream evaluator's one-task process lifetime. Each child
uses the same pinned model, seeds, vector width, controller and horizon through the independent
native environment construction path. The parent checks task and episode coverage before writing
the merged result and retains child logs, raw records, commands and result hashes. A child failure
rejects the complete result. Parent timing reports sequential child phase sums and the entire
command's end-to-end time, including process startup and repeated model loads. These timing
boundaries differ from the persistent SDK Worker and are recorded for correctness reproduction.

## Real-policy correctness

All 1,100 SDK/native episode pairs match on profile/task identity, seed, initial-state digest,
first action, success-once, success-at-end, episode return and executed steps. The comparison
uses zero action/return tolerance. Maximum first-action and return differences are both zero.
There are 88,000 executed environment control steps, 1,010 success-once episodes and 621
success-at-end episodes on each path. Every SDK job exits successfully, with zero worker failures,
OOMs, missing or duplicate records. The 90 unsuccessful task episodes are retained.

Each table row contains 100 episodes on each path. The success and mean-return columns are equal
between SDK and native. These comparisons cover initial inputs and first actions; per-step
input/action traces are optional separate diagnostics.

| Task | Steps, each path | Success once / at end, each path | Mean episode return, each path |
| --- | ---: | ---: | ---: |
| PickCube-v1 | 5,000 | 100 / 100 | 43.740302 |
| RollBall-v1 | 8,000 | 63 / 0 | 20.062334 |
| PushT-v1 | 10,000 | 98 / 72 | 62.870129 |
| LiftPegUpright-v1 | 5,000 | 100 / 50 | 39.815195 |
| PokeCube-v1 | 5,000 | 70 / 36 | 38.040053 |
| PullCube-v1 | 5,000 | 100 / 8 | 19.601826 |
| PushCube-v1 | 5,000 | 100 / 24 | 23.559583 |
| StackCube-v1 | 5,000 | 84 / 81 | 39.302787 |
| TwoRobotPickCube-v1 | 10,000 | 95 / 95 | 80.106171 |
| TwoRobotStackCube-v1 | 10,000 | 100 / 56 | 83.247011 |
| AnymalC-Reach-v1 | 20,000 | 100 / 99 | 146.416451 |

## Per-profile SDK baseline

The eleven jobs are separate process launches. Their complete command times sum to 482.689 s
and their formal evaluation times sum to 427.454 s. This sum describes the per-profile batch.
A monolithic suite loads its Policy once and has its own timing evidence. Each row below is a
single measured run, so it establishes a reproducible baseline without a run-to-run variance estimate.

Formal time is the SDK `evaluation_s` interval. Complete command time includes startup,
checkpoint preparation, model loading, runtime preparation, evaluation, output and cleanup.
Native timers have different boundaries; correctness pairing does not establish a speedup.

| Task | Complete s | Formal s | Formal episodes/h | Formal steps/s | RPC mean ms/batch | RPC wait sum s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PickCube-v1 | 38.361 | 33.010 | 10,905.7 | 151.47 | 2.457 | 1.229 |
| RollBall-v1 | 41.618 | 37.117 | 9,699.0 | 215.53 | 2.260 | 1.808 |
| PushT-v1 | 51.091 | 46.449 | 7,750.4 | 215.29 | 2.931 | 2.931 |
| LiftPegUpright-v1 | 37.956 | 33.530 | 10,736.7 | 149.12 | 2.633 | 1.316 |
| PokeCube-v1 | 37.611 | 33.102 | 10,875.4 | 151.05 | 1.963 | 0.982 |
| PullCube-v1 | 36.379 | 31.920 | 11,278.4 | 156.64 | 2.149 | 1.075 |
| PushCube-v1 | 39.861 | 34.181 | 10,532.2 | 146.28 | 2.719 | 1.359 |
| StackCube-v1 | 43.049 | 36.743 | 9,797.9 | 136.08 | 2.484 | 1.242 |
| TwoRobotPickCube-v1 | 46.906 | 42.009 | 8,569.6 | 238.05 | 2.583 | 2.583 |
| TwoRobotStackCube-v1 | 53.849 | 48.620 | 7,404.3 | 205.68 | 2.317 | 2.317 |
| AnymalC-Reach-v1 | 56.007 | 50.773 | 7,090.4 | 393.91 | 2.429 | 4.859 |

RPC round trip includes request transport, server inference and response validation. Independent
server queue time and GPU actor latency are N/A in these unprofiled runs. The state profiles do
not request image observations. Pure render CPU/GPU time was not instrumented and remains N/A.

| Task | GPU mean, complete / formal % | GPU sampled peak % | VRAM mean / sampled peak GiB | CPU mean / sampled peak cores |
| --- | ---: | ---: | ---: | ---: |
| PickCube-v1 | 22.68 / 25.63 | 63 | 3.751 / 5.113 | 0.949 / 1.104 |
| RollBall-v1 | 23.96 / 26.68 | 64 | 3.821 / 4.913 | 0.988 / 1.115 |
| PushT-v1 | 22.71 / 25.04 | 52 | 3.966 / 4.915 | 0.992 / 1.108 |
| LiftPegUpright-v1 | 25.75 / 28.24 | 65 | 3.798 / 5.046 | 0.986 / 1.115 |
| PokeCube-v1 | 25.32 / 28.43 | 61 | 3.865 / 5.113 | 0.982 / 1.073 |
| PullCube-v1 | 19.45 / 20.94 | 57 | 3.641 / 4.913 | 0.989 / 1.115 |
| PushCube-v1 | 20.42 / 23.90 | 63 | 3.594 / 4.913 | 0.912 / 1.101 |
| StackCube-v1 | 22.47 / 25.19 | 67 | 3.721 / 5.113 | 0.932 / 1.111 |
| TwoRobotPickCube-v1 | 36.58 / 41.02 | 76 | 4.071 / 5.198 | 0.995 / 1.112 |
| TwoRobotStackCube-v1 | 40.22 / 44.28 | 77 | 4.230 / 5.263 | 0.986 / 1.112 |
| AnymalC-Reach-v1 | 33.50 / 37.07 | 74 | 3.940 / 4.950 | 0.914 / 1.114 |

Resources are sampled every 250 ms. GPU utilization and VRAM are device-wide NVML samples.
Peaks are sampled maxima. CPU mean uses accounted process-tree CPU time divided by command wall
and peak uses sampled process CPU. One core is 100% of one CPU core. SM throughput, Tensor activity
and RT throughput were not collected in this batch and remain N/A.
NVML busy rates do not measure those hardware counters.

## Measured lifecycle budget

The following sums combine eleven sequential K1/G1 jobs. Worker phases are CPU wall intervals.
The environment step includes physics, controller work and generated observations. Rollout
contains step, input staging and RPC wait, so those nested rows must not be added again.

| Recorded phase | Sum over eleven jobs, s |
| --- | ---: |
| Policy load | 19.978 |
| Runtime preparation | 7.070 |
| Environment creation, 110 vector groups | 95.221 |
| Explicit seeded reset, 110 vector groups | 50.317 |
| Rollout | 217.863 |
| Environment step including observation, 8,800 vector control ticks | 193.365 |
| Policy input staging | 0.650 |
| Client inference wait, 8,800 batches | 21.700 |
| Environment close | 24.163 |

Interpreter startup, result writing and finalization are included in command wall time but were
not independently instrumented. Per-lane elapsed time represents the shared N10 group rollout.
It does not provide independent per-episode latency samples. Raw worker group metadata retains
creation/reset/rollout/close values for each cold and subsequent group.

## Octo visual-policy diagnostics

The separate Octo runtime loads the real public `rail-berkeley/octo-small` checkpoint on
`cuda:0`. Its immutable model revision is `03d88976c54a58e10480d2043a8c762b35bc2611`;
the main weight file is 546,696,551 bytes and is verified against SHA-256
`590df097f8a37bbc1c3aac2a488c0fb08e72bae8abaedfb89f0685677d848962`.
This is the public OXE-pretrained visual policy using Bridge action statistics, rather than a
ManiSkill-finetuned checkpoint. [The Octo example](../../examples/maniskill/OCTO.md) records its
source pins, independent per-episode RNG convention, inputs and native controller mapping.

The initial `octo-small-*-r2` real-environment diagnostic runs one episode for each of the four
Bridge tasks through both SDK and independent native paths. All four initial-state and full
initial-observation digests match exactly. Both paths record one success across the four episodes,
with equal episode returns and executed step counts. First actions differ by approximately
`1.6e-6` to `1.25e-5`, so this diagnostic does not establish exact policy equivalence.

The unmodified pinned upstream inference also produces approximately `1e-5` action differences
between fresh GPU processes on three fixed synthetic image frames. Inside each process, the
adapter and upstream outputs match bit for bit. The pinned XLA
[algorithm selector](https://github.com/openxla/xla/blob/ca31652cdbeb6ea187589dea546ff8019274f8b2/xla/service/gpu/stream_executor_util.cc#L537)
otherwise chooses the candidate with the shortest measured startup runtime. Enabling only
`--xla_gpu_deterministic_ops=true` makes the two fresh-process three-frame diagnostics bit-identical,
while the adapter still matches the upstream inference in each process. This flag is now set before
JAX initialization; default autotuning level, precision, model parameters and RNG streams are unchanged.
A separate diagnostic with autotuning disabled also passed, but that extra setting is not retained.

The native ManiSkill3 controller consumes intrinsic Euler XYZ rotations, while the legacy Simpler
controller consumes axis-angle rotations. The adapter converts the same predicted Bridge rotation
into the pinned native controller's Euler representation. A 1,003-case rotation-matrix diagnostic,
including singular-angle cases, has maximum matrix error `4.72e-16` before action casting and
`1.03e-7` after float32 casting. This verifies the rotation representation conversion, rather than
claiming physical equivalence with the legacy controller's different target-update semantics.

The final four-task `octo-small-*-r3` real-environment diagnostic passes with the deterministic
setting. All four SDK/native episode records match exactly, with maximum first-action and return
differences of zero. The complete per-step traces also match: 300 actions and 304 observations,
including the four initial observations. There are zero comparison errors.

| Four-task, one-episode-per-task diagnostic | Time, s |
| --- | ---: |
| SDK complete command | 179.390 |
| SDK formal evaluation | 164.150 |
| Native complete command | 177.177 |

These are single diagnostic runs with different timing boundaries, not a performance comparison.
They validate all four real visual-policy paths at N1. The complete 100-episode-per-task comparison
below subsequently passes at N10. The full matched matrix now covers fifteen tasks: eleven PPO
state tasks and four Octo Bridge visual tasks.

Raw evidence is retained under `/autodl-fs/data/roboscope/maniskill-review-20261109` in
`results/octo-small-sdk-r2/`, `results/octo-small-native-r2/`, `octo-reference-comparison.json`,
`octo-reference-repeat2.json`, `octo-determinism-only-comparison.json` and
`octo-reference-deterministic-only-{1,2}.json`. The final small diagnostic is in
`results/octo-small-sdk-r3/`, `results/octo-small-native-r3/` and
`results/octo-small-comparison-r3.json`; the corresponding `*-r3-traces/` directories retain
the complete paired traces. The local validation workspace also retains
`check_octo_rotation_mapping.json` alongside its script, outside the checkout. These diagnostic
artifacts are not committed to Git.

### Completed 400-episode visual-policy comparison

The `octo400-sdk-r1` run completes `bridge_rgb_v1` with 100 seeds (0–99) per task, K1/G1 and N10.
Autotune, MPS, video, profiling and per-step trace diagnostics are disabled. The model, RGB inputs,
native controller, full horizons and independent episode RNG protocol are unchanged. All 400
episodes complete, executing 30,000 control steps. The selected success-once metric is 110/400
(27.5%); success at end is 107/400 (26.75%). The 290 unsuccessful task episodes are retained.
The independent `octo400-native-r1` run completes the same 400 episodes and 30,000 steps. All
episode pairs match exactly on identity, seed, full initial-observation and initial-state digests,
first action, return, both success metrics and executed steps. Maximum first-action and return
differences are zero with zero tolerances. The strict comparison reports zero errors, missing
records or duplicate records; both commands exit successfully without execution failures.
The SDK records no worker failures or OOM events.

| Task, 100 episodes on each path | Steps, each path | Success once / at end, each path | Success-once rate |
| --- | ---: | ---: | ---: |
| PutCarrotOnPlateInScene-v1 | 6,000 | 18 / 17 | 18% |
| PutEggplantInBasketScene-v1 | 12,000 | 60 / 60 | 60% |
| StackGreenCubeOnYellowCubeBakedTexInScene-v1 | 6,000 | 0 / 0 | 0% |
| PutSpoonOnTableClothInScene-v1 | 6,000 | 32 / 30 | 32% |

Reward mode is `none`, so every episode return is zero. The strict native comparison also checks
success metrics, steps, initial inputs and first actions. The full 400-episode batch does not
collect per-step traces; those are covered by the separate four-episode diagnostic above.

| Metric, one SDK run | Complete command | Formal evaluation |
| --- | ---: | ---: |
| Time, s | 2,409.006 | 2,393.123 |
| Episodes/h | 597.757 | 601.724 |
| Control steps/s | 12.453 | 12.536 |
| GPU mean busy, % | 52.911 | 53.259 |
| GPU sampled peak busy, % | 99 | 99 |
| Device VRAM mean / sampled peak, GiB | 7.331 / 7.650 | 7.373 / 7.650 |
| CPU sampled mean / peak, cores | 1.293 / 12.147 | 1.296 / 12.147 |
| Process-tree RSS mean / sampled peak, GiB | 7.453 / 7.831 | 7.497 / 7.831 |

Policy loading takes 11.703 s and runtime preparation takes 0.659 s. Accounted process-tree CPU
time is 3,223.369 core-seconds, or 1.338 cores averaged over the complete command. Resource
sampling uses the same 250 ms, device-wide NVML and process-tree boundaries described above;
peaks are sampled maxima. One CPU core corresponds to 100% of one core.

The 3,000 client inference batches accumulate 748.197 s of RPC round-trip wait, with a mean of
249.399 ms per batch. This includes transport, server inference and response validation, and is
nested inside formal evaluation time. Independent server queue wait, server GPU inference
latency and isolated render CPU/GPU duration are N/A in this unprofiled run. SM throughput,
Tensor activity and RT throughput also remain N/A; the retained counter probes report
`ERR_NVGPUCTRPERM`. GPU busy percentages do not substitute for these counters.

Raw SDK records are in `results/octo400-sdk-r1/` and its matching `measurements/` artifacts under
the external validation directory. Native records are in `results/octo400-native-r1/result.json`;
`results/octo400-comparison-r1.json` records strict pairing. `octo-results-summary.json` retains
both measured commands, parsed results and per-task phase summaries. The local workspace also
retains `octo400-sdk-measurement.json`, `octo400-sdk-result.json` and `octo400-sdk-metrics.json`
outside the checkout. Each path has one baseline measurement; no run-to-run variance or runtime
speedup is claimed.

The native reference's own evaluation clock covers 2,326.281 s, and its complete command takes
2,342.355 s. Startup and model loading take 11.753 s. Its evaluation clock and lossless image-pipe
transport differ from the SDK boundaries, so these measurements are correctness baselines.

| Native baseline metric | Value |
| --- | ---: |
| Complete command / own evaluation episodes/h | 614.766 / 619.014 |
| Complete command / own evaluation control steps/s | 12.808 / 12.896 |
| GPU mean / sampled peak busy, complete command % | 52.146 / 99 |
| Device VRAM mean / sampled peak, complete command GiB | 7.335 / 7.650 |
| CPU sampled mean / peak, complete command cores | 1.278 / 12.106 |
| Process-tree RSS mean / sampled peak, complete command GiB | 7.380 / 7.725 |

Native accounted process-tree CPU time is 2,999.561 core-seconds, averaging 1.281 cores over the
complete command. The native resource sampler has no evaluation-stage marker, so resource values
for its evaluation interval remain N/A. Its actor round-trip sum is 693.174 s, nested within rollout.
Independent GPU actor latency, queue time, render duration and core counters remain N/A on this path.

### Visual-policy lifecycle budget

Each SDK task has ten N10 vector groups. The table reports CPU wall intervals summed over those
groups, in seconds. Step includes physics, controller work and observation generation; it does
not isolate render GPU time. Step and RPC intervals are nested within rollout.

| SDK task | Create | Reset | Rollout | Step with observation | RPC wait |
| --- | ---: | ---: | ---: | ---: | ---: |
| Carrot on plate | 24.027 | 15.544 | 377.351 | 211.222 | 161.396 |
| Eggplant in basket | 23.062 | 17.453 | 1,062.805 | 760.674 | 292.061 |
| Stack green on yellow | 20.206 | 15.585 | 370.609 | 218.656 | 146.903 |
| Spoon on cloth | 22.391 | 16.816 | 404.244 | 251.408 | 147.837 |
| Total | 89.686 | 65.400 | 2,215.010 | 1,441.960 | 748.197 |

SDK observation staging adds 23.458 s, environment close 10.801 s and policy cleanup 0.052 s.
Native phase sums are environment creation 88.067 s, reset 65.283 s, rollout 2,156.615 s and
environment close 14.957 s. These are sequential run budgets, not isolated GPU execution times.

| SDK task | First group rollout, s | Subsequent nine groups median [min, max], s |
| --- | ---: | ---: |
| Carrot on plate | 52.212 | 36.100 [34.955, 37.943] |
| Eggplant in basket | 106.742 | 106.936 [99.006, 110.804] |
| Stack green on yellow | 37.403 | 37.108 [35.704, 38.478] |
| Spoon on cloth | 39.842 | 40.701 [39.274, 41.202] |

These distributions describe shared vector-group rollout, not independent per-episode latency.
The first task includes the initial actor compilation cost; later task-first groups run in the
same persistent process. They are therefore reported as first versus subsequent groups rather
than treated as identical cold starts.

## Final source and build checks

`runtime-r3-final-source-verification.json` verifies all 86 installed SDK Python files against
the candidate, all measured executable source files against the final source snapshot, and the
protocol, supported suites and defaults against the installed runtime manifest. All checks pass.
The final local check snapshot in `checks-final.json` records 265 passing pytest tests, passing
Ruff checks including the ManiSkill benchmark and examples, passing `bash -n` and ShellCheck for
both builders, and a passing `ci/python_support.py check`.

The actual `runtime-r3` build passes `--self-test`, including real rendering. The final
current-code Octo CLI quick test in `results/octo-quick-final/` exits successfully with one
completed episode, 60 steps and zero task successes. Its complete command is 60.091 s and formal
evaluation is 44.925 s. The result is an execution check; its unsuccessful episode is retained.
Raw timing is in `measurements/octo-quick-final.json` and the pipeline log is
`logs/octo-quick-final-pipeline.log`.

## Remaining acceptance gaps

- The pinned task inventory contains 55 evaluation tasks. Fifteen now have complete matched
  100-episode real-policy checks: eleven state PPO tasks and four Bridge visual-policy tasks.
  The remaining 40 tasks still require selected compatible policies and matched native comparisons.
- The original eighteen tabletop RGB profiles still require compatible visual policies and
  matched native evaluation. The four accepted Bridge visual profiles are a separate task family.
  Environment rendering diagnostics remain separate from learned-policy acceptance.
- Five tasks remain blocked by unavailable UCSD archives: AssemblingKits, PickClutterYCB, TurnFaucet,
  OpenCabinetDoor and OpenCabinetDrawer. These asset and policy gaps overlap.
- Zero-action environment diagnostics complete all 76 configured profiles. Of these, 68 pass,
  covering 50 of the 55 evaluation tasks. Eight profiles for the five asset-blocked tasks fail
  explicitly with missing-asset errors. These short reset, action and render checks do not count
  as trained-policy or full-horizon native acceptance.
- Issue 25 full acceptance remains pending the complete task/policy/native matrix.

## Reproduction and raw results

The eleven-model batch uses task-specific copies of the installed manifest, each selecting one
profile with its frozen seeds and horizon. For each copy, run the SDK and native commands below,
using a fresh output path. `models.json` is produced by `download_models.py --all`.

```bash
"$runtime/env/bin/python" -m roboscope_kit.runtime.benchmark --benchmark maniskill \
  --benchmark-manifest "$manifest" --suite "$suite" --checkpoint-source "$models" \
  --policy-config "$policy" --episodes 100 --concurrency 1 --replicas 1 --output-dir "$sdk_output"
"$runtime/env/bin/python" examples/maniskill/native_eval.py \
  --benchmark-manifest "$manifest" --suite "$suite" --checkpoint "$models" \
  --episodes 100 --num-envs 10 --output "$native_result"
python examples/maniskill/compare_native.py --sdk-output "$sdk_output" \
  --native-result "$native_result" --benchmark-manifest "$manifest" \
  --suite "$suite" --output "$comparison"
```

Raw files remain outside Git under `/autodl-fs/data/roboscope/maniskill-review-20261109`.
`model-results-summary.json` records this measured batch. `results/models-r2-<profile>-sdk/`,
`results/models-r2-<profile>-native/`, `results/models-r2-<profile>-comparison.json` and matching
`measurements/` files retain complete records, commands, logs and 250 ms resource samples.
The final SDK suite is in `results/full1100-sdk/`; its paired comparison is
`results/full1100-merged-reference-comparison.json`. `results/native1100-merged.json` retains
the source task-reference hashes. `results/parallel110-comparison.json` records the parallel
110-episode comparison, and `results/native-repeat110-comparison.json` records the independent
native repetition. `profiles-r3-summary.json` records environment-only coverage.
The failed monolithic native command and `native_lifecycle_probe.py` diagnostics are retained
alongside the passing runs.

The separate evidence archive is named `issue25-review-evidence-20261109.tar.gz` and is stored
outside Git in the external validation directory. It collects raw results, commands, logs,
measurements, source verification, provenance and the evidence manifest; it excludes simulator
assets, checkpoints, caches and installed environments.
