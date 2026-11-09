# RoboCasa365 startup validation — 2026-11-07

This historical startup record retains its original source attribution. The
[2026-11-09 review update](#review-update--2026-11-09) records the reviewed build,
tests and real five-slot Quick Test; the separate
[reduced native comparison](native-comparison-20261109.md) reports later paired
rollouts across all 50 tasks.

**Result: all 50 official target tasks passed the startup integration check.**
This pass covers 18 Atomic-Seen, 16 Composite-Seen and 16 Composite-Unseen tasks
in `pretrain` scenes. Its scope was explicitly reduced to startup validation:
one environment and evaluated reset per task, followed by two real-policy
8-action chunks. The formal 50-rollout-per-task protocol is unchanged.

For that startup pass, the longer Kit/native collection and automatic repeats were stopped; their
partial records are retained for diagnosis. This record does not claim a
benchmark success rate, full native equivalence, support for all 365 tasks, or
a throughput improvement. [Issue #97](https://github.com/MINT-SJTU/RoboScope-Kit/issues/97)
remains the integration reference; GR1 Tabletop and issue #16 are separate.

## Inputs and information ownership

| Item | Tested configuration / source of truth |
| --- | --- |
| Startup-tested Kit code | `0.8.0b36`, `12272399a902e9a19307df9ab06f10fc642a0cac`; the PR refresh is recorded below |
| Task inventory, horizons and protocol | [Protocol](../../packages/roboscope-kit/docs/robocasa365-protocol.md), [ordered task data](../../packages/roboscope-kit/src/roboscope_kit/runtime/benchmarks/robocasa365/protocol.py), [manifest](benchmark-manifest.json) |
| Simulator | RoboCasa 1.0.1, robosuite 1.5.2, MuJoCo 3.3.1, Python 3.11.13, NumPy 2.2.5; exact revisions in [source inputs](source-inputs.json) and [package pins](runtime-requirements.txt) |
| Assets and corrections | Six archives in [asset inputs](asset-inputs.json); four checksum-verified patches in [runtime patches](runtime-patches.json), applied by [the builder](build-runtime.sh) |
| Model | Official multi-task Diffusion Policy `pretrain_human300`, epoch 0500, FP32, 100 denoising steps; code, checkpoint and CLIP hashes in [model inputs](../../examples/robocasa365_diffusion/model-inputs.json) |
| Inference environment | Python 3.11.15, torch 2.6.0+cu124; installation and locked dependencies in the [Policy example](../../examples/robocasa365_diffusion/README.md) |
| Machine | RTX 4090 reporting 49,140 MiB, driver 580.105.08; Xeon 6459C, 12-core CPU quota, 96 GiB memory limit |
| Startup run | K=1/G=1, one slot per task, constructor seed 1111111, stream episode 0, original rendering, videos and MPS off |

The protocol document owns robot/controller settings, camera and language inputs,
state preprocessing, action conversion, frequencies, reset streams, termination
and scoring. The tables and hashes are not duplicated here. The SDK Policy loads
verified real weights through `SessionContext.checkpoint_path`; no inference or
runtime optimization was enabled for this check.

## Startup results

The external startup harness uses the unchanged Kit environment, observation and
action helpers, plus `PolicyProcessAdapter` and the real SDK client/server path.
It checks reset, three RGB histories, state shapes/finite values, nonempty language,
finite `(8,12)` actions, physical stepping, another observation/inference cycle,
`end_episode`, and environment cleanup. The 16-step limit belongs only to this
diagnostic harness; it does not modify task horizons or emit formal episode scores.

| Task set | Checked / expected | Physical steps |
| --- | ---: | ---: |
| Atomic-Seen | 18 / 18 | 288 |
| Composite-Seen | 16 / 16 | 256 |
| Composite-Unseen | 16 / 16 | 256 |
| Total | **50 / 50** | **800** |

There were **100 inference calls**, zero failed/retried tasks and zero missing or
duplicate task records. Each task executed all 16 steps. The policy and simulator
processes exited after the run; GPU memory returned to zero. Results explicitly
record `full_suite: false`, `native_equivalence_verified: false` and zero completed
benchmark episodes.

| Supporting check on b36 | Result |
| --- | --- |
| Fresh builder and registry/import self-test | Passed in `runtime-acceptance36`; self-test `self-test.AUCAyA` |
| Existing pytest / Ruff | 188 passed; Ruff passed |
| Build/install shell checks | Bash syntax and ShellCheck passed |
| Normal CLI real-model Quick Test | Completed one 900-step CloseBlenderLid episode, 0/1 task successes, no execution error |
| Native five-slot repeatability pilot | Two independent five-episode Kettle runs on each path; every run had 3/5 successes and 2,250 steps; reset, observation, action and state traces matched exactly |
| Identical-input native loader checks | CloseBlenderLid and CloseFridge first batches matched preprocessing, CPU/CUDA RNG and actions exactly; maximum action difference 0 |
| Full 50-by-50 native comparison | Cancelled for this validation pass; not an acceptance result |

Startup wall time through worker completion was **684.687 s** (11.4 minutes),
including 9.888 s in the Policy load call. Sequential worker intervals were
282.449 s for construction, 284.324 s for evaluated resets, 62.330 s for Policy
round trips, 33.224 s for native steps/observations and 6.229 s for cleanup.
Two-second GPU samples recorded 3.95% mean utilization and 1,604.9 / 3,012 MiB
mean / peak memory. These are startup diagnostics, not formal evaluation
throughput; RPC and native-step timers do not isolate GPU inference or rendering.

### PR refresh — 2026-11-08

After syncing main `406cf3d`, source `f165f3717c4dea0e3216d4bc723f39124026a6bd`
passed **265 existing pytest tests**, Ruff, Bash syntax, ShellCheck and the Python
support check. Its registry/import self-test passed, and a fresh normal CLI Quick
Test completed one 900-step CloseBlenderLid episode (0/1 successes, no execution
error). Benchmark, Policy, build and pin files match the startup-tested source
byte for byte; the only SDK changes are the three inherited autotune/orchestrator
modules listed in the external source proof. The later main `9174d1b` sync and
report update change documentation only. The original 50-task run is not relabeled
as a new-head run.

At that refresh, the CLI self-test checked imports/registry rather than creating
an environment; Quick Test used one slot rather than the formal five-slot group;
profiling used a private client request to keep a group capture open. The external
startup harness did not establish those follow-ups. The review update below
records their subsequent fixes and additional tests requested by the maintainer.

### Review update — 2026-11-09

Reviewed integration source `04cdd858a548aba13ef63a21f06f6301a58a0a2d`
uses version `0.8.0b34` on main `9174d1b6fb557c0285deb4b3d7d28cc66e2f4ba4`.
The version was reset to the next free release after rebasing; the b36 and b40
labels elsewhere identify historical experiments, not the current release order.
The fresh builder produced a new pinned source/runtime prefix, verified all six
asset archive hashes and reused the verified extracted asset content.

| Reviewed-source check | Result |
| --- | --- |
| Build and self-test | Passed; one real environment reset and rendered twice, with RGB/state validation |
| pytest / Ruff | 298 passed; Ruff passed, including simulator-free config, aggregation, shard and grouped lifecycle checks |
| Build/install shell checks | Bash syntax and ShellCheck passed |
| Normal CLI real-model Quick Test | Five native CloseBlenderLid slots, 900 steps each, 4,500 steps total, 0/5 task successes; execution passed |
| Public SDK integration | Grouped Policy lifecycle and environment profiling use public hooks |

The maintainer requested additional simulator-free tests in the
[PR review](https://github.com/MINT-SJTU/RoboScope-Kit/pull/100#pullrequestreview-5816939401).
Frozen task horizons are now included in the manifest, and EGL cleanup handles
missing or already-freed contexts while preserving an original exception.

The all-task comparison collected 250 paired episodes on historical
b40 source `97de6073c737c15446121519342159189f69e4e2`, using a shared 128-step cap.
Native and Kit each scored 3/250 and executed 32,000 steps, with zero paired
outcome or terminal-step disagreements. Initial integration state and scenario
matched in 250/250; initial observations matched in 245/250 and full recorded
traces in 172/250. One Kit Worker exit with signal 9 required a successful retry.
These results remain attributed to that earlier source.

On the reviewed runtime, targeted Kit/native checks of three tasks covered 15
pairs with zero outcome or terminal-step differences and eight exact traces.
Independent native A/A repeats on those tasks covered another 15 pairs with
zero outcome or terminal-step differences and nine exact traces. Native pixel
variation was reproduced, but attribution of all Kit/native trajectory differences
remains unresolved. The [native report](native-comparison-20261109.md) owns the
per-task table, exact scope, reproduction command and durable evidence link.

The maintainer approved 250 paired rollouts instead of 2,500; the user also
allowed a shorter diagnostic horizon. These checks do not establish full
native-horizon success rates or full trajectory equivalence. Passing CI on the
final PR head remains a separate acceptance requirement; the integration remains
labelled partial.

### Focused correctness follow-up — 2026-11-09

The external follow-up runs source `43df866a3c23e83ee6edade8f68615f3eae1be6c`,
adding a second Kit pass and Composite-Seen `KettleBoiling` /
Composite-Unseen `WeighIngredients` to the three reviewed atomic tasks.
The executable-file proof supports reuse of
the earlier final-runtime atomic Native A/A and Kit A records; 55 additional
episodes complete the focused collection. Both paths retain original RGB/MSAA,
the real FP32/100-step Diffusion Policy, five constructor-seeded slots and the
same 128-step diagnostic cap. These instrumented runs are not speed data.

| Comparison | Pairs | Left / right successes | Outcome / step differences | Initial physics / scenarios exact | Exact traces |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native A/A | 25 | 2 / 2 | 0 / 0 | 25 / 25 | 19 |
| Kit A/A | 25 | 2 / 2 | 0 / 0 | 25 / 25 | 19 |
| Native / Kit, pass A | 25 | 2 / 2 | 0 / 0 | 25 / 25 | 16 |
| Native / Kit, pass B | 25 | 2 / 2 | 0 / 0 | 25 / 25 | 16 |

The initial numerical differences are confined to RGB, with a maximum of one
integer intensity level in all four comparisons. Initial state/language inputs
remain exact. Every action/physics divergence follows an image difference at
an earlier retained boundary; same-boundary ordering would be ambiguous and is
not assigned image-first causality. In Kit A/A, all 22 slots with identical
initial inputs have identical first action chunks. Two changed first action
chunks accompany different Blender pixels; one further changed input yields
identical actions. No action/physics error tolerance is adopted. The
[focused report](correctness-followup-20261109.md) records per-slot outcomes,
phase/step attribution, numerical deltas, commands and the new evidence archive.
These five-task checks do not individually explain all historical 78 trace
differences or establish native-horizon accuracy equivalence.

The current-runtime fixed-state Blender wrist control also reproduces the
rendering variation. Thirty redraws with original four-sample MSAA produce 15
distinct images with maximum intensity difference 1; repeated framebuffer reads
are exact. A separate diagnostic with MSAA disabled produces one identical image
across all redraws. Both conditions have the same initial integration-state hash
and preserve integration state after every draw/read. MSAA remains enabled in
every evaluation and performance run; disabling it is only a rendering diagnosis.

## Corrections and comparison limits

The builder's EGL patches free resources in their owning context and preserve an
interrupted context during cleanup. The b31 multi-slot `group-smoke-01` result is
invalid as a corrected-renderer reference. The b36 sampling patches preserve
counter-region insertion order and pass the seeded generator to fixture stack
bases. They change formerly nondeterministic scene realizations; native comparisons
must use the same corrected runtime and must not be called unmodified-upstream runs.

CloseBlenderLid also exposed native MSAA rendering variation: with fixed physics,
a few RGB channels can differ by one intensity level, changing subsequent actions.
Native A/A repeats reproduce this behavior. MSAA remains enabled; full trajectory
or success-rate equivalence is unresolved. Similar aggregate scores are not used
as evidence of equivalence.

For historical profiling context only, b35 Kettle K=1/G=1 on the same machine
recorded 86.465 s in native steps/observations, 39.198 s in Policy round trips,
27.833 s in reset and 30.395 s in construction. That five-slot run had 2/5 successes
and 2,250 steps. It is not a b36 speed or accuracy comparison. Nsight and independent
Policy Server GPU attribution remain unverified.

## Reproduction and external evidence

Build/install instructions remain in the [benchmark README](README.md#build-and-run)
and [Policy README](../../examples/robocasa365_diffusion/README.md). On the validated host:

```bash
WORK=/root/autodl-tmp/robocasa365-20261107
EVIDENCE=/autodl-fs/data/robocasa365/acceptance-20261107
PYTHONPATH="$WORK/sdk-acceptance36" "$WORK/control-acceptance36/bin/python" \
  "$EVIDENCE/startup50-v36/harness/run_startup_smoke.py" \
  --output "$EVIDENCE/startup50-repeat"
```

Use a new output directory. `startup50-v36/` contains `inputs.json`, `command.json`,
`policy-capabilities.json`, `tasks.jsonl`, `summary.json`, `coverage-check.json`,
worker/Policy logs, resource samples and the exact harness with checksums. These
are also included in the **b36 startup evidence** attachment on the existing
[integration evidence release](https://github.com/QLL-CCDD/RoboScope-Kit/releases/tag/robocasa365-issue97-evidence-20261107).
The archive includes per-file hashes, supporting pilot records and the PR refresh
checks/source proof; the separate verification JSON records its SHA256. The older
b35 attachment is historical evidence. No weights, logs, captures or bulk task
result lists are added to Git.
The final source recheck logs are `startup50-v36/pytest.log` and `ruff.log`.

The same evidence root retains `build36.log`, `pytest36-final.log`,
`ruff36-final.log`, `quick-test36-02/`, `repeatability36.json`, and the first-batch
comparisons under `full50x50-v36-original-rendering/first-batch-comparisons/`.
`native-blender-repeatability.json` and `render-repeat-*` retain the rendering diagnosis.
The cancelled full run records its cancellation separately and is not merged into
startup coverage. That collection was not resumed; later reduced runs have
separate source and evidence records.
