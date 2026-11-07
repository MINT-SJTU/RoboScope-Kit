# RoboCasa365 integration readiness — 2026-11-07

This is an **integration-first draft** contributing to
[issue #97](https://github.com/MINT-SJTU/RoboScope-Kit/issues/97), not completion
of its acceptance requirements. The full **2,500 rollouts and matched native
evaluator comparison have not been run**. No native-equivalent accuracy,
unchanged success rate or runtime speedup is claimed. GR1 Tabletop and its
[issue #16](https://github.com/MINT-SJTU/RoboScope-Kit/issues/16) remain separate.

This branch contains the isolated benchmark/official Policy connection and the
mandatory EGL cleanup correctness fix. The official trained checkpoint, model
precision, denoising schedule, scenario ordering, resets, action conversion and
scoring rules are unchanged.

## Revisions and machine

| Component | Pin / configuration |
|---|---|
| Integration SDK | 0.8.0b35; branch-specific validation below |
| RoboCasa | 1.0.1, `3443cdf3dd20f9ae8ed41adba442059b506891d0` |
| RoboSuite | 1.5.2, `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` |
| Simulator | Python 3.11.13, MuJoCo 3.3.1, NumPy 2.2.5, CPU torch 2.5.1+cpu |
| Official DP source | `41212698b6f481ed92a55f0d5f1778ec56bea417` |
| RoboCasa robomimic fork | `7dc18461c4fc497095f1044f4c40cd44fff62e6a` |
| Checkpoint | Official `pretrain_human300`, epoch 0500; checkpoint and CLIP hashes in the model lock |
| Inference | Python 3.11.15, torch 2.6.0+cu124, torchvision 0.21.0, NumPy 1.26.4, diffusers 0.11.1, transformers 4.46.3; FP32 and 100 denoising steps |
| Hardware | RTX 4090, reported 49,140 MiB, driver 580.105.08; 12-core CPU quota, 96 GiB memory limit; MPS disabled |

Source/archive/asset inputs are locked in this benchmark's JSON files. Model
sources, weights and dependencies are locked in the
[Policy example](../../examples/robocasa365_diffusion/README.md). Source snapshots
are verified against their archives on repeat installation; edits to installed
simulator sources are rejected unless they match a committed reviewed patch.

## Mandatory EGL correction and historical erratum

Sequential hard resets exposed a renderer ownership error: deleting an old
renderer without binding its EGL context could free another slot's GPU
resources. Matching initial images/physics/first actions did not exclude
corrupted later step images in the earlier slots.

The committed source patches bind the owner before freeing MuJoCo resources,
restore a new renderer after replacement, and preserve the interrupted EGL API,
display, context and read/draw surfaces during delayed destruction. Per-context
cleanup no longer releases another context's thread binding; final thread
release occurs at process exit. Factory self-test and workers require marker 2.
This is a correctness correction, not a rendering-rate or evaluation-rule change.

The old b31 `group-smoke-01` five-slot score and timings are **invalid as a
corrected-renderer reference**. They remain archived for diagnosis, but are not
reused as validation or a performance baseline. A single-slot run and a first
chunk comparison did not exercise the failing five-slot lifecycle.

## Branch-specific validation (b35)

Results in this table must come from this integration branch and its fresh
runtime. Earlier branch results below are not substituted for them.

| Required check | Current integration-branch result |
|---|---|
| Clean simulator build with committed source/asset/patch locks | Passed in a fresh `runtime-integration35` prefix; build completed with `INTEGRATION_B35_READY` |
| Final builder self-test / marker-2 validation | Passed frozen registry/imports and required cleanup marker 2; output `self-test.qJMmIc` |
| SDK wheel installation | Fresh b35 wheel built and installed in separate control and inference overlays |
| SDK source identity | All 85 SDK Python files byte-identical across LF-normalized local integration source, remote integration source and both installed overlays |
| Simulator source repeat check | Frozen sources reverified and both reviewed EGL patch preflights passed |
| Existing pytest and Ruff | Linux: 188 passed in 8.61 s; required source/tests/RoboTwin and RoboCasa365/example Ruff checks passed |
| Bash syntax and ShellCheck | Both build/install shell scripts passed Bash syntax; ShellCheck passed on LF content |
| Real native-five-slot group / original official Policy | Passed: TurnOnElectricKettle, 5 unique records, 2/5 task successes, 2,250 steps, no worker failure/retry/OOM or missing/duplicate record |
| Full target50 / matched native evaluation | Not run; acceptance outstanding |

The b35 runtime prefix is
`/root/autodl-tmp/robocasa365-20261107/runtime-integration35`; its self-test output
is `self-test.qJMmIc` below that prefix. The build log is retained outside Git at
`/autodl-fs/data/robocasa365/integration35/build.log`. The branch's real five-slot
TurnOnElectricKettle run uses a separate schema-2 Policy configuration and the
b35 SDK overlay; its completed result is in
`/autodl-fs/data/robocasa365/integration35/native-five-kettle`.

Wheel SHA256:
`d33845d63be87088b021a83006b589bda63fe3b333f68a6206a8572dd1b26b19`.
`installed-sdk-proof.json` and `build-proof.json` under the same persistent
`integration35` evidence directory record source identity and repeat verification.
These source checks and the completed episode results below are separate evidence.

### Fresh b35 real-group result

All five TurnOnElectricKettle episodes reached their 450-step horizon, producing
five unique records and **2/5 task successes (40%)**. The run completed 2,250
physical steps with zero worker failures, retries, OOMs, missing or duplicate
records. One worker-attempt log, empty recovery failures, no failed queue JSON
and no errors JSONL were independently checked. The original official Policy,
native observations and fixed K=1/G=1 were used with videos disabled.
`full_suite` and `native_equivalence_verified` remain false. The validation
summary is `integration35/native-five-kettle-validation.json`.

| Fresh b35 interval / resource | Measured value |
|---|---:|
| Helper end-to-end | 240.276 s |
| Evaluation | 184.902 s |
| Policy process/startup/load | 52.592 s (cold) |
| Runtime preparation | 0.932 s |
| Worker task | 184.704 s |
| Environment construction | 30.395 s |
| Evaluated reset | 27.833 s |
| Native steps, including controllers/observations/rendering | 86.465 s |
| Policy round trips | 39.198 s |
| Environment close | 0.426 s |
| Video writes | 0 s |
| Sampled mean GPU utilization | 7.23% |
| Peak GPU memory | 6,616 MiB |
| Mean CPU usage | 1.081 cores |

These are descriptive smoke measurements. Worker phases are nested inside task,
evaluation and end-to-end intervals. RPC is not isolated inference time, native
steps are not isolated physics/rendering, and GPU samples combine rendering and
inference. Average GPU memory and isolated GPU kernel times were not measured.

The earlier b33 run's 3/5 and this b35 run's 2/5 are **different smoke runs, not
a matched parity or performance comparison**. They must not be used to infer an
accuracy change or speedup. These samples do not establish native or full-suite
success rates.

## Prior evidence, explicitly separate from b35

The corrected **b33** default five-slot run used TurnOnElectricKettle,
`pretrain`, five constructor streams, the full 450-step horizon, eight-action
chunks, last two physical frames, FP32, 100 denoising steps, K=1/G=1 and no video.
It used the original official Policy. This is prior operational evidence, **not
proof for this new integration branch**.

| Prior corrected b33 run | Result |
|---|---|
| Completed episodes / steps | 5 / 2,250 |
| Task successes | 3/5; a smoke sample, not a benchmark score |
| Worker failures / retries / OOM | 0 / 0 / 0 |
| End-to-end / evaluation | 201.173 s / 187.052 s; descriptive only |
| Policy startup/load / preparation | 11.598 s / 0.832 s |
| Source | `425e9fd5e84f1c63c3e50d64f2bb316a5f483c64`; a different branch/version |

Earlier b31 checks verified the six asset archives by expected sizes, SHA256
and ZIP CRC, built a fresh simulator/inference environment and loaded real
weights. The source installer exposes the pinned DP namespace through a `.pth`
entry, since upstream `find_packages()` omits it. The inference lock includes
the fork's imported `six` dependency. Two Lightwheel ZIPs contain top-level
attribution READMEs; only those explicitly locked metadata entries are excluded
from extraction. All of these installation requirements remain reproducible
from committed build inputs.

For captured real batches of sizes one and five, the earlier native comparison
independently loaded the upstream workspace/EMA model and verified preprocessing,
language embeddings, loader CPU/CUDA RNG and first actions exactly, with maximum
action difference **0.0**. It covers the first chunk only, not complete physics
rollouts or native success rates. `native_equivalence_verified` remains false.

## Profiling limits

Historical **b31 single-slot** CloseBlenderLid wall timers recorded 65.053 s in
Policy round trips and 47.020 s in native simulation/observation/rendering during
a 900-step smoke episode. These are prior measurements, not b35 timings or a
corrected multi-slot baseline. GPU operations were not separately attributed.

Worker timing includes native stepping, controllers, observations and rendering;
Policy RPC includes transfers, preprocessing, inference and waits. GPU samples
combine rendering and inference. Shared group durations repeat in five episode
records and must be counted once. The `--profile` capture starts after group
environment resets and excludes independent Policy Server GPU execution.
Nsight hardware capture and isolated server GPU timing remain unverified. No
speed ratio or kernel-level bottleneck conclusion is drawn from these smoke runs.

## Reproduction and evidence

Build a fresh runtime using the [benchmark instructions](README.md#build-and-run),
install this branch's SDK in both control and inference environments, and use
absolute paths in the schema-2 Policy process configuration. If using isolated SDK overlays, preserve their recorded `PYTHONPATH` in both
the control and Policy process commands. Confirm that each imports b35 rather
than the earlier deployed SDK. The inference
`load()` reads real weights from `SessionContext.checkpoint_path`. The older
adaptation prompt's schema-1 process configuration is incompatible.

From the repository root, choose fresh output paths and the newly built manifest:

```bash
KIT_PYTHON=/root/autodl-tmp/robocasa365-20261107/control-integration35/bin/python
SIM_RUNTIME=/root/autodl-tmp/robocasa365-20261107/runtime-integration35
POLICY_CONFIG=/autodl-fs/data/robocasa365/integration35/evostudio-policy.json
OUT_ROOT="$(mktemp -d /tmp/robocasa365-integration.XXXXXX)"
"$KIT_PYTHON" -c 'from importlib.metadata import version; assert version("roboscope-kit") == "0.8.0b35"'

"$KIT_PYTHON" -m roboscope_kit.runtime.benchmark --benchmark robocasa365 \
  --benchmark-manifest "$SIM_RUNTIME/benchmark-manifest.json" \
  --self-test --output-dir "$OUT_ROOT/self-test"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$KIT_PYTHON" examples/robocasa365_diffusion/run_group_smoke.py \
  --task TurnOnElectricKettle \
  --benchmark-manifest "$SIM_RUNTIME/benchmark-manifest.json" \
  --policy-config "$POLICY_CONFIG" \
  --checkpoint-source /autodl-fs/data/robocasa365/models/official-dp \
  --output-dir "$OUT_ROOT/native-five"
```

The group helper defaults to CloseBlenderLid; the explicit official task above
uses a 450-step horizon. The CLI Quick Test uses one slot, not this five-slot
path. Its K=G=1 is selected automatically. Use full native groups for comparison;
check actual steps, terminal reasons, expected/unique records and worker errors.

Original assets, model weights and build caches stay under the persistent
`/autodl-fs/data/robocasa365` root. Extracted runtime files may live on the
container SSD because of the shared filesystem's file-count quota. Keep raw
logs, captures, source/dependency proofs and results outside Git. Prior evidence
includes `quick-test-04`, `group-smoke-01` (invalidated multi-slot run), native
first-batch comparison JSONs and `default-b33-fast`; these labels must not be
renamed into b35 results. Current-branch artifact paths are recorded with its
completed checks.

## Remaining acceptance

The integration PR must remain **Draft** and reference issue #97 without closing
it until the required full target50 matrix and matched native comparison exist.
Those evaluations must retain source/model/scenario protocol and report complete
per-task/per-split coverage, successes, actual steps, termination and errors.
This smoke-ready implementation does not establish their result. Performance
optimization and any speed evidence are separate follow-up work; GR1 issue #16
is not a RoboCasa365 performance result.
