# RoboCasa365 focused native follow-up, 2026-11-09

The current integration executable snapshot is `43df866a3c23e83ee6edade8f68615f3eae1be6c`.
This diagnostic extends the earlier reviewed-runtime atomic checks with
one Composite-Seen and one Composite-Unseen task and a second Kit pass.
It retains original RGB/MSAA, the official human300 Diffusion Policy, FP32,
100 denoising steps, pretrain scenes and constructor seeds 1111111–1111115.
Every path uses the same 128-step diagnostic cap, five native slots and
original success/termination rules. Kit uses K=1/G=1. Runs are sequential
with no timing claim; video and MPS are off.

The five tasks are `CloseBlenderLid`, `OpenStandMixerHead`, `SlideDishwasherRack`, `KettleBoiling`, `WeighIngredients`.
The existing final-runtime atomic Native A/A and Kit A traces are reused
only with the recorded executable-file correspondence proof. New collection
adds 55 episodes: two native passes for the composite tasks (20), their
first Kit pass (10), and a second Kit pass for all five tasks (25).
Each comparison below pairs the same 25 task/slot identities; the four
comparisons are not 100 independent scenario identities.

The [earlier all-task report](native-comparison-20261109.md) retains the
250-pair collection at historical source `97de607`. The maintainer approved
50×5=250 rollouts in place of the default 2,500. This follow-up is focused
current-code evidence and does not relabel the historical all-task source.

## Paired results

| Comparison | Pairs | Left / right successes | Outcome differences | Terminal-step differences | Initial physics / scenario exact | Initial observation exact | Exact traces |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native A/A | 25 | 2 / 2 | 0 | 0 | 25 / 25 | 22 | 19 |
| Kit A/A | 25 | 2 / 2 | 0 | 0 | 25 / 25 | 22 | 19 |
| Native / Kit, pass A | 25 | 2 / 2 | 0 | 0 | 25 / 25 | 20 | 16 |
| Native / Kit, pass B | 25 | 2 / 2 | 0 | 0 | 25 / 25 | 21 | 16 |

Actual left/right control-step totals per comparison: 3,200 / 3,200.


| Task | Native A | Native B | Kit A | Kit B |
| --- | --- | --- | --- | --- |
| `CloseBlenderLid` | `00000` | `00000` | `00000` | `00000` |
| `OpenStandMixerHead` | `00011` | `00011` | `00011` | `00011` |
| `SlideDishwasherRack` | `00000` | `00000` | `00000` | `00000` |
| `KettleBoiling` | `00000` | `00000` | `00000` | `00000` |
| `WeighIngredients` | `00000` | `00000` | `00000` | `00000` |

Success vectors follow the five constructor-seed slots. Actual per-slot
step counts, first divergences and raw official/Kit scores are in the archive.

## Difference attribution and numerical scope

| Comparison | Initial RGB maximum intensity delta | Changed RGB channels in two-frame input | Initial non-image differences | Image before changed action/physics | Image only | Same-boundary ambiguous | Action or physics trace differences |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native A/A | 1 | 302 | 0 | 5 | 1 | 0 | 5 |
| Kit A/A | 1 | 34 | 0 | 5 | 1 | 0 | 5 |
| Native / Kit, pass A | 1 | 112 | 0 | 8 | 1 | 0 | 8 |
| Native / Kit, pass B | 1 | 336 | 0 | 8 | 1 | 0 | 8 |

Initial raw arrays are canonicalized to the exact SDK input dtypes and
two-frame histories; their hashes are checked against each recorded reset.
The channel count includes both identical history frames, so a changed
single-frame channel is counted twice. Initial state and language inputs
are checked separately from RGB. The trace audit records the phase and step
of the first image, action and integration-state difference. Observations
and integration states are sampled after a chunk; each action hash covers
the already-executed block ending at that step. Same-boundary differences
cannot establish ordering within a chunk and are labelled ambiguous.
Image-first ordering at an earlier retained boundary is compatible with
pixel differences propagating into
subsequent Policy actions and physics. Native A/A establishes that this can
occur without the Kit adapter. It does not excuse every historical difference
or prove accuracy equivalence.

The numerical and ordering checks above cover these five tasks only.
They do not individually attribute all 78 historical trace differences
and do not replace an all-task run on a different executable revision.

The saved Kit first inference batches also permit a numerical A/A action
check, independently of the converted-action trace hashes. These action
deltas describe variation and introduce no accepted action tolerance.

| Task, Kit A/A | Exact first action chunks | Maximum absolute first-action delta | Different actions on identical initial inputs |
| --- | ---: | ---: | ---: |
| `CloseBlenderLid` | 3/5 | 9.43243504e-05 | 0 |
| `OpenStandMixerHead` | 5/5 | 0 | 0 |
| `SlideDishwasherRack` | 5/5 | 0 | 0 |
| `KettleBoiling` | 5/5 | 0 | 0 |
| `WeighIngredients` | 5/5 | 0 | 0 |

Any identical-input action disagreement remains an inference/RNG
repeatability question and is not attributed to RGB variation.

### Fixed-state renderer control

The current runtime redraws the CloseBlenderLid wrist camera 30 times
at fixed integration state, and separately reads without redrawing.
Both conditions start with the same integration-state hash and preserve
that state after every draw/read.

| Diagnostic | MSAA samples | Read-only distinct images | Redraw distinct images | Redraw maximum intensity delta | Integration state unchanged |
| --- | ---: | ---: | ---: | ---: | --- |
| Original MSAA | 4 | 1 | 15 | 1 | True |
| MSAA disabled diagnostically | 0 | 1 | 1 | 0 | True |

This control only diagnoses rendering at fixed physics state. Disabling
MSAA changes observations and is never enabled in evaluation or performance
measurements. The production runtime retains its original rendering settings.

## Acceptance decisions and remaining limits

- Task/slot/seed identities, record coverage, initial integration state/scenario,
  initial state/language inputs, outcomes and actual terminal steps use exact checks.
- RGB is reported both by exact hashes and numerical deltas. The observed native
  A/A envelope describes repeatability in these samples; it is not a general
  accepted tolerance for different Policy inputs.
- No action or physics drift tolerance is adopted. Any such difference keeps
  full trajectory equivalence unverified, even when terminal scores agree.
- The 128-step prefixes do not establish native-horizon success-rate equivalence
  or statistical equivalence. The integration remains partial.

## Reproduction

```bash
W=/root/autodl-tmp/robocasa365-20261107
E=/autodl-fs/data/robocasa365/acceptance-20261107
SOURCE="/root/autodl-tmp/robocasa365-20261107/followup-native-src"
PYTHONPATH="$SOURCE/packages/roboscope-kit/src" \
  "$W/control-reviewed-native/bin/python" -B "$W/followup-native/run_followup.py" \
  --root "$E/native-followup-repeat" --source "$SOURCE" \
  --commit 43df866a3c23e83ee6edade8f68615f3eae1be6c --runtime "/root/autodl-tmp/robocasa365-20261107/runtime-reviewed-native" --cap 128
```

Use a new output directory and the frozen earlier atomic trace roots
recorded in the plan. The source correspondence proof is required for reuse.

The renderer controls run in separate subprocesses and do not modify the
installed runtime or evaluation settings:

```bash
R="$W/runtime-reviewed-native"
"$R/env/bin/python" -B "$W/followup-native/render_attribution.py" \
  --output "$E/render-original-repeat"
"$R/env/bin/python" -B "$W/followup-native/render_attribution.py" \
  --output "$E/render-no-msaa-repeat" --disable-msaa
```

[Raw records, numerical audit, frozen harness and source proof](https://github.com/QLL-CCDD/RoboScope-Kit/releases/tag/robocasa365-review-followup-evidence-20261109)
