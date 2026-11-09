# Approved RoboCasa365 all-task native comparison, 2026-11-09

All 50 frozen target tasks completed five episodes each in two fresh native
passes and the current serial Kit evaluator. Each run has **250 unique terminal
records, 3/250 successes and 32,000 actual control steps**. Every paired success
and terminal step agrees, including the identities of the three successful slots.
This is the approved **250-episode, 128-step-cap comparison** for PR #100.

The maintainer [approved five rollouts per task](https://github.com/MINT-SJTU/RoboScope-Kit/pull/100#issuecomment-6718404495)
instead of the default 50; the user approved the common 128-step cap for this
round. The external harness applies that cap equally. The committed formal
50-rollout default and v1.0.1 task horizons remain unchanged. These results are
not native-horizon accuracy, coverage of all 365 tasks, or a full-trajectory
bit-equality claim. Neither 2,500 episodes nor perfect trajectory hashes is an
additional gate for this approved round.

## Sources, model and execution conditions

Collection identifies submitted integration head
`0b876848d10f10536a457e1bb499e01a3b9f4532`, version **0.8.0b34**, with runnable
snapshot `43df866a3c23e83ee6edade8f68615f3eae1be6c`. The source proof checks all
127 runnable SDK/test/benchmark/model files and pins against the tested tree.
Subsequent documentation-only heads require their own correspondence proof;
measurements are not relabelled as a different executable revision. The
[historical 250-pair collection](native-comparison-20261109.md) at `97de607`
and [focused five-task controls](correctness-followup-20261109.md) remain separate.
No historical episodes are reused in these two fresh native passes.

| Input | Frozen value |
| --- | --- |
| RoboCasa / robosuite | `3443cdf3dd20f9ae8ed41adba442059b506891d0` (v1.0.1) / `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` |
| Native Diffusion Policy evaluator | `41212698b6f481ed92a55f0d5f1778ec56bea417` |
| robomimic / checkpoint repository | `7dc18461c4fc497095f1044f4c40cd44fff62e6a` / `c484448aba1a9b60a04c9b0ca117241518ea69f3` |
| Model | Official human300 epoch0500, EMA settings from the checkpoint, FP32, 100 DDPM steps |
| Checkpoint SHA256 | `e19011f1db25ca75d5047b5b0ab749592d48005e7222c542faadc0c6df88424c` |
| CLIP weight SHA256 | `a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb` |
| Scenes / constructor streams | `pretrain`; seeds 1111111–1111115; first stream episode of each slot |
| Observations / actions | Original full RGB and original four-sample MSAA; last two consecutive step histories; native eight-action chunk and conversions |
| Grouping / scoring | Five native slots per task; native per-slot success accumulation and shared group termination |
| Concurrency | Native: one task/vector group at a time; Kit serial: K=2 Workers / G=1 Policy replica |
| Output / CUDA MPS | Video off / MPS on for every path |
| Hardware | RTX 4090 reporting 49,140 MiB, driver 580.105.08; Xeon Gold 6459C, 12-core quota, 96 GiB memory limit |

The clean builder-produced runtime includes two EGL ownership/restoration fixes
and two sampling fixes: stable XML counter-region order and the environment RNG
for fixture stack bases. Both native and Kit use this same corrected runtime.
This is an official-evaluator comparison on a corrected pinned simulator, not
an unmodified-upstream comparison. The source and asset pins, original/patch/
resulting-file hashes, model hashes and runtime package inventory accompany the
records. A CPU-only provenance audit also verifies that the native bootstrap's
eight core simulator dependencies match the reviewed runtime byte for byte
(6,599 payload files); RoboCasa and robosuite resolve to the reviewed sources.

Native A, Native B and the SDK arms run sequentially under one exclusive GPU
lease. CPU-only provenance/storage maintenance overlaps some native resource
samples, which are tagged in `side-load-analysis.json`. The nine last-created
native task output directories use local storage after shared file quota pressure;
all four SDK arms use the same local output backing before any SDK command starts.
`output-storage-routing.json` records the exact mappings. Models and assets are
unchanged. Native subprocess wall times are not used as a Kit speedup denominator.
Performance and full resource analysis belong to dependent
[PR #107](https://github.com/MINT-SJTU/RoboScope-Kit/pull/107).

## Coverage and paired outcomes

| Run | Tasks | Expected / unique episodes | Successes | Actual steps | Missing / duplicate / unknown | Failed attempts / retries / timeouts |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Native A | 50 | 250 / 250 | 3/250 (1.2%) | 32,000 | 0 / 0 / 0 | 0 / 0 / 0 |
| Native B | 50 | 250 / 250 | 3/250 (1.2%) | 32,000 | 0 / 0 / 0 | 0 / 0 / 0 |
| Kit fixed serial | 50 | 250 / 250 | 3/250 (1.2%) | 32,000 | 0 / 0 / 0 | 0 / 0 / 0 |

Each record is keyed by `(task, slot)`, with slot and episode index 0–4,
constructor seed `1111111 + slot`, stream episode zero, and rollout label
`100000 + slot`. Validation rejects missing, duplicate, unknown or mismatched
identities. The native evaluator computes its own scores; its rollout labels
are not substituted for constructor seeds.

| Split | Tasks / episodes per run | Native A success | Native B success | Kit serial success | Steps per run |
| --- | ---: | ---: | ---: | ---: | ---: |
| Atomic-Seen | 18 / 90 | 3/90 (3.33%) | 3/90 (3.33%) | 3/90 (3.33%) | 11,520 |
| Composite-Seen | 16 / 80 | 0/80 (0%) | 0/80 (0%) | 0/80 (0%) | 10,240 |
| Composite-Unseen | 16 / 80 | 0/80 (0%) | 0/80 (0%) | 0/80 (0%) | 10,240 |
| All tasks | 50 / 250 | 3/250 (1.2%) | 3/250 (1.2%) | 3/250 (1.2%) | 32,000 |

| Pair | Episodes | Success disagreements | Actual terminal-step disagreements | Initial full integration / scenario disagreements | Initial observation hash disagreements |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native A / Native B | 250 | 0 | 0 | 0 / 0 | 4 |
| Native A / Kit fixed serial | 250 | 0 | 0 | 0 / 0 | 5 |
| Native B / Kit fixed serial | 250 | 0 | 0 | 0 / 0 | 4 |

An independently tuned serial run also chooses K=2/G=1 and completes the same
250 identities, success vectors and 32,000 steps, with zero paired outcome or
step differences from fixed serial. This supplies a second current-source Kit
outcome repeat. Its timing and tuning details remain in PR #107.

The predefined rule flags **every success or actual-step disagreement** for
focused replay. None occurs. No numerical tolerance or significance threshold
is chosen after inspecting these scores. Five scenarios per task and a shortened
cap do not establish arbitrary statistical equivalence. The native score output
does not expose a distinct terminal reason, so it remains `null`; Kit reports
`horizon` for all these capped records. This report compares actual terminal
steps, not an invented native termination-reason field.

## All 50 task results

Success vectors follow slots 0–4; `1` means success and each task rate is the
count divided by five. Every slot in every listed run executes exactly 128
control steps. Thus every task has 640 steps per run, including the already
successful slots that continue to the native group barrier.

| Task | Split | Native A count (vector) | Native B count (vector) | Kit serial count (vector) | A / B / Kit steps |
| --- | --- | --- | --- | --- | --- |
| `CloseBlenderLid` | `atomic_seen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `CloseFridge` | `atomic_seen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `CloseToasterOvenDoor` | `atomic_seen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `CoffeeSetupMug` | `atomic_seen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `NavigateKitchen` | `atomic_seen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `OpenCabinet` | `atomic_seen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `OpenDrawer` | `atomic_seen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `OpenStandMixerHead` | `atomic_seen` | 2/5 (`00011`) | 2/5 (`00011`) | 2/5 (`00011`) | 640 / 640 / 640 |
| `PickPlaceCounterToCabinet` | `atomic_seen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `PickPlaceCounterToStove` | `atomic_seen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `PickPlaceDrawerToCounter` | `atomic_seen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `PickPlaceSinkToCounter` | `atomic_seen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `PickPlaceToasterToCounter` | `atomic_seen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `SlideDishwasherRack` | `atomic_seen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `TurnOffStove` | `atomic_seen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `TurnOnElectricKettle` | `atomic_seen` | 1/5 (`00100`) | 1/5 (`00100`) | 1/5 (`00100`) | 640 / 640 / 640 |
| `TurnOnMicrowave` | `atomic_seen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `TurnOnSinkFaucet` | `atomic_seen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `DeliverStraw` | `composite_seen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `GetToastedBread` | `composite_seen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `KettleBoiling` | `composite_seen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `LoadDishwasher` | `composite_seen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `PackIdenticalLunches` | `composite_seen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `PreSoakPan` | `composite_seen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `PrepareCoffee` | `composite_seen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `RinseSinkBasin` | `composite_seen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `ScrubCuttingBoard` | `composite_seen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `SearingMeat` | `composite_seen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `SetUpCuttingStation` | `composite_seen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `StackBowlsCabinet` | `composite_seen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `SteamInMicrowave` | `composite_seen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `StirVegetables` | `composite_seen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `StoreLeftoversInBowl` | `composite_seen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `WashLettuce` | `composite_seen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `ArrangeBreadBasket` | `composite_unseen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `ArrangeTea` | `composite_unseen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `BreadSelection` | `composite_unseen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `CategorizeCondiments` | `composite_unseen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `CuttingToolSelection` | `composite_unseen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `GarnishPancake` | `composite_unseen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `GatherTableware` | `composite_unseen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `HeatKebabSandwich` | `composite_unseen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `MakeIceLemonade` | `composite_unseen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `PanTransfer` | `composite_unseen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `PortionHotDogs` | `composite_unseen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `RecycleBottlesByType` | `composite_unseen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `SeparateFreezerRack` | `composite_unseen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `WaffleReheat` | `composite_unseen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `WashFruitColander` | `composite_unseen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |
| `WeighIngredients` | `composite_unseen` | 0/5 (`00000`) | 0/5 (`00000`) | 0/5 (`00000`) | 640 / 640 / 640 |

Only `OpenStandMixerHead` slots 3/4 and `TurnOnElectricKettle` slot 2 succeed,
with the same identities on every path. A larger official horizon may change
these outcomes; the low capped score must not be presented as the checkpoint's
official multi-task success rate.

## Image and trajectory diagnostics

Initial `mjSTATE_INTEGRATION`, scenario metadata and the ordinary initial
physics fingerprint agree in all 250 pairs above. The external full-state hash
uses float64 `mj_getState` without stepping, rendering or consuming RNG; the
same hook runs in SDK spawned imports. All fingerprints are present.

The two native passes have **208/250 exact recorded traces**. Of the 42 first
divergences, four appear at reset and 38 at a retained chunk boundary; the
first differing fields are observation/feature hashes. Initial observation
hash agreement is 246/250. This demonstrates native image/trajectory variation
with exact initial integration/scenario state in this sample. Kit timing runs
omit heavy step tracing, so this collection does not claim complete Native/Kit
trajectory equality.

The [earlier numerical controls](correctness-followup-20261109.md#fixed-state-renderer-control)
reproduce low-bit MSAA redraw variation at fixed physics state. They support a
rendering explanation, but are not a new numerical audit of each of these 42
native trace differences or each Kit input-hash difference. No RGB, action or
physics tolerance is adopted, and MSAA is never disabled for evaluation.
Equal terminal results establish the reported paired agreement within the
approved cap, not full-horizon or bitwise trajectory equivalence.

## Reproduction and evidence

The frozen harness and complete per-attempt `command.json`, Policy config,
manifest, environment settings, records, logs and checksums are in the
[approved all-task evidence release](https://github.com/QLL-CCDD/RoboScope-Kit/releases/tag/robocasa365-approved250-evidence-20261109).
No weights or installed assets are included. Rebuild/download the checksum-pinned
runtime and model first using the benchmark and Policy READMEs. Use new output
roots when repeating; the exact original paths are retained in the records.

The native driver runs both fresh passes:

```bash
W=/root/autodl-tmp/robocasa365-20261107
E=/autodl-fs/data/robocasa365/acceptance-20261107
B="$W/followup-native-src"
BO="$W/followup-native-overlay"
C="$W/control-reviewed-native/bin/python"
R="$W/runtime-reviewed-native"
H="$W/full250-approved"
PYTHONPATH="$B/packages/roboscope-kit/src" "$C" -B "$H/native.py" \
  --root "$E/approved250-native-repeat" --source "$B" \
  --commit 0b876848d10f10536a457e1bb499e01a3b9f4532 --runtime "$R"
```

Replay the serial Kit path with the frozen per-arm manifest and Policy config.
The entry helper applies the common cap and read-only initial-state hook:

```bash
A="$E/full250-approved-20261109/kit/fixed-serial/attempt-001"
OUT="$E/approved250-kit-serial-repeat"
export PYTHONPATH="$B/packages/roboscope-kit/src:$BO"
export PYTHONHASHSEED=42 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export ROBOSCOPE_POLICY_ENCODING=json
export RC_SLOT_WORKLOAD="$("$C" -c 'import json; from roboscope_kit.runtime.benchmarks.robocasa365.protocol import SUITES; print(json.dumps({"tasks":list(SUITES["target50"]),"cap":128}))')"
export RC_SLOT_MPS_OWNERSHIP="$OUT/mps-ownership.jsonl"
"$C" -B "$H/entry.py" --benchmark robocasa365 --suite target50 --episodes 5 \
  --benchmark-manifest "$A/manifest.json" --policy-config "$A/policy.json" \
  --checkpoint-source /autodl-fs/data/robocasa365/models/official-dp \
  --output-dir "$OUT" --concurrency 2 --replicas 1 --cuda-mps
```

The complete six-run orchestration uses SDK configuration SHA256
`5e64c67750c4a475e266cbc92ad58cf74f417d1eb27bf14bd8262aa83d75c7bf`
and scenario-plan SHA256
`082a66a08f5a066f4903419f381796b811dff88a6121391233f13e5750fc8aab`.
`native/summary.json`, `kit/native-comparison.json`, `kit/comparison.json`,
the source/runtime/model proofs and the per-file archive manifest preserve
source correspondence and all paired records. The comparison script is run as:

```bash
"$C" -B "$H/compare_native.py" "$E/full250-approved-20261109/kit" \
  "$E/full250-approved-20261109/native"
```

This report completes the approved all-task capped collection and native
comparison for the recorded executable files. The fresh runtime double-reset/
render self-test, real five-slot Quick Test, 298 pytest checks, Ruff and shell
checks remain attributed in the [build/review record](validation-20261107.md).
The specifically requested simulator-free tests are retained. Passing exact-head
CI remains an independent delivery gate; `action_required` is not green CI.
The PR remains Draft until that external gate is satisfied. Accuracy beyond
the cap and complete trajectory equality are measurement limits, not newly
introduced requirements for this approved round.
