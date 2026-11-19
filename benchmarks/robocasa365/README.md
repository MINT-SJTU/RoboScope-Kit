# RoboCasa365 runtime

This runtime is independent of `benchmarks/robocasa`, which owns GR-1 Tabletop.
The [protocol](../../packages/roboscope-kit/docs/robocasa365-protocol.md) freezes
the official 50-task multi-task workload in **pretrain scenes**, with 50 rollouts
per task (2,500 records), the v1.0.1 task horizons and native five-slot groups.
It is tracked in [issue #97](https://github.com/MINT-SJTU/RoboScope-Kit/issues/97).

## Build and run

Use Linux x86_64, Python 3.11, uv 0.11.14, Git, NVIDIA EGL libraries and a CUDA
GPU. Install the [official Diffusion Policy example](../../examples/robocasa365_diffusion/README.md)
in its separate inference environment. The simulator uses CPU torch for
imported playback utilities and excludes interactive device and training tools.
Allow space for six asset ZIPs (about 10.99 GB), their extracted files (about
147,000 files: check the filesystem's file-count quota), source checkouts,
environments and model weights.

```bash
python -m pip install ./packages/roboscope-kit
KIT_PYTHON="$(command -v python)" \
  ROBOSCOPE_ROBOCASA365_ASSETS_DIR=/data/robocasa365-downloads \
  bash benchmarks/robocasa365/build-runtime.sh /sim/robocasa365
```

`ROBOSCOPE_ROBOCASA365_PYTHON` can select an existing absolute Python 3.11
executable instead of downloading one; it must remain available after building.
`ROBOSCOPE_PYPI_INDEX` selects a reachable package mirror without changing pins.
`--environment-only` prepares imports without installing a manifest or claiming
a usable asset-backed runtime. Normal builds verify all asset hashes and finish
with the Kit self-test. Downloads use GET: Box may reject HEAD for valid links.

The builder applies three source patches, checking original, patch and result
hashes:

- **EGL cleanup.** Renderer resources are freed in their owning context, and
  delayed destruction restores the interrupted context. Without it, closing or
  hard-resetting one slot breaks rendering in another slot of the same process.
- **Counter regions in XML order** (`robocasa/models/fixtures/counter.py`).
  Upstream iterates a `set()` of regions, whose order follows object identity,
  so the same constructor seed places objects differently in two processes
  ([robocasa#236](https://github.com/robocasa/robocasa/issues/236)).
- **Seeded fixture stack bases** (`robocasa/models/fixtures/fixture_stack.py`).
  Upstream samples the stack base without the environment's generator
  ([robocasa#237](https://github.com/robocasa/robocasa/issues/237)).

The last two are reproducibility corrections: without them two native runs with
the same constructor seeds start from different scenes, and no per-episode
comparison is possible. Results are those of the official evaluator on the
corrected pinned simulator; tasks, seeds and success criteria are unchanged.

`check-scene-repeatability.py` shows the problem. On 2026-11-18 it built six
tasks (PickPlaceCounterToCabinet, WashLettuce, ArrangeTea, StirVegetables,
PrepareCoffee, OpenDrawer) with constructor seed 1111111 in three fresh
processes each, with `PYTHONHASHSEED=42`. On the uncorrected pin, WashLettuce
started from two different simulator states; with the corrections, every task
started from one state and one scenario. Fix `PYTHONHASHSEED` when comparing
scenario metadata digests: with a random hash seed, ArrangeTea's metadata
serializes in a different order although its simulator state is the same.

```bash
python -m roboscope_kit.runtime.benchmark \
  --benchmark robocasa365 \
  --benchmark-manifest /sim/robocasa365/benchmark-manifest.json \
  --self-test --output-dir /results/robocasa365-self-test

python -m roboscope_kit.runtime.benchmark \
  --benchmark robocasa365 --suite target50 \
  --benchmark-manifest /sim/robocasa365/benchmark-manifest.json \
  --policy-config /workspace/robocasa365-policy.json \
  --checkpoint-source /models/robocasa365-diffusion \
  --quick-test --output-dir /results/robocasa365-quick-test
```

The self-test verifies the frozen registry and resets and renders one
environment twice. The Quick Test runs one native five-slot group of the first
task at its official horizon. For the full matrix remove `--quick-test` and set
`--concurrency K --replicas G`; each Worker runs five simulator environments.

## Native comparison

The integration was compared with the pinned official evaluator on 2026-11-18:
target50, five rollouts per task (one native five-slot group, 250 episodes), at the
official task horizons, FP32 with 100 DDPM steps, full RGB, no video. Machine: one
RTX 4090 (driver 595.71.05), Xeon Platinum 8470Q with a 20-core and 72 GiB cgroup
limit. Kit commit `303a376`, CLI `--episodes 5 --concurrency 4 --replicas 1
--cuda-mps`. The [example README](../../examples/robocasa365_diffusion/README.md#native-comparison)
has the commands; `compare_rollouts.py` states the tolerance, fixed before the runs.

| Arm | Successes | atomic_seen | composite_seen | composite_unseen | Steps | Wall time |
|---|---:|---:|---:|---:|---:|---:|
| Native A (official `eval_robocasa.py`) | 15/250 | 14/90 | 1/80 | 0/80 | 452,250 | 17,868 s, beside B |
| Native B (repeat) | 15/250 | 14/90 | 1/80 | 0/80 | 452,250 | 17,900 s, beside A |
| Kit serial, K4/G1 | 14/250 | 14/90 | 0/80 | 0/80 | 452,250 | 10,584 s, alone |

The two native passes agree on all 250 outcomes and step counts, and 147 of their
250 trajectories are bit-identical. Kit and native start every pair from the same
scene and simulator state (250/250 scenario and initial-state digests), execute the
same step counts, and disagree on one outcome: ScrubCuttingBoard, episode 0, succeeds
natively and fails in Kit. As with every other divergence in the comparison, it
begins in rendered images (there, `agentview_left` at step 208), not in state or
actions: a one-level MSAA pixel difference changes the next diffusion action. One
disagreement against none between native passes is within the declared tolerance
(exact binomial p = 1.0; direction p = 1.0). 102 of the 250 Kit trajectories are
bit-identical to native.

No group of this checkpoint has all five slots succeed, so every episode ran to its
horizon (termination `horizon` in all 250 Kit records); the group-success barrier
and `simulator_done` occur in neither path. Horizons that are not a multiple of 8,
such as 450, end with a partial chunk on both paths with equal step counts.

<details>
<summary>Per-task results</summary>

| Task | Split | Horizon | Native A | Native B | Kit serial | Kit/native A disagreements | Steps per arm | Trajectories identical to native A; first divergence |
|---|---|---:|---:|---:|---:|---:|---:|---|
| CloseBlenderLid | atomic_seen | 900 | 0/5 | 0/5 | 0/5 | 0 | 4,500 | 0/5; earliest step 0 |
| CloseFridge | atomic_seen | 900 | 1/5 | 1/5 | 1/5 | 0 | 4,500 | 0/5; earliest step 8 |
| CloseToasterOvenDoor | atomic_seen | 450 | 3/5 | 3/5 | 3/5 | 0 | 2,250 | 1/5; earliest step 8 |
| CoffeeSetupMug | atomic_seen | 600 | 0/5 | 0/5 | 0/5 | 0 | 3,000 | 4/5; earliest step 368 |
| NavigateKitchen | atomic_seen | 450 | 0/5 | 0/5 | 0/5 | 0 | 2,250 | 2/5; earliest step 24 |
| OpenCabinet | atomic_seen | 1050 | 1/5 | 1/5 | 1/5 | 0 | 5,250 | 3/5; earliest step 24 |
| OpenDrawer | atomic_seen | 750 | 0/5 | 0/5 | 0/5 | 0 | 3,750 | 3/5; earliest step 32 |
| OpenStandMixerHead | atomic_seen | 450 | 2/5 | 2/5 | 2/5 | 0 | 2,250 | 3/5; earliest step 64 |
| PickPlaceCounterToCabinet | atomic_seen | 750 | 0/5 | 0/5 | 0/5 | 0 | 3,750 | 2/5; earliest step 16 |
| PickPlaceCounterToStove | atomic_seen | 600 | 0/5 | 0/5 | 0/5 | 0 | 3,000 | 2/5; earliest step 64 |
| PickPlaceDrawerToCounter | atomic_seen | 750 | 0/5 | 0/5 | 0/5 | 0 | 3,750 | 4/5; earliest step 72 |
| PickPlaceSinkToCounter | atomic_seen | 900 | 0/5 | 0/5 | 0/5 | 0 | 4,500 | 4/5; earliest step 144 |
| PickPlaceToasterToCounter | atomic_seen | 600 | 0/5 | 0/5 | 0/5 | 0 | 3,000 | 2/5; earliest step 72 |
| SlideDishwasherRack | atomic_seen | 450 | 2/5 | 2/5 | 2/5 | 0 | 2,250 | 3/5; earliest step 0 |
| TurnOffStove | atomic_seen | 750 | 0/5 | 0/5 | 0/5 | 0 | 3,750 | 4/5; earliest step 560 |
| TurnOnElectricKettle | atomic_seen | 450 | 3/5 | 3/5 | 3/5 | 0 | 2,250 | 4/5; earliest step 256 |
| TurnOnMicrowave | atomic_seen | 450 | 1/5 | 1/5 | 1/5 | 0 | 2,250 | 1/5; earliest step 8 |
| TurnOnSinkFaucet | atomic_seen | 600 | 1/5 | 1/5 | 1/5 | 0 | 3,000 | 5/5 |
| DeliverStraw | composite_seen | 2550 | 0/5 | 0/5 | 0/5 | 0 | 12,750 | 0/5; earliest step 40 |
| GetToastedBread | composite_seen | 3000 | 0/5 | 0/5 | 0/5 | 0 | 15,000 | 2/5; earliest step 16 |
| KettleBoiling | composite_seen | 1500 | 0/5 | 0/5 | 0/5 | 0 | 7,500 | 2/5; earliest step 480 |
| LoadDishwasher | composite_seen | 1800 | 0/5 | 0/5 | 0/5 | 0 | 9,000 | 2/5; earliest step 960 |
| PackIdenticalLunches | composite_seen | 3900 | 0/5 | 0/5 | 0/5 | 0 | 19,500 | 0/5; earliest step 0 |
| PreSoakPan | composite_seen | 2400 | 0/5 | 0/5 | 0/5 | 0 | 12,000 | 4/5; earliest step 936 |
| PrepareCoffee | composite_seen | 1800 | 0/5 | 0/5 | 0/5 | 0 | 9,000 | 2/5; earliest step 120 |
| RinseSinkBasin | composite_seen | 1350 | 0/5 | 0/5 | 0/5 | 0 | 6,750 | 2/5; earliest step 88 |
| ScrubCuttingBoard | composite_seen | 1200 | 1/5 | 1/5 | 0/5 | 1 | 6,000 | 3/5; earliest step 208 |
| SearingMeat | composite_seen | 4350 | 0/5 | 0/5 | 0/5 | 0 | 21,750 | 3/5; earliest step 32 |
| SetUpCuttingStation | composite_seen | 2400 | 0/5 | 0/5 | 0/5 | 0 | 12,000 | 4/5; earliest step 8 |
| StackBowlsCabinet | composite_seen | 2100 | 0/5 | 0/5 | 0/5 | 0 | 10,500 | 2/5; earliest step 8 |
| SteamInMicrowave | composite_seen | 2100 | 0/5 | 0/5 | 0/5 | 0 | 10,500 | 2/5; earliest step 296 |
| StirVegetables | composite_seen | 2400 | 0/5 | 0/5 | 0/5 | 0 | 12,000 | 3/5; earliest step 24 |
| StoreLeftoversInBowl | composite_seen | 2550 | 0/5 | 0/5 | 0/5 | 0 | 12,750 | 0/5; earliest step 24 |
| WashLettuce | composite_seen | 1650 | 0/5 | 0/5 | 0/5 | 0 | 8,250 | 2/5; earliest step 88 |
| ArrangeBreadBasket | composite_unseen | 4350 | 0/5 | 0/5 | 0/5 | 0 | 21,750 | 0/5; earliest step 16 |
| ArrangeTea | composite_unseen | 2250 | 0/5 | 0/5 | 0/5 | 0 | 11,250 | 2/5; earliest step 72 |
| BreadSelection | composite_unseen | 1950 | 0/5 | 0/5 | 0/5 | 0 | 9,750 | 3/5; earliest step 88 |
| CategorizeCondiments | composite_unseen | 1650 | 0/5 | 0/5 | 0/5 | 0 | 8,250 | 1/5; earliest step 48 |
| CuttingToolSelection | composite_unseen | 1200 | 0/5 | 0/5 | 0/5 | 0 | 6,000 | 5/5 |
| GarnishPancake | composite_unseen | 2700 | 0/5 | 0/5 | 0/5 | 0 | 13,500 | 1/5; earliest step 152 |
| GatherTableware | composite_unseen | 2250 | 0/5 | 0/5 | 0/5 | 0 | 11,250 | 2/5; earliest step 8 |
| HeatKebabSandwich | composite_unseen | 2700 | 0/5 | 0/5 | 0/5 | 0 | 13,500 | 0/5; earliest step 8 |
| MakeIceLemonade | composite_unseen | 3000 | 0/5 | 0/5 | 0/5 | 0 | 15,000 | 0/5; earliest step 176 |
| PanTransfer | composite_unseen | 1800 | 0/5 | 0/5 | 0/5 | 0 | 9,000 | 3/5; earliest step 112 |
| PortionHotDogs | composite_unseen | 2250 | 0/5 | 0/5 | 0/5 | 0 | 11,250 | 0/5; earliest step 40 |
| RecycleBottlesByType | composite_unseen | 2850 | 0/5 | 0/5 | 0/5 | 0 | 14,250 | 2/5; earliest step 464 |
| SeparateFreezerRack | composite_unseen | 2400 | 0/5 | 0/5 | 0/5 | 0 | 12,000 | 0/5; earliest step 0 |
| WaffleReheat | composite_unseen | 4050 | 0/5 | 0/5 | 0/5 | 0 | 20,250 | 0/5; earliest step 8 |
| WashFruitColander | composite_unseen | 3150 | 0/5 | 0/5 | 0/5 | 0 | 15,750 | 2/5; earliest step 24 |
| WeighIngredients | composite_unseen | 3000 | 0/5 | 0/5 | 0/5 | 0 | 15,000 | 1/5; earliest step 104 |

</details>
