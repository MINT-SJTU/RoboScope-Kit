# Reduced RoboCasa365 target50 native comparison, 2026-11-09

This paired diagnostic covers all 50 target50 tasks and one five-slot group per task.
Both Kit and the pinned upstream `eval_robocasa.py` use the official human300
Diffusion Policy, pretrain scenes, constructor seeds 1111111–1111115, the original
RGB observations and a shared 128-step cap. The upstream evaluator still creates
its native vector environment and computes its own rewards and scores. The harness
changes the diagnostic horizon and disables video on both paths. The simulator
uses the same builder-produced EGL and seeded scene fixes on both paths.

Kit collection used `97de6073c737c15446121519342159189f69e4e2` (historical 0.8.0b40).
The reviewed integration subsequently adds grouped public profiling calls,
tests, frozen manifest task horizons and guards for already-freed EGL contexts.
The numerical Policy and physical step loop are retained. Fresh-build and
targeted real-simulator checks cover that later source. The 250-pair result
below remains attributed to its recorded source. The builder verified the six
asset archive checksums and reused extracted content in a fresh runtime prefix.
The host was an Intel Xeon Gold 6459C with a 12-core quota, a 96 GiB container
limit and one RTX 4090 with 49,140 MiB VRAM and driver 580.105.08. The two paths
shared the host during collection, so their times are not performance data.

## Paired results

Both paths completed 250 unique episodes. Native scored
3/250 and Kit scored 3/250.
Native executed 32,000 steps and Kit executed
32,000. There were 0 paired
success disagreements and 0 terminal-step
disagreements. The native output does not expose a separate terminal reason.
Initial integration state matched in 250/250,
scenario metadata in 250/250, and raw initial
observations in 245/250. All recorded
trace fields matched in 172/250. The trace includes
physical state and observation and action hashes at action-chunk boundaries.
Earlier repeated native Blender runs on this host also showed wrist pixel
variation. Current divergences remain visible in the paired evidence.

One Kit Worker exited with signal 9 while starting PickPlaceCounterToStove.
The SDK requeued the item and the rerun completed all five episodes. The raw
incomplete trace and recovery event are retained. No OOM was classified by the
SDK. Every native task completed on its first attempt. Of the 78 episodes with
a trace difference, 71 later differed in action and physics hashes. Their first
recorded divergence was an image or observation hash. Thirteen tasks had exact
traces in all five slots. Equal terminal scores cover the shared prefix only.

| Split | Pairs | Native success | Kit success | Outcome differences | Step differences |
| --- | ---: | ---: | ---: | ---: | ---: |
| atomic_seen | 90 | 3 | 3 | 0 | 0 |
| composite_seen | 80 | 0 | 0 | 0 | 0 |
| composite_unseen | 80 | 0 | 0 | 0 | 0 |

Initial observation fields with differences by episode count are `images.agentview_left` (2), `images.wrist` (3).

| Task | Native success | Kit success | Native / Kit steps | Initial obs exact | Trace exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| `CloseBlenderLid` | 0/5 | 0/5 | 640 / 640 | 2/5 | 1/5 |
| `CloseFridge` | 0/5 | 0/5 | 640 / 640 | 5/5 | 0/5 |
| `CloseToasterOvenDoor` | 0/5 | 0/5 | 640 / 640 | 5/5 | 3/5 |
| `CoffeeSetupMug` | 0/5 | 0/5 | 640 / 640 | 5/5 | 5/5 |
| `NavigateKitchen` | 0/5 | 0/5 | 640 / 640 | 5/5 | 3/5 |
| `OpenCabinet` | 0/5 | 0/5 | 640 / 640 | 5/5 | 4/5 |
| `OpenDrawer` | 0/5 | 0/5 | 640 / 640 | 5/5 | 3/5 |
| `OpenStandMixerHead` | 2/5 | 2/5 | 640 / 640 | 5/5 | 5/5 |
| `PickPlaceCounterToCabinet` | 0/5 | 0/5 | 640 / 640 | 5/5 | 3/5 |
| `PickPlaceCounterToStove` | 0/5 | 0/5 | 640 / 640 | 5/5 | 4/5 |
| `PickPlaceDrawerToCounter` | 0/5 | 0/5 | 640 / 640 | 5/5 | 4/5 |
| `PickPlaceSinkToCounter` | 0/5 | 0/5 | 640 / 640 | 5/5 | 4/5 |
| `PickPlaceToasterToCounter` | 0/5 | 0/5 | 640 / 640 | 5/5 | 4/5 |
| `SlideDishwasherRack` | 0/5 | 0/5 | 640 / 640 | 4/5 | 3/5 |
| `TurnOffStove` | 0/5 | 0/5 | 640 / 640 | 5/5 | 5/5 |
| `TurnOnElectricKettle` | 1/5 | 1/5 | 640 / 640 | 5/5 | 5/5 |
| `TurnOnMicrowave` | 0/5 | 0/5 | 640 / 640 | 5/5 | 2/5 |
| `TurnOnSinkFaucet` | 0/5 | 0/5 | 640 / 640 | 5/5 | 5/5 |
| `DeliverStraw` | 0/5 | 0/5 | 640 / 640 | 5/5 | 3/5 |
| `GetToastedBread` | 0/5 | 0/5 | 640 / 640 | 5/5 | 5/5 |
| `KettleBoiling` | 0/5 | 0/5 | 640 / 640 | 5/5 | 3/5 |
| `LoadDishwasher` | 0/5 | 0/5 | 640 / 640 | 5/5 | 5/5 |
| `PackIdenticalLunches` | 0/5 | 0/5 | 640 / 640 | 4/5 | 0/5 |
| `PreSoakPan` | 0/5 | 0/5 | 640 / 640 | 5/5 | 5/5 |
| `PrepareCoffee` | 0/5 | 0/5 | 640 / 640 | 5/5 | 1/5 |
| `RinseSinkBasin` | 0/5 | 0/5 | 640 / 640 | 5/5 | 3/5 |
| `ScrubCuttingBoard` | 0/5 | 0/5 | 640 / 640 | 5/5 | 4/5 |
| `SearingMeat` | 0/5 | 0/5 | 640 / 640 | 5/5 | 5/5 |
| `SetUpCuttingStation` | 0/5 | 0/5 | 640 / 640 | 5/5 | 3/5 |
| `StackBowlsCabinet` | 0/5 | 0/5 | 640 / 640 | 5/5 | 3/5 |
| `SteamInMicrowave` | 0/5 | 0/5 | 640 / 640 | 5/5 | 5/5 |
| `StirVegetables` | 0/5 | 0/5 | 640 / 640 | 5/5 | 4/5 |
| `StoreLeftoversInBowl` | 0/5 | 0/5 | 640 / 640 | 5/5 | 3/5 |
| `WashLettuce` | 0/5 | 0/5 | 640 / 640 | 5/5 | 4/5 |
| `ArrangeBreadBasket` | 0/5 | 0/5 | 640 / 640 | 5/5 | 2/5 |
| `ArrangeTea` | 0/5 | 0/5 | 640 / 640 | 5/5 | 4/5 |
| `BreadSelection` | 0/5 | 0/5 | 640 / 640 | 5/5 | 5/5 |
| `CategorizeCondiments` | 0/5 | 0/5 | 640 / 640 | 5/5 | 3/5 |
| `CuttingToolSelection` | 0/5 | 0/5 | 640 / 640 | 5/5 | 4/5 |
| `GarnishPancake` | 0/5 | 0/5 | 640 / 640 | 5/5 | 4/5 |
| `GatherTableware` | 0/5 | 0/5 | 640 / 640 | 5/5 | 3/5 |
| `HeatKebabSandwich` | 0/5 | 0/5 | 640 / 640 | 5/5 | 2/5 |
| `MakeIceLemonade` | 0/5 | 0/5 | 640 / 640 | 5/5 | 5/5 |
| `PanTransfer` | 0/5 | 0/5 | 640 / 640 | 5/5 | 3/5 |
| `PortionHotDogs` | 0/5 | 0/5 | 640 / 640 | 5/5 | 2/5 |
| `RecycleBottlesByType` | 0/5 | 0/5 | 640 / 640 | 5/5 | 5/5 |
| `SeparateFreezerRack` | 0/5 | 0/5 | 640 / 640 | 5/5 | 1/5 |
| `WaffleReheat` | 0/5 | 0/5 | 640 / 640 | 5/5 | 2/5 |
| `WashFruitColander` | 0/5 | 0/5 | 640 / 640 | 5/5 | 4/5 |
| `WeighIngredients` | 0/5 | 0/5 | 640 / 640 | 5/5 | 4/5 |

## Native repeatability after the EGL cleanup review

The final guarded-EGL runtime ran CloseBlenderLid, OpenStandMixerHead and
SlideDishwasherRack twice, each with the same five constructor seeds and
128-step cap. The 15 pairs scored 2/15 and 2/15,
with 0 outcome differences and 0
terminal-step differences. 9/15 traces were exact.
All initial physical states and scenarios matched. Three Blender wrist
images differed initially in 9, 139 and 3 RGB channels out of 196,608,
each by one integer intensity level. The native path therefore also has
pixel variation. Attribution of all Kit/native differences and native-horizon
success rates remains open.

A targeted rerun on the reviewed integration paired the same three tasks
against the first native pass. All 15 initial integration states and scenarios
matched, with zero success or terminal-step disagreements. Eight traces were
exact. These checks exercise the reviewed runtime separately from the earlier
250-pair collection.

## Reproduction and acceptance

Extract the harness from the evidence archive on the validation host and use
the recorded checkout, simulator, Policy and native dependency overlay.

```bash
W=/root/autodl-tmp/robocasa365-20261107
S=/autodl-fs/data/robocasa365/acceptance-20261107
K="$W/pr100-refresh-acceptance-src"
PYTHONPATH="$K/packages/roboscope-kit/src" "$W/control-pr100-refresh/bin/python" -B \
  "$W/pr100-paired250-b40/run.py" --root "$S/pr100-paired128-target50-repeat" \
  --source "$K" --commit 97de6073c737c15446121519342159189f69e4e2 \
  --runtime "$W/runtime-pr100-refresh" --cap 128 --native-jobs 2 --kit-workers 2 --kit-replicas 2
```

The fresh build, render/reset self-test, real five-slot Quick Test, 298
pytest checks, Ruff, Bash syntax, and ShellCheck passed on reviewed b34
`04cdd858a548aba13ef63a21f06f6301a58a0a2d`.
The reduced task coverage and upstream pairing are complete for this cap.
The maintainer approved 250 paired rollouts in place of the original 2,500.
This run also uses the user-requested shorter horizon. Native-horizon success
rates, review of observation and trajectory differences, and passing PR-head CI
remain outside the completed diagnostic coverage.
[Raw results, traces and harness](https://github.com/QLL-CCDD/RoboScope-Kit/releases/tag/robocasa365-review-evidence-20261109) · [Issue #97](https://github.com/MINT-SJTU/RoboScope-Kit/issues/97)
