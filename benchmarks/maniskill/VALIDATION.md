# ManiSkill validation status

**2026-11-13: integration acceptance is incomplete.** The pinned inventory declares
55 evaluation TaskIDs and 77 selectable task/robot/observation/controller profiles.
All original task IDs and tabletop state/RGB configurations remain in scope.
Removing borrowed-task/robot model examples does not remove benchmark tasks.

Acceptance requires a checkpoint trained for the intended task and robot, useful
task competence where success is defined, the prescribed episode count,
independently resolved upstream configuration, matching frozen vector grouping
and results from the final source. Shape compatibility, a five-step prefix or an
SDK/configured-reference match cannot establish these requirements. Return-scored
tasks retain their native metric without fabricated binary success. See the
[maintainer decision](https://github.com/MINT-SJTU/RoboScope-Kit/pull/99#issuecomment-6800683447).

## Retained model scope

| Model family | Selected scope | Acceptance limit |
|---|---|---|
| Official state PPO | Eleven task-trained checkpoints, including the genuine 16D dual-Panda actors | Prescribed-count native comparison matches exactly at `6dcbf53`; rates agree with the original evaluator at the same width ([results](#official-ppo-prescribed-count-results)) |
| Author G1 PPO | TransportBox and PlaceAppleInBowl task/robot checkpoints | Prescribed-count records match the configured native reference ([results](#remaining-model-prescribed-count-results)); author simulator/horizon differences and independent configuration remain unresolved |
| NEWT | Eight task/robot checkpoints: six control tasks, PlaceSphere and PullCubeTool | Prescribed-count records match the configured native reference ([results](#remaining-model-prescribed-count-results)); inference is the deterministic actor prior, not the author's default MPC evaluator, and Hopper and PlaceSphere score well below the published MPC results |
| Bridge Octo-small | Four original WidowX Bridge profiles | Prescribed-count records match the native reference ([results](#remaining-model-prescribed-count-results)); provisional baseline, since a generalist model does not establish per-task training or competence |
| Author ManiSkill Octo | Existing PegInsertionSide/PlugCharger RGB profiles with complete 7D pose actions | The author's published Octo results have zero success; no task acceptance is claimed |

The eleven official PPO tasks, two author G1 tasks and eight NEWT tasks identify
21 TaskIDs with a selected task-trained candidate. The remaining 34 have no
selected intended task/robot checkpoint in this inventory. This is a model
inventory, not accepted task coverage or proof that no other public model exists.
The provisional visual examples are not added to that count. The original 18
tabletop RGB configurations still require compatible visual-policy acceptance.
The [inventory](TASK_INVENTORY.md) and [compatibility notes](REMAINING_COMPATIBILITY.md)
record the exact profiles and model limitations.

## Machine, protocol and source

The historical simulator machine reports an NVIDIA GeForce RTX 4090 with
49,140 MiB VRAM, driver 595.71.05 and a 20-core CPU quota. The simulator source is
`baab60ede2e89167c1b7aaed41a9aa8e690a9d1e`, SAPIEN 3.0.0b1, Python 3.11.15,
Torch 2.6.0/CUDA 12.4 and NumPy 1.26.4. Assets, checkpoint provenance and native
horizons are pinned separately. The [evaluation protocol](../../packages/roboscope-kit/MANISKILL_EVALUATION_PROTOCOL.md)
defines observations, controllers, scores and the frozen ten-lane vector group.

Quick Test uses the first complete ten-episode group, with seeds 0-9. Reduced
episode counts also retain complete groups. Scalar seed equality does not make
older width-one/two records equivalent initial states. Diagnostic self-tests at
other widths are separate from task acceptance.

## Recorded evidence

Only source and scoring/coverage checks are reported here. Historical borrowed
policy runs are excluded from the retained model scope and its acceptance evidence.
Stored-record replay is a CPU regression fixture, not a new model evaluation.

| Check cohort | Source | Elapsed / scope | Limit |
|---|---|---|---|
| Scoring and shard arithmetic | [`fad25d36b5207d1ff47ee4cf8af68721f98241e0`](https://github.com/QLL-CCDD/RoboScope-Kit/commit/fad25d36b5207d1ff47ee4cf8af68721f98241e0) | CPU replay of stored records and corrupted result documents | No simulator, model competence or throughput evidence |
| Source review checks | [`f40fbe816ba2d4198ea799c03b78a016bb3caf1b`](https://github.com/QLL-CCDD/RoboScope-Kit/commit/f40fbe816ba2d4198ea799c03b78a016bb3caf1b) | 270 existing pytest cases: 8.71 s; checks below | Precedes the model-scope cleanup; not final-head acceptance |
| Retained-scope cleanup | Current candidate; source/check hashes bound externally at delivery | 270 existing pytest cases: 7.88 s; eleven-suite CPU factory/quick/reduced checks pass | No new GPU evaluation, runtime build or simulator self-test |
| Official PPO prescribed count | [`6dcbf53dac517f4ab7bf55f3dc4cf042e716da20`](https://github.com/QLL-CCDD/RoboScope-Kit/commit/6dcbf53dac517f4ab7bf55f3dc4cf042e716da20) | 1,100 episodes per side, exact record and per-step trajectory match; 403.8 s evaluation | Eleven official PPO tasks only; see below |

Raw commands, source/check identities and regression fixtures remain outside Git.
CPU regression receipts do not count any fixture model or TaskID toward acceptance.
The [source-review CPU evidence](https://github.com/QLL-CCDD/RoboScope-Kit/releases/tag/maniskill-issue25-review-source-20261112)
is published for `f40fbe816ba2d4198ea799c03b78a016bb3caf1b`; it predates this scope cleanup.
SDK compatibility passed on the earlier
[`634d867811401f4812b7580bce0e782af3cdcd19`](https://github.com/MINT-SJTU/RoboScope-Kit/actions/runs/43082168613);
that does not establish CI or simulator acceptance of a later source revision.
The [`f40fbe8` workflow](https://github.com/MINT-SJTU/RoboScope-Kit/actions/runs/43158764888)
still requires approval; the cleanup's eventual exact-head CI result must be checked separately.

## Official PPO prescribed-count results

Source `6dcbf53dac517f4ab7bf55f3dc4cf042e716da20` was built with `build-runtime.sh` into a new
directory; the simulator self-test passed with Kit 0.8.0b36. The runtime manifest SHA-256 is
`ca92cfe3621518d8fa88234bb0e308e4b692834e12b60f9e4afd5d2c6a9a784c`. Checkpoints are the eleven
[official PPO files](../../examples/maniskill/official-models.json) at demonstration revision
`3558f72cf99d51d04a6ac4779e88b3da4bbf1628`, verified by SHA-256. The suite is
`official_ppo_state_v1`: 100 episodes per task, seeds 0-99, ten-lane groups, one Worker and one
Policy replica, no autotune. The native side resolves each configuration from pinned upstream
sources, uses the same checkpoints without Policy RPC, and runs each task in its own process.

| Task | Success once | Success at end | Mean return | Control steps |
| --- | ---: | ---: | ---: | ---: |
| `PickCube-v1` | 100/100 | 100/100 | 43.74 | 5,000 |
| `RollBall-v1` | 63/100 | 0/100 | 20.06 | 8,000 |
| `PushT-v1` | 98/100 | 72/100 | 62.87 | 10,000 |
| `LiftPegUpright-v1` | 100/100 | 50/100 | 39.82 | 5,000 |
| `PokeCube-v1` | 70/100 | 36/100 | 38.04 | 5,000 |
| `PullCube-v1` | 100/100 | 8/100 | 19.60 | 5,000 |
| `PushCube-v1` | 100/100 | 24/100 | 23.56 | 5,000 |
| `StackCube-v1` | 84/100 | 81/100 | 39.30 | 5,000 |
| `TwoRobotPickCube-v1` | 95/100 | 95/100 | 80.11 | 10,000 |
| `TwoRobotStackCube-v1` | 100/100 | 56/100 | 83.25 | 10,000 |
| `AnymalC-Reach-v1` | 100/100 | 99/100 | 146.42 | 20,000 |
| Total | 1,010/1,100 | 621/1,100 | – | 88,000 |

SDK and native each produced 1,100 records with no duplicate or missing episode. Seeds, lanes,
initial-state hashes, first actions, returns, success flags, steps and terminal masks match exactly,
with zero tolerance. A second complete run recorded per-step observation and action hashes for
every episode; all 1,100 trajectory files match on both sides. The 90 unsuccessful episodes are
scored task outcomes: there were no Worker failures, retries or out-of-memory events.

The timed run disables trajectory hashing. SDK evaluation takes 403.8 s (command 407.4 s):
9,807 episodes/h and 217.0 control steps/s. Mean GPU busy is 33.7%, peak VRAM 4.66 GiB and mean
CPU use 1.14 cores. Policy loading takes 1.56 s and the total Policy wait 19.5 s. Inside the
evaluation, environment construction takes 20.3%, seeded resets 11.0%, rollout 53.5%,
environment closing 6.2%, and Worker recycling between tasks 9.0%. The native run takes 386.1 s.

### Throughput by Worker count

The same source, checkpoints and 1,100 episodes were evaluated with more simulator Workers sharing one
GPU and one Policy replica. Every run matches the native records exactly and has no Worker failure,
retry or out-of-memory event. Each Worker holds its own ten-lane vector environment, so Worker count
does not change initial states or seed grouping.

| Workers | Evaluation (s) | Speedup | Episodes/h | Mean GPU busy | Peak VRAM (GiB) | Mean CPU cores |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 403.8 | 1.00x | 9,807 | 34% | 4.7 | 1.1 |
| 2 | 311.1 | 1.30x | 12,729 | 56% | 8.8 | 2.1 |
| 4 | 244.7 | 1.65x | 16,184 | 76% | 17.2 | 3.9 |
| 6 | 220.9 | 1.83x | 17,927 | 82% | 22.8 | 5.8 |
| 8 | 214.0 | 1.89x | 18,505 | 84% | 26.8 | 7.5 |
| 12 | 193.6 | 2.09x | 20,455 | 94% | 35.8 | 10.6 |

CUDA MPS does not help this workload: with MPS, 8 Workers take 223.0 s and 12 Workers 197.9 s and 197.7 s
in two runs. Sixteen Workers with MPS take 197.9 s at 46.7 GiB peak VRAM. The GPU is saturated near
twelve Workers on this machine. A second Policy replica is slower: 8 and 12 Workers take 231.4 s and
198.3 s with two replicas. The small actor is not the bottleneck; the extra replica competes with the
simulators for the GPU.

Autotune previously capped ManiSkill at two Workers; the cap is now 16. Autotune then selects eight
Workers: evaluation takes 213.0 s after 391.4 s of probing 1, 2, 4, 8 and 16 Workers. Sixteen Workers
miss the knee, which requires a doubling to deliver a quarter of linear scaling. The prescribed runs
agree: twelve Workers are 10.5% faster than eight, below the 12.5% the same rule requires for that step.
Autotune therefore selects the efficient point, not the fastest one; a fixed `--concurrency 12` is
fastest on this machine and avoids the probing time. Without a cap, a probe after an accepted 16-Worker
step requested 30 Workers from the CPU estimate and failed with a GPU solver allocation error. Two probe
items per Worker leave the steady-state window too short at 16 Workers: it reported 504.6 steps/s against
an elapsed rate of 353.1, so the manifest keeps three.

To reproduce a row, evaluate with a fixed Worker count and compare with the native records as in
the [example README](../../examples/maniskill/README.md):

```bash
<install-dir>/env/bin/python -m roboscope_kit.runtime.benchmark --benchmark maniskill \
  --suite official_ppo_state_v1 --benchmark-manifest <install-dir>/benchmark-manifest.json \
  --checkpoint-source <official-checkpoints> --policy-config <roboscope-policy.json> \
  --output-dir <out> --concurrency 8 --replicas 1
```

Evaluation time is the last item completion in the run's `result.json` `item_timeline`; GPU busy,
peak VRAM and CPU cores are its `resources` fields.

### Comparison with the original evaluator

The unmodified upstream `ppo.py --evaluate` ran each checkpoint with 100 evaluation environments
and seeds 0, 1 and 2. Its reset schedule draws different initial states, so episodes cannot be
paired. Rates are compared with a two-sided Fisher exact test. To separate seed sampling from
protocol differences, the native harness also evaluated seeds 100-399 under the Kit protocol.

| Task | Kit, seeds 0-99 | Kit, seeds 100-399 | Original evaluator (seed 0/1/2) | Fisher p, Kit 0-399 vs original |
| --- | ---: | ---: | --- | ---: |
| `PickCube-v1` | 100/100 | 300/300 | 100/100/100 | 1.0 |
| `RollBall-v1` | 63/100 | 201/300 | 67/65/69 | 0.81 |
| `PushT-v1` | 98/100 | 287/300 | 87/92/91 | 0.001 |
| `LiftPegUpright-v1` | 100/100 | 299/300 | 100/100/100 | 1.0 |
| `PokeCube-v1` | 70/100 | 220/300 | 79/75/76 | 0.22 |
| `PullCube-v1` | 100/100 | 300/300 | 100/100/100 | 1.0 |
| `PushCube-v1` | 100/100 | 300/300 | 100/100/98 | 0.18 |
| `StackCube-v1` | 84/100 | 269/300 | 85/92/91 | 0.72 |
| `TwoRobotPickCube-v1` | 95/100 | 292/300 | 94/98/98 | 1.0 |
| `TwoRobotStackCube-v1` | 100/100 | 298/300 | 96/99/97 | 0.023 |
| `AnymalC-Reach-v1` | 100/100 | 300/300 | 100/100/100 | 1.0 |

Nine tasks agree with the original evaluator. `TwoRobotStackCube-v1` (99.5% against 97.3%) is
not significant after correcting for eleven tasks. `PushT-v1` differs consistently: 96.3% under the
Kit protocol against 90.0% in the original evaluator, p = 0.001, and every Kit seed block exceeds
every original seed. The gap follows the vector width. The same unmodified evaluator with ten
evaluation environments and seeds 0-29 succeeds in 286/300 episodes (95.3%). That agrees with the
Kit protocol (p = 0.57) and differs from its own 100-environment result (p = 0.018). The Kit
protocol freezes ten lanes, so its PushT rate is the ten-lane rate; it is not a Kit defect.

The native harness's diagnostic `--reuse-environment` option is not safe for these comparisons.
With it, `AnymalC-Reach-v1` seeds 200-299 fail in two of two attempts with a CUDA out-of-memory
error in the contact impulse query of the first reset after the environment is reused; with a new
environment per group the same seeds complete 100/100. The seed 200-299 AnymalC row above uses
the per-group environment. The SDK always builds a new environment for each group.

## Remaining-model prescribed-count results

On 2026-11-13, the selected checkpoints for the other tasks ran at the prescribed 100 episodes per
task: Kit 0.8.0b37 (this source plus #126's Worker scheduling and out-of-memory changes, which do
not affect records), a new runtime built with
`build-runtime.sh` and checkpoints verified against their pinned SHA-256 by the example downloaders.
Four Workers and one Policy replica; records do not depend on Worker count. The native side is
`native_eval_policy.py`, which builds native Gym environments without the Kit Worker or RPC and
reuses the same Policy and manifest configuration. Its exact agreement verifies the Kit's execution
path; it does not independently resolve the authors' task configuration.

| Task | Checkpoint | Metric | Kit result | Published result | Records vs native |
| --- | --- | --- | ---: | --- | --- |
| `UnitreeG1TransportBox-v1` | Author PPO | Success once / at end | 100/100, 99/100 | – | Exact |
| `UnitreeG1PlaceAppleInBowl-v1` | Author PPO | Success once / at end | 26/100, 22/100 | – | Exact |
| `MS-AntWalk-v1` | NEWT | Mean return | 993.7 | none | Exact |
| `MS-AntRun-v1` | NEWT | Mean return | 201.9 | none | Exact |
| `MS-CartpoleBalance-v1` | NEWT | Mean return / 1000 | 0.9998 | 0.999 | Exact |
| `MS-CartpoleSwingUp-v1` | NEWT | Mean return / 1000 | 0.991 | 0.991 | Exact |
| `MS-HopperStand-v1` | NEWT | Mean return / 600 | 0.217 | 0.806 | Exact |
| `MS-HopperHop-v1` | NEWT | Mean return / 600 | 0.088 | 0.797 | Exact |
| `PlaceSphere-v1` | NEWT | Success once / at end | 34/100, 16/100 | 0.9 | Exact |
| `PullCubeTool-v1` (pose controller) | NEWT | Success once / at end | 93/100, 93/100 | 1.0 | Exact |
| `PutCarrotOnPlateInScene-v1` | Octo-small | Success once / at end | 19/100, 18/100 | 0.097 | Exact |
| `PutEggplantInBasketScene-v1` | Octo-small | Success once / at end | 59/100, 59/100 | 0.569 | Exact |
| `PutSpoonOnTableClothInScene-v1` | Octo-small | Success once / at end | 31/100, 31/100 | 0.472 | Exact |
| `StackGreenCubeOnYellowCubeBakedTexInScene-v1` | Octo-small | Success once / at end | 1/100, 1/100 | 0.042 | Exact |

All 1,400 episodes match: seeds, lanes, initial-state hashes, first actions, returns, success
flags, steps and terminal masks, with zero tolerance. The G1 pair takes 84.7 s for 200 episodes
and 20,000 steps; the NEWT suite 356.2 s for 800 episodes and 535,000 steps. The Bridge suite runs
the Octo-small RGB Policy in its own inference environment, built with `build-octo-policy.sh`,
with one Worker; it takes 2,583.7 s for 400 episodes and 30,000 steps, and its native side is
`native_eval_octo.py`.

Cartpole reproduces NEWT's published scores. PullCubeTool is close. Hopper and PlaceSphere are far
below them: NEWT's published single-task scores plan with MPC at its own horizons (Hopper 500
ticks, PlaceSphere 25),
from CPU simulation on a later ManiSkill nightly, while this example runs the actor prior at the
benchmark's horizons on GPU physics. These tasks therefore have a working intended-task model but
no matching competence. NEWT publishes no usable score for the Ant tasks.

The Bridge published results are SIMPLER visual-matching scores on the original ManiSkill2 real2sim
tasks, not these ManiSkill3 profiles. Octo-small is a generalist model trained on Open
X-Embodiment data that includes Bridge, not a per-task checkpoint, so the Bridge tasks remain a
provisional visual baseline.

The other 30 of the 55 TaskIDs have no checkpoint that meets the
[checkpoint search](TASK_INVENTORY.md#public-checkpoint-search) rule and remain unvalidated.

## Scoring and shard correction

Return-scored episodes contribute neither binary successes nor failures nor the
success-rate denominator. For example, a regression fixture with two successful,
one unsuccessful and six return-only records has nine completed episodes, two
successes, one failure and success rate 2/3; it is not 2/9. A return-only result has
zero binary successes/failures and a null success rate. Completed episodes and
steps still include every record.

Each profile's return sum/count merges independently into a sample-weighted mean,
including unequal shard sizes and negative values. No return average spans
different task reward scales. Official PPO and Octo native multi-profile summaries
therefore report a null aggregate `mean_return`, retaining each profile's mean.
An Octo single-profile result preserves its own mean. CPU fixtures verify that
reporting boundary without executing a model. CPU replay checks arithmetic through both merge
paths, including corruption and content-identity fixtures. Replayed historical
grouping is declared explicitly and does not validate ten-lane initial states.

Result-only reduction reports `coverage="accounted"`; it cannot inspect episode
identities. Record-tree merging reports `coverage="complete"` only after checking
the deterministic WorkItem partition, actual profile/index/seed/lane/record IDs,
steps, boolean terminal masks and stored summaries. Neither label establishes
task/model acceptance. Counts and identities compare exactly. Only floating
reduction uses `rel_tol=1e-12, abs_tol=1e-10` for addition-order rounding; model
inputs, observations, actions and terminal masks retain strict comparison rules.

## Source checks and outstanding validation

At source `f40fbe816ba2d4198ea799c03b78a016bb3caf1b`, all 270 existing Linux
pytest cases pass in 8.71 s. Ruff, including all then-present ManiSkill examples,
the Python-support check, and canonical-LF `bash -n`/ShellCheck pass. External
CPU checks cover:

- Eleven independently resolved official PPO configurations and constructor probes,
  266 rejected corrupt configurations, eleven shared-annotation isolation cases,
  and four corrupt native-parent provenance cases. These are configuration
  fixtures, not an actual upstream evaluator/model/GPU comparison.
- The first complete ten-lane WorkItem, probe rounding from eleven to twenty
  episodes, and six rejected invalid count/width/default configurations.
- Four trajectory retry cases, four actual-queue context-recycling cases,
  timeout-output logging and separate RGB/render shapes.
- Stored-record shard reconstruction, altered-record rejection and content
  identities, independently of trained-policy acceptance.

The retained-scope cleanup also passes all 270 existing pytest cases in 7.88 s,
Ruff across the remaining ManiSkill examples and the Python-support check. New
CPU dispatch fixtures verify retained catalog selection/hash checks, unchanged
7D action forwarding at the Policy boundary and rejection of removed transfer
contracts. They use inference stubs: learned-model outputs and neural parity are
not newly verified. Four corrupt native-parent provenance fixtures are rejected,
and native aggregate-return scope passes the CPU fixture above. Documentation
checks match all 55 inventory task/profile/horizon/metric rows, eleven suite
counts and eight NEWT profiles to the manifest, with no callers of deleted files.

Independent CPU checks exercise the actual factory for all eleven suites,
including full plans, Quick Test and complete ten-episode reduced groups. Every
Quick Test matches its full plan's first ten-lane group. Manifest, SDK task
specifications, registry and suite sizes agree; the suite and inventory unions
both contain exactly 77 profiles, with no orphan. All 55 original TaskIDs, five
original suites and their profile dictionaries are unchanged from `f40fbe8`.
The source AST inventory still resolves to 58 registrations minus three helpers,
or 55 evaluation tasks. All selected candidates remain marked unvalidated.
This verifies planning and declared scope, not episode execution, installed
runtime/asset availability, model competence or new upstream pairing.

The b36 wheel built and installed into an owned CPU checker. Its artifact and
check identities remain external; installing a wheel is not building a simulator.
Initial process-fixture empty-PID-file and Windows CRLF failures remain in the
receipts. The unchanged process test passed alone in 1.85 s and then in the full
passing suite; script checks use canonical LF bytes. No test was changed to hide
either failure. `AGENTS.md` prohibits adding repository tests, so additional
regression checks remain external.

The retained-scope source/check hashes are bound externally at delivery. No current
ten-lane model rollout, fresh builder run or GPU self-test is claimed for this
cleanup. Required acceptance remains prescribed-count execution with acceptable
same-task/robot checkpoints, independently configured native comparison,
appropriate competence, complete records and exact-head CI.

## Timing and resource coverage

The [official PPO results](#official-ppo-prescribed-count-results) report evaluation time,
episodes/h, steps/s, Policy wait, GPU busy, peak VRAM and CPU use for the eleven official tasks.
Other retained models have no current-source timing. SM/Tensor/RT throughput is unavailable:
the machine denies GPU performance counters. Do not reuse a
borrowed-policy cohort as that measurement. CPU wall spans can include
asynchronous GPU work, and concurrent Worker times cannot be summed into
application wall time. Issue #31 performance work remains separate.

## Original assets

AssemblingKits, TurnFaucet, OpenCabinetDoor and OpenCabinetDrawer still lack their
original assets. Six retained profiles are blocked: the state/RGB pairs for the
first two tasks and the two original cabinet state profiles. Model and asset gaps
can overlap. Unknown outcomes are not zero-success results. Original asset
availability and exact clutter-corpus recovery are documented in the
[compatibility notes](REMAINING_COMPATIBILITY.md#original-asset-gaps).
