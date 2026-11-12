# ManiSkill validation status

**2026-11-12: integration acceptance is incomplete.** The pinned inventory declares
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
| Official state PPO | Eleven task-trained checkpoints, including the genuine 16D dual-Panda actors | Final-source, prescribed-count independent upstream comparison remains pending |
| Author G1 PPO | TransportBox and PlaceAppleInBowl task/robot checkpoints | Author simulator/horizon differences and independent native comparison remain unresolved |
| NEWT | Eight task/robot checkpoints: six control tasks, PlaceSphere and PullCubeTool | Current inference is the deterministic actor prior, not the author's default MPC evaluator; version, backend and horizon differences require validation |
| Bridge Octo-small | Four original WidowX Bridge profiles | Provisional baseline; matching per-task training and competence are not established |
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

Raw commands, source/check identities and regression fixtures remain outside Git.
CPU regression receipts do not count any fixture model or TaskID toward acceptance.
The [source-review CPU evidence](https://github.com/QLL-CCDD/RoboScope-Kit/releases/tag/maniskill-issue25-review-source-20261112)
is published for `f40fbe816ba2d4198ea799c03b78a016bb3caf1b`; it predates this scope cleanup.
SDK compatibility passed on the earlier
[`634d867811401f4812b7580bce0e782af3cdcd19`](https://github.com/MINT-SJTU/RoboScope-Kit/actions/runs/43082168613);
that does not establish CI or simulator acceptance of a later source revision.
The [`f40fbe8` workflow](https://github.com/MINT-SJTU/RoboScope-Kit/actions/runs/43158764888)
still requires approval; the cleanup's eventual exact-head CI result must be checked separately.

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

This source-only correction batch has no new rollout timing or resource samples.
Total/formal evaluation time, episodes/h, steps/s, inference/render/wait latency,
GPU busy, SM/Tensor/RT throughput, average/peak VRAM, CPU use and task outcomes are
therefore unavailable for the retained current-source scope. Do not reuse a
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
