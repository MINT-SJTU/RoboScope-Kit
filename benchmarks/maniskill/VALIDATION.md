# ManiSkill validation status

**2026-11-12: integration acceptance is incomplete.** The pinned inventory has 55
evaluation TaskIDs and 89 selectable task/robot/observation/controller profiles.
Historical paired execution evidence covers 75 profiles and 51 TaskIDs across
different source commits. Those figures describe execution coverage, not accepted
tasks or a current-source full evaluation. They must not be added into a single
success rate or relabeled as 100-episode-per-task validation.

Acceptance requires a public checkpoint trained for the intended task and robot,
nonzero task competence where success is defined, the prescribed episode count,
independently resolved upstream configuration, matching frozen vector grouping and
results from the final code. Cross-task/robot transfers and zero-success samples
do not meet that model requirement. Return-scored tasks retain their own native
metric rather than fabricated binary success. See [the maintainer decision](https://github.com/MINT-SJTU/RoboScope-Kit/pull/99#issuecomment-6800683447).

## Machine, protocol and source

The recorded machine reports an NVIDIA GeForce RTX 4090 with 49,140 MiB VRAM,
driver 595.71.05 and a 20-core CPU quota. The simulator source is
`baab60ede2e89167c1b7aaed41a9aa8e690a9d1e`, SAPIEN 3.0.0b1, Python 3.11.15,
Torch 2.6.0/CUDA 12.4 and NumPy 1.26.4. Assets, model provenance and individual
native horizons are pinned separately. The canonical [evaluation protocol](../../packages/roboscope-kit/MANISKILL_EVALUATION_PROTOCOL.md)
defines observations, controller semantics, scoring and grouping.

The current protocol freezes ten simulator lanes per group. Older width-one/two
records below are preserved under their original configuration; equal seed
numbers do not make them width-ten initial states. The Quick Test must use the
first complete frozen group. Diagnostic self-tests at other widths are explicitly
separate from task acceptance.

Statistics/shard correction source
[`fad25d36b5207d1ff47ee4cf8af68721f98241e0`](https://github.com/QLL-CCDD/RoboScope-Kit/commit/fad25d36b5207d1ff47ee4cf8af68721f98241e0)
has CPU-only verification and published evidence. The subsequent frozen-group,
trajectory-retry, self-test-report and upstream-configuration changes also pass
the local checks described below. Their current working-source hashes and check
receipts are recorded externally; a final commit binding is required at delivery.
**No new GPU evaluation, fresh runtime build or simulator self-test is claimed for
these corrections.** Current ten-lane model evaluation remains pending. SDK compatibility passed on the earlier
[`634d867811401f4812b7580bce0e782af3cdcd19`](https://github.com/MINT-SJTU/RoboScope-Kit/actions/runs/43082168613);
that does not establish CI or simulator acceptance of later code. The statistics
correction's exact-head workflow still awaits approval.

## Recorded evidence

The table lists separate, complete-horizon historical cohorts. It excludes short
integration prefixes and does not combine overlapping episode identities.
Command time includes loading/startup; evaluation intervals follow each path's
recorded boundaries. Native child-phase sums and SDK wall time are not equivalent
performance denominators. These are traced correctness cohorts, not speed claims.

| Workload | Profiles / episodes each | Vector width | Source commit | SDK / native command s | SDK / native evaluation s | Scope and limit |
|---|---:|---:|---|---:|---:|---|
| Broad original-horizon model matrix | 66 / 1 | 1 | [`42e6b24bbde9e26bfe714e92786d2891bfc37d4b`](https://github.com/QLL-CCDD/RoboScope-Kit/commit/42e6b24bbde9e26bfe714e92786d2891bfc37d4b) | 1409.283 / 1796.505 | 1194.185 / 1177.403 | 19,240 action pairs; includes transfer models; one sample per profile |
| Additional canonical controller/state profiles | 6 / 2 | 2 | [`db473261e41c4663a7350bbb79816789fc632f48`](https://github.com/QLL-CCDD/RoboScope-Kit/commit/db473261e41c4663a7350bbb79816789fc632f48) | 87.704 / 90.953 | 79.502 / 67.161 | 1,400 action pairs; includes LiftPeg and dual-arm transfers |
| Direct-pull and cross-robot actor priors | 2 / 2 | 2 | [`c660435b235cd9d3db303fcc3c14ee6beba1ebe1`](https://github.com/QLL-CCDD/RoboScope-Kit/commit/c660435b235cd9d3db303fcc3c14ee6beba1ebe1) | 24.146 / 24.700 | 20.857 / 17.400 | 600 action pairs; no successes; not intended-task experts |
| LiftCube actor on existing LiftPeg position profile | 1 / 1 | 1 | [`575b49c03c7566f112bd1b1a1df4f1b5a15a4668`](https://github.com/QLL-CCDD/RoboScope-Kit/commit/575b49c03c7566f112bd1b1a1df4f1b5a15a4668) | 10.170 / 9.547 | 6.413 / 5.126 | 50 action pairs; transfer prior; no success |
| Stored-record scoring and shard replay | 25 stored episodes per reconstruction | Original grouping retained | [`fad25d36b5207d1ff47ee4cf8af68721f98241e0`](https://github.com/QLL-CCDD/RoboScope-Kit/commit/fad25d36b5207d1ff47ee4cf8af68721f98241e0) | CPU only; not a rollout clock | N/A | No simulator or model execution; verifies arithmetic and record coverage |

The broad matrix contains 52 success-scored episodes, with 13 successes (25%),
and fourteen return-scored episodes. That is a description of that historical
matrix, not a task-averaged benchmark score. An earlier 1,500-episode reference
predates actual native terminal-mask recording and remains excluded from strict
mask-complete acceptance. Historical reports and failed attempts are preserved
at [their immutable source snapshot](https://github.com/QLL-CCDD/RoboScope-Kit/tree/634d867811401f4812b7580bce0e782af3cdcd19/benchmarks/maniskill).

Raw records, traces, commands, model/source hashes and audits remain outside Git:
[broad and canonical-state evidence](https://github.com/QLL-CCDD/RoboScope-Kit/releases/tag/maniskill-issue25-native-review-20261111),
[canonical-prior evidence and retained failures](https://github.com/QLL-CCDD/RoboScope-Kit/releases/tag/maniskill-issue25-canonical-priors-20261111),
[LiftCube-prior evidence](https://github.com/QLL-CCDD/RoboScope-Kit/releases/tag/maniskill-issue25-lift-prior-20261111),
and [statistics/shard evidence](https://github.com/QLL-CCDD/RoboScope-Kit/releases/tag/maniskill-issue25-statistics-shards-20261112).
The last release includes a per-profile historical statistics table with actual
model, input mode, episode count, success or mean return, horizon and source.

## Scoring and shard correction

Unchanged stored WorkResults reproduce a real disagreement between the former
generic summary and the task-aware official result:

| Stored workload | Episodes / steps | Former generic success rate | Correct success denominator | Return-only episodes |
|---|---:|---:|---:|---:|
| NEWT mixed profiles | 9 / 5,400 | 2/9 = 22.22% | 2/3 = 66.67%; one unsuccessful scored episode | 6 |
| Humanoid profiles | 5 / 5,000 | 0%; five incorrectly counted failures | N/A; zero binary failures | 5 |
| ANYmal Spin | 1 / 200 | 0%; one incorrectly counted failure | N/A | 1 |
| Unitree standing | 2 / 2,000 | 0%; two incorrectly counted failures | N/A | 2 |
| Official PPO profiles | 11 / 880 | 10/11 = 90.91% | Unchanged | 0 |

The correction keeps return-only samples out of successes, failures and the
success-rate denominator. Return sums/counts merge separately for each profile;
their mean is sample-weighted, including unequal shard sizes and negative values.
No return average spans different task reward scales.

CPU replay verifies the actual NEWT, Humanoid and PPO records with two and three
shards through both existing merge paths. Record-tree merging checks the complete
deterministic WorkItem partition and actual profile/index/seed/lane/record IDs,
steps, boolean terminal masks and stored summaries. Thirty altered record-tree
cases and 36 altered result-only cases are rejected, alongside eight standard
statistics corruptions. Content-identity fixtures check changed model/Policy
bytes and equal relative content at different locations. Their synthetic fixture
hashes do not establish new live model provenance.

Result-only reduction reports `coverage="accounted"`; it cannot inspect actual
episode identities. Record-tree merging reports `coverage="complete"` only after
its stronger record checks. Neither label establishes full task/model acceptance.
Integer counts and identities compare exactly. Only floating arithmetic reduction
uses `rel_tol=1e-12, abs_tol=1e-10` for addition-order rounding; model inputs,
actions, observations and terminal masks retain their strict comparison rules.

## Current source checks

All 270 existing Linux pytest cases pass in 8.71 s. Ruff, including every ManiSkill
example, and the Python-support check pass. `bash -n` and ShellCheck pass on the
canonical LF build-script bytes. These are source checks; the changed builder has
not been used for a fresh runtime build or GPU self-test in this correction batch.

The external CPU checks additionally verify:

- Independent resolution of all eleven official PPO task configurations and eleven
  constructor probes, with 266 corrupt configurations rejected. Eleven annotation-
  isolation checks ensure shared model metadata cannot select environment settings.
  Four additional corrupt native-parent provenance cases are rejected.
  These are AST/configuration/constructor fixtures, not actual upstream evaluator,
  model or paired GPU execution.
- Quick Test selects the exact first full WorkItem with ten lanes and seeds 0-9.
  A probe request for eleven episodes rounds to twenty with width ten. Six invalid
  count/width/default configurations are rejected.
- Four trace failure/retry cases preserve failed partial attempts and publish only
  completed canonical traces; four actual-queue cases cover context recycling and
  unchanged default behavior. Timeout logging preserves captured bytes, strings
  and absent output before raising. Separate RGB/render shapes remain correct.
- The original 25 stored records replay through record-tree merging in six two-
  or three-shard partitions, rejecting thirty record corruptions and passing three
  content-identity checks. This replay explicitly declares its archival vector
  width one; it does not validate current ten-lane simulator initial states.

The current 0.8.0b36 wheel also builds and installs into the owned CPU checker.
Its 261,955-byte artifact and source/check identities remain outside Git with the
validation receipts; an installed CPU wheel is not a fresh simulator build.

An initial complete pytest run hit an existing process fixture's empty-PID-file
race. That unchanged test then passed alone in 1.85 s, followed by the passing
complete suite. An initial build-script syntax check encountered Windows checkout
CRLF; subsequent checks use the canonical LF bytes. Original failed logs remain
available with the receipts; no repository test was changed to hide either result.

The current `AGENTS.md` prohibits adding repository tests, so the new checks remain
external; this does not fulfil the review's separate request for additional
repository regression tests. Follow-on source evidence is being prepared under
`maniskill-issue25-review-source-20261112`; that release is not yet published.

## Timing and resource coverage

Recorded evidence separates command startup/loading, environment creation/reset,
rollout, RPC, cleanup and tuning where each source emits them. Device-wide GPU
busy and sampled CPU/memory/VRAM are available in the linked raw summaries;
unavailable SM/Tensor/RT counters, isolated inference/Vulkan GPU duration and
queue-only latency remain N/A. CPU wall spans can include asynchronous GPU work.
Concurrent Worker seconds are not added into an application wall-time gain.
The [historical lifecycle budget](https://github.com/QLL-CCDD/RoboScope-Kit/blob/634d867811401f4812b7580bce0e782af3cdcd19/benchmarks/maniskill/REVIEW_VALIDATION.md#measured-lifecycle-budget)
retains its original source and does not validate the corrected current runtime.
Issue #31 throughput work is separate from this integration record.

## Remaining acceptance

The original selected task suite has historical execution evidence for 45 of 55
profiles. Six original state contracts lack a selected compatible model:
`pick_clutter_ycb_state`, `fmb_assembly1_easy_state`, `ms_humanoid_stand_state`,
`unitree_h1_stand_state`, `unitree_g1_stand_state`, and `anymal_c_spin_state`.
Their exact missing input contracts are in [the compatibility notes](REMAINING_COMPATIBILITY.md).
An alternative RGB/root/controller profile does not validate its original sibling.

Four TaskIDs still lack original assets: AssemblingKits, TurnFaucet,
OpenCabinetDoor and OpenCabinetDrawer. Eight associated state/RGB/depth profiles
are unexecuted; model and asset blockers can overlap. Unknown results are not
zero-success results. Cross-task/robot priors do not close these acceptance gaps.
Transfer-policy code and its historical evidence remain in this candidate; moving
them to follow-up work and completing the intended-task checkpoint search are
still outstanding review requirements.

The [HORA contact repeatability finding](HORA_NATIVE_REPEATABILITY.md) remains a
strict observation-comparison failure even though actions and outcomes match.
There is no quantization or tolerance waiver, and that transfer model also does
not meet the intended-policy acceptance condition.

Required next evidence is current-source evaluation at the frozen grouping and
prescribed counts with acceptable same-task/robot checkpoints, a configuration-
independent upstream comparison, task competence where applicable, complete
record coverage and exact-head CI. Source configuration checks and CPU replay
cannot substitute for those simulator/model runs.
