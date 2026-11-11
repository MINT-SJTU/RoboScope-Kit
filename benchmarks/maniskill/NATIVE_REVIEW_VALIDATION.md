# ManiSkill native-reference review validation

**2026-11-11: selected R12/b51, R13/b52, R16/b55 and R17/b56 correctness cohorts PASS; acceptance remains partial.**
Cumulative coverage is **75 profiles / 51 task IDs**: 66 compared in R12, six additional canonical
profiles in R13, two in R16 and canonical LiftPeg4 in R17. This is not a fresh b56 rerun of all 75 profiles, acceptance of all 55 tasks, or
100-episode-per-task acceptance. [Issue 25](https://github.com/MINT-SJTU/RoboScope-Kit/issues/25)
and [PR 99](https://github.com/MINT-SJTU/RoboScope-Kit/pull/99) remain open.

R12 binds b51 `source-r12` at code head `42e6b24bbde9e26bfe714e92786d2891bfc37d4b`;
R13 binds b52 `source-r13` at `db473261e41c4663a7350bbb79816789fc632f48`.
R16 binds b55 `source-r16` at `c660435b235cd9d3db303fcc3c14ee6beba1ebe1`.
R17 binds b56 `source-r17` at `575b49c03c7566f112bd1b1a1df4f1b5a15a4668`.
The final documentation commit follows measurement without changing the measured runtime or Policy
code. The candidate incorporates main `329ef98`, preserving its RLBench registration; this ManiSkill
integration has not been merged.

## Scope and strict comparison

The user deferred original-asset recovery for AssemblingKits, TurnFaucet, OpenCabinetDoor and
OpenCabinetDrawer. The inventory is **89 profiles = 75 compared + six unmatched available + eight
asset-deferred**. Active verification covers the other 51 task IDs and their claimed profiles.
Selected siblings do not validate unmatched observation/controller contracts; see the
[current gaps](REMAINING_COMPATIBILITY.md#remaining-profile-acceptance-gaps).

The 75 configurations comprise 45 profiles selected by the named per-task inventory,
19 additional RGB profiles, seven root-state profiles and four controller alternatives.
They cover 51 distinct tasks. Six task IDs currently have comparison evidence only for an
alternative profile; that does not validate their original state configuration. Optional model
compatibility variants are retained separately from task coverage and the regression matrix for
each change. The original `upstream_tasks_v1` still has 45 of 55 selected profiles validated:
six model-contract gaps and four asset-deferred profiles remain. LiftPeg4 is an existing extra
tabletop controller profile; the original per-task inventory already selects its validated 7D PPO
sibling. The performance workload selects only `rollball_state`; no full-profile rerun is implied.

## R17 canonical LiftPeg4 extension

The [LiftCube actor prior](../../examples/maniskill/LIFTCUBE.md) uses the author's complete
four-output trained policy, explicit `mpc=False, eval_mode=True`, real native state and the
unchanged 50-tick LiftPeg4 profile. It does not reproduce the author's default stochastic MPC
evaluation or claim peg-uprighting competence. Both SDK and native samples have zero successes.

| R17 cohort | Seeds / vector groups | Paired samples | Action / observation pairs | SDK command / eval s | Native command / eval s |
| --- | --- | ---: | --- | --- | --- |
| Ordinary Quick Test, full 50 ticks | 0 / 1 | 1 | 50 / 51 | 10.170 / 6.413 | 9.547 / 5.126 |
| Five-tick diagnostic prefix, original horizon retained | 0-4 / 2,2,1 | 5 | 25 / 30 | 16.163 / 12.441 | 18.690 / 9.449 |

All 75 actions, 81 observations, initial hashes, returns, success fields, episode identities,
lanes and actual boolean terminal masks match at zero tolerance. There are six paired samples
across the cohorts, with five distinct identities; seed 0 appears in both. The full episode
normally truncates at 50; the five short prefixes retain false/false masks. There are no
missing/duplicate records, Worker failures or OOMs. The N2 run retains its process between the
two full-width groups and performs one planned recycle before the smaller tail.

Native evaluation is the GPU-vectorized Gym reference with the same declared actor prior.
Its tail cohort sums sequential child evaluation intervals; the parent command includes the
extra interpreter/model startup. SDK and native clocks have different boundaries. Tracing is
enabled for these correctness checks; the small-sample throughput below is diagnostic.

| Path / cohort | Episodes or prefixes/h | Actual steps/s | GPU busy mean % | VRAM mean / peak GiB | Accounted CPU cores |
| --- | ---: | ---: | ---: | --- | ---: |
| SDK full N1 | 561.394 | 7.797 | 5.683 | 1.928 / 3.942 | 0.952 |
| Native full N1 | 702.366 | 9.755 | 9.128 | 1.914 / 3.942 | 0.957 |
| SDK N2 prefix | 1446.853 | 2.010 | 2.369 | 1.987 / 5.072 | 0.984 |
| Native N2 prefix | 1905.009 | 2.646 | 2.240 | 1.877 / 5.072 | 0.954 |

The raw summary retains model-loading/preparation, creation/reset/rollout/cleanup, RPC and
resource clocks. RPC includes inference, transport and validation. Isolated inference GPU,
render/Vulkan duration, queue-only wait and SM/Tensor/RT throughput remain N/A. NVML busy is
an activity indicator. There are too few full episodes to estimate success rates, stable
throughput or an independent episode-latency distribution. No runtime speedup is claimed.

The source-to-root frame conversion saturates 43 XYZ components in the full episode and
4/3/4/3/3 in the prefixes, identically on both paths. Native controller clipping is retained;
this is a measured transfer limitation, not exact source-controller equivalence. The fourth
trained gripper output and the two-tick source action hold remain intact.

The fresh simulator build and self-test pass in 51 s. All 337 regular source files and 94
installed SDK files match their snapshot; version and runtime manifest match b56. The actual
production model/policy CPU proof binds its four inputs to canonical-LF Git content, explicitly
retaining the original CRLF projection-file receipt and AST equivalence. All 60 tensors load
strictly and remain unchanged; 147 synthetic inputs plus one historical measured reset reproduce
the author's complete actor branch bitwise. Existing 270 pytest cases and Ruff pass. Older
prototype evidence is retained separately and is not relabeled as production execution.

[R17 raw evidence, CPU proofs and audits](https://github.com/QLL-CCDD/RoboScope-Kit/releases/tag/maniskill-issue25-lift-prior-20261111)
include commands, traces, logs, source binding and measurements. Final-head CI is checked
separately. Four original-asset tasks, the six original state-model gaps and the documented
HORA repeatability limit remain outside this extension's passed scope.

All listed cohorts retain exact actions, complete observation hashes, native returns/outcomes,
episode identities, horizons and actual boolean terminal masks. Full horizons require
`terminated=false`, `truncated=true`; five-tick prefixes retain false/false. No observation
quantization, tolerance change or acceptance waiver was applied. One or two full-horizon samples
per profile check integration; they do not estimate success rates or establish task competence.
Generic native references independently construct Gym environments/features/action execution and
bypass Kit Worker/RPC while sharing the Policy implementation. Author-code/CPU probes separately
check their stated inference and transfer components.

## Completed cohorts

All rows PASS their strict record comparison and exhaustive trace audit, with zero action,
observation or structural differences and zero worker failures/OOMs. K1/G1 is fixed. R12 uses
seed 0; R13/R16 prefixes use seed 0 at N1 and seeds 0/1/2 at N2; the supplemental R16
multi-group prefix uses seeds 0/1/2/3/4. Full N2 uses seeds 0/1.

| Cohort | Profiles / task IDs | N / episodes | Action / observation pairs | SDK command / eval s | Native command / eval s |
| --- | --- | --- | --- | --- | --- |
| R12 full original horizons | 66 / 51 | 1 / 66 | 19,240 / 19,306 | 1,409.283 / 1,194.185 | 1,796.505 / 1,177.403 |
| R13 N1 five-tick retry1 | 6 / 6 | 1 / 6 | 30 / 36 | 65.543 / 57.603 | 70.542 / 42.039 |
| R13 N2 five-tick tail | 6 / 6 | 2 / 18 | 90 / 108 | 74.340 / 66.520 | 84.656 / 55.241 |
| R13 full original horizons | 17 / 15 | 1 / 17 | 1,580 / 1,597 | 215.670 / 203.217 | 218.853 / 155.793 |
| R13 new-six full horizons | 6 / 6 | 2 / 12 | 1,400 / 1,412 | 87.704 / 79.502 | 90.953 / 67.161 |
| R16 new-two full horizons | 2 / 2 | 1 / 2 | 300 / 302 | 24.707 / 20.854 | 25.125 / 18.387 |
| R16 N2 five-tick tail | 2 / 2 | 2 / 6 | 30 / 36 | 38.351 / 35.147 | 44.807 / 27.782 |
| R16 new-two full horizons | 2 / 2 | 2 / 4 | 600 / 604 | 24.146 / 20.857 | 24.700 / 17.400 |
| R16 old-PPO five-tick regression | 11 / 11 | 1 / 11 | 55 / 66 | 103.356 / 96.980 | 117.450 / 66.543 |
| R16 N2 multi-group five-tick prefix | 2 / 2 | 2 / 10 | 50 / 60 | 34.990 / 31.584 | 41.895 / 25.008 |

R13's 17-profile N1 cohort includes the six new profiles and eleven official PPO regressions;
dual task IDs overlap across controller profiles. The N2 prefix explicitly exercises lane groups
of two then one. R12 recorded 54 planned profile recycles, separately from zero failure recovery.
Its 52 success-scored samples had 13 success-once / seven success-at-end; fourteen return-only
samples retain null success. The new six R13 full-horizon profiles had one success-once at N1 and two at N2;
matching records do not mean that the transfer policies solve their target tasks.

The added profiles have the following full-horizon paired records. N1 uses one episode at
seed 0; N2 uses two episodes at seeds 0/1. Success-at-end counts equal success-once counts in
these full-horizon cohorts. Counts below describe the observed samples, not population success rates.

| Cohort | Added profile | Original horizon | N1 action / observation pairs | N2 action / observation pairs | Successes N1; N2 |
| --- | --- | ---: | ---: | ---: | --- |
| R13/b52 | `unitree_g1_transport_box_state` | 100 | 100 / 101 | 200 / 202 | 1 of 1; 2 of 2 |
| R13/b52 | `unitree_g1_place_apple_in_bowl_state` | 100 | 100 / 101 | 200 / 202 | 0 of 1; 0 of 2 |
| R13/b52 | `peg_insertion_side_state` | 100 | 100 / 101 | 200 / 202 | 0 of 1; 0 of 2 |
| R13/b52 | `plug_charger_state` | 200 | 200 / 201 | 400 / 402 | 0 of 1; 0 of 2 |
| R13/b52 | `two_robot_pickcube_state` | 100 | 100 / 101 | 200 / 202 | 0 of 1; 0 of 2 |
| R13/b52 | `two_robot_stackcube_state` | 100 | 100 / 101 | 200 / 202 | 0 of 1; 0 of 2 |
| R16/b55 | `pull_cube_tool_state` | 100 | 100 / 101 | 200 / 202 | 0 of 1; 0 of 2 |
| R16/b55 | `unitree_go2_reach_state` | 200 | 200 / 201 | 400 / 402 | 0 of 1; 0 of 2 |

The R16 N2 prefix executes three episodes per profile, in groups of two then one. Its three
planned Worker recycles isolate profile/width changes; both full-horizon R16 cohorts have one
planned profile recycle each. These are distinct from zero failure recovery/OOM. Both new priors
have zero success-once and success-at-end in all listed samples. The original tabletop RGB
coverage remains 16/18 profiles, each with one full-horizon real-policy comparison in R12.
The supplemental five-episode N2 prefix checks groups of two, two, then one for each new profile:
consecutive full-width groups retain their process before the smaller tail gets a fresh process.
It passes ten episodes / 50 action pairs / 60 observations with three planned recycles and no
failures/OOM. The earlier three-episode prefix checks the tail without this consecutive-group coverage.

The first R13 support attempt FAILED after dual SDK completion because AutoFS had exhausted
200,000/200,000 inodes despite 221 GiB free. Its status and logs remain retained. Relocating 351
historical `source-r11` cache files was hash-verified; immutable archives were retained. Source,
runtime and models were unchanged. New metadata uses `LOCAL/review-r13-data`; successful retry
labels are distinct and do not overwrite the failed attempt.

## Changes and build checks

The b50 generic native evaluator runs sequential per-task children and preserves the virtual-env
Python path. The b51 optional `WorkerLoop.recycle_on_group_change=False` is enabled for ManiSkill:
before a profile change it leaves one new claim unevaluated; the orchestrator requeues it for a
planned Worker recycle while retaining the Policy replica. Same-profile groups reuse their Worker.
In b55, this existing ManiSkill opt-in also recycles before the group width changes. Native
references isolate the corresponding full-width and tail segments in separate child processes,
preserving global episode indices, seeds, lanes, unique profile summaries and child provenance.
Same-width groups retain their process. Other benchmarks keep the default opt-out. This fixes
SAPIEN contact-query lifetime failures; it is a correctness change with **no speed gain claimed**.
The Ant/Hopper A,A,B,B GPU check passed four full episodes / 3,200 steps with two Worker PIDs,
one retained Policy PID, one planned recycle and zero failures/OOMs.

Canonical Humanoid Walk/Run reconstruct the 67D actor input from their original 185D state;
joint-name metadata adds no observation features. The five-profile R12 Humanoid family passed
5,000 action pairs / 5,005 observation pairs. R13 adds G1 author state PPOs, real Peg/Plug prefix32
grasp-prior transfers and the [dual-Panda 4D actor composition](../../examples/maniskill/DUAL_PANDA_PPO.md).
The external `b52-frozen-protocol-proof.json` confirms all 89 profile definitions and existing
suite memberships unchanged; two suites were added.

R13 fresh builds, ordinary quick test and source binding PASS: 330 snapshot files, 249 audited
executable/config files and 94 installed SDK files in each simulator/Octo environment. Existing
b52 checks PASS: Ruff, 270 source tests (6.87 s), wheel/sdist/Twine, 92 installed module imports
and 270 installed-wheel tests (3.52 s). These local checks do not replace exact-head CI.

R16 fresh simulator/Octo builds (72 s), self-test and source binding PASS: 333 snapshot files,
251 executable/config files and 94 installed SDK files per environment. Ordinary Quick Test
completed its original 100-tick PullCubeTool episode with 0/1 success: command 14.079 s,
evaluation 10.819 s. Existing b55 checks PASS: 270 source tests (6.91 s), Ruff/Python support,
wheel/sdist/Twine, 92 installed module imports and 270 installed-wheel tests (4.09 s).

Original b54 CPU model receipts are reused through a seven-file unchanged-source binding to b55;
they are not relabeled as new inference runs. Separate CPU fixtures verify six Worker lifecycle
cases and native width planning/aggregation, including return-only null summaries and global
tail indices. The actual R14 per-task joint/controller-order probes carry forward through unchanged
profiles, source pin and calibration helper; they are not new R16 GPU measurements.

## Retained failures and HORA repeatability

**R14/b53 FAILED:** Go2's action first differed at step 2 despite identical observation hashes.
The NumPy/CUDA action-normalization probe reproduced 72 differing components out of 1,536,
maximum `2.9802322387695312e-8`. b54 uses the same Torch-device conversion before readback
as the native path. Weights, FP32 precision and all zero tolerances remain unchanged. R14 also
retains the Quick Test helper's missing alias-parent failure after successful SDK evaluation,
a stopped missing-environment probe and the same-process AnyMal-to-Go2 contact-query OOM.
Corrected separate-task joint-order probes passed; they do not erase the original failures.

**R15/b54 FAILED:** its N1 full-horizon cohort passed 300 action pairs / 302 observations,
but the N2 three-episode prefix hit the Go2 width-2-to-1 transition. SDK recovery completed
records after one Worker OOM; the native child failed at its smaller final environment.
The b55 lifecycle fix is verified by R16's clean prefix/tail and full-horizon cohorts. R14/R15
statuses, failed/partial trajectories, logs and source snapshots retain their original identities.

R9 retained a native NEWT Hopper-reset CUDA failure and HORA observation mismatch; R10 stopped on
the virtual-env symlink/base-interpreter error fixed in b50. Historical b48/R9 checks retain their
own scope: ten synthetic mask cases, N2 full PickCube50/PutCarrot60 comparisons and generic PPO
PASS. The b47/R8 64-profile prefix mask recheck also remains historical.

**b50/R11 FAILED:** final paired records cover 64 profiles, 17,240 actions and 17,304 observations,
but SDK NEWT crashed at Hopper reset and automatically requeued one item (no OOM). Its eventual
strict comparison passed; the collector retained recovery as failure. HORA levels 2/3 differed at
observations 212 and 260: four observation differences, zero action differences, equal returns and
outcomes. R11 SDK command/eval sums were 954.625/741.810 s; native sums were 1,588.277/1,041.329 s.

The separate [HORA repeatability investigation](HORA_NATIVE_REPEATABILITY.md) still **FAILS strict
observation equality**, despite the exact single R12 cohort. Twelve full 300-tick runs contain
3,600 actions / 3,612 observations; all actions, record fields, outcomes, returns and masks match.
Native level 3 repeats vary at observation 260; SDK repeats at 212/260. Differences are one ULP
at state index 70 (FSR 3). At frozen native level 3/tick 212, 64 contact queries produced two raw
vector/norm variants without step/evaluate/get_obs and with qpos unchanged. SAPIEN float atomicAdd
is a source-consistent mechanism, not a proven cause. Raw failed comparisons remain failed.

## R12 traced measurements

Each row is a separate family command. RPC is SDK mean host round trip. GPU/VRAM are device-wide
250 ms NVML samples over that command; peaks are sampled. CPU is waited-child accounted mean cores.

| Family | SDK command / eval s | Native command / eval s | RPC ms | GPU busy % | VRAM mean / peak GiB | CPU cores |
| --- | --- | --- | --- | --- | --- | --- |
| Humanoid | 102.837 / 98.115 | 114.600 / 88.086 | 1.456 | 28.812 | 2.964 / 3.881 | 0.949 |
| Panda | 345.303 / 251.316 | 716.131 / 373.623 | 21.558 | 8.117 | 2.812 / 5.385 | 0.754 |
| G1 | 64.999 / 45.725 | 86.754 / 45.596 | 98.134 | 9.337 | 3.423 / 5.559 | 1.438 |
| PPO | 122.656 / 118.009 | 132.254 / 92.477 | 1.321 | 10.484 | 2.127 / 4.080 | 0.813 |
| NEWT | 126.384 / 119.560 | 162.839 / 96.559 | 1.031 | 14.042 | 2.575 / 4.256 | 0.834 |
| HORA | 103.106 / 98.988 | 122.179 / 97.898 | 3.270 | 52.995 | 4.445 / 5.922 | 0.940 |
| TriFinger | 89.442 / 84.592 | 95.092 / 69.579 | 2.202 | 30.255 | 2.110 / 3.571 | 0.864 |
| ROBEL | 59.810 / 56.428 | 59.574 / 44.835 | 0.847 | 21.110 | 1.808 / 3.309 | 0.922 |
| Unitree | 72.689 / 67.173 | 67.360 / 56.820 | 2.813 | 44.747 | 3.223 / 4.131 | 0.848 |
| Go2 | 15.941 / 10.819 | 13.910 / 8.800 | 2.257 | 7.492 | 1.645 / 3.820 | 0.629 |
| Spin | 14.806 / 10.623 | 15.761 / 10.980 | 1.805 | 12.831 | 1.975 / 3.820 | 0.754 |
| Octo | 291.309 / 232.836 | 210.050 / 192.151 | 156.545 | 34.939 | 3.719 / 5.320 | 1.049 |

R12 SDK command/eval rates are 168.596/198.964 episodes/h and 13.652/16.111 steps/s; native rates
are 132.257/201.800 and 10.710/16.341. These are sequential family sums, not a persistent-suite
clock. SDK evaluation is `evaluation_s`; native uses post-load `formal_wall_s`, except PPO/Octo
`evaluation_wall_s`. Tracing, hashing and serialization are included. Different lifecycle clocks
prevent interpreting ratios as speedups. Host step/observation, staging and RPC intervals nest
inside rollout; RPC includes transport, inference and response validation. Isolated GPU inference,
CPU/GPU rendering, queue-only wait and SM/Tensor/RT throughput are **N/A**, not zero. GPU busy is
activity, not arithmetic throughput. Dedicated execution-timeout counters are null; normal native
horizon truncation is recorded separately.

## Evidence and remaining gates

The current R16 traced cohorts retain these SDK measurements. Evaluation rates include
construction, reset, Worker lifecycle, hashing and result handling; they are not steady-state
performance measurements. RPC includes transport, server inference and response validation.

| R16 cohort | Eval episodes/h / steps/s | RPC ms | GPU busy mean / peak % | VRAM mean / sampled peak GiB | Accounted CPU cores | Success-once |
| --- | --- | --- | --- | --- | --- | --- |
| New-two N1 full | 345.259 / 14.386 | 1.726 | 12.727 / 66 | 2.006 / 3.983 | 0.723 | 0/2 |
| New-two N2 prefix/tail | 614.565 / 0.854 | 6.652 | 1.399 / 22 | 1.613 / 3.983 | 0.642 | 0/6 |
| New-two N2 full | 690.417 / 28.767 | 1.957 | 12.536 / 61 | 2.052 / 3.983 | 0.757 | 0/4 |
| Old-eleven N1 prefix | 408.331 / 0.567 | 2.678 | 1.764 / 48 | 1.669 / 4.080 | 0.653 | 0/11 |
| New-two N2 multi-group prefix | 1,139.809 / 1.583 | 5.025 | 2.557 / 40 | 1.860 / 5.113 | 0.774 | 0/10 |

Success-at-end is zero in all five R16 cohorts. There are no Worker failures/OOMs; ten planned
profile recycles in the old-PPO regression are expected. Prefix truncation flags remain false,
while full horizons normally truncate. Independent GPU inference/render, queue-only wait and
SM/Tensor/RT throughput remain N/A. The accompanying summaries retain native resource samples,
all lifecycle fields and original commands. These measurements make no speed gain claim.

Raw R9/R10/R11 evidence remains outside Git under
`ROOT=/autodl-fs/data/roboscope/maniskill-review-20261109` and in a retrieved 1,483-file archive,
SHA256 `e9a62adf371934edad08da6c6681e2c3f9468218f06e94f3a28e0fb6dd19b7fa`.
Independent integrity checking verified every member, all 64 R11 trace pairs and 24 family result
files; this preserves the failed measurements, rather than passing their evaluation.

`LOCAL=/root/autodl-tmp/maniskill-issue25-review-20261109` holds results under
`review-r12-results`, `review-r13-results` and `review-r16-results`; generation-specific
source/manifests/metadata use `review-r13-data` / `review-r16-data`. Failed R14/R15 outputs remain
in their original directories. Retrieved `mask-review-20261110/full-review-r12-summary.json`,
`r13/*-summary.json` and `r16/*-summary.json`, with their
exhaustive trace audits, retain commands, lifecycle/resource fields, outcomes, failures and provenance.
The R11 Markdown summary's stale R9 heading does not change its JSON label `full-review-r11`.

The [canonical-prior evidence release](https://github.com/QLL-CCDD/RoboScope-Kit/releases/tag/maniskill-issue25-canonical-priors-20261111)
stores current R16 and failed R14/R15 data, source snapshots, explicit failure inventories,
build/self-test/check receipts and final-head bindings. The
[previous native-review release](https://github.com/QLL-CCDD/RoboScope-Kit/releases/tag/maniskill-issue25-native-review-20261111)
stores the R13 archive and source/head binding, with separate immutable R12 and failed R9/R10/R11
archives and their checksums. It preserves the R13 inode failure, successful retry and HORA failed
repeats instead of relabeling them. The
[historical R8 evidence release](https://github.com/QLL-CCDD/RoboScope-Kit/releases/tag/maniskill-issue25-compatibility-20261109)
retains its original b47 scope. Historical fifteen-task, 100-episode comparisons predate complete
native-mask checking and are not relabeled. Exact-head CI must be checked after the PR update;
the local checks do not assert a GitHub CI pass. Issue 31 remains separate; these traced correctness checks
make no throughput claim.
