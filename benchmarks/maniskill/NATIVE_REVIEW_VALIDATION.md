# ManiSkill native-reference review validation

**2026-11-11: selected R12/b51 and R13/b52 correctness cohorts PASS; acceptance remains partial.**
Cumulative coverage is **72 profiles / 51 task IDs**: 66 compared in R12 and six additional canonical
profiles in R13. This is not a fresh b52 rerun of all 72 profiles, acceptance of all 55 tasks, or
100-episode-per-task acceptance. [Issue 25](https://github.com/MINT-SJTU/RoboScope-Kit/issues/25)
and [PR 99](https://github.com/MINT-SJTU/RoboScope-Kit/pull/99) remain open.

R12 binds b51 `source-r12` at code head `42e6b24bbde9e26bfe714e92786d2891bfc37d4b`;
R13 binds b52 `source-r13` at `db473261e41c4663a7350bbb79816789fc632f48`.
The final documentation commit follows measurement without changing the measured runtime or Policy
code. The candidate incorporates main `329ef98`, preserving its RLBench registration; this ManiSkill
integration has not been merged.

## Scope and strict comparison

The user deferred original-asset recovery for AssemblingKits, TurnFaucet, OpenCabinetDoor and
OpenCabinetDrawer. The inventory is **89 profiles = 72 compared + nine unmatched available + eight
asset-deferred**. Active verification covers the other 51 task IDs and their claimed profiles.
Selected siblings do not validate unmatched observation/controller contracts; see the
[current gaps](REMAINING_COMPATIBILITY.md#remaining-profile-acceptance-gaps).

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
seed 0; R13 prefixes use seed 0 at N1 and seeds 0/1/2 at N2; full N2 uses seeds 0/1.

| Cohort | Profiles / task IDs | N / episodes | Action / observation pairs | SDK command / eval s | Native command / eval s |
| --- | --- | --- | --- | --- | --- |
| R12 full original horizons | 66 / 51 | 1 / 66 | 19,240 / 19,306 | 1,409.283 / 1,194.185 | 1,796.505 / 1,177.403 |
| R13 N1 five-tick retry1 | 6 / 6 | 1 / 6 | 30 / 36 | 65.543 / 57.603 | 70.542 / 42.039 |
| R13 N2 five-tick tail | 6 / 6 | 2 / 18 | 90 / 108 | 74.340 / 66.520 | 84.656 / 55.241 |
| R13 full original horizons | 17 / 15 | 1 / 17 | 1,580 / 1,597 | 215.670 / 203.217 | 218.853 / 155.793 |
| R13 new-six full horizons | 6 / 6 | 2 / 12 | 1,400 / 1,412 | 87.704 / 79.502 | 90.953 / 67.161 |

R13's 17-profile N1 cohort includes the six new profiles and eleven official PPO regressions;
dual task IDs overlap across controller profiles. The N2 prefix explicitly exercises lane groups
of two then one. R12 recorded 54 planned profile recycles, separately from zero failure recovery.
Its 52 success-scored samples had 13 success-once / seven success-at-end; fourteen return-only
samples retain null success. The new six R13 full-horizon profiles had one success-once at N1 and two at N2;
matching records do not mean that the transfer policies solve their target tasks.

The six added profiles have the following full-horizon paired records. N1 uses one episode at
seed 0; N2 uses two episodes at seeds 0/1. Success-at-end counts equal success-once counts in
these six samples. Counts below describe the observed samples, not population success rates.

| Added profile | Original horizon | N1 action / observation pairs | N2 action / observation pairs | Successes N1; N2 |
| --- | ---: | ---: | ---: | --- |
| `unitree_g1_transport_box_state` | 100 | 100 / 101 | 200 / 202 | 1 of 1; 2 of 2 |
| `unitree_g1_place_apple_in_bowl_state` | 100 | 100 / 101 | 200 / 202 | 0 of 1; 0 of 2 |
| `peg_insertion_side_state` | 100 | 100 / 101 | 200 / 202 | 0 of 1; 0 of 2 |
| `plug_charger_state` | 200 | 200 / 201 | 400 / 402 | 0 of 1; 0 of 2 |
| `two_robot_pickcube_state` | 100 | 100 / 101 | 200 / 202 | 0 of 1; 0 of 2 |
| `two_robot_stackcube_state` | 100 | 100 / 101 | 200 / 202 | 0 of 1; 0 of 2 |

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

## Retained failures and HORA repeatability

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

Raw R9/R10/R11 evidence remains outside Git under
`ROOT=/autodl-fs/data/roboscope/maniskill-review-20261109` and in a retrieved 1,483-file archive,
SHA256 `e9a62adf371934edad08da6c6681e2c3f9468218f06e94f3a28e0fb6dd19b7fa`.
Independent integrity checking verified every member, all 64 R11 trace pairs and 24 family result
files; this preserves the failed measurements, rather than passing their evaluation.

`LOCAL=/root/autodl-tmp/maniskill-issue25-review-20261109` holds R12/R13 results under
`review-r12-results` / `review-r13-results`; R13 source/manifests/metadata are in `review-r13-data`.
Retrieved `mask-review-20261110/full-review-r12-summary.json` and `r13/*-summary.json`, with their
exhaustive trace audits, retain commands, lifecycle/resource fields, outcomes, failures and provenance.
The R11 Markdown summary's stale R9 heading does not change its JSON label `full-review-r11`.

The [current evidence release](https://github.com/QLL-CCDD/RoboScope-Kit/releases/tag/maniskill-issue25-native-review-20261111)
stores the R13 archive and source/head binding, with separate immutable R12 and failed R9/R10/R11
archives and their checksums. It preserves the R13 inode failure, successful retry and HORA failed
repeats instead of relabeling them. The
[historical R8 evidence release](https://github.com/QLL-CCDD/RoboScope-Kit/releases/tag/maniskill-issue25-compatibility-20261109)
retains its original b47 scope. Historical fifteen-task, 100-episode comparisons predate complete
native-mask checking and are not relabeled. Exact-head CI remains pending; the old PR head `b443`
reports `action_required` with zero jobs. Issue 31 remains separate; these traced correctness checks
make no throughput claim.
