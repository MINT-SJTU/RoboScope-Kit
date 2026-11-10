# ManiSkill native-reference review validation

**2026-11-11: b51 validation PENDING; historical b50/R11 matrix FAILED.** This report tracks
[issue 25](https://github.com/MINT-SJTU/RoboScope-Kit/issues/25) and
[PR 99](https://github.com/MINT-SJTU/RoboScope-Kit/pull/99). Neither issue closure nor full acceptance
of all 55 upstream task IDs is supported. The candidate incorporates main `329ef98`, preserving
its RLBench registration alongside this unmerged ManiSkill integration.

The subsequent b52 candidate adds six model-side references to existing canonical profiles.
G1 TransportBox/Apple use released state PPO actors; Peg/Plug project real TCP/object fields into
the unchanged LiftPeg actor; dual-Panda tasks compose unchanged 4D actors with pinned frame
calibration. Their independent CPU contracts pass; fresh b52 builds and real comparisons are pending.
All 89 existing profile definitions, seeds and scoring rules remain unchanged. These six candidates
leave nine available profiles without a selected compatible model, plus eight asset-deferred profiles.

## Active scope and protocol

The user deferred original-asset recovery for AssemblingKits, TurnFaucet, OpenCabinetDoor and
OpenCabinetDrawer on 2026-11-11. Active verification covers the other **51 task IDs and their claimed
profiles**. The b51 accounting was **89 profiles = 64 R11-selected + two additional Humanoid
profiles + 15 then-unmatched + eight asset-deferred**; b52 splits those fifteen into six pending
candidates and nine unmatched profiles. Selected siblings do not validate
unmatched state, RGB, root-state or controller contracts; see the
[profile gaps](REMAINING_COMPATIBILITY.md#remaining-profile-acceptance-gaps).

The strict comparator checks actual `terminated` and `truncated` booleans from both paths and rejects
missing, malformed or horizon-inconsistent masks. Under `ignore_terminations=True`, `auto_reset=False`,
full episodes end with `terminated=false`, `truncated=true`; pre-horizon prefixes keep both false.
Success/failure metrics remain separate. No action, observation, return or mask tolerance was relaxed.

Generic native references independently build Gym environments, features and action execution,
bypassing Kit Worker/RPC while sharing the Policy implementation. This checks SDK integration;
independent author-code/numerical probes have narrower model-transfer scope. Agreement alone does
not independently validate Policy inference or task competence.

## Current b51 changes and evidence

The b50 generic native reference uses sequential per-task child processes, retains child commands,
logs and result hashes, and preserves the virtual-environment interpreter path. Its child phase sums
remain distinct from the parent command clock.

The b51 `WorkerLoop` adds optional `recycle_on_group_change=False`; ManiSkill opts in. Before a group
change, the worker leaves the new claim unevaluated and exits for the existing orchestrator's planned
recycle. The orchestrator requeues that one item, retains the Policy replica, and reuses a worker for
same-profile groups. CPU queue probes and a four-episode full-horizon GPU transition check passed;
the complete multi-task matrix remains pending.

Canonical HumanoidWalk/Run now project the 185-value ManiSkill state to the same 67-value actor input.
`include_joint_names` adds actual joint names as metadata only; all 96 Root-input CPU samples are unchanged.
The 64-sample CPU canonical features were bitwise equal to the old Root features. Actor/action ASTs
were unchanged; independent native action checking had maximum absolute error `1.19e-7`.
Both canonical profiles passed N2 five-tick GPU prefixes and R12 full-horizon GPU comparisons.
The complete Humanoid family passed five profiles at 1,000 ticks each: 5,000 action pairs and
5,005 observation pairs, covering the three Root profiles and canonical Walk/Run without adding task IDs.

## Historical checks retained with their source identities

| Evidence | Recorded result |
| --- | --- |
| b48/R9 synthetic masks | All ten contract/rejection cases passed. |
| b48/R9 real-model masks | N2 PPO: two complete 50-tick PickCube episodes; N2 Octo: two complete 60-tick PutCarrot episodes; strict records, actions, observations and masks matched. Generic PPO reference also passed. |
| b47/R8 prefix recheck | All 64 profiles / 51 task IDs passed the stricter offline mask check; original evidence remains b47. |
| R9 full-horizon failures | Native NEWT crashed at Hopper reset; HORA level 3 observation first differed at tick 212. Failure logs and partial traces were retained. |
| R10 stopped batch | Resolving the virtual-environment Python symlink selected the base interpreter without NumPy; fixed in b50, with stopped logs retained. |
| b50/R11 fresh runtime | Builder and self-test passed; 247 executable/config files and 94 installed SDK files per environment matched the source audits. Ant/Hopper two-task five-tick child smoke passed. |
| b50 source checks | 270 existing pytest cases, Ruff and Python-support checks passed; unchanged build scripts retain their syntax/ShellCheck pass. |
| Earlier b48 wheel | Clean build/install, 92 module imports and 270 existing pytest cases passed; this is not a b51 wheel check. |

## b50/R11 full-horizon result

All **64 selected profiles / 51 task IDs** have final SDK/native records at K1/G1/N1, seed 0, one
episode per original native horizon. Exhaustive auditing covered **17,240 action pairs and 17,304
observation pairs**: zero action differences, four observation differences, and no structural errors.
This is one-sample integration evidence; it does not replace the prescribed 100-episode acceptance.

| Failure family | Final records and retained failure |
| --- | --- |
| NEWT | SDK worker crashed with CUDA illegal memory access during Hopper reset; automatic recovery requeued one item. Final strict NEWT comparison passed, but the collector correctly failed the family for recovery. No OOM occurred. |
| HORA | Levels 2 and 3 each differed in observation hash at ticks 212 and 260. Actions, episode returns and final outcomes matched; strict trajectory comparison failed. |

R11 is **FAILED**, with one recovery family and one hash-mismatch family. Counts of final records or
matching profiles must not be presented as clean accepted coverage.

The R11 numeric diagnostic initially had console-only scope. The subsequent
[R12 native repeatability investigation](HORA_NATIVE_REPEATABILITY.md) completed twelve independent
300-tick runs: 3,600 actions and 3,612 observations. Within each level, all actions, record fields,
outcomes, returns and actual masks matched. Native level 3 repeats varied at observation 260; SDK
repeats varied at 212/260. Numeric captures isolate one-ULP differences to state index 70 (FSR 3).
At frozen native level 3/tick 212, 64 contact queries returned two raw-vector/norm variants without
step, evaluate or get_obs, while qpos stayed bitwise unchanged. This directly demonstrates native
contact-readback variation. SAPIEN float atomicAdd is a consistent mechanism, not a proven cause.
**Strict observation comparison still fails**; no quantization, tolerance change or waiver was applied.

## R11 traced measurements

Each row is a separate family command. RPC is SDK mean host round-trip time; GPU busy and VRAM
are SDK device-wide samples over the command. CPU is accounted mean cores. VRAM uses GiB.

| Family | SDK command / eval s | Native command / eval s | RPC ms | GPU busy % | VRAM mean / peak | CPU cores |
| --- | --- | --- | --- | --- | --- | --- |
| Humanoid | 59.755 / 46.669 | 69.626 / 51.675 | 1.676 | 27.874 | 3.407 / 4.935 | 0.811 |
| Panda | 153.903 / 127.406 | 594.683 / 334.190 | 22.187 | 18.270 | 4.950 / 6.573 | 1.127 |
| G1 | 50.626 / 35.277 | 76.056 / 42.970 | 97.223 | 10.709 | 4.131 / 6.808 | 1.631 |
| PPO | 53.394 / 43.975 | 126.244 / 82.339 | 1.396 | 21.186 | 3.631 / 5.238 | 0.770 |
| NEWT | 131.783 / 69.457 | 177.326 / 91.709 | 1.113 | 22.549 | 3.694 / 5.376 | 0.504 |
| HORA | 88.109 / 79.414 | 111.263 / 88.250 | 3.520 | 58.486 | 6.282 / 8.042 | 0.909 |
| TriFinger | 54.735 / 46.651 | 85.893 / 60.759 | 2.356 | 47.259 | 3.443 / 4.758 | 0.854 |
| ROBEL | 37.023 / 30.738 | 58.905 / 40.718 | 0.894 | 27.657 | 2.986 / 4.363 | 0.773 |
| Unitree | 61.720 / 54.357 | 66.329 / 53.890 | 2.971 | 49.970 | 3.951 / 5.372 | 0.889 |
| Go2 | 18.182 / 9.795 | 11.724 / 6.690 | 2.532 | 6.261 | 1.552 / 3.810 | 0.563 |
| Spin | 17.131 / 9.570 | 13.638 / 8.447 | 2.383 | 9.734 | 1.785 / 3.810 | 0.651 |
| Octo | 228.263 / 188.501 | 196.590 / 179.692 | 157.218 | 41.725 | 5.005 / 6.575 | 1.304 |

SDK family sums are **954.625 s command / 741.810 s evaluation**: command/evaluation rates are
241.4/310.6 episodes/h and 18.059/23.240 steps/s. Native sums are 1,588.277/1,041.329 s.
These are sequential family sums, not one persistent-suite clock. SDK evaluation is `evaluation_s`;
native uses family-specific post-load intervals (`formal_wall_s`, or PPO/Octo `evaluation_wall_s`).
Different clocks and simulator lifetimes prevent interpreting ratios as speedups. Hashing,
synchronization and serialization are included; this is not an untraced performance baseline.

Step/observation, staging and RPC host intervals are nested in rollout and cannot be added as
independent phases. RPC includes transport, inference and response validation. Pure GPU inference,
isolated CPU/GPU rendering, queue-only wait and SM/Tensor/RT throughput are **N/A**, not zero.
NVML busy is activity rather than arithmetic throughput; sampled peaks are not hardware maxima.

## Remaining gates and evidence location

| Gate | Current status |
| --- | --- |
| Current b51 GPU build and source/installed audits | R12 fresh simulator and Octo builds PASS after SSH restoration on November 11. All 247 executable/config files and 94 SDK files in each environment match frozen source `42e6b24`. |
| Current b51 source checks and wheel | 270 existing source tests PASS in 8.11 s; Ruff and maintained-Python checks PASS. Wheel/sdist and Twine PASS; 92 installed modules import; 270 installed-wheel tests PASS in 5.09 s. |
| Fresh runtime smoke / ordinary quick test | R12 GPU self-test and real PickCube quick test PASS: 1/1 success, native 50-tick horizon; 19.258 s command / 11.025 s evaluation. Ant/Hopper A,A,B,B: four full-horizon episodes, 3,200 steps, exact strict comparison, one planned recycle, zero failures/OOMs, two Worker PIDs and one retained Policy PID. Canonical Humanoid Walk/Run N2 prefixes PASS: four episodes, 20 actions / 24 observations per path. |
| Full-horizon GPU comparisons | R12 matrix is RUNNING for 66 profiles / 51 task IDs. Humanoid's five profiles PASS, including canonical Walk/Run; remaining families are running. HORA independent native variation is demonstrated, but strict observation equality still FAILS. |
| Exact-head CI | PENDING. The old remote PR head `b443` reports `action_required` with zero jobs; it is not CI evidence for the current head. |
| Current evidence publication | PENDING; no new R11/b51 evidence release has been created. |

Raw R9/R10/R11 evidence remains outside Git under
`ROOT=/autodl-fs/data/roboscope/maniskill-review-20261109` and in a retrieved 1,483-file archive,
SHA256 `e9a62adf371934edad08da6c6681e2c3f9468218f06e94f3a28e0fb6dd19b7fa`.
Independent integrity checking verified every member, all 64 R11 trace pairs and 24 family result
files; this preserves the failed measurements, rather than passing their evaluation. The sibling
`mask-review-20261110/` also contains the summaries and exhaustive trace audit. The JSON label is
`full-review-r11`; its retained Markdown companion has a stale R9 heading. Commands, lifecycle
intervals, resources, failures and source identities are retained. No new public release exists yet.

The [historical R8 evidence release](https://github.com/QLL-CCDD/RoboScope-Kit/releases/tag/maniskill-issue25-compatibility-20261109)
retains its original b47 scope. Historical fifteen-task, 100-episode comparisons predate complete
native-mask checking and are not relabeled. Issue 31 remains separate; throughput work follows the
available-scope correctness verification. This review makes no new throughput claim.
