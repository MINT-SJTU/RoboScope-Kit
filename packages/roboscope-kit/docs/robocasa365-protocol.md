# RoboCasa365 multi-task evaluation protocol

The `robocasa365` benchmark is separate from the existing `robocasa/tabletop`
GR1 benchmark. It evaluates the official RoboCasa365 multi-task task sets in
**pretrain scenes** with PandaOmron. The task category `composite_unseen` does
not select the distinct `target` scene split.

This integration contributes to
[issue #97](https://github.com/MINT-SJTU/RoboScope-Kit/issues/97). The full
2,500-rollout matrix and matched native evaluator comparison are pending, so
acceptance remains draft work. The b35 clean build and real five-slot smoke are
recorded in the benchmark validation document; they do not replace those full
acceptance checks. GR1 work in issue #16 is separate.

## Frozen workload

RoboCasa is pinned to [`3443cdf3dd20f9ae8ed41adba442059b506891d0`](https://github.com/robocasa/robocasa/tree/3443cdf3dd20f9ae8ed41adba442059b506891d0)
(v1.0.1), with robosuite
[`5ce6643f3092639d08f7b0f90ed1c6a84f50552c`](https://github.com/ARISE-Initiative/robosuite/tree/5ce6643f3092639d08f7b0f90ed1c6a84f50552c).
The complete ordered task lists and per-task horizons are frozen in
[`protocol.py`](../src/roboscope_kit/runtime/benchmarks/robocasa365/protocol.py).
The build and every worker validate them against the installed upstream registry.
These horizons already include the v1.0.1 1.5x increase; the runtime does not
multiply them again or apply the older GR1 suite's 720-step limit.

| Suite | Tasks | Rollouts per task | Expected records |
|---|---:|---:|---:|
| `target50` | 50 | 50 | 2,500 |
| `atomic_seen` | 18 | 50 | 900 |
| `composite_seen` | 16 | 50 | 800 |
| `composite_unseen` | 16 | 50 | 800 |

`target50` is the concatenation of the other three ordered sets. Smaller runs
are labeled with `full_suite: false`. `--quick-test` runs the first task of the
selected suite once, using one native environment; it keeps that task's full
horizon and is only a smoke test. `--episodes` accepts multiples of five up to
50 for reduced runs with the native five-slot grouping. No dummy or padded
episodes are counted.

## Scenario sampling and environment lifecycle

The reference is the official Diffusion Policy
[`eval_robocasa.py`](https://github.com/robocasa-benchmark/diffusion_policy/blob/41212698b6f481ed92a55f0d5f1778ec56bea417/eval_robocasa.py)
and its `RobomimicImageRunner` / `MultiStepWrapper` execution path. Each task
creates five environment slots with constructor seeds `1111111 + slot`. A
slot retains its environment RNG across ten sequential episode resets.
The official runner's displayed rollout labels `100000 + episode_index` are
**identifiers, not seeds passed to reset**: its per-rollout `env.seed(seed)`
call is commented out. Kit preserves that distinction.

The pinned kitchen sets `load_model_on_init=False`. The Gym wrapper's
constructor performs the first environment reset; each evaluated episode then
performs one more unseeded reset. Kit uses that same wrapper and does not add
an extra warmup reset. Episode index `i` belongs to slot `i % 5` and the
slot's stream episode `i // 5`. It records constructor seed, slot, stream
index, rollout label, scenario metadata digest, initial physics state digest,
and initial observation digest with each result. A deterministic scenario-plan
description and its SHA-256 appear in run metadata.

A work item owns an entire task and all five slots, so changing Kit's K/G
does not redistribute the constructor streams or split the native termination
barrier. K is the number of Kit workers; **each formal worker manages five
simulation environments**. G is the independent number of Policy replicas.
Environment wrappers persist between episode groups of a task and close when
the task completes; native hard resets can rebuild their underlying model and
renderer. Wrappers are never reused across different tasks.

The first implementation steps each slot in the worker process. The reference
runner uses subprocess slots. This process-layout difference and model RNG
behavior require measured comparisons; matching the intended sampling rules
alone does not establish native equivalence. `native_equivalence_verified`
remains `false` in results until a separate evidence-backed comparison exists.

The reproducible runtime requires the reviewed EGL ownership/restoration fix,
checked by factory self-test and worker marker-2 validation. It preserves a live
renderer when another slot is destroyed, including delayed hard-reset cleanup.
It changes no scenario ordering, seeds, reset counts, actions, observation
frequency or success rules. Earlier uncorrected multi-slot scores/timings are
superseded; see the
[integration validation](../../../benchmarks/robocasa365/validation-20261107.md).

## Observations and actions

The robot is PandaOmron at 20 control steps/s. The official Gym wrapper renders
three RGB cameras at 256 x 256, vertically flips each image exactly once, and
supplies raw float32 state and episode language. No depth is requested.

| Kit feature | Upstream source | Shape / dtype |
|---|---|---|
| `images.agentview_left` | `video.robot0_agentview_left` | `(2,256,256,3)`, uint8 |
| `images.agentview_right` | `video.robot0_agentview_right` | `(2,256,256,3)`, uint8 |
| `images.wrist` | `video.robot0_eye_in_hand` | `(2,256,256,3)`, uint8 |
| `state.end_effector_position_relative` | same state suffix | `(2,3)`, float32 |
| `state.end_effector_rotation_relative` | same state suffix, xyzw quaternion | `(2,4)`, float32 |
| `state.gripper_qpos` | same state suffix | `(2,2)`, float32 |
| `state.base_position` | same state suffix | `(2,3)`, float32 |
| `state.base_rotation` | same state suffix, xyzw quaternion | `(2,4)`, float32 |
| `language.instruction` | `annotation.human.task_description` | string |

Images and states contain the last **two consecutive physical-step frames**,
not the previous and current inference-boundary frames. The initial frame is
duplicated. Inference at step 8 sees frames 7 and 8. Every physical step still
generates its native observation; no rendering omission is enabled in this
baseline. Model normalization, language embeddings and model inference belong
to the Policy example and are not credited as runtime acceleration.

The pinned Gym wrapper declares every state space as `Box(-1, 1)`, including
raw world-frame `state.base_position`. Real kitchen coordinates can exceed those
bounds, so Gym can warn that an observation is outside its declared space. The
checker does not clip or modify observations. The released diffusion model does
not consume base position or base rotation; Kit validates the actual shapes,
dtypes and finite values required by its model contract. Preserve the native
raw values rather than clipping them to silence this declaration warning.

The action representation is `robocasa365.panda-omron-hybrid-v1`, with an
`(8,12)` chunk: end-effector delta position 3, delta rotation 3, gripper 1,
base/torso 4, and hybrid control mode 1. The worker passes each vector through
upstream `convert_action` and the Gym wrapper's controller conversion. It
does not introduce action clipping or another normalization. Gripper and
control mode use the wrapper's threshold of 0.5. Model and benchmark must
agree on this representation; GR1's 29-dimensional joint action is incompatible.

## Success, termination and results

The worker checks native success after every physical step and scores an
episode as successful if any step reports success, as the native runner's
maximum sparse reward does. An individual success does not stop its slot.
At each eight-step boundary, the group stops when all slots are done or when
all slots' `info["success"][0]` values are true. Here `[0]` is the first of the
latest two physical-step values; substituting the latest value or any-ever
success would change the native rule. Each slot truncates at the exact
task-specific horizon, including a partial last action chunk. A done slot
continues to be present in batched inference and ignores further actions,
matching the native MultiStepWrapper.

Kit stores actual executed steps, episode outcomes, task/split/stream
identities, terminal reasons and reset fingerprints. Aggregation validates
record identity, duplicate records, stream ownership and step bounds. It
reports per-task success, per-split pooled episode success and per-split mean
task success, with expected/completed coverage. These averages coincide for
a complete 50-rollout-per-task matrix. The official leaderboard weights the
three split means by their task counts: `(18 * Atomic-Seen + 16 * Composite-Seen
+ 16 * Composite-Unseen) / 50`, rounded to one decimal only for presentation
([pinned scoring code](https://github.com/robocasa-benchmark/leaderboard/blob/0bdce13f42ff3c5bb96f9ff33c6ee6113c205ef5/scripts/weighted_overall.py)).
Kit retains raw rates; its overall pooled rate equals that macro average only
for a complete, balanced target50 matrix. It is not a three-split equal average.
The published diffusion leaderboard entry used RoboCasa 1.0.0; this 1.0.1
integration does not claim to reproduce its published percentage.
Formal terminal accounting additionally
requires every expected record, without gaps or duplicates.

Optional video uses the already generated right-agent image at 10 fps, encoding
physical steps 1, 3, 5 and so on, matching the native recorder's reset counter
and the checkpoint's render camera. Kit's `save_videos` records every rollout;
native comparisons must disable video on both sides or match the recorded
rollout selection. Files have unique worker/task/rollout paths. Keep output
settings identical between compared runs. A run also saves each task's first
real Policy input batch and returned actions under `evidence/worker-*/` for
native preprocessing/action parity checks. These artifacts stay outside Git.

## Profiling and correctness limits

Each work result separates environment construction, resets, stepping,
Policy round trips, video writes, cleanup and total task wall time. Those
worker-local inclusive timings locate bottlenecks; summing them across workers
does not produce evaluation wall time. Episode elapsed values share the group
interval and must not be summed across the five slots. Use the runtime's
formal wall clock, completed episodes/h and actual steps/s for throughput.

`--profile` captures the first native episode **group** of a worker. One
NVTX episode range covers the group's five Policy resets, batched inference,
physical steps and end notifications; successive slot resets do not end the
capture prematurely. Environment construction and simulation resets precede
that range and use separate wall timers. Nsight's report covers the worker;
the independent Policy Server's GPU work requires separate capture. Profiled
runs are diagnostic and are not formal speed measurements.

The real smoke test, full matrix and matched native comparison are separate
evidence requirements. A passing
import self-test does not establish a simulator rollout, a successful task,
native score equivalence or unchanged success rate. In
particular, the released diffusion model uses random sampling: match the
Policy RNG, batch size/order, observation tensors and scenario fingerprints
when comparing native and Kit; report unmatched randomness explicitly.
