# ManiSkill Evaluation Protocol

Protocol `maniskill-v1` is defined by the [benchmark manifest](../../benchmarks/maniskill/benchmark-manifest.json). It pins ManiSkill
`baab60ede2e89167c1b7aaed41a9aa8e690a9d1e` (`3.0.0b15`), SAPIEN `3.0.0b1`,
CPython `3.11.15`, Torch `2.6.0` with CUDA `12.4`, and NumPy `1.26.4`. The runtime builder
records source, package, PhysX library, manifest and asset provenance. Model versions, byte counts,
URLs and SHA-256 digests are declared separately in the model example.

The source pin is retained to keep the existing source, asset and checkpoint evidence
reproducible. It is not the latest ManiSkill release or an inventory of current upstream main.
Changing the pin requires a regenerated task inventory, protocol and model verification,
followed by new correctness evidence. This records the current reproducibility policy;
it does not assert an undocumented original reason for selecting this revision.

## Scope and task settings

The [task inventory](../../benchmarks/maniskill/TASK_INVENTORY.md) covers 58 literal task-directory registrations and nine
runtime scene registrations. Source-backed exclusions identify three helper/reference task-directory
environments and the scene helpers, leaving 55 evaluation tasks. The minimal source template is
inventoried separately. These counts describe the pinned source. A task remains unvalidated until
the exact task and declared observation profile run with a compatible real Policy and matched
native evaluator.

Each named suite uses 100 episodes per task and explicit seeds 0–99. The manifest freezes the robot,
native controller, observation mode, camera mapping, action dimension where known, full horizon,
reconfiguration, success metric and asset IDs for each profile. The selected robot/controller is part
of the protocol. Adding another embodiment requires its own profile and correctness evidence.

| Family | Declared robot/controller | Physics/control Hz | Horizon |
| --- | --- | ---: | --- |
| Tabletop | Profile-specific Panda, PandaStick or dual Panda controller | 100/20 | Per-task upstream registration |
| TriFinger cube rotation | `trifingerpro`, `pd_joint_delta_pos` | 100/20 | 250 |
| Valve rotation | `dclaw`, `pd_joint_delta_pos` | 100/20 | 80, 150 or 300 by level |
| In-hand object rotation | `allegro_hand_right_touch`, `pd_joint_delta_pos` | 100/20 | 300 |
| DM-Control Ant and Humanoid | Native robot, `pd_joint_delta_pos` | 200/40 | 1,000 |
| DM-Control Hopper | Native robot, `pd_joint_delta_pos` | 100/25 | 600 |
| DM-Control Cartpole | Native robot, `pd_joint_delta_pos` | 100/20 | 1,000 |
| Quadruped | `anymal_c` or `unitree_go2_simplified_locomotion`, `pd_joint_delta_pos` | 100/20 | 200 |
| Unitree humanoid | Profile-specific G1/H1, `pd_joint_delta_pos` | 100/20 | 100 or 1,000 |
| FMB assembly | `panda`, `pd_ee_delta_pose` | 100/20 | 500 |
| Cabinet manipulation | `fetch`, `pd_joint_delta_pos` | 100/20 | 100 |
| Bridge digital twins | Native WidowX class and `arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos` | 500/5 | 60 or 120 |

The family frequencies follow the pinned simulation defaults and task overrides. Each task's
registration supplies its horizon. The original tabletop profiles retain their existing controller
selection. The public LiftPegUpright PPO uses a separate `pd_ee_delta_pose` profile, and the two
dual-Panda PPOs use separate `pd_joint_delta_pos` profiles. The inventory links every task definition.
[Simulation defaults](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/utils/structs/types.py)
and [task definitions](https://github.com/mani-skill/ManiSkill/tree/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks)
are the source of these settings.

## GPU grouping, reset and termination

The frozen `benchmark_protocols.maniskill.vector_envs` value is ten environments per Worker.
It is not an autotune bound or `defaults.envs_per_worker`; manifests using that obsolete default
are rejected. Each queue item contains one profile and an ordered complete seed group. Prescribed
and explicitly reduced episode counts must be multiples of this width. The upstream
`ManiSkillVectorEnv` uses `auto_reset=False`, `ignore_terminations=True` and `record_metrics=True`.
Every group performs a full reset with its explicit seed vector and `reconfiguration_freq=1`.
Per-environment random initialization can depend on vector grouping, so a native comparison must
retain the same lane width and seed grouping.

Quick Test selects exactly the first complete vector group of the first task in the suite:
ten episodes with seeds 0-9, matching the first WorkItem of full evaluation. It is not a
one-environment evaluation with a different reset distribution. Probes round their requested
episode count up to complete groups and cap it at the available complete seed groups; neither
autotune nor CLI concurrency changes vector width. The N1/N2 simulator self-tests are separate
integration diagnostics and do not claim to reproduce full-protocol initial states.

Each task executes its complete registered horizon. A Worker plans a process recycle before evaluating
another profile, while the loaded Policy replica and same-profile episode groups remain in place.
The orchestrator requeues the unevaluated next-profile item and restarts that Worker context;
planned recycling is recorded separately from failures and retries. This avoids reusing the PhysX
contact-query context across profiles without changing episode identities or seed grouping.

The Worker option is named `recycle_on_context_change`: its context includes both profile ID
and vector-group width. Other benchmarks keep its default disabled.

Native task failures and success tracking
remain in the environment, while the wrapper's ignored terminations keep each lane running to
the time limit. The Worker checks terminal masks and native episode lengths. One control action
per environment is requested at every control tick. Episode-local Policy state is reset and ended
using a distinct episode ID for every lane. The benchmark protocol is implemented through the
pinned [vector wrapper](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/vector/wrappers/gymnasium.py)
and [BaseEnv lifecycle](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/sapien_env.py).

## Observation and action mapping

The Policy uses the existing `maniskill.native-controller-v1` action representation.
`reset(EpisodeContext)` supplies task/profile identity, robot, controller, observation mode and
actual action dimension. Each action is checked against the upstream Gym action space, then
forwarded to the native controller. Multi-agent Dict action spaces use the pinned upstream
`FlattenActionSpaceWrapper` and its native ordering.

`state.proprioception` preserves the upstream flattened state order. A `state_dict` profile uses
upstream insertion-order flattening. Visual profiles expose the state already present in that
upstream observation and map declared camera RGB pixels to `images.primary`, `images.wrist` or
`images.auxiliary`. Native image shape and uint8 pixels are retained. The optional
`language.instruction` comes from the profile. Model-specific normalization and preprocessing
belong to the Policy implementation. The optional `include_joint_names` profile setting adds
actual active/controller joint identities to episode metadata without adding observation features.

The four Bridge tasks require the native `rgb+segmentation` observation mode, background overlay,
500/5 Hz configuration and task-specific WidowX robot class. The declared Policy camera is
`3rd_view_camera`. Their segmentation channel supports the native overlay path, and the Policy
receives the resulting RGB features. These profiles require a compatible visual Policy and their
own native comparison. They cannot be accepted through state PPO results.
[Bridge configuration](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/digital_twins/bridge_dataset_eval/base_env.py)
defines that native path.

## Scoring and coverage

For tasks with native success metrics, the profile selects `success_once` or `success_at_end`.
Both native fields and episode return are recorded. The nine `MS-*` control tasks, Unitree H1/G1
standing and AnymalC spin have no native success field, so their profiles select mean episode
return. Their success counts and rates are null. Overall success uses only episodes belonging to
success-scored tasks and records that denominator explicitly. The unweighted task mean uses
those same success-scored tasks. Mean returns remain per task, since reward scales differ.

Every expected profile, episode index and seed must appear exactly once. Aggregation rejects
unknown or duplicate work items, missing or incorrectly identified records, wrong horizons and
inconsistent scores. `formal` requires complete coverage of the named suite at its configured
episode counts. `--quick-test` and reduced `--episodes` runs are incomplete supporting evidence.
Record-tree shard merging follows the same coverage validation. JSON-only `shard_reduce` has
a dedicated ManiSkill reducer keyed by profile ID. Each profile retains `return_sum` and
`return_count`; merging computes its sample-weighted mean and never averages returns across
different tasks. Return-scored episodes contribute neither binary successes nor failures.
The standard result records `success_evaluated_episodes` and `return_evaluated_episodes` when
return-only samples are present. With no success-scored samples, standard successes/failures
are zero and its success rate is null; official return-profile success fields remain null.

Both merge paths check the complete declared profile/count protocol and identical vector width.
New ManiSkill shards additionally record checkpoint and Policy content identities in `shard.build`.
Checkpoint identity covers file content and relative names (excluding Git internals and Python
bytecode); Policy identity covers command arguments, file arguments, Python sources under its
working directory and declared benchmark configuration. Hashing runs once before sharded
evaluation and is timed as `checkpoint_identity_s`. Keep the checkpoint immutable and runtime
caches outside it. This binds the declared inputs, not every dependency imported from another
environment. Absolute installation paths are not a substitute for content identity. Older
shards without these identities must be rerun; a merger cannot reconstruct their provenance.

`shard_merge` reads completed records, validates the deterministic WorkItem partition, episode
index/seed/lane/profile/record identity, actual boolean terminal masks, steps and per-shard
summaries, then reaggregates them. Its coverage is `complete` only after those checks pass.
`shard_reduce` reads only result documents: it rejects inconsistent counts, means, profiles,
protocols and model identities, but coverage remains `accounted`. Consistent fabricated counts
cannot prove that actual records were neither omitted nor duplicated. Use record-tree merging
for that stronger guarantee. Neither path adds concurrent elapsed times or throughput rates.

Native comparisons use the same checkpoint, precision, profile, seeds, vector groups and full
horizons. Compare initial state and observation, actions, native returns, both success outcomes
where available, terminal masks and executed steps. A matching overall score alone cannot pass
correctness acceptance. Repeated runs measure native variation and any systematic divergence.

Every paired SDK and native record must explicitly store boolean `terminated` and `truncated`
values returned by the vector wrapper. Full-horizon records require `false`/`true`; prefixes
require `false`/`false` until the native horizon is reached. These are post-wrapper protocol masks:
`ignore_terminations=True` preserves the full rollout and `auto_reset=False` prevents a reset before
recording. Success/failure signals remain separate native metrics. The comparator rejects missing,
non-boolean or incorrect masks on either side. Historical native PPO archives without these fields
remain historical and require new native output; fields must not be inferred or backfilled.

Multi-profile PPO and generic native references run sequential per-task child processes. This
isolates the upstream simulator lifetime without changing seed groups, observations, actions or
horizons. Each child must return its exact expected episode identities before its unchanged records
are merged. Child phase sums and complete parent command time are reported separately; failed child
logs and commands remain available. This native-reference strategy is not a Kit runtime speedup.

## Assets and build provenance

`benchmarks/maniskill/build-runtime.sh <install-dir>` builds the simulator into a fresh directory.
`MANISKILL_ASSET_SUITES` selects comma-separated named suites. `MANISKILL_ASSET_DIR` selects
shared asset storage. `MANISKILL_ASSET_TASKS` adds comma-separated task IDs, and
`MANISKILL_ASSET_ARCHIVE_DIR` selects an archive cache. The asset helper
accepts repeated `--suite` and `--task-id`, estimates transfer size with `--estimate`, and records
installed-file/tree hashes. Manifest asset IDs include robot downloads as well as task objects.
Source-supplied checksums and immutable official archive overrides are verified before use.

```bash
/path/to/runtime/env/bin/python /path/to/runtime/prepare-assets.py \
  --benchmark-manifest /path/to/runtime/benchmark-manifest.json \
  --asset-dir /path/to/runtime/assets --suite upstream_tasks_v1 --estimate
```

The source/library cache overrides are `MANISKILL_SOURCE_ARCHIVE` and
`MANISKILL_PHYSX_GPU_LIBRARY`. The required uv version is `0.11.14`. Assets and models retain
their upstream licenses and download requirements. Their technical availability is tracked
separately from license permission and real-policy/native acceptance.

## Timing and evidence

Record preparation, cold startup, model loading, environment creation, seeded reset, rollout,
result writing/cleanup, tuning and end-to-end wall time. Formal throughput uses completed
episodes divided by `metrics.json.evaluation_s`, with actual environment steps/s reported beside
it. Include total wall time to account for work outside that timer. Worker CPU intervals can
contain asynchronous CUDA work, and concurrent Worker interval sums can overlap.

Run profiling separately from speed measurements. Measure CPU, GPU busy, memory and VRAM with
their sample window and method. Report unavailable hardware counters or uninstrumented phases
explicitly. Retain raw per-episode records, logs, commands, runtime provenance, resource samples
and exact Kit source identities outside Git. The [validation status](../../benchmarks/maniskill/VALIDATION.md)
separates source/scoring checks from current task/model acceptance. A stored-record regression
fixture is not trained-policy or throughput evidence.

## Integration-prefix diagnostics and model candidates

A manifest may explicitly set `diagnostic_step_limit` to a positive integer no larger than the
profile's native horizon. This bounds only the evaluator loop. Gym still receives the original
`max_episode_steps`, and records retain the actual native terminal masks, prefix length and native
horizon. Such runs are always `evaluation_kind="integration_smoke"` and non-formal, even if the
selected episode count is complete. Do not use prefix throughput as a full-evaluation speedup.
The generic `native_eval_policy.py` reference supports the same prefix; the older PPO/Octo native
entry points reject this setting.

The model-specific suites do not change the original task inventory or tabletop
state/RGB profiles. `reference_ppo_state_v1` selects the two author G1 task-trained
PPO profiles. `newt_state_v1` selects six control tasks, PlaceSphere and the
PullCubeTool pose-controller variant. NEWT currently runs its deterministic actor
prior, not original MPC; source/backend/horizon differences remain unresolved.
`panda_visual_rgb_v1` selects only the existing PegInsertionSide and PlugCharger
RGB profiles with complete seven-dimensional pose actions. Its author baseline
has zero reported success, so these candidates remain unvalidated. Bridge Octo
remains a separate provisional visual example. The [model compatibility notes](../../benchmarks/maniskill/REMAINING_COMPATIBILITY.md)
record these boundaries; a named suite is not accepted task/model coverage.
