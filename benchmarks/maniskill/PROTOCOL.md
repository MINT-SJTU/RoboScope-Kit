# ManiSkill evaluation protocol

Protocol `maniskill-v1` is defined by `benchmark-manifest.json`. It pins ManiSkill
`baab60ede2e89167c1b7aaed41a9aa8e690a9d1e` (`3.0.0b15`), SAPIEN `3.0.0b1`,
CPython `3.11.15`, Torch `2.6.0` with CUDA `12.4`, and NumPy `1.26.4`. The runtime builder
records source, package, PhysX library, manifest and asset provenance. Model versions, byte counts,
URLs and SHA-256 digests are declared separately in the model example.

## Scope and task settings

The [task inventory](TASK_INVENTORY.md) covers 58 literal task-directory registrations and nine
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

The default vector width is ten environments per Worker. Each queue item contains one profile and
an ordered seed group. Its last group uses the actual remaining lane count. The upstream
`ManiSkillVectorEnv` uses `auto_reset=False`, `ignore_terminations=True` and `record_metrics=True`.
Every group performs a full reset with its explicit seed vector and `reconfiguration_freq=1`.
Per-environment random initialization can depend on vector grouping, so a native comparison must
retain the same lane width and seed grouping.

Each task executes its complete registered horizon. Native task failures and success tracking
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
belong to the Policy implementation.

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
no ManiSkill reducer.

Native comparisons use the same checkpoint, precision, profile, seeds, vector groups and full
horizons. Compare initial state and observation, actions, native returns, both success outcomes
where available, terminal masks and executed steps. A matching overall score alone cannot pass
correctness acceptance. Repeated runs measure native variation and any systematic divergence.

## Assets and build provenance

`benchmarks/maniskill/build-runtime.sh <install-dir>` builds the simulator into a fresh directory.
`MANISKILL_ASSET_SUITES` selects comma-separated named suites. `MANISKILL_ASSET_DIR` selects
shared asset storage and `MANISKILL_ASSET_ARCHIVE_DIR` selects an archive cache. The asset helper
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
and exact Kit source identities outside Git. The historical [three-task report](MULTITASK_VALIDATION.md)
belongs to its recorded source snapshot. Current task coverage and blockers must be reported
against the current candidate.

## Integration-prefix diagnostics and root observations

A manifest may explicitly set `diagnostic_step_limit` to a positive integer no larger than the
profile's native horizon. This bounds only the evaluator loop. Gym still receives the original
`max_episode_steps`, and records retain the actual native terminal masks, prefix length and native
horizon. Such runs are always `evaluation_kind="integration_smoke"` and non-formal, even if the
selected episode count is complete. Do not use prefix throughput as a full-evaluation speedup.
The generic `native_eval_policy.py` reference supports the same prefix; the older PPO/Octo native
entry points reject this setting.

`transfer_reference_state_v1` names four optional root-kinematics profiles and the separate
PullCubeTool NEWT pose-controller profile. Original profiles remain unchanged. Root profiles add
`state.root_pose` (scene-local xyz and wxyz quaternion), `state.root_linear_velocity` (world xyz)
and `state.root_angular_velocity` (world xyz), read from the native articulation API. Their native
`state.proprioception` vector is unchanged. Episode metadata includes active/controller joint
names and physical/control frequencies. Model-specific frame, joint and control conversions stay
in the example Policy. See [completion validation](COMPLETION_VALIDATION.md) for scope and evidence.
