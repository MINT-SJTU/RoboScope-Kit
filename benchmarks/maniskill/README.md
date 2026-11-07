# ManiSkill multi-task evaluation

The `maniskill` adapter runs named task suites from `benchmark-manifest.json`. Task IDs, robots,
controllers, observation modes, camera mappings, action dimensions and horizons come from each
profile. Adding a registered task or another fixed suite requires a manifest change, not Worker code.
The built-in catalog covers all 18 tabletop tasks in the pinned ManiSkill source, with separate
state and RGB suites. These are versioned RoboScope suites rather than an upstream aggregate score.

## Suites and protocol

| Suite | Tasks | Episodes per task | Purpose |
| --- | ---: | ---: | --- |
| `pickcube_state_v1` | 1 | 100 | Single-task state baseline |
| `baseline_state_v1` | 3 | 100 | PickCube, RollBall and PushT with official PPO checkpoints |
| `tabletop_state_v1` | 18 | 100 | Complete pinned tabletop task catalog, state observations |
| `tabletop_rgb_v1` | 18 | 100 | Same task catalog, RGB plus upstream visual-mode state |

Protocol `maniskill-v1` pins ManiSkill `baab60ede2e89167c1b7aaed41a9aa8e690a9d1e`
(`3.0.0b15`), SAPIEN `3.0.0b1`, CPython `3.11.15`, Torch `2.6.0`, and NumPy `1.26.4`.
Each profile uses seeds 0-99, GPU PhysX, its listed full horizon, 100 Hz physics / 20 Hz control,
one action per Policy request, `auto_reset=False`, and `ignore_terminations=True`.
Primary scoring is upstream `success_once`; `success_at_end` and actual steps are also recorded.
Results include each task/profile, episode and seed; overall results include episode-weighted success
and the unweighted task mean. Reduced `--episodes` runs and Quick Test are marked incomplete.

| Task | Robot | Controller | Action dimensions | Horizon |
| --- | --- | --- | ---: | ---: |
| AssemblingKits-v1 | panda_wristcam | pd_ee_delta_pose | 7 | 200 |
| LiftPegUpright-v1 | panda | pd_ee_delta_pos | 4 | 50 |
| PegInsertionSide-v1 | panda_wristcam | pd_ee_delta_pose | 7 | 100 |
| PickClutterYCB-v1 | panda | pd_ee_delta_pose | 7 | 100 |
| PickCube-v1 | panda | pd_ee_delta_pos | 4 | 50 |
| PickSingleYCB-v1 | panda_wristcam | pd_ee_delta_pose | 7 | 50 |
| PlaceSphere-v1 | panda | pd_ee_delta_pos | 4 | 50 |
| PlugCharger-v1 | panda_wristcam | pd_ee_delta_pose | 7 | 200 |
| PokeCube-v1 | panda | pd_ee_delta_pos | 4 | 50 |
| PullCube-v1 | panda | pd_ee_delta_pos | 4 | 50 |
| PullCubeTool-v1 | panda | pd_ee_delta_pos | 4 | 100 |
| PushCube-v1 | panda | pd_ee_delta_pos | 4 | 50 |
| PushT-v1 | panda_stick | pd_ee_delta_pos | 3 | 100 |
| RollBall-v1 | panda | pd_ee_delta_pos | 4 | 80 |
| StackCube-v1 | panda_wristcam | pd_ee_delta_pos | 4 | 50 |
| TurnFaucet-v1 | panda_wristcam | pd_ee_delta_pose | 7 | 200 |
| TwoRobotPickCube-v1 | panda_wristcam, panda_wristcam | pd_ee_delta_pos | 8 | 100 |
| TwoRobotStackCube-v1 | panda_wristcam, panda_wristcam | pd_ee_delta_pos | 8 | 100 |

## Build and assets

```bash
UV_BIN=/path/to/uv bash benchmarks/maniskill/build-runtime.sh /path/to/runtime
```

The builder is the only simulator installation entry point. It verifies the source archive and PhysX
library hashes, installs pinned dependencies and the SDK, writes provenance, installs the selected
assets, and runs representative N1/N2 self-tests. Verified source/library cache overrides are
`MANISKILL_SOURCE_ARCHIVE` and `MANISKILL_PHYSX_GPU_LIBRARY`; uv is pinned to `0.11.14`.

By default the three-model baseline uses bundled robots and procedural objects. To install all
external assets needed by the tabletop suites, set `MANISKILL_ASSET_SUITES=tabletop_state_v1`
when building. `MANISKILL_ASSET_ARCHIVE_DIR=/absolute/archive-cache` reuses verified archives
with the original archive suffix (for example `ycb.zip`).
`MANISKILL_ASSET_DIR=/absolute/shared/assets` selects shared storage; the runtime
records that location through its `assets` link and sets `MS_ASSET_DIR` before importing ManiSkill.

```bash
/path/to/runtime/env/bin/python /path/to/runtime/prepare-assets.py   --benchmark-manifest /path/to/runtime/benchmark-manifest.json   --asset-dir /path/to/runtime/assets --suite tabletop_state_v1 --estimate
# Install selected assets after checking the size estimate:
/path/to/runtime/env/bin/python /path/to/runtime/prepare-assets.py   --benchmark-manifest /path/to/runtime/benchmark-manifest.json   --asset-dir /path/to/runtime/assets --suite tabletop_state_v1 --download   --output /path/to/runtime/tabletop-assets.json
```

The helper supports repeated `--suite` or `--task-id`, lists registered upstream tasks with
`--list-tasks`, reuses existing assets and records installed tree hashes. Source-supplied checksums
are verified by the upstream downloader. Some PartNet archives have no upstream checksum; their
installed contents are recorded after download. YCB is pinned to the immutable official dataset revision
`4c134e2e33b11b1fa751c1d7b6afb2e89231b875`; its archive retains the original ManiSkill checksum
`174001ba1003cc0c5adda6453f4433f55ec7e804f0f0da22d015d525d02262fb`.

The three task configurations AssemblingKits, PickClutterYCB and TurnFaucet require archives hosted
at UCSD, whose current outage is tracked by [upstream #1489](https://github.com/mani-skill/ManiSkill/issues/1489).
Their configuration and installer are included; those assets were unavailable during this acceptance.
YCB was installed successfully; the other 15 tabletop task configurations have bundled/procedural
or available asset dependencies. This is availability accounting, not a claim that all 15 were evaluated.

Asset terms remain those of their upstream sources;
Panda/Franka/table retain the existing conservative CC BY-NC 4.0 record. Weights and assets stay
outside Git.

## Policy mapping

Use the existing Policy API and `maniskill.native-controller-v1` action representation.
`reset(EpisodeContext)` identifies the upstream task and includes profile, robot, controller and
actual action dimension. The Worker checks actions against that task's Gym space and forwards them
without extra normalization or clipping. Dual-arm Dict actions use upstream `FlattenActionSpaceWrapper`.

`state.proprioception` contains the original flattened state; `state_dict` uses upstream insertion-order
flattening. RGB profiles expose only state already present in the upstream visual observation,
plus camera pixels mapped to `images.primary`, `images.wrist`, or `images.auxiliary` and the configured
`language.instruction`. No privileged state is added. Image shapes and state/action dimensions may
vary between task profiles; Policy episode state stays isolated by episode ID.

Distributed evaluations use record-tree shard merging. JSON-only `shard_reduce` has no ManiSkill reducer.

The [official multi-model example](../../examples/maniskill/README.md) supplies actual checkpoints
and both SDK and native GPU-vectorized commands. A Policy for the 18-task suite must supply models
for those tasks; the three-model PPO example intentionally serves its baseline suite.

## Integration acceptance and performance work

Integration acceptance uses representative tasks, variable action dimensions, RGB and dual-arm
smokes rather than baseline testing all 18 tasks. See [multi-task validation](MULTITASK_VALIDATION.md)
for actual results, timings and reproduction commands. This issue 25 integration keeps create/close
per vector group; environment reuse, controller optimizations, allocation tuning and additional MPS behavior
belong to issue 31 and are measured separately.
