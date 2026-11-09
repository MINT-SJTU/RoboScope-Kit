# ManiSkill evaluation

The `maniskill` adapter executes manifest-configured tasks through ManiSkill GPU vector
environments and the existing Policy lifecycle. The pinned inventory contains 55 evaluation
tasks across tabletop, dexterity, control, quadruped, humanoid, mobile manipulation and Bridge
digital-twin families. The [inventory](TASK_INVENTORY.md) identifies every task and helper with
upstream source evidence. Protocol settings and observation/action mapping are in
[PROTOCOL.md](PROTOCOL.md).

**Integration acceptance is partial.** Eleven state PPO tasks and four Bridge visual tasks pass
100-episode-per-task comparisons with independent native evaluation. The
[review validation](REVIEW_VALIDATION.md) records all 1,500 paired episodes, metrics and remaining
model, asset and profile gaps. The [completion phase](COMPLETION_VALIDATION.md) adds real-model
five-step diagnostics for 27 more task IDs. The [remaining-compatibility phase](REMAINING_COMPATIBILITY.md)
raises five-step real-policy/native coverage to 51 of 55 task IDs and validates 16 of the original
18 tabletop RGB profiles. Four task IDs still require their original upstream assets.
The [historical three-task results](MULTITASK_VALIDATION.md) describe the previous source snapshot.
Task declarations and simulator checks alone do not establish real-policy acceptance.

| Suite | Tasks | Episodes per task | Configuration |
| --- | ---: | ---: | --- |
| `pickcube_state_v1` | 1 | 100 | Original PickCube state reference |
| `baseline_state_v1` | 3 | 100 | PickCube, RollBall and PushT state PPO |
| `official_ppo_state_v1` | 11 | 100 | Available official state PPO checkpoints |
| `tabletop_state_v1` | 18 | 100 | Pinned tabletop catalog with state inputs |
| `tabletop_rgb_v1` | 18 | 100 | Pinned tabletop catalog with RGB inputs |
| `upstream_state_v1` | 51 | 100 | All evaluation tasks supporting the declared state profile |
| `bridge_rgb_v1` | 4 | 100 | Native Bridge RGB and segmentation observations |
| `upstream_tasks_v1` | 55 | 100 | Complete evaluation-task inventory at the declared observation mode |
| `transfer_reference_state_v1` | 5 | 100 | Explicit root-observation and model-compatible controller variants |
| `humanoid_control_transfer_state_v1` | 3 | 100 | Explicit root observations for TD-MPC2 actor transfer |
| `panda_visual_rgb_v1` | 19 | 100 | Tabletop and FMB visual/control transfers using ManiSkill-finetuned Octo |
| `g1_visual_rgb_v1` | 2 | 100 | Explicit head-camera and root-observation G1 visual transfers |
| `fetch_mshab_depth_v1` | 2 | 100 | Native Fetch depth/contact inputs for MS-HAB transfer; original assets required |

These are versioned Kit suites. ManiSkill is an environment platform without one unified official
full benchmark. The original profile IDs remain selectable.

## Build and run

```bash
MANISKILL_ASSET_DIR=/path/to/shared/assets \
MANISKILL_ASSET_SUITES=official_ppo_state_v1 \
UV_BIN=/path/to/uv bash benchmarks/maniskill/build-runtime.sh /path/to/runtime

/path/to/runtime/env/bin/python examples/maniskill/download_models.py /path/to/models --all
```

The builder verifies the source and PhysX hashes, installs pinned dependencies and selected assets,
records provenance and runs N1/N2 GPU self-tests. Use an empty install prefix. Select additional
asset suites with `MANISKILL_ASSET_SUITES` and a verified archive cache with
`MANISKILL_ASSET_ARCHIVE_DIR`; `MANISKILL_ASSET_TASKS` selects additional comma-separated task IDs.
The Clutter archive override recovers the exact original 5,000-episode file from a checksum-pinned
author archive. Asset availability and terms are recorded separately from task
validation. Keep weights, assets, caches and raw results outside Git.

Set the Python and repository paths in the [Policy configuration](../../examples/maniskill/roboscope-policy.json),
then follow the [official PPO example](../../examples/maniskill/README.md) for `--self-test`,
real-model `--quick-test`, full evaluation and native GPU-vectorized comparison commands.
The separate [Octo-small example](../../examples/maniskill/OCTO.md) serves `bridge_rgb_v1`.
Additional real-model references document [Panda Octo](../../examples/maniskill/PANDA_OCTO.md),
[Humanoid TD-MPC2](../../examples/maniskill/HUMANOID.md) and
[G1 visual transfer](../../examples/maniskill/G1_VISUAL.md), including their transfer limitations.
The [Fetch depth example](../../examples/maniskill/FETCH_MSHAB.md) supplies real MS-HAB Open
weights and a native depth/contact adapter; original PartNet assets still block cabinet task runs.
The [protocol](PROTOCOL.md) describes per-task coverage, metrics and timing boundaries.
