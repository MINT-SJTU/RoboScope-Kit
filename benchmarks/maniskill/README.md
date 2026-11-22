# ManiSkill evaluation

The `maniskill` adapter runs manifest-selected task/robot/observation/controller profiles through
GPU vector environments and the existing Policy lifecycle. The pinned [inventory](TASK_INVENTORY.md)
declares 55 evaluation TaskIDs across tabletop, dexterity, control, locomotion, mobile manipulation
and Bridge families. The [evaluation protocol](../../packages/roboscope-kit/MANISKILL_EVALUATION_PROTOCOL.md)
defines their frozen grouping, input/action mapping, scores and shard handling.

**Full integration acceptance is incomplete.** [Validation status](VALIDATION.md) separates current
source checks, retained model candidates and missing acceptance evidence/assets.
Identical SDK/configured-reference records alone do not establish task competence or acceptance.
Weights, simulator assets, caches and raw records remain outside Git.

| Suite | Profiles | Episodes per profile | Configuration |
| --- | ---: | ---: | --- |
| `pickcube_state_v1` | 1 | 100 | Original PickCube state reference |
| `baseline_state_v1` | 3 | 100 | PickCube, RollBall and PushT state PPO |
| `tabletop_state_v1` | 18 | 100 | Original tabletop state profiles |
| `tabletop_rgb_v1` | 18 | 100 | Original tabletop RGB profiles; visual acceptance pending |
| `official_ppo_state_v1` | 11 | 100 | Official task-trained state PPO checkpoints |
| `upstream_tasks_v1` | 55 | 100 | All original evaluation TaskIDs at declared observation modes |
| `upstream_state_v1` | 51 | 100 | Original evaluation-task state profiles |
| `bridge_rgb_v1` | 4 | 100 | Original Bridge RGB/segmentation; provisional Octo baseline |
| `panda_visual_rgb_v1` | 2 | 100 | Peg/Plug RGB with author-compatible 7D pose actions; unvalidated |
| `reference_ppo_state_v1` | 2 | 100 | Two author G1 task-trained PPO candidates |
| `newt_state_v1` | 8 | 100 | Eight NEWT task/robot candidates; actor prior, not original MPC |

These are versioned Kit suites. ManiSkill is an environment platform without one unified official
full benchmark. The current manifest has 77 profiles across eleven suites; all
55 original TaskIDs and the original tabletop state/RGB definitions remain selectable.

## Build and run

```bash
MANISKILL_ASSET_DIR=/path/to/shared/assets \
MANISKILL_ASSET_SUITES=official_ppo_state_v1 \
UV_BIN=/path/to/uv bash benchmarks/maniskill/build-runtime.sh /path/to/runtime

/path/to/runtime/env/bin/python examples/maniskill/download_models.py /path/to/models --all
```

The builder verifies the source and PhysX hashes, installs pinned dependencies and selected assets,
records provenance and runs one-/two-lane GPU diagnostic self-tests. Use an empty install prefix.
Select additional
asset suites with `MANISKILL_ASSET_SUITES` and a verified archive cache with
`MANISKILL_ASSET_ARCHIVE_DIR`; `MANISKILL_ASSET_TASKS` selects additional comma-separated task IDs.
The Clutter archive override recovers the exact original 5,000-episode file from a checksum-pinned
author archive. Asset availability and terms are recorded separately from task
validation. Keep weights, assets, caches and raw results outside Git.

Set the Python and repository paths in the [Policy configuration](../../examples/maniskill/roboscope-policy.json),
then follow the [official PPO example](../../examples/maniskill/README.md) for `--self-test`,
real-model `--quick-test`, full evaluation and native GPU-vectorized comparison commands.
The separate [Octo-small example](../../examples/maniskill/OCTO.md) serves `bridge_rgb_v1`.
Additional examples document [eight task-trained NEWT candidates](../../examples/maniskill/NEWT.md)
and the [author Panda Octo candidate](../../examples/maniskill/PANDA_OCTO.md), restricted to
Peg/Plug RGB. The PPO example also describes two author G1 checkpoints. None of these model
catalogs establishes final-source prescribed-count task acceptance. The
[protocol](../../packages/roboscope-kit/MANISKILL_EVALUATION_PROTOCOL.md) describes observations,
scores, grouping and timing boundaries.
