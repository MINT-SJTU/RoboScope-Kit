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
model, asset and profile gaps. Forty evaluation tasks still lack a selected compatible policy.
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
`MANISKILL_ASSET_ARCHIVE_DIR`. Asset availability and terms are recorded separately from task
validation. Keep weights, assets, caches and raw results outside Git.

Set the Python and repository paths in the [Policy configuration](../../examples/maniskill/roboscope-policy.json),
then follow the [official PPO example](../../examples/maniskill/README.md) for `--self-test`,
real-model `--quick-test`, full evaluation and native GPU-vectorized comparison commands.
The separate [Octo-small example](../../examples/maniskill/OCTO.md) serves `bridge_rgb_v1`.
The [protocol](PROTOCOL.md) describes per-task coverage, metrics and timing boundaries.
