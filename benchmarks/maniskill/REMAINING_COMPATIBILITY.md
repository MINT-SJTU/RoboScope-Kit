# ManiSkill model compatibility notes

The [validation status](VALIDATION.md) separates retained model candidates from
acceptance. A matching tensor shape, an SDK/configured-reference match or a
successful prefix cannot establish intended-task training or competence.

## Selected candidates and profile boundaries

The selected task-trained catalog has eleven official PPO TaskIDs, two author G1
PPO TaskIDs and eight NEWT TaskIDs. Thirty-four original TaskIDs have no selected
intended task/robot checkpoint in this inventory; no claim is made that public
models do not exist elsewhere. The original 55 TaskIDs remain selectable.

| Candidate | Compatible declared scope | Remaining boundary |
|---|---|---|
| Official PPO | Eleven profiles in `official_ppo_state_v1` | LiftPeg uses the separate pose profile; the two genuine dual-Panda actors use separate 16D joint-action profiles. They do not validate the original controller siblings or RGB inputs. |
| Author G1 PPO | Two profiles in `reference_ppo_state_v1` | Retain the Apple's source action clipping. TransportBox's source simulator/default horizon differ, and independent author validation remains open. |
| NEWT | Eight profiles in `newt_state_v1` | Six control tasks, PlaceSphere and the separate PullCubeTool pose profile match the released task/robot. Deterministic actor-prior inference is not original MPC evaluation; source version, CPU/GPU backend and Hopper 500/600-tick differences remain explicit. |
| Bridge Octo-small | Four original Bridge RGB profiles | Provisional baseline; per-task training match and competence are not established. |
| Author ManiSkill Octo | Existing PegInsertionSide/PlugCharger RGB profiles | Only full 7D `pd_ee_delta_pose` actions are supported. Published Octo results are zero; acceptance remains open. |

The author's other named Octo tasks use different controllers in the original Kit
RGB profiles. Those definitions stay unchanged and are not validated by this
candidate. The original 18 tabletop RGB configurations need compatible visual
policies and their own native comparisons. Similarly, a candidate for a separate
controller profile does not validate the original state/controller sibling.

At the pinned FMB source, the state path constructs object/TCP extras but returns
an empty dictionary. A future candidate must consume the actual original input
contract or introduce a separately reviewed source patch with new comparisons.
Inputs are not filled with fabricated root, object or image fields to fit a model.

## Original-asset gaps

AssemblingKits, TurnFaucet, OpenCabinetDoor and OpenCabinetDrawer still require
their original processed assets. The six retained blocked profiles are
`assembling_kits_{state,rgb}`, `turn_faucet_{state,rgb}`,
`open_cabinet_door_state` and `open_cabinet_drawer_state`. Model and asset blockers
can overlap. Unknown outcomes stay unknown, and a different scene cannot replace
the original asset distribution for task acceptance.

The pinned builder and asset helper are the reproducible installation path.
An archive override is acceptable only when its original content is verified;
an arbitrary replacement mesh or regenerated scene is not an asset repair.
Successful clutter recovery below does not imply recovery of the four other
task assets. The current candidate does not claim those downloads now succeed.

## Exact original clutter-asset recovery


The original UCSD clutter endpoint was unavailable. The
[CenterGrasp author's installation instructions](https://github.com/robot-learning-freiburg/CenterGrasp/blob/72bc8f1121937fb5b50164d4f5616b3074309281/README.md)
provide a public mirror bundle containing the byte-identical original scene corpus:

| Pin | Value |
| --- | --- |
| Bundle | `https://centergrasp.cs.uni-freiburg.de/download/maniskill_ycb.tar.gz` |
| Bundle bytes | 43,528,012 |
| Bundle SHA-256 | `fdddd43f17c4e51d3e9458d1229e56c8b47aa7be339c66bdcf36118c4f038bfa` |
| Member | `maniskill_ycb/pick_clutter/ycb_train_5k.json.gz` |
| Original member bytes | 16,616,172 |
| Original member SHA-256 | `70ec176c7036f326ea7813b77f8c03bea9db5960198498957a49b2895a9ec338` |
| Parsed source episodes | 5,000 |

The member digest equals the frozen upstream registry checksum. No scene seeds, objects,
collision meshes or episode identities are regenerated. Original asset terms still apply.

The manifest pins the bundle, author revision and `archive_member` metadata.
`prepare-assets.py` verifies the complete bundle digest, extracts only the named regular file,
checks its size, checks its digest against both the member pin and the original upstream checksum,
then passes that original archive to the upstream installer. Provenance records both digest
levels. A verified cached bundle can be reused through `MANISKILL_ASSET_ARCHIVE_DIR`.

The builder installs this corpus through the committed asset-preparation path.
`MANISKILL_ASSET_TASKS` selects comma-separated original task IDs without requiring
a complete suite containing unavailable assets. It supplements `MANISKILL_ASSET_SUITES`.
