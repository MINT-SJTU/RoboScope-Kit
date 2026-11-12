# ManiSkill model compatibility notes

These notes describe selected model input/controller contracts and unresolved original-profile
requirements. The [validation status](VALIDATION.md) is the single current-state record. A compatible
shape or a historical SDK/configured-reference match does not establish an intended-task trained
checkpoint, nonzero task competence or acceptance under the latest maintainer decision.

## Original-profile gaps

The following six available original state profiles have no selected compatible model:

| Profile | Remaining input/model requirement |
|---|---|
| `pick_clutter_ycb_state` | The selected Octo model requires RGB. Its state sibling lacks the object information required by inspected alternatives. |
| `fmb_assembly1_easy_state` | The selected Octo model requires RGB. At the pin, upstream constructs object/TCP extras but returns an empty dictionary; changing this contract needs an explicit source patch and regenerated comparisons. |
| `ms_humanoid_stand_state` | Its egocentric state omits world-frame information consumed by the selected DM-Control Stand conversion. |
| `unitree_h1_stand_state` | The selected locomotion actor needs root orientation/angular velocity absent from the canonical numeric state. |
| `unitree_g1_stand_state` | The selected locomotion actor likewise needs root orientation/angular velocity; joint-name metadata cannot supply them. |
| `anymal_c_spin_state` | The selected Reach-to-Spin conversion needs root pose/orientation absent from this state. |

These are unresolved selected input/model contracts, not six demonstrated download failures or a
proof that no public compatible model exists. An alternative RGB/root/controller profile is a
different protocol and does not validate its original sibling. Root/IMU constants, fabricated object
or image fields, truncated actors and untrained output padding are not used to fill missing inputs.
The earlier inspected G1 encoder-only lead did not provide a complete compatible actor/config;
new model availability requires a new source-backed search rather than repeating that conclusion.

## Original-asset gaps

AssemblingKits, TurnFaucet, OpenCabinetDoor and OpenCabinetDrawer still require their original assets.
Their eight unexecuted profiles are `assembling_kits_{state,rgb}`, `turn_faucet_{state,rgb}`,
`open_cabinet_{door,drawer}_state` and `open_cabinet_{door,drawer}_mshab_depth`. Model and asset
blockers can overlap. Selected Panda RGB/Fetch depth references do not validate an asset-blocked
scene or supply a compatible policy for its canonical state sibling. Unknown outcomes stay unknown.

## Selected technical contracts

The following source and frame-conversion notes are retained for reproducibility. Borrowed
checkpoints remain diagnostic references, not accepted intended-task models. Their chronological
rollout/build tables have moved out of the current documentation; original reports and raw evidence
remain available at the [immutable source snapshot](https://github.com/QLL-CCDD/RoboScope-Kit/blob/634d867811401f4812b7580bce0e782af3cdcd19/benchmarks/maniskill/REMAINING_COMPATIBILITY.md)
and the [linked evidence releases](VALIDATION.md#recorded-evidence).

### Humanoid policy provenance

[HUMANOID.md](../../examples/maniskill/HUMANOID.md) and
[humanoid-checkpoints.json](../../examples/maniskill/humanoid-checkpoints.json) pin three original
seed-1 checkpoints from
[`nicklashansen/tdmpc2`](https://huggingface.co/nicklashansen/tdmpc2/tree/8fb2a82efb3bae96941da440128fe1332e4394fd),
revision `8fb2a82efb3bae96941da440128fe1332e4394fd`, totaling 95,944,422 bytes. Inference follows
source `332f19e4aba4bffe2dc736f45ea7f7bdc585903a`, `mpc=False`, `eval_mode=True`. All original
checkpoint tensors load strictly; training-only heads are not fabricated or silently discarded.

The Policy reconstructs the 67D source input from real native state/root kinematics and maps joint
and motor identities by name. Source motor gear, spring and damping values produce a requested
torque, converted into the original native current-relative PD action. Instantaneous unsaturated
drive-torque agreement does not establish trajectory agreement: PhysX holds a PD target, while
the source holds torque across different physics/contact solvers. Episode-local two-tick motor
history preserves source action repeat while native observations and checks still run each tick.

### Visual policy provenance

[PANDA_OCTO.md](../../examples/maniskill/PANDA_OCTO.md) and
[panda-octo-checkpoint.json](../../examples/maniskill/panda-octo-checkpoint.json) pin the released
ManiSkill-fine-tuned Octo checkpoint at revision `9622afab2b7ce2312a6cf1febc526928589b77eb`, step
`1000000`, from `robotics-diffusion-transformer/maniskill-model`. Inference checkpoint files total
549,407,955 bytes. The author's evaluator is pinned at
[`98e6d7b725797b9640d219a8fd385009e13fa824`](https://github.com/thu-ml/RoboticsDiffusionTransformer/blob/98e6d7b725797b9640d219a8fd385009e13fa824/eval_sim/eval_octo.py);
Octo v1.5 and dlimp sources, offline tokenizer and individual checkpoint files are also hashed.

The model uses actual primary RGB, the author's Lanczos3 resizing/truncating uint8 conversion,
two-frame history and four-action predictions. Its configuration has no proprioception tokenizer.
The author's absent wrist view remains absent. PandaStick/G1 observations are not padded with
invented Panda joints. The author sends normalized EE7 directly to the native controller; Bridge
statistics and action denormalization are not applied. Per-episode PRNG/history/queues isolate
concurrent episodes; this intentionally differs from the author's one global episode generator.

The author's published five-task comparison reports zero success for this Octo baseline. Its real
weights are suitable for exercising image inference and native control, but successful loading or
five-tick execution does not imply useful success on the requested tasks.

### G1 physical conversion

[G1_VISUAL.md](../../examples/maniskill/G1_VISUAL.md) documents the transfer and its limits. FK and
DLS use the unchanged native upper-body URDF, SHA-256
`3ab8835fb251c8313974c13cb9e943e09fbeed9575ec86c36002ee6436868344`, at the frozen ManiSkill
revision. Apple placement commands the right TCP. Box transport uses two separately reset visual
streams with left/right role prefixes and one stacked IK problem sharing the torso.

The pinned Panda GPU controller norm-clips raw rotation and multiplies it by **-0.1**, then uses
the resulting root-frame angular displacement directly in differential geometric-Jacobian IK.
The transfer integrates this intent into a finite G1 goal; it does not claim exact Panda joint or
trajectory parity. Native CPU control instead solves the separately constructed Euler-XYZ target.
G1's 12 bimanual pose components share 11 torso/arm joints, so bounded DLS is approximate.

Hand opening is native zero position for the six flexion axes. Closing interpolates toward
mirrored URDF limits; thumb abduction holds its measured value. This full-limit convention is a
model-side transfer choice. Joint targets become the original 0.2/0.5 rad normalized native PD
deltas in actual controller order. Clipped-joint counts and final commanded-target Cartesian
residuals are diagnostics, not task scores or measured physical tracking errors.

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

The clean `runtime-r6` builder installed clutter through this committed path. The asset was not
inserted by a one-off installed-environment edit. The new comma-separated `MANISKILL_ASSET_TASKS`
selector resolves original task IDs to declared profiles/assets, allowing available tasks to be
prepared without selecting a full suite containing unavailable assets. It supplements the existing
`MANISKILL_ASSET_SUITES` selector.

## Fetch model implementation and native API probe

Kit **0.8.0b47** adds two explicit `fetch_mshab_depth_v1` profiles and a
[pinned MS-HAB model example](../../examples/maniskill/FETCH_MSHAB.md). The real fridge and
kitchen-counter PPO Open weights total 136,311,860 bytes. Both strictly load all 27 released
tensors and reproduce the unmodified author network bitwise on matching CPU batches. Independent
CPU state/contact fixtures reproduce the source 42-vector exactly. Public interleaved Policy calls
match within 1.43e-6, with exact frame history and reset isolation. These are transfer models trained
on ReplicaCAD articulations, not trained PartNet-cabinet experts.

The optional SDK observations contain native head/hand depth, measured base/TCP/finger poses,
the actual target handle position and raw pairwise contact forces. The Policy owns all coordinate
transforms, the source's absent-object sentinel, grasp threshold, depth normalization and three-frame
history. Original cabinet state profiles and native control/reset/horizon/scoring remain unchanged.

The initial historical GPU probe exposed an over-restrictive depth positivity check. The pinned renderer
uses signed int16 millimeter data, including negative encoded values. The final implementation
retains those raw values, and the fresh **R8** simulator/Octo builds install the corrected b47 source.
historical's failed probe is retained separately in the external evidence.

An explicitly non-cabinet **PickCube/Fetch API fixture** exercised both real models at N1 and N2
through the public SDK socket using MessagePack, comparing independently extracted native features
and direct model calls. All **30 actions and 36 observations** matched; independently reconstructed
source state and native grasp flags also matched exactly. ID reuse reproduced the first action.
Observed raw depth includes -32768 mm, which survived SDK transport unchanged. The complete probe
took 9.547 s. `fetch-native-gpu-proof.json` declares `counts_toward_cabinet_acceptance=false`.
This fixture validates the physical input/control interfaces; it does not stand in for the missing
PartNet cabinet scenes or add two task IDs to the accepted matrix.

Every evaluation task now has a selected real-policy reference or explicit transfer candidate.
That model inventory is separate from **51/55 canonical task IDs with real-policy/native prefixes**:
the four original-asset blockers and the two unvalidated tabletop RGB profiles remain open.

## Evidence limits

The configured-policy references reuse manifest configuration and Policy implementations. They
check transport/interface behavior, not independent upstream task configuration or published-model
performance. Existing float32 HORA contact variation remains a [strict repeatability failure](HORA_NATIVE_REPEATABILITY.md),
without a tolerance/quantization waiver. Full current-source prescribed-count evaluation remains
pending; historical complete-horizon and prefix cohorts are not combined.
