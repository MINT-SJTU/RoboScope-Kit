# ManiSkill remaining compatibility validation

This phase continues [issue 25](https://github.com/MINT-SJTU/RoboScope-Kit/issues/25) on
2026-11-09. The pinned scope remains the [55 upstream evaluation task IDs](TASK_INVENTORY.md),
with observation/controller profiles counted separately. The preceding
[completion report](COMPLETION_VALIDATION.md) records 42 task IDs with matched real-model
five-tick prefixes. The [review report](REVIEW_VALIDATION.md) retains the separate fifteen-task,
100-episode-per-task, full-horizon comparisons. This phase does not retroactively change either
result or claim a runtime speedup.

**The R6 real-model GPU matrix passed for 51/55 distinct task IDs across 64 selected profiles.**
All 64 SDK/native pairs, 320 actions and 384 observations per path matched exactly. This adds nine
task IDs: three Humanoid control tasks, four Panda manipulation tasks, and two G1 manipulation
tasks. Sixteen of the original eighteen tabletop RGB profiles now have matched real visual-policy
prefixes. Four tasks remain blocked by original assets; cabinet tasks also lack a compatible
implemented Fetch policy. Five-tick integration coverage does not establish full benchmark acceptance.

## New policy and profile scope

| Candidate | Suite / profiles | New task IDs relative to the preceding 42 | Contract and limits |
| --- | --- | --- | --- |
| Official TD-MPC2 Humanoid policies | `humanoid_control_transfer_state_v1`: Stand, Walk and Run named root-state profiles | `MS-HumanoidStand-v1`, `MS-HumanoidWalk-v1`, `MS-HumanoidRun-v1` | Task-specific DM-Control policies; deterministic learned prior, with measured native state and explicit torque-to-PD conversion. Cross-simulator transfer, not ManiSkill training or the full MPC planner. |
| RDT-author-released ManiSkill-fine-tuned Octo | `panda_visual_rgb_v1`: the original 18 tabletop RGB profiles plus `fmb_assembly1_easy_rgb` | `PegInsertionSide-v1`, `PlugCharger-v1`, `FMBAssembly1Easy-v1`, `PickClutterYCB-v1` | Actual RGB and language; source EE7 decoding and declared projection to each frozen native controller. New tasks, PandaStick and dual-arm composition are transfer evaluations. |
| The same visual model with explicit G1 retargeting | `g1_visual_rgb_v1`: apple placement and box transport RGB profiles | `UnitreeG1PlaceAppleInBowl-v1`, `UnitreeG1TransportBox-v1` | Panda-to-G1 visual/controller transfer using actual 25-joint state and the original native articulation. The checkpoint is not G1-trained. |

The visual Policy implements all 19 Panda-family RGB profiles. Two remain asset-blocked:
`assembling_kits_rgb` and `turn_faucet_rgb`. All 17 asset-available RGB profiles passed the R6
real-model SDK/native prefixes, including 16 of the original 18 tabletop RGB profiles and FMB.
Declared capability, CPU inference, real GPU prefix execution and full-horizon task competence
are separate evidence levels.

The new Humanoid profiles are `ms_humanoid_{stand,walk,run}_root_state`. They add measured root
pose and velocities; the original canonical state profiles remain selectable and unchanged.
G1 RGB profiles retain native numerical state and expose actual joint identities in episode
metadata. Existing state-only profiles are not silently credited with these new RGB/root-state
comparisons. No old robot, controller, camera, horizon, reset, success or termination setting is
replaced to fit a model.

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

## Independent CPU and native-FK evidence

| Check | Result and coverage | What it does not establish |
| --- | --- | --- |
| Humanoid source-state reconstruction | 96 actual DM-Control states; independent native joint permutation; maximum source-observation error 4.77e-7 | Cross-simulator trajectory or score equivalence |
| Humanoid learned actors and conversion | All three source encoder/prior actors bitwise equal; source actuator plus passive torque exact; native-action error at most 1.20e-7 against float64 reference | Full stochastic TD-MPC2 planner behavior |
| Humanoid episode isolation | Two interleaved ten-step episodes per task, 60 public-Policy actions; hold/reset/cleanup passed with CUDA hidden | Native GPU rollout acceptance |
| ManiSkill Octo source functions | Actual checkpoint; five actions including the next chunk boundary and image resize bitwise equal to pinned author functions, maximum action error 0 | Real-image task success; the inference probe uses deterministic synthetic RGB |
| Unused Octo proprioception | Altering or omitting the example proprioception field gives bitwise-identical actions; no state tokenizer configured | A fabricated state input is never used to claim embodiment compatibility |
| G1 independent CPU kinematics | 64 random interior states; independent SciPy recursive FK and finite differences; maximum all-link FK error 8.88e-16, Jacobian error 2.38e-10; unrelated joints unchanged and no limit violations | Dynamics or arbitrary-goal reachability |
| G1 reachable small-goal DLS | 64 single-hand and 64 bimanual goals; worst final/initial weighted residual 0.01441 and 0.008694 | G1-expert policy competence |
| G1 actual native GPU FK | Both tasks at N1 and N2, six snapshots including five real action steps; maximum world-link matrix error 9.44e-7 after composing the measured root pose | Learned-model execution; this probe uses random bounded joint actions |
| Visual controller conversion probes | All 19 Panda-family action dimensions agree; 100 G1 Cartesian commands, six hand endpoints and three DLS cases passed, including permuted state/controller identities | Model inference or SDK/native rollout pairing |

The R6 simulator was built from an empty prefix using the committed builder, with Kit 0.8.0b46
and the existing pinned simulator/physics dependencies. Its GPU self-test passed. These build and
kinematics checks are independent of the pending learned-policy matrix. Humanoid inference uses
the existing Torch/NumPy environment. ManiSkill Octo uses the pinned Octo inference environment
with the eight versioned imports in `panda-octo-requirements.txt` installed through an isolated
checkpoint import overlay; the Bridge Octo source/environment is not overwritten.

## New real-model GPU matrix: pending

No new GPU Policy pass count, success result or throughput value is asserted here until the
batch records and strict SDK/native comparison have been collected. The planned scope is three
Humanoid root-state profiles, 17 asset-available Panda-family RGB profiles and two G1 RGB profiles.
Task IDs already covered by the previous 42 are counted once.

Use the existing five-tick diagnostic method: preserve the full native environment horizon and
set only `diagnostic_step_limit=5`. Each prefix records seed, profile, checkpoint/manifest hashes,
actual steps, native masks, success fields where available, return and full observation/action
traces. It is `integration_smoke`, not a completed episode or shortened benchmark protocol.
The native Gym path constructs observations and routes actions independently of SDK Worker/RPC;
the independent source-model checks above cover model math separately. Missing/duplicate records,
first divergence and numerical comparison tolerances must remain visible.

| Required result | Current new-phase status |
| --- | --- |
| Expected / actual SDK/native records and task-ID/profile coverage | Pending |
| Initial observations, five actions and terminal observation pairing | Pending |
| Native horizon, executed ticks, terminal/truncated masks, missing/duplicate records | Pending |
| Total command wall and evaluation wall; diagnostic episodes/h and control steps/s | Pending |
| Model load, create, reset, step+observation, input staging, RPC round trip, close/save clocks | Pending where collected |
| Device-wide GPU busy, average/peak VRAM, CPU and RSS | Pending sampler output |
| Independent GPU inference, isolated rendering and queue/wait duration | N/A unless independently instrumented |
| SM, Tensor and RT hardware throughput | N/A; NVML busy is not any of these counters |
| Native success counts/rates, return-only tasks, execution failures, OOM and actual timeouts | Pending; diagnostic cutoff is not a timeout |

Do not sum overlapping worker/server clocks as total wall time or compare five-tick prefix
throughput with full-horizon evaluation as a speedup. The phase introduces model/profile/asset
compatibility, not a runtime optimization. Final exact-head pytest, Ruff, shell syntax/ShellCheck,
build provenance, real quick test and durable evidence links belong with the completed matrix;
prior-phase checks do not establish the status of the new numerical source.

## Reproduction and unresolved scope

Build selected available assets without requesting the blocked full RGB suite:

```bash
MANISKILL_ASSET_DIR=/autodl-fs/roboscope/assets \
MANISKILL_ASSET_SUITES=official_ppo_state_v1,humanoid_control_transfer_state_v1,g1_visual_rgb_v1 \
MANISKILL_ASSET_TASKS=PickClutterYCB-v1,PickSingleYCB-v1 \
MANISKILL_ASSET_ARCHIVE_DIR=/autodl-fs/roboscope/asset-archives \
UV_BIN=/path/to/uv bash benchmarks/maniskill/build-runtime.sh /path/to/new-runtime
```

Follow [HUMANOID.md](../../examples/maniskill/HUMANOID.md),
[PANDA_OCTO.md](../../examples/maniskill/PANDA_OCTO.md) and
[G1_VISUAL.md](../../examples/maniskill/G1_VISUAL.md) for pinned downloads, model preparation and
Policy commands. The existing [SDK/native instructions](../../examples/maniskill/README.md#additional-real-policy-references)
describe the generic reference and strict comparator. Diagnostic manifests must retain each
profile's frozen values and be distinguished from the ordinary full-horizon manifest. Exact new
GPU batch commands, records and metric tables remain pending with the matrix above.

| Remaining task | Concrete unresolved requirement |
| --- | --- |
| `AssemblingKits-v1` | Original kit asset archive remains unavailable from the probed endpoint; the declared visual Policy cannot be executed without it. |
| `TurnFaucet-v1` | Required original processed PartNet assets remain blocked by the probed authenticated download; the declared visual Policy is not validated. |
| `OpenCabinetDoor-v1` | Original processed PartNet assets and a compatible real Fetch model/action path are still missing. |
| `OpenCabinetDrawer-v1` | Original processed PartNet assets and a compatible real Fetch model/action path are still missing. |

Public raw PartNet mirrors do not supply the frozen loader's required `mobility_cvx.urdf` or
`mobility_fixed.urdf`. Other inspected archives split/fix object joints or provide point clouds
instead of the original processed assets. Renaming raw URDFs, generating new collision geometry,
accepting download gates or substituting different scenes has not been used to close these gaps.
Asset failures remain recorded failures, rather than skipped passes.

Even a successful 51/55 five-tick matrix leaves these four task IDs, two original tabletop RGB
profiles, canonical/sibling-profile distinctions and the repository's full-horizon acceptance
requirements unresolved. The historical fifteen full-horizon task comparisons remain fifteen.
Weights, downloads, raw probes and logs stay outside Git under `/autodl-fs`; new evidence has not
yet been attributed to the prior phase's published evidence archive.
