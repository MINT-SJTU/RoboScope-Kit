# ManiSkill remaining compatibility validation

This phase continues [issue 25](https://github.com/MINT-SJTU/RoboScope-Kit/issues/25) on
2026-11-09. The pinned scope remains the [55 upstream evaluation task IDs](TASK_INVENTORY.md),
with observation/controller profiles counted separately. The preceding
[completion report](COMPLETION_VALIDATION.md) records 42 task IDs with matched real-model
five-tick prefixes. The [review report](REVIEW_VALIDATION.md) retains the separate fifteen-task,
100-episode-per-task, full-horizon comparisons. This phase does not retroactively change either
result or claim a runtime speedup.

The [historical R8 results](#historical-b47-validation-r8) use Kit 0.8.0b47 and predate the b48
terminal-mask recording/comparison fix. Their measurements retain that original source identity;
they do not validate the updated head. The R6 section retains the preceding phase's evidence.
The [native-reference review](NATIVE_REVIEW_VALIDATION.md) records the mask fixes, retained R9
full-horizon failures and pending final R10 validation separately.

**The R6 real-model GPU matrix passed for 51/55 distinct task IDs across 64 selected profiles.**
All 64 SDK/native pairs, 320 actions and 384 observations per path matched exactly. This adds nine
task IDs: three Humanoid control tasks, four Panda manipulation tasks, and two G1 manipulation
tasks. **16/18 original tabletop RGB profiles** now have matched real visual-policy prefixes.
Four tasks remain blocked by original assets. A subsequent Fetch implementation and native API
probe are recorded below, without adding the blocked cabinet tasks to the pass count.
Five-tick integration coverage does not establish full benchmark acceptance.

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
kinematics checks are independent of the learned-policy matrix below. Humanoid inference uses
the existing Torch/NumPy environment. ManiSkill Octo uses the pinned Octo inference environment
with the eight versioned imports in `panda-octo-requirements.txt` installed through an isolated
checkpoint import overlay; the Bridge Octo source/environment is not overwritten.

## R6 real-model GPU results

The final matrix uses the fresh `runtime-r6` and `octo-policy-runtime-r6` builds, Kit **0.8.0b46**,
Python 3.11.15, ManiSkill 3.0.0b15 / SAPIEN 3.0.0b1, Torch 2.6.0 / CUDA 12.4 and NumPy 1.26.4 on
the adaptation host's NVIDIA GeForce RTX 4090. All SDK runs used K1/G1/N1 with MPS disabled.
The separate policy process retains its pinned JAX/Flax/TensorFlow inference environment.

`collect_remaining.py` read the completed twelve-family batch and passed with zero integrity
errors. All 36 SDK/native/comparison commands exited successfully. Expected/actual coverage is
64/64 profiles, 51/51 selected task IDs, 320/320 actions and 384/384 observations per path. Both
records and complete observation/action traces match at zero tolerance. There are no missing or
duplicate records, worker failures, OOM events, true terminal masks or actual timeouts. All 52
success-scored prefixes have zero success; the other twelve retain null success fields and returns.
This is a five-tick compatibility result, not a claim that the policies solve all tasks.

The sequential family-command resource aggregate is below. GPU/VRAM means are sample-weighted
across commands; CPU mean is accounted child CPU seconds divided by summed command wall. Neither
is a full-horizon steady-state estimate. Family measurements and clocks follow.

| Path | GPU busy mean % | VRAM mean / peak GiB | CPU accounted mean / sampled peak cores |
| --- | --- | --- | --- |
| SDK | 6.158 | 3.228 / 8.042 | 1.102 / 14.376 |
| NATIVE | 8.085 | 3.612 / 8.042 | 1.251 / 19.702 |

Generated 2026-11-09T08:45:10.516821+00:00; final label `remaining-all-r6`.

These are traced five-native-tick integration prefixes at K1/G1/N1, one seed per profile.
The original native horizon is retained and both terminal masks remain false. These results
are non-formal and do not add full-horizon accepted tasks. Each family is a separate command.
Family command and evaluation sums are sequential command sums, not one persistent-suite wall time.
SDK/native boundaries and their transports differ; no runtime speedup is inferred.

Collector integrity checks: **PASS** (0 errors).

Final coverage is 64 SDK and 64 native records,
covering 64 profiles and 51/55 unique task IDs.
Task IDs intentionally repeat across distinct observation/controller profiles.
There are 320 control steps on each path. Native success is available for
52 prefixes (0 success-once, 0 success at prefix end); 12 are return-only.
Recorded worker failures: 0; OOM events: 0; true terminated/truncated masks: 0 / 0.

### SDK clocks and diagnostic throughput

| Family | Records | Steps | Command s | Evaluation s | Episodes/h command / eval | Steps/s command / eval |
| --- | --- | --- | --- | --- | --- | --- |
| newt | 9 | 45 | 33.072 | 17.619 | 979.7 / 1,838.9 | 1.361 / 2.554 |
| hora | 4 | 20 | 16.671 | 12.398 | 863.8 / 1,161.5 | 1.200 / 1.613 |
| trifinger | 5 | 25 | 16.689 | 11.897 | 1,078.6 / 1,513.0 | 1.498 / 2.101 |
| robel | 5 | 25 | 13.492 | 10.387 | 1,334.1 / 1,732.9 | 1.853 / 2.407 |
| unitree | 2 | 10 | 16.035 | 10.945 | 449.0 / 657.8 | 0.624 / 0.914 |
| go2 | 1 | 5 | 10.589 | 5.745 | 340.0 / 626.6 | 0.472 / 0.870 |
| spin | 1 | 5 | 11.652 | 6.113 | 309.0 / 588.9 | 0.429 / 0.818 |
| ppo | 11 | 55 | 28.971 | 22.212 | 1,366.9 / 1,782.8 | 1.898 / 2.476 |
| octo | 4 | 20 | 69.075 | 47.926 | 208.5 / 300.5 | 0.290 / 0.417 |
| humanoid | 3 | 15 | 14.700 | 8.944 | 734.7 / 1,207.5 | 1.020 / 1.677 |
| panda | 17 | 85 | 66.380 | 53.337 | 922.0 / 1,147.4 | 1.281 / 1.594 |
| g1 | 2 | 10 | 40.334 | 25.105 | 178.5 / 286.8 | 0.248 / 0.398 |
| Sequential sum | 64 | 320 | 337.658 | 232.629 | 682.3 / 990.4 | 0.948 / 1.376 |

### Native clocks and diagnostic throughput

| Family | Records | Steps | Command s | Evaluation s | Episodes/h command / eval | Steps/s command / eval |
| --- | --- | --- | --- | --- | --- | --- |
| newt | 9 | 45 | 21.374 | 14.575 | 1,515.9 / 2,223.0 | 2.105 / 3.087 |
| hora | 4 | 20 | 14.955 | 10.357 | 962.9 / 1,390.4 | 1.337 / 1.931 |
| trifinger | 5 | 25 | 13.587 | 9.545 | 1,324.8 / 1,885.8 | 1.840 / 2.619 |
| robel | 5 | 25 | 12.495 | 9.378 | 1,440.6 / 1,919.3 | 2.001 / 2.666 |
| unitree | 2 | 10 | 12.562 | 7.896 | 573.2 / 911.9 | 0.796 / 1.267 |
| go2 | 1 | 5 | 8.812 | 4.331 | 408.6 / 831.2 | 0.567 / 1.154 |
| spin | 1 | 5 | 8.600 | 4.118 | 418.6 / 874.2 | 0.581 / 1.214 |
| ppo | 11 | 55 | 23.693 | 19.039 | 1,671.4 / 2,079.9 | 2.321 / 2.889 |
| octo | 4 | 20 | 59.117 | 42.416 | 243.6 / 339.5 | 0.338 / 0.472 |
| humanoid | 3 | 15 | 11.194 | 6.164 | 964.8 / 1,752.0 | 1.340 / 2.433 |
| panda | 17 | 85 | 60.764 | 47.624 | 1,007.2 / 1,285.1 | 1.399 / 1.785 |
| g1 | 2 | 10 | 36.107 | 21.604 | 199.4 / 333.3 | 0.277 / 0.463 |
| Sequential sum | 64 | 320 | 283.259 | 197.048 | 813.4 / 1,169.3 | 1.130 / 1.624 |

Native evaluation is the reference's `formal_wall_s` field; the field name does not make a
prefix formal. Native command clocks include interpreter imports, model process startup and
lossless pipe transport. SDK evaluation excludes separately timed Policy loading/preparation.

### Prefix outcomes

| Family | Success-scored | Return-only | Success once / end | Success rates once / end | Unsuccessful scored prefixes | Worker failures | OOM | Terminated / truncated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| newt | 3 | 6 | 0 / 0 | 0.000000 / 0.000000 | 3 | 0 | 0 | 0 / 0 |
| hora | 4 | 0 | 0 / 0 | 0.000000 / 0.000000 | 4 | 0 | 0 | 0 / 0 |
| trifinger | 5 | 0 | 0 / 0 | 0.000000 / 0.000000 | 5 | 0 | 0 | 0 / 0 |
| robel | 5 | 0 | 0 / 0 | 0.000000 / 0.000000 | 5 | 0 | 0 | 0 / 0 |
| unitree | 0 | 2 | 0 / 0 | N/A / N/A | 0 | 0 | 0 | 0 / 0 |
| go2 | 1 | 0 | 0 / 0 | 0.000000 / 0.000000 | 1 | 0 | 0 | 0 / 0 |
| spin | 0 | 1 | 0 / 0 | N/A / N/A | 0 | 0 | 0 | 0 / 0 |
| ppo | 11 | 0 | 0 / 0 | 0.000000 / 0.000000 | 11 | 0 | 0 | 0 / 0 |
| octo | 4 | 0 | 0 / 0 | 0.000000 / 0.000000 | 4 | 0 | 0 | 0 / 0 |
| humanoid | 0 | 3 | 0 / 0 | N/A / N/A | 0 | 0 | 0 | 0 / 0 |
| panda | 17 | 0 | 0 / 0 | 0.000000 / 0.000000 | 17 | 0 | 0 | 0 / 0 |
| g1 | 2 | 0 | 0 / 0 | 0.000000 / 0.000000 | 2 | 0 | 0 | 0 / 0 |

Success rates are fractions over success-scored prefixes only. An unsuccessful five-tick prefix
is not a full-horizon task failure. Return-only tasks have null success fields, and the
diagnostic cutoff is not an actual timeout.

### SDK resources

| Family | GPU busy command / eval % | VRAM mean / peak GiB | CPU accounted mean / sampled peak cores | RSS mean / peak GiB |
| --- | --- | --- | --- | --- |
| newt | 2.792 / 5.116 | 2.550 / 5.376 | 0.686 / 1.071 | 2.029 / 4.667 |
| hora | 13.303 / 17.447 | 4.086 / 8.042 | 0.880 / 1.069 | 2.823 / 5.536 |
| trifinger | 2.682 / 3.319 | 2.284 / 4.758 | 0.877 / 1.105 | 2.034 / 3.934 |
| robel | 4.815 / 4.780 | 2.172 / 4.363 | 0.901 / 1.104 | 1.612 / 3.381 |
| unitree | 0.317 / 0.476 | 2.302 / 5.372 | 0.724 / 1.073 | 2.012 / 5.035 |
| go2 | 0.634 / 0.333 | 1.513 / 3.810 | 0.779 / 1.042 | 1.202 / 3.391 |
| spin | 2.087 / 1.125 | 1.467 / 3.810 | 0.765 / 1.073 | 1.166 / 3.356 |
| ppo | 5.239 / 5.791 | 3.047 / 5.238 | 0.802 / 1.093 | 2.630 / 4.617 |
| octo | 16.777 / 24.118 | 3.553 / 6.575 | 1.167 / 14.376 | 3.991 / 7.264 |
| humanoid | 0.069 / 0.118 | 2.083 / 4.935 | 0.766 / 1.072 | 1.686 / 4.114 |
| panda | 3.738 / 4.120 | 4.327 / 6.573 | 1.474 / 13.594 | 5.357 / 7.722 |
| g1 | 1.538 / 2.227 | 3.705 / 6.808 | 1.644 / 12.194 | 4.405 / 8.032 |

### Native resources

| Family | GPU busy command / eval % | VRAM mean / peak GiB | CPU accounted mean / sampled peak cores | RSS mean / peak GiB |
| --- | --- | --- | --- | --- |
| newt | 3.345 / N/A | 3.339 / 5.376 | 0.914 / 1.033 | 2.639 / 4.597 |
| hora | 15.627 / N/A | 4.416 / 8.042 | 0.937 / 1.060 | 2.917 / 5.512 |
| trifinger | 10.774 / N/A | 2.603 / 4.758 | 0.906 / 1.035 | 2.147 / 3.892 |
| robel | 3.000 / N/A | 2.573 / 4.363 | 0.876 / 1.061 | 1.816 / 3.332 |
| unitree | 1.600 / N/A | 2.409 / 5.372 | 0.911 / 1.043 | 2.089 / 5.039 |
| go2 | 0.571 / N/A | 1.479 / 3.810 | 0.853 / 1.035 | 1.257 / 3.278 |
| spin | 1.176 / N/A | 1.443 / 3.810 | 0.867 / 1.041 | 1.232 / 3.331 |
| ppo | 4.516 / N/A | 3.617 / 5.238 | 0.922 / 1.063 | 3.013 / 4.583 |
| octo | 22.826 / N/A | 4.002 / 6.575 | 1.289 / 17.786 | 4.428 / 7.242 |
| humanoid | 5.136 / N/A | 2.236 / 4.935 | 0.902 / 1.045 | 1.818 / 4.082 |
| panda | 3.179 / N/A | 4.569 / 6.699 | 1.510 / 16.219 | 5.514 / 7.788 |
| g1 | 1.583 / N/A | 3.852 / 6.808 | 1.975 / 19.702 | 4.734 / 8.277 |

NVML means are device-wide sampled busy, not SM/Tensor/RT throughput. Peaks are sampled
maxima; CPU accounted means and sampled peaks use different explicit measurements.
Native stage-aligned resources are N/A because it emits no SDK stage marker.

### SDK lifecycle and inference

| Family | Load s | Create s | Reset s | Step+obs s | Input staging s | RPC wait s | RPC mean ms | Rollout s | Close s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| newt | 11.570 | 7.576 | 3.875 | 0.692 | 0.003 | 0.167 | 3.707 | 0.874 | 1.905 |
| hora | 1.722 | 5.086 | 2.507 | 0.945 | 0.001 | 0.305 | 15.227 | 1.257 | 1.167 |
| trifinger | 1.856 | 5.511 | 2.443 | 0.612 | 0.002 | 0.105 | 4.186 | 0.723 | 1.119 |
| robel | 0.457 | 5.107 | 2.005 | 0.309 | 0.002 | 0.017 | 0.694 | 0.332 | 0.979 |
| unitree | 2.035 | 6.571 | 1.325 | 0.240 | 0.008 | 0.164 | 16.381 | 0.417 | 0.623 |
| go2 | 1.920 | 3.306 | 0.475 | 0.150 | 0.001 | 0.090 | 17.954 | 0.243 | 0.294 |
| spin | 1.745 | 3.527 | 0.478 | 0.179 | 0.001 | 0.094 | 18.900 | 0.277 | 0.246 |
| ppo | 2.771 | 9.351 | 4.935 | 1.108 | 0.003 | 0.154 | 2.804 | 1.281 | 2.173 |
| octo | 17.899 | 13.626 | 6.312 | 4.891 | 0.024 | 19.771 | 988.570 | 24.732 | 0.835 |
| humanoid | 2.369 | 4.004 | 1.421 | 0.194 | 0.003 | 0.097 | 6.449 | 0.298 | 0.668 |
| panda | 9.303 | 16.423 | 8.626 | 2.304 | 0.019 | 16.063 | 188.980 | 18.424 | 3.881 |
| g1 | 10.813 | 4.797 | 1.174 | 0.348 | 0.004 | 15.090 | 1,508.999 | 15.450 | 0.696 |

Step+observation, input staging and RPC wait are nested within rollout. CPU wall timings
do not isolate asynchronous GPU execution. The trace-writing overhead is included.

### Native phase boundaries

| Family | Actor load s | Create s | Reset s | Rollout s | Actor pipe round trip s | Close s | Reference end-to-end s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| newt | 3.696 | 7.743 | 3.948 | 0.767 | 0.173 | 1.618 | 20.368 |
| hora | 1.498 | 5.491 | 2.494 | 1.252 | 0.309 | 1.049 | 13.870 |
| trifinger | 1.250 | 5.492 | 2.212 | 0.665 | 0.078 | 0.992 | 12.571 |
| robel | 0.153 | 5.724 | 2.156 | 0.308 | 0.013 | 1.056 | 11.109 |
| unitree | 1.610 | 5.233 | 1.514 | 0.422 | 0.136 | 0.632 | 11.486 |
| go2 | 1.603 | 3.429 | 0.431 | 0.251 | 0.114 | 0.179 | 7.922 |
| spin | 1.729 | 3.229 | 0.436 | 0.262 | 0.094 | 0.181 | 7.741 |
| ppo | 1.619 | 9.874 | 5.252 | 1.269 | 0.137 | 2.302 | 22.597 |
| octo | 11.723 | 10.977 | 6.424 | 23.852 | 18.960 | 0.739 | 56.901 |
| humanoid | 1.829 | 4.044 | 1.216 | 0.275 | 0.077 | 0.553 | 9.956 |
| panda | 8.939 | 15.764 | 8.669 | 17.540 | 15.408 | 3.138 | 59.399 |
| g1 | 9.502 | 5.139 | 1.123 | 14.779 | 14.454 | 0.409 | 34.502 |

Independent GPU actor latency, server queue time, isolated render CPU/GPU duration and
SM/Tensor/RT hardware throughput are N/A. Prefix task failures are not execution failures.

### Per-task records

| Family | Task | Profile | Native H / actual steps | Success once / at prefix end | Prefix return | Truncated |
| --- | --- | --- | --- | --- | --- | --- |
| newt | MS-AntRun-v1 | `ms_ant_run_state` | 1000 / 5 | N/A (return only) | 0.064228646 | false |
| newt | MS-AntWalk-v1 | `ms_ant_walk_state` | 1000 / 5 | N/A (return only) | 4.652080059 | false |
| newt | MS-CartpoleBalance-v1 | `ms_cartpole_balance_state` | 1000 / 5 | N/A (return only) | 4.993438244 | false |
| newt | MS-CartpoleSwingUp-v1 | `ms_cartpole_swing_up_state` | 1000 / 5 | N/A (return only) | 0.348500609 | false |
| newt | MS-HopperHop-v1 | `ms_hopper_hop_state` | 600 / 5 | N/A (return only) | 0.000000000 | false |
| newt | MS-HopperStand-v1 | `ms_hopper_stand_state` | 600 / 5 | N/A (return only) | 0.000000000 | false |
| newt | PickSingleYCB-v1 | `pick_single_ycb_state` | 50 / 5 | 0 / 0 | 0.204430893 | false |
| newt | PlaceSphere-v1 | `place_sphere_state` | 50 / 5 | 0 / 0 | 0.666350126 | false |
| newt | PullCubeTool-v1 | `pull_cube_tool_newt_state` | 100 / 5 | 0 / 0 | 0.538895905 | false |
| hora | RotateSingleObjectInHandLevel0-v1 | `rotate_single_object_in_hand_level0_state` | 300 / 5 | 0 / 0 | 0.258283556 | false |
| hora | RotateSingleObjectInHandLevel1-v1 | `rotate_single_object_in_hand_level1_state` | 300 / 5 | 0 / 0 | 0.384920150 | false |
| hora | RotateSingleObjectInHandLevel2-v1 | `rotate_single_object_in_hand_level2_state` | 300 / 5 | 0 / 0 | 0.381053925 | false |
| hora | RotateSingleObjectInHandLevel3-v1 | `rotate_single_object_in_hand_level3_state` | 300 / 5 | 0 / 0 | 0.352225482 | false |
| trifinger | TriFingerRotateCubeLevel0-v1 | `tri_finger_rotate_cube_level0_state` | 250 / 5 | 0 / 0 | 2.966431618 | false |
| trifinger | TriFingerRotateCubeLevel1-v1 | `tri_finger_rotate_cube_level1_state` | 250 / 5 | 0 / 0 | 2.238461018 | false |
| trifinger | TriFingerRotateCubeLevel2-v1 | `tri_finger_rotate_cube_level2_state` | 250 / 5 | 0 / 0 | 2.939495802 | false |
| trifinger | TriFingerRotateCubeLevel3-v1 | `tri_finger_rotate_cube_level3_state` | 250 / 5 | 0 / 0 | 2.945262432 | false |
| trifinger | TriFingerRotateCubeLevel4-v1 | `tri_finger_rotate_cube_level4_state` | 250 / 5 | 0 / 0 | 1.671286225 | false |
| robel | RotateValveLevel0-v1 | `rotate_valve_level0_state` | 80 / 5 | 0 / 0 | 0.049172748 | false |
| robel | RotateValveLevel1-v1 | `rotate_valve_level1_state` | 150 / 5 | 0 / 0 | 0.049172748 | false |
| robel | RotateValveLevel2-v1 | `rotate_valve_level2_state` | 150 / 5 | 0 / 0 | 0.049172748 | false |
| robel | RotateValveLevel3-v1 | `rotate_valve_level3_state` | 150 / 5 | 0 / 0 | 0.046286553 | false |
| robel | RotateValveLevel4-v1 | `rotate_valve_level4_state` | 300 / 5 | 0 / 0 | 0.054438535 | false |
| unitree | UnitreeG1Stand-v1 | `unitree_g1_stand_root_state` | 1000 / 5 | N/A (return only) | 5.000000000 | false |
| unitree | UnitreeH1Stand-v1 | `unitree_h1_stand_root_state` | 1000 / 5 | N/A (return only) | 5.000000000 | false |
| go2 | UnitreeGo2-Reach-v1 | `unitree_go2_reach_root_state` | 200 / 5 | 0 / 0 | 1.548131824 | false |
| spin | AnymalC-Spin-v1 | `anymal_c_spin_root_state` | 200 / 5 | N/A (return only) | -0.251348883 | false |
| ppo | AnymalC-Reach-v1 | `anymal_c_reach_state` | 200 / 5 | 0 / 0 | 1.617087960 | false |
| ppo | LiftPegUpright-v1 | `lift_peg_upright_ppo_state` | 50 / 5 | 0 / 0 | 1.151947737 | false |
| ppo | PickCube-v1 | `pickcube_state` | 50 / 5 | 0 / 0 | 0.925253153 | false |
| ppo | PokeCube-v1 | `poke_cube_state` | 50 / 5 | 0 / 0 | 0.972863138 | false |
| ppo | PullCube-v1 | `pull_cube_state` | 50 / 5 | 0 / 0 | 1.263951659 | false |
| ppo | PushCube-v1 | `pushcube_state` | 50 / 5 | 0 / 0 | 1.076675296 | false |
| ppo | PushT-v1 | `pusht_state` | 100 / 5 | 0 / 0 | 0.573944926 | false |
| ppo | RollBall-v1 | `rollball_state` | 80 / 5 | 0 / 0 | 0.069395043 | false |
| ppo | StackCube-v1 | `stackcube_state` | 50 / 5 | 0 / 0 | 1.116416693 | false |
| ppo | TwoRobotPickCube-v1 | `two_robot_pickcube_ppo_state` | 100 / 5 | 0 / 0 | 0.095330618 | false |
| ppo | TwoRobotStackCube-v1 | `two_robot_stackcube_ppo_state` | 100 / 5 | 0 / 0 | 0.142150789 | false |
| octo | PutCarrotOnPlateInScene-v1 | `put_carrot_on_plate_in_scene_rgb` | 60 / 5 | 0 / 0 | 0.000000000 | false |
| octo | PutEggplantInBasketScene-v1 | `put_eggplant_in_basket_scene_rgb` | 120 / 5 | 0 / 0 | 0.000000000 | false |
| octo | PutSpoonOnTableClothInScene-v1 | `put_spoon_on_table_cloth_in_scene_rgb` | 60 / 5 | 0 / 0 | 0.000000000 | false |
| octo | StackGreenCubeOnYellowCubeBakedTexInScene-v1 | `stack_green_cube_on_yellow_cube_baked_tex_in_scene_rgb` | 60 / 5 | 0 / 0 | 0.000000000 | false |
| humanoid | MS-HumanoidRun-v1 | `ms_humanoid_run_root_state` | 1000 / 5 | N/A (return only) | 2.188179016 | false |
| humanoid | MS-HumanoidStand-v1 | `ms_humanoid_stand_root_state` | 1000 / 5 | N/A (return only) | 4.714434624 | false |
| humanoid | MS-HumanoidWalk-v1 | `ms_humanoid_walk_root_state` | 1000 / 5 | N/A (return only) | 2.503074884 | false |
| panda | FMBAssembly1Easy-v1 | `fmb_assembly1_easy_rgb` | 500 / 5 | 0 / 0 | 0.000000000 | false |
| panda | LiftPegUpright-v1 | `lift_peg_upright_rgb` | 50 / 5 | 0 / 0 | 0.965016484 | false |
| panda | PegInsertionSide-v1 | `peg_insertion_side_rgb` | 100 / 5 | 0 / 0 | 0.047485210 | false |
| panda | PickClutterYCB-v1 | `pick_clutter_ycb_rgb` | 100 / 5 | 0 / 0 | 0.000000000 | false |
| panda | PickSingleYCB-v1 | `pick_single_ycb_rgb` | 50 / 5 | 0 / 0 | 0.038820539 | false |
| panda | PickCube-v1 | `pickcube_rgb` | 50 / 5 | 0 / 0 | 0.115265667 | false |
| panda | PlaceSphere-v1 | `place_sphere_rgb` | 50 / 5 | 0 / 0 | 0.103202179 | false |
| panda | PlugCharger-v1 | `plug_charger_rgb` | 200 / 5 | 0 / 0 | 0.000000000 | false |
| panda | PokeCube-v1 | `poke_cube_rgb` | 50 / 5 | 0 / 0 | 0.166022211 | false |
| panda | PullCube-v1 | `pull_cube_rgb` | 50 / 5 | 0 / 0 | 0.183611140 | false |
| panda | PullCubeTool-v1 | `pull_cube_tool_rgb` | 100 / 5 | 0 / 0 | 0.037859678 | false |
| panda | PushCube-v1 | `pushcube_rgb` | 50 / 5 | 0 / 0 | 0.161663428 | false |
| panda | PushT-v1 | `pusht_rgb` | 100 / 5 | 0 / 0 | 0.480547607 | false |
| panda | RollBall-v1 | `rollball_rgb` | 80 / 5 | 0 / 0 | 0.031737462 | false |
| panda | StackCube-v1 | `stackcube_rgb` | 50 / 5 | 0 / 0 | 0.190350384 | false |
| panda | TwoRobotPickCube-v1 | `two_robot_pickcube_rgb` | 100 / 5 | 0 / 0 | 0.030750170 | false |
| panda | TwoRobotStackCube-v1 | `two_robot_stackcube_rgb` | 100 / 5 | 0 / 0 | 0.053793415 | false |
| g1 | UnitreeG1PlaceAppleInBowl-v1 | `unitree_g1_place_apple_in_bowl_rgb` | 100 / 5 | 0 / 0 | 0.119086802 | false |
| g1 | UnitreeG1TransportBox-v1 | `unitree_g1_transport_box_rgb` | 100 / 5 | 0 / 0 | 0.154105559 | false |

`return_only` tasks retain null success fields. The SDK structural boolean is not
counted as failure or success for those tasks. Prefix-end success is not native horizon-end success.

### Per-profile SDK lifecycle

| Family | Profile | Prefix elapsed s | Create s | Reset s | Step+obs s | Input staging s | RPC wait s | RPC mean ms | Close s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| newt | `ms_ant_run_state` | 0.040 | 0.483 | 0.438 | 0.031 | 0.000 | 0.007 | 1.460 | 0.267 |
| newt | `ms_ant_walk_state` | 0.197 | 2.892 | 0.424 | 0.092 | 0.000 | 0.103 | 20.642 | 0.199 |
| newt | `ms_cartpole_balance_state` | 0.033 | 0.451 | 0.344 | 0.023 | 0.000 | 0.008 | 1.641 | 0.170 |
| newt | `ms_cartpole_swing_up_state` | 0.025 | 0.452 | 0.347 | 0.019 | 0.000 | 0.005 | 0.958 | 0.233 |
| newt | `ms_hopper_hop_state` | 0.037 | 0.461 | 0.477 | 0.027 | 0.000 | 0.008 | 1.650 | 0.178 |
| newt | `ms_hopper_stand_state` | 0.030 | 0.437 | 0.384 | 0.018 | 0.000 | 0.011 | 2.102 | 0.156 |
| newt | `pick_single_ycb_state` | 0.154 | 0.964 | 0.484 | 0.144 | 0.000 | 0.008 | 1.692 | 0.267 |
| newt | `place_sphere_state` | 0.177 | 0.841 | 0.429 | 0.167 | 0.000 | 0.009 | 1.707 | 0.172 |
| newt | `pull_cube_tool_newt_state` | 0.181 | 0.595 | 0.549 | 0.171 | 0.000 | 0.008 | 1.512 | 0.264 |
| hora | `rotate_single_object_in_hand_level0_state` | 0.519 | 2.999 | 0.542 | 0.261 | 0.000 | 0.257 | 51.339 | 0.225 |
| hora | `rotate_single_object_in_hand_level1_state` | 0.239 | 0.592 | 0.825 | 0.223 | 0.000 | 0.014 | 2.720 | 0.406 |
| hora | `rotate_single_object_in_hand_level2_state` | 0.254 | 0.769 | 0.576 | 0.233 | 0.000 | 0.020 | 3.960 | 0.314 |
| hora | `rotate_single_object_in_hand_level3_state` | 0.245 | 0.725 | 0.563 | 0.229 | 0.000 | 0.014 | 2.888 | 0.222 |
| trifinger | `tri_finger_rotate_cube_level0_state` | 0.248 | 3.235 | 0.444 | 0.199 | 0.000 | 0.047 | 9.382 | 0.188 |
| trifinger | `tri_finger_rotate_cube_level1_state` | 0.110 | 0.556 | 0.390 | 0.098 | 0.000 | 0.010 | 2.070 | 0.196 |
| trifinger | `tri_finger_rotate_cube_level2_state` | 0.137 | 0.586 | 0.600 | 0.112 | 0.001 | 0.023 | 4.600 | 0.304 |
| trifinger | `tri_finger_rotate_cube_level3_state` | 0.123 | 0.626 | 0.636 | 0.108 | 0.000 | 0.013 | 2.583 | 0.233 |
| trifinger | `tri_finger_rotate_cube_level4_state` | 0.106 | 0.509 | 0.373 | 0.093 | 0.000 | 0.011 | 2.295 | 0.199 |
| robel | `rotate_valve_level0_state` | 0.102 | 3.261 | 0.374 | 0.097 | 0.000 | 0.004 | 0.809 | 0.207 |
| robel | `rotate_valve_level1_state` | 0.064 | 0.423 | 0.411 | 0.058 | 0.000 | 0.005 | 0.917 | 0.213 |
| robel | `rotate_valve_level2_state` | 0.054 | 0.510 | 0.409 | 0.048 | 0.000 | 0.004 | 0.845 | 0.201 |
| robel | `rotate_valve_level3_state` | 0.055 | 0.450 | 0.376 | 0.053 | 0.000 | 0.002 | 0.369 | 0.158 |
| robel | `rotate_valve_level4_state` | 0.057 | 0.463 | 0.435 | 0.053 | 0.000 | 0.003 | 0.528 | 0.200 |
| unitree | `unitree_g1_stand_root_state` | 0.171 | 0.844 | 0.611 | 0.126 | 0.007 | 0.036 | 7.176 | 0.378 |
| unitree | `unitree_h1_stand_root_state` | 0.245 | 5.727 | 0.714 | 0.114 | 0.001 | 0.128 | 25.586 | 0.245 |
| go2 | `unitree_go2_reach_root_state` | 0.243 | 3.306 | 0.475 | 0.150 | 0.001 | 0.090 | 17.954 | 0.294 |
| spin | `anymal_c_spin_root_state` | 0.277 | 3.527 | 0.478 | 0.179 | 0.001 | 0.094 | 18.900 | 0.246 |
| ppo | `anymal_c_reach_state` | 0.065 | 0.738 | 0.429 | 0.056 | 0.000 | 0.007 | 1.408 | 0.174 |
| ppo | `lift_peg_upright_ppo_state` | 0.122 | 0.484 | 0.410 | 0.113 | 0.000 | 0.007 | 1.490 | 0.181 |
| ppo | `pickcube_state` | 0.218 | 3.753 | 0.472 | 0.131 | 0.000 | 0.085 | 16.958 | 0.184 |
| ppo | `poke_cube_state` | 0.126 | 0.496 | 0.412 | 0.117 | 0.000 | 0.007 | 1.461 | 0.240 |
| ppo | `pull_cube_state` | 0.108 | 0.463 | 0.757 | 0.101 | 0.000 | 0.005 | 0.972 | 0.176 |
| ppo | `pushcube_state` | 0.109 | 0.476 | 0.365 | 0.100 | 0.000 | 0.007 | 1.461 | 0.172 |
| ppo | `pusht_state` | 0.112 | 0.562 | 0.431 | 0.101 | 0.000 | 0.009 | 1.843 | 0.194 |
| ppo | `rollball_state` | 0.128 | 0.531 | 0.463 | 0.120 | 0.000 | 0.006 | 1.286 | 0.289 |
| ppo | `stackcube_state` | 0.117 | 0.863 | 0.392 | 0.109 | 0.000 | 0.006 | 1.264 | 0.176 |
| ppo | `two_robot_pickcube_ppo_state` | 0.084 | 0.478 | 0.391 | 0.075 | 0.000 | 0.007 | 1.392 | 0.193 |
| ppo | `two_robot_stackcube_ppo_state` | 0.092 | 0.507 | 0.414 | 0.084 | 0.000 | 0.007 | 1.310 | 0.195 |
| octo | `put_carrot_on_plate_in_scene_rgb` | 19.641 | 6.787 | 1.425 | 1.038 | 0.008 | 18.576 | 3,715.145 | 0.231 |
| octo | `put_eggplant_in_basket_scene_rgb` | 1.761 | 2.711 | 1.844 | 1.348 | 0.006 | 0.393 | 78.656 | 0.208 |
| octo | `put_spoon_on_table_cloth_in_scene_rgb` | 1.721 | 2.346 | 1.644 | 1.317 | 0.005 | 0.392 | 78.444 | 0.194 |
| octo | `stack_green_cube_on_yellow_cube_baked_tex_in_scene_rgb` | 1.610 | 1.782 | 1.399 | 1.188 | 0.005 | 0.410 | 82.036 | 0.202 |
| humanoid | `ms_humanoid_run_root_state` | 0.053 | 0.510 | 0.510 | 0.045 | 0.001 | 0.005 | 0.924 | 0.205 |
| humanoid | `ms_humanoid_stand_root_state` | 0.185 | 3.038 | 0.464 | 0.100 | 0.001 | 0.083 | 16.552 | 0.253 |
| humanoid | `ms_humanoid_walk_root_state` | 0.061 | 0.457 | 0.447 | 0.049 | 0.001 | 0.009 | 1.872 | 0.210 |
| panda | `fmb_assembly1_easy_rgb` | 0.194 | 0.610 | 0.502 | 0.122 | 0.001 | 0.069 | 13.888 | 0.220 |
| panda | `lift_peg_upright_rgb` | 10.526 | 4.164 | 0.516 | 0.258 | 0.001 | 10.264 | 2,052.716 | 0.235 |
| panda | `peg_insertion_side_rgb` | 0.274 | 1.124 | 0.519 | 0.190 | 0.001 | 0.081 | 16.210 | 0.195 |
| panda | `pick_clutter_ycb_rgb` | 0.162 | 2.632 | 0.722 | 0.115 | 0.001 | 0.044 | 8.853 | 0.367 |
| panda | `pick_single_ycb_rgb` | 0.237 | 0.652 | 0.528 | 0.143 | 0.001 | 0.091 | 18.183 | 0.199 |
| panda | `pickcube_rgb` | 0.184 | 0.876 | 0.490 | 0.110 | 0.001 | 0.070 | 13.938 | 0.216 |
| panda | `place_sphere_rgb` | 0.205 | 0.558 | 0.456 | 0.108 | 0.001 | 0.094 | 18.782 | 0.202 |
| panda | `plug_charger_rgb` | 0.249 | 0.550 | 0.500 | 0.153 | 0.001 | 0.093 | 18.615 | 0.260 |
| panda | `poke_cube_rgb` | 0.184 | 0.664 | 0.539 | 0.136 | 0.001 | 0.045 | 8.954 | 0.245 |
| panda | `pull_cube_rgb` | 0.198 | 0.623 | 0.534 | 0.114 | 0.001 | 0.080 | 16.030 | 0.258 |
| panda | `pull_cube_tool_rgb` | 0.217 | 0.547 | 0.420 | 0.118 | 0.001 | 0.096 | 19.180 | 0.200 |
| panda | `pushcube_rgb` | 0.173 | 0.517 | 0.424 | 0.098 | 0.001 | 0.072 | 14.415 | 0.208 |
| panda | `pusht_rgb` | 0.192 | 0.511 | 0.469 | 0.095 | 0.001 | 0.094 | 18.776 | 0.265 |
| panda | `rollball_rgb` | 0.184 | 0.561 | 0.445 | 0.098 | 0.001 | 0.082 | 16.477 | 0.192 |
| panda | `stackcube_rgb` | 0.206 | 0.576 | 0.480 | 0.132 | 0.001 | 0.071 | 14.240 | 0.213 |
| panda | `two_robot_pickcube_rgb` | 4.777 | 0.642 | 0.566 | 0.156 | 0.001 | 4.618 | 923.515 | 0.207 |
| panda | `two_robot_stackcube_rgb` | 0.261 | 0.617 | 0.516 | 0.158 | 0.001 | 0.099 | 19.894 | 0.198 |
| g1 | `unitree_g1_place_apple_in_bowl_rgb` | 10.664 | 3.845 | 0.575 | 0.168 | 0.003 | 10.491 | 2,098.109 | 0.295 |
| g1 | `unitree_g1_transport_box_rgb` | 4.786 | 0.952 | 0.599 | 0.181 | 0.002 | 4.599 | 919.889 | 0.401 |

Each profile has one five-tick record at N1. Its lifecycle fields belong to that work item;
family-level model loading/resources are not allocated artificially to individual profiles.
These one-record timings do not estimate an episode-time distribution.

### Trace and record comparisons

| Family | SDK/native records exact | Full prefix traces exact | Actions | Observations including terminal |
| --- | --- | --- | --- | --- |
| newt | True | True | 45 | 54 |
| hora | True | True | 20 | 24 |
| trifinger | True | True | 25 | 30 |
| robel | True | True | 25 | 30 |
| unitree | True | True | 10 | 12 |
| go2 | True | True | 5 | 6 |
| spin | True | True | 5 | 6 |
| ppo | True | True | 55 | 66 |
| octo | True | True | 20 | 24 |
| humanoid | True | True | 15 | 18 |
| panda | True | True | 85 | 102 |
| g1 | True | True | 10 | 12 |

### Pilot overlap repeatability

Pilot `remaining-pilot-r6` is compared with the matching profiles in `remaining-all-r6`.
This is a repeated five-profile subset, not a second full 64-profile matrix.
Rows compare semantic records and every observation/action digest across overlapping profiles.
Timing and resource values are intentionally excluded from equality checks.

| Family | SDK records | Native records | SDK traces | Native traces |
| --- | --- | --- | --- | --- |
| panda | True | True | True | True |
| g1 | True | True | True | True |

### Additional N2 vector checks

`remaining-vector-r6` passed all nine SDK/native/comparison commands at K1/G1/N2, with seeds 0 and 1
in their respective lanes. Three Humanoid profiles, clutter/PandaStick/dual-Panda RGB profiles,
and both G1 RGB profiles produced **16 paired records, 80 actions and 96 observations per path**.
Every prefix retains its full native horizon, executes five ticks and has false native terminal
and truncated masks. The ten success-scored prefixes have zero success; six Humanoid prefixes
are return-only. Exact records and full traces pass independently of the N1 comparison. No
worker/OOM/sampler errors occurred. This checks real batched images, independent model histories,
joint-order routing and recurrent/action-repeat isolation; it is not an autotune or MPS result.

#### N2 SDK clocks

| Family | Records | Steps | Command s | Evaluation s | Episodes/h command / eval | Steps/s command / eval |
| --- | --- | --- | --- | --- | --- | --- |
| humanoid | 6 | 30 | 14.985 | 8.614 | 1,441.5 / 2,507.4 | 2.002 / 3.483 |
| panda | 6 | 30 | 40.049 | 26.791 | 539.3 / 806.3 | 0.749 / 1.120 |
| g1 | 4 | 20 | 37.093 | 24.327 | 388.2 / 591.9 | 0.539 / 0.822 |

#### N2 NATIVE clocks

| Family | Records | Steps | Command s | Evaluation s | Episodes/h command / eval | Steps/s command / eval |
| --- | --- | --- | --- | --- | --- | --- |
| humanoid | 6 | 30 | 12.269 | 7.216 | 1,760.5 / 2,993.4 | 2.445 / 4.157 |
| panda | 6 | 30 | 40.502 | 26.644 | 533.3 / 810.7 | 0.741 / 1.126 |
| g1 | 4 | 20 | 34.958 | 22.099 | 411.9 / 651.6 | 0.572 / 0.905 |

#### N2 resources and request latency

| Family | Path | GPU busy command / eval % | VRAM mean / peak GiB | CPU accounted mean / sampled peak cores | SDK RPC mean ms |
| --- | --- | --- | --- | --- | --- |
| humanoid | SDK | 0.441 / 0.647 | 2.075 / 4.935 | 0.743 / 1.035 | 6.879 |
| humanoid | NATIVE | 2.064 / N/A | 2.333 / 4.935 | 0.877 / 1.024 | N/A |
| panda | SDK | 0.535 / 0.724 | 3.606 / 6.466 | 1.829 / 17.499 | 934.698 |
| panda | NATIVE | 1.633 / N/A | 3.665 / 6.466 | 1.874 / 16.802 | N/A |
| g1 | SDK | 2.575 / 3.830 | 3.862 / 6.939 | 1.893 / 16.236 | 1,432.742 |
| g1 | NATIVE | 2.599 / N/A | 3.960 / 6.939 | 2.068 / 19.011 | N/A |

### Environment sweep, self-test and real quick test

The independent zero-action/render sweep completed all **87/87 declared profiles** at N2 and five
ticks per lane: **81 passed**, covering 51 task IDs. Six profiles failed explicitly with missing-asset
`FileNotFoundError`: state/RGB AssemblingKits, state/RGB TurnFaucet, OpenCabinetDoor state and
OpenCabinetDrawer state. These are the four known blocked tasks, not skipped or converted into
passes. There were no other environment failures. This sweep does not count as learned-policy
acceptance.

The builder's six N1/N2 self-test cases passed for PickCube state/RGB and PandaStick, including
actual CUDA simulation, reset and nonconstant rendered frames. The separate untraced real PPO
`--quick-test` completed PickCube's original 50-tick horizon with **1/1 success** and zero worker/OOM
errors. It used the ordinary manifest without a diagnostic step limit. It is a single full-horizon
quick test, rather than a complete formal suite.

| Quick-test metric | Measured value |
| --- | --- |
| Command / evaluation wall s | 13.826 / 7.392 |
| Episodes/h command / evaluation | 260.4 / 487.0 |
| Steps/s command / evaluation | 3.616 / 6.764 |
| Client inference round trip mean ms | 2.719 |
| Device GPU busy mean % | 4.769 |
| VRAM mean / sampled peak GiB | 1.543 / 3.982 |
| CPU accounted mean / sampled peak cores | 0.687 / 1.043 |
| Success / native control ticks | 1/1 / 50 |
| Independent GPU inference, render, queue time; SM/Tensor/RT counters | N/A |

The external collector preserves original WorkResults, resource samplers, phase clocks and source
hashes in `remaining_summary.json`; `remaining_summary.md` renders every family/profile metric.
`remaining-final-auxiliary-audit.json` independently checks N2, all 87 environment rows and the
untraced quick test. Sampler/trace/phase data remain outside Git. Final repository checks and
exact-head CI are recorded separately; prior-phase CI is not attributed to the new source.

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
describe the generic reference and strict comparator. Diagnostic manifests retain each profile's
frozen values and remain separate from the ordinary full-horizon manifest. The recorded host
commands are `run_remaining_matrix.py --families humanoid
panda g1 ppo newt hora trifinger robel unitree go2 spin octo --label remaining-all-r6`, followed by
`run_remaining_vector.py` for the eight named N2 profiles, `smoke_profiles_final.py --runtime
RUNTIME --label profiles-r6 --workers 2`, and the ordinary `--quick-test`. The external
`run_remaining_final.sh` preserves exact invocations and paths; `collect_remaining.py --root ROOT`
recomputes the tables and strict pairing from original records without executing a model.

| Remaining task | Concrete unresolved requirement |
| --- | --- |
| `AssemblingKits-v1` | Original kit asset archive remains unavailable from the probed endpoint; the declared visual Policy cannot be executed without it. |
| `TurnFaucet-v1` | Required original processed PartNet assets remain blocked by the probed authenticated download; the declared visual Policy is not validated. |
| `OpenCabinetDoor-v1` | Original processed PartNet assets remain unavailable. The pinned real MS-HAB Fetch Open model and depth/contact adapter are implemented; native API probe passes, canonical task validation remains blocked. |
| `OpenCabinetDrawer-v1` | Original processed PartNet assets remain unavailable. The pinned real MS-HAB Fetch Open model and depth/contact adapter are implemented; native API probe passes, canonical task validation remains blocked. |

Public raw PartNet mirrors do not supply the frozen loader's required `mobility_cvx.urdf` or
`mobility_fixed.urdf`. Other inspected archives split/fix object joints or provide point clouds
instead of the original processed assets. Renaming raw URDFs, generating new collision geometry,
accepting download gates or substituting different scenes has not been used to close these gaps.
Asset failures remain recorded failures, rather than skipped passes.

The completed 51/55 five-tick matrix leaves these four task IDs, two original tabletop RGB
profiles, canonical/sibling-profile distinctions and the repository's full-horizon acceptance
requirements unresolved. The historical fifteen full-horizon task comparisons remain fifteen.
Weights, downloads, raw probes and logs stay outside Git under `/autodl-fs`. New R6 records retain
their own source labels and are not attributed to the prior phase's published evidence archive.

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

The initial R7 GPU probe exposed an over-restrictive depth positivity check. The pinned renderer
uses signed int16 millimeter data, including negative encoded values. The final implementation
retains those raw values, and the fresh **R8** simulator/Octo builds install the corrected b47 source.
R7's failed probe is retained separately in the external evidence.

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

## Historical b47 validation: R8

The final experiments ran on 2026-11-09; this report was finalized on 2026-11-10. Fresh
`runtime-r8` and `octo-policy-runtime-r8` install the corrected b47 source. The pre/post-run
source audit passed for all 226 executable/configuration files and all 86 SDK Python files in
each environment. The extended audit also verifies the model/source pins and preserves R6/R7
identities separately. The R8 source archive SHA-256 is
`5f33019eea153a73cf673ae566e7b418c2b187e71f0f94ef3a23a35a79c0f69b`.
The machine, pinned simulator dependencies and K1/G1/N1 configuration match the R6 description;
MPS is disabled. R8 remains evidence for b47, not validation of the b48 changes; R6 above retains
the preceding phase's measurements.

All 36 main-matrix commands passed: **64/64 profile pairs, 51/55 distinct task IDs, 320 actions
and 384 observations per path**, with exact SDK/native pairing and no missing or duplicate
records. Original tabletop RGB coverage remains **16/18**. The 52 success-scored five-tick
prefixes have zero successes; twelve return-scored prefixes retain null success fields. There
are no worker failures, OOM events or actual timeouts. These intentionally incomplete prefixes
are `integration_smoke`, not formal full-horizon evaluations or estimates of task competence.

The sequential command totals below exclude comparator commands. Each family command is measured
separately; their sum is not the elapsed wall time of the complete pipeline. Evaluation clocks
include simulator startup/reset/cleanup inside their respective evaluator boundary. CPU phase sums
can overlap server/GPU work and must not be added to infer total time. Trace copying/hashing is
enabled for the matrix. No runtime speedup is claimed.

| Metric | SDK R8 | Native R8 |
| --- | ---: | ---: |
| Summed command wall / evaluation wall (s) | 370.052 / 248.424 | 313.285 / 214.938 |
| Prefixes/h, command / evaluation | 622.6 / 927.4 | 735.4 / 1071.9 |
| Actual control steps/s, command / evaluation | 0.865 / 1.288 | 1.021 / 1.489 |
| Sample-weighted device GPU busy / sampled peak (%) | 5.487 / 96 | 6.152 / 94 |
| Sample-weighted device VRAM / sampled peak (GiB) | 3.116 / 8.042 | 3.556 / 8.042 |
| Accounted CPU mean / sampled peak (cores) | 1.018 / 19.954 | 1.181 / 18.073 |

Per-family SDK metrics retain the cold command/model/JIT effects; RPC time includes model work,
IPC and response validation. It is not isolated GPU inference latency. Resource samples cover the
whole family command and device-wide VRAM, while CPU/RSS follow the process tree.

| Family | Records | Command / eval (s) | RPC mean (ms) | GPU busy mean (%) | VRAM mean / peak (GiB) | CPU mean (cores) | RSS mean / peak (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| newt | 9 | 30.820 / 20.755 | 3.742 | 4.783 | 2.795 / 5.376 | 0.705 | 2.258 / 4.688 |
| hora | 4 | 18.719 / 12.732 | 14.506 | 5.068 | 3.589 / 8.042 | 0.749 | 2.500 / 5.510 |
| trifinger | 5 | 20.044 / 14.430 | 4.127 | 1.474 | 2.242 / 4.758 | 0.738 | 1.965 / 3.917 |
| robel | 5 | 17.615 / 13.229 | 0.822 | 3.652 | 2.085 / 4.363 | 0.711 | 1.604 / 3.388 |
| unitree | 2 | 17.334 / 9.750 | 19.761 | 0.254 | 2.069 / 5.372 | 0.680 | 1.748 / 5.038 |
| go2 | 1 | 14.327 / 7.392 | 17.336 | 0.143 | 1.433 / 3.810 | 0.628 | 1.127 / 3.389 |
| spin | 1 | 15.387 / 6.714 | 19.138 | 0.119 | 1.382 / 3.810 | 0.562 | 1.034 / 3.385 |
| ppo | 11 | 33.541 / 24.174 | 2.951 | 3.992 | 3.023 / 5.238 | 0.701 | 2.635 / 4.631 |
| octo | 4 | 69.251 / 48.611 | 1008.123 | 16.820 | 3.623 / 6.575 | 1.192 | 4.094 / 7.241 |
| humanoid | 3 | 17.917 / 9.772 | 6.621 | 3.714 | 1.941 / 4.935 | 0.686 | 1.584 / 4.130 |
| panda | 17 | 75.453 / 56.405 | 189.463 | 2.695 | 4.119 / 6.573 | 1.338 | 5.112 / 7.834 |
| g1 | 2 | 39.644 / 24.460 | 1466.856 | 3.019 | 3.586 / 6.808 | 1.641 | 4.324 / 8.104 |

| SDK host interval | Sum (s) |
| --- | ---: |
| `environment_initialization_s` | 88.361876 |
| `environment_reset_s` | 35.042850 |
| `rollout_s` | 64.644225 |
| `step_with_observation_cpu_wall_s` | 12.246377 |
| `observation_prepare_cpu_wall_s` | 0.087308 |
| `inference_wait_s` | 52.159164 |
| `policy_reset_s` | 0.242735 |
| `policy_cleanup_s` | 0.031797 |
| `environment_close_s` | 14.132153 |

The native model-load clocks sum to **48.865 s**. SDK model loading, process startup
and first-query JIT are not independently separated by this collector. Aggregate SDK RPC time is
**162.997 ms/request** over 320 requests. The combined SDK step/observation
host clock above includes physics and rendering; **independent GPU inference, isolated rendering,
queue-only wait and SM/Tensor/RT throughput are N/A**, rather than zero. NVML busy is an activity
indicator, not an SM or ray-tracing utilization counter. Stable-run timing and episode distributions
are not inferred from one five-tick sample per profile.

| Additional validation | R8 result |
| --- | --- |
| N2 paired rollouts | All nine commands pass; 16 record pairs across eight profiles, 80 actions and 96 observations per path match exactly. |
| Complete environment sweep | All 89 profiles attempted: 81 pass across 51 task IDs; eight explicit `FileNotFoundError` results correspond only to AssemblingKits, TurnFaucet, OpenCabinetDoor and OpenCabinetDrawer. |
| Fresh GPU self-test | Six N1/N2 cases pass. |
| Ordinary PPO quick test | PickCube: original 50-tick horizon, 1/1 success; 12.368 s command / 7.114 s evaluation, 291.1 / 506.0 episodes/h and 4.043 / 7.028 steps/s. |
| R6 to R8 offline regression | All 64 SDK records/traces and all 64 native records/traces are exact: 320 actions and 384 observations on each path, including outcomes, returns, steps, masks and initial hashes; no tolerance change or R6 rerun. |
| Existing checks | Linux pytest: 265 passed; Ruff, bash syntax, ShellCheck and maintained-Python support checks passed. |

| N2 family | Pairs | SDK command / eval (s) | SDK RPC mean (ms) | SDK GPU busy (%) | SDK VRAM mean / peak (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| humanoid | 6 | 18.091 / 10.174 | 7.178 | 1.143 | 1.989 / 4.935 |
| panda | 6 | 47.453 / 31.168 | 1041.106 | 2.232 | 3.541 / 6.466 |
| g1 | 4 | 40.820 / 25.087 | 1516.167 | 1.019 | 3.661 / 6.939 |

The untraced quick test records 10.612% mean GPU busy,
1.681/3.982 GiB mean/peak VRAM,
0.775 mean CPU cores and
1.369/3.583 GiB mean/peak RSS. Its mean inference RPC
is 2.756 ms; step-with-observation host time is 1.236 s over 50 ticks. Missing GPU/queue counters
have the same N/A scope as the matrix.

The successful Fetch API fixture above remains excluded from the 51-task count. Historical
full-horizon, 100-episode native comparisons cover 15 of the 55 task IDs. The other 40 IDs lack
corresponding 100-episode-per-task comparisons: in that historical matrix, 36 had only five-tick
prefixes and four remained asset-blocked.
These task-ID counts do not establish acceptance of every declared observation/controller profile.
The original tabletop RGB prefixes cover 16 of 18 profiles; their two asset-blocked profiles overlap
the four blocked task IDs and remain a separate profile-coverage gap.

### Remaining profile acceptance gaps

The [manifest](benchmark-manifest.json) declares **89 profiles: 64 selected by the R8 model matrix,
eight asset-blocked profiles and 17 initially unmatched asset-available profiles**. The b51 candidate
adds exact state-only Humanoid Walk/Run conversions: 64 source inputs match bitwise in CPU checks,
and their N2 five-tick GPU/native prefixes pass; full-horizon comparisons remain pending. This leaves
**15 available profiles without a compatible selected Policy**, plus the two candidates awaiting full-horizon validation. The environment-only
sweep does not supply missing model comparisons. The 15 remaining model-contract gaps are:

| Unmatched profiles | Count | Mismatch with the selected model contract |
| --- | ---: | --- |
| `lift_peg_upright_state` | 1 | Canonical EE position control is 4D; the official PPO requires EE pose control, 7D. |
| `two_robot_pickcube_state`, `two_robot_stackcube_state` | 2 | Canonical EE position control is 8D; the official PPO requires joint delta control, 16D. |
| `pull_cube_tool_state` | 1 | Canonical EE position control is 4D; NEWT requires EE pose control, 7D. |
| `peg_insertion_side_state`, `pick_clutter_ycb_state`, `plug_charger_state`, `fmb_assembly1_easy_state` | 4 | The selected Octo model requires RGB; these profiles provide state observations. No compatible state model is selected. |
| `ms_humanoid_stand_state` | 1 | The egocentric state omits the world-frame quantities required by the selected Stand checkpoint. Walk/Run have exact b51 state reconstructions, pending GPU validation. |
| `unitree_{h1,g1}_stand_state`, `unitree_go2_reach_state`, `anymal_c_spin_state` | 4 | H1/G1 omit required root orientation/angular velocity. Go2 and Anymal already contain world velocities but omit root pose/orientation needed by the selected conversions; joint identities alone are static metadata. |
| `unitree_g1_{place_apple_in_bowl,transport_box}_state` | 2 | The G1 visual transfer requires RGB and root/joint metadata absent from these state profiles. |

The eight asset-blocked profiles are `assembling_kits_{state,rgb}`, `turn_faucet_{state,rgb}`,
`open_cabinet_{door,drawer}_state` and `open_cabinet_{door,drawer}_mshab_depth`. Selected Panda RGB
and Fetch depth Policies cover four of these contracts once the original assets are available;
their four canonical state siblings also lack selected compatible Policies. The categories above
are disjoint for counting, but asset and model blockers can overlap within a profile.

Closing these gaps requires compatible models or separately validated Policy adaptations. Switching
to a sibling RGB/root-state profile or changing its controller validates a different profile; it
does not close the canonical profile's acceptance gap.

A review identified that the full-horizon comparator did not validate native terminal masks.
Historical PPO native records did not persist the actual `terminated` and `truncated` fields.
The b48 change addresses this recording/comparison gap. Historical records and measurements remain
under their original source identities, without fabricated fields or a retroactive claim of
mask-complete validation or validation of the updated head. The generic native reference reuses the
same Policy implementation while independently constructing the Gym environment, features and action
execution path without the Kit Worker or RPC. Its agreement isolates SDK integration; the separate
author-code and numerical checks retain their stated scope and do not establish equivalence to the
original authors' complete algorithms or task competence.

[Compatibility evidence and checksums](https://github.com/QLL-CCDD/RoboScope-Kit/releases/tag/maniskill-issue25-compatibility-20261109)
contain `remaining_summary_r8.json`, `remaining-final-auxiliary-audit-r8.json`,
`remaining-r6-r8-regression.json`, both final source audits, exact runner/collector commands,
per-profile records/traces, lifecycle timings and resource samples. Per-profile and native/N2
resource detail stays in those raw files rather than being duplicated here. The archive excludes
model weights, simulator assets and installed environments; R6, the failed R7 probe and final R8
evidence retain their original source labels.

The user deferred the four original-asset tasks on 2026-11-11. They remain in the 55-task inventory;
the active phase targets the other 51 task IDs and their declared profiles. Deferral does not close
issue 25 or make the full 55-task benchmark accepted.
