# Canonical dual-Panda PPO transfers

`dual_panda_policy.py:ManiSkillDualPandaPolicy` composes unchanged, published official PPO actors
for the existing `two_robot_pickcube_state` and `two_robot_stackcube_state` profiles. It requests
only `state.proprioception`, retains their native 8D `pd_ee_delta_pos` controller, and returns one
action per tick. **R13 strict N1/N2 SDK/native comparisons passed at the original 100-tick
horizons.** These are cross-task transfers, not task-trained dual-arm experts. The
[current review](../../benchmarks/maniskill/NATIVE_REVIEW_VALIDATION.md) records cohort counts,
source identities and remaining acceptance gates.

For TwoRobotPickCube, each arm uses the PushCube35 actor with its own measured joints/TCP, the
actual cube and the actual task goal. This is a weak planar-pushing-to-lifting/handover transfer;
neither source actor learned handover or lifting. For TwoRobotStackCube, the left StackCube48
actor receives the real A/B cubes, while the right PushCube35 actor moves real B toward the
observed goal region. This follows the task roles, but coordination is untrained. Neither mapping
uses a fabricated grasp flag, intermediate goal, extra object or hidden simulator state.

## Source models and unchanged inference

[official-models.json](official-models.json) pins the author dataset
`haosulab/ManiSkill_Demonstrations`, revision `3558f72cf99d51d04a6ac4779e88b3da4bbf1628`:

| Actor | Published checkpoint | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| PushCube, state35/action4 | `demos/PushCube-v1/rl/ppo_pd_ee_delta_pos_ckpt.pt` | 1,137,698 | `5210b0d48bed5b1b6b94aeef54d46bb6b5dceaebe812d1356983241d6b0d1b75` |
| StackCube, state48/action4 | `demos/StackCube-v1/rl/ppo_pd_ee_delta_pos_ckpt.pt` | 1,164,322 | `ab2525bf1ead6930e5344c8ff08ed3292698a7ce34a056dfdce1690b3ea92e52` |

The existing `ppo_model.load_model_bundle` strictly loads all original tensors; the Policy also
checks both official hashes and dimensions. Actor inference uses the unchanged three 256-unit
Tanh layers and deterministic actor mean in FP32 with TF32 disabled. No output dimensions are
removed or projected. Each 4D output is forwarded to its corresponding root-frame controller,
left then right, without another action rotation. The native controller enforces its original
bounds. No actor, controller, horizon, reward, reset or termination rule is changed.

## Observed state and frames

The pinned ManiSkill revision is `baab60ede2e89167c1b7aaed41a9aa8e690a9d1e`. State flattening
preserves insertion order: left qpos9/qvel9, right qpos9/qvel9, then task extras. Half-open slices:

| Slice | TwoRobotPickCube66 | TwoRobotStackCube76 |
| --- | --- | --- |
| `0:18`, `18:36` | Left/right qpos9 + qvel9 | Same |
| `36:43`, `43:50` | Left/right TCP xyz+wxyz | Same |
| `50:57` | Cube xyz+wxyz | — |
| `57:60`, `60:63`, `63:66` | Left TCP→cube, right TCP→cube, cube→goal | — |
| `50:53` | — | Goal-region xyz |
| `53:60`, `60:67` | — | A/B xyz+wxyz |
| `67:70`, `70:73`, `73:76` | — | Left TCP→A, right TCP→B, A→B |

PushCube35 receives qpos9, qvel9, TCP pose7, goal3 and object pose7. StackCube48 receives qpos9,
qvel9, TCP pose7, A pose7, B pose7, TCP→A3, TCP→B3 and A→B3. Missing left TCP→B is computed from
the two observed positions. Pick's actual goal is reconstructed explicitly in FP32 as
`cube_xyz + observed_cube_to_goal`; floating-point subtraction/addition need not recover the
unexposed original goal bits exactly. No PickCube42 actor is used: its contact-based grasp flag
is absent from these canonical observations and cannot be replaced truthfully by finger width.

The final table initializer, which overrides task loading poses, places left/right bases at
`(0,-0.75,0), yaw +pi/2` and `(0,+0.75,0), yaw -pi/2`. The single-arm source base is
`(-0.615,0,0), yaw 0`. These are pinned static calibration, not newly read observation features.
Each arm's TCP/object/goal poses are re-expressed with
`T_source_base * inverse(T_target_base)`; full wxyz quaternions and relative vectors rotate,
while its measured qpos/qvel remain unchanged. Both source and target actions are current-relative
root translations (physical xyz limits ±0.1 m) plus one mimic-gripper action. Input calibration
therefore permits direct forwarding of the four actor outputs in each target root frame.

Source and target differ in robot base placement, initial joint distribution and task geometry.
PushCube uses a single Panda, while the target uses wrist-camera Pandas. Dual-pick adds lifting
and handover; dual-stack requires two arms to coordinate moving supports and stacking. State-frame
conversion does not remove these distribution shifts or turn a pushing actor into a grasp expert.
The original 100-tick target horizons and native success criteria remain authoritative.

Primary source: [state flattening](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/utils/common.py#L176),
[dual pick observations](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/tabletop/two_robot_pick_cube.py#L142),
[dual stack observations](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/tabletop/two_robot_stack_cube.py#L196),
[final base calibration](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/utils/scene_builder/table/scene_builder.py#L204-L240),
and [EE controller semantics](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/agents/controllers/pd_ee_pose.py#L76-L108).

## Reproduction and validation scope

Build the pinned runtime using the benchmark builder, then download only the two source models:

```bash
PYTHON=/path/to/maniskill-runtime/env/bin/python
MODELS=/path/to/models/dual-panda
"$PYTHON" examples/maniskill/download_models.py "$MODELS" --tasks PushCube-v1 StackCube-v1
```

Use `dual_panda_policy.py` as the SDK Policy command and `MODELS` as its checkpoint directory.
The manifest suite `dual_panda_transfer_state_v1` selects the two existing canonical profiles.
Run matching SDK/native episodes, seeds and vector groups. The generic native reference bypasses
Kit Worker/RPC and constructs the original Gym tasks while loading this same Policy:

```bash
"$PYTHON" examples/maniskill/native_eval_policy.py \
  --benchmark-manifest /path/to/maniskill-runtime/benchmark-manifest.json \
  --suite dual_panda_transfer_state_v1 --checkpoint "$MODELS" \
  --policy-python "$PYTHON" \
  --policy-module /absolute/path/to/kit/examples/maniskill/dual_panda_policy.py \
  --policy-class ManiSkillDualPandaPolicy --episodes 1 --num-envs 1 \
  --output /path/to/evidence/dual-panda-native.json \
  --trajectory-dir /path/to/evidence/dual-panda-native-traces
```

The external `cpu-proof.json` passed with CUDA uninitialized. For 128 synthetic inputs per task,
the independent SAPIEN Pose oracle's maximum projected-input error was `1.4901161193847656e-7`
for both tasks; action-frame error was below `8.95e-8`. The `2e-6` bound applies only to this CPU
geometry cross-check; no benchmark tolerance changed. All 17 original tensors per actor loaded
strictly; actor/critic outputs on identical inputs and mixed-task action routing were bitwise equal
to author code. Metadata, shape/nonfinite rejection and episode cleanup checks passed. Independently
generated unit poses and unconstrained joint arrays are not kinematically consistent simulator states.

R13 then passed N1/N2 prefixes and full SDK/native comparisons: N1 had two 100-tick episodes,
200 action pairs / 202 observation pairs; N2 had four episodes, 400 / 404 pairs. All actions,
observations, actual masks and outcomes matched exactly, with no worker failure/OOM. Both cohorts
had zero task successes (0/2 and 0/4), so integration equality does not imply useful coordination
or task competence. These samples do not establish success rates or 100-episode acceptance.
Checkpoints, CPU receipts and raw trajectories remain outside Git.
