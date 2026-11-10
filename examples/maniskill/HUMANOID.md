# TD-MPC2 Humanoid transfer references

This example transfers the official TD-MPC2 policies for DM-Control Humanoid
Stand, Walk and Run to the corresponding native ManiSkill tasks. They are real
trained, task-specific Humanoid policies, with an explicit simulator/controller
conversion. They are not ManiSkill-trained experts, and this example does not
claim to reproduce the original DM-Control scores.

The [checkpoint catalog](humanoid-checkpoints.json) pins all three seed-1 files
at Hugging Face revision `8fb2a82efb3bae96941da440128fe1332e4394fd` of
[`nicklashansen/tdmpc2`](https://huggingface.co/nicklashansen/tdmpc2/tree/8fb2a82efb3bae96941da440128fe1332e4394fd).
They total 95,944,422 bytes. Each original 60-tensor checkpoint is loaded
strictly, including the unused dynamics, reward, online and target critic
parameters. The encoder is 67→256→512 with LayerNorm/Mish and SimNorm; the
Gaussian-prior network is 512→512→512→42. Evaluation executes its 21D tanh mean.

Inference follows the official
[`mpc=False`, `eval_mode=True` path](https://github.com/nicklashansen/tdmpc2/blob/332f19e4aba4bffe2dc736f45ea7f7bdc585903a/tdmpc2/tdmpc2.py),
with source revision `332f19e4aba4bffe2dc736f45ea7f7bdc585903a`. It is the learned
policy-prior reference, rather than the full stochastic MPC planner. Existing
Torch 2.6 and NumPy 1.26 inference dependencies suffice.

## Observations and physical actions

The native robot and source DM-Control Humanoid share the same 21 named hinge
joints. Native articulation order, source joint-state order and motor order
are mapped by name; source motor order is not identical to source qpos order.
Reset receives the actual native state/controller joint identities.

The `ms_humanoid_{stand,walk,run}_root_state` profiles retain the original
numeric state and add measured root pose and world linear/angular velocity
through the native articulation API. Stand's native 154D state contains head
height, torso-frame extremities, and egocentric link and COM velocities.

The canonical `ms_humanoid_walk_state` and `ms_humanoid_run_state` profiles
also supply every numerical input required by the same checkpoints. Their
[pinned 185D state](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/control/humanoid.py#L208)
already contains world root linear velocity at `[42:45]`, world root angular
velocity at `[45:48]`, and the WXYZ root quaternion at `[181:185]`. The Policy
reads these actual measurements directly for those exact profile IDs. It
does not construct unused root XYZ values. Both paths reconstruct the original
67D input in source order:

```text
joint angles21 | head height1 | extremities12 | torso vertical3 |
world COM velocity3 | world root velocity3 | body root angular velocity3 |
joint velocity21
```

Stand's COM velocity is rotated back to the world frame. The native measured
root quaternion supplies torso vertical and converts world angular velocity
to MuJoCo's free-joint body frame. Canonical Stand remains unsupported: its
egocentric state omits orientation and world velocities required by this model.
No orientation, velocity or missing body feature is replaced with a constant
or padded to imitate another embodiment.

The session requires `state.proprioception`; each root-state episode also
requires finite `state.root_pose`, `state.root_linear_velocity` and
`state.root_angular_velocity`. Reset requires actual `active_joint_names` and
`action_joint_names` metadata on every supported profile, plus `profile_id` to
select the canonical Walk/Run path. Joint identities describe the existing
articulation/controller order and add no numerical observations. Native state
shape, robot, controller, action dimension and frequencies remain checked.
The `humanoid_control_canonical_state_v1` suite selects these two canonical
profiles at their original 1,000-tick horizons and 100 episodes per profile.

The [native controller](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/agents/robots/humanoid/humanoid.py)
remains `pd_joint_delta_pos`, with its original per-joint gains and ±2 rad
normalized delta range. At each observation, the source motor command `u`
requests joint torque:

```text
tau = motor_gear * u - source_spring * (qpos - spring_reference)
                         - source_damping * qvel
native_action = clip((tau + native_damping * qvel)
                     / native_stiffness / 2, -1, 1)
```

Motor gears and passive spring/damping values are taken from the actual
DM-Control 1.0.14 / MuJoCo 2.3.7 model, cross-checked against its real actuator
and passive joint forces. This matches requested drive torque at the observed
state when unsaturated. PhysX holds the PD target across its substeps, whereas
MuJoCo holds motor torque; clipping, solver/contact dynamics, native foot-only
floor contacts and the different simulator remain transfer differences.
Per-episode clipping counts are reported in the Policy log.

The source wrapper repeats a motor command over two 25 ms control ticks. The
Policy retains that two-tick motor history independently per episode, while
recomputing the native PD delta from each current observation. Physics,
observations and native reward/termination checks still execute each tick.
Native 200 Hz physics, 40 Hz control and 1,000-tick horizons are unchanged.
Reset clears only the selected episode's held command; cleanup releases it.

## Download and validation

```bash
PYTHON=/path/to/maniskill-runtime/env/bin/python
"$PYTHON" examples/maniskill/humanoid_model.py /autodl-fs/models/maniskill-humanoid
```

Use `humanoid_policy.py` as the SDK command and the resulting directory as its
checkpoint. Models and reference evidence stay outside Git.

CPU verification used 96 actual states from DM-Control 1.0.14 / MuJoCo 2.3.7,
with an independently permuted native joint order. Reconstructing the 67D
source observation differed by at most 4.77e-7. Against unmodified source
encoder/prior operations on identical inputs, all three actors were bitwise
equal. The source torque formula matched MuJoCo actuator plus passive forces
exactly; normalized native actions differed from the independent float64
reference by at most 1.20e-7. Two interleaved ten-step episodes per task
(60 public-Policy actions), history reset and cleanup passed with CUDA hidden.
The source numerical operations ran unchanged; only their nonnumeric
TensorDict result container was replaced with a plain dict for this check.
This evidence does not establish cross-simulator physics equivalence. Native
GPU rollout validation is recorded in the benchmark's compatibility report.

The canonical Walk/Run adapter was checked separately on the existing 64 actual
DM-Control Walk/Run reference states with independently permuted joint order.
Its 67D inputs were bitwise identical to the preceding root-feature conversion;
the root-feature path, actor operations and physical action conversion remain
unchanged. This CPU evidence does not establish canonical-profile native rollout
acceptance; those profiles require their own matched SDK/native comparisons.

## Upstream license notice

MIT License

Copyright (c) Nicklas Hansen (2023).

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
