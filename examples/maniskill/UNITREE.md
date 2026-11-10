# Unitree standing transfer references

The official [Unitree RL Gym](https://github.com/unitreerobotics/unitree_rl_gym/tree/276801e46c5d433564f24658bac64f254b7d2d4b)
publishes recurrent H1 and G1 locomotion actors. This example applies them with
zero velocity commands to ManiSkill's H1/G1 standing tasks. They are real trained
models transferred across simulators and robot variants, rather than
ManiSkill-trained experts. Task rewards, termination, initial-state sampling,
physical parameters and episode horizons remain the benchmark's native values.

[`unitree-checkpoints.json`](unitree-checkpoints.json) pins the source commit,
checkpoint sizes and hashes, actor joint order, native controller joint order,
and both sets of nominal joint positions. The two TorchScript files total
285,090 bytes and remain outside Git. The existing ManiSkill runtime already
contains the needed Torch and NumPy dependencies.

| Model | Actor state/action | Native state/action | Native robot |
| --- | --- | --- | --- |
| Official H1 locomotion | 41 / 10 | 38 / 19 | unitree_h1_simplified |
| Official G1 locomotion | 47 / 12 | 74 / 37 | unitree_g1_simplified_legs |

Only explicitly named root-state profiles can run these references. The
canonical `state.proprioception` vector remains unchanged. The profiles add
`state.root_pose` (seven values: xyz position followed by wxyz quaternion),
`state.root_linear_velocity` (three world-frame values), and
`state.root_angular_velocity` (three world-frame values) from the native public
robot API. This actor consumes root pose and angular velocity; it does not use
root linear velocity. `active_joint_names` metadata defines state joint order;
`action_joint_names` defines controller order. Missing orientation is never
replaced by an upright constant.

The actor input follows the official
[deployment script](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/deploy/deploy_mujoco/deploy_mujoco.py):
body angular velocity scaled by 0.25, projected gravity, zero velocity command,
leg joint positions relative to the model's default angles, leg velocities
scaled by 0.05, previous actor action, and sine/cosine of a 0.8-second gait phase.
World angular velocity is rotated into the root frame using the observed
quaternion. Joint names are reordered explicitly to the model's MuJoCo joint
order, verified against its pinned XML.

Each actor predicts absolute leg targets as
`target = policy_default + 0.25 * actor_action`. The native controller accepts
normalized position deltas, so the example returns
`clip((target - current_qpos) / 0.2, -1, 1)` in native controller order. Torso,
arms and fingers retain the native standing keyframe targets. It uses native
control bounds, rather than changing the controller to fit the model.

The official actors contain a 64-unit LSTM and mutate recurrent buffers. Each
episode owns an independent actor copy, reset LSTM state and previous-action
history. Episode cleanup releases that state. Gait phase starts at zero and
advances using the benchmark's actual control period. The authors' deployment
period was 20 ms; the native ManiSkill standing period is 50 ms. PD gains and
robot geometry also differ, so these transfer results must not be presented as
reproductions of the authors' locomotion performance.

```bash
PYTHON=/path/to/maniskill-runtime/env/bin/python
"$PYTHON" examples/maniskill/download_unitree.py /autodl-fs/models/maniskill-unitree
```

Use `unitree_policy.py` as the SDK policy command, pass the generated
`models.json` as the checkpoint path, and select the explicitly named H1/G1
root-state profiles. Rollout acceptance and native comparisons are recorded in
the benchmark acceptance report.

CPU inference was compared with the original TorchScript actors and independently
assembled deployment observations on two interleaved episodes of ten steps per
model. The maximum native action difference was 2.15e-6 for H1 and 1.35e-6 for
G1, including quaternion transforms and native action conversion. Reset cleared
recurrent state; episode cleanup released all histories. These are inference
checks, not physical rollout results.

The upstream model/deployment source is distributed under the following license:

```text
BSD 3-Clause License
Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
