# TriFinger real-robot expert transfer

This example executes the public RRC push expert on native ManiSkill
`TriFingerRotateCubeLevel0-v1` and `Level1`, and its lift expert on `Level2`,
`Level3` and `Level4`. This is zero-shot transfer, not a ManiSkill-trained policy
or a reproduction of the original robot scores. No simulator profile or
evaluation rule is changed. The push expert does not condition on desired
orientation, so its use on level 1 does not establish orientation competence.

## Run

The existing ManiSkill runtime's PyTorch and NumPy are sufficient. Download the
two weights with:

```bash
python examples/maniskill/trifinger_model.py /autodl-fs/models/rrc-trifinger
```

Set the Policy command to the absolute Python executable and absolute
`trifinger_policy.py` path, with its directory as `cwd`. The checkpoint path
must be the directory containing `push.pt` and `lift.pt`. Total download size is
1,110,214 bytes. Both source-pinned downloads and loads verify SHA-256.

Inference runs the small released networks on CPU; the simulator retains GPU
physics. Each episode tracks its own previous physical torque command and clears
it on reset. Independent episodes can be batched. Logs report saturation in the
native controller and clipping to the source policy's physical input bounds.

## Reproducible source and inputs

Source: [rr-learning/trifinger-rl-example](https://github.com/rr-learning/trifinger-rl-example/tree/e4841da99127a19e319bd06b71d2becd52067e38),
revision `e4841da99127a19e319bd06b71d2becd52067e38`. Its README identifies these
as the experts used to collect the real-robot benchmark's push and lift datasets.

| Weight | Bytes | SHA-256 |
| --- | ---: | --- |
| `push.pt` | 544,099 | `e469bb3690eee61deb073e3fe63a1609253b495ca15060d9a06b406ffa7212b7` |
| `lift.pt` | 566,115 | `680c06886538e8bd609200aef1663b5dfdca9c41d2cd736ab408bb3ecaa05d49` |

The original full observation format is defined in
[trifinger_rl_datasets](https://github.com/rr-learning/trifinger_rl_datasets/tree/f23729f90634f570d7c3dfc2fba5d8c0e838cf44).
Inspection of the actual exported TorchScript graphs establishes that their actor
networks consume only these fields, in this order:

| Physical input | Push dimensions | Lift dimensions |
| --- | ---: | ---: |
| Nine joint angles, finger-first order | 9 | 9 |
| Nine joint angular velocities, same order | 9 | 9 |
| Current cube's eight world-space corners | 24 | 24 |
| Desired cube center / desired eight corners | 3 | 24 |
| Previous requested physical joint torque | 9 | 9 |
| Total | 54 | 75 |

The example invokes the original checkpoint's actor modules, calibration constants
and normalizer on exactly these consumed inputs. It does not fabricate force
measurements, camera confidence, robot IDs or unused observation fields. CPU
comparison of both real checkpoints against their original full-observation
TorchScript path, with 19 independent valid inputs each, produced bitwise-identical
torques (maximum absolute difference `0.0`). This checks model conversion, not
environment trajectory equivalence.

The native 62D state contains all required measured fields: joint positions and
velocities first, followed by fingertip poses and velocities, desired cube pose
and current cube pose. The unused fingertip data is not sent to the actor. Cube
corners follow the source's binary corner order and use the same 0.065 m side
length. ManiSkill quaternions use WXYZ; the explicit vector rotation preserves
the source's geometric meaning without confusing XYZW components. Coordinates
remain in the arena's world frame in metres, with positive Z upwards. The source's
physical observation bounds and frozen normalization are retained.

## Physical controller conversion and limitations

The source experts output nine joint torques in newton metres, bounded by
`[-0.397, 0.397]`. Sending these numbers directly as ManiSkill position actions
would be incorrect. The native controller uses stiffness `Kp=100`, damping
`Kd=10`, and `target_position = current_position + 0.1 * native_action`.
At the observation instant its PD torque is:

```text
tau_PD = Kp * (target_position - current_position) - Kd * measured_velocity
desired_delta = (requested_torque + Kd * measured_velocity) / Kp
native_action = clamp(desired_delta / 0.1, -1, 1)
```

This inverse matches the requested torque in the controller's PD force law at
that instant when unsaturated. PhysX solves its drive implicitly, and the PD
controller responds to the moving joint during subsequent physics substeps;
this conversion does not reproduce a held torque trajectory. The source policy ran at 50 Hz; native
ManiSkill remains at its frozen 20 Hz with its original robot initialization,
physics and episode termination. Dynamics, controller gains, physical actuation,
and frequency differences are transfer limitations. Five-step acceptance checks
real learned inference and finite native execution, not preservation of RRC task
success or torque trajectories.

Native inspection on November 9, 2026, at ManiSkill revision
`baab60ede2e89167c1b7aaed41a9aa8e690a9d1e` confirmed these orders:

| Array | Joint order |
| --- | --- |
| Observed positions and velocities | Base joints `0,120,240`, upper joints `0,120,240`, lower joints `0,120,240` |
| Controller and RRC model | Base/upper/lower of finger `0`, then finger `120`, then finger `240` |
| Gather observed arrays into model/controller order | `0,3,6,1,4,7,2,5,8` |

The same inspection confirmed state dimension 62, force-drive mode,
`interpolate=False`, `use_target=False`, gains `100/10`, a 20 Nm force limit and
normalized current-relative delta bounds of ±0.1 rad. The adapter keeps these
permutations explicit; it does not map actuators between different robot hands.

## Source license

BSD 3-Clause License

Copyright (c) 2022, Max Planck Gesellschaft.
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
