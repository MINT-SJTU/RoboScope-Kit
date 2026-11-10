# Pretrained ROBEL DClaw transfer

This example uses the released ROBEL `DClawScrewRandom-v0` actor for zero-shot
transfer onto all five native ManiSkill `RotateValveLevel{0,1,2,3,4}-v1` state
profiles. It is a genuine trained nine-joint, three-finger DClaw policy. It is
not a ManiSkill-trained policy or a reproduction of the original ROBEL scores.
The benchmark's geometry, reset, observations, PD controller, simulation,
success thresholds, termination masks and episode horizons remain unchanged.

## Run

The existing runtime's NumPy is sufficient; no old TensorFlow, MuJoCo or hardware
package is installed. Download the 609,486-byte checkpoint with:

```bash
python examples/maniskill/robel_model.py /autodl-fs/models/robel-dclaw
```

Configure the Policy command with an absolute Python path and absolute
`robel_policy.py` path, and use that script's directory as `cwd`. The checkpoint
path may be the directory or its `DClawScrewRandom-v0-policy.pkl` file.

The checkpoint comes from author-maintained
[vikashplus/robel_dev](https://github.com/vikashplus/robel_dev/tree/039466811957f8d046213eddaa7d0e764971030d),
revision `039466811957f8d046213eddaa7d0e764971030d`, path
`robel/scripts/data/softlearning/DClawScrewRandom-v0-policy.pkl`.
SHA-256 is `7c536de59cfe1a75e7e16e2a8e446df6d2ea9ff59bc0dd12b3b8332693ebadf5`.
Both downloader and loader verify the digest. The loader extracts only the
serialized NumPy arrays through explicitly allowed object constructors;
TensorFlow and Softlearning objects are not instantiated.

## Model contract

The checkpoint records `21 -> 256 -> 256 -> 18` dense layers with ReLU, ReLU
and linear activations, and no input preprocessor. The 18 outputs split into
nine Gaussian means and nine log standard deviations. The source
`enjoy_softlearning.py` selects deterministic evaluation, which uses
`tanh(mean)`. This formula is established by the checkpoint-era
[Softlearning GaussianPolicy](https://github.com/rail-berkeley/softlearning/blob/7bbfcdba8ad6efd56811df946718c3f175f25499/softlearning/policies/gaussian_policy.py)
and its `SquashBijector`, revision `7bbfcdba8ad6efd56811df946718c3f175f25499`.

The 21 physical inputs, in source order, are nine measured joint angles,
cosine and sine of valve angle, the previous nine normalized commands, and
wrapped desired-angle error. No unmeasured force, velocity or shape input is
invented. The native 51D state contains the needed joint angles, valve joint
angle, valve root pose and requested rotation direction. Episode reset clears
the previous command and the policy's desired trajectory.

The screw policy internally tracks a moving desired angle. On transfer this
trajectory starts at the measured valve angle and moves at 0.5 rad/s in the
native requested direction. Its integration uses the native 20 Hz command rate.
This is policy-side goal construction, not a change to native success criteria.
The source trained random target velocities in `[-0.75, 0.75]` rad/s; transfer
uses the fixed magnitude above.

The NumPy actor uses the original six weight arrays without training. A
comparison across 31 observations against the source dense/ReLU/tanh formula evaluated in
float64 measured maximum absolute error `1.46e-6`; all actions were finite and
bounded. This is a numerical inference check, not TensorFlow rollout parity.

## Joint, frame and controller mapping

Native environment inspection confirmed that both observed joints and controller
commands use breadth-first order:
`f1_0,f2_0,f3_0,f1_1,f2_1,f3_1,f1_2,f2_2,f3_2`.
ROBEL uses finger-first order `FFJ10..12,MFJ20..22,THJ30..32`.
Its three-finger MJCF and the native CAD-derived URDF have different hinge signs
and finger labels. Mapping only by array dimension would be wrong.

The source chain is defined in
[robel-scenes_dev](https://github.com/vikashplus/robel-scenes_dev/blob/9d774735f5d6a599774d01d8808f411054f8cf28/dclaw/assets/chain3xh.xml),
using the ROBEL source revision's exact submodule commit.
Comparing mounted joint axes and finger locations with ManiSkill revision
`baab60ede2e89167c1b7aaed41a9aa8e690a9d1e` gives the following explicit homologous
joint mapping. Signs apply to both measured angles and commanded targets.

| ROBEL finger joints | Native joints | Signs |
| --- | --- | --- |
| `FFJ10, FFJ11, FFJ12` | `f1_0, f1_1, f1_2` | `+, -, +` |
| `MFJ20, MFJ21, MFJ22` | `f3_0, f3_1, f3_2` | `-, +, +` |
| `THJ30, THJ31, THJ32` | `f2_0, f2_1, f2_2` | `-, +, -` |

ROBEL's planar palm frame corresponds to a -30 degree world yaw in the native
mounting. This matches its finger locations at approximately 60, -60 and 180
degrees to native f1, f3 and f2 at approximately 30, -90 and 150 degrees. The
source uses rounded MJCF angles and the native uses CAD geometry, so link and
contact geometry are transfer differences rather than asserted identical bodies.

The source valve hinge rotates about +Z. In native `build_robel_valve`, the
SAPIEN revolute +X axis is conjugated by `Ry(pi/2)`, giving -Z. With the observed
root yaw and the palm-frame calibration, source valve phase is:

```text
source_angle = native_root_yaw - native_valve_joint_angle + pi/6
source_desired_velocity = -native_rotate_direction * 0.5
target_error = wrap_to_pi(source_desired_angle - source_angle)
```

This preserves physical hinge direction. The canonical random root pose and
valve shape remain in place. Novel valve shapes, including unobserved individual
head offsets, remain a zero-shot domain shift.

ROBEL denormalizes the actor's commands to absolute joint targets. For each
finger the ranges are `+/-[0.48, pi/3, pi/2]` radians. After the signed permutation:

```text
desired_source_qpos = normalized_source_action * source_joint_range
desired_native_qpos = signed_permutation(desired_source_qpos)
native_action = clamp((desired_native_qpos - measured_native_qpos) / 0.1, -1, 1)
```

This respects the native PD delta bound; its target equals the requested
absolute target when unsaturated. Each episode logs the number of clipped native
joints and steps. ROBEL's source gains and 10 Hz execution differ from native
ManiSkill's fixed 20 Hz, gains and force limits. Five-step validation establishes
real pretrained inference and native execution compatibility; it does not
establish useful task success or equivalent source trajectories.

## Attribution

ROBEL source and the released example policies are provided in the author
repository under Apache License 2.0. Copyright 2019 The ROBEL Authors.
See the [source license](https://github.com/vikashplus/robel_dev/blob/039466811957f8d046213eddaa7d0e764971030d/LICENSE).
The adapter implements the documented mathematical inference formula and does
not redistribute TensorFlow or Softlearning source.
