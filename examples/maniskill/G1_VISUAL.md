# ManiSkill Octo transfer to G1 manipulation

`g1_visual_policy.py` applies the real RDT-author-released, ManiSkill-fine-tuned
Octo model to `UnitreeG1PlaceAppleInBowl-v1` and `UnitreeG1TransportBox-v1`.
This is an explicit Panda-to-G1 visual transfer reference. The checkpoint was
not trained on these G1 tasks, and neither its original Panda results nor
G1-expert success rates are claimed.

The Policy reuses the pinned [ManiSkill Octo model](PANDA_OCTO.md), its
[checkpoint/source catalog](panda-octo-checkpoint.json), original image
processing, two-frame history, per-episode PRNG and four-action output queue.
The released observation encoder has no proprioception tokenizer. It consumes
the actual configured RGB image and task text; G1 joint positions are used
only for physical action retargeting. No fictitious Panda joint vector is
constructed or supplied to the model.

Use the named G1 visual profiles with the original native upper-body robot,
camera, 25D `pd_joint_delta_pos` controller, 100/20 Hz physics/control rates,
100-tick horizon and task resets/rewards/success checks. The task's numeric
state remains unchanged. Reset metadata supplies the actual state-joint and
controller-joint identities independently. The model accepts the same
checkpoint directory as the Panda Octo example, with the additional native
kinematics file downloaded below.

## Cartesian retargeting

The [kinematics helper](g1_kinematics.py) pins the unmodified native
`g1_simplified_upper_body.urdf` at ManiSkill revision
`baab60ede2e89167c1b7aaed41a9aa8e690a9d1e`, SHA-256
`3ab8835fb251c8313974c13cb9e943e09fbeed9575ec86c36002ee6436868344`.
It evaluates the real robot's joint transforms and TCP poses in the articulation
root frame. No simulator state is modified by FK or IK.

The source model returns normalized `[dx, dy, dz, rx, ry, rz, gripper]` actions.
For its pinned controller, translation clips each raw coordinate to ±1 and
scales by 0.1 m. Rotation is handled separately: norm-clip the raw rotation
vector to length one, then multiply by **−0.1**. Do not coordinate-clip the
rotation before its norm clipping, reverse its sign, or interpret its units
as physical radians before scaling.

The pinned native GPU controller feeds these scaled values directly to its
differential geometric-Jacobian IK path. This transfer integrates that
root-frame angular intent into a finite G1 target using Rodrigues rotation:

```text
target_position = current_position + 0.1 * clip(raw_translation, -1, 1)
angular_delta = -0.1 * raw_rotation / max(1, norm(raw_rotation))
target_rotation = Exp(angular_delta) @ current_rotation
```

The CPU controller's separate Euler-XYZ target-pose path does not describe
the GPU differential solver. The G1 finite-target DLS conversion is a transfer
approximation; it does not reproduce Panda joint targets or trajectories.

For apple placement, a single independent visual stream commands the right
TCP. For box transport, two independently reset streams use left/right-hand
role prefixes on the unchanged native task instruction and share the actual
scene image. Their two TCP targets enter one stacked DLS problem, including
the shared torso joint. This is a compositional transfer strategy, rather
than a bimanual policy trained on the G1 task. Uncommanded joint targets remain
at their current measured positions.

## Native hand and joint controls

The source Panda gripper's positive end is open. The native G1 source defines
open-hand distance as mean absolute position of its six flexion joints
`one` through `six`, so zero is open. For each commanded hand, this transfer
uses `amount = (1-clip(raw_gripper,-1,1))/2`, between zero and the furthest
joint-limit endpoint from zero. The URDF's limits are mirrored between hands;
their signs are not copied from one hand to the other. The `zero` thumb-abduction
joint is held at its current measured position. Choosing the full-limit
closed posture is this transfer's controller convention, not an upstream
native hand-close primitive.

The resulting joint target is converted into the original current-relative
native controller: `(target-current)/0.2` for the eleven torso/arm axes and
`(target-current)/0.5` for the fourteen hand axes, in actual controller order.
Actions are clipped to the native normalized bounds. The Policy logs clipped
joint counts and the Cartesian residual of the final clipped joint targets
against the requested TCP goals. This is a model-side commanded-target
residual, rather than a task success score or a measured physical tracking error.

Each native episode owns its hand streams, instructions and source model
history. Reset clears only that episode's state, and cleanup releases every
corresponding visual stream. The immutable FK structure can be shared between
episodes; qpos, goals and model histories are not shared.

## Run and evidence

```bash
PYTHON=/path/to/octo-policy-runtime/env/bin/python
"$PYTHON" examples/maniskill/download_panda_octo.py /autodl-fs/models/maniskill-octo
"$PYTHON" examples/maniskill/g1_kinematics.py /autodl-fs/models/maniskill-octo
```

Use `g1_visual_policy.py` as the SDK command and that model directory as the
checkpoint. The additional conversion needs only NumPy; source model/runtime
dependencies are those of the ManiSkill Octo example. Weights, URDF downloads,
observations and logs stay outside Git. The compatibility report records
actual native-GPU FK, real-model rollout and SDK/native trajectory validation
separately from model-side kinematic conversion checks.
