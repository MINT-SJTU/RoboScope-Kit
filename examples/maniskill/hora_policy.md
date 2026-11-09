# HORA-RMA Allegro transfer policy

This example runs the released HORA-RMA stage 2 student on all four native
`RotateSingleObjectInHandLevel{0,1,2,3}-v1` state profiles. It is an explicitly
zero-shot transfer from Isaac Gym, not a model trained on these ManiSkill tasks
and not a reproduction of HORA's original success or rotation scores. The
benchmark's robot, observations, reset, controller, physics, success checks and
episode horizons remain unchanged. The student has no task-goal input and was
trained for continuous rotation around one axis; level 3 requests other axes.

## Reproduce

Use the ManiSkill runtime's Python, which already provides PyTorch and NumPy:

```bash
python examples/maniskill/hora_model.py /autodl-fs/models/hora-rma
```

Configure the Policy command as an absolute Python path followed by the absolute
path to `hora_policy.py`, with its directory as `cwd`. Set the checkpoint path to
`/autodl-fs/models/hora-rma` or its `hora-stage2.pth` file. Each episode maintains
its own 30-frame history and cumulative target; reset clears both. The policy
supports batched independent episodes and returns one native action per call.

The download is 7,116,047 bytes and retains only the 1,145,020-byte student;
teacher weights and TensorBoard event files are excluded. Both archive and
checkpoint SHA-256 digests are verified before loading. No training or additional
simulator dependency is required.

## Source and semantic mapping

- Source: [HORA](https://github.com/HaozhiQi/hora/tree/b1f009128f9adad3b07bc8aff91d7916feed09fc),
  revision `b1f009128f9adad3b07bc8aff91d7916feed09fc`.
- Public checkpoint: the repository's README links Google Drive file
  `1AKecNsQZ56TCyJU49DU06GxnQRbeawMu`; use archive member `stage2_nn/last.pth`.
- Archive SHA-256: `beed1cfdc3095899e3e03687d1ebee80539174bf49b8686041e5aface85678f0`.
- Student SHA-256: `b8e16f5d64c3af111205c4b088b9888acaa4d7fc91e2a42c97b6c544ab87d2aa`.

The student consumes normalized joint angles and previous commanded target angles,
not privileged object state, touch values or fabricated observations. Its actor
uses the last three 32D frames and a learned temporal adaptation embedding from
the last 30 frames. The checkpoint includes both frozen observation normalizers.
We use the deterministic deployment path, including its action clamp and target
joint limits, without training-time sensor noise or a custom initial grasp.

The source and native Allegro URDFs have identical 16 joint names and angular
limits. Their *array orders differ*. A real native environment inspection at
ManiSkill revision `baab60ede2e89167c1b7aaed41a9aa8e690a9d1e` established these orders:

| Array | Joint numeric suffix order |
| --- | --- |
| Native observed `qpos` | `0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15` |
| Native controller action | `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15` |
| HORA model | `0,1,2,3,12,13,14,15,4,5,6,7,8,9,10,11` |

The source's `hora/algo/deploy/deploy.py` defines the last permutation. Both
policies issue commands at 20 Hz. The native 105D flattened state starts with
16 measured joint positions in the observed order. All permutation operations
use the documented physical joint identity.

HORA integrates a target in radians:

```text
desired_target = clamp(previous_target + clamp(actor_output, -1, 1) / 24, joint_limits)
native_action = clamp((desired_target - measured_qpos) / 0.1, -1, 1)
```

The final expression follows ManiSkill's fixed `pd_joint_delta_pos` controller,
which commands `measured_qpos + 0.1 * native_action`. It preserves the desired
target whenever this target is within the native controller's allowed delta;
otherwise it saturates at that existing bound. The source model's target history
retains its intended target. Each episode logs source target clipping and native
controller saturation counts to stderr so the transfer's limits remain visible.

This conversion does not reproduce the original closed-loop dynamics: source
HORA uses torque-limited low-gain control and a different simulation frequency;
ManiSkill retains its own PD gains, force limits, contact simulation and initial
open hand. Five-step validation establishes real model inference and native
execution compatibility; it does not establish useful task success.

## Upstream permission notice

Copyright (c) 2022 Haozhi Qi

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
