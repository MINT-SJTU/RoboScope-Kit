# Community Go2 locomotion transfer

This example uses the real public
[`HexPlex0xFF/Maniskill-Unitree-GO2`](https://huggingface.co/HexPlex0xFF/Maniskill-Unitree-GO2/tree/5e51960de0e2f989f0b0a4a84300ce324a55544c)
checkpoint as a velocity controller for native `UnitreeGo2-Reach-v1`.
The publisher declares MIT licensing but does not document its training
simulator. It is therefore labeled a community locomotion transfer reference,
not a ManiSkill-trained expert.

[`go2-checkpoint.json`](go2-checkpoint.json) pins the 4,546,539-byte checkpoint,
its SHA-256, the accompanying configuration digest, model joint order and
observation/control conventions. The architecture is the released RSL PPO
ActorCritic: 45 inputs, ELU hidden layers 512/256/128 and 12 actor outputs. All
actor, critic and standard-deviation tensors are loaded strictly. Inference uses
the deterministic actor mean without observation normalization.

The configuration matches the official Genesis
[walking observation and latency contract](https://github.com/Genesis-Embodied-AI/Genesis/blob/5dc441df4961fc0d84784adcf412048f720f5def/examples/locomotion/go2_env.py).
Inputs are body angular velocity scaled by 0.25, projected gravity, commanded
body velocity scaled by `[2, 2, 0.25]`, joint positions relative to the published
defaults, joint velocities scaled by 0.05 and the previous actor output. No
missing state values are padded.

Use an explicitly named root-state profile. The existing native 35-value Reach
observation remains unchanged; it supplies joint positions/velocities and the
actual world-frame robot-to-goal vector. Optional root pose and world angular
velocity features allow quaternion-based body-frame transforms. Active state
joint names and controller joint names are mapped independently to the
checkpoint's explicit joint order.

The goal follower converts the observed goal vector into body-frame desired
`vx`, `vy` and yaw rate, clipping to the published training command ranges
`[-0.5, 0.5]`, `[-0.32, 0.32]` and `[-0.2, 0.2]`. It commands zero velocity when
the native reached-goal flag is true. Task goals and success checks are not
modified.

The source model has one control-tick actuator latency. Each episode therefore
maintains its own previous prediction: the actor observes that history, the
current native tick executes the previous prediction, and the new prediction
is saved for the next tick. Reset clears the history. Absolute joint targets
`published_default + 0.25 * delayed_prediction` are converted to the native
normalized delta controller as `clip((target-current_qpos)/0.7, -1, 1)`.

Native physics, PD gains, control frequency and complete evaluation horizons are
unchanged. The reference source period was 20 ms; native evaluation uses its
specified period, including one native tick of latency. These dynamics differ
from the locomotion reference, so results are transfer results.

```bash
PYTHON=/path/to/maniskill-runtime/env/bin/python
"$PYTHON" examples/maniskill/download_go2.py /autodl-fs/models/maniskill-go2
```

Use `go2_policy.py` as the SDK policy command and pass that directory or its
`model_1000.pt` as the checkpoint. The existing runtime's Torch and NumPy are
sufficient; no training dependencies are installed.

CPU source-contract checks used two interleaved episodes of ten steps, with
independently permuted native state/controller joint orders. Maximum actor
action difference was 4.77e-7 and maximum normalized native action difference
was 2.38e-7. The check also verified latency-history reset and episode cleanup.
Physical rollout acceptance is recorded separately in the benchmark report.
