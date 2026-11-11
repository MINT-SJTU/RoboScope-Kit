# LiftCube actor prior for the existing LiftPeg position profile

`liftcube_policy.py` selects the existing `lift_peg_upright_state` through
`liftcube_prior_state_v1`. Its complete 32D observation, 4D position/gripper
controller, 100/20 Hz simulation/control rates, 50-tick horizon, seeds and upright
success rule remain unchanged. No profile or simulator variant is added.

This is an explicit **actor-mean transfer prior**, not an upright-peg expert or
the authors' complete stochastic MPC evaluator. The existing 7D LiftPeg PPO
reference remains separate and already covers this TaskID in `upstream_tasks_v1`.
The 4D profile belongs to the original tabletop state suite. Validating it adds
profile coverage, not another distinct TaskID.

## Model and source configuration

[liftcube-checkpoint.json](liftcube-checkpoint.json) pins the authors' actual
31,468,066-byte `maniskill2/lift-cube-1.pt` at Hugging Face revision
[`8fb2a82efb3bae96941da440128fe1332e4394fd`](https://huggingface.co/nicklashansen/tdmpc2/tree/8fb2a82efb3bae96941da440128fe1332e4394fd/maniskill2).
SHA256 is `da231968f24a40dd12cbb9415cd2c2ca41c1f7ff964d09b521a14c8866ea2d6a`.
The source/model license is MIT. The original checkpoint metadata supplies the
5M configuration: two encoder layers, encoder256, latent/MLP512, five critics,
101 bins and SimNorm groups of eight. All 60 original FP32 world-model tensors
load strictly, including the unused dynamics, reward and critic parameters.
The unchanged encoder and four-output actor mean execute without external state
normalization, action slicing, padding, fabricated goals or grasp flags.

The pinned
[TD-MPC2 source](https://github.com/nicklashansen/tdmpc2/tree/7ec6bc83a82a5188ca3faddc59aea83f430ab570)
supports `mpc=False`, `act(..., eval_mode=True)`, returning the learned policy
mean. That is this example's declared configuration. Its standard
`python evaluate.py task=lift-cube model_size=5 checkpoint=... seed=1` instead
uses `mpc=True` and omits `eval_mode`, whose default is false. Its stochastic
planner, candidate sampling and final action noise are not reproduced here.
Actor parity must not be described as equality to that full source evaluation.

## Actual input and complete action mapping

The source is
[ManiSkill2 v0.4.1](https://github.com/mani-skill/ManiSkill/tree/90a628059cbdbcd76303b899b192efc4a38beddc).
Its LiftCube42 observation can be constructed from the actual native LiftPeg32:

| Source field | Source offsets | Native measurement |
| --- | --- | --- |
| Joint positions / velocities | 0:18 | Complete native0:18, including both fingers |
| Panda base pose, XYZ/WXYZ | 18:25 | Actual fixed table-scene base `[-.615,0,0,1,0,0,0]` |
| TCP pose, XYZ/WXYZ | 25:32 | Complete native18:25 |
| Object pose, XYZ/WXYZ | 32:39 | Complete real peg pose, native25:32 |
| TCP-to-object displacement | 39:42 | Actual peg position minus actual TCP position |

Both pinned table-scene builders use that base pose. Non-target controllers
emit no additional controller state. The source grasp/lift reward trains an
**active fourth gripper output**, unlike FetchReach's ignored grip channel.

Source XYZ translates in the current TCP frame; the target translates in robot
root axes. `liftcube_projection.py` rotates the full clipped source XYZ by the
current measured TCP quaternion. Both translation bounds are +/-0.1 m. Rotation
can produce a target component outside the native normalized bounds. The native
controller applies its real component clipping; the Policy does not shrink all
axes to hide this mismatch. Per-episode logs count saturated XYZ components.
Intended Cartesian displacement is preserved only when representable inside
the target bounds. This is not general physical-controller equivalence.

The learned fourth value is retained unchanged. Source and native symmetric
finger targets both use the normalized range mapped to [-.01,.04] m. The source
wrapper repeats the complete 4D action twice. Each episode therefore holds its
original source action for two native ticks, reapplying the frame transform with
each current TCP pose. Reset and cleanup isolate this state between episodes.
Observations, simulation and success/termination checks still run every tick.

## Validation status

The external CPU prototype passed original `WorldModel.encode/pi` and complete
`TDMPC2.act` actor-mean comparisons: all four outputs bitwise equal for 147
synthetic rows and one existing real reset observation from the 7D PPO sibling.
All 60 tensors remained unchanged. Two interleaved episodes passed 24 public API
actions with two-tick holds and isolated reset/cleanup. The source and target
observation/controller functions were checked separately with synthetic CPU
fixtures: exact input/gripper mapping; target saturation remains a measured
limitation. The synthetic fixtures are not native rollouts.

**b56 is a code candidate; clean-build, self-test and real SDK/native validation
are pending.** These CPU checks do not increase the current 74-profile/51-TaskID
rollout count or establish success rates. Cube-to-peg geometry, source500Hz versus
target100Hz physics, and the different task objective remain transfer limits.

## Run

Use the existing runtime's Torch and NumPy; no new inference package is needed.
Download verified weights outside Git:

```bash
/path/to/runtime/env/bin/python examples/maniskill/liftcube_model.py /path/to/liftcube-model
```

Set a policy manifest's command to the absolute runtime Python and
`examples/maniskill/liftcube_policy.py`, with the example directory as `cwd`.
Use `--checkpoint-source /path/to/liftcube-model/lift-cube-1.pt` and suite
`liftcube_prior_state_v1`. The generic native reference uses the same declared
actor prior and frozen target protocol:

```bash
/path/to/runtime/env/bin/python examples/maniskill/native_eval_policy.py \
  --benchmark-manifest /path/to/runtime/benchmark-manifest.json \
  --policy-module /path/to/kit/examples/maniskill/liftcube_policy.py \
  --policy-class ManiSkillLiftCubePolicy --policy-python /path/to/runtime/env/bin/python \
  --checkpoint /path/to/liftcube-model/lift-cube-1.pt --suite liftcube_prior_state_v1 \
  --episodes 1 --num-envs 1 --output /path/to/native-liftcube.json
```

This native path independently constructs the target environment and execution
loop while reusing the Policy. It tests SDK integration separately from the
author-function CPU proof. Preserve original horizons for full-horizon checks;
any five-tick diagnostic must remain explicitly non-formal.
