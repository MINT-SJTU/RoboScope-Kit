# Canonical state PPO priors

`canonical_prior_state_v1` selects the existing PullCubeTool and UnitreeGo2 Reach
profiles. It preserves their native observations, controllers, seeds, horizons and
success rules. These are explicit transfers of released learned policies, not
task-trained experts. Weights remain outside Git.

## PullCubeTool direct-pull prior

The unchanged official PullCube actor accepts 35 values and emits all four native
EE-position/gripper actions. Its pinned checkpoint and source are recorded in
[reference-ppo-models.json](reference-ppo-models.json). The target's actual 39D state
contains qpos9, qvel9, TCP pose7, cube pose7 and tool pose7. The adapter retains the
first 25 values and complete cube pose, adding the native task's fixed dense-reward
workspace target between them. No quaternion or action component is discarded.

At pinned ManiSkill `baab60ede2e89167c1b7aaed41a9aa8e690a9d1e`,
[`TableSceneBuilder`](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/utils/scene_builder/table/scene_builder.py)
fixes the Panda base at `[-0.615, 0, 0]`; the
[`PullCubeTool` dense reward](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/tabletop/pull_cube_tool.py)
adds `[0.05, 0, 0]` in FP32. This authored constant is distinct from the progress
center's 0.035 offset and the success test's XY radius 0.6. It is not a sampled or
observed hidden goal. Native reward and success remain unchanged.

The source actor ignores tool pose: this is a weak direct-pull prior, with no claim
of learned tool use. Source PullCube uses a 50-tick horizon; the target retains
100 ticks. [pull_tool_prior.py](pull_tool_prior.py) contains the NumPy Policy and
native Torch mappings. Independent CPU checks execute the pinned native observation,
flattening, base-pose and reward-source components on 147 synthetic samples. Both
input mappings, original actor/critic outputs and all 17 FP32 tensors match their
stated source contracts. These synthetic checks do not establish task competence.

## Go2 Reach joint prior

The official ANYmal Reach actor and the canonical Go2 Reach environment both
provide 35 fields: qpos12, qvel12, world linear/angular velocity6, reached-goal1,
world goal XY2 and robot-to-goal XY2. All fields and twelve learned outputs are
retained. No root orientation or synthetic sensor measurement is introduced.

[go2_reach_prior.py](go2_reach_prior.py) pins the complete corresponding joint
names, native home poses and normalization bounds. Corresponding frozen URDF axes
agree in the base frame at zero joint configuration; no hind-leg sign flip is
assumed. Input joint positions use source-home plus target displacement from
target-home. Joint velocities and the remaining eleven measured fields are
unchanged. The source controller clips actor outputs before scaling them to
physical deltas within +/-0.225 radians. Inverse normalization expresses every
delta through the unchanged Go2 controller's +/-0.7 radian bounds.

CPU checks retain all 17 original FP32 tensors and execute the source controller's
actual affine clip/scale function. Home-coordinate inversion has at most
`1.1921e-7` radian roundoff; target normalization has at most `1.4902e-8` radian
physical-delta error in the synthetic sample. These are transfer-map diagnostics,
not benchmark tolerances. Native target SDK/reference comparisons remain exact.

The SDK and direct native evaluator run the action conversion on the same Torch
device before copying actions to the host. NumPy and CUDA scalar division can
round differently in FP32; b53's mixed-backend conversion first diverged at control step
2 in the Go2 full-horizon check. A synthetic probe reproduced 72 differing
components out of 1,536, maximum `2.9802e-8`. The failed b53 records remain separate.

The [shared native observation builder](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/quadruped/quadruped_reach.py)
and frozen [ANYmal](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/agents/robots/anymal/anymal_c.py)
and [Go2 controllers](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/agents/robots/unitree_go/unitree_go2.py)
define these contracts. Limb geometry, joint limits, PD gains and contact dynamics
differ. Home alignment is not an exact kinematic mapping or a claim of learned
Go2 competence. Source and target retain their original 200-tick horizons.

## Current real-model validation

R16/b55 at code head `c660435b235cd9d3db303fcc3c14ee6beba1ebe1` passed strict
SDK/native checks for these two existing profiles. All actions, complete observation
hashes, actual terminal masks, native returns and outcomes agree exactly, with zero
Worker failures/OOM. Task/controller/seed/scoring definitions remain unchanged.

| Cohort | Episodes per profile | Seeds | N | Steps per episode | Action / observation pairs |
| --- | ---: | --- | ---: | --- | --- |
| Full horizons N1 | 1 | 0 | 1 | PullCubeTool 100; Go2 200 | 300 / 302 |
| Five-tick tail check | 3 | 0/1/2 | 2 | 5 | 30 / 36 |
| Full horizons N2 | 2 | 0/1 | 2 | PullCubeTool 100; Go2 200 | 600 / 604 |
| Five-tick multi-group check | 5 | 0/1/2/3/4 | 2 | 5 | 50 / 60 |

The three-episode prefix exercises two then one vector lanes. The supplemental
five-episode check executes groups of two, two, then one, verifying consecutive
same-width reuse before a fresh tail process. Both retain global episode indices
and seeds. Both priors have zero success-once and success-at-end in every
listed sample. Equality verifies this integration; these small samples do not
establish useful tool manipulation, Go2 task competence or 100-episode acceptance.
Eleven old official PPO profiles also passed a separate N1 five-tick regression.

R14/b53 remains failed for the step-2 rounding divergence above. R15/b54 passed
N1 full horizons but failed at the N2 width-two-to-one transition: SDK recovered a
Worker OOM and the native child failed. b55 recycles the opted-in Worker before
width changes and isolates native width segments in separate processes. Same-width
groups retain their process. This is a correctness fix with no speed gain claimed.

The original model CPU proofs executed with b54 and remain source-identified.
An exact seven-file source binding reuses them for unchanged b55 model code;
separate CPU fixtures cover Worker/native process lifetimes. Actual per-task R14
joint/controller-order probes carry through identical profiles, calibration helper
and source pin; they are not relabeled as fresh R16 measurements. Failed attempts
and the new passing cohorts are retained in the
[canonical-prior evidence release](https://github.com/QLL-CCDD/RoboScope-Kit/releases/tag/maniskill-issue25-canonical-priors-20261111).

## Reproduction

```bash
/path/to/runtime/env/bin/python examples/maniskill/download_models.py /path/to/models \
  --catalog examples/maniskill/reference-ppo-models.json --tasks PullCubeTool-v1 UnitreeGo2-Reach-v1
```

Use the existing `ppo_policy.py` configuration and `canonical_prior_state_v1`.
SDK/native cohort sizes, source versions and real-model results are recorded in the
[native review](../../benchmarks/maniskill/VALIDATION.md).
