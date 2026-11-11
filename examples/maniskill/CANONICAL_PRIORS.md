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

The [shared native observation builder](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/quadruped/quadruped_reach.py)
and frozen [ANYmal](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/agents/robots/anymal/anymal_c.py)
and [Go2 controllers](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/agents/robots/unitree_go/unitree_go2.py)
define these contracts. Limb geometry, joint limits, PD gains and contact dynamics
differ. Home alignment is not an exact kinematic mapping or a claim of learned
Go2 competence. Source and target retain their original 200-tick horizons.

The b53 candidates require fresh-build and real-model verification; they are not
included in the passing R12/R13 coverage until those gates run.

## Reproduction

```bash
/path/to/runtime/env/bin/python examples/maniskill/download_models.py /path/to/models \
  --catalog examples/maniskill/reference-ppo-models.json --tasks PullCubeTool-v1 UnitreeGo2-Reach-v1
```

Use the existing `ppo_policy.py` configuration and `canonical_prior_state_v1`.
SDK/native cohort sizes, source versions and real-model results are recorded in the
[native review](../../benchmarks/maniskill/NATIVE_REVIEW_VALIDATION.md).
