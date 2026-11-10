# ANYmal Reach-to-Spin orbit transfer

This policy applies the existing official `AnymalC-Reach-v1` PPO checkpoint to
`AnymalC-Spin-v1` through an explicit internal waypoint planner. It is a zero-shot
transfer policy, not an expert trained on Spin. The original checkpoint,
architecture and deterministic actor-mean inference are unchanged. Its source
revision, filename, size and SHA-256 are already pinned in
[`official-models.json`](official-models.json).

The native task definitions share the same robot and `pd_joint_delta_pos`
controller. Spin's 30-value state is exactly Reach's physical prefix: joint
positions/velocities followed by world root linear/angular velocities. The
Reach actor additionally consumes a reached-goal flag, world goal xy and
robot-to-goal xy. Here those five values are computed for an actual internal
planner waypoint, rather than padded or presented as a native task goal.

Use the explicitly named Spin root-state profile. Its original
`state.proprioception` vector remains unchanged; the optional `state.root_pose`
feature supplies xyz+wxyz root pose. On the first observation, the planner fixes
an orbit center 0.75 m to the robot's initial body-left. Each subsequent waypoint
lies 90 degrees counterclockwise ahead of the robot's current radial direction
on that circle. The planner computes world goal coordinates, relative goal
coordinates and the 0.35 m reached-goal flag from the observed position.

Waypoint distance is at least 0.75 m, so the computed reached flag remains false.
Each episode owns and resets its center; no simulator state is changed. Native
Spin rewards positive world yaw velocity and has no horizontal-motion penalty,
making a counterclockwise orbit a meaningful transfer strategy. Physical
rollout results must be reported as transfer results without a Spin-expert
performance claim.

Download the existing public model with:

```bash
PYTHON=/path/to/maniskill-runtime/env/bin/python
"$PYTHON" examples/maniskill/download_models.py /autodl-fs/models/maniskill \
  --tasks AnymalC-Reach-v1
```

Use `anymal_spin_policy.py` as the SDK policy command and pass the generated
`AnymalC-Reach-v1.pt` as the checkpoint. The native benchmark retains its task
rewards, physics, resets, control frequency, full horizon and failure checks.

CPU source-contract checks covered two interleaved episodes of ten observations.
The 30 physical input values were preserved exactly; computed waypoint inputs
differed from an independent orbit reference by at most 2.38e-7, and actor actions
differed from the unmodified official PPO by at most 7.15e-7. Reset cleared only
the selected episode's center, and cleanup released all planner state. Physical
rollout acceptance is recorded separately in the benchmark report.
