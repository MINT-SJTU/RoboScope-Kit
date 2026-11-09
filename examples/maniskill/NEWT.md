# NEWT state policy priors

This example loads real, task-specific [NEWT checkpoints](https://huggingface.co/nicklashansen/newt/tree/7eef11eb63c8ed53d61d739693d7140135ea0876).
It executes the official deterministic policy-prior path (`mpc=False`,
`eval_mode=True`), rather than model predictive control. Results from this example
must not be presented as NEWT's published MPC performance.

The source is pinned to
[`1d3fc058b81ddf8d36a5457c29ea407dd374c1b8`](https://github.com/nicklashansen/newt/tree/1d3fc058b81ddf8d36a5457c29ea407dd374c1b8).
The upstream README and model card declare the MIT license. Checkpoint filenames,
sizes, SHA-256 digests and native task contracts are in
[`newt-checkpoints.json`](newt-checkpoints.json). Weights stay outside Git.

| Native task | State dimension | Controller action dimension |
| --- | ---: | ---: |
| MS-AntWalk-v1 / MS-AntRun-v1 | 114 | 8 |
| MS-CartpoleBalance-v1 / MS-CartpoleSwingUp-v1 | 10 | 1 |
| MS-HopperStand-v1 / MS-HopperHop-v1 | 15 | 4 |
| PlaceSphere-v1 | 39 | 4 |
| PickSingleYCB-v1 | 45 | 7 |
| PullCubeTool-v1 (named pose-controller profile) | 39 | 7 |

These are native ManiSkill policies, including the six `ms-*` control models.
They are not the similarly named DeepMind Control Suite torque policies.
The control tasks use native `pd_joint_delta_pos`; PlaceSphere uses
`pd_ee_delta_pos`; YCB and PullCubeTool use `pd_ee_delta_pose`. PullCubeTool's
existing position-controller profile is unchanged; select the explicit new pose
profile for this model. No observation normalization is added.

The YCB model was trained on `013_apple`. It is an object-transfer reference when
evaluated on the canonical complete YCB object distribution. The adapter does
not select objects or change task sampling, and no all-object expert performance
is claimed.

The upstream [ManiSkill wrapper](https://github.com/nicklashansen/newt/blob/1d3fc058b81ddf8d36a5457c29ea407dd374c1b8/tdmpc2/envs/maniskill.py)
consumes the native flattened state and repeats each model action twice.
Its [vector wrapper](https://github.com/nicklashansen/newt/blob/1d3fc058b81ddf8d36a5457c29ea407dd374c1b8/tdmpc2/envs/wrappers/vectorized_multitask.py)
zero-pads that state to 128 values and pads actions to 16 values. This example
reproduces that trained architecture contract and validates the checkpoint's
native action mask. All 143 world-model tensors are loaded strictly, including
the ten identical masks retained by the released single-task checkpoints.
Inference runs only the encoder and actor, using the masked `tanh` actor mean.

Each episode maintains its own two-tick action hold. `reset()` clears the hold;
the simulator still computes observations, rewards, termination and physics on
every native tick. Benchmark control frequencies and complete episode horizons
are unchanged. In particular, canonical Hopper evaluation lasts 600 native
ticks although NEWT's training wrapper used 500.

NEWT used `mani_skill-nightly==2025.9.19.39`. Its Ant, Cartpole and Hopper task
observations and controller definitions match the benchmark's pinned ManiSkill
revision; Ant's state-mode predicate changed, with identical state values and
ordering. PlaceSphere's task observation method is unchanged. Training used CPU
simulation; the benchmark retains its specified GPU backend, so no assertion is
made that trajectories match the authors' training simulator.

Use the existing ManiSkill runtime built by `benchmarks/maniskill/build-runtime.sh`.
The small inference adapter needs only its existing Torch 2.6.0 and NumPy 1.26.4
dependencies; no upstream training packages or separate environment are needed.

```bash
PYTHON=/path/to/maniskill-runtime/env/bin/python
"$PYTHON" examples/maniskill/download_newt.py /autodl-fs/models/maniskill-newt
cat > /workspace/newt-policy.json <<EOF
{
  "schema_version": 1,
  "command": ["$PYTHON", "/path/to/RoboScope-Kit/examples/maniskill/newt_policy.py"],
  "cwd": "/path/to/RoboScope-Kit"
}
EOF
```

Pass the generated `models.json` bundle as the checkpoint path and select a
matching numeric state profile. Each model is approximately 72.3 MB; the nine
pinned files total approximately 650 MB. The downloader verifies cached files
and reports progress and estimated remaining time.

The nine model adapters were compared against the pinned upstream
`WorldModel.encode()` / `WorldModel.pi()` implementation on 32 states per model
using Torch 2.6.0 on CPU: all action means were bitwise equal (maximum absolute
error 0). This checks inference equivalence; task rollout acceptance is recorded
in the benchmark's acceptance report.
