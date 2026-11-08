# ManiSkill multi-task PPO example

This example serves official RL-trained PPO state actors through the SDK's ManiSkill adapter.
Task coverage is defined by the benchmark manifest; model coverage is supplied separately by a
checkpoint bundle. `baseline_state_v1` preserves the original three-task reference.
`official_ppo_state_v1` selects all eleven tasks with compatible official state checkpoints.
The task inventory records remaining policies and native comparisons needed for full acceptance.

## Models and routing

The eleven checkpoints total **12.78 MB**, from immutable revision
`3558f72cf99d51d04a6ac4779e88b3da4bbf1628` of
[`haosulab/ManiSkill_Demonstrations`](https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations/tree/3558f72cf99d51d04a6ac4779e88b3da4bbf1628).
URLs, sizes and SHA-256 digests are pinned in [official-models.json](official-models.json).

| Task | Robot | State dimension | Action dimension | Horizon |
| --- | --- | ---: | ---: | ---: |
| PickCube-v1 | panda | 42 | 4 | 50 |
| RollBall-v1 | panda | 44 | 4 | 80 |
| PushT-v1 | panda_stick | 31 | 3 | 100 |
| LiftPegUpright-v1 | panda | 32 | 7 | 50 |
| PokeCube-v1 | panda | 54 | 4 | 50 |
| PullCube-v1 | panda | 35 | 4 | 50 |
| PushCube-v1 | panda | 35 | 4 | 50 |
| StackCube-v1 | panda_wristcam | 48 | 4 | 50 |
| TwoRobotPickCube-v1 | two panda_wristcam | 66 | 16 | 100 |
| TwoRobotStackCube-v1 | two panda_wristcam | 76 | 16 | 100 |
| AnymalC-Reach-v1 | anymal_c | 35 | 12 | 200 |

The dual-arm and quadruped checkpoints use `pd_joint_delta_pos`, LiftPegUpright uses
`pd_ee_delta_pose`, and the other checkpoints use `pd_ee_delta_pos`. Evaluation uses deterministic
FP32 actor means. The loader reproduces the
three-layer, 256-unit Tanh actor/critic architecture in the pinned upstream PPO implementation and
strictly loads the complete checkpoint. Input/output dimensions are inferred from the weights.
Observations retain the native flattened `state` order and scale; actions pass unchanged to the
configured controller. No external normalization, clipping or precision change is used.
The task manifest owns the evaluation horizon and seed list.

```bash
/path/to/runtime/env/bin/python examples/maniskill/download_models.py /path/to/maniskill-models
```

The downloader verifies cached files, fetches only missing selected checkpoints and writes
`models.json`. Its default preserves the three baseline models. Use `--all` for all eleven or
`--tasks PickCube-v1 RollBall-v1` for a subset. Store models/assets outside Git in writable shared storage.

A bundle maps task IDs to relative checkpoint paths and expected hashes:

```json
{
  "schema_version": 1,
  "models": {
    "PickCube-v1": {
      "path": "PickCube-v1.pt",
      "sha256": "0b17b5ed9690ccf83111d0f09af8d1599f69ee7ba0077e1aac48814ac78ce99c",
      "robot_uid": "panda",
      "obs_mode": "state",
      "control_mode": "pd_ee_delta_pos",
      "max_episode_steps": 50
    }
  }
}
```

`Policy.load()` reads the bundle from `context.checkpoint_path` and loads each actor once.
`reset()` associates the episode ID with its task and checks the observation/controller configuration.
`act_batch()` groups independent episodes by task, runs each actor with its own input/action dimensions,
and restores request order. `end_episode()` removes episode routing state. A standalone official `.pt`
file is accepted when its pinned digest identifies its task. For another task suite, supply a bundle
covering its tasks with compatible PPO state checkpoints.

## SDK evaluation

Build the simulator through [the benchmark builder](../../benchmarks/maniskill/README.md).
The protocol is `maniskill-v1`; actions use `maniskill.native-controller-v1`, whose controller and
shape are provided in each task's episode metadata. The adapter supports state and RGB observation
paths. This PPO example consumes `state.proprioception`; RGB integration uses separate rendering and
transport smoke checks, without claiming these actors are trained image policies.

Fill [roboscope-policy.json](roboscope-policy.json) with absolute executable, script and working
directory paths from your installation. The contract contains portable placeholders. Use the runtime's
manifest so asset and PhysX locations resolve against that build directory.

Quick Test runs the first task in the selected suite for one episode:

```bash
/path/to/runtime/env/bin/python -m roboscope_kit.runtime.benchmark \
  --benchmark maniskill --suite baseline_state_v1 \
  --benchmark-manifest /path/to/runtime/benchmark-manifest.json \
  --checkpoint-source /path/to/maniskill-models \
  --policy-config /path/to/roboscope-policy.json \
  --quick-test --output-dir /path/to/new-quick-run
```

For a representative multi-task run, fix K/G and request ten episodes per task:

```bash
/path/to/runtime/env/bin/python -m roboscope_kit.runtime.benchmark \
  --benchmark maniskill --suite baseline_state_v1 \
  --benchmark-manifest /path/to/runtime/benchmark-manifest.json \
  --checkpoint-source /path/to/maniskill-models \
  --policy-config /path/to/roboscope-policy.json \
  --episodes 10 --concurrency 1 --replicas 1 \
  --output-dir /path/to/new-three-task-run
```

This runs 30 episodes and 2,300 control steps, exercising two robots, different state/action shapes
and task horizons. Omit `--episodes` for the declared 100 episodes per task. Use
`--suite official_ppo_state_v1` with an `--all` checkpoint bundle to exercise all eleven models.
Each record retains its profile, seed, initial-observation hash, first action, actual steps,
`success_once` and `success_at_end`. Broader suites require a compatible real policy for each task.
The published state actors cannot validate RGB policies, and RGB acceptance remains separate.
The separate [Octo-small example](OCTO.md) serves the four Bridge visual tasks.

## Native GPU-vectorized comparison

`native_eval.py` runs the same actors directly on ManiSkill GPU tensors with the selected manifest
and task suite. It constructs the environment independently through upstream Gym and wrappers,
using GPU physics, explicit seed groups and fixed horizons.
Multi-task commands launch one child process per task before CUDA initialization, retaining each
child's logs, result and SHA-256 before validating and merging the complete task matrix.

```bash
/path/to/runtime/env/bin/python examples/maniskill/native_eval.py \
  --benchmark-manifest /path/to/runtime/benchmark-manifest.json \
  --suite baseline_state_v1 --checkpoint /path/to/maniskill-models \
  --episodes 10 --num-envs 10 --output /path/to/native-three-task.json
```

Default environment lifetime matches the SDK baseline: create/close per vector group. Add
`--reuse-environment` for a persistent native comparison and label that difference. Keep the same N,
seed groups and task configuration when comparing initial-state hashes, first actions, outcomes and
actual steps. Reset inputs can change with N even when scalar seeds match.

Verify the complete matching episode matrix with the simulator-free comparator. Pass the same
`--episodes` override when comparing reduced diagnostic runs.

```bash
python examples/maniskill/compare_native.py --sdk-output /path/to/sdk-run \
  --native-result /path/to/native-result.json \
  --benchmark-manifest /path/to/runtime/benchmark-manifest.json \
  --suite official_ppo_state_v1 --output /path/to/comparison.json
```

It rejects missing, duplicate and unexpected episodes, wrong task/horizon mappings, different
initial states, first actions, success metrics and episode returns. Numeric comparisons are exact
by default; any explicit tolerances are retained in the report. A disagreement exits nonzero.

For separate correctness diagnostics, `--trajectory-dir /path/to/new-native-traces` saves each
consumed observation and action digest with its step number. The SDK worker supports matching
traces through `ROBOSCOPE_MANISKILL_TRAJECTORY_DIR`. Compare these diagnostic runs separately
from throughput measurements because copying and hashing each step adds overhead.
Pass both `--sdk-trajectories` and `--native-trajectories` to the comparator to locate the first
divergent consumed observation or action in each mismatched episode.

Output separates loading, imports/setup, creation, resets, rollout, closing and bookkeeping.
Single-task native formal timers include imports/setup and simulator lifecycle, excluding loading
and result writing. Multi-task output labels summed sequential child phases and reports the parent
command's complete wall time, including child startup and repeated model loads. Preserve external
command wall time; SDK and native timer boundaries differ. `inference_host_submission_s` measures
asynchronous CPU submission, not GPU execution latency. Per-task primary scores follow
`success_metric` in the manifest; both success
rates are retained explicitly.

## Original upstream entrypoint

The separate launcher runs the unmodified pinned `examples/baselines/ppo/ppo.py`. It verifies the
source blob, checkpoint and PhysX hashes and disables training, recording and model saving.
Select a checkpoint from the bundle with `--task-id`:

```bash
/path/to/runtime/env/bin/python examples/maniskill/run_upstream.py \
  --runtime /path/to/runtime --checkpoint /path/to/maniskill-models \
  --output-dir /path/to/new-upstream-smoke --task-id RollBall-v1 --num-eval-envs 10 --seed 0
```

`command.json`, `upstream.log` and `result.json` retain the exact invocation, evaluated steps/episodes,
metrics and complete child-process time. Upstream retains its scalar-seed and subsequent unseeded
reset schedule; this checks the genuine GPU-vectorized evaluator separately from the protocol-aligned
native comparison. Runtime pins, assets and acceptance results are in the
[benchmark documentation](../../benchmarks/maniskill/README.md).
