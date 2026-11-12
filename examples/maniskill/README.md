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

## Additional PPO state references

`reference_ppo_state_v1` uses four existing canonical profiles with the unchanged
three-layer Tanh actor/critic and all 17 original FP32 tensors. Immutable URLs,
dimensions, hashes and source identities are in [reference-ppo-models.json](reference-ppo-models.json).
It selects the first four rows below; `canonical_prior_state_v1` selects the last two.

| Target | Native / actor state | Actions | Reference |
| --- | ---: | ---: | --- |
| G1 TransportBox | 77 / 77 | 25 joint deltas | Author task-trained PPO |
| G1 Apple | 74 / 74 | 25 joint deltas | Author nominal PPO expert, retaining source clipping |
| PegInsertionSide | 43 / 32 | 7 EE pose deltas | Explicit LiftPeg grasp/lift-prior transfer |
| PlugCharger | 46 / 32 | 7 EE pose deltas | Explicit LiftPeg grasp/lift-prior transfer |
| PullCubeTool | 39 / 35 | 4 EE position deltas | Explicit direct-pull prior using the authored static workspace target |
| Go2 Reach | 35 / 35 | 12 joint deltas | Explicit ANYmal cross-robot joint prior |

The [TransportBox author checkpoint](https://github.com/tjvitchutripop/stream-rl-robotics/tree/30b2d8ecca093d9b9b492598aa9eeba70b628fd5/pretrained-models/transport-box)
uses raw state and actor means. The [Apple author reference](https://github.com/tabz23/Latent-Safety-Filters/tree/ea456084576b16053e6de79c52b489e33409479e)
clips its deterministic actions to the normalized controller bounds. Only its
catalog record sets `action_clip_bounds=[-1,1]`; the SDK Policy and direct native
evaluator preserve that operation. Other actors in the four-task reference suite retain their
original raw actions; the canonical priors have their separately documented conversions.
Independent CPU checks against the released author actors/critics, including Apple clipping,
are bitwise equal. Historical configured-policy checks cover these profiles and the
two [dual-Panda transfers](DUAL_PANDA_PPO.md). The [current validation status](../../benchmarks/maniskill/VALIDATION.md)
retains their exact source, grouping and limits. They do not establish intended-task competence,
independent upstream configuration or 100-episode acceptance.

Peg/Plug expose the same first 32 fields as LiftPeg: qpos/qvel18, TCP pose7 and
object pose7. Their records explicitly select this prefix while traces retain
the full target observations. Both quaternions and all seven action components
are retained; no hidden object/goal query, synthetic field or orientation removal
is used. Panda/WristCam share joint and controller frames; contact geometry and
task objectives differ. These are grasp/lift priors, not insertion experts.

```bash
/path/to/runtime/env/bin/python examples/maniskill/download_models.py \
  /autodl-fs/maniskill-reference-models --catalog examples/maniskill/reference-ppo-models.json
```

Use the same `ppo_policy.py` configuration and the generic configured-policy diagnostic:

```bash
/path/to/runtime/env/bin/python examples/maniskill/native_eval_policy.py \
  --benchmark-manifest /path/to/runtime/benchmark-manifest.json \
  --suite reference_ppo_state_v1 --checkpoint /path/to/maniskill-reference-models \
  --policy-module /absolute/path/to/kit/examples/maniskill/ppo_policy.py \
  --policy-class ManiSkillPPOPolicy --policy-python /path/to/runtime/env/bin/python \
  --episodes 100 --num-envs 10 --output /path/to/reference-policy-native.json
```

This diagnostic takes its task configuration from the manifest and reuses the same Policy.
It is not the independent upstream configuration comparison used for official PPO acceptance.
The manifest retains every target's original horizon, controller, observations and scoring. TransportBox's source
evaluator defaults to 200 ticks on CPU and its lock uses ManiSkill 3.0.1; Apple's
source simulator revision is not pinned. These references transfer checkpoints
to the pinned Kit runtime, without claiming the papers' evaluation protocols.
TransportBox specifies no source license; Apple code is MIT and its model card
specifies no license. Weights remain external to Git and evidence bundles.

The additional [canonical priors](CANONICAL_PRIORS.md) use `canonical_prior_state_v1`,
with unchanged target profiles. Their earlier smaller-width full-horizon and prefix
checks remain historical evidence rather than the current frozen-width protocol.
Their source models and transfer semantics differ from the four-task
`reference_ppo_state_v1` suite. Both new priors had zero successes in these samples;
they do not establish tool-use or Go2 task competence.

## SDK evaluation

Build the simulator through [the benchmark builder](../../benchmarks/maniskill/README.md).
The protocol is `maniskill-v1`; actions use `maniskill.native-controller-v1`, whose controller and
shape are provided in each task's episode metadata. The adapter supports state and RGB observation
paths. This PPO example consumes `state.proprioception`; separate visual Policies supply
RGB references. Historical visual interface checks and remaining model/asset gates are
listed in the [validation status](../../benchmarks/maniskill/VALIDATION.md).

Fill [roboscope-policy.json](roboscope-policy.json) with absolute executable, script and working
directory paths from your installation. The contract contains portable placeholders. Use the runtime's
manifest so asset and PhysX locations resolve against that build directory.

Quick Test runs the first complete ten-lane vector group of the first selected profile.
It uses the same initial-state grouping as the corresponding full run:

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

`native_eval.py` is restricted to the eleven official PPO task checkpoints. It resolves task
configuration from the pinned upstream source independently of the SDK manifest and rejects
configuration disagreements. It runs the same actors directly on ManiSkill GPU tensors through
upstream Gym and wrappers, with GPU physics, matching seed groups and fixed horizons.
Multi-task commands launch one child process per task before CUDA initialization, retaining each
child's logs, result and SHA-256 before validating and merging the complete task matrix.

```bash
/path/to/runtime/env/bin/python examples/maniskill/native_eval.py \
  --benchmark-manifest /path/to/runtime/benchmark-manifest.json \
  --suite baseline_state_v1 --checkpoint /path/to/maniskill-models \
  --episodes 10 --output /path/to/native-three-task.json
```

Environment lifetime matches the SDK baseline: create/close per vector group. The native width
defaults to the frozen protocol width; an explicit `--num-envs` must match it. Retain complete
groups when reducing episode counts. Reset inputs can change with width even when scalar seeds match.

Verify the complete matching episode matrix with the simulator-free comparator. Pass the same
`--episodes` override when comparing reduced diagnostic runs.

```bash
/path/to/runtime/env/bin/python examples/maniskill/compare_native.py \
  --sdk-output /path/to/new-three-task-run --native-result /path/to/native-three-task.json \
  --benchmark-manifest /path/to/runtime/benchmark-manifest.json \
  --suite baseline_state_v1 --episodes 10 --output /path/to/comparison.json
```

It rejects missing, duplicate and unexpected episodes, wrong task/horizon mappings, different
initial states, first actions, success metrics and episode returns. Numeric comparisons are exact
by default; any explicit tolerances are retained in the report. A disagreement exits nonzero.

Terminal masks are checked on both sides for full-horizon and prefix runs, with strict boolean
types. Use new native output containing the actual returned masks. Older PPO records lacking mask
fields are rejected; do not infer those values from the requested horizon or rewrite old evidence.

For separate correctness diagnostics, `--trajectory-dir /path/to/new-native-traces` saves each
consumed observation and action digest with its step number. The SDK worker supports matching
traces through `ROBOSCOPE_MANISKILL_TRAJECTORY_DIR`. This variable enables correctness diagnostics
only and must be unset during throughput measurements. Each attempt writes exclusive
`*.jsonl.partial` files; failed attempts remain available without blocking retries. Only complete
groups whose Policy/environment cleanup succeeds publish canonical episode `*.jsonl` files.
The comparator discovers those canonical files and ignores partial attempts. Copying and hashing
each step adds diagnostic overhead.
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
reset schedule. This smoke invocation alone does not establish equal initial inputs or acceptance;
the configuration-independent comparison also needs a matching real current-source run.
Runtime pins, assets and acceptance results are in the
[benchmark documentation](../../benchmarks/maniskill/README.md).

## Additional real-policy references

The [LiftCube actor prior](LIFTCUBE.md) targets the existing canonical LiftPeg4
profile through `liftcube_prior_state_v1`. It retains the authors' complete
42D state input and active learned gripper, with explicit actor-mean mode,
EE-to-root conversion and two-tick action holds. Its historical smaller-width checks are indexed
in the validation record; this borrowed cube policy is not an intended upright-peg expert.

Task-specific [NEWT priors](NEWT.md) cover six control tasks, PlaceSphere, PickSingleYCB and a
separate PullCubeTool pose profile. Explicit transfer references are available for
[HORA Allegro](hora_policy.md), [TriFinger](trifinger_policy.md), [ROBEL DClaw](robel_policy.md),
[Unitree H1/G1](UNITREE.md), [Go2](GO2.md) and [ANYmal Spin](ANYMAL_SPIN.md). Their source-trained
robot/controller contracts and cross-simulator differences are documented separately; executing
these models does not reproduce their original expert success rates.

Further references cover [Humanoid control](HUMANOID.md), the
[ManiSkill-finetuned Octo visual policy](PANDA_OCTO.md), [G1 visual transfer](G1_VISUAL.md) and
[MS-HAB Fetch depth transfer](FETCH_MSHAB.md). Each records its exact source model/input/control
contract and separates interface validation from task competence and original-asset blockers.

The generic `native_eval_policy.py` constructs the native Gym environment, features
and action execution without the Kit Worker or RPC. It takes configuration from the same manifest
and reuses the same Policy, so agreement isolates SDK transport/integration; it does not independently
validate task configuration, model inference or transfer mappings. Author-code and numerical checks provide separate, explicitly scoped
evidence for those components. Multi-profile references run sequential per-task child processes,
as the PPO reference does, to isolate native simulator lifetimes. Child logs and record hashes are
retained; child phase sums and parent command time have separate boundaries. Specify
`--policy-module /absolute/path/policy.py`, `--policy-class`,
`--policy-python`, `--checkpoint`, `--benchmark-manifest`, `--suite` and `--output`. Add matched
`--episodes`, `--num-envs` and `--trajectory-dir` for the native workload, then pass both trace
directories to `compare_native.py`. A diagnostic manifest can set
`benchmark_protocols.maniskill.diagnostic_step_limit=5`; retain its original environment horizon.
All such prefixes are non-formal and keep the frozen vector width. The [validation record](../../benchmarks/maniskill/VALIDATION.md)
retains source identities, measured scope, failures and unresolved gates. Historical paired execution
of 75 profiles / 51 TaskIDs is not accepted task coverage; current-source prescribed-count evaluation
with intended task/robot checkpoints and nonzero competence remains required.
