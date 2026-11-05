# robomimic Evaluation Protocol

This document defines the SDK-native RoboScope robomimic evaluation contract. The kit's benchmark runtime
owns benchmark enumeration, simulator processes, concurrency, recovery, and result aggregation. User
inference code owns only the public `Policy` lifecycle and model-specific preprocessing and
postprocessing.

`roboscope-kit` requires Python 3.11 or newer in every process that imports it, including the Policy and
the simulator workers. Python 3.8–3.10 are unsupported. See the
[Python support policy](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/packages/roboscope-kit/README.md#python-support).

Status: code candidate, build-verified on Linux x86_64 (clean build, `--self-test`, Quick Test, full
evaluation, and native comparison in section 9). Not merged or released until the version that contains it
(`0.8.0b26` or later) is on PyPI.

## 1. Benchmark scope

The `benchmark` identifier is `robomimic`, the protocol version is `robomimic-v1`, and the only suite is
`ph_low_dim`: the five robomimic v0.1 (CoRL 2021) simulation tasks with low-dimensional observations, the
setting of the model zoo's proficient-human (PH) low-dim BC-RNN checkpoints.

| Task | robosuite environment | Robots | Horizon (steps) | `object` size | Action size |
| --- | --- | --- | --- | --- | --- |
| `lift` | `Lift` | one Panda | 400 | 10 | 7 |
| `can` | `PickPlaceCan` | one Panda | 400 | 14 | 7 |
| `square` | `NutAssemblySquare` | one Panda | 400 | 14 | 7 |
| `transport` | `TwoArmTransport`, `single-arm-opposed` | two Panda | 700 | 41 | 14 |
| `tool_hang` | `ToolHang` | one Panda | 700 | 44 | 7 |

Every task uses the robomimic v0.1 environment arguments stored in the checkpoints (the same `env_args` as
the PH low-dim datasets): the `OSC_POSE` controller with delta control (`kp=150`, `damping=1`, output
limits 0.05 m and 0.5 rad), `control_freq=20`, `ignore_done=True`, object observations on, no cameras, no
reward shaping. The horizon is each checkpoint's `experiment.rollout.horizon`. These values are frozen in
the kit's
[task table](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/packages/roboscope-kit/src/roboscope_kit/runtime/benchmarks/robomimic/tasks.py);
the worker builds each environment from that table and never reads a model checkpoint.

A formal `ph_low_dim` evaluation runs every task with 50 episodes, 250 in total. Quick Test runs Lift with
seed 0 only. `--episodes N` (1 to 50) runs seeds `0..N-1` of every task and is reported as non-formal.

### What is not supported, and why

| Combination | Status | Reason |
| --- | --- | --- |
| PH, low-dim, BC-RNN: Lift, Can, Square, Transport, Tool Hang | Supported (`ph_low_dim`) | Released model-zoo checkpoints; reproduced here (section 9) |
| PH, image, BC-RNN: the same five tasks | Not supported | Checkpoints exist, but they were trained on mujoco-py offscreen renders of the `offline_study` robosuite; the MuJoCo 2.3.7 renderer here is a different image pipeline whose effect on the 84x84 policies has not been verified. It needs its own suite once a reproduction shows its success rates hold |
| MH (multi-human) and MG (machine-generated) datasets, any observation | Not supported | robomimic publishes no MH or MG checkpoints, so there is no real model to verify an evaluation with |
| Other algorithms of the study (BC, BCQ, CQL, HBC, IRIS) | Not supported | Not in the model zoo |
| Real-robot tasks (Lift, Can, Tool Hang real) | Out of scope | Not simulation |
| Training, offline validation loss | Out of scope | This protocol measures closed-loop rollout success only |

Policies other than the model zoo may be evaluated on `ph_low_dim`, but a score is comparable to the
robomimic study only for the same task setting and the same aggregation (section 7).

## 2. Environment, versions, and license gates

The robomimic simulator is not a dependency of `roboscope-kit`. Build it with
[`benchmarks/robomimic/build-runtime.sh`](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/robomimic/build-runtime.sh) `<install-dir>`
(Linux x86_64, uv 0.11.14, git, network access). The script installs a uv-managed CPython 3.11 under
`<install-dir>/python` and a venv under `<install-dir>/env` with exactly
[`runtime-requirements.txt`](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/robomimic/runtime-requirements.txt),
all from PyPI and GitHub:

| Component | Pin |
| --- | --- |
| robomimic | v0.4.0, commit `7c66e7a41b5d9dcc905b1a68346bfee1b49b79c9` |
| robosuite | head of the `v1.4.1` branch, commit `c848ca848020d0c4ccdd10c5056bd06f2a195ba2` (reports 1.4.1) |
| MuJoCo | `mujoco==2.3.7` (DeepMind bindings) |
| torch | `torch==2.4.1` (PyPI CUDA 12.1 wheels), `numpy==1.26.4`, `numba==0.68.1` |
| Python | CPython 3.11 (MuJoCo 2.3.7 has no 3.12 wheel) |

The script checks the installed versions and the commits uv recorded, checks that the upstream entry point
`robomimic.scripts.run_trained_agent` is installed, regenerates the 50 seeded initial states of every task
(section 5) and compares their digests with the task table, records `pip-freeze.txt`, and copies
[`benchmark-manifest.json`](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/robomimic/benchmark-manifest.json)
into `<install-dir>`. The manifest's `runtime.python` is `env/bin/python`, relative to the manifest file.
Workers import the kit from its source tree through `PYTHONPATH`, so `roboscope-kit` is not installed into
this environment. `--self-test` regenerates the Lift initial states through the worker's code and checks
their digest.

**Why not the original stack.** The robomimic v0.1 model zoo was trained with robomimic v0.1 on the
`offline_study` branch of robosuite (1.2), which depends on `mujoco-py==2.0.2.x` and the MuJoCo 2.0 binary
with an activation key. mujoco-py 2.0 is a Cython extension built against MuJoCo 2.0 headers and system
OpenGL/OSMesa development headers, robomimic v0.1 and v0.3 import `mujoco_py` unconditionally, and none of
it is maintained for Python 3.11. This protocol therefore uses robomimic v0.4.0, the first release whose
robosuite wrapper runs without mujoco-py, with the robosuite v1.4.1 branch that robomimic documents for its
regenerated datasets. robomimic v0.4.0 loads v0.1 checkpoints through its own backward-compatibility path
(`FileUtils.update_config`). The physics (MuJoCo 2.0 to 2.3.7, robosuite 1.2 to 1.4.1) is not the
`offline_study` physics, so success rates are not assumed equal to the published ones: section 9 reports
what this stack reproduces. robosuite v1.5 is not used, because it replaces the controller stack the
checkpoints were trained with.

**Licenses.** robomimic and robosuite are MIT-licensed, MuJoCo is Apache-2.0, torch is BSD-3-Clause. The
build accepts no license on anyone's behalf and downloads no model or dataset. The model-zoo checkpoints are
published on `downloads.cs.stanford.edu` without a license of their own; users download them themselves
(`examples/robomimic/download_model_zoo.py`, which checks the pinned SHA-256 of each file) and are
responsible for their terms. No dataset is needed: the evaluation samples initial states from the
environment and does not replay demonstrations.

Run an evaluation with the kit installed in any supported Python:

```bash
python -m roboscope_kit.runtime.benchmark \
    --benchmark robomimic --suite ph_low_dim --checkpoint-source /models/robomimic-zoo \
    --benchmark-manifest <install-dir>/benchmark-manifest.json \
    --policy-config /path/to/roboscope-policy.json \
    --output-dir /runs/robomimic-ph-low-dim
```

Use a new or empty output directory for each evaluation or self-test. A directory containing only
`errors.jsonl` can be retried, preserving earlier errors; other run artifacts require a new directory.

## 3. User policy lifecycle

The user entrypoint implements `Policy` and calls `run(policy)`. The runtime starts one process per Policy
replica with the command from `--policy-config` and calls `load(SessionContext)` once per replica,
`reset(EpisodeContext)` before every episode, `act(Observation)` whenever the worker needs a new action
chunk, `end_episode(episode_id)` when the episode ends or the worker stops, and `unload()` at the end. One
replica serves many workers concurrently, so a Policy with recurrent or sampling state keys it by
`episode_id`.

`EpisodeContext.task_id` is the task name (`lift`, ..., `tool_hang`). `EpisodeContext.metadata` carries
`protocol`, `seed`, `env_name` (the robosuite environment), and `horizon`. A Policy that samples actions
may seed its sampler from `seed`; the example does, which makes each episode reproducible (section 9).
`PolicyCapabilities` must declare `robomimic` and `ph_low_dim`. The
[example Policy](https://github.com/MINT-SJTU/RoboScope-Kit/tree/main/examples/robomimic) serves the five
model-zoo checkpoints and keeps observation normalization, BC-RNN hidden state, and GMM sampling on the
model side.

## 4. Observation and action contract

Every observation carries the robomimic low-dim keys as contiguous `float32` vectors, converted from
robomimic's `EnvRobosuite.get_observation()` (`float64`) without any other change:

| Feature | Shape | Source |
| --- | --- | --- |
| `state.robot0_eef_pos` | `(3,)` | `robot0_eef_pos` |
| `state.robot0_eef_quat` | `(4,)` | `robot0_eef_quat` (x, y, z, w) |
| `state.robot0_gripper_qpos` | `(2,)` | `robot0_gripper_qpos` |
| `state.robot1_eef_pos`, `state.robot1_eef_quat`, `state.robot1_gripper_qpos` | `(3,)`, `(4,)`, `(2,)` | Transport only; optional features |
| `state.object` | `(object size,)` | robomimic `object` (robosuite `object-state`); size per task in section 1 |

`step` counts environment steps from 0. There are no images and no language instruction.

An action is the robosuite `OSC_POSE` delta command per arm, `[dx, dy, dz, dax, day, daz, gripper]` in
`[-1, 1]` (scaled by the controller to 0.05 m and 0.5 rad), concatenated `robot0` then `robot1` for
Transport (`robomimic.osc-pose-delta-v1`, `float32`, 7 or 14 values). The worker passes each action to
`env.step` unchanged; the controller clips its inputs, as it does upstream. Action chunks are executed in
order and the remainder is discarded when the episode ends; the model-zoo Policy returns one action per
call, as upstream calls the policy every step.

## 5. Seeds and initial states

Episode `i` of every task uses seed `i` (`i = 0..49`). Before the episode the worker calls
`np.random.seed(i)`, `env.reset()`, `state = env.get_state()`, and `env.reset_to(state)`, which is exactly
what upstream `run_trained_agent.py --seed i --n_rollouts 1` does before its first step (robosuite's
placement samplers and robot initialization noise draw from NumPy's global RNG). The worker reuses one
environment per task across episodes; with robosuite's hard resets, every reset rebuilds the model, so the
sampled state depends only on the seed.

The digest of one initial state is the SHA-256 of its flattened MuJoCo state rounded to 9 decimals
(absorbing last-bit `libm` differences between CPUs); a task's digest is the SHA-256 over
`"<seed>:<state digest>\n"` for seeds 0 to 49. Each record carries its `initial_state_sha256`. The build
regenerates all 250 states and rejects an environment whose task digests differ from the frozen table; a
formal aggregate recomputes each task's digest from its records and fails on a mismatch. The frozen task
digests are:

| Task | Initial-state digest (seeds 0–49) |
| --- | --- |
| `lift` | `33ab18606642725c9c43b52bf753b34d2518bdec0e85f5aceaa7e176f7ebae27` |
| `can` | `645962b9270d0ac982f2422e85e1dcf5431f028dcbc769b7c4621903d30a2e33` |
| `square` | `7bb5a5ae88197ef482c3e7188ed773fc2f5802d8d31b98223657c0f31575f99a` |
| `transport` | `0a657d8dbe3fc132cdcdaf691b87c295d3b7fdcf7dff90dee7ccd83fbe0a5963` |
| `tool_hang` | `5e6d17058a2803bcc1acce745aebb84c2db71d5e13adf9557ebc981a7e628576` |

Seeds `0..49` are a RoboScope reproduction parameter. robomimic's own training-time rollouts (50 episodes
every 50 epochs) and its `run_trained_agent.py` default (27 episodes, no seed) use unseeded or single-seed
streams, so there is no upstream list of initial states to adopt.

## 6. Rollout and success

The worker reproduces upstream `rollout()` of `run_trained_agent.py`: for up to `horizon` steps it gets an
action, calls `env.step(action)`, then `success = env.is_success()["task"]`, and stops when `done or
success`. robosuite environments never report `done` here (`ignore_done=True`), so an episode ends on the
first step whose task success check passes, or at the horizon. The record's `success` is that last
`success` and its `steps` is the number of executed environment steps (upstream `Horizon`). A record that
fails before its horizon is rejected. With robosuite on the DeepMind bindings, upstream catches no
simulation exception (`EnvRobosuite.rollout_exceptions` is empty), so a simulator error fails the worker
instead of becoming a failed episode; the runtime records the failure and requeues the work item on a
restarted worker, up to its restart limit, so a crash is never counted as a failed episode.

## 7. Aggregation

`official_result` has `metric = "per_task_success_rate"`, `formal`, and one row per task with `task`,
`env_name`, `episodes`, `successes`, and `success_rate`, in the task order of section 1. robomimic reports
success rates per task; there is no official cross-task score, and this protocol does not define one. The
standard `result` adds the total episodes, successes, executed steps, episodes/s, and steps/s.

A record is identified by `<task>-seed-<seed:04d>`. The aggregate rejects records outside the work
manifest, duplicates, out-of-range steps, and early failures. A formal result requires all 250 records
and verified initial-state digests (`initial_states_verified: true`); Quick Test and `--episodes` runs are
not formal. Sharded runs sum the per-task rows.

## 8. Validation sequence

1. Build the runtime with `build-runtime.sh` in a fresh directory; it verifies versions and all initial-state
   digests.
2. `--self-test --benchmark-manifest <install-dir>/benchmark-manifest.json`.
3. `--quick-test` with the real Policy (Lift, seed 0).
4. The full `ph_low_dim` evaluation.
5. `examples/robomimic/compare_native.py` against upstream `run_trained_agent.py` with the same checkpoints,
   seeds, and horizons.

## 9. Correctness baseline

Measured on 2026-11-05 with `roboscope-kit` 0.8.0b26 (this change), on a shared Linux host: 2 x Intel Xeon
Silver 4214R (48 logical CPUs, other jobs keeping the load average at 10–50), 188 GB RAM, one NVIDIA RTX 3080
10 GB (driver 560.35.03, CUDA 12.6; about 3.3 GB of it used by other processes). Checkpoints: the five
model-zoo files of section 2 with their pinned SHA-256. Policy: `examples/robomimic` run with the runtime's
Python. Workers do not use the GPU.

### Build, self-test, and Quick Test

- `build-runtime.sh` into two fresh directories: both builds verified the versions, commits, and all five
  initial-state digests, ran `--self-test`, and produced identical `pip-freeze.txt` files. The second build
  (uv cache warm, so package downloads excluded) took 266 s: 32 s to install CPython, 3 s to install the
  73 locked packages, about 3 minutes to regenerate the 250 initial states.
- `--quick-test`: Lift seed 0 succeeded in 59 steps; upstream `run_trained_agent.py --seed 0 --n_rollouts 1`
  also succeeded in 59 steps. Policy load 4.1 s, runtime preparation 1.8 s, evaluation 16.8 s, 24 s end to
  end.

### Full evaluation

`--concurrency 16 --replicas 1` (fixed K/g, so no autotune phase), all 250 records present exactly once,
no worker failures, restarts, or OOM events, and `initial_states_verified: true`:

| Task | Success | Executed steps | Mean steps | Mean episode time (s) | Model-zoo name (training-time rollout) |
| --- | --- | --- | --- | --- | --- |
| `lift` | 50/50 (100%) | 2,122 | 42.4 | 1.5 | `succ_100` |
| `can` | 50/50 (100%) | 5,553 | 111.1 | 4.4 | `succ_100` |
| `square` | 39/50 (78%) | 10,454 | 209.1 | 8.9 | `succ_84` |
| `transport` | 27/50 (54%) | 29,014 | 580.3 | 56.0 | `succ_78` |
| `tool_hang` | 7/50 (14%) | 33,624 | 672.5 | 33.7 | `succ_14` |
| Total | 173/250 | 80,767 | | | |

Episode time includes the reset: every reset compiles the MuJoCo model three times (the hard reset, and
`reset_to`'s reset and XML reload), 1.3–2.3 s each on this host, as upstream does.

| Phase | Time |
| --- | --- |
| Checkpoint preparation | < 0.01 s |
| Policy load (five checkpoints, SHA-256 checks) | 4.5 s |
| Runtime preparation (workers ready) | 2.6 s |
| Evaluation | 449.2 s |
| End to end (process wall clock, including writing results) | 458.0 s |
| Throughput | 0.55 episodes/s, 178 executed steps/s (185 in the steady window) |
| Resources | 13.1 CPU cores on average (`time -v`), 1.3 GB maximum RSS (`time -v`), Policy replica 341 MiB VRAM, 3.6% mean GPU utilization; workers spend 20% of their item time waiting for the Policy |

A second full pass with `--concurrency 8 --replicas 1` on the second build produced the same success and
the same executed steps for all 250 episodes, so the outcome does not depend on concurrency, worker
assignment, or which build ran it. That pass took 718.9 s (0.35 episodes/s, 112 steps/s).

### Native comparison

`examples/robomimic/compare_native.py` ran the pinned, unmodified upstream entry point
`python -m robomimic.scripts.run_trained_agent --agent <checkpoint> --n_rollouts 1 --seed <seed>` for each
of the 250 task and seed pairs, with the same checkpoints and the checkpoints' own horizons:

| Task | Kit success | Native success | Kit steps | Native steps | Identical episodes (success and steps) |
| --- | --- | --- | --- | --- | --- |
| `lift` | 50/50 | 50/50 | 2,122 | 2,122 | 50/50 |
| `can` | 50/50 | 50/50 | 5,553 | 5,553 | 50/50 |
| `square` | 39/50 | 39/50 | 10,454 | 10,454 | 50/50 |
| `transport` | 27/50 | 27/50 | 29,014 | 29,014 | 50/50 |
| `tool_hang` | 7/50 | 7/50 | 33,624 | 33,624 | 50/50 |

All 250 episodes match one to one in success and in executed steps. That checks the observation mapping
(keys, order, `float32` conversion), the action path, recurrent-state handling, success detection, and
horizon, episode by episode, rather than by total. It is a pairing under one controlled source of
randomness: the seed determines both the initial state (NumPy) and the GMM samples (torch), and the
example Policy gives each episode its own seeded torch RNG state, so episodes that share one replica do
not consume each other's random numbers.

What is not controlled, and why the result is not a claim of statistical equivalence with other setups:

- The upstream default is one RNG stream across many consecutive episodes (`--n_rollouts 50 --seed 0`),
  where each episode's initial state and samples depend on every earlier episode's length. Those episodes
  cannot be paired with independently seeded ones. Run once per task, that protocol gave Lift 48/50, Can
  50/50, Square 33/50, Transport 21/50, Tool Hang 7/50: a different sample of initial states and GMM draws,
  not a discrepancy. The Square and Transport gaps (6 of 50 each) come from one run of each protocol; they
  are consistent with sampling variance, but no repeated runs measured that variance.
- The identity holds for the same CPU and GPU types and the same torch, cuDNN, and MuJoCo builds. Other
  hardware can change floating-point results in the LSTM or the physics and break the one-to-one match;
  per-task rates would then agree only statistically.
- The model-zoo file names carry the success rate robomimic measured during training, on the
  `offline_study` stack, at the epoch it selected as best. That value comes from other physics, other
  initial states, and a best-of-many selection, so it is a reference, not an expected value. Transport (54%
  here, 42% with the upstream default stream, 78% in the file name) is the clearest difference; Square
  (78% and 66% against 84%) is smaller. The comparison above rules out a difference between the kit and
  the native entry point on this stack and these seeds. It does not identify the cause of the gap to the
  file-name rates: the robosuite 1.4.1 and MuJoCo 2.3.7 stack, best-epoch selection, and different
  evaluation seeds and protocol may each contribute, and no run on the `offline_study` stack separated them.

### Raw results and reproduction

Each run directory keeps `result.json`, `manifest.json` (expected work items and record count),
`metrics.json`, `events.jsonl`, the per-episode records under `queue/completed/`, worker and Policy logs,
and, for the comparison, `comparison.json`, `native-episodes.jsonl`, and one upstream log per episode. The
commands, with `CUDA_VISIBLE_DEVICES=0`:

```bash
benchmarks/robomimic/build-runtime.sh /sim/robomimic
python examples/robomimic/download_model_zoo.py /models/robomimic-zoo
python -m roboscope_kit.runtime.benchmark --benchmark robomimic --self-test \
    --benchmark-manifest /sim/robomimic/benchmark-manifest.json --output-dir /runs/robomimic-self-test
python -m roboscope_kit.runtime.benchmark --benchmark robomimic --suite ph_low_dim --quick-test \
    --checkpoint-source /models/robomimic-zoo --benchmark-manifest /sim/robomimic/benchmark-manifest.json \
    --policy-config roboscope-policy.json --output-dir /runs/robomimic-quick
python -m roboscope_kit.runtime.benchmark --benchmark robomimic --suite ph_low_dim \
    --concurrency 16 --replicas 1 \
    --checkpoint-source /models/robomimic-zoo --benchmark-manifest /sim/robomimic/benchmark-manifest.json \
    --policy-config roboscope-policy.json --output-dir /runs/robomimic-full
python examples/robomimic/compare_native.py --runtime /sim/robomimic --checkpoint-dir /models/robomimic-zoo \
    --kit-run /runs/robomimic-full --output /runs/robomimic-native --jobs 6 --batch-rollouts 50
```

Autotune was not measured: on this shared host its worker search would take most of the CPUs other jobs
were using, so the baseline uses a fixed K/g. Throughput work (for example, avoiding the extra model
compilations of `reset_to`, or batching `act` across workers) is left to a separate change, measured
against this baseline.
