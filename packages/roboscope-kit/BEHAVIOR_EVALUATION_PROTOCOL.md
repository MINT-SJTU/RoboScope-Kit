# BEHAVIOR Evaluation Protocol

This document defines the SDK-native RoboScope BEHAVIOR evaluation contract. The kit's benchmark runtime
owns benchmark enumeration, simulator processes, concurrency, recovery, and result aggregation. User
inference code owns only the public `Policy` lifecycle and model-specific preprocessing and
postprocessing.

`roboscope-kit` requires Python 3.11 or newer in every process that imports it, including the Policy and
the simulator workers. Python 3.8–3.10 are unsupported. See the
[Python support policy](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/packages/roboscope-kit/README.md#python-support).

Status: integration candidate, not accepted. Section 9 records the runtime build, the self-test, the
real-model Quick Tests and one episode of every task (instance 301 of all 100 tasks) with BEHAVIOR2026VLA
on `challenge_2026_rgb224`; section 10 the performance baseline. The formal 1,000 episodes, the native
comparison of the full workload and the full evaluation of `challenge_2026` have not been run.

## 1. Benchmark scope

The `benchmark` identifier is `behavior` and the protocol version is `behavior-challenge-2026-v1`. Both
suites are the **2026 BEHAVIOR Challenge public-test profile**, as BEHAVIOR-1K
[`v3.9.3-post1`](https://github.com/StanfordVL/BEHAVIOR-1K/tree/v3.9.3-post1) runs it. That release is
the one the organizers use for the 2026 final evaluation
([updates, 09/28/2026](https://github.com/StanfordVL/BEHAVIOR-1K/blob/v3.9.3-post1/docs/challenge/updates.md)).
They run the same tasks, instances, step limits and metric and differ only in the observation wrapper,
the evaluator's `--env-wrapper`, which a challenge submission chooses and submits with its policy
([evaluation rules](https://github.com/StanfordVL/BEHAVIOR-1K/blob/v3.9.3-post1/docs/challenge/evaluation.md)):
`challenge_2026` uses `RGBDFullResWrapper` (RGB and linear depth at the data-collection resolutions, which
the rules name for official RGB-D evaluation), `challenge_2026_rgb224` uses `DefaultWrapper` (224x224 RGB,
the evaluator's default). Section 4 lists the features of each. A result names its suite; results of the
two suites are not merged.

| Profile | Tasks | Status |
| --- | --- | --- |
| 2026 challenge, public test instances 0–9 (`challenge_2026`, `challenge_2026_rgb224`) | 100 activities in 7 scenes, R1Pro | Supported |
| 2026 challenge, hidden test instances | 100 | Not supported: held out by the organizers |
| 2025 challenge (BEHAVIOR-1K v3.7.x, Isaac Sim 4.5) | 50 (tasks 0–49 of 2026) | Not supported: its simulator stack requires Python 3.10, and its instances, timeouts (2x human length) and observation tracks differ; it needs its own profile |
| All BEHAVIOR-1K activities | 1,000 | Not supported: there is no evaluation protocol, instance set or released policy for them |
| Training, data collection, leaderboard submission, real robots | – | Out of scope |

The 100 tasks are BEHAVIOR activities, not all of BEHAVIOR-1K. Their IDs and step limits are frozen in the
kit's [task table](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/packages/roboscope-kit/src/roboscope_kit/runtime/benchmarks/behavior/tasks.py)
in challenge task-ID order (`metadata/B100_task_misc.csv` of the 2026 task instances; tasks 0–49 are the
2025 challenge's 50 tasks in the same order). Scenes: `house_single_floor` (34 tasks),
`house_double_floor_lower` (32), `house_double_floor_upper` (10), `restaurant_diner` (8), `Rs_int` (6),
`hotel_suite_large` (5), `office_cubicles_right` (5).

A formal evaluation of either suite runs public test instance indices 0–9 (instances 301–310) of every
task, one rollout each: 1,000 episodes, as the challenge reports results. `--episodes N` (1 to 10) runs
instance indices `0..N-1` of every task; values below 10 are non-formal. Quick Test runs instance 301 of
`turning_on_radio` (task 0) only.

## 2. Environment, versions, and licence gates

The simulator is not a dependency of `roboscope-kit`. Build it with
[`benchmarks/behavior/build-runtime.sh`](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/behavior/build-runtime.sh)
`--accept-licenses <install-dir>` (Linux x86_64, an NVIDIA GPU with a Vulkan ICD, uv 0.11.14, git, curl,
network access). It installs a uv-managed CPython 3.11.15 under `<install-dir>/python` and a venv under
`<install-dir>/env`:

| Component | Pin |
| --- | --- |
| BEHAVIOR-1K (OmniGibson, BDDL, JoyLo) | `StanfordVL/BEHAVIOR-1K` tag `v3.9.3-post1`, commit `bd049de3119acdcdf2334fe9e1ebe060fa20c108`; OmniGibson 3.9.3, BDDL and JoyLo installed in development mode from that checkout, as its `setup.sh` does |
| Isaac Sim | `isaacsim[all,extscache]==5.1.0.0` from `pypi.nvidia.com` |
| Python packages | [`runtime-requirements.txt`](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/behavior/runtime-requirements.txt): the exact environment `setup.sh --omnigibson --bddl --joylo --eval` leaves on this commit, including `torch==2.7.0+cu128`, `numpy` 1.x, `cffi==1.17.1`, `websockets` 15, `torch-cluster`, and LeRobot at `wensi-ai/lerobot` commit `436812bd8ee39b768c645c248c91f1330834e687` |
| Scene and object assets | `behavior-1k-assets-3.9.0.zip` (31,457,673,073 bytes, SHA-256 `09e9fce6…095e`) |
| Robot assets | `omnigibson-robot-assets-3.8.2.zip` (SHA-256 `3d813b21…e126`) |
| Task instances | `2026-challenge-task-instances.zip` (SHA-256 `d4ac3d72…1f81`); the build and the self-test also check its `B100_task_misc.csv` and `task.jsonl` against the frozen digests in the task table |

**Primitives.** The challenge installs with `setup.sh --primitives` as well. Of that extra the lock keeps
`warp-lang==1.12.0`, which OmniGibson imports at module level, and leaves out cuRobo, a CUDA 12.8 source
build used only by OmniGibson's motion-planning action primitives, which the evaluator does not import.

**LeRobot.** `v3.9.3-post1`'s OmniGibson requires `lerobot[dataset]` from the moving `release/b1k` branch
of `wensi-ai/lerobot`, which has since moved to a release that needs Python 3.12. BEHAVIOR-1K `v3.9.4` pins
it to commit `436812bd`, "the 2026 challenge release" (StanfordVL/BEHAVIOR-1K#2374); the lock uses that
commit. The evaluator does not import LeRobot. For the same reason the lock takes `pyarrow==25.0.1`, the
newest below 26 as `v3.9.4` pins it (StanfordVL/BEHAVIOR-1K#2369): pyarrow 26 needs NumPy 2, and Isaac Sim
pins `numpy==1.26.0`. `av==15.1.0` is the newest release LeRobot's `dataset` extra allows; the evaluator's
video writer imports it.

The three archives come from the Hugging Face dataset `behavior-1k/zipped-datasets` at revision
`9f0d57d465726976ed98138d3f8b8ca3e2186775`, the versions `v3.9.3-post1`'s `asset_utils` downloads; the full
digests are in the build script.

**Licences.** These are accepted by the user, never by the build or the kit:

- Isaac Sim is distributed under the NVIDIA Isaac Sim End User License Agreement. OmniGibson only starts it
  with `OMNI_KIT_ACCEPT_EULA=YES`; the kit sets it for the runtime it starts.
- The BEHAVIOR Data Bundle (the scene, object and robot assets) is licensed for non-commercial academic
  use inside OmniGibson only and may not be redistributed. Its assets are encrypted; OmniGibson's
  `download_key()` installs the decryption key once the licence is accepted.
- BEHAVIOR-1K's code is MIT.

**Running `build-runtime.sh --accept-licenses` means that you have read both licences and accept them.**
Without `--accept-licenses` the build stops before downloading anything. Nothing licensed is committed to
this repository.

The build installs exactly the lock (`--no-deps`, so nothing is re-resolved), the three source packages
from the checkout, and removes the `websockets` and `packaging` copies bundled with Isaac Sim's
extensions, as `setup.sh` does. It then compares `pip freeze` with the lock, checks the torch, Isaac Sim
and `cffi` versions and that BDDL, OmniGibson and JoyLo import from the checkout, downloads (or, with
`BEHAVIOR_ZIPS`, reuses) and verifies the three archives, unpacks them under `<install-dir>/data`
(`OMNIGIBSON_DATA_PATH`), installs the dataset key, copies
[`benchmark-manifest.json`](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/behavior/benchmark-manifest.json)
into `<install-dir>`, and runs the check of `--self-test` (below). Its runtime paths (`python`, `data`)
are relative to the manifest. Workers import the kit from its source tree through `PYTHONPATH`, so
`roboscope-kit` is not installed into this environment.

`--self-test` checks the dataset versions, the task metadata digests and the key, then, once per suite,
builds the worker's evaluator for `turning_on_radio` with the suite's wrapper (scene load), loads public
test instance 301 through the evaluator's own load path, and checks that the first observation maps to
every SDK feature of the suite with the frozen shapes, non-empty RGB and, for `challenge_2026`, finite
depth. No action is executed. The build's own check (`check.json`) covers `challenge_2026`.

**Resources.** BEHAVIOR's documentation asks for an RTX 2070 or better with 8 GB, 32 GB of RAM, and
reports 150–300 s of scene loading and about 13.5 simulated steps/s with full-resolution RGB-D on an RTX
4090. The unpacked data take about 36 GB and `env/` about 25 GB. Isaac Sim renders through Vulkan: check
that `vulkaninfo --summary` lists the NVIDIA GPU before building.

Run an evaluation with the kit installed in any supported Python, for example the BEHAVIOR2026VLA
example on `challenge_2026_rgb224`:

```bash
python -m roboscope_kit.runtime.benchmark \
  --benchmark behavior --suite challenge_2026_rgb224 \
  --benchmark-manifest <install-dir>/benchmark-manifest.json \
  --policy-config examples/behavior2026vla/roboscope-policy.json \
  --checkpoint-source <behavior2026vla>/checkpoints/BEHAVIOR2026VLA \
  --output-dir runs/behavior --concurrency 1 --replicas 1
```

The GR00T example runs on `challenge_2026` the same way (`examples/gr00t/roboscope-policy.json`).

## 3. User policy lifecycle

The runtime calls `load(SessionContext)` once per replica, then for each episode `reset(EpisodeContext)`,
one `act(Observation)` per control step, and `end_episode(episode_id)`. `EpisodeContext.task_id` is the
task name; its metadata carries `task_index` (the challenge task ID), `instance_index`, `instance_id`
and `max_steps`. A Policy declares `behavior` in `supported_benchmarks` and returns an action chunk of
one or more actions per call. The worker executes them one per control step and asks again when they
are used up, as the native client does with `--replay-action-chunk-size`; every step is simulated,
rendered and scored as usual, and only the observations of the steps in between are not sent. A chunk
is therefore equivalent to one action per call only when the Policy would not have read those
observations: the GR00T example returns one action per call, the BEHAVIOR2026VLA example the 13
actions its model call executes. A Policy that keeps the state of each open episode can serve several
workers per replica.

## 4. Observation and action contract

The worker builds every observation from the native evaluator's own preprocessed observation
(`BatchedEvaluator._preprocess_obs` under the suite's official wrapper), without rescaling. The wrapper's
`camera_spec()` sets the camera resolutions and modalities when the environment is created; the worker
rejects an image or depth map of another size:

| Feature | `challenge_2026` (`RGBDFullResWrapper`) | `challenge_2026_rgb224` (`DefaultWrapper`) | Native source |
| --- | --- | --- | --- |
| `images.head` | 720x720x3 `uint8` | 224x224x3 `uint8` | `robot_r1::robot_r1:zed_link:Camera:0::rgb` without its alpha channel |
| `images.left_wrist`, `images.right_wrist` | 480x480x3 `uint8` | 224x224x3 `uint8` | `...:left_realsense_link:Camera:0::rgb`, `...:right_realsense_link:Camera:0::rgb` without alpha |
| `state.head_depth`, `state.left_wrist_depth`, `state.right_wrist_depth` | 720x720 / 480x480 `float32`, metres | absent | `...::depth_linear` |
| `state.proprioception` | 61 `float32` | 61 `float32` | `robot_r1::proprio`, ordered as `PROPRIOCEPTION_INDICES["R1Pro"]` (base velocity, arm, end-effector and gripper states, trunk) |
| `state.camera_poses` | 21 `float32` | 21 `float32` | `robot_r1::cam_rel_poses`: head, left wrist, right wrist camera pose relative to the robot base (x, y, z, qx, qy, qz, qw) |
| `state.task_id` | 1 `int64` | 1 `int64` | `task_id`, the challenge task ID |

The head camera has the evaluator's 40° horizontal aperture in both suites. The observation contains only
RGB, depth (`challenge_2026`) and proprioception (plus the camera poses and task ID the native observation
also carries), as the challenge track allows; there is no language instruction in the native observation, so the profile has none.

The action (`behavior.r1pro-joint-v1`) is 23 `float32` values: holonomic base velocity x, y, yaw (0:3),
torso joint positions (3:7), left arm joint positions (7:14), left gripper (14), right arm joint positions
(15:22), right gripper (22), as the default R1Pro configuration (`omnigibson/eval/r1pro.yaml`, absolute
joint position controllers, `action_normalize: false`) receives it. The worker passes the action to the
evaluator unchanged.

## 5. Test instances and initial states

Each episode is one public test instance: the evaluator resets the robot to its default configuration,
loads the instance's stored task-relevant object states and presampled robot pose
(`2026-challenge-task-instances/scene_test/public/...-tro_state.json`), settles physics for 25 steps, and
resets the environment. Instance index `i` is instance `301 + i`. Each worker process starts with
`seed_everything(0)` and `PYTHONHASHSEED=0`, builds one task's scene, evaluates one instance, and exits, as
the native command with one `--instance-indices` value does; no scene is reused across episodes.

## 6. Rollout, termination, and success

The evaluator steps physics at 120 Hz and the policy at 30 Hz, rendering once per control step. An episode
ends when the BDDL goal is satisfied (`success`), when another OmniGibson task termination fires
(`task_termination`), or when the task's `Timeout` termination fires (`max_steps`). `max_steps` is
`int(1.5 x mean human demonstration length)` from `task.jsonl`, frozen per task in the task table and
checked against the evaluator's `Timeout`. OmniGibson evaluates terminations before it counts the step
(`episode_steps >= max_steps`), so a timed-out episode executes `max_steps + 1` actions; the native result
reports that number in `steps` and `time.simulator_steps`, and so does the kit (for `turning_on_radio`,
`max_steps` is 3224 and a timed-out episode runs 3225 actions). `v3.9.3-post1`'s evaluator has no
wall-clock budget, and neither does the kit.

Each record keeps the evaluator's own result (`native_result`): `success`, `steps`, the Q score
(`q_score.final`), the time and agent-distance metrics, and the same JSON the native command writes
(`<output-dir>/behavior/json/<task>_<instance>_0.json`).

## 7. Aggregation

The challenge ranks by **Q**: per episode, the largest fraction of a BDDL goal option's predicates that
were not satisfied at the start and are at the end (1 for a success); per task, the mean over its
instances; overall, the mean over the 100 tasks (`score_utils.compute_final_q_score`, a missing rollout
counts as 0). The kit reports `mean_q_score` that way and, separately, `mean_success_rate` (the mean over
tasks of the fraction of episodes with full success). Partial credit is never counted as success.
A formal result requires exactly one record for each of the 1,000 episodes; the aggregate rejects missing,
duplicate, unknown or out-of-contract records. Merging sharded runs sums per-task episode, success and
Q-score counts, recomputes each task's mean Q and the overall mean Q and success rate from the sums, rejects
an episode reported by two shards, and marks the merged result formal only when it covers all 1,000
episodes.

## 8. Validation sequence

1. Build with `build-runtime.sh --accept-licenses <install-dir>` and run `--self-test`.
2. Quick Test with the real Policy (one episode of `turning_on_radio`), then the same episode with the
   example's native runner (`examples/gr00t/run_native.py` or `examples/behavior2026vla/run_native.py`).
3. The full 1,000-episode evaluation and the same workload natively, compared with
   `examples/gr00t/compare_native.py`.

## 9. Native comparison and acceptance

### Nondeterminism

OmniGibson's physics and rendering are not deterministic across runs (the challenge's evaluation
documentation states that rollouts of the same policy may differ), GR00T's diffusion action head
samples noise from an unseeded torch generator in its server, and BEHAVIOR2026VLA's flow-matching sampler
draws its noise from the loaded policy's JAX key, which advances with every model call and so depends on
the episodes a replica served before. Paired episodes therefore need not agree step for step, and one
episode per (task, instance) and path is not enough to compare a task's success. The comparison pairs every (task, instance) of the Kit and native runs, rejects missing,
duplicate and unpaired results, and reports per-task successes, mean Q and mean executed steps of both
sides and every episode whose outcome differs. Acceptance requires, for the full workload, that the
difference in overall mean Q and in each task's success count lies within the run-to-run spread of the
native path measured by repeating a subset of tasks natively.

### BEHAVIOR2026VLA on `challenge_2026_rgb224`

The behavior code of this branch built as Kit `0.8.0b75`, before it was rebased onto `0.8.0b77` (the
rebase changed no behavior code), `IliaLarchenko/BEHAVIOR2026VLA` at revision
`dc8c443f`, solution commit `68c7f912`, BEHAVIOR-1K `v3.9.3-post1`; one RTX 4090 (48 GB, driver
580.105.08) with a container quota of 20 CPU cores (Xeon Platinum 8470Q) and 96 GiB of RAM; the runtime
was built with this branch's `build-runtime.sh` on 2026-11-23 and its build check and `--self-test`
passed for both suites.

| Run | Workload | Result |
| --- | --- | --- |
| Kit Quick Test, K=1 G=1, six runs on successive heads of this branch, the last on Kit `0.8.0b78` after the rebase | `turning_on_radio` instance 301 | 2 successes (Q 1.0, 1,866 and 2,073 actions), 4 time-outs (Q 0.0, 3,225 actions) |
| Native `run_native.py` | `turning_on_radio` instances 301–303 | 301: time-out, Q 0.0, 3,225 actions; 302: success, 2,035 actions; 303: time-out, Q 0.0 |
| Kit, 4 shortest tasks after the Quick Test task, K=2 G=2 | instance 301 of 4 tasks | 4 of 4 records, 1 success (`installing_a_modem`) |
| **Kit, every task, K=2 G=2, 2026-11-24/25** | instance 301 of all 100 tasks (100 episodes, non-formal) | 100 of 100 records, no missing or duplicate record, no worker failure or retry; 23 successes, 77 time-outs; **mean Q 0.410**, mean success rate 0.23; 1,430,844 actions |

The solution's authors report a Q of 0.392 for this checkpoint over instances 301–310 of every task
(1,000 episodes). The Kit's 0.410 over instance 301 alone is of the same size but is not a comparison:
it is one episode per task, and the two runs of the same 16 tasks that this branch has (the full run
above and an earlier, stopped run with K=3 on an earlier head) agree on success for 15 of them but
differ in mean Q, 0.382 against 0.506, with single-task differences up to 0.625. The run-to-run spread
of one episode is that large on the Kit path alone.

What these runs establish: the Kit path runs the full frozen task list with the real model end to end
(runtime, Policy load in its own environment, observation and action mapping, the evaluator's own
termination and Q score, records, aggregation) with complete coverage and no failure, and its outcomes
are of the size the model's authors report. What they do not establish: agreement with the native
evaluator per task. The native path has run three episodes of one task; the paired comparison of the
same 100-episode workload, the native run-to-run spread and the formal 1,000 episodes are open.

### GR00T N1.7 on `challenge_2026`

All runs on one RTX 4090 (48 GB, driver 580.105.08) in a container limited to 12 CPU cores and 96 GiB of
RAM, the behavior code of this branch (built as Kit 0.8.0b58 before rebasing onto 0.8.0b66, with the same
behavior code), BEHAVIOR-1K `v3.9.3-post1`, GR00T checkpoint `kmy17518/gr00t-n1.7-b1k-multitask`
`checkpoint-238000` (revision `5831e9dc`), 2026-11-14.

| Step | Result |
| --- | --- |
| Clean build in an empty directory | Passed in 27 min with the downloaded archives reused through `BEHAVIOR_ZIPS`: lock installed and compared with `pip freeze`, archives verified and unpacked, key installed, the build's instance check (cold caches: scene load 442 s) and the kit's `--self-test` |
| `--self-test` on a built runtime | Passed in 198 s: scene load 189 s, instance 301 load 6.4 s, `max_steps` 3224, every feature with its frozen shape, non-empty RGB, finite depth |
| Kit Quick Test (`turning_on_radio`, instance 301) | Complete, 1 record: no success, Q 0.0, 3225 actions (`max_steps`) |
| Native, same episode, twice (`run_native.py`) | No success, Q 0.0, 3225 actions, both times |
| `compare_native.py` | 1 of 1 episodes paired; success, Q and steps agree; no missing, duplicate or unpaired records |

The checkpoint does not solve this instance on either path, so the agreement of outcomes is weak
evidence: a zero against a zero would also hide a wrong mapping. The inputs were therefore compared
directly: for the same evaluator observation, the request the Kit path sends to the GR00T server
(evaluator observation, SDK observation, kit msgpack frame, `native_request`) equals the native
`WebsocketPolicy` request, value for value, on every key the server reads (section 4). The paired runs
also differ in the robot's travelled distance (base 8.5–12.3 m across the two Kit and two native runs),
which shows the run-to-run nondeterminism described above on both paths.

GR00T's 100-task evaluation and its native comparison have not been run. At the measured rate
(section 10) and K=1, one instance of every task is at most 1,581,928 actions (every episode timing out),
about 133 hours of rollout plus about 4 hours of scene loads; the formal 1,000 episodes are about ten
times that on each path. The CPU quota (12 cores, about 7 per worker) leaves little room for more
workers on this machine, so the full matrix needs more machines or a maintainer-approved smaller
instance count per task.

## 10. Performance baseline

### BEHAVIOR2026VLA on `challenge_2026_rgb224`

The 100-task run of section 9, K=2 workers and G=2 replicas on one GPU, one instance per worker process
(99 planned recycles):

| Measure | Value |
| --- | --- |
| Policy load (two replicas, JAX compilation of both model calls included) | 129 s |
| Formal evaluation | 86,749 s (24.1 h) for 100 episodes and 1,430,844 actions |
| Throughput | 16.49 actions/s (steady window 16.42), 9.90 actions/s per worker during rollouts; 0.00115 episodes/s (4.15 per hour) |
| GPU | mean utilization 34 % (median 27 %, max 100 %), peak memory 42.4 GB, of which 6.3 GB per replica |
| CPU | 11.6 cores on average of the 20-core quota, throttled in 0.7 % of the cgroup's periods |
| Disk | each worker's RTX texture cache reached 9–12 GB for a large scene and is removed when the worker exits |

Where the 173,499 worker-seconds (2 workers x evaluation time) went, from the per-episode records and the
Policy's per-episode timing:

| Part | Seconds | Share | Per action |
| --- | ---: | ---: | ---: |
| Isaac Sim launch, scene and instance load (mean 284 s, median 269 s, max 643 s per instance) | 28,416 | 16.4 % | 19.9 ms |
| Simulator step and render (rollout minus observation and Policy wait) | 115,001 | 66.3 % | 80.4 ms |
| Observation build | 1,135 | 0.7 % | 0.8 ms |
| Policy wait | 28,419 | 16.4 % | 19.9 ms |
| — of which the Policy's `act` (model and chunk replay) | 16,226 | 9.4 % | 11.3 ms |
| — of which transport (Kit protocol and process boundary) | 12,193 | 7.0 % | 8.5 ms |
| Process start and teardown outside these | 527 | 0.3 % | 0.4 ms |

The model ran 110,114 times, once per 13 actions; an `act` with a model call averages at most 147 ms.

The simulator dominates: one worker steps about 12 actions/s while it is not waiting, and the scene
load of every instance adds a sixth. At this rate the formal 1,000 episodes take about ten days on this
machine.

### Throughput changes (Kit `0.8.0b86`)

Three changes leave every simulated step, rendered image and score as before and raise the throughput:

- **Chunks.** The BEHAVIOR2026VLA Policy returns the 13 actions of a model call at once instead of one per
  request (12 of 13 Policy round trips go), and the worker replays them.
- **Several workers per replica.** The Policy keeps each open episode's rollout state, so 4 workers fit on
  one 48 GB GPU with 2 replicas (6.3 GB each).
- **A texture cache per scene.** The run keeps one RTX texture cache per scene under
  `<output-dir>/texture-cache`, evaluates one scene after another and removes a scene's cache once no
  pending or running item needs it. A warm cache halves a scene load (`turning_on_radio` 250 s to 127 s).

Checks on the same machine:

- With a fixed, observation-independent action sequence, the head-camera image of a warm-cache run
  differs from a cold-cache run by no more than two cold-cache runs differ from each other (mean
  absolute difference 1–4 of 255).
- 16 tasks of 6 scenes, instance 301. The "before" runs are the 100-task run of section 9 and a run of the
  same code the day after; of the "after" runs, the first two used a prototype of these changes behind
  switches and the third this change, built as `0.8.0b83` before its rebase onto `main`:

| Configuration | Mean Q | Successes | Evaluation | Actions/s |
| --- | ---: | ---: | ---: | ---: |
| Before (K=2, G=2, one action per request, a cold cache per worker) | 0.500, 0.413 | 6, 5 | 7,820 s | 14.3–16.5 |
| After (K=4, G=2, chunks, scene caches) | 0.290, 0.531, 0.555 | 4, 6, 6 | 4,413–4,881 s | 22.5–24.2 |

The Q values of a configuration differ as much between its runs as between the configurations: one
episode per task does not resolve a difference of this size, and no change of the outcome was found.
The throughput is 1.57–1.69 times that of the run made the same day with the configuration before (1.57
with this version). The CPU now limits it: 17.7–18.5 of the 20 cores on average (GPU memory peak
46.5 GB). In the run of this version the scene caches took at most 22 GB and none was left afterwards.

Render skipping was measured and rejected: rendering only the step whose observation the Policy
receives, with the 12 steps in between simulated but not rendered, leaves the physics bit for bit the
same and doubles the step rate, but the RTX renderer accumulates over frames (OmniGibson notes that a
pose change takes two renders to appear) and the image at a query step differs from the every-step
image by a mean of 8–27 of 255. BEHAVIOR's evaluator renders every step and has no setting that
removes the accumulation, so rendered steps cannot be skipped.

Left for later: running the instances of a task in one process (the evaluator's `--num-envs`; four
environments in one process use 14.3 GB of GPU memory and load each instance 46 % faster) applies
to the formal 10 instances per task, and batching the model calls of several workers.

### GR00T N1.7 on `challenge_2026`

Kit Quick Test, K=1 worker and G=1 replica (`runs/kit-quick-2`):

| Measure | Value |
| --- | --- |
| Policy load (GR00T server start and model load, warm file cache) | 28 s |
| Formal evaluation | 1,088 s for 1 episode |
| Scene load and instance load inside it | 110 s |
| Rollout | 977 s for 3,225 actions: 3.30 actions/s |
| Of which waiting for the Policy | 664 s (206 ms per action) |
| End-to-end actions/s (evaluation window) | 2.95 |
| GPU | mean utilization 14 % (max 36 %), peak memory 20.9 GB, of which the GR00T replica 6.8 GB |
| CPU | 7.2 cores on average, throttled in 30 % of the cgroup's periods |

The native runs took 1,304–1,311 s for the same 3,225 actions (2.46 actions/s) after about 190 s of
launch and scene load. The Kit path is faster per action mainly because its Policy sends the server only
the five arrays it reads (about 2.9 MB), while the native client sends the whole observation, including
RGBA and depth (about 6.8 MB), which the server deep-copies.

Where the time goes, and candidates for a later `perf(behavior)` issue: the Policy wait is 68 % of the
rollout (GR00T infers at every step in `temporal_ensemble` mode, about 85 ms on the server, plus the
transfer of three full-resolution images and depth through the kit protocol to the Policy and of the RGB
on to the server); each episode pays a scene load of 110–190 s because a process evaluates one instance;
the CPU quota throttles the simulator. Kit worker concurrency is limited by the GR00T server, which
serves one episode per connection, so K workers need K replicas (6.8 GB each).
