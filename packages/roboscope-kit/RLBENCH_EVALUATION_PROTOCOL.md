# RLBench Evaluation Protocol

This document defines the SDK-native RoboScope RLBench evaluation contract. The kit's benchmark runtime
owns benchmark enumeration, simulator processes, concurrency, recovery, and result aggregation. User
inference code owns only the public `Policy` lifecycle and model-specific preprocessing and
postprocessing.

`roboscope-kit` requires Python 3.11 or newer in every process that imports it, including the Policy and
the simulator workers. Python 3.8–3.10 are unsupported. See the
[Python support policy](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/packages/roboscope-kit/README.md#python-support).

Status: code candidate, build-verified on Linux x86_64 (clean build, `--self-test`, Quick Test, full
evaluation, and native comparison in section 9). Not merged or released until the version that contains it
(`0.8.0b27` or later) is on PyPI.

## 1. Benchmark scope

The `benchmark` identifier is `rlbench`, the protocol version is `rlbench-peract-v1`, and the only suite is
`peract_18`: the **PerAct 18-task reproduction profile**. It is the RLBench multi-task evaluation of PerAct
(Shridhar et al., CoRL 2022) as PerAct's pinned `eval.py` runs it on PerAct's released test split. It is
not RLBench as a whole (RLBench has more than 100 tasks), not RLBench's own task sets, and not the
evaluation of later work that reuses these tasks with other test episodes or settings; scores of other
profiles must be named and reported separately.

| Task | Variations | Task | Variations |
| --- | --- | --- | --- |
| `close_jar` | 20 | `put_groceries_in_cupboard` | 9 |
| `insert_onto_square_peg` | 20 | `put_item_in_drawer` | 3 |
| `light_bulb_in` | 20 | `put_money_in_safe` | 3 |
| `meat_off_grill` | 2 | `reach_and_drag` | 20 |
| `open_drawer` | 3 | `slide_block_to_color_target` | 4 |
| `place_cups` | 3 | `stack_blocks` | 60 |
| `place_shape_in_shape_sorter` | 5 | `stack_cups` | 20 |
| `place_wine_at_rack_location` | 3 | `sweep_to_dustpan_of_size` | 2 |
| `push_buttons` | 50 | `turn_tap` | 2 |

Each task covers all of its variations: every test episode was generated with a variation drawn at random
(`--all_variations=True`), and the episode replays that variation. A formal `peract_18` evaluation runs
test episodes 0–24 of every task, 450 episodes in total, as PerAct's README test command does
(`framework.eval_episodes=25`, `eval_from_eps_number=0`). Quick Test runs `open_drawer` test episode 0
only. `--episodes N` (1 to 25) runs episodes `0..N-1` of every task and is reported as non-formal. The
task list, episode count, cameras, action representation, and episode length are frozen in the kit's
[task table](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/packages/roboscope-kit/src/roboscope_kit/runtime/benchmarks/rlbench/tasks.py).

### What is not supported, and why

| Combination | Status | Reason |
| --- | --- | --- |
| PerAct's 18 tasks, released test split, PerAct cameras and key-frame actions | Supported (`peract_18`) | Released test episodes and checkpoint; reproduced against PerAct's `eval.py` (section 9) |
| Other RLBench tasks, RLBench's own task sets (for example FS10, MT15/MT30, or all 100+ tasks) | Not supported | No released test episodes or checkpoint for them; each needs its own pinned profile and test set |
| Freshly generated test episodes (`dataset_generator.py`) | Not supported | Every generation samples new scenes; PerAct recommends its released split for reproducibility |
| Other action modes (joint velocities, end-effector deltas, IK without planning) | Not supported | The profile's action is PerAct's `MoveArmThenGripper(EndEffectorPoseViaPlanning(), Discrete())` |
| The `val` split and checkpoint selection | Out of scope | The released checkpoint is evaluated as is (`eval_type=last`), section 2 |
| Training, data collection, real robots | Out of scope | This protocol measures closed-loop simulation success only |

Policies other than PerAct may be evaluated on `peract_18`; a score is comparable to the PerAct paper's
RLBench table only for the same test split and the same aggregation (section 7).

## 2. Environment, versions, and licence gates

The RLBench simulator is not a dependency of `roboscope-kit`. Build it with
[`benchmarks/rlbench/build-runtime.sh`](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/rlbench/build-runtime.sh) `<install-dir>`
(Linux x86_64, uv 0.11.14, git, curl, `tar` with xz, network access, and an Xvfb server; no GPU is needed
for the simulator). It installs a uv-managed CPython 3.11.15 under `<install-dir>/python` and a venv under
`<install-dir>/env`:

| Component | Pin |
| --- | --- |
| CoppeliaSim | Edu 4.1.0, Ubuntu 20.04 build, `CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz` from Coppelia Robotics' download server, SHA-256 `512de3a7…a8f` (the full digest is in the script) |
| PyRep | `stepjam/PyRep` commit `231a1ac6b0a179cff53c1d403d379260b9f05f2f` (4.1.0.3), its CFFI bindings built against this CoppeliaSim |
| RLBench | `MohitShridhar/RLBench` branch `peract`, commit `ad991951bc53e4f3b73b803a75cf4b7d55295cf7` (1.2.0) |
| YARR | `MohitShridhar/YARR` branch `peract`, commit `8d0f2e228b97efa207eeccedfc6dac9334342ba3` |
| PerAct | `peract/peract` commit `02fb87681c5a47be9dbf20141bedb836ee2d3ef9`, with `patches/peract-python311.patch` |
| Python packages | [`runtime-requirements.txt`](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/rlbench/runtime-requirements.txt): `numpy==1.26.4`, `torch==2.4.1` (PyPI CUDA 12.1 wheels), OpenAI CLIP commit `d05afc4`, `hydra-core==1.3.2`, `transformers==4.46.3`, and the rest of PerAct's imports |
| Test split | PerAct's released `test` zips, one per task, from the Google Drive folder PerAct's README links, each with its pinned size and SHA-256 in [`peract-test-set.txt`](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/rlbench/peract-test-set.txt) |

The test episodes were generated in May 2022, and the RLBench fork's task and scene code has not changed
since then (later commits only touch the dataset generator, the demo loader, and the README), so the pinned
commit samples the same scenes (section 5).

**Python 3.11.** PyRep, RLBench, YARR, and PerAct were written for Python 3.8 with torch 1.7. PyRep's CFFI
bindings and the RLBench fork build and run unchanged on Python 3.11 with NumPy 1.26. PerAct needs one
source patch: it imports `pytorch3d`, which has no Python 3.11 build, at module level; the patch moves the
import into the training-time augmentation function that uses it and removes two unused imports. Nothing in
evaluation changes. PerAct pins `hydra-core==1.0.5`, whose OmegaConf 2.0 does not import on Python 3.11, so
the lock uses hydra-core 1.3.2; `eval.py` runs under it unchanged (it warns that `version_base` is unset).
`transformers` is pinned to 4.46.3, the last release that supports torch 2.4, because PerAct imports it at
module level for a training-only scheduler.

**Headless rendering.** CoppeliaSim 4.1 renders its vision sensors through Qt's OpenGL context and needs
an X display even in headless mode (`-h`); Qt's `offscreen` and `minimalegl` platforms crash in
`CVisionSensorGlStuff`. Each worker therefore starts its own Xvfb server (`-displayfd`, so concurrent
workers never collide), and CoppeliaSim renders with Mesa's llvmpipe in the worker process
(`LP_NUM_THREADS` = the manifest's `client_cpu_threads`). The build links the host's Xvfb into
`<install-dir>/bin/Xvfb`; install it from the distribution (Ubuntu: `xvfb`) or point `XVFB` at a binary.
PerAct's test split was rendered by its authors on their own OpenGL stack, so the RGB images a worker renders
are not bit-identical to the recorded ones; the Kit and PerAct's `eval.py` render with the same stack here,
so the comparison in section 9 is paired.

**Licences.** These are accepted by the user, never by the build or the kit:

- CoppeliaSim Edu (Coppelia Robotics GmbH) is free only for educational entities (hobbyists, students,
  teachers, professors, schools, and universities) and only for non-commercial use; companies and research
  institutions need a commercial licence. The RoboScope Kit maintainers approved its use for this
  evaluation on 2026-11-05.
- RLBench (Imperial College London) may be used only for non-commercial, internal, or academic research.
  The PerAct test split was generated with RLBench and is published by the PerAct authors for reproducing
  their results, without a licence of its own.
- PyRep is MIT, YARR and PerAct are Apache-2.0, except PerAct's voxelization and data preprocessing,
  which are under the ARM licence it ships (Imperial College London, non-commercial research use like
  RLBench's). OpenAI CLIP is MIT, torch is BSD-3-Clause.

The build refuses to run until `ROBOSCOPE_ACCEPT_COPPELIASIM_EDU_LICENSE=YES` and
`ROBOSCOPE_ACCEPT_RLBENCH_LICENSE=YES` are set, and nothing licensed is committed to this repository:
CoppeliaSim, the RLBench sources, and the test split are downloaded from their official sources and checked
against the pinned digests and commits. The model (the PerAct checkpoint and CLIP RN50) is not part of the
runtime; `examples/peract/download_assets.py` downloads it with pinned SHA-256s.

The build checks the installed versions and that RLBench and YARR import from their pinned checkouts,
downloads (or, with `RLBENCH_PERACT_TEST_ZIPS`, reuses) and verifies the 18 test zips, extracts them to
`<install-dir>/peract-test/test`, recomputes every task's test-episode digest (section 5) against the
frozen table, launches CoppeliaSim headless and renders one reset through the worker's own code, records
`pip-freeze.txt`, and copies
[`benchmark-manifest.json`](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/rlbench/benchmark-manifest.json)
into `<install-dir>`. Its runtime paths (`python`, `coppeliasim`, `test_set`, `xvfb`) are relative to the
manifest. Workers import the kit from its source tree through `PYTHONPATH`, so `roboscope-kit` is not
installed into this environment. `--self-test` re-checks the `open_drawer` digest and renders its first
test episode.

Run an evaluation with the kit installed in any supported Python:

```bash
python -m roboscope_kit.runtime.benchmark \
    --benchmark rlbench --suite peract_18 --checkpoint-source /models/peract \
    --benchmark-manifest <install-dir>/benchmark-manifest.json \
    --policy-config /path/to/roboscope-policy.json \
    --output-dir /runs/rlbench-peract-18
```

Use a new or empty output directory for each evaluation or self-test. A directory containing only
`errors.jsonl` can be retried, preserving earlier errors; other run artifacts require a new directory.

## 3. User policy lifecycle

The user entrypoint implements `Policy` and calls `run(policy)`. The runtime starts one process per Policy
replica with the command from `--policy-config` and calls `load(SessionContext)` once per replica,
`reset(EpisodeContext)` before every episode, `act(Observation)` before every action, `end_episode` when the
episode ends or the worker stops, and `unload()` at the end. One replica serves many workers concurrently,
so a Policy with per-episode state keys it by `episode_id`.

`EpisodeContext.task_id` is the task name. `EpisodeContext.metadata` carries `protocol`, `episode` (the test
episode index), `variation`, `episode_length` (25), `instruction`, and `cameras`. `PolicyCapabilities` must
declare `rlbench` and `peract_18`. The
[example Policy](https://github.com/MINT-SJTU/RoboScope-Kit/tree/main/examples/peract) serves the released
PerAct checkpoint and keeps PerAct's observation dict, time feature, CLIP tokenization, voxelization, and
inference on the model side.

## 4. Observation and action contract

The worker uses PerAct's observation configuration (`create_obs_config`): cameras `front`, `left_shoulder`,
`right_shoulder`, and `wrist` at 128x128 with RLBench's `OPENGL` render mode, RGB and point clouds on,
masks and the overhead camera off, joint positions and velocities, gripper pose, open state, matrix, and
finger positions on. It additionally keeps the depth buffer, which RLBench captures for the point cloud
anyway, so rendering and point clouds are unchanged. Every observation carries, converted from RLBench's
`Observation` without any other change:

| Feature | Shape, dtype | Source |
| --- | --- | --- |
| `images.<camera>` | `(128, 128, 3)`, `uint8` | `<camera>_rgb` |
| `state.<camera>_depth` | `(128, 128)`, `float32` | `<camera>_depth` converted to metres with the camera's near and far planes |
| `state.<camera>_point_cloud` | `(128, 128, 3)`, `float32` | `<camera>_point_cloud`: world-frame x, y, z in metres per pixel |
| `state.<camera>_camera_extrinsics` | `(4, 4)`, `float32` | `misc["<camera>_camera_extrinsics"]` (camera-to-world) |
| `state.<camera>_camera_intrinsics` | `(3, 3)`, `float32` | `misc["<camera>_camera_intrinsics"]` |
| `state.gripper_open` | `(1,)`, `float32` | `gripper_open` (1 open, 0 closed) |
| `state.gripper_joint_positions` | `(2,)`, `float32` | `gripper_joint_positions`, unclipped |
| `state.gripper_pose` | `(7,)`, `float32` | `gripper_pose`: x, y, z, qx, qy, qz, qw |
| `state.joint_positions`, `state.joint_velocities` | `(7,)`, `float32` | Panda arm joints |
| `language.instruction` | string | `get_task_descriptions()[0]` (section 5) |

`<camera>` is each of the four cameras. `step` counts the actions executed so far, from 0. PerAct's
`float32` conversions happen in YARR's rollout loop; doing them in the worker gives the Policy the same
values. PerAct's own preprocessing (clipping the finger positions to `[0, 0.04]`, its time feature, and CLIP
tokens) stays in the Policy.

An action is one key frame for RLBench's `MoveArmThenGripper(EndEffectorPoseViaPlanning(), Discrete())`, 9
`float64` values (`rlbench.ee-pose-planning-v1`): the gripper position and orientation in the world frame
(`x, y, z, qx, qy, qz, qw`, a unit quaternion), gripper open (`> 0.5`) or closed, and ignore collisions (1)
or not (0). The worker passes it to `TaskEnvironment.step` unchanged. RLBench plans a path to the pose with
CoppeliaSim's RRT-Connect (100 trials, 10 ms each, with collision checking unless ignore collisions is set;
a failed collision-checked plan is retried once without it), executes the path in physics (`0.05 s` per
simulation step), stopping early when the task succeeds, and then opens or closes the gripper. `float64` is
the dtype PerAct passes; a `float32` round trip would move the planner's target. The worker accepts exactly
one action per `act`, as `eval.py` calls the agent before every step.

## 5. Test episodes and initial states

Test episode `i` of a task is directory `episode<i>` of `<task>/all_variations/episodes` in the test split
(PerAct takes the `i`-th of the natural-sorted episode directories; the build requires exactly
`episode0`–`episode24`). The worker reproduces PerAct's `CustomMultiTaskRLBenchEnv.reset_to_demo`:

1. read `variation_number.pkl` and the `random_seed` stored in `low_dim_obs.pkl` (the NumPy `MT19937`
   state captured just before that episode's scene was sampled during data generation);
2. `set_variation(variation)`, `np.random.set_state(random_seed)`, and `TaskEnvironment.reset()`, which
   samples the scene, places it, and lets objects settle; then the same three calls once more
   (see below);
3. take the first observation, then the instruction `get_task_descriptions()[0]`.

The worker reads the two pickles with an unpickler that only resolves the RLBench `Demo` and
`Observation` classes and NumPy arrays, and does not load the recorded images. A test episode's digest is
the SHA-256 of its variation and its full random state; a task's digest is the SHA-256 over
`"<episode>:<episode digest>\n"` for episodes 0 to 24. Each record carries its `episode_state_sha256`. The
build rejects a test split whose task digests differ from the frozen table, and a formal aggregate
recomputes each task's digest from its records.

Each record also carries `initial_scene_max_abs_diff`: the largest difference between the task's
low-dimensional state (object poses and joint positions) after the reset and the same state in the
episode's recorded first frame. It shows that the pinned simulator samples the recorded scene (section 9).

**Deviation from `eval.py`: two seeded resets instead of one.** PerAct resets once. A reset that starts
from the end of a played episode can place objects differently from the recorded scene: RLBench's
placement rejects samples that collide with objects the previous episode left behind, so the number of
random draws, and with it the scene, depends on how the previous episode ended (for example, after a
`close_jar` episode that screwed the lid on, the next test episode's jars land elsewhere). In PerAct's
`eval.py`, which plays the 25 episodes of a task in order, episode `i`'s scene therefore depends on the
planner's random outcome in episode `i - 1`. The Kit runs episodes on whichever worker is free, so it
cannot reproduce that history, and a single reset would make the scene depend on scheduling. The worker
therefore resets twice with the same variation and random state: the first reset clears the previous
episode, and the second samples the recorded scene, whatever ran before. Records carry
`clearing_reset_max_abs_diff`, the first reset's distance from the recorded scene, which is what a
single reset would have given. Section 9 measures both, and the same distance for every episode of
PerAct's own `eval.py`.

**Upstream behaviour kept as is: the instruction re-runs the task's `init_episode`.**
`get_task_descriptions()` calls the task's `init_episode` again after the first observation was taken. In
tasks whose `init_episode` samples object poses (for example `close_jar` re-samples both jars,
`insert_onto_square_peg` its pegs), those objects move to a second, unvalidated random placement that the
Policy does not see until its first action has been executed, and some tasks register their success
condition twice. PerAct's evaluation, and therefore its published numbers, include this; the profile keeps it
so that it reproduces them. Each record carries `descriptions_moved_objects_max_abs`, the largest change
of the task state that call made. A corrected profile, which takes the instruction from `reset()`, would be a
separate suite with its own results.

## 6. Rollout, termination, and success

The worker reproduces YARR's `RolloutGenerator` with PerAct's environment wrapper, for at most 25 actions
(`rlbench.episode_length`):

- the Policy is asked for an action with the latest observation, and the action is executed with
  `TaskEnvironment.step`;
- `IKError`, `ConfigurationPathError`, or `InvalidActionError` (no reachable or collision-free path, a pose
  outside the workspace, a non-unit quaternion) ends the episode as a failure at that step, as PerAct's
  wrapper does; the step counts as executed. Any other exception fails the worker; the runtime records it and
  requeues the work item on a restarted worker, up to its restart limit, so a crash is never counted as a
  failed episode;
- a step that returns reward >= 1 (RLBench's reward is 1 when all of the task's success conditions hold)
  succeeds and terminates the episode. A step whose task reports termination without success (a fail
  condition) ends it as a failure;
- after the 25th action without success the episode times out as a failure.

A record's `success` is whether the episode succeeded, `steps` is its executed actions (1–25), and
`metadata.termination` is `success`, `fail_condition`, `timeout`, or the planning error's name. It also
carries `simulation_steps` (physics steps of the executed arm paths, counted through RLBench's scene step
callback; the few steps of gripper actuation are not counted), every executed `actions` vector, and the
`instruction`. PerAct's evaluation also calls the agent once more on the terminal observation and records
videos; neither affects the outcome, and the worker does neither.

## 7. Aggregation

`official_result` has `metric = "per_task_success_rate"`, `formal`, one row per task with `task`,
`episodes`, `successes`, and `success_rate` in the task order of section 1, and `mean_success_rate`, the
unweighted mean of the 18 task success rates. PerAct reports each task's success rate (its evaluation logs
`return` = 100 x the mean episode reward per task) and their average; the mean is derived from the rows.
`diagnostics` adds, per task, executed actions, simulation steps, and termination counts; it is not part of
the official metric. The standard `result` adds the total episodes, successes, executed actions, episodes/s,
and actions/s.

A record is identified by `<task>-episode-<episode:02d>`. The aggregate rejects records outside the work
manifest, duplicates, out-of-range action counts, inconsistent terminations, and timeouts before 25 actions.
A formal result requires all 450 records and verified test-episode digests (`test_episodes_verified: true`);
Quick Test and `--episodes` runs are not formal. Sharded runs sum the per-task rows.

## 8. Validation sequence

1. Build the runtime with `build-runtime.sh` in a fresh directory; it verifies versions, the test split's
   digests, and a rendered reset.
2. `--self-test --benchmark-manifest <install-dir>/benchmark-manifest.json`.
3. `--quick-test` with the real Policy (`open_drawer`, test episode 0).
4. The full `peract_18` evaluation.
5. `examples/peract/run_native.py` (PerAct's own `eval.py` on the same runtime, checkpoint, and test
   episodes) and `examples/peract/compare_native.py`.

## 9. Correctness baseline

RESULTS
