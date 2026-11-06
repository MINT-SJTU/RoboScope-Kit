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
[`benchmarks/rlbench/build-runtime.sh`](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/rlbench/build-runtime.sh) `--accept-licenses <install-dir>`
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
evaluation changes. PerAct pins `hydra-core==1.0.5`, which does not import on Python 3.11 (its config
dataclasses use mutable defaults, which Python 3.11 rejects), so the lock uses hydra-core 1.3.2; `eval.py`
runs under it unchanged (it warns that `version_base` is unset). `transformers` is pinned to 4.46.3, which
still supports torch 2.4 (current releases need torch 2.6), because PerAct imports it at module level for a
training-only scheduler.

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

**Running `build-runtime.sh --accept-licenses` means that you have read the CoppeliaSim Edu and RLBench
licences above and accept them.** Without `--accept-licenses` the build stops before downloading anything.
Nothing licensed is committed to this repository:
CoppeliaSim, the RLBench sources, and the test split are downloaded from their official sources and checked
against the pinned digests and commits. The model (the PerAct checkpoint and CLIP RN50) is not part of the
runtime; `examples/peract/download_assets.py` downloads it with pinned SHA-256s.

The build checks the installed versions and that RLBench and YARR import from their pinned checkouts,
downloads (or, with `RLBENCH_PERACT_TEST_ZIPS`, reuses) and verifies the 18 test zips, extracts them to
`<install-dir>/peract-test/test`, recomputes every task's test-episode digest (section 5) against the
frozen table, launches CoppeliaSim headless and renders one reset through the worker's own code, resets
each of the 450 test episodes once on a freshly loaded task and compares its scene with the recorded one
against the frozen list in `peract-initial-scenes.json` (section 5; `RLBENCH_CHECK_JOBS` CoppeliaSim
processes, by default half the CPUs and at most 6, with the per-episode results in
`<install-dir>/initial-scenes.json`), records `pip-freeze.txt`, and copies
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
tokens) stays in the Policy. PerAct's time feature lags one action behind `step`: its environment wrapper
extracts the observation after an action before incrementing its step counter, so in `eval.py` the agent sees
the counter value `max(step - 1, 0)`, the same value at its first and second action. The example Policy
reproduces this; a Policy computing PerAct's feature from `step` itself gets a different input from the
second action on, and in section 9 that alone shifted most episodes' key frames by one.

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
2. `set_variation(variation)`, `np.random.set_state(random_seed)`, and one `TaskEnvironment.reset()`, which
   samples the scene, places it, and lets objects settle; whatever episode the worker ran before;
3. take the first observation, then the instruction `get_task_descriptions()[0]`.

The worker reads the two pickles with an unpickler that only resolves the RLBench `Demo` and
`Observation` classes and NumPy arrays, and does not load the recorded images. A test episode's digest is
the SHA-256 of its variation and its full random state; a task's digest is the SHA-256 over
`"<episode>:<episode digest>\n"` for episodes 0 to 24. Each record carries its `episode_state_sha256`. The
build rejects a test split whose task digests differ from the frozen table, and a formal aggregate
recomputes each task's digest from its records.

**Initial-scene diagnostic.** Each record carries `initial_scene_position_error_m`, the largest distance,
in metres and per coordinate, between the position of any object of the task after the reset and its
position in the episode's recorded first frame (`task_low_dim_state` of frame 0 in `low_dim_obs.pkl`), and
`initial_scene_worst_object`, that object's name. RLBench's task state lists every object of the task,
including invisible helpers (waypoints, proximity sensors), so a large value does not always mean a visibly
different scene; positions rather than full poses are compared, so quaternion signs and contact-force
readings do not count. Objects are matched by their CoppeliaSim handle, not by their index in the vector:

- The recorded vector has no names. `Task.get_low_dim_state` (`rlbench/backend/task.py:84`, at the pinned
  RLBench commit) concatenates, for each object of `Task._initial_objs_in_scene`, its pose (7 values), then a
  joint's position (+1) or a force sensor's forces and torques (+6). `Scene.reset` rebuilds that list from
  the task's object tree at every reset (`rlbench/backend/scene.py:168`, `task.py:169`).
- That order changes within one loaded task. When the gripper grasps an object, PyRep re-parents it to the
  gripper (`pyrep/robots/end_effectors/gripper.py:50`); the `release()` at the start of the next
  `Scene.reset` (`scene.py:152`) re-parents it to its old parent (`gripper.py:63`), where CoppeliaSim lists
  it after its siblings; `Task.restore_state` (`task.py:342`) restores poses, not that order. On this
  runtime the tree was observed back in its original order by the end of the same reset (during
  `validate()`'s path planning), after `_initial_objs_in_scene` had already been taken: a second reset, or
  re-reading the list after the first, lists the objects in the recorded order again. The worker does not
  rely on that.
- The worker therefore captures the task's object list once, at the first reset after the task is loaded
  (`LoadedTask` in the worker), lays the recorded vector out over that list with a strict length check, and
  compares object by object. A task whose object set changes after loading fails the worker.

An earlier version of this profile compared the vectors by index. After one reset following a played
episode it then paired different objects: 105 of 450 Kit episodes and 117 of 450 episodes of PerAct's own
`eval.py` appeared more than 1 cm off, and that version reset twice to avoid it. The difference was in the
measurement, not in the scene. In a replay of 8 tasks (episodes 1–24 each, 192 episodes, each reset after
replaying the previous episode's Kit actions), compared by object, one reset and a second reset left 21
and 23 episodes more than 1 cm off; the NumPy random state and the placement attempts after the reset were
the same in all 192; compared by index, the first reset showed 110 and its object list was out of order in
95.

**Upstream's `task_low_dim_state` order is unreliable after a grasp, and this profile does not provide it.**
RLBench's `task_low_dim_state` observation is the same vector. After an episode in which the gripper grasped
something, a later episode's vector in the same loaded task lists that object in another place than the
recorded one, so a Policy reading it by index would see objects swapped. The worker's observation config sets
`task_low_dim_state=False` (PerAct does not use it), and the Policy never receives it. The recorded test
set has the same property: the data generator loads the task afresh for every episode
(`tools/dataset_generator.py:375`), but `TaskEnvironment._get_live_demos` (`rlbench/task_environment.py:140`)
retries a failed demonstration in the same loaded task, after a demonstration that may have grasped an
object (`scene.py:425`).

**Build-time check, episode by episode.** `build-runtime.sh` runs
[`initial_scenes`](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/packages/roboscope-kit/src/roboscope_kit/runtime/benchmarks/rlbench/initial_scenes.py):
for every task and test episode it loads the task afresh, resets it once with the worker's code, requires
that the recorded vector has exactly the length of the fresh object list's layout, and compares the scene by
handle. For an episode more than 1 cm off it lays the recorded vector out over the order that releasing each
graspable object, or each ordered pair, would have left (its model of CoppeliaSim's order must first
reproduce the task's own list), and classifies the episode. The outcome is frozen in
[`peract-initial-scenes.json`](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/rlbench/peract-initial-scenes.json)
with its digest, and the build fails when an episode is off that the file does not list, when a listed
recorded order is no longer explained by releasing the same objects, or when a recorded vector does not fit
the layout. Each task runs in its own CoppeliaSim process with episodes in order, because contact settling
depends on what the process simulated before; settling past the tolerance in a task already known to settle
is reported rather than failed. On this runtime all 450 recorded vectors fit the fresh layout, and 20
episodes start more than 1 cm off (section 9):

| Cause | Episodes | Meaning |
| --- | --- | --- |
| `recorded_order` | 7: `place_cups` 13, `place_shape_in_shape_sorter` 7 and 16, `put_groceries_in_cupboard` 7, `stack_blocks` 12, `stack_cups` 9 and 12 | The recorded vector was laid out after one object (`released`) had been grasped and released; over that order the scene is within 1 cm (at most 5 mm off) |
| `recorded_order_and_settling` | 4: `place_cups` 11 and 16, `place_shape_in_shape_sorter` 13 and 24 | Such an order explains part of the difference; objects are still 5–24 cm off |
| `settling` | 9: `place_cups` 19, `place_shape_in_shape_sorter` 1, 2, 4, 5, 12, 14, 15, 18 | No release order explains it; objects that settle into contact during the reset land elsewhere |

Episodes on the list are still scored. For the 7 `recorded_order` episodes the record's
`initial_scene_position_error_m` reports the misordered vector, not a different scene. Contact settling is
not reproducible on this simulator build across processes: the same episode, loaded fresh, settles
differently depending on what the CoppeliaSim process simulated before, so in the worker (which runs
episodes in scheduling order) the settling episodes of `place_cups` and `place_shape_in_shape_sorter` are not
the same set as the build's.

**Upstream behaviour kept as is: the instruction re-runs the task's `init_episode`.**
`get_task_descriptions()` calls the task's `init_episode` again after the first observation was taken. In
tasks whose `init_episode` samples object poses (for example `close_jar` re-samples both jars,
`insert_onto_square_peg` its pegs), those objects move to a second, unvalidated random placement that the
Policy does not see until its first action has been executed, and some tasks register their success
condition twice. PerAct's evaluation, and therefore its published numbers, include this; the profile keeps it
so that it reproduces them. Each record carries `descriptions_moved_objects_m`, the largest distance an
object moved in that call. A corrected profile, which takes the instruction from `reset()`, would be a
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
   digests, a rendered reset, and every test episode's initial scene against the frozen list.
2. `--self-test --benchmark-manifest <install-dir>/benchmark-manifest.json`.
3. `--quick-test` with the real Policy (`open_drawer`, test episode 0).
4. The full `peract_18` evaluation.
5. `examples/peract/run_native.py` (PerAct's own `eval.py` on the same runtime, checkpoint, and test
   episodes) and `examples/peract/compare_native.py`.

## 9. Correctness baseline

Measured on 2026-11-05 with `roboscope-kit` 0.8.0b27 (this change) on a shared Linux host: 2 x Intel Xeon
Silver 4214R (48 logical CPUs, other jobs keeping the load average at 5–16), 188 GB RAM, Ubuntu 22.04, one
NVIDIA RTX 3080 10 GB (driver 560.35.03, CUDA 12.6; about 4.4 GB of it used by other processes throughout).
Xvfb is Ubuntu 22.04's `xvfb` 21.1.4 (unpacked from its package without root), rendering with Mesa 23.0.4
llvmpipe. Checkpoint: PerAct 600k (`seed0`) and CLIP RN50 with the SHA-256s of `examples/peract/assets.py`.
Policy: `examples/peract`, run with the runtime's Python on GPU 0. Workers do not use the GPU.

### Build, self-test, and Quick Test

- Two clean builds into fresh directories with the committed script, reusing the uv cache and the 18
  downloaded test zips (`RLBENCH_PERACT_TEST_ZIPS`; downloading them from Google Drive took about 40
  minutes at 6–8 MB/s over parallel streams). Both verified the versions, the pinned checkouts, all 18
  test-episode digests, and all 18 variation counts, and rendered a headless reset. The second build ran
  `--self-test` itself (the first build's self-test step exposed the report-parsing bug fixed before the
  second; `--self-test` on the first build passed afterwards). Their `pip-freeze.txt` files are identical
  apart from the install path. The second build took 10 min 6 s: about 1 minute for CoppeliaSim, 1.5 for
  the four source checkouts, 0.5 for CPython and the 90 locked packages, 7 to verify and extract the 19 GB
  test split, and 20 s for the build check and `--self-test` (6–7 s).
- A third clean build (build 3, 2026-11-06) with the script that adds the per-episode initial-scene check
  (section 5) passed all of the above. The check took 114.7 s with 6 CoppeliaSim processes (112–193 s in
  standalone runs on build 2, with the host's other load varying); the whole build took 15 min 51 s, this time
  downloading CoppeliaSim and fetching the four sources again. All 450 recorded vectors fit their fresh
  layout, and build 3's per-episode distances equal build 2's to 1e-6 m in all 450 episodes, and so does
  `peract-initial-scenes.json` (20 episodes). On build 2 the result was also identical between 6 and 3
  processes, and between two runs with 6. Before each task got its own process, the result depended on
  how tasks were grouped in a process: with 6 and with 3 processes, 6 `place_shape_in_shape_sorter` episodes
  were more than 1 cm off in one grouping and not in the other.
- `--quick-test` on build 1: `open_drawer` test episode 0 succeeded in 3 actions (93 simulation steps).
  Policy load 11.7 s, runtime preparation 1.6 s, evaluation 8.4 s, 23 s end to end; the scene matched the
  recorded test episode to 1e-6 m. On build 2, with the final Policy (the time-feature fix of the native
  comparison below): success in 3 actions (94 simulation steps); Policy load 18.7 s, runtime preparation
  6.2 s, evaluation 9.0 s, 40 s end to end (load and preparation times vary with the other jobs on the host).

### Full evaluation

`--concurrency 12 --replicas 1` (fixed K/g, so no autotune phase) on build 1: all 450 records present exactly
once, no worker failures, restarts, or OOM events, `formal: true` and `test_episodes_verified: true`:

| Task | Success | Executed actions | Simulation steps | Mean episode time (s) | Planning errors | Timeouts | State > 1 cm off (first / second reset) | Authors' 420k test run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `close_jar` | 8/25 (32%) | 492 | 4,977 | 42.6 | 0 | 17 | 13 / 0 | 56% |
| `insert_onto_square_peg` | 1/25 (4%) | 606 | 10,521 | 56.9 | 0 | 24 | 0 / 0 | 8% |
| `light_bulb_in` | 7/25 (28%) | 503 | 5,171 | 45.7 | 0 | 18 | 0 / 0 | 20% |
| `meat_off_grill` | 17/25 (68%) | 322 | 5,663 | 32.2 | 0 | 8 | 5 / 0 | 92% |
| `open_drawer` | 24/25 (96%) | 83 | 2,326 | 13.6 | 1 | 0 | 0 / 0 | 92% |
| `place_cups` | 0/25 (0%) | 556 | 9,078 | 56.6 | 6 | 19 | 15 / 4 | 0% |
| `place_shape_in_shape_sorter` | 2/25 (8%) | 607 | 7,893 | 60.6 | 0 | 23 | 15 / 12 | 8% |
| `place_wine_at_rack_location` | 14/25 (56%) | 221 | 5,994 | 25.3 | 7 | 4 | 0 / 0 | 48% |
| `push_buttons` | 7/25 (28%) | 514 | 6,306 | 44.2 | 0 | 18 | 0 / 0 | 56% |
| `put_groceries_in_cupboard` | 1/25 (4%) | 428 | 16,583 | 55.7 | 12 | 12 | 3 / 1 | 8% |
| `put_item_in_drawer` | 16/25 (64%) | 384 | 11,922 | 46.5 | 4 | 5 | 15 / 0 | 60% |
| `put_money_in_safe` | 12/25 (48%) | 404 | 13,678 | 40.2 | 0 | 13 | 0 / 0 | 32% |
| `reach_and_drag` | 24/25 (96%) | 168 | 4,098 | 17.3 | 1 | 0 | 23 / 0 | 100% |
| `slide_block_to_color_target` | 17/25 (68%) | 241 | 4,925 | 23.6 | 1 | 7 | 0 / 0 | 72% |
| `stack_blocks` | 6/25 (24%) | 560 | 14,096 | 54.2 | 1 | 18 | 9 / 1 | 8% |
| `stack_cups` | 3/25 (12%) | 532 | 12,177 | 51.9 | 4 | 18 | 7 / 2 | 4% |
| `sweep_to_dustpan_of_size` | 12/25 (48%) | 359 | 7,058 | 35.3 | 2 | 11 | 0 / 0 | 56% |
| `turn_tap` | 21/25 (84%) | 149 | 3,464 | 15.1 | 0 | 4 | 0 / 0 | 96% |
| Total | 192/450; mean 42.7% | 7,129 | 145,930 | | 39 | 219 | 105 / 20 | mean 45.3% |

"State > 1 cm off" counts episodes in which some object of the task (section 5) was more than 1 cm from its
recorded first-frame position: after the first (clearing) reset, which is what a single reset gives, and after
the second, which the episode starts from. The second reset brings 85 of those 105 episodes back to the
recorded state; the 20 that remain are the contact-settling tasks of section 5. The last column is not an
expected value: it is the authors' own test run in `test_data.csv` inside the released checkpoint zip, for
their 420k weights rather than the released 600k ones, on their renderer. No task reported a fail condition;
every failure was a timeout or a planning error.

| Phase | Time |
| --- | --- |
| Checkpoint preparation | < 0.01 s |
| Policy load (agent build, 600k weights, CLIP RN50, SHA-256 checks) | 9.1 s |
| Runtime preparation (12 workers, each with its Xvfb and CoppeliaSim) | 1.6 s |
| Evaluation | 1,618.8 s |
| End to end (process wall clock, including writing results) | 1,635.5 s |
| Throughput | 0.277 episodes/s (0.288 between the 10th and 90th percentile completions), 4.38 executed actions/s, 90.1 simulation steps/s |
| Resources | 12.0 CPU cores on average (the kit's sampling; `time -v` sees 5.9 because it counts only waited-for processes), 1.8 GB maximum RSS of the CLI, Policy replica 1.17 GB VRAM after load and up to about 4.2 GB in use, 90% mean GPU utilization |

The evaluation is bound by the single Policy replica: PerAct's 100^3-voxel Perceiver takes about 0.2 s per
action on this GPU, the replica was busy 90% of the time, and workers spent 67% of their episode time waiting
for it. A second replica does not fit next to the other processes on this 10 GB GPU. Batching `act` across
workers, or a lower-precision Policy, is the first target of a throughput change, measured against this
baseline.

### Native comparison

`examples/peract/run_native.py` ran PerAct's pinned `eval.py` with its README's test command (all 18 tasks in
one process, `eval_envs=1`, test episodes 0–24, `eval_type=last`, headless) on build 1, with the same
checkpoint, CLIP file, and test split, and logged every episode through read-only hooks. Its own
`eval_data.csv` agrees with that log (for example `close_jar` return 44.0 = 11/25, length 16.88 = 422/25
actions). `compare_native.py` paired it with the Kit run above episode by episode, and a second Kit pass on
build 2 with `--concurrency 6` gives the Kit-to-Kit spread of the same protocol:

| Task | Kit (build 1, K=12) | Kit (build 2, K=6) | Native `eval.py` | Kit vs native: same outcome | Kit vs Kit: same outcome | First action equal (Kit 1 vs native) | Executed actions (Kit 1 / native) | Simulation steps (Kit 1 / native) | Native start > 1 cm off |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `close_jar` | 8 | 10 | 11 | 20/25 | 19/25 | 20/25 | 492 / 422 | 4,977 / 4,797 | 17 |
| `insert_onto_square_peg` | 1 | 2 | 2 | 24/25 | 24/25 | 17/25 | 606 / 587 | 10,521 / 10,165 | 0 |
| `light_bulb_in` | 7 | 5 | 7 | 19/25 | 21/25 | 22/25 | 503 / 494 | 5,171 / 5,169 | 0 |
| `meat_off_grill` | 17 | 18 | 16 | 20/25 | 22/25 | 25/25 | 322 / 330 | 5,663 / 5,506 | 7 |
| `open_drawer` | 24 | 23 | 23 | 22/25 | 22/25 | 25/25 | 83 / 86 | 2,326 / 2,345 | 0 |
| `place_cups` | 0 | 1 | 0 | 25/25 | 24/25 | 13/25 | 556 / 565 | 9,078 / 8,787 | 14 |
| `place_shape_in_shape_sorter` | 2 | 1 | 0 | 23/25 | 24/25 | 15/25 | 607 / 625 | 7,893 / 7,664 | 16 |
| `place_wine_at_rack_location` | 14 | 11 | 10 | 19/25 | 16/25 | 24/25 | 221 / 239 | 5,994 / 8,561 | 0 |
| `push_buttons` | 7 | 9 | 6 | 20/25 | 19/25 | 23/25 | 514 / 507 | 6,306 / 6,024 | 0 |
| `put_groceries_in_cupboard` | 1 | 3 | 3 | 21/25 | 21/25 | 17/25 | 428 / 304 | 16,583 / 10,656 | 6 |
| `put_item_in_drawer` | 16 | 14 | 13 | 14/25 | 17/25 | 19/25 | 384 / 430 | 11,922 / 12,196 | 13 |
| `put_money_in_safe` | 12 | 12 | 12 | 19/25 | 19/25 | 23/25 | 404 / 422 | 13,678 / 14,206 | 0 |
| `reach_and_drag` | 24 | 23 | 23 | 22/25 | 24/25 | 18/25 | 168 / 157 | 4,098 / 3,817 | 23 |
| `slide_block_to_color_target` | 17 | 23 | 22 | 18/25 | 17/25 | 23/25 | 241 / 127 | 4,925 / 3,282 | 0 |
| `stack_blocks` | 6 | 8 | 6 | 21/25 | 23/25 | 13/25 | 560 / 544 | 14,096 / 14,095 | 13 |
| `stack_cups` | 3 | 0 | 3 | 23/25 | 22/25 | 16/25 | 532 / 527 | 12,177 / 10,202 | 8 |
| `sweep_to_dustpan_of_size` | 12 | 13 | 11 | 20/25 | 22/25 | 22/25 | 359 / 387 | 7,058 / 6,839 | 0 |
| `turn_tap` | 21 | 21 | 22 | 22/25 | 21/25 | 24/25 | 149 / 126 | 3,464 / 3,423 | 0 |
| Total (mean success) | 192 (42.7%) | 197 (43.8%) | 190 (42.2%) | 372/450 | 377/450 | 359/450 | 7,129 / 6,879 | 145,930 / 137,734 | 117 |

What the comparison checks, and what it found:

- **Inputs and mapping.** All 450 pairs have the same variation and the same instruction. Where both sides
  start from the recorded state (329 episodes), the first action, which depends only on the first
  observation, is bit-identical in 278, and the two runs then execute on average the same first 3.5 actions.
  Of the 51 that differ, 26 differ only in rotation, 14 in position by one 1 cm voxel, 7 only in the gripper
  or collision flag, and 4 by more than a voxel: the reset does not reproduce the scene bit for bit even on one
  side (two fresh loads of the same episode leave the arm joints
  up to 3e-4 rad apart, which changes thousands of wrist-camera pixels), and PerAct's arg-max sometimes falls
  the other way.
- **The time feature (found by this comparison).** The first Kit pass computed PerAct's time feature from
  `step`. Its mean success was 46.9% (211/450), but its action sequences left the native ones after 1.4
  actions on average, because `eval.py` repeats the first time value (section 4). With the lag reproduced,
  the sequences agree for 3.2 actions on average and the totals match (192 against 190).
- **Initial states.** 117 of the 450 native episodes started more than 1 cm from their recorded state (none of
  them a task's first episode), because `eval.py` resets once after the previous episode (section 5). The Kit
  started 20 episodes that far off, all in the contact-settling tasks.
- **Outcomes.** Per-episode outcomes cannot be paired: RLBench's motion planner samples IK solutions and
  RRT-Connect paths from CoppeliaSim's own generator, which neither side seeds, and the reset jitter above
  feeds into every later step. The Kit and native outcomes agree in 372 of 450 episodes, and the two Kit passes
  agree in 377: the Kit differs from PerAct's `eval.py` about as much as from itself. The task totals behave
  the same way; the largest Kit-to-native gap, `slide_block_to_color_target` (17 against 22), is smaller than
  the gap between the two Kit passes on that task (17 against 23). One run per side does not establish
  statistical equivalence; it shows no systematic difference at this sample size, and the overall means (42.7%
  and 43.8% against 42.2%) agree within that spread.

Native timing on the same GPU: 8,330 s end to end (2 h 19 min) for 450 episodes, 0.054 episodes/s and 0.83
executed actions/s, 2.3 CPU cores on average and 6.3 GB maximum RSS (`time -v`). Besides running one episode
at a time, `eval.py` loads every recorded image of each test episode (`get_stored_demos` with
`image_paths=False`), which the Kit worker does not need. The Kit run above was 5.1 times faster in
episodes/s on the same GPU.

### Raw results and reproduction

Each Kit run directory keeps `result.json`, `manifest.json` (expected work items and record count),
`metrics.json`, `events.jsonl`, the per-episode records under `queue/completed/` (with every executed action
and the reset diagnostics), and worker and Policy logs; the native run keeps `native-episodes.jsonl`,
`run.json`, PerAct's `eval_data.csv`, and its log. The commands, with `CUDA_VISIBLE_DEVICES=0`:

```bash
# --accept-licenses accepts the CoppeliaSim Edu and RLBench licences (section 2).
XVFB=/path/to/Xvfb KIT_PYTHON=/path/to/kit/python \
    benchmarks/rlbench/build-runtime.sh --accept-licenses /sim/rlbench
python examples/peract/download_assets.py /models/peract
python -m roboscope_kit.runtime.benchmark --benchmark rlbench --self-test \
    --benchmark-manifest /sim/rlbench/benchmark-manifest.json --output-dir /runs/rlbench-self-test
python -m roboscope_kit.runtime.benchmark --benchmark rlbench --suite peract_18 --quick-test \
    --checkpoint-source /models/peract --benchmark-manifest /sim/rlbench/benchmark-manifest.json \
    --policy-config roboscope-policy.json --output-dir /runs/rlbench-quick
python -m roboscope_kit.runtime.benchmark --benchmark rlbench --suite peract_18 \
    --concurrency 12 --replicas 1 \
    --checkpoint-source /models/peract --benchmark-manifest /sim/rlbench/benchmark-manifest.json \
    --policy-config roboscope-policy.json --output-dir /runs/rlbench-full
PYTHONPATH=packages/roboscope-kit/src /sim/rlbench/env/bin/python examples/peract/run_native.py \
    --runtime /sim/rlbench --checkpoint-source /models/peract --output /runs/rlbench-native
python examples/peract/compare_native.py --kit-run /runs/rlbench-full --native /runs/rlbench-native \
    --output /runs/rlbench-compare.json
```

The second Kit pass (build 2, `--concurrency 6 --replicas 1`) ran at the same rate as K=12: 1,633.3 s
evaluation, 0.274 episodes/s (0.276 steady), 4.18 executed actions/s, with workers waiting 42% instead of 67%
of their episode time; its GPU memory peaked at 9.9 GB device-wide, close to this card's limit. Autotune was
not measured: the Policy replica is the bottleneck from about K=6 on this GPU, and the worker search would
take CPUs other jobs were using, so the baseline uses a fixed K/g.
