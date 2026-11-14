# SimplerEnv Evaluation Protocol

This document defines the SDK-native RoboScope SimplerEnv evaluation contract. The kit's benchmark runtime
owns the episode matrix, simulator processes, concurrency, recovery, and result aggregation. User inference
code owns only the public `Policy` lifecycle and model-specific preprocessing and postprocessing.

`roboscope-kit` requires Python 3.11 or newer in every process that imports it, including the Policy and the
simulator workers. Python 3.8–3.10 are unsupported. See the
[Python support policy](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/packages/roboscope-kit/README.md#python-support).

## 1. Benchmark scope

The `benchmark` identifier is `simpler_env`, the protocol version is `simpler-env-v1`, and the action
representation is `simpler_env.ee-delta-axangle-v1`. The backend is SimplerEnv's `main` branch: SAPIEN 2.2 with
the CPU-simulated ManiSkill2 environments of `ManiSkill2_real2sim`. The GPU-parallel ManiSkill3 port of the
Bridge environments (SimplerEnv's `maniskill3` branch) is a different simulator and renderer and is not
integrated; its speedups and results are not validated by this integration.

There are three suites. They are separate protocols: their task counts, results, and performance numbers are
never pooled.

| Suite | Robot | Setup | Tasks | Variants | Episodes |
| --- | --- | --- | --- | --- | --- |
| `google_robot_visual_matching` | Google Robot (`google_robot_static`) | Real-image background overlay on the policy camera and the four robot/scene recolor settings (`urdf_version`) | 5 | 244 | 864 |
| `google_robot_variant_aggregation` | Google Robot | No overlay; base scene plus background, lighting, distractor, table-texture, and cabinet variants | 5 | 84 | 1,722 |
| `widowx_visual_matching` | WidowX (`widowx`, `widowx_sink_camera_setup`) | Real-image overlay of the Bridge table or sink | 4 | 4 | 96 |

Every variant is one `simpler_env/main_inference.py` invocation from the pinned SimplerEnv commit's official
RT-1 / RT-1-X scripts, with the checkpoint and policy arguments removed:

| Suite | Task | Variants x episodes | Steps | Script |
| --- | --- | --- | --- | --- |
| Visual matching | `pick_coke_can` (horizontal, vertical, standing) | 3 orientations x 4 `urdf_version` x 25 object positions (5 x 5 grid) = 300 | 80 | `rt1_pick_coke_can_visual_matching.sh` |
| | `move_near` | 4 `urdf_version` x 60 predefined episodes = 240 | 80 | `rt1_move_near_visual_matching.sh` |
| | `open_drawer` (top, middle, bottom) | 3 drawers x 4 `urdf_version` x 9 robot poses with their overlays = 108 | 113 | `rt1_drawer_visual_matching.sh` |
| | `close_drawer` (top, middle, bottom) | 108, as `open_drawer` | 113 | `rt1_drawer_visual_matching.sh` |
| | `place_apple_in_closed_top_drawer` | 4 `urdf_version` x 3 robot poses x 9 apple positions = 108 | 200 | `rt1_put_in_drawer_visual_matching.sh` |
| Variant aggregation | `pick_coke_can` | 3 orientations x 9 variants x 25 = 675 | 80 | `rt1_pick_coke_can_variant_agg.sh` |
| | `move_near` | 8 variants x 60 = 480 | 80 | `rt1_move_near_variant_agg.sh` |
| | `open_drawer` | 3 drawers x 7 variants x 9 robot poses (3 x 3 grid) = 189 | 113 | `rt1_drawer_variant_agg.sh` |
| | `close_drawer` | 189, as `open_drawer` | 113 | `rt1_drawer_variant_agg.sh` |
| | `place_apple_in_closed_top_drawer` | 7 variants x 3 robot poses x 9 apple positions = 189 | 200 | `rt1_put_in_drawer_variant_agg.sh` |
| WidowX visual matching | `put_spoon_on_tablecloth`, `put_carrot_on_plate`, `stack_green_block_on_yellow_block` | 24 predefined episodes each | 60 | `rt1x_bridge.sh` |
| | `put_eggplant_in_basket` | 24 predefined episodes | 120 | `rt1x_bridge.sh` |

The coke-can variants are `base`, two table textures, two distractor settings, two backgrounds, and two
lightings; move near has `base`, `no_distractor`, two backgrounds, two lightings, and two table textures; the
drawer and apple tasks have `base`, two backgrounds, two lightings, and two cabinets (all ray-traced). The
variant names, their arguments, and the episode counts are frozen in
`roboscope_kit/runtime/benchmarks/simpler_env/tasks.py`; `examples/rt1/compare_native.py` checks every
invocation the official scripts make against them (section 8).

A formal run evaluates every episode of one suite. `--quick-test` runs the first episode of the suite's first
variant; `--episodes N` takes at most `N` episodes from every variant, and any run short of the full matrix is
reported with `formal: false`. `official_result.formal` is true only when the result holds a record for every
episode of the full matrix: one `--shard-index` shard of a full run reports `formal: false` and `partial: true`,
and only the merge of the complete shard set (`shard_merge`) is formal. `benchmark.formal_selection` records
whether the requested workload is the full matrix; it is the same on every shard.

### What is not supported, and why

- **The two alternative-camera variant-aggregation variants** (`*AltGoogleCamera*`): the official scripts run
  them, but `calc_metrics_evaluation_videos.py` leaves them out of the variant-aggregation average ("we only
  consider camera variants when investigating the effect of different robustness factors"). They are not in
  the suite.
- **`google_robot_pick_object`, the middle and bottom "place into closed drawer" tasks, and the prepackaged
  `simpler_env.make()` environments**: not part of the official scripts' evaluation (the scripts comment the two
  drawers out, and the paper evaluates the top drawer only); the prepackaged environments sample their
  configuration per reset instead of sweeping it.
- **Octo's three seeds on WidowX**: `octo_bridge.sh` repeats the 96 episodes with `--octo-init-rng 0 2 4`, the
  policy's own sampling seed; the environment matrix is the one above. A stochastic Policy can be evaluated
  three times; the kit does not pool runs.
- **MMRV and Pearson correlation**: they measure how a set of policies ranks in simulation against the real
  world and need the real results and several policies; they cannot be derived from one model's success rates,
  and the kit does not report them.
- **The ManiSkill3 backend**: see above.
- **Videos and per-episode action plots**: the official evaluator writes them; the kit records outcomes,
  steps, and episode statistics instead (section 5).

### Environment and invocation

The simulator is not a dependency of `roboscope-kit`. Build it with
[`benchmarks/simpler_env/build-runtime.sh`](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/simpler_env/build-runtime.sh)
`<install-dir>` (Linux x86_64, git, uv 0.11.14, network access, an NVIDIA GPU with a Vulkan driver). It
installs uv-managed CPython 3.11.15 and a venv with exactly
[`runtime-requirements.txt`](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/simpler_env/runtime-requirements.txt),
then SimplerEnv and its `ManiSkill2_real2sim` submodule editable from a checkout at the pinned commits
(`<install-dir>/src/SimplerEnv`), as SimplerEnv's README installs them:

| Component | Pin |
| --- | --- |
| SimplerEnv | `eddb5690749a1e34267807c0f7b5eede323918d7` (main, 2026-09-30) |
| ManiSkill2_real2sim (scenes, objects, robot assets, real-image overlays) | `ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682`, the commit SimplerEnv records for its submodule |
| CPython | 3.11.15 |
| SAPIEN | 2.2.2 |
| numpy | 1.24.4 (SimplerEnv's README: numpy < 2.0 for its IK) |
| gymnasium | 0.29.1 (`ManiSkill2_real2sim` requires < 1.0) |
| TensorFlow / TF-Agents / TF Hub | 2.15.1 (with its CUDA wheels) / 0.19.0 / 0.16.0, what `main_inference.py` and its RT-1 policy import |

The TensorFlow stack is in the runtime because SimplerEnv's native entry point imports it; the kit's workers
never import it, and the example Policy uses it from this environment so that the Kit and native runs share one
TensorFlow. The build checks both commits and that the checkouts are unmodified, the versions, that
`simpler_env` and the assets resolve to the checkout, and then builds, resets, renders, and steps one episode of
every suite plus one ray-traced drawer variant through the kit worker's own code. `--self-test` repeats that
check. Set `ROBOSCOPE_PYPI_INDEX_URL` to install the locked packages from a PyPI mirror.

SAPIEN renders through Vulkan, and SAPIEN 2.2 breaks when the Vulkan loader loads any ICD (installable client
driver) besides the regular NVIDIA one. With NVIDIA's Vulkan SC ICD (`nvidia_icd_vksc.json`) installed, every
renderer teardown segfaults inside `libnvidia-vksc-core.so`; upstream's own evaluator, which drops one
environment per episode, crashes the same way. With Mesa's ICDs installed next to the NVIDIA one, creating the
renderer can fail with `ErrorExtensionNotPresent`. Export `VK_ICD_FILENAMES=<dir>/nvidia_icd.json` (`<dir>` is
`/usr/share/vulkan/icd.d` or `/etc/vulkan/icd.d`) for the build and every run. While another ICD is installed
and the variable is unset, the build, `--self-test`, and every run stop before they render anything. Merging
shard results renders nothing and needs no ICD setting.

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json   # or /etc/vulkan/icd.d/; see above
benchmarks/simpler_env/build-runtime.sh /sim/simpler-env
python -m roboscope_kit.runtime.benchmark \
    --benchmark simpler_env --suite google_robot_visual_matching \
    --checkpoint-source /models/my-checkpoint \
    --benchmark-manifest /sim/simpler-env/benchmark-manifest.json \
    --policy-config /path/to/roboscope-policy.json \
    --output-dir /runs/simpler-vm
```

Use a new or empty output directory for each evaluation or self-test.

## 2. User policy lifecycle

The runtime starts one process per Policy replica with the command from `--policy-config` and calls:

1. `Policy.load(SessionContext)` once per replica with the resolved `BenchmarkSpec` of the suite;
2. `Policy.reset(EpisodeContext)` after the episode's environment is built and reset, before its first action;
3. `Policy.act(Observation)` whenever the worker needs an action;
4. `Policy.end_episode(episode_id)` when the episode ends;
5. `Policy.unload()` when evaluation ends.

`EpisodeContext.task_id` is the task (`pick_coke_can`, ...), and `EpisodeContext.metadata` carries `protocol`,
`suite`, `variant`, `policy_setup` (`google_robot` or `widowx_bridge`, the official script's `--policy-setup`),
`robot`, `env_name`, `control_freq`, and `max_episode_steps`. A replica serves several workers, so a stateful
Policy keeps its state per `episode_id`.

## 3. Observation and action contract

| Feature | Type | Shape | Construction |
| --- | --- | --- | --- |
| `images.primary` | `uint8` RGB | Google Robot `(512, 640, 3)`; WidowX `(480, 640, 3)` | `get_image_from_maniskill2_obs_dict(env, obs)`: `overhead_camera` (Google Robot) or `3rd_view_camera` (WidowX), from `obs_mode="rgbd"` with segmentation; in the visual-matching suites the real-image overlay is composited into it |
| `language.instruction` | string | | `env.get_language_instruction()` after reset and after every step; it changes when the long-horizon apple task advances its subtask |

The action is one `float64` row of 8 values per environment step: end-effector delta translation (3), delta
rotation as an axis-angle vector (3), and the gripper command (1), which go to `env.step` unchanged, followed by
a terminate signal (`> 0` terminates). The controller conventions are the environments' own:

| Robot | Control mode | Control / simulation frequency |
| --- | --- | --- |
| Google Robot | `arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner` | 3 Hz / 513 Hz |
| WidowX | `arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos` | 5 Hz / 500 Hz |

The two robots keep their upstream frequencies; nothing is unified. Normalization, rotation conversion, and
gripper semantics (for example RT-1's gripper inversion and binarization on WidowX) are the Policy's job,
exactly as they are the upstream policy wrapper's. A Policy may return several rows; the worker executes them
in order and drops the rest when the instruction changes.

## 4. Episodes, poses, and seeds

For each work item the worker parses the variant's frozen arguments with upstream
`simpler_env.evaluation.argparse.get_args` (so `np.linspace` sweeps, the robot quaternion
`Pose(q=euler2quat(r, p, y)) * Pose(q=center)`, and the `key=value` build keywords are upstream's) and takes the
episode in `maniskill2_evaluator`'s loop order: robot x, robot y, robot orientation, then object x and y (or the
predefined episode id). The episode's robot and object poses are recorded with its result.

As upstream, every episode builds a **fresh environment** with `build_maniskill2_env(env_name,
**build_kwargs, obs_mode="rgbd", robot=..., sim_freq=..., control_mode=..., control_freq=...,
max_episode_steps=..., scene_name=..., camera_cfgs={"add_segmentation": True}, rgb_overlay_path=...)`, with
`shader_dir="rt"` put first when the variant enables ray tracing, and resets it with
`env.reset(options={"robot_init_options": ..., "obj_init_options": ...})` and **no seed**. A fresh
ManiSkill2 environment seeds its main generator with 2022 on construction, so the per-episode seed, and every
draw the scene makes from it, is the same as upstream's. The environment is dropped, not closed, after the
episode, as upstream does. The custom-scene code draws only from the environment's own generators (a search of
the pinned `ManiSkill2_real2sim` and `simpler_env` sources finds no use of NumPy's global generator outside a
context manager that restores it).

## 5. Rollout, termination, and success

The worker runs upstream's `run_maniskill2_eval_single_episode` loop:

1. `is_final_subtask = env.is_final_subtask()`, the instruction, and the first image after reset;
2. while neither the policy has terminated nor the time limit has truncated: one Policy action; if its
   terminate signal is set on a non-final subtask, `env.advance_to_next_subtask()` is called instead of
   terminating; `env.step(action[:7])`; then the instruction, `is_final_subtask`, and the image are read again;
3. the episode **succeeds when the last executed step returned `terminated`** (the task's success predicate at
   that step). The loop does not stop when the task first succeeds: a later step can undo it. Upstream names
   its video `success` or `failure` from the same value.

Each record carries the executed steps, whether the policy terminated and whether the time limit truncated,
the instructions, the number of subtask advances, the final `episode_stats`, a SHA-256 of the first policy
image, a SHA-256 over every `float64` action passed to `env.step`, and an 8-hex-digit digest of each step's
action (`step_action_digests`). The aggregate rejects a record outside
the work manifest, a duplicate, steps outside `[1, max_episode_steps]`, and an episode that ended neither by
policy termination nor by truncation at the step limit.

## 6. Aggregation

The official metric is the success rate per task, computed as `calc_metrics_evaluation_videos.py` does: each
variant's success rate, then their unweighted mean over the task's variants (for coke can, per orientation and
then over the three orientations). Every variant of a task has the same number of episodes, so this equals the
task's pooled success rate on a formal run. `official_result.tasks` holds the task rates, `subtasks` the coke
can orientations and the three drawers, and `variants` every variant. The WidowX suite also reports
upstream's partial success: the share of episodes whose final `episode_stats` has `is_src_obj_grasped`.

SimplerEnv does not define one number over tasks, and the kit reports none. A sharded run merges the per-task
counts; its task rate is the pooled rate, which equals the official one only when every shard set covers the
whole matrix.

## 7. Validation sequence

1. Build into a fresh directory with the committed script; the build's own check and `--self-test` must pass.
2. Run `--quick-test` with the real Policy.
3. Run the full suite and check that `official_result.formal` is true and every frozen episode has exactly one
   record.
4. Run SimplerEnv's own scripts with `examples/rt1/run_native.py` into a new output directory and pair the runs
   with `examples/rt1/compare_native.py --strict`, which fails unless both sides cover every frozen episode
   exactly once, every episode is paired, and every native record carries the observation camera, policy model
   and policy setup of its invocation.

## 8. Correctness baseline

The full three-suite comparison below comes from one RTX 4090 host; the WidowX suite was compared again at
commit `0af5b90` on an RTX 3080 host. Both used the RT-1-X checkpoint served by `examples/rt1/` and a runtime
built with the committed script. Commit hashes are those of
[pull request #88](https://github.com/MINT-SJTU/RoboScope-Kit/pull/88).

### Build, self-test, and Quick Test

A clean build of commit `932d85d` into a new directory finished in 60 s (warm uv cache, PyPI through a
mirror). The build's checks and `--self-test` passed.

`--quick-test` on `google_robot_visual_matching` runs the episode `pick_coke_can/horizontal/urdf_none/000`. It
succeeded in 47 steps on `932d85d`, and again on commit `8769edc` on 2026-11-10 with the same runtime: Policy
load 37.54 s, evaluation 14.83 s, sampled peak GPU memory 2,415 MiB.

### Native comparison

Measured between 2026-11-07 and 2026-11-10. The Kit ran commit `932d85d`, one suite at a time on an idle GPU,
with fixed `--concurrency` (K simulator Workers) and `--replicas` (G Policy replicas). The native side ran
`examples/rt1/run_native.py` on the same host, runtime, and checkpoint: SimplerEnv's unmodified
`main_inference.py` with the official scripts' RT-1-X command lines, `--jobs` invocations at once.
`examples/rt1/compare_native.py --strict` paired the two runs of each suite on variant, robot pose, and object
pose or episode id. It exited 0 for all three suites: every native invocation matched one frozen variant with
an identical episode list, both sides held exactly one record for every frozen episode, and every native
record carried the configuration of its invocation.

| Suite | Episodes paired | Kit / native successes | Different outcomes (Kit only / native only) | Same action trajectory | Same first image | Kit evaluation time | Native wall time |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `widowx_visual_matching` | 96 of 96 | 3 / 3 | 2 (1 / 1) | 73 | 96 | 195.7 s (K=4, G=2) | 231 s (4 jobs) |
| `google_robot_visual_matching` | 864 of 864 | 399 / 392 | 27 (17 / 10) | 642 | 864 | 3,815.3 s (K=4, G=2) | 3,821 s (6 jobs) |
| `google_robot_variant_aggregation` | 1,722 of 1,722 | 622 / 622 | 44 (22 / 22) | 1,324 | 1,722 | 7,267.5 s (K=8, G=3) | 6,955 s (8 jobs), then 1,210 s (1 job) for three invocations |

"Kit only" counts the pairs in which only the Kit episode succeeded. A pair has the same action trajectory
when the SHA-256 over every `float64` action passed to `env.step` is identical on both sides, and the same
first image when the SHA-256 of the first image sent to the Policy is.

- The first image is identical in every pair, so the robot pose, object pose, overlay, recolor settings, and
  camera reach the Policy as upstream renders them.
- Every pair with a different outcome also has a different action trajectory.
- Kit and native differ in 27 visual-matching outcomes; two runs of the same side differ in 22 (Kit) and 28
  (native), as the next section shows.
- At 8 jobs, three native variant-aggregation invocations failed with SAPIEN's `RuntimeError: cannot create
  buffer` (GPU memory). They were rerun with `--jobs 1`; the other 81 were reused after `run_native.py` checked
  their run identity, command line, and record digests.

Variant-aggregation success by task (mean of variant rates) in the runs above:

| Task | Kit | Native |
| --- | ---: | ---: |
| `pick_coke_can` | 48.9% | 48.9% |
| `move_near` | 32.5% | 31.9% |
| `open_drawer` | 8.5% | 7.9% |
| `close_drawer` | 50.8% | 52.9% |
| `place_apple_in_closed_top_drawer` | 12.7% | 12.7% |

### WidowX comparison at `0af5b90` on an RTX 3080

Measured on 2026-11-14 on one RTX 3080 (10 GB) host, commit `0af5b90`, with a runtime built from scratch with
the committed script (10 min 48 s, cold uv cache); `--self-test` passed. `--quick-test` on
`widowx_visual_matching` ran `put_spoon_on_tablecloth/base/000` to the end (60 steps, not a success): Policy load
36.87 s, evaluation 22.86 s. The full suite then ran on both sides, one after the other on the same idle GPU,
and `compare_native.py --strict` exited 0.

| Suite | Episodes paired | Kit / native successes | Different outcomes | Same action trajectory | Same first image | Kit evaluation time | Native wall time |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `widowx_visual_matching` | 96 of 96 | 3 / 3 | 0 | 84 | 96 | 434.4 s (K=3, G=1) | 549 s (2 jobs) |

Sampled peak GPU memory was 4.6 GB for the Kit run and 9.8 GB for the native run with 2 jobs.

### Run-to-run variation

Two runs of the same side can give different outcomes. The repeats below ran on the same host from 2026-11-06
with Kit commit `97ccf22`, whose SimplerEnv adapter, worker, and example files are those of `932d85d`. The
result files do not record the commit; `97ccf22` is what the checkout held during the runs. Each repeat ran
alone on the GPU with the settings of the first run (Kit K=4, G=2; native 3 jobs). The last four rows pair the
`932d85d` Kit runs of the table above with the `97ccf22` Kit runs of the same suite.

| Suite | Runs compared | Different outcomes |
| --- | --- | ---: |
| `google_robot_visual_matching` | Kit `97ccf22`, run 1 and run 2 | 22 of 864 |
| `google_robot_visual_matching` | Native, run 1 and run 2 | 28 of 864 |
| `google_robot_visual_matching` | Kit `932d85d` and Kit `97ccf22` run 1 | 32 of 864 |
| `google_robot_visual_matching` | Kit `932d85d` and Kit `97ccf22` run 2 | 26 of 864 |
| `widowx_visual_matching` | Kit `932d85d` and Kit `97ccf22` | 0 of 96 |
| `google_robot_variant_aggregation` | Kit `932d85d` and Kit `97ccf22` | 39 of 1,722 |

The first image is identical in all 864 pairs of both visual-matching repeats. The four Kit-and-native pairings
of those four runs differ in 24 to 28 outcomes.

### Resource use

Per-process sampling every 20 s during a Kit `google_robot_visual_matching` run on an RTX 3080 (Kit `0.8.0b28`,
`--concurrency 2 --replicas 1`, TensorFlow capped at 1,536 MB): the two Workers used 1.42 CPU cores together
and the Policy replica 0.68; peak resident memory was 1.96 GB per Worker and 4.07 GB for the replica; peak GPU
memory was 1.42 GB per Worker (ray-traced scenes) and 1.78 GB for the replica.

### Limits

- Each comparison pairs one run with one run. The counts show no difference beyond run-to-run variation; they
  do not establish statistical equivalence.
- Repeats at one commit cover visual matching only, with one repeat per side.
- The source of the variation was not isolated. In a separate probe, five passes on the same rendered frame
  produced different actions at steps 0 and 3.
- The Google Robot suites were compared at commit `932d85d` only. Since then the adapter's result flags and
  shard metadata, the place of the Vulkan check, and the build's lock check changed, and `main` was merged;
  the worker, the frozen matrix, and the example Policy did not change. At the current code the WidowX suite
  was compared again, on an RTX 3080.
- Ray-traced images depend on the GPU model. A Kit visual-matching run on another GPU, an RTX 3080, gave the
  same first image as the RTX 4090 in all 540 rasterized episodes and a different one in all 324 ray-traced
  episodes. Run the Kit and the native evaluation on the same host.
- The times are not a throughput comparison: the two sides ran with different parallelism, and the native side
  loads the model once per invocation.
- RT-1-X is the only model evaluated.

### Raw results and reproduction

A Kit run directory holds `result.json` and the per-episode records under `queue/completed/`. A native run
directory holds `run-identity.json`, `plan.jsonl`, `episodes/*.jsonl`, and `logs/`. `compare_native.py` writes
the comparison JSON. The records of the runs above have no public archive yet. Reproduce with the commands of
section 1 and of the
[RT-1-X example](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/examples/rt1/README.md).
