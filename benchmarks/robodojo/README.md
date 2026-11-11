# RoboDojo simulation

`build-runtime.sh <install-dir>` builds the pinned Linux x86-64 simulator, downloads the benchmark assets,
and writes `benchmark-manifest.json` under the install directory. It requires `git`, `curl`, `sha256sum`,
the Debian system packages `cmake`, `build-essential`, and `ffmpeg`, a working CUDA build toolchain, and a
compatible NVIDIA GPU. The builder installs an isolated Miniforge, runs RoboDojo's `scripts/install.sh -i`,
then applies the Kit patches and checks the resulting environment. An existing `$HOME/miniconda3` is left
alone. The asset download is approximately 41 GB. The first simulator launch on an account also downloads
Isaac Sim extensions from NVIDIA into `~/.local/share/ov`, and can sit silent for minutes. Isaac Sim is
licensed by NVIDIA; read its
[license](https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html)
and set `OMNI_KIT_ACCEPT_EULA=YES` yourself before building or running it. The build does not set
`PRIVACY_CONSENT`.

The `arx_x5` suite contains the 42 base simulation tasks at the pinned RoboDojo revision.
`arx_x5_randomized` contains the 12 additional randomized variants. The manifest records each task's
upstream default of 25 or 50 layouts and uses seed 1. `--episodes` can request a smaller common prefix;
requests above an upstream task limit are rejected. Work is divided into non-overlapping fixed-layout
shards. A shard missing a layout or returning incomplete results fails rather than reporting a partial run.

The simulator abandons a layout after a PhysX break or an unstable reset. The Worker evaluates such a layout
again, up to three passes per shard. The experimental manifest default `envs_per_worker` (1 to 10, default 1)
runs that many environments in one simulator, one Kit episode each. A retry keeps that environment count: when
fewer layouts are missing than environments, it fills the batch with layouts already evaluated and discards
their new results. The last pass runs only the layouts still missing, so they can run with fewer environments:
`make_kong` layout 1 resets unstably with three environments every time but not with one. Each episode records
the environment count it ran with as `evaluator_num_envs`.

Isaac Sim does not exit when Vulkan runs out of GPU memory; it waits in an error dialog. The Worker watches the
native log for `ERROR_OUT_OF_DEVICE_MEMORY`, stops the simulator and reports an OOM, so the Kit lowers the GPU's
concurrency and requeues the shard.

The Policy receives `images.head`, `images.left_wrist`, and `images.right_wrist` as RGB uint8 arrays,
`language.instruction` as a string, and `state.joints` as 14 float32 values ordered as six left-arm joints,
left gripper, six right-arm joints, right gripper. `state.ee_pose` contains both end-effector poses as
`[x, y, z, qw, qx, qy, qz]` in the environment frame. The default `robodojo.x5-joint-v1` action uses the
joint-state order for absolute targets and gripper commands. The `robodojo.x5-ee-compact-quat-v1` action
instead has `[x, y, z, qx, qy, qz, gripper]` for each arm; the missing quaternion `w` is reconstructed as
nonnegative. A Policy that needs every intermediate observation can declare
`step_observation_updates=("robodojo",)` and implement `observe(Observation)`: `act()` receives the
first observation of a chunk, and `observe()` receives the intermediate observations in step order.

The manifest also exposes the experimental `async_video`, `unpaced_render`, and `control_state_reuse`
switches, plus `physics_threads` (1, 4 or 8). They are disabled by default. `observation_queue_size` defaults
to 0 (synchronous) and can be set to 1–10 for bounded asynchronous observation delivery when the Policy
declares step updates. Pending updates finish before the next `act()` or episode end. These candidates
originate in @QLL-CCDD's RoboDojo work and should be evaluated for correctness and throughput for the
intended workload.

The simulator source, IsaacLab, CuRobo, XPolicyLab, package constraints, and the asset dataset are pinned
by the build script. Its submodule selections keep the upstream installer's `--remote` updates at those
tested revisions.
Model code and checkpoints are supplied separately through the standard Kit Policy configuration;
[`examples/xpolicylab/`](../../examples/xpolicylab/README.md) serves XPolicyLab policies this way.
The experimental [`examples/pyrua/`](../../examples/pyrua/README.md) uses the interactive Policy
contract to preserve PyRUA-Lean's waiting, early finish and native verdict on one environment per Worker;
it also provides an optional local SAM3 service shared across Policy replicas.
The layout-sharding patch is adapted from @QLL-CCDD's RoboDojo work.

The asset downloader verifies the pinned inventory and each file's hash. `HF_ENDPOINT` selects the metadata
and primary download host; `ROBODOJO_ASSET_TRANSPORT=modelscope` uses ModelScope as a first byte source
with the same pinned hashes. `ROBODOJO_ASSET_INVENTORY` may point to pre-fetched inventory metadata when
the metadata service is unavailable, and `ROBODOJO_ASSET_WORKERS` controls download concurrency.

For a space-constrained machine with an existing pinned RoboDojo runtime, `build-runtime.sh` can
create a separate overlay environment with `ROBODOJO_BASE_RUNTIME=/path/to/base`. Set
`ROBODOJO_ASSET_SOURCE` to a read-only copy of the pinned assets, `ROBODOJO_ASSET_INVENTORY` to
the pinned inventory, and `ROBODOJO_WHEELHOUSE` to an offline directory containing a Kit wheel
built from this source and wheels for any dependencies missing from the base. The builder verifies
source revisions, checks the wheel against the staged SDK code, installs into the new environment,
and hashes every asset. Generated robot configuration files stay in the new runtime.

This mode reuses the base environment's installed dependencies and CuRobo source: it does not make
a self-contained runtime. Keep the base unchanged while evaluating, and retain the generated
`base-provenance.json` for comparisons. The default full build remains available when an independent
environment is required.

## Profiling

Profiled with the Kit's `--profile`; how to run and read a capture is in the
[profiling notes](../../packages/roboscope-kit/docs/profiling.md).

### 2026-11-05: simulator Worker with a placeholder Policy

- **Machine:** AutoDL, RTX 4090 48 GB, driver 580.105.08, 208 cores; Nsight Systems 2026.5.1.
- **Configuration:** RoboDojo `arx_x5` with a placeholder Policy that returns the same action every step
  and takes almost no time, so the run measures mostly the simulator's own cost. Kit 0.8.0b16 development build
  of `--profile` (PR #55).
- **Results:**

| Run | Worker | `episode` | `policy.act` | GPU busy | Share |
|---|---|---:|---:|---:|---:|
| `--quick-test` | 000 | 19.0 s | 20 calls, 0.8 s | 2.2 s | 12% |
| K=3, g=1 | 000 | 110.2 s | 105 calls, 3.8 s | 11.2 s | 10% |
| K=3, g=1 | 001 | 19.7 s | 20 calls, 0.8 s | 2.2 s | 11% |
| K=3, g=1 | 002 | 101.7 s | 105 calls, 3.9 s | 11.0 s | 11% |

- **Conclusion:** the captured Worker's GPU kernel intervals cover about 10–12% of its episode and Policy
  round trips about 4%. The Worker capture excludes the separate Policy Server and GPU graphics pipeline;
  it does not partition the remaining time among CPU work, synchronization and waiting. These figures
  describe the placeholder workload, not full-device utilization with a real Policy.
- **Next steps:** enable CPU sampling in the Kit's `nsys` arguments (currently `--sample=none`) to find where the
  CPU time goes, and profile again with a real Policy to include its cost.

## Throughput

[#18](https://github.com/MINT-SJTU/RoboScope-Kit/issues/18) sets a target of at least 2× the evaluation
throughput of the #17 example, separately for a compute-bound Policy (`Simate_beta`) and for a latency-bound
agent (PyRUA-Lean). `Simate_beta` averages 1.96× over four runs on the #17 scope, just below the target.
PyRUA-Lean with four Workers and a shared SAM3 service reaches 3.16× on a four-task sample; its full task scope
has not been measured.

Throughput is episodes per hour of evaluation time, which excludes setup and autotuning. A speed-up is the ratio
of two throughputs unless its section says otherwise. K is the number of Workers, one simulator each, and G the
number of Policy replicas. The sections below follow the order of this table.

| Policy | Configuration | Baseline | Speed-up | Outcomes | Details |
|---|---|---|---|---|---|
| `Simate_beta` | K=3, stride 10, 4 Policy threads | K=2, JSON, Kit 0.8.0b26 | 1.57× and 1.60× | 28 and 29 of 126 successes against 23 (paired p ≥ 0.19) | [one environment per Worker](#2026-11-07-simate_beta-with-one-environment-per-worker) |
| `Simate_beta` | K=3 | K=1, same Kit commit | 1.95×, one pair of runs | 11 of 126 layouts differ in verdict | [one environment per Worker](#2026-11-07-simate_beta-with-one-environment-per-worker) |
| `Simate_beta` | K=3 × 3 environments, stride 10, 4 Policy threads | K=2, JSON, Kit 0.8.0b26 | 1.96×, mean of four runs (1.89–2.04×) | 26–31 of 126 successes against 23 (paired p ≥ 0.12) | [three environments per Worker](#2026-11-08-simate_beta-with-three-environments-per-worker) |
| PyRUA-Lean, GPT-6.1 Terra | K=G=2 | K=G=1 | 1.78× and 2.01× | 4 paired layouts changed verdict | [two Workers](#2026-11-08-pyrua-lean-with-two-workers) |
| PyRUA-Lean, GPT-6.1 Terra | K=G=4, shared SAM3 | K=G=1, local SAM3 | 3.16× | 2 of 8 layouts differ in each round, as between repeats | [shared SAM3](#2026-11-08-pyrua-lean-with-shared-sam3) |
| PyRUA-Lean, scripted agent | K=G=4, shared SAM3, `cover_blocks` | two native single-Worker runs | 3.59× | 24 of 24 verdicts match | [one table per Worker](#2026-11-10-pyrua-lean-with-one-table-per-worker) |
| PyRUA-Lean, scripted agent | K=G=5, shared SAM3, `cover_blocks` | two native single-Worker runs | 4.18× | 24 of 24 verdicts match | [one table per Worker](#2026-11-10-pyrua-lean-with-one-table-per-worker) |
| PyRUA-Lean, GPT-6.1 Terra | K=G=4, shared SAM3, `cover_blocks` | no paired run | estimated 3.39× | 48 of 48 successes | [one table per Worker](#2026-11-10-pyrua-lean-with-one-table-per-worker) |
| PyRUA-Lean, GPT-6.1 Terra | 2 processes × 24 tiled tables, `cover_blocks` | four single-table Workers | 2.05×, partial run | camera images differ from native | [one table per Worker](#2026-11-10-pyrua-lean-with-one-table-per-worker) |
| Latency Policy (synthetic) | K=G=5 | K=G=1 | 4.69× | not applicable: the Policy holds still | [latency-bound Policy](#2026-11-07-latency-bound-policy-synthetic) |

### 2026-11-07: `Simate_beta` with one environment per Worker

Measured on an AutoDL RTX 4090 48 GB with a 20-CPU cgroup quota on a host that reports 208 cores. The workload
is the #17 scope: `arx_x5`, 42 tasks × 3 layouts = 126 episodes, the same layouts in every run. The baseline is
the #17 Kit default on Kit 0.8.0b26 with JSON Policy messages: autotune chose K=2, G=1 there, and the baseline
run fixes K=2. The other runs use this branch at `155ff86` (a development build labelled 0.8.0b31, which is not
the published 0.8.0b31) with MessagePack and G=1. `stride` is `--observation-stride 10`
([XPolicyLab notes](../../examples/xpolicylab/README.md#skip-observations-the-model-never-reads)), and
`4 threads` sets `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4` for the Policy process.

| Run | Configuration | Evaluation | Episodes/h | Steps/s | Successes | Gain (episodes/h) |
|---|---|---:|---:|---:|---:|---:|
| `g0` | K=2, JSON (baseline) | 15,810 s | 28.69 | 5.63 | 23/126 | 1.00× |
| `g1` | K=2 | 14,837 s | 30.57 | 5.93 | 28/126 | 1.07× |
| `g1` | K=3, stride | 11,216 s | 40.44 | 7.71 | 32/126 | 1.41× |
| `h1` | K=3, 4 threads | 11,054 s | 41.03 | 8.05 | 28/126 | 1.43× |
| `h2` | K=3, 4 threads | 10,597 s | 42.80 | 8.19 | 27/126 | 1.49× |
| `h1` | K=3, stride, 4 threads | 9,897 s | 45.83 | 8.84 | 28/126 | 1.60× |
| `h2` | K=3, stride, 4 threads | 10,075 s | 45.02 | 8.60 | 29/126 | 1.57× |
| `auto` | autotune (chose K=2), stride, 4 threads | 12,409 s | 36.55 | 6.83 | 30/126 | 1.27× |

- **Result:** K=3 with the stride and four threads reaches 1.57× and 1.60×, short of the 2× target. End to end,
  the #17 default takes 17,576 s, including its 30 min of autotuning. A fixed K=3 run needs no autotuning and
  takes about 9,900–10,100 s (1.74–1.77×). Policy load takes 6 s in every run.
- **Stride and threads:** stride 10 gives `Simate_beta` identical actions. Without the thread cap, PyTorch opens
  104 threads and the 20-CPU quota throttles the whole container (throttled fraction 0.26–0.44).
- **Outcomes:** every run returned 126 unique episodes with no OOM, and each head-camera video has one frame per
  step plus one. Every later run succeeded more often than the baseline (27–32 against 23). Paired by task and
  layout, the largest difference has p = 0.022 (exact McNemar, `g1` K=3), which is not significant after
  correcting for the seven comparisons; the others have p ≥ 0.19. #17 found that the same layout can succeed in
  one run and fail in the next.
- **Autotune:** `auto` kept K=2, about 20% slower than fixed K=3; end to end it takes 14,213 s (1.24×). Its K=4
  probe ran out of GPU memory and waited until the 900 s timeout, so the 30 min budget ran out before K=3 was
  tried. The run predates the OOM detection.
- **Limits:** each simulator takes 8.0–10.3 GB of GPU memory and the Policy 15.2 GB, so K=3 is the most that
  fits in 48 GB (peak 42.8 GB). GPU utilization is 0.77 at K=3. The baseline differs from the other runs in Kit
  version and message encoding.

On one Kit commit, `e4e5682`, with the same 126 task/layout pairs and one environment per Worker, K=1/G=1 took
24,002.5 s (24 successes) and K=3/G=1 took 12,295.1 s (31 successes): a 1.95× elapsed-time ratio. Eleven paired
layouts had different verdicts. The baseline here is K=1, not the autotuned K=2 above, and one pair of runs
establishes neither outcome equivalence nor variance.

### 2026-11-08: `Simate_beta` with three environments per Worker

Same machine, scope and baseline as above. Each run uses MessagePack, G=1 and `envs_per_worker: 3`, so K=3 runs
nine environments in three simulators. The `m` runs used a local build of the retry change (`dcc5da2`); the `o`
runs used `10b3497`, which adds the OOM handling.

| Run | Configuration | Evaluation | Episodes/h | Steps/s | Successes | Gain (episodes/h) |
|---|---|---:|---:|---:|---:|---:|
| `g0` | K=2, one environment, JSON (baseline) | 15,810 s | 28.69 | 5.63 | 23/126 | 1.00× |
| `m5` | K=3 | 9,339 s | 48.57 | 9.30 | 29/126 | 1.69× |
| `m5` | K=3, stride, 4 threads | 8,094 s | 56.04 | 10.61 | 29/126 | 1.95× |
| `m7` | K=3, stride, 4 threads | 7,759 s | 58.46 | 11.20 | 28/126 | 2.04× |
| `o2` | K=3, stride, 4 threads | 8,107 s | 55.95 | 10.65 | 31/126 | 1.95× |
| `o3` | K=3, stride, 4 threads | 8,375 s | 54.16 | 10.50 | 26/126 | 1.89× |
| `o1` | K=4 → 3 after an OOM, stride, 4 threads | 8,385 s | 54.09 | 10.40 | 29/126 | 1.89× |

- **Result:** the four K=3 runs with the stride and four threads average 56.15 episodes/h, 1.96× (1.89–2.04×;
  steps/s 1.91×): close to the 2× target but below it. End to end, the #17 default's 17,576 s, including 30 min
  of autotuning, drops to 7,764–8,380 s (2.10–2.26×); policy load takes 5–6 s.
- **Stride and threads:** without them the 20-CPU quota throttles the container 38% of the time and GPU
  utilization is 0.59; with them throttling is 0 and utilization 0.75.
- **Outcomes:** each listed run returned 126 unique episodes, and every head-camera video has one frame per step
  plus one. Successes paired with the baseline give p ≥ 0.12 (exact McNemar).
- **Limits:** GPU memory has no headroom. Every run peaked at 48.5 GB on the 48 GB card, and one more run of
  this configuration, before the OOM handling, ran out of memory when `play_tic_tac_toe` ran beside two `pour_*`
  tasks. With the OOM handling the Kit recovers: `o1` started four Workers, two ran out of memory, and the Kit
  retired one Worker, requeued three shards and finished at K=3. `make_kong` layout 1 resets unstably in every
  three-environment attempt, so every run evaluated it on its own on the last retry pass; the cause is open.
  The baseline differs from these runs in Kit version and message encoding.

### 2026-11-08: PyRUA-Lean with two Workers

On the same RTX 4090, [PyRUA-Lean](../../examples/pyrua/README.md) at `c686120` used GPT-6.1 Terra with max
reasoning, Kit `e4e5682` and the PyRUA example at `73078f7`. Each run evaluated `align_blocks`, `deposit_coin`,
`insert_key` and `solve_equation`, layouts 0 and 1 at seed 1: 8 episodes, with one environment per Worker. The
runs are listed in the order they ran.

| Run | K/G | Evaluation | Episodes/h | Steps/s | Successes | Requests | Peak GPU (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline 1 | 1/1 | 5,278.3 s | 5.46 | 0.274 | 4/8 | 182 | 15,557 |
| optimized 1 | 2/2 | 2,971.3 s | 9.69 | 0.489 | 3/8 | 205 | 30,653 |
| optimized 2 | 2/2 | 2,971.6 s | 9.69 | 0.578 | 3/8 | 172 | 30,609 |
| baseline 2 | 1/1 | 5,976.1 s | 4.82 | 0.270 | 4/8 | 215 | 15,557 |

- **Result:** the two paired rounds give 1.78× and 2.01×. Dividing the combined baseline time by the combined
  optimized time gives 1.89×.
- **Outcomes:** the 16 single-Worker episodes had 3,058 steps and 8 successes; the 16 two-Worker episodes had
  3,172 steps and 6 successes. Four of the 16 paired layouts changed verdict. The two single-Worker runs differ
  from each other on 4 of 8 layouts, and the two two-Worker runs on none.
- **Where the time goes:** in the two single-Worker runs, a streamed model request took a median 15.5–16.7 s,
  and the request intervals totaled 3,425 s and 4,207 s, 65% and 70% of evaluation time. They include network
  and response streaming and can overlap simulator and tool work. With two Workers, the union of the request
  intervals covered 88% of evaluation time in each run.
- **Limits:** these are 8-episode samples from four tasks, not the 126-episode scope from #17. The model's
  actions and trajectory lengths vary from run to run, so the sample neither establishes statistical
  equivalence nor attributes the verdict differences to concurrency. The table excludes interrupted attempts
  and recovery time.

### 2026-11-08: PyRUA-Lean with shared SAM3

These runs used the same RTX 4090, PyRUA-Lean `c686120`, GPT-6.1 Terra at max reasoning, Kit `e4e5682`, the
`73078f7` Policy and the eight episodes above. The baseline is one Worker with a local SAM3 model. The
optimized configuration is four Workers, four Policy replicas and one loopback SAM3 service from PyRUA-Lean
`5238f76`, whose `sam3_local.py` is byte-identical to the pinned local version. Both use the same checkpoint
and storage path. The runs are listed in the order they ran.

| Run | Workers / SAM3 | Successes | Steps | Kit wall | End-to-end | Peak GPU | Model requests |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline 1 | 1 / local | 3/8 | 1,503 | 5,830.0 s | 5,832.2 s | 15,557 MiB | 195 |
| optimized 1 | 4 / shared | 3/8 | 1,620 | 2,013.3 s | 2,032.3 s | 41,247 MiB | 192 |
| optimized 2 | 4 / shared | 5/8 | 1,365 | 1,426.2 s | 1,446.3 s | 41,279 MiB | 170 |
| baseline 2 | 1 / local | 5/8 | 1,455 | 5,026.7 s | 5,029.3 s | 15,685 MiB | 170 |

The next table samples the same runs. GPU samples were taken about every 30 seconds: they are snapshots, not
time-weighted utilization. Request stream time includes network and streaming and can overlap across the four
agents.

| Run | Request stream total | GPU zero samples / samples | Sampled GPU p95 |
|---|---:|---:|---:|
| baseline 1 | 4,117.3 s | 173/189 | 15.8% |
| optimized 1 | 4,018.0 s | 53/69 | 60.0% |
| optimized 2 | 3,019.9 s | 34/47 | 31.7% |
| baseline 2 | 3,387.2 s | 141/163 | 27.5% |

- **Result:** the two single-Worker evaluation phases totaled 10,855.4 s for 16 episodes (5.31 episodes/h,
  0.272 steps/s), and the two four-Worker phases 3,438.6 s (16.75 episodes/h, 0.868 steps/s): 3.16×. The
  evaluation phase excludes shared-service startup. End to end, the paired speed-ups are 2.87× and 3.48×, and
  the combined times give 3.12×. End-to-end time starts after input checks and includes service startup and
  evaluator execution.
- **Outcomes:** the verdict differed on 2/8 paired layouts in each round. Repeating one configuration also
  changed 2/8 baseline and 2/8 optimized verdicts. The shared service matched local segmentation for 16
  text/point queries on eight retained camera images, including 16 concurrent shared queries.
- **Limits:** this is a four-task sample of 8 episodes per run, so it establishes neither episode-by-episode
  equivalence nor a full-suite success rate. The speed-up combines more concurrency with one shared model copy;
  it does not measure a SAM3 speed-up. The measured wrapper differs from the committed
  `pyrua-shared-sam3-wrapper.py` in import order and whitespace only.

### 2026-11-10: PyRUA-Lean with one table per Worker

A table is one simulated environment: `--num_envs` sets how many tables one Isaac Sim process holds. On the same
RTX 4090, these runs used Kit `e4e5682` with its unmodified RoboDojo runtime, K=G, one shared SAM3 service as
above, seed 1 and the task `cover_blocks`.

A scripted agent replayed a fixed tool program per layout without model calls, on layouts 0–23. Native A and B
are two single-Worker repeats; their differences give the native run-to-run spread, and gains are against their
mean.

| Run | K=G | Evaluation | Episodes/h | Gain | Peak GPU | Same steps as A / B | Same verdict | 30/37/43-step episodes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| native A | 1 | 2,412.6 s | 35.8 | — | 14,729 MiB | — / 17 | 24/24 | 13/3/8 |
| native B | 1 | 2,497.5 s | 34.6 | — | 14,697 MiB | 17 / — | 24/24 | 13/4/7 |
| 4 Workers | 4 | 684.6 s | 126.2 | 3.59× | 38,803 MiB | 21 / 20 | 24/24 | 13/2/9 |
| 5 Workers | 5 | 587.5 s | 147.1 | 4.18× | 46,895 MiB | 18 / 17 | 24/24 | 13/5/6 |

- **Result:** four single-table Workers run 3.59× faster than one and agree with native as closely as native
  agrees with itself; all 24 tool sequences matched in every pair. Use K=4 on a 48 GB card: K=5 leaves about
  2 GB of headroom.
- **Real agent:** with GPT-6.1 Terra at max reasoning, K=4 evaluated layouts 0–47 in 5,585.6 s: 48/48 successes,
  30.9 episodes/h, 17,823 steps (3.19 steps/s) and 721 model requests, with a 38,803 MiB peak. Episodes took
  326–494 s (median 387 s); a single-Worker `cover_blocks` episode took 366 s. Summed episode time is 3.39 times
  the evaluation time. No single-Worker run of these 48 layouts exists, so this gain is an estimate.
- **Several tables per process:** `num_envs` > 1 is faster but changes the camera images. RoboDojo renders each
  camera type for all tables as one tiled render product, and with the scripted agent, 11/24 episodes at 24
  tables ended at 44 steps, which native never produced. Rendering each camera separately removes most of the
  difference, but 3/24 episodes at 8 tables still ended at 44 steps, and 24 tables (72 render products) fail to
  allocate render resources. Physics state matched native to about 1e-8 in both cases.
- **Tiled tables, speed only:** two processes × 24 tiled tables ran the same 48 layouts with GPT-6.1 Terra on
  another RTX 4090 48 GB (same 580.105.08 driver, video off, idle tables asleep). A tool-transport error on one
  agent's `finish` call stopped the run 1,987 s after launch. By then, 35 episodes had ended normally, all
  successful, after 1,415–1,824 s each; 13 were cut short. Over the 1,987 s, that is 63.4 completed episodes/h,
  2.05× the four single-table Workers measured the same way (48 episodes in 5,590.6 s). Each episode took about
  4.6 times as long, and the run used 1,653 model requests, about 34 per episode against 15.
- **Limits:** every run here is one task. The tiled figure is a partial-run rate, not a bound on full
  48-episode throughput: the 13 cut-short episodes may have needed longer.

### 2026-11-07: latency-bound Policy (synthetic)

[`examples/latency-policy/`](../../examples/latency-policy/README.md) sleeps 3 s per `act()` and returns 10
actions. It characterizes scheduling only; acceptance for #18 needs a real agent. Measured on the same machine
and development build (`155ff86`) as the `Simate_beta` runs, with six step-limited episodes per Worker on one
task.

| K = G | Episodes/h | Gain | Peak GPU memory |
|---:|---:|---:|---:|
| 1 | 38.4 | 1.00× | 8.2 GB |
| 4 | 148.3 | 3.86× | 32.5 GB |
| 5 | 180.1 | 4.69× | — |
| 6 | 214.9 | 5.60× | 48.4 GB |

- **Other workloads:** across all 42 tasks at K=5, two runs each completed 42/42 episodes (43.6 and 44.1
  episodes/h) with a 44.6 GB peak. With 0.5 s, 1 s and 10 s per call, K=5 gains 4.22×, 4.48× and 4.88× over K=1.
- **Autotune:** it chose K=1, G=1 for this Policy and never tried K=G. It adds Workers before replicas
  (`bound_hint` is `gpu_infer`), and K=2 sharing one replica serves the same number of requests per second as
  K=1, so its probe took twice as long and timed out at 900 s.
- **Limits:** autotune's K=1 evaluation of the 42 tasks reached 33 episodes in about 12,400 s (about 9.6
  episodes/h) before a 4 h guard stopped it. Those are not all 42 tasks, so K=5's gain on the full mix, about
  4.6×, is approximate.
