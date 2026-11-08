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
A layout the simulator abandons after a PhysX break or an unstable reset is evaluated again, up to three passes
per shard. A retry with fewer missing layouts than environments is filled with layouts already evaluated, whose
new results are discarded, so the simulator keeps the shard's environment count. Layouts still missing on the
last pass run on their own: `make_kong` layout 1 resets unstably with three environments every time but not
with one. Each episode records the `--num_envs` it ran with.

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
contract to preserve PyRUA-Lean's waiting, early finish and native verdict on one environment per Worker.
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

Draft for [#18](https://github.com/MINT-SJTU/RoboScope-Kit/issues/18). For `Simate_beta`, three environments
per Worker with the observation stride and a thread cap reach 1.95× and 2.04× in two complete runs, but a third
attempt ran out of GPU memory, so 2× is not yet reliable. A real-agent PyRUA comparison below measures four tasks;
the full declared task scope and the 2× target remain open for that workload.

### 2026-11-07: `Simate_beta` (compute-bound VLA)

- **Machine:** AutoDL, RTX 4090 48 GB, a 20-CPU cgroup quota on a host that reports 208 cores.
- **Workload:** the #17 scope, `arx_x5`, 42 tasks × 3 layouts = 126 episodes, the same layouts in every run.
- **Configurations:** the baseline is the #17 Kit default (autotune chose K=2, G=1) on 0.8.0b26 with JSON
  Policy messages, run with K=2 fixed. The other runs use 0.8.0b31 with MessagePack and G=1; the run names are
  batches. `stride` is `--observation-stride 10` ([XPolicyLab notes](../../examples/xpolicylab/README.md#skip-observations-the-model-never-reads));
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

Every run returned 126 unique episodes with no OOM; four runs re-evaluated one layout each after a PhysX break.
Head-camera videos have one frame per step plus one in every episode. Evaluation excludes setup and autotuning,
about 30 min in `auto` and in the baseline.

- **Successes:** every later run succeeded more often than the baseline (27–32 against 23). Paired by task
  and layout, the largest difference has p = 0.022 (exact McNemar, `g1` K=3), which is not significant after
  correcting for the seven comparisons; the others have p ≥ 0.19. #17 found the same layout can succeed in
  one run and fail in the next.
- **MessagePack:** at K=2 it gains 7%; most of the gain comes from K=3.
- **Stride:** `Simate_beta` reads the newest observation at the start of each 10-step chunk and history frames
  20 steps apart. Stride 10 gives identical actions and skips the other observations: 5% and 12% faster at K=3
  in the two repeats.
- **Policy threads:** PyTorch opens one thread per reported core, 104 here, and the 20-CPU quota throttles the
  whole container (throttled fraction 0.26–0.44). Four threads remove the throttling.
- **Limits:** each simulator takes 8.0–10.3 GB and the Policy 15.2 GB, so K=3 is the most that fits in 48 GB
  (peak 42.8 GB); K=4 would need up to 56 GB on the larger tasks. GPU utilization is already 0.77 at K=3, and
  the simulator's CPU side dominates each step (see [Profiling](#profiling)).
- **Autotune:** it measured K=1 and K=2, then doubled to K=4, where two simulators ran out of GPU memory. A
  simulator that runs out of memory waits in an error dialog instead of exiting, so the probe hit its 900 s
  timeout, the 30 min budget ran out, and autotune kept K=2 without trying K=3, about 20% slower than fixed K=3.
- **End to end:** policy load takes 6 s in every run. A fixed K=3 run needs no autotuning, so the #17 default's
  17,576 s, including its 30 min of autotuning, drops to about 9,900–10,100 s (1.74–1.77×). With autotune
  (`auto`), it takes 14,213 s (1.24×).
- **Conclusion:** 1.57–1.60× in formal throughput, short of the 2× target for this workload. Raising it further
  needs more environments per GPU without another 8–10 GB simulator each, as `envs_per_worker` does.

A separate same-version comparison on Kit `e4e5682` ran the same 126 task/layout pairs with one environment per
Worker. K=1/G=1 took 24,002.5 s (24 successes) and K=3/G=1 took 12,295.1 s (31 successes), a 1.95× elapsed-time
ratio. Eleven paired layouts had different verdicts. K=1 is not the autotuned K=2 baseline above, and one pair
of runs does not establish outcome equivalence or variance.

### 2026-11-08: `Simate_beta` with three environments per Worker

Same machine, scope and baseline as above. Kit 0.8.0b44 (measured as an earlier local build of the same retry
change) with MessagePack, K=3, G=1 and `envs_per_worker: 3`
(nine environments in three simulators). `make_kong` layout 1 resets unstably in every three-environment attempt,
so in every run it was evaluated on its own on the last retry pass; the other 125 episodes ran with three
environments.

| Run | Configuration | Evaluation | Episodes/h | Steps/s | Successes | Gain (episodes/h) |
|---|---|---:|---:|---:|---:|---:|
| `g0` | K=2, one environment, JSON (baseline) | 15,810 s | 28.69 | 5.63 | 23/126 | 1.00× |
| `m5` | 3 × 3 environments | 9,339 s | 48.57 | 9.30 | 29/126 | 1.69× |
| `m5` | 3 × 3 environments, stride, 4 threads | 8,094 s | 56.04 | 10.61 | 29/126 | 1.95× |
| `m7` | 3 × 3 environments, stride, 4 threads | 7,759 s | 58.46 | 11.20 | 28/126 | 2.04× |

- **Completeness:** each run returned 126 unique episodes, and every head-camera video has one frame per step
  plus one. A repeat between `m5` and `m7` stopped at 78 of 126 episodes when `play_tic_tac_toe` ran beside two
  `pour_*` tasks and a simulator ran out of GPU memory (`ERROR_OUT_OF_DEVICE_MEMORY`); without an external
  guard the run would have hung. Every run peaked at 48.5 GB of 48 GB, so this configuration has no headroom.
- **Stride and threads:** on top of three environments they add 15–20%. Without them the 20-CPU quota throttles
  the container 38% of the time and GPU utilization is 0.59; with them throttling is 0 and utilization 0.75.
- **Successes:** paired with the baseline, p = 0.21–0.27 (exact McNemar).
- **End to end:** the #17 default's 17,576 s, including 30 min of autotuning, drops to 7,764–8,099 s
  (2.17–2.26×); policy load takes 5 s.
- **Conclusion:** formal throughput is 1.95× and 2.04× in two complete runs (steps/s 1.88× and 1.99×), with
  one of three attempts lost to GPU memory. Reliable acceptance needs either GPU memory headroom, such as a
  smaller Policy footprint or two Workers with more environments each, or an OOM detector that requeues the
  shard.

### 2026-11-08: PyRUA-Lean with a remote model

On the same 48 GiB RTX 4090, [PyRUA-Lean](../../examples/pyrua/) at `c686120` used GPT-6.1 Terra with max
reasoning, Kit `e4e5682` and the PyRUA example at `73078f7`. Each run evaluated `align_blocks`, `deposit_coin`,
`insert_key` and `solve_equation`, layouts 0 and 1 at seed 1. K=G=1 and K=G=2 ran in the order shown, with one
environment per Worker. These are 8-episode samples, not the 126-episode scope from #17.

| Run | K/G | Evaluation | Episodes/h | Steps/s | Successes | Requests | Peak GPU (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline 1 | 1/1 | 5,278.3 s | 5.46 | 0.274 | 4/8 | 182 | 15,557 |
| optimized 1 | 2/2 | 2,971.3 s | 9.69 | 0.489 | 3/8 | 205 | 30,653 |
| optimized 2 | 2/2 | 2,971.6 s | 9.69 | 0.578 | 3/8 | 172 | 30,609 |
| baseline 2 | 1/1 | 5,976.1 s | 4.82 | 0.270 | 4/8 | 215 | 15,557 |

Formal episode-throughput ratios for the two paired rounds are 1.78× and 2.01×; dividing the combined baseline
time by the combined optimized time gives 1.89×. The 16 single-Worker episodes had 3,058 steps and 8 successes;
the 16 two-Worker episodes had 3,172 steps and 6 successes. Four paired layouts changed verdict. Model-generated
actions and trajectory lengths vary, so this sample neither establishes statistical equivalence nor attributes
the verdict differences to concurrency. The first optimized-2 attempt stopped after two episodes on a model-stream
failure and was repeated; the table excludes interrupted attempts and recovery time. Each completed run's native,
Kit and agent records and tool receipts were independently checked.

The model guard recorded a median 15.5–16.7 s per streamed request in the two single-Worker runs. Request
intervals totaled 3,425 s and 4,207 s, or 65% and 70% of evaluation time. This includes network and response
streaming, can overlap simulator/tool work, and is not a measurement of model compute alone. Two-Worker request
intervals overlap; their union covered 88% of evaluation time in each run. These measurements motivate overlapping
independent episodes without changing PyRUA's observations or actions.

### 2026-11-07: latency-bound Policy (synthetic)

[`examples/latency-policy/`](../../examples/latency-policy/README.md) sleeps 3 s per `act()` and returns 10
actions. It characterizes scheduling only; acceptance for #18 needs a real agent.

| K = G | Episodes/h | Gain | Peak GPU memory |
|---:|---:|---:|---:|
| 1 | 38.4 | 1.00× | 8.2 GB |
| 4 | 148.3 | 3.86× | 32.5 GB |
| 5 | 180.1 | 4.69× | — |
| 6 | 214.9 | 5.60× | 48.4 GB |

Six step-limited episodes per Worker on one task. Across all 42 tasks at K=5, two runs each completed 42/42
episodes (43.6 and 44.1 episodes/h) with a 44.6 GB peak. With 0.5 s, 1 s and 10 s per call, K=5 gains 4.22×, 4.48×
and 4.88× over K=1.

Autotune chose K=1, G=1 for this Policy. It adds Workers before replicas (`bound_hint` is `gpu_infer`), and K=2
sharing one replica serves the same number of requests per second as K=1, so its probe took twice as long and
timed out at 900 s. It never tried K=G. Its K=1 evaluation of the 42 tasks reached 33 episodes in about 12,400 s
(about 9.6 episodes/h) before a 4 h guard stopped it. Those are not all 42 tasks, so K=5's gain on the full mix,
about 4.6×, is approximate.
