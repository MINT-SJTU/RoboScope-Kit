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
The layout-sharding patch is adapted from @QLL-CCDD's RoboDojo work.

The asset downloader verifies the pinned inventory and each file's hash. `HF_ENDPOINT` selects the metadata
and primary download host; `ROBODOJO_ASSET_TRANSPORT=modelscope` uses ModelScope as a first byte source
with the same pinned hashes. `ROBODOJO_ASSET_INVENTORY` may point to pre-fetched inventory metadata when
the metadata service is unavailable, and `ROBODOJO_ASSET_WORKERS` controls download concurrency.

## Profiling

Profiled with the Kit's `--profile`; how to run and read a capture is in the
[profiling notes](../../packages/roboscope-kit/docs/profiling.md).

### 2026-11-05: `arx_x5` keeps the GPU busy only about 11% of the time

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

- **Conclusion:** GPU kernels take about 11% of an episode and Policy round trips about 4%. The rest is
  spent outside the GPU: the simulator's CPU work and the synchronization and waiting between kernels. To
  raise RoboDojo throughput, work on the simulator side; more GPU work will not help. This matches the
  earlier manual `nsys` measurement of about 11%.
- **Next steps:** enable CPU sampling in the Kit's `nsys` arguments (currently `--sample=none`) to find where the
  CPU time goes, and profile again with a real Policy to include its cost.

## Throughput

Draft for [#18](https://github.com/MINT-SJTU/RoboScope-Kit/issues/18); repeats, the autotune comparison and the
MessagePack-only ablation are still running, and the numbers below may change.

### 2026-11-07: `Simate_beta` (compute-bound VLA)

- **Machine:** AutoDL, RTX 4090 48 GB, a 20-CPU cgroup quota on a host that reports 208 cores.
- **Workload:** the #17 scope, `arx_x5`, 42 tasks × 3 layouts = 126 episodes, the same layouts in every run.
- **Configurations:** the baseline is the #17 Kit default (autotune chose K=2, G=1) on 0.8.0b26 with JSON
  Policy messages. The other runs use 0.8.0b31 with MessagePack, K=3, G=1. `stride` is
  `--observation-stride 10` ([XPolicyLab notes](../../examples/xpolicylab/README.md#skip-observations-the-model-never-reads));
  `4 threads` sets `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4` for the Policy process.

| Run | Configuration | Evaluation | Episodes/h | Steps/s | Successes | Gain (episodes/h) |
|---|---|---:|---:|---:|---:|---:|
| `g0` | K=2, JSON (baseline) | 15,810 s | 28.69 | 5.63 | 23/126 | 1.00× |
| `g1` | K=3, stride | 11,216 s | 40.44 | 7.71 | 32/126 | 1.41× |
| `h1` | K=3, 4 threads | 11,054 s | 41.03 | 8.05 | 28/126 | 1.43× |
| `h1` | K=3, stride, 4 threads | 9,897 s | 45.83 | 8.84 | 28/126 | 1.60× |

Every run returned 126 unique episodes with no OOM; `g1` and the first `h1` re-evaluated one layout after a PhysX
break. Evaluation excludes setup and the baseline's 30 min of autotuning.

- **Stride:** `Simate_beta` reads the newest observation at the start of each 10-step chunk and history frames
  20 steps apart. Stride 10 gives identical actions and skips the other observations, about 12% at K=3.
- **Policy threads:** PyTorch opens one thread per reported core, 104 here, and the 20-CPU quota throttles the
  whole container (throttled fraction 0.35–0.44). Four threads remove the throttling.
- **Limits:** each simulator takes 8.0–10.3 GB and the Policy 15.2 GB, so K=3 is the most that fits in 48 GB
  (peak 42.8 GB); K=4 would need up to 56 GB on the larger tasks. GPU utilization is already 0.77 at K=3, and
  the simulator's CPU side dominates each step (see [Profiling](#profiling)).
- **Conclusion so far:** about 1.6× in formal throughput, short of the 2× target for this workload.

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
episodes (43.6 episodes/h) with a 44.6 GB peak. With 0.5 s, 1 s and 10 s per call, K=5 gains 4.22×, 4.48×
and 4.88× over K=1.
