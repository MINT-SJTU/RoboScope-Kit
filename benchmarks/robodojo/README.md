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
left gripper, six right-arm joints, right gripper. `state.ee_pose` contains both end-effector poses as `[x, y,
z, qw, qx, qy, qz]` in the environment frame. The default `robodojo.x5-joint-v1` action uses the joint-state
order for absolute targets and gripper commands. The `robodojo.x5-ee-compact-quat-v1` action instead has `[x,
y, z, qx, qy, qz, gripper]` for each arm; the missing quaternion `w` is reconstructed as nonnegative. The
cameras render 480 × 640; a Policy can request any `(height, width)` from 64 to 2048 pixels per side for each
camera with `PolicyCapabilities.image_sizes`. A camera with a negotiated size renders natively at that size
with its own lens and pose, and is neither cropped nor resized. The build's `camera-sizes.patch` passes these
sizes to RoboDojo's camera templates and changes only each camera's `resolution`, so the horizontal field of
view stays the camera's own and the vertical one follows the aspect ratio, as RoboDojo's camera manager
derives its intrinsics. A size whose aspect ratio is not 4:3 therefore shows more or less of the scene
vertically: the Gemini head camera at 224 × 224 sees about 96° vertically instead of about 80°. Saved
videos pad odd sizes by one pixel. Negotiated sizes need roboscope-kit 0.8.0b83 or later in the
Policy environment. Runtimes built before this patch have no `negotiated_camera_sizes` in their installed
manifest, so the Kit offers them 480 × 640 only; rebuild the runtime to use other sizes. Without `image_sizes`
the cameras, observations and videos are unchanged. A Policy that needs every intermediate observation can
declare `step_observation_updates=("robodojo",)` and implement `observe(Observation)`: `act()` receives the
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
