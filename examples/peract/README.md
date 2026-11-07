# PerAct Policy

A `roboscope-kit` `Policy` that serves the released [PerAct](https://github.com/peract/peract) 600k multi-task
checkpoint ("PerAct - 2048 Latents", ID `seed0`) on the `rlbench` benchmark's `peract_18` suite: the 18 RLBench
tasks of the PerAct paper with their released test episodes. The benchmark contract is the
[RLBench protocol](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/packages/roboscope-kit/RLBENCH_EVALUATION_PROTOCOL.md).

| File | Purpose |
| --- | --- |
| `peract_policy.py` | The Policy: builds PerAct's agent from the checkpoint's `config.yaml` and calls its own `act` |
| `assets.py` | The pinned checkpoint and CLIP RN50 files: URL, size, SHA-256 |
| `download_assets.py` | Downloads the checkpoint and CLIP RN50 and verifies their SHA-256 |
| `run_native.py` | Runs PerAct's own `eval.py` on the RLBench runtime and logs every episode |
| `native_hooks.py` | The read-only logging hooks `run_native.py` installs in `eval.py`'s evaluation process |
| `compare_native.py` | Pairs a Kit run with a native run episode by episode |
| `roboscope-policy.json` | Policy process contract template |

## What the Policy keeps from upstream

The model-specific parts of PerAct's `eval.py` stay in this Policy, not in the kit:

- the agent is PerAct's `peract_bc.launch_utils.create_agent(config)` with `build(training=False)` on
  `cuda:0` (cuDNN benchmarking on, as `eval.py`'s `get_device`) and `load_weights` of the 600k weights;
  CLIP RN50 is read from the pinned local file instead of `~/.cache/clip`;
- the observation becomes the dict YARR's `RolloutGenerator` feeds the agent: per camera the channel-first
  RGB (`uint8`) and point cloud (`float32`) with the camera's extrinsics and intrinsics, the 4-value
  `low_dim_state` of PerAct's environment wrapper (gripper open, the two finger positions clipped to
  `[0, 0.04]`, and the time feature `(1 - i / 24) * 2 - 1`), and the OpenAI CLIP tokens of the instruction,
  each with batch and time dimensions of one. The time feature uses `i = max(t - 1, 0)` before action `t`
  (counting from 0), as `eval.py` computes it: PerAct's wrapper extracts the observation after a step
  before incrementing its step counter, so the agent sees `i = 0` at both its first and second action.
  Using `t` instead shifts every later key frame by one (section 9 of the protocol);
- voxelization, the Perceiver, and the arg-max over translation, rotation, gripper, and collision classes
  are PerAct's own `act(..., deterministic=True)`.

The Policy returns PerAct's 9-value action as `float64`, exactly the array `eval.py` passes to RLBench: the
key-frame gripper position and unit quaternion `(x, y, z, qx, qy, qz, qw)`, gripper open, and ignore
collisions. PerAct's agent keeps no state between steps, so one replica serves concurrent episodes. `act` runs
under `torch.no_grad()`, which `eval.py` does not use; it changes no value, only the memory autograd would
keep.

## Run

1. Build the RLBench runtime; it also holds the pinned PerAct source and torch this Policy uses:

   ```bash
   benchmarks/rlbench/build-runtime.sh --accept-licenses /sim/rlbench
   ```

   **Passing `--accept-licenses` means that you accept the licences of CoppeliaSim Edu (free only for
   educational entities and non-commercial use) and RLBench (non-commercial, internal or academic research
   only).** Read them first: [CoppeliaSim licensing](https://www.coppeliarobotics.com/helpFiles/en/licensing.htm),
   [RLBench licence](https://github.com/stepjam/RLBench/blob/master/LICENSE). Without the flag the build stops
   before downloading anything. The build also needs an Xvfb server on `PATH` or in `XVFB` (Ubuntu: `xvfb`).

2. Download the checkpoint (124 MB, Apache-2.0 like the PerAct source) and CLIP RN50 (256 MB, MIT):

   ```bash
   python examples/peract/download_assets.py /models/peract
   ```

3. Copy `roboscope-policy.json` and replace the paths. The command runs the runtime's own Python with the kit
   source on `PYTHONPATH`, so the Policy uses exactly the PerAct and torch of the native comparison and nothing
   is installed into the runtime. `PERACT_ROOT` is the runtime's PerAct checkout; `LD_LIBRARY_PATH` lets
   PerAct's helpers import PyRep.

4. Evaluate with the benchmark CLI from any Python 3.11+ that has `roboscope-kit` installed:

   ```bash
   CUDA_VISIBLE_DEVICES=0 python -m roboscope_kit.runtime.benchmark \
       --benchmark rlbench --suite peract_18 --quick-test \
       --checkpoint-source /models/peract \
       --benchmark-manifest /sim/rlbench/benchmark-manifest.json \
       --policy-config /path/to/roboscope-policy.json \
       --output-dir /runs/peract-quick
   ```

   Drop `--quick-test` for the full 18 x 25 evaluation; add `--concurrency K --replicas G` to skip autotuning.
   RLBench does not support cross-instance sharding until the Policy and checkpoint identity can be verified
   across shards.

5. Run PerAct's own `eval.py` on the same test episodes and compare:

   ```bash
   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/path/to/RoboScope-Kit/packages/roboscope-kit/src \
       /sim/rlbench/env/bin/python examples/peract/run_native.py --runtime /sim/rlbench \
       --checkpoint-source /models/peract --output /runs/peract-native
   python examples/peract/compare_native.py --kit-run /runs/peract-full \
       --native /runs/peract-native --output /runs/peract-compare.json
   ```

   `run_native.py` runs all 18 tasks in one process, as PerAct's README does; `--task` selects a subset.
   It reports success only after the episode identities and per-task success rates agree with PerAct's CSV;
   an incomplete native child run fails. The comparison requires the full declared workload and rejects
   missing or mismatched variation, instruction or stored-state digest; older native logs without per-episode
   state digests cannot pass this check.
