# RT-1-X Policy for SimplerEnv

A `roboscope-kit` `Policy` that serves the released RT-1-X TF-Agents checkpoint
(`rt_1_x_tf_trained_for_002272480_step`, Open X-Embodiment) on all three suites of the `simpler_env`
benchmark: Google Robot visual matching, Google Robot variant aggregation, and WidowX (Bridge) visual
matching. It runs SimplerEnv's own RT-1 inference code. The benchmark contract is the
[SimplerEnv protocol](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/packages/roboscope-kit/SIMPLER_ENV_EVALUATION_PROTOCOL.md).

| File | Purpose |
| --- | --- |
| `rt1_policy.py` | The Policy: SimplerEnv's `RT1Inference` with its state kept per episode |
| `assets.py` | The pinned checkpoint and Universal Sentence Encoder: URL, size, SHA-256 of the archive and of every loaded file |
| `download_assets.py` | Downloads both archives, verifies them, and extracts them |
| `run_native.py` | Runs SimplerEnv's official RT-1-X scripts with their unmodified `main_inference.py` and logs every episode |
| `native_hooks.py` | The read-only logging hooks `run_native.py` installs in `main_inference.py` |
| `compare_native.py` | Checks the native invocations against the frozen Kit variants and pairs the episodes |
| `roboscope-policy.json` | Policy process contract template |

## What the Policy keeps from upstream

Every model-specific step is SimplerEnv's `simpler_env.policies.rt1.rt1_model.RT1Inference` at the pinned
commit, called the way `maniskill2_evaluator.py` calls it:

- `RT1Inference.reset(instruction)` before an episode's first action and `step(image, instruction)` for each
  action; `step` resets the recurrent TF-Agents state itself when the instruction changes (the long-horizon
  drawer task advances its instruction);
- image `resize_with_pad` to 320x256, the Universal Sentence Encoder embedding, and the policy's own action
  post-processing: on Google Robot the small gripper-action filter, axis-angle rotation, and the raw gripper
  value; on WidowX the unnormalized translation and rotation, RPY to axis-angle, and the inverted, binarized
  gripper. The episode's `policy_setup` (`google_robot` or `widowx_bridge`, from the official script's
  `--policy-setup`) picks the setup;
- TensorFlow's GPU memory cap, as `main_inference.py --tf-memory-limit` sets it
  (`ROBOSCOPE_RT1_TF_MEMORY_MB`, default 3072 MB as upstream).

One replica serves many concurrent episodes. `RT1Inference` keeps one episode's state on the instance (its
observation buffer, TF-Agents policy state, instruction, and embedding), so the Policy stores those fields per
episode and swaps them in around each upstream call. The `widowx_bridge` instance runs upstream's `__init__`
too, but reuses the SavedModel and encoder the first instance loaded. The returned action is the float64 vector
`main_inference.py` passes to `env.step` (`world_vector`, `rot_axangle`, `gripper`) plus
`terminate_episode[0]`.

## Run

1. Build the SimplerEnv runtime; it also holds the pinned TensorFlow, TF-Agents, and SimplerEnv this Policy
   uses. On a host with any Vulkan ICD besides `nvidia_icd.json` (the protocol explains why), export
   `VK_ICD_FILENAMES=<dir>/nvidia_icd.json` first, for the build and every run below:

   ```bash
   benchmarks/simpler_env/build-runtime.sh /sim/simpler-env
   ```

2. Download the checkpoint (775 MB zip) and the Universal Sentence Encoder (550 MB). Neither ships a license
   file of its own; check the Open X-Embodiment and TF Hub terms before you use them:

   ```bash
   python examples/rt1/download_assets.py /models/rt1
   ```

3. Copy `roboscope-policy.json` and replace the paths. The command runs the runtime's own Python with the kit
   source on `PYTHONPATH`, so the Policy uses exactly the TensorFlow and SimplerEnv of the native comparison and
   nothing is installed into the runtime.

4. Evaluate with the benchmark CLI from any Python 3.11+ that has `roboscope-kit` installed:

   ```bash
   CUDA_VISIBLE_DEVICES=0 python -m roboscope_kit.runtime.benchmark \
       --benchmark simpler_env --suite google_robot_visual_matching --quick-test \
       --checkpoint-source /models/rt1 \
       --benchmark-manifest /sim/simpler-env/benchmark-manifest.json \
       --policy-config /path/to/roboscope-policy.json \
       --output-dir /runs/rt1x-quick
   ```

   Drop `--quick-test` for the full suite; add `--concurrency K --replicas G` to skip autotuning. The other
   suites are `google_robot_variant_aggregation` and `widowx_visual_matching`.

5. Run SimplerEnv's own scripts on the same suite and compare:

   ```bash
   python examples/rt1/run_native.py --runtime /sim/simpler-env --checkpoint-source /models/rt1 \
       --suite google_robot_visual_matching --output /runs/native-vm --jobs 2
   PYTHONPATH=packages/roboscope-kit/src /sim/simpler-env/env/bin/python examples/rt1/compare_native.py \
       --suite google_robot_visual_matching --kit-run /runs/rt1x-vm --native-run /runs/native-vm \
       --output /runs/compare-vm.json --strict
   ```

   `run_native.py` first executes the official scripts with a `python` shim that only records their
   `main_inference.py` command lines, keeps the RT-1-X invocations (the scripts sweep four RT-1 checkpoints),
   and drops the two alternative-camera variants that SimplerEnv's own aggregation leaves out. It then runs each
   kept command line unchanged, `--jobs` at a time, from a directory that links the runtime's checkout so the
   scripts' relative paths resolve. Videos and per-episode action plots are skipped unless `--write-videos` is
   given; they do not feed back into the episode.

   An output directory belongs to one native run: `run-identity.json` records the suite, runtime, SimplerEnv
   commit, checkpoint digests, TensorFlow memory limit and planned invocations, and a re-run with a different
   identity is refused. A finished invocation is reused only when its command line, output settings, and
   episode records still match; otherwise it runs again.

   `compare_native.py` exits 1 on protocol or integrity failures (duplicated or unknown episodes, native
   records that contradict their invocation). `--strict` is the acceptance mode and also requires complete
   coverage of the suite on both sides, every episode paired, and observation/policy fields on every native
   record; without it a partial run is reported with status 0.
