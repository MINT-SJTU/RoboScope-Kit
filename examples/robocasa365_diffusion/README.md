# Official RoboCasa365 Diffusion Policy

This example serves the officially published `pretrain_human300` multi-task
Diffusion Policy through the RoboScope Kit lifecycle. It serves the independent
**`robocasa365`** benchmark. The existing `robocasa/tabletop` suite uses GR1,
29-dimensional actions and a different model interface.

The checkpoint was trained on RoboCasa365 human demonstrations across 300 tasks.
It is linked in the [official multi-task documentation](https://github.com/robocasa/robocasa/blob/3443cdf3dd20f9ae8ed41adba442059b506891d0/docs/benchmarking/multitask_learning.md).
This example performs no training or model inference optimization. Its
full-horizon comparison with the official evaluator is in the
[benchmark README](../../benchmarks/robocasa365/README.md#native-comparison).

## Pinned inputs

| Input | Revision |
| --- | --- |
| Diffusion Policy source | `41212698b6f481ed92a55f0d5f1778ec56bea417` |
| RoboCasa robomimic fork (`robocasa-dp`) | `7dc18461c4fc497095f1044f4c40cd44fff62e6a` |
| Official checkpoint repository | `c484448aba1a9b60a04c9b0ca117241518ea69f3` |
| `openai/clip-vit-large-patch14` | `32bd64288804d66eefd0ccbe215aa642df71cc41` |

`model-inputs.json` records source archive and weight hashes, checkpoint size and
download locations. The installer uses canonical commit-pinned codeload archives,
without requiring Git or inheriting host Git mirror rewrites. It verifies archives
before extraction, writes `.roboscope-source.json` provenance, and checks existing
source files against the archive on repeated builds. An existing unverified source
directory is rejected rather than overwritten. The downloader and `load()` verify
both weight files. The checkpoint is about
1.70 GB; the full pinned CLIP weights are about 1.71 GB. Allow additional space
for caches and CUDA wheels. The official inference guidance recommends at least
8 GB GPU memory; five simulator slots need additional memory.

Use Linux x86_64, Python 3.11 and a CUDA GPU. Keep the inference and simulator
environments separate. The official model needs the RoboCasa robomimic fork's
language-conditioned visual encoder, which ARISE robomimic does not provide.
The installer uses `--no-deps` for that source because its package metadata
specifies older training dependencies. Direct inference dependencies are in `requirements.txt`; `requirements-lock.txt`
freezes the complete validated inference dependency closure used by the installer.
The pinned Diffusion Policy tree uses implicit namespace packages while its
`setup.py` uses `find_packages()`. The installer therefore writes a `.pth` entry
for the verified source root into the inference environment and checks the actual
model imports with Python's isolated mode. Imports do not depend on the working
directory or `PYTHONPATH`.

## Install and download

From the repository root, with `uv` available:

```bash
bash examples/robocasa365_diffusion/install-policy.sh /sim/robocasa365-policy

HF_HOME=/sim/robocasa365-policy/huggingface \
  /sim/robocasa365-policy/env/bin/python \
  examples/robocasa365_diffusion/download_model.py /models/robocasa365-diffusion
```

Set `ROBOSCOPE_POLICY_PYTHON` to an existing Python 3.11 executable to avoid
downloading an interpreter. `ROBOSCOPE_PYPI_INDEX` selects a reachable package
index; versions remain pinned. The CUDA wheel index is selected explicitly.
Set `ROBOSCOPE_POLICY_SOURCE_ARCHIVES_DIR` to keep the two verified source archives
in persistent storage. Pre-seed that directory with
`diffusion_policy-<revision>.tar.gz` and `robomimic-<revision>.tar.gz` when the build
host cannot access codeload directly. The same committed hashes are required for
cached and downloaded archives. `install_sources.py` can also run directly with
any Python 3.11 interpreter before installing GPU dependencies.

This creates `model.ckpt` and a local `clip/` encoder directory. Pass the model
directory as `--checkpoint-source`. `load()` uses the provided
`SessionContext.checkpoint_path`; it also accepts a checkpoint file with `clip/`
as its sibling. The MetaWorld StarVLA checkpoint is incompatible and is rejected.

Replace paths in `roboscope-policy.json` with your absolute Python, repository
and working-directory paths. In an EvoStudio adaptation container,
`/workspace/user_policy.py` can symlink to this example's `policy.py`, preserving
access to its adjacent metadata and downloader module. Set that entrypoint in
`/workspace/evostudio-policy.json`.

The installer installs the local Kit package; reinstall it after modifying Kit.
Build the simulator separately with `benchmarks/robocasa365/build-runtime.sh`.

## Observation and action semantics

The checkpoint config is authoritative. The Policy verifies its exact feature
shapes, delta actions, two observation frames and eight-action chunk. It selects
`ema_model` when the checkpoint requests EMA and loads the full state dictionary,
including observation/action normalization. Precision, 100 diffusion sampling
steps and trained model parameters are unchanged.

| Worker feature | Model input |
| --- | --- |
| `images.agentview_right` | `robot0_agentview_right_image` |
| `images.agentview_left` | `robot0_agentview_left_image` |
| `images.wrist` | `robot0_eye_in_hand_image` |
| `state.end_effector_position_relative` | `robot0_base_to_eef_pos` |
| `state.end_effector_rotation_relative` | `robot0_base_to_eef_quat` |
| `state.gripper_qpos` | `robot0_gripper_qpos` |
| `language.instruction` | `lang_emb` (CPU CLIP projection) |

Images arrive as `(2, 256, 256, 3)` uint8. The native gym wrapper already flipped
them vertically. The Policy applies robomimic `process_frame` to convert to CHW
float32 and divide by 255. State histories have float32 shape `(2, 3)`, `(2, 4)`
and `(2, 2)`. History contains the **last two consecutive simulation steps**;
the initial frame is duplicated. Previous inference-boundary frames would be
incorrect inputs. Temporal history is owned and reset by the worker.

The CPU language encoder uses the pinned local CLIP weights, upstream
`padding="max_length"` tokenization and 768-dimensional `text_embeds` projection.
It creates one embedding per episode, matching the fork's `LangEncoder` without
a network download during evaluation.

Native `predict_action` normalizes observations, samples diffusion and
denormalizes actions. The Policy returns its eight 12-dimensional actions as
float32 without clipping or scheduler changes. The worker applies native action
conversion and scoring.

`reset()` clears per-episode language state. The checkpoint's native model reset
is a no-op because it has no recurrent hidden state. Each task owns a torch RNG
stream initialized from checkpoint training seed 42. Batches within a task use
that stream in native slot order; interleaved tasks cannot consume one another's
diffusion noise. The worker's first-slot/first-episode reset restarts the stream
for each new workload or retry, so autotune probes cannot consume formal-run
noise. CLIP initialization and encoding do not consume these streams.

## Native comparison

The scripts below compare full rollouts with the pinned upstream
`eval_robocasa.py`, at the official task horizons and with videos off on both
sides. `rollout_trace.py` wraps the RoboCasa Gym environment in every simulator
process of both paths; it records per-slot SHA-256 digests of the initial scene
and state, and every eight physical steps of the Policy features, executed
actions and MuJoCo integration state. It changes no reset, action or score.

Run the official evaluator for each task, in the inference environment with
`native-requirements.txt` installed and an unpacked `gym==0.21.0` source tree
(the version in upstream `conda_environment.yaml`; it is pure Python):

```bash
HF_HOME=/models/huggingface HF_HUB_OFFLINE=1 PYTHONHASHSEED=42 \
/sim/robocasa365-policy/env/bin/python examples/robocasa365_diffusion/run_native.py \
  --task CloseBlenderLid --simulator /sim/robocasa365 --gym-source /sim/gym-0.21.0 \
  --diffusion-policy /sim/robocasa365-policy/diffusion_policy \
  --checkpoint /models/robocasa365-diffusion/model.ckpt \
  --output /results/native-a/CloseBlenderLid
```

Run the Kit CLI with the same tracing (all arguments pass to
`python -m roboscope_kit.runtime.benchmark`):

```bash
PYTHONHASHSEED=42 ROBOCASA365_TRACE_DIR=/results/kit/trace \
python examples/robocasa365_diffusion/run_kit.py \
  --benchmark robocasa365 --suite target50 --episodes 5 \
  --benchmark-manifest /sim/robocasa365/benchmark-manifest.json \
  --policy-config /workspace/robocasa365-policy.json \
  --checkpoint-source /models/robocasa365-diffusion \
  --concurrency 4 --replicas 1 --cuda-mps --output-dir /results/kit/result
```

Then pair two native passes with one or more Kit runs:

```bash
python examples/robocasa365_diffusion/compare_rollouts.py \
  --native /results/native-a /results/native-b --kit serial=/results/kit \
  --output /results/comparison.json
```

`compare_rollouts.py` rejects missing, duplicate or unknown episodes and any
disagreement between an arm's score and its trace. Its docstring fixes the
tolerance: identical initial scenes and states for every pair, Kit/native
outcome disagreements no more frequent than native/native ones, and no
direction among them. It reports per-task successes and steps and each pair's
first divergence.

`compare_native.py` is a narrower check that needs no simulator: it feeds a
captured first Policy batch (`evidence/worker-*/<task>-first-batch.npz` of any
Kit run) through the upstream workspace loader and observation wrapper, and
requires every prepared feature and the first action chunk to match exactly.

```bash
python examples/robocasa365_diffusion/compare_native.py \
  --checkpoint /models/robocasa365-diffusion \
  --batch /results/kit/result/evidence/worker-000/CloseBlenderLid-first-batch.npz \
  --output /results/first-batch-parity.json
```
