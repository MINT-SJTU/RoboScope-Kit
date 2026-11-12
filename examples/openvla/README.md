# OpenVLA on LIBERO with RoboScope Kit

This example implements the public Kit `Policy` lifecycle for the four suite-specific OpenVLA-7B
LIBERO checkpoints. The SDK owns Policy processes, simulator Workers, scheduling, recovery, and scoring.
The example does not support the base OpenVLA checkpoint, `libero_90`, LIBERO-Plus, or LIBERO-Pro.

## Build the Policy environment and standard LIBERO Worker

Use Linux x86_64 with an NVIDIA GPU supporting BF16 and FlashAttention2, a compatible
NVIDIA driver, `uv`, `curl`, `git`, and conda or micromamba. Build into two separate
directories from this checkout:

```bash
bash examples/openvla/build-policy-env.sh /absolute/openvla-policy
KIT_PYTHON=/absolute/openvla-policy/env/bin/python \
  bash benchmarks/libero/build-runtime.sh /absolute/libero-runtime
```

The Policy builder installs CPython 3.11.15, the SDK, and hash-locked inference
dependencies: torch 2.2.0, Transformers 4.40.1, TensorFlow 2.15.0, NumPy 1.24.4,
and checksum-pinned FlashAttention 2.5.5. It checks imports and CPU preprocessing,
and writes `build-manifest.json`; it does not load a checkpoint or build a simulator.
Use a new Policy directory when moving from the former combined OpenVLA runtime.

The standard LIBERO builder owns the simulator, source checkout, manifest, and
self-test. Its existing pins remain CPython 3.10.20, robosuite 1.4.0, MuJoCo 3.2.3,
NumPy 1.24.4, and LIBERO revision `8f1084e3132a39270c3a13ebe37270a43ece2a01`.
The benchmark CLI and Policy run in the Policy environment; the manifest starts
Workers in `/absolute/libero-runtime/env`. No OpenVLA-specific simulator is built.

Optional inputs: `OPENVLA_INDEX_URL` and `DOWNLOAD_CACHE` configure the Policy
package index and verified FlashAttention wheel cache; `LIBERO_URL` selects a
source mirror for the standard simulator builder. Change environments through
these scripts and their pin files.

After building, validate the separated layout with the self-test and a bounded
real-policy evaluation. Historical results below used a different Worker stack
and do not validate current full-suite results.

## Prepare a checkpoint and the model-code cache

Use the Hugging Face `hf` CLI from a separate tooling environment; the pinned inference environment
uses its own older `huggingface-hub` dependency. Select the checkpoint matching the requested suite:

| Suite | Hugging Face repository | Revision |
| --- | --- | --- |
| `libero_spatial` | `openvla/openvla-7b-finetuned-libero-spatial` | `962318cec55ac10993ff0f5f43eda9a270b4c873` |
| `libero_object` | `openvla/openvla-7b-finetuned-libero-object` | `287d6cfdf12d07b1449505f66d9bf3550257e9b3` |
| `libero_goal` | `openvla/openvla-7b-finetuned-libero-goal` | `fa5ae1e7509348889295bba8e08621d8b55e9baf` |
| `libero_10` | `openvla/openvla-7b-finetuned-libero-10` | `80970322773f81baa2e22fe495d0487b93a05cfa` |

For Spatial:

```bash
export HF_HOME=/absolute/local-disk/openvla-hf-cache
hf download openvla/openvla-7b-finetuned-libero-spatial \
  --revision 962318cec55ac10993ff0f5f43eda9a270b4c873 \
  --local-dir /absolute/local-disk/models/openvla/spatial

# Cache the external Python implementation; base-model weights are not needed.
hf download openvla/openvla-7b \
  --revision 47a0ec7fc4ec123775a391911046cf33cf9ed83f --include '*.py'
```

Keep checkpoints and `HF_HOME` on local disk rather than network storage to avoid
slow checkpoint loading. Keep the same `HF_HOME` when running the evaluation. The
Policy loads only local files and pins the external model/configuration/image-processor code to `47a0ec7fc4ec123775a391911046cf33cf9ed83f`.
Each checkpoint directory must retain its weights, tokenizer, processor configuration, and
`dataset_statistics.json`. The statistics must contain the selected suite's key.
The processor is assembled from explicitly pinned components because Transformers 4.40.1's nested
processor loaders do not consistently forward `code_revision`.

## Evaluate through the SDK

Copy [`roboscope-policy.json`](roboscope-policy.json) and replace its absolute placeholder paths.
Its command uses the Policy environment's Python and this example's `openvla_policy.py`;
`cwd` points to the example directory. Keep those paths fixed during a run.

Run a bounded smoke evaluation in a new output directory. The standard manifest
retains its 300-second readiness timeout:

```bash
export HF_HOME=/absolute/local-disk/openvla-hf-cache
export OPENVLA_SEED=7
/absolute/openvla-policy/env/bin/python -m roboscope_kit.runtime.benchmark \
  --benchmark libero --suite libero_spatial \
  --checkpoint-source /absolute/local-disk/models/openvla/spatial \
  --benchmark-manifest /absolute/libero-runtime/benchmark-manifest.json \
  --policy-config /absolute/roboscope-policy.json \
  --quick-test --output-dir /absolute/new-smoke-run
```

For a fixed evaluation, replace `--quick-test` with `--episodes 5 --concurrency 2 --replicas 2`.
This evaluates initial states 0–4 for each of ten tasks (50 episodes): partial
initial-state coverage, five of the 50 states per task. Acceptance of this reduced
workload requires maintainer approval; it is not the full 500-episode suite.
Omit both fixed concurrency flags to use SDK Autotune. To evaluate another suite, change both
`--suite` and the suite-specific checkpoint path. Each run needs a new output directory.
CUDA MPS and profiling are optional SDK features and are off unless explicitly requested.

The runtime produces `result.json`, `metrics.json`, progress, terminal episode records, and process logs.
A successful smoke run has no execution errors and produces a terminal episode; task success itself
is a model outcome rather than a prerequisite for correct execution.

## Policy behavior

- `load` loads one checkpoint per replica, enables model evaluation mode, and declares the supported
  suite, `images.primary`, `language.instruction`, batch size 1, and action chunk size 1.
- `reset` registers the episode ID. Separate IDs allow one replica to serve multiple Workers without
  mixing lifecycle bookkeeping; the model does not keep recurrent state between requests.
- `act` processes the primary image and instruction, calls `predict_action` under `torch.inference_mode`,
  and returns one finite float32 action in the SDK's `libero.delta-eef-v1` representation.
- `end_episode` releases episode bookkeeping; `unload` releases model and processor references.

The image pipeline follows the OpenVLA LIBERO helpers at source revision
`c8f03f48af692657d3060c19588038c7220e9af9`: Kit already rotates the camera by 180 degrees, then the
Policy applies JPEG encoding/decoding, TensorFlow Lanczos resize to 224x224, and a 90%-area center crop.
TensorFlow preprocessing is explicitly placed on CPU, matching the native evaluator; the Policy
disables TensorFlow GPU visibility before initialization. PyTorch model inference remains on GPU.
Inference uses BF16, FlashAttention2, a lowercase instruction prompt, greedy decoding, suite-specific
action unnormalization, and the OpenVLA gripper binarization/sign conversion.

`OPENVLA_SEED` defaults to 7 and controls the model process RNGs. The simulator seed is configured by
the benchmark manifest. One inference call returns one control step, not a multi-step action sequence.
Multiple Workers may share a replica; requests to each standard Policy server execute serially.
Increasing `--replicas` creates independent model processes rather than a batch of observations.

## Experiment results

These historical evaluations cover all ten tasks per suite but only initial-state
indices 0–4: 50 episodes per suite, with partial coverage of the 50 states per task. Kit 2/2 means two Workers and two Policy replicas on one GPU; Native runs serially.

| Suite | Native successes | Kit 2/2 successes | Native formal time (min) | Kit 2/2 formal time (min) |
| --- | --- | --- | --- | --- |
| Spatial | 43/50 (86%) | 40/50 (80%) | 22.96 | 19.30 |
| Object | 42/50 (84%) | 44/50 (88%) | 28.66 | 23.83 |
| Goal | 33/50 (66%) | 39/50 (78%) | 29.11 | 21.73 |
| Long (`libero_10`) | 28/50 (56%) | 23/50 (46%) | 64.20 | 59.35 |
| Total | 146/200 (73%) | 146/200 (73%) | 144.93 | 124.21 |

Formal time excludes checkpoint loading and other setup. These measurements used Kit `0.8.0b33`, source commit
`b432786`, on November 9–10, 2026. Each path used one NVIDIA GeForce RTX 4090 reporting
49,140 MiB; Native used driver `595.71.05` and Kit 2/2 used `580.105.08` on a different GPU.
The CPU model was not recorded for these full evaluations.
The paired outcomes disagree on 36 episodes, so equal totals do not establish per-episode agreement
or statistical equivalence. These historical measurements used the combined robosuite 1.4.1 runtime and Kit GPU
preprocessing; they validate neither the current CPU Policy nor the standard
robosuite 1.4.0 Worker layout. They are not a controlled speedup claim.

Durable external evidence links and current-code results are pending.
