# OpenVLA on LIBERO with RoboScope Kit

This example implements the public Kit `Policy` lifecycle for the four suite-specific OpenVLA-7B
LIBERO checkpoints. The SDK owns Policy processes, simulator Workers, scheduling, recovery, and scoring.
The example does not support the base OpenVLA checkpoint, `libero_90`, LIBERO-Plus, or LIBERO-Pro.

## Build the runtime

Use Linux x86_64 with an NVIDIA GPU supporting BF16 and FlashAttention2, a compatible NVIDIA driver,
`uv`, and `curl`. From this repository:

```bash
bash benchmarks/libero/build-runtime.sh --openvla-kit /absolute/openvla-kit-runtime
```

The builder installs Python 3.11.15, the SDK, and hash-locked dependencies: torch 2.2.0,
Transformers 4.40.1, TensorFlow 2.15.0, FlashAttention 2.5.5, robosuite 1.4.1, MuJoCo 3.2.3,
and NumPy 1.24.4. The LIBERO source is pinned to `8f1084e3132a39270c3a13ebe37270a43ece2a01`;
`openvla-kit/libero-source.json` and `materialize_source.py` verify every source file.
The builder writes a benchmark manifest and runs the SDK environment self-test. The self-test
checks the simulator runtime; it does not load an OpenVLA checkpoint or run an episode.

Optional build inputs:

- `LIBERO_SOURCE_CACHE`: an existing LIBERO tree; only files matching the pinned inventory are reused.
- `DOWNLOAD_CACHE`: directory containing the checksum-verified FlashAttention wheel.
- `OPENVLA_INDEX_URL`: Python package index; package versions and hashes remain fixed.

Make environment changes through this builder and its pin files, rather than patching an installed runtime.

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
export HF_HOME=/absolute/openvla-hf-cache
hf download openvla/openvla-7b-finetuned-libero-spatial \
  --revision 962318cec55ac10993ff0f5f43eda9a270b4c873 \
  --local-dir /absolute/models/openvla/spatial

# Cache the external Python implementation; base-model weights are not needed.
hf download openvla/openvla-7b \
  --revision 47a0ec7fc4ec123775a391911046cf33cf9ed83f --include '*.py'
```

Keep the same `HF_HOME` when running the evaluation. The Policy loads only local files and pins the
external model/configuration/image-processor code to `47a0ec7fc4ec123775a391911046cf33cf9ed83f`.
Each checkpoint directory must retain its weights, tokenizer, processor configuration, and
`dataset_statistics.json`. The statistics must contain the selected suite's key.
The processor is assembled from explicitly pinned components because Transformers 4.40.1's nested
processor loaders do not consistently forward `code_revision`.

## Evaluate through the SDK

Copy [`roboscope-policy.json`](roboscope-policy.json) and replace its absolute placeholder paths.
The command must use the built runtime's Python and this example's `openvla_policy.py`; `cwd` must
point to the example directory. Keep the repository and runtime at those paths while the run is active.

Run a bounded smoke evaluation in a new output directory:

```bash
export HF_HOME=/absolute/openvla-hf-cache
export OPENVLA_SEED=7
export TF_FORCE_GPU_ALLOW_GROWTH=true
/absolute/openvla-kit-runtime/env/bin/python -m roboscope_kit.runtime.benchmark \
  --benchmark libero --suite libero_spatial \
  --checkpoint-source /absolute/models/openvla/spatial \
  --benchmark-manifest /absolute/openvla-kit-runtime/benchmark-manifest.json \
  --policy-config /absolute/roboscope-policy.json \
  --quick-test --output-dir /absolute/new-smoke-run
```

For a fixed evaluation, replace `--quick-test` with `--episodes 5 --concurrency 2 --replicas 2`.
This evaluates five initial states for each of ten tasks in the selected suite (50 episodes).
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
Inference uses BF16, FlashAttention2, a lowercase instruction prompt, greedy decoding, suite-specific
action unnormalization, and the OpenVLA gripper binarization/sign conversion.

`OPENVLA_SEED` defaults to 7 and controls the model process RNGs. The simulator seed is configured by
the benchmark manifest. One inference call returns one control step, not a multi-step action sequence.
Multiple Workers may share a replica; requests to each standard Policy server execute serially.
Increasing `--replicas` creates independent model processes rather than a batch of observations.

## Validation evidence

See [historical experiment records](EVIDENCE.md) for runtime versions, hardware,
exact commands, raw outcomes and logs. These old-revision runs do not validate
the current PR head; paired divergence and repeatability checks remain pending.
