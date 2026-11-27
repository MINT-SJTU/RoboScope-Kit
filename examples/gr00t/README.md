# GR00T N1.7 BEHAVIOR Policy

A `roboscope-kit` `Policy` that serves
[`kmy17518/gr00t-n1.7-b1k-multitask`](https://huggingface.co/kmy17518/gr00t-n1.7-b1k-multitask), a GR00T N1.7
fine-tune on all 100 tasks of the 2026 BEHAVIOR Challenge, on the `behavior` benchmark's `challenge_2026`
suite. The benchmark contract is the
[BEHAVIOR protocol](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/packages/roboscope-kit/BEHAVIOR_EVALUATION_PROTOCOL.md).

It is the only public checkpoint found (November 2026) that covers all 100 tasks with the 2026 evaluator's
native server. Its card reports no score. Other public BEHAVIOR checkpoints cover the 2025 profile's 50
tasks (Comet's π0.5 `pi05-b1kpt50`, the 2025 winners' solution) with that year's observation layout, or a
single task (the official 2026 baselines on `turning_on_radio`).

| File | Purpose |
| --- | --- |
| `gr00t_behavior_policy.py` | The Policy: starts the checkpoint's native server and forwards each step to it |
| `build-model-env.sh` | Builds the GR00T environment (Python 3.10) and fetches the checkpoint and its backbone, all pinned |
| `run_native.py` | Runs BEHAVIOR's own `omnigibson.eval.eval`, one process per instance, against the same server |
| `compare_native.py` | Pairs a Kit run with a native run episode by episode |
| `roboscope-policy.json` | Policy process contract template |

## Processes

The checkpoint's server, `scripts/b1k/serve_b1k.py` of
[`kmy17518/Isaac-GR00T`](https://github.com/kmy17518/Isaac-GR00T) commit `080d830d`, needs Python 3.10, which
cannot import the kit. The Policy therefore runs on Python 3.11+ (the BEHAVIOR runtime's own Python, with
the kit source on `PYTHONPATH`), starts the server as a child process on a free local port, and speaks
BEHAVIOR's websocket protocol to it (msgpack with NumPy arrays), the protocol of the native evaluator's
`WebsocketPolicy`. Each request carries the native observation keys the server reads, with a batch
dimension of one: the three RGB images, the 61-dim proprioception and `task_id`. The server keeps every
model-specific step as natively: the prompt for the task ID (the dataset's task name, the kind the
checkpoint was trained on), the 224x224 letter-box resize, the state slices, the 40-step action horizon
and `temporal_ensemble` (an inference per control step, averaged over the five newest chunks). Its
temporal-ensemble buffer belongs to the connection, so one replica serves one episode at a time: run as
many replicas as workers.

## Model environment

`build-model-env.sh` installs the fork's `uv.lock` exactly, with two additions: `websockets==16.0`, which the
fork's `pyproject.toml` pins for `serve_b1k.py` but its lock predates, and
`patches/transformers-4.57.3-offline-tokenizer.patch`, a backport of transformers 4.57.4's fix that lets
a tokenizer load offline. The server runs offline (`HF_HUB_OFFLINE=1`) from the cached Cosmos-Reason2-2B,
so a gated download is never attempted during evaluation; without the patch, transformers 4.57.3 asks the
Hub whether the tokenizer is a Mistral one and fails offline. Model code and weights are unchanged.

For the same evaluator observation, the request this Policy sends equals the native client's request,
value for value, on every key the server reads (the protocol document, section 9).

## Run

1. Build the simulator runtime (`benchmarks/behavior/build-runtime.sh --accept-licenses /sim/behavior`)
   and the model environment:

   ```bash
   examples/gr00t/build-model-env.sh /models/gr00t
   ```

   It prints `GR00T_ROOT`, `GR00T_PYTHON` and `HF_HOME`. GR00T N1.7 and its Cosmos-Reason2-2B backbone
   are under NVIDIA's model licences; read them first. The backbone is gated on Hugging Face: set
   `HF_TOKEN` to use it from there, otherwise the script takes NVIDIA's ModelScope copy and checks it
   against the Hugging Face digests.

2. Copy `roboscope-policy.json` and replace the paths.

3. Evaluate from any Python 3.11+ with `roboscope-kit` (`--quick-test` runs one worker and one replica;
   a full run passes `--concurrency K --replicas K`, one replica per worker):

   ```bash
   python -m roboscope_kit.runtime.benchmark \
       --benchmark behavior --suite challenge_2026 --quick-test \
       --checkpoint-source /models/gr00t/checkpoints/gr00t-n1.7-b1k-multitask/checkpoint-238000 \
       --benchmark-manifest /sim/behavior/benchmark-manifest.json \
       --policy-config /path/to/roboscope-policy.json \
       --output-dir /runs/behavior-quick
   ```

4. Run the same workload natively and compare:

   ```bash
   GR00T_ROOT=... GR00T_PYTHON=... HF_HOME=... /sim/behavior/env/bin/python examples/gr00t/run_native.py \
       --behavior-runtime /sim/behavior --task turning_on_radio --instances 1 \
       --checkpoint /models/gr00t/checkpoints/gr00t-n1.7-b1k-multitask/checkpoint-238000 \
       --output-dir /runs/behavior-native-quick
   python examples/gr00t/compare_native.py --kit-run /runs/behavior-quick --native /runs/behavior-native-quick
   ```
