# OpenVLA LIBERO Policy

A `roboscope-kit` `Policy` for the four suite-specific OpenVLA-7B checkpoints on
LIBERO-Spatial, Object, Goal and Long (`libero_10`). The benchmark contract is the
[LIBERO protocol](../../packages/roboscope-kit/LIBERO_EVALUATION_PROTOCOL.md).
This example is not verified to support the base OpenVLA checkpoint, `libero_90`, LIBERO-Plus,
or LIBERO-Pro.

| File | Purpose |
| --- | --- |
| `openvla_policy.py` | Loads a suite-specific checkpoint and implements image preprocessing, inference and action conversion |
| `build-policy-env.sh` | Builds the pinned Policy and benchmark CLI environment, separately from the simulator |
| `requirements.in` / `requirements.lock` | Inference dependencies and their hash-locked versions |
| `roboscope-policy.json` | Policy process configuration template |
| `run_native.py` | Runs the pinned official evaluator and exports per-episode results |
| `compare_native.py` | Checks case coverage and compares Kit/Native outcomes and steps |

## What the Policy keeps from upstream

The model-specific behavior follows the official OpenVLA LIBERO helpers at
revision `c8f03f48af692657d3060c19588038c7220e9af9`:

- Kit supplies the camera image already rotated by 180 degrees. The Policy applies
  JPEG encoding/decoding, TensorFlow Lanczos resize to 224×224 and a 90%-area center crop.
- TensorFlow preprocessing runs on CPU, matching the official evaluator; PyTorch
  inference uses GPU BF16 and FlashAttention2 with greedy decoding.
- The prompt uses the lowercase task instruction. Actions use suite-specific
  unnormalization, falling back to `<suite>_no_noops` when the suite key is absent,
  and the upstream gripper binarization/sign conversion.
- Each call produces one 7-dimensional action. The Policy returns it as float64
  in Kit's `libero.delta-eef-v1` representation, matching Native's executed action precision.
  Both the CLI and Policy environments need roboscope-kit 0.8.0b91 or later.
- The model keeps no recurrent state between requests or episodes. Episode IDs
  track lifecycle bookkeeping when a replica serves multiple Workers.

`OPENVLA_SEED` defaults to 7 and seeds the model process RNGs. Simulator seeds and
settling steps come from the benchmark manifest. Requests execute serially within
each Policy replica; additional replicas create independent model processes.

## Run

Use Linux x86_64, an NVIDIA GPU supporting BF16 and FlashAttention2, a compatible
NVIDIA driver, `uv`, `curl`, `git`, and conda or micromamba. Run the commands below
from the repository root.

1. Build the Policy environment and the standard LIBERO simulator into separate directories:

   ```bash
   bash examples/openvla/build-policy-env.sh /sim/openvla-policy
   KIT_PYTHON=/sim/openvla-policy/env/bin/python \
     bash benchmarks/libero/build-runtime.sh /sim/libero
   ```

   The benchmark CLI and Policy use `/sim/openvla-policy/env`; the benchmark
   manifest launches simulator Workers in `/sim/libero/env`. The Policy builder
   pins its inference dependencies; the standard LIBERO builder owns simulator pins.

2. Download the checkpoint matching the suite. Use `hf` from a separate tooling
   environment, keeping the pinned inference environment unchanged:

   ```bash
   export HF_HOME=/models/openvla-hf-cache
   hf download openvla/openvla-7b-finetuned-libero-spatial \
     --revision 962318cec55ac10993ff0f5f43eda9a270b4c873 \
     --local-dir /models/openvla/spatial

   # Cache the pinned model implementation; base-model weights are not needed.
   hf download openvla/openvla-7b \
     --revision 47a0ec7fc4ec123775a391911046cf33cf9ed83f --include '*.py'
   ```

   Keep checkpoints and `HF_HOME` on local disk, and reuse the same `HF_HOME` for
   evaluation. The Policy loads local files only. Retain the checkpoint's weights,
   tokenizer, processor configuration and `dataset_statistics.json`.

3. Copy [`roboscope-policy.json`](roboscope-policy.json) and replace its absolute
   paths with the Policy environment's Python, this checkout's `openvla_policy.py`
   and the example directory for `cwd`.

4. Run a quick evaluation in a new output directory:

   ```bash
   export OPENVLA_SEED=7
   /sim/openvla-policy/env/bin/python -m roboscope_kit.runtime.benchmark \
     --benchmark libero --suite libero_spatial \
     --checkpoint-source /models/openvla/spatial \
     --benchmark-manifest /sim/libero/benchmark-manifest.json \
     --policy-config /absolute/roboscope-policy.json \
     --quick-test --output-dir /runs/openvla-spatial-quick
   ```

   Drop `--quick-test` for the default ten-task × 50-state evaluation (500 episodes).
   For a 50-episode block using states 0–4, replace it with `--episodes 5`.
   Add `--concurrency 2 --replicas 2` for fixed two-Worker/two-replica execution;
   omit both flags to use SDK Autotune. To change suites, change both `--suite`
   and the matching checkpoint path. Use a new output directory for each run.

## Checkpoints

| Suite | Hugging Face repository | Revision |
| --- | --- | --- |
| `libero_spatial` | `openvla/openvla-7b-finetuned-libero-spatial` | `962318cec55ac10993ff0f5f43eda9a270b4c873` |
| `libero_object` | `openvla/openvla-7b-finetuned-libero-object` | `287d6cfdf12d07b1449505f66d9bf3550257e9b3` |
| `libero_goal` | `openvla/openvla-7b-finetuned-libero-goal` | `fa5ae1e7509348889295bba8e08621d8b55e9baf` |
| `libero_10` | `openvla/openvla-7b-finetuned-libero-10` | `80970322773f81baa2e22fe495d0487b93a05cfa` |

## Native comparison

Use an existing, separately prepared official OpenVLA/LIBERO environment and an
OpenVLA Git checkout pinned to `c8f03f48af692657d3060c19588038c7220e9af9`.
The Policy-only environment built above does not contain the Native simulator.
The reference experiments use Native robosuite 1.4.1; Kit uses the standard
Worker's robosuite 1.4.0. Keep the Native environment unchanged when comparing.

Run the following with the Native environment's Python, using the same local
checkpoint as Kit and a new output directory:

```bash
/path/to/native/env/bin/python examples/openvla/run_native.py \
  --openvla-root /path/to/openvla \
  --checkpoint /models/openvla/spatial --suite libero_spatial \
  --episodes-per-task 5 --seed 7 --env-seed 7 \
  --output /runs/openvla-spatial-native

python examples/openvla/compare_native.py \
  --kit-run /runs/openvla-spatial-kit \
  --native /runs/openvla-spatial-native \
  --output /runs/openvla-spatial-comparison.json
```

Generate the Kit input using the evaluation command above with `--episodes 5`
instead of `--quick-test`. Both runs cover all ten tasks and initial states 0–4:
this is **partial initial-state coverage**, 50 of the standard 500 episodes per
suite. The compact comparison supports consecutive states starting at zero.

The Native entry point calls the official evaluation loop with center cropping,
ten settling steps and CPU TensorFlow preprocessing. It retains task-major
execution, reset behavior, action conversion and suite horizons. It explicitly
sets environment seed 7 to match the reference campaign (the upstream helper
otherwise uses 0), replaces video writing with small episode records, and rejects
runs with caught evaluator errors. It does not record images or action traces.
Its elapsed time includes model loading and setup, so it is not formal evaluation
time and must not be compared directly with Kit's evaluation-only timing.

`run.json` records the Native invocation and completion status;
`native-episodes.jsonl` contains outcomes and executed policy steps, excluding
settling. The comparison rejects missing, duplicate or unexpected episode IDs and
mismatched suites, environment seeds or coverage. It reports per-task successes,
step totals, Kit-only/Native-only successes and individual outcome/step differences.
It does not certify identical checkpoints, scene conditions, observations or
trajectories; matching task/state labels alone cannot establish those properties.
Outputs stay in the selected run directories and are not repository fixtures.

## Evaluation

### Float64 action-precision validation (K/g=1/1)

The current Policy returns `float64` actions with `roboscope-kit 0.8.0b91` or later,
matching Native's executed precision. The November 26, 2026 comparison used an
RTX 4090, K/g=1/1, seed 7, and all ten tasks × states 0–4 per suite (50 episodes,
partial initial-state coverage).

| Suite | Float64 Kit success | Native success | Kit-only / Native-only success |
| --- | ---: | ---: | ---: |
| Spatial | 38/50 | 43/50 | 0 / 5 |
| Object | 43/50 | 42/50 | 1 / 0 |
| Goal | 37/50 | 33/50 | 4 / 0 |
| Long | 27/50 | 28/50 | 4 / 5 |
| **Total** | **145/200** | **146/200** | **9 / 10** |

Goal uses a fresh Native run; the other suites reuse the validated November 14
references with identical Native scripts, checkpoint paths and seeds. All four
comparisons pass coverage checks. There are 19 outcome disagreements, so similar
total scores do not establish episode or trajectory equivalence.

On Goal, the original float32 Policy reproduces every outcome and step count
before and after the SDK change (36/50 in both runs). Changing only the patched
Policy to float64 yields 37/50 and reduces disagreements with Native from five
to four. Fixed-sequence checks also confirm that the SDK preserves action bytes
and reproduces direct simulator states for both precisions.

At matching K/g=1/1, the historical float32 Kit scored 150/200 versus 145/200 for
the new float64 Kit, a 2.5-percentage-point difference. The SDK revisions differ,
so this is an aggregate comparison, not proof that precision has no effect. See
the [float64 validation](EXPERIMENT_REPORT.md#7-float64-action-preservation-issue-188)
for controls, all remaining disagreements and simulator-version/reset-history
limitations.

### Throughput and configuration benchmark (historical float32, K/g=2/2)

Completed performance measurements below used Kit `da280f0` (`0.8.0b58`, float32
actions) on an RTX 4090 at K/g=2/2 and the earlier serial Native reference with
upstream OpenVLA `c8f03f4`; each suite covers ten tasks × states 0–4. Times
exclude loading and tuning. These measurements were not rerun after the float64
change and remain the baseline for Kit's concurrency speedup.

| Suite | Native success | Kit 2/2 success | Native eval (min) | Kit eval (min) |
| --- | ---: | ---: | ---: | ---: |
| Spatial | 43/50 | 40/50 | 22.96 | 19.22 |
| Object | 42/50 | 43/50 | 28.66 | 24.19 |
| Goal | 33/50 | 38/50 | 29.11 | 22.66 |
| Long | 28/50 | 30/50 | 64.20 | 57.62 |
| Total | 146/200 | 151/200 | 144.93 | 123.68 |

The configuration study retains 2/2: 22.71% higher throughput than 1/1, with
similar success totals across configurations. Autotune selects 2/2 for all four
suites.

The [experiment report](EXPERIMENT_REPORT.md) documents evaluation coverage,
Native comparisons, performance measurements and diagnostic limitations.
