# CALVIN StarVLA dynamic batching

On 2026-11-22, two runs evaluated the same 1,000 official `task_D_D` sequences on one
RTX 4090 with an Intel Core i9-14900K and NVIDIA driver 580.173.02. Both used EGL,
CUDA MPS off, the run cache off, and the same StarVLA checkpoint and scoring rules.
GPU process samples show the evaluation and desktop processes in both accepted runs.
This comparison used the original process-wide action noise RNG. The current CALVIN
example supports seeding each observation independently for a new comparison.

| Metric | Serial baseline | Dynamic batching |
| --- | ---: | ---: |
| Workers / Policy replicas / batch cap | 1 / 1 / 1 | 16 / 1 / 8 |
| Completed sequences | 1,000 | 1,000 |
| Formal evaluation | 4,222.094 s | 1,947.093 s |
| End to end | 4,234.009 s | 1,964.676 s |
| Policy load / runtime preparation | 7.878 s / 2.603 s | 11.370 s / 3.611 s |
| Sequences per second | 0.236849 | 0.513586 |
| Simulation steps per second | 103.400 | 228.157 |
| Device peak memory | 9,547 MiB | 14,359 MiB |
| Total simulation steps | 436,566 | 444,243 |
| Mean completed subtasks | 3.826 | 3.863 |
| Completion through subtask 1 | 92.1% | 92.4% |
| Completion through subtask 2 | 83.6% | 85.1% |
| Completion through subtask 3 | 75.0% | 76.9% |
| Completion through subtask 4 | 69.1% | 70.1% |
| Completion through subtask 5 | 62.8% | 61.8% |

The combined configuration gave **2.17× formal throughput** and **2.16× end to end**.
The measured mean batch size was 5.73. The full-run ratio includes the Worker increase
and batching together. At a fixed four Workers and one replica, an earlier 40-sequence
prototype comparison measured 0.299 sequences/s at cap 1 and 0.396 at cap 4 (1.32×).
A separate timing sample on the selected configuration measured 141 ms per prediction
batch on average, of which the Qwen forward stream spanned 102 ms and the action head
spanned 21 ms. [Timing sample](bottleneck-profile.json)

The audit confirmed every sequence index and its ordered official task list exactly once in
each arm. Relative to the baseline, 109 sequences completed more subtasks, 784 tied, and 107
completed fewer. Across all pairs, 84 became full successes and 94 ceased to be full successes. The mean
increased by 0.037, while full success fell by 1.0 percentage point. One paired full run
measures these differences. Batch composition can change the order in which a seeded Policy
draws random numbers.

- [Summary and paired counts](summary.json)
- [Serial work items](baseline/records.jsonl) and [batched work items](batched/records.jsonl)
- [Serial logs](baseline/logs.tar.gz) and [batched logs](batched/logs.tar.gz), including events,
  worker and Policy logs, and GPU process samples. Log paths abbreviate the experiment account's home directory.
- [Per-call batch trace](batched/actual-batches.tsv.gz), with batch size and prediction wall time in nanoseconds.

## Reproduction

The serial arm ran Kit `4f9f0dbef8d49c3d04212c401167e23224ea1060` with evaluator and Policy sources
identical to upstream `4fe7e50ca2700fe63a2926af4f1130db2108465a`. The batched arm used a prototype
over the same sources, adapted from PR #22 for the current wire encodings. The first PR #138
revision used the prototype dispatch implementation. The measured StarVLA example also emitted
the batch trace and accepted an unused clip transform.
[Serial evaluator source](https://github.com/MINT-SJTU/RoboScope-Kit/tree/4fe7e50ca2700fe63a2926af4f1130db2108465a)
[Batching prototype upstream PR](https://github.com/MINT-SJTU/RoboScope-Kit/pull/22)

Build a fresh runtime with the submitted builder. The simulator uses CPython 3.11.15 and
PyTorch 2.6.0+cpu, and the Policy process uses PyTorch 2.6.0+cu124. Keep the StarVLA checkout
at `985c2238627481f25476fbee478083812dd82a21`. The demo pins the checkpoint revision
`962182bdf2cd07958c1fda4a05ea21e8423ab779`. Verify the selected
`steps_30000_pytorch_model.pt` against SHA-256
`abf79ab7c4426c2eb618032a0a8258d9aaaf4f92542e86d4ff569cdc6fd6a3c0`.
The runtime builder verifies the sequence list against SHA-256
`c3689911565037d399d779c655e9e943cbd5bfbfbf44045e2b3b1daa8b1a2317`.
[StarVLA environment and snapshot setup](../../../../examples/starvla/README.md#run-a-demo)

Use two copies of the same snapshot, leaving the serial snapshot at the default cap of 1
and adding `max_batch_size: 8` under the batched snapshot's `inference` mapping. Keep all
other snapshot settings equal. In the Policy process contract, use the pinned StarVLA
checkout as `cwd` and the command `[policy_python, seeded_policy.py, starvla_policy.py]`
with absolute paths. Point `starvla_policy.py` at the corresponding Kit checkout for each arm.
For the historical batched arm, use the prototype or the initial dispatch implementation at
[`2b1a002`](https://github.com/MINT-SJTU/RoboScope-Kit/pull/138/commits/2b1a0021d44187fa00ec5b1725bf9f21fed229e7).
The launcher fixes the process startup seeds.

```python
import random
import runpy
from pathlib import Path
import sys

import numpy as np
import torch

entry = Path(sys.argv.pop(1)).resolve()
sys.argv[0] = str(entry)
sys.path.insert(0, str(entry.parent))
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
runpy.run_path(str(entry), run_name="__main__")
```

Set the variables to absolute paths and use a new output directory for each arm.

```bash
"$kit_with_batching/benchmarks/calvin/build-runtime.sh" "$runtime"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
unset CALVIN_ACTION_TYPE

PYTHONPATH="$kit_serial/packages/roboscope-kit/src" \
  "$runtime/env/bin/python" -u -m roboscope_kit.runtime.benchmark \
  --benchmark calvin --suite task_D_D --episodes 1000 \
  --benchmark-manifest "$runtime/benchmark-manifest.json" \
  --checkpoint-source "$serial_snapshot" --policy-config "$serial_policy" \
  --run-cache off --concurrency 1 --replicas 1 --output-dir "$serial_output"

PYTHONPATH="$kit_with_batching/packages/roboscope-kit/src" \
  "$runtime/env/bin/python" -u -m roboscope_kit.runtime.benchmark \
  --benchmark calvin --suite task_D_D --episodes 1000 \
  --benchmark-manifest "$runtime/benchmark-manifest.json" \
  --checkpoint-source "$batched_snapshot" --policy-config "$batched_policy" \
  --run-cache off --concurrency 16 --replicas 1 --output-dir "$batched_output"
```
