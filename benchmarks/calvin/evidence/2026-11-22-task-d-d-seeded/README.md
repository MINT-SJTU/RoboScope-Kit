# CALVIN seeded StarVLA batching

On 2026-11-22, both arms ran the 1,000 official `task_D_D` sequences on the same RTX 4090
with an Intel Core i9-14900K and NVIDIA driver 580.173.02. Both used EGL, CUDA MPS off,
the run cache off, the same StarVLA checkpoint, and `inference.action_noise_seed: 0`.
Kit commit [`7b34ef3`](https://github.com/MINT-SJTU/RoboScope-Kit/pull/138/commits/7b34ef346eabe9c758a0cf418af00e6299f1247a)
(`0.8.0b73`) was used for both arms.

| Metric | A, serial | B, dynamic batching |
| --- | ---: | ---: |
| Workers / Policy replicas / batch cap | 1 / 1 / 1 | 16 / 1 / 8 |
| Completed sequences | 1,000 | 1,000 |
| Formal evaluation | 4,216.824 s | 1,927.250 s |
| End to end | 4,228.310 s | 1,945.407 s |
| Policy load / runtime preparation | 7.612 s / 2.203 s | 11.166 s / 3.609 s |
| Sequences/s | 0.23715 | 0.51887 |
| Simulation steps/s | 105.40 | 228.27 |
| Total simulation steps | 444,444 | 439,935 |
| Peak device memory | 9,547 MiB | 14,677 MiB |
| Mean device utilization | 55.63% | 88.21% |
| Mean actual batch size | 1.00 | 5.62 |
| Mean completed subtasks | 3.805 | 3.800 |
| Completion through subtask 1 | 92.3% | 92.3% |
| Completion through subtask 2 | 84.3% | 83.4% |
| Completion through subtask 3 | 76.3% | 74.7% |
| Completion through subtask 4 | 67.6% | 68.5% |
| Completion through subtask 5 | 60.0% | 61.1% |

The configuration raised formal sequence throughput by **2.188×** and end-to-end throughput by
**2.173×**. The comparison measures the combined effect of increasing Worker count and enabling
batching. The record audit matched every official sequence index and ordered task list exactly
once in each arm, with no Worker failures or out-of-memory events.

Across paired sequences, 113 completed more subtasks in B, 772 tied, and 115 completed fewer.
Ninety-five sequences gained full success and 84 lost it. Mean completion changed by −0.005
subtasks. Completion through subtask 3 fell 1.6 percentage points and full completion rose
1.1 points. Each observation samples initial action noise from the same episode and step seed
in both arms. Batch shape can still change model numerics. These scores describe one paired run.

The GPU process monitor accepted both recorded runs with only the evaluation and desktop
graphics processes present. The logs contain 4,101 and 1,880 distinct process sample timestamps,
respectively, with at most two seconds between successive timestamps in either run. Two earlier
serial attempts ended after detection of foreign compute clients. The table uses the subsequent
complete serial run and the complete batched run. The collection service was restored after both.

- [Summary and paired counts](summary.json)
- [Serial records](baseline/records.jsonl) and [batched records](batched/records.jsonl)
- [Serial logs](baseline/logs.tar.gz) and [batched logs](batched/logs.tar.gz), including manifests,
  results, worker and Policy logs, captured commands, and GPU process samples
- [Serial batch trace](baseline/actual-batches.tsv.gz) and [batched batch trace](batched/actual-batches.tsv.gz),
  with batch size and action-prediction wall time in nanoseconds

## Reproduction

1. Check out Kit at `7b34ef346eabe9c758a0cf418af00e6299f1247a`. Run
   `"$kit/benchmarks/calvin/build-runtime.sh" "$runtime"` on Linux x86_64. The simulator uses
   CPython 3.11.15 and PyTorch 2.6.0+cpu. The builder verifies the sequence list against SHA-256
   `c3689911565037d399d779c655e9e943cbd5bfbf44045e2b3b1daa8b1a2317`.
2. Prepare the [CALVIN StarVLA demo](../../../../examples/starvla/README.md#run-a-demo).
   The Policy process used PyTorch 2.6.0+cu124 and StarVLA revision
   `985c2238627481f25476fbee478083812dd82a21`. Apply the patch with
   `git -C "$starvla" apply "$kit/examples/starvla/calvin/starvla-action-noise.patch"`.
   Use the demo checkpoint revision `962182bdf2cd07958c1fda4a05ea21e8423ab779` and verify
   `steps_30000_pytorch_model.pt` against SHA-256
   `abf79ab7c4426c2eb618032a0a8258d9aaaf4f92542e86d4ff569cdc6fd6a3c0`.
3. Use two copies of the same snapshot. In each snapshot's `config.yaml`, set
   `inference.action_noise_seed: 0`. Set `inference.max_batch_size: 1` for A and
   `inference.max_batch_size: 8` for B. Use the pinned, patched StarVLA checkout as
   the Policy process contract's `cwd` and use
   `[policy_python, traced_policy.py, starvla_policy.py]` as its command. All paths
   in the contract must be absolute. Both arms used this launcher:

```python
import os
from pathlib import Path
import random
import sys
from time import perf_counter_ns

import numpy as np
import torch

entrypoint = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(entrypoint.parent))
from starvla_policy import StarVLAPolicy
from roboscope_kit.server import run

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

with Path(os.environ["CALVIN_PR138_BATCH_TRACE"]).open("w") as trace:
    class TracedStarVLAPolicy(StarVLAPolicy):
        def act_batch(self, observations):
            start = perf_counter_ns()
            actions = super().act_batch(observations)
            trace.write(f"{len(observations)}\t{perf_counter_ns() - start}\n")
            return actions

    run(TracedStarVLAPolicy())
```

Use absolute paths for the variables below. Run the arms sequentially against a GPU with no other
compute tasks. Use a new output directory for each arm. For process evidence, run
`nvidia-smi pmon -i 0 -s um -d 1 -o DT` in parallel and check that every compute PID belongs to
the evaluation.

```bash
export PYTHONPATH="$kit/packages/roboscope-kit/src"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
unset CALVIN_ACTION_TYPE

CALVIN_PR138_BATCH_TRACE="$serial_trace" \
  "$runtime/env/bin/python" -u -m roboscope_kit.runtime.benchmark \
  --benchmark calvin --suite task_D_D --episodes 1000 \
  --benchmark-manifest "$runtime/benchmark-manifest.json" \
  --checkpoint-source "$serial_snapshot" --policy-config "$policy" \
  --concurrency 1 --replicas 1 --run-cache off --output-dir "$serial_output"

CALVIN_PR138_BATCH_TRACE="$batched_trace" \
  "$runtime/env/bin/python" -u -m roboscope_kit.runtime.benchmark \
  --benchmark calvin --suite task_D_D --episodes 1000 \
  --benchmark-manifest "$runtime/benchmark-manifest.json" \
  --checkpoint-source "$batched_snapshot" --policy-config "$policy" \
  --concurrency 16 --replicas 1 --run-cache off --output-dir "$batched_output"
```
