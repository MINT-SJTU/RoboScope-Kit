# Native batch comparison

The opt-in `ROBOSCOPE_STARVLA_MAX_BATCH_SIZE` accepts a positive integer; only the LIBERO-Pro QwenOFT
example opts in above one. The Policy Server collects only already-queued requests from distinct
episodes, without an intentional collection delay, and replies to each request with its own validated
action chunk and original frame encoding. Lifecycle calls remain serialized. A batch cap is not the
actual batch size; MPS can be combined with batching but is an independent switch.

`capture_batch_observations.py` runs in the pinned simulator environment and saves four task-0-through-3
observations after native reset and 10 settling steps. `check_native_batch.py` runs in the model
environment against the same checkpoint, compares B=2 and B=4 (including reversed ordering) against
single predictions, requires exact gripper commands and `rtol=0.01, atol=0.005` on the action arrays,
and records the maximum absolute differences. This small check is necessary but cannot prove full
closed-loop trajectory equivalence.

`run_batch_comparison.py` uses a new output directory and defaults to caps 1, 2, 4, the
`libero_goal_object` suite, 50 episodes per task, fixed K=12/g=3, MPS on, MessagePack, per-request
connections, video off and chunk camera rendering on. Each arm has a separate Quick Test warmup and
separate formal batch metrics. Install the same Kit revision in both controller and model environments;
the Policy config must point to this checkout. Preserve the pinned simulator, task assets and model.

```bash
python benchmarks/libero_pro/run_batch_comparison.py \
  --benchmark-manifest /sim/libero-pro/benchmark-manifest.json \
  --policy-config /models/starvla-policy.json --checkpoint-source /models/starvla-libero-pro \
  --output-dir /runs/libero-batch-01 --dry-run
```

Remove `--dry-run` to execute. `--batch-sizes 1 2 4` sets both the caps and the run order. The driver
requires an idle GPU, checks again between phases, and uses the existing owned MPS lifecycle. Each arm
must have complete records, requested K/g, all MPS replicas attached and a matching Policy capability.
Its batch log records size, Policy wall time and the `predict_action` wall interval, including input
processing, GPU work and output synchronization. These are wall times, not pure GPU kernel time.
Metrics consume request-scoped timing spans; no per-request duration is stored on the Policy instance.
The metrics directory is read at server construction. Metrics alone do not enable CUDA diagnostic hooks.
Batch logging is enabled consistently in every arm; warmup and formal measurements are separate. The output
contains per-size counts, mean/median/p95, sampled peak GPU memory, scores, actual steps and paired
differences against cap 1. Failed/retried work requires explicit review rather than trusting the batch
accounting. One 500-episode suite is neither full-matrix acceptance nor proof of statistical equivalence.
