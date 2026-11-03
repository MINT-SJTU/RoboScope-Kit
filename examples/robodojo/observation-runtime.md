# Observation RPC and bounded FIFO candidate

This page covers an optional runtime path in `0003-parallel-autotune.patch`; it does not change the SDK search algorithm. Synchronous mode is the default. `--overlap-observation` enables asynchronous waiting for acknowledgements with a default queue limit of 1; only an explicit `--observation-queue-limit 10` enables the FIFO10 candidate described here.

In the original experiments, the full 50-episode single-pending candidate showed a success rate drop from 60% to 50%, and the cause was not fully identified; FIFO10 was only diagnosed on three layouts. Neither is an accepted optimization with no success-rate regression.

## Ordering and state constraints

- A deep copy of the observation is taken before submission and stored in a bounded queue; frames are never dropped, merged, or downsampled.
- Each connection has only one RPC in flight; a later request must wait for and validate the previous response. Earlier errors propagate forward, and queued requests are not sent after a failure.
- The queue is drained before inference, reset, other model RPCs, and close; an interrupted wait still keeps the outstanding request.
- Model methods still go through the original global lock; history, random stream allocation, model code, and action queue execution are unchanged.
- Only the `update_observation` returns-None contract already verified for this pinned recipe is supported; it is not generalized to arbitrary model client semantics.

Ordering within one worker is protected, but this does not mean the global RNG consumption order across workers matches a synchronous run; this remains a limitation for success-rate analysis.

## Apply and small-scale comparison

First complete the preparation in the main [README](README.md) and the [autotune environment](autotune.md), keeping episode-count applied; keep the video, render, thread, and control state candidates in the before state. After evaluation and model processes exit, use the original version of the tools to roll back the old parallel patch and then the old measurement patch, and keep the old tools for restoring. The new tools only accept their pinned original or applied state; do not skip the check to overwrite an old patch.

```bash
RECIPE=/path/to/roboscope-kit/examples/robodojo
BASE=/path/to/robodojo
ROOT="$BASE/RoboDojo"
SDK="$BASE/roboscope-kit-2792b6e"
PYTHON="$HOME/miniconda3/envs/RoboDojo/bin/python"
EVIDENCE="$BASE/evidence"
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch measurement # before
"$PYTHON" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch measurement
"$PYTHON" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch parallel
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch parallel # after
export PYTHONPATH="$SDK/packages/roboscope-kit/src:$RECIPE:$ROOT"
# This reproduces the three-layout diagnostic setup with the same profiling in both groups; output directories must be new.
export ROBODOJO_EXECUTION_PROFILE=1 ROBODOJO_PYTHON_PROFILE=1
unset ROBODOJO_CAUSAL_TRACE_DIR
"$PYTHON" "$ROOT/recipe_parallel_runtime.py" "$ROOT" \
  --recipe-dir "$RECIPE" --profile "$RECIPE/recipe.toml" --sdk-root "$SDK" \
  --mode fixed --workers 3 --replicas 1 --maximum-workers 3 --maximum-replicas 1 \
  --episodes 3 --output "$EVIDENCE/observation-off-NEW"
"$PYTHON" "$ROOT/recipe_parallel_runtime.py" "$ROOT" \
  --recipe-dir "$RECIPE" --profile "$RECIPE/recipe.toml" --sdk-root "$SDK" \
  --mode fixed --workers 3 --replicas 1 --maximum-workers 3 --maximum-replicas 1 \
  --episodes 3 --output "$EVIDENCE/observation-fifo10-NEW" \
  --overlap-observation --observation-queue-limit 10
```

The earlier single-pending comparison used `--overlap-observation --observation-queue-limit 1`. With `fixed`, search time is 0; this is not an autotune on/off comparison. The earlier 50-episode single-pending experiment enabled only execution profiling with cProfile off, so do not merge it with this example into one result.

Afterwards, run `verify_videos.py` and `compare_runs.py` on the summary/result paths as described in the [full validation guide](full-series.md); keep failures, incomplete runs, and per-layout differences. After processes exit, roll back in reverse order:

```bash
"$PYTHON" "$RECIPE/apply_runtime_patch.py" rollback "$ROOT" --patch parallel
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch parallel # before
"$PYTHON" "$RECIPE/apply_runtime_patch.py" rollback "$ROOT" --patch measurement
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch measurement # before
# If an older measurement version was applied before, restore it last with its original tools; do not operate on sources that are in use.
```

Timing of lock wait, executor dispatch, method body, and return to the event loop belongs to `0001-measurement.patch`; records are written after the original model lock is released and add no GPU synchronization. The optional large causal trace is for diagnostics only; raw observation, RNG, and action records stay outside the repository and are not committed as runtime dependencies.
