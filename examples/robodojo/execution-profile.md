# Single-process execution overhead breakdown

This step breaks down execution overhead that the earlier 50-episode baseline did
not itemize. It does not modify parallel/autotune or
`XPolicyLab/policy/Simate_beta`. The model, seed, layouts, action mapping,
550-step limit, history/chunk, physics stepping, camera fence, and video
parameters stay the same.

`0001-measurement.patch` adds an optional `--execution-profile`: within
`run_eval` for a single episode it records, for the synchronous call tree, call
counts, cumulative host time, self time (excluding measured child calls on the
same thread), and maximum duration. Coverage includes observation, action,
control interpolation, robot state, physics, scene sync, render fence, camera
readback, video, and RPC. No CUDA synchronize calls are added, so asynchronous
CUDA work may be attributed to a later wait; do not call these times GPU kernel
time.

Inclusive times on the synchronous path are nested, and the main thread's RPC
wait overlaps with the background codec thread, so they cannot be summed.
Asynchronous server time is tagged with `async/` and should only be read as
cumulative intervals that include waiting. Server statistics cover the whole
process lifetime, while simulator statistics cover only the measured execution;
reset and initialization use the existing measurement records. Report these
scopes separately in comparisons.

`--python-profile` also saves a main-thread cProfile for deeper hotspot
inspection. It adds overhead, so do not compute speedups against runs without
it. Formal performance acceptance runs disable both profiling options.
Diagnostic output goes to the external `events/execution-profile/`; do not
commit the JSON or `.prof` files.

## Update, run, and roll back

First make sure all model and simulation processes from this run have exited and
the GPU is idle. Roll back the old measurement patch with the originally deployed
version of the tools and confirm `before`, then apply with the new committed
tools. Each stage's report keeps its own measurement version; do not combine
samples from the new version with timings from the old version as one
measurement.

```bash
BASE=/path/to/robodojo
ROOT="$BASE/RoboDojo"
OLD=/path/to/previous/recipe
RECIPE=/path/to/committed/examples/robodojo
PYTHON="$HOME/miniconda3/envs/RoboDojo/bin/python"

"$PYTHON" "$OLD/apply_runtime_patch.py" check "$ROOT" --patch parallel # before
"$PYTHON" "$OLD/apply_runtime_patch.py" rollback "$ROOT" --patch measurement
"$PYTHON" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch measurement
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch measurement # after
# episode-count stays applied.
"$PYTHON" "$RECIPE/run_eval.py" "$ROOT" --mode measured --seed 1 \
  --num-envs 1 --episodes 1 --label execution-diagnostic \
  --execution-profile --python-profile
# After the breakdown, use a separate evidence directory for the fixed six-episode pilot and the 50-episode comparison with profiling off.
"$PYTHON" "$RECIPE/verify_videos.py" /path/to/completed/run

# After all processes exit, restore the old measurement version:
"$PYTHON" "$RECIPE/apply_runtime_patch.py" rollback "$ROOT" --patch measurement
"$PYTHON" "$OLD/apply_runtime_patch.py" apply "$ROOT" --patch measurement
```

The native entry point does not enable profiling by default. The settings
snapshot records the actual device, physics dt, decimation, observation
interval, render interval, Kit wall-clock rate limiting, and the zero-delay
option; reading them does not change any setting. The statistics module is
installed by the benchmark patch and does not speed up or change computation in
user inference code.

## RPC server breakdown and per-request tracing

The latest measurement patch keeps the original model lock and separates lock wait, executor dispatch, method body, and return-to-event-loop time. Records are written after the lock is released and add no GPU synchronization. Timing and cProfile remain diagnostic switches and must match between the two groups; thread pool wait time is not model compute time.

The optional `ROBODOJO_CAUSAL_TRACE_DIR` enables per-request diagnostic records for observations, RNG, actions, and so on. The output is large and affects timing, so point it only at a new directory outside the repository. Disable this trace for formal throughput experiments, and do not infer that unrecorded values match just because counts match. See [observation runtime](observation-runtime.md) for the runtime switches.
