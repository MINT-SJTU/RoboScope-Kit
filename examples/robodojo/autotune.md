# Parallel and autotune comparison on the fixed six episodes

The six episodes are the earlier pilot and search subset. The current formal scope is 50 episodes of a single
task (see [full-series.md](full-series.md)).
The entry point accepts `--output` for an evidence directory that must not exist yet. Before updating the
patch, roll back the old parallel patch with the old version of the tool, check the `before` state, then
apply the new version. Never overwrite source code that is in use by a running evaluation.

This is an optional experiment. Acceptance of the main recipe still uses the single-environment
native/measured comparison.
It reuses the roboscope-kit SDK autotune (`two-phase` strategy) pinned at
`2792b6e160f06e24d18499b5ca5f50cca5f3fc31` and adds no search algorithm of its own.
The reported results below were measured with an earlier, unreleased version of the kit's two-phase
autotune. roboscope-kit's `two-phase` keeps the same implementation, but the current pin has not been
re-verified on GPU yet. The figures are reference results from the original experiments.
The controller needs Python 3.11+; the simulator and model use the dedicated environments from the main
[README](README.md).

## Scope and correctness constraints

- Fixed Simate-beta / stack_blocks / arx_x5, seed 1, layouts 0–5, 550-step limit. Every group keeps the
  native cameras, video, FP32, and 10-step flow inference.
- `off` uses K=1/G=1. `on` lets the existing controller test K=1, 2, 3 under its own search rules; G has a
  default upper bound of 1. K is the number of independent simulator processes, one environment per
  process; G is the number of model servers. This is a bounded experiment for the available resources and
  does not claim to search every parallel combination. `--maximum-workers 1|2|3` tightens the bound. The
  default bound of 3 is the experimental range of the 49,140 MiB GPU used here, not a recommendation for
  24 GB cards. One simulator process takes about 8 GiB and the model about 15 GiB; choose bounds from the
  actual capacity. The SDK can still reject candidates early.
- Every candidate runs the same six episodes in full. The SDK resource, OOM, failure, and budget checks
  stay in place; the tolerated success-rate drop is 0. The six-episode threshold is an engineering screen
  and does not prove statistical equivalence.
- Shards are assigned as `range(episodes)[worker_index::K]` and keep the original layout IDs. After the
  run, every shard and the union are checked for missing, duplicated, or substituted layouts. Anomalies
  are recorded as incomplete, and the run's child process groups are cleaned up.
- A shared model uses its existing `reset_evaluation` interface, which clears only the history of the
  current evaluation. Model-side forward, RNG, precision, chunking, and preprocessing code are unchanged.
- Changing K changes the order in which the shared model RNG is consumed and can change physics results.
  Report per-episode success/failure, steps, termination, and repeated measurements; equal success totals
  for two groups are not enough.

## Optional multi-replica comparison (2026-10-26)

The earlier experiment fixed G=1 in the entry point and in the SDK bounds and did not compare G=2/3, so it
does not show that autotune proved a single replica optimal. The current `0003-parallel-autotune.patch`
supports `--maximum-replicas 1|2` (default 1) and hands the replica dimension to the existing SDK search
without adding a tuning algorithm.
Each replica uses the same checkpoint/seed with its own port and process. Workers connect to replica
`worker_index % G`, and each replica serves at least one worker. The model implementation and scoped reset
are unchanged. Independent replicas have independent RNG streams, so trajectories are not guaranteed to
match the shared server; compare them layout by layout.

To isolate the effect of G, use `fixed` to measure a given K/G directly. It calls the same measurement entry
point without search overhead and is not an autotune `on` result. Run the commands below with the existing
environments idle and after the patches above are applied. Before updating, roll back the old parallel
patch with the old tool, confirm `before`, then apply parallel with the new, locally committed tool. Do not
switch versions under running source code.

```bash
for G in 1 2; do
  "$PYTHON" "$ROOT/recipe_parallel_runtime.py" "$ROOT" \
    --recipe-dir "$RECIPE" --profile "$RECIPE/recipe.toml" --sdk-root "$SDK" \
    --mode fixed --workers 2 --replicas "$G" \
    --maximum-workers 2 --maximum-replicas 2 --episodes 6 \
    --output "$BASE/evidence/replicas-k2g${G}-NEW"
done
# To search the replica dimension, use the existing controller:
# --mode on --maximum-workers 2 --maximum-replicas 2 --episodes 50 --probe-episodes 6
```

In the earlier runs one loaded model used 13.11 GiB of GPU memory; the K3/G1 formal peak was 39.07 GiB and
the K2/G1 probe peak 30.75 GiB. Adding one replica of the same size gives rough estimates of 52.18 /
43.86 GiB, not counting extra inference peaks. On this 49,140 MiB GPU only K2/G2 was tested; K3/G2, which
was expected to exceed capacity, was not started. Do not read these results as capacity conclusions for a
standard 24 GiB card. Tighten the bounds for the target machine. On a launch failure or OOM, keep the full
logs and clean up every child process of the run.

Total GPU usage is still sampled every 500 ms. Per-PID GPU memory of each model/simulator process is also
sampled every 5 seconds, and each model's post-load memory and load time are recorded. CPU/RAM are read
from the cgroup every second; RAM includes file cache and other processes in the cgroup and is not the
model's resident set. Raw samples, process PIDs, ports, and run JSON stay outside the repository.
After a complete run, use the rollback steps on this page. Record these performance results separately
from the earlier G=1 results.

Six-episode diagnostic from the original experiments: K2/G2 completed with a peak of 44.09 GiB GPU memory.
Formal steps/s went from 5.33 to 6.63 and mean GPU utilization from 32.12% to 43.45%. However, success
went from 3/6 to 5/6, the load was more balanced, and the trajectories differed, so the observed 1.2419×
cannot be attributed entirely to the extra replica. The capacity check passed; the default stays at G1.
This short test does not replace the fixed 50-episode acceptance run.

## Apply order and run

Finish the native six episodes from the main [README](README.md) first, then apply measurement. Modify
source code only through the committed patch tool. Adjust the paths below for your machine; keep the SDK
checkout and the recipe checkout separate.

```bash
BASE=/path/to/robodojo
ROOT="$BASE/RoboDojo"
RECIPE=/path/to/roboscope-kit/examples/robodojo
SDK="$BASE/roboscope-kit-2792b6e"
# The upstream simulator environment; roboscope-kit itself needs only numpy, so it is also the controller.
PYTHON="$HOME/miniconda3/envs/RoboDojo/bin/python"

git clone https://github.com/mint-sjtu/roboscope-kit.git "$SDK"
git -C "$SDK" checkout --detach 2792b6e160f06e24d18499b5ca5f50cca5f3fc31
# If the checkout already exists, check its revision and reuse it; do not overwrite local changes.
export PYTHONPATH="$SDK/packages/roboscope-kit/src:$RECIPE:$ROOT"
"$PYTHON" -c 'from roboscope_kit.runtime.benchmark import autotune; print(autotune.__file__)'

"$PYTHON" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch measurement
"$PYTHON" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch episode-count
# Upstream's own single-process two-environment reference; run it before applying the parallel patch.
"$PYTHON" "$RECIPE/run_eval.py" "$ROOT" \
  --mode measured --seed 1 --episodes 6 --num-envs 2 --label upstream-vector-two

"$PYTHON" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch parallel
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch parallel # after
for MODE in off on; do
  "$PYTHON" "$ROOT/recipe_parallel_runtime.py" "$ROOT" \
    --recipe-dir "$RECIPE" --profile "$RECIPE/recipe.toml" --sdk-root "$SDK" \
    --mode "$MODE" --episodes 6 --probe-episodes 6 --maximum-workers 3 \
    --candidate-timeout-s 1800 --tune-budget-s 3600
done
```

`--simulator-env`/`--policy-env` default to the upstream conda environments `RoboDojo` and
`simate-beta-eval`; `--vulkan-icd` and `--metrics-root` accept host paths. By default the two common NVIDIA ICD directories are
searched. The controller records the source revisions and rejects unknown modifications, model modifications, a
wrong patch digest, or a simulator/Python/PyTorch that does not match the profile. The generic SDK source
needs no changes.

## Results and timing

Run output goes to `ROOT/../evidence/autotune/<off|on>-<id>/`. `experiment.json` records the selected K/G,
search time, total time including search, and the formal run path; `selection/` holds the existing SDK's
candidates and rejection reasons. Each candidate's and formal run's `summary.json`, `result.json`, worker
logs, and GPU samples stay outside the repository.

The formal evaluation window runs from the earliest worker's evaluation_start to the latest
evaluation_end. It includes reset, execution, output written inside the window, and tail imbalance; it is
not the sum of per-worker times. Total wall clock also includes model loading, simulator startup, and
cleanup. The total cost of `on` must also include every search candidate of the run, not only the selected
candidate's execution time. Preparation/download is listed separately. GPU is sampled every 500 ms; CPU
uses cumulative cgroup counters. Low GPU utilization alone does not identify the bottleneck.

When comparing serial, upstream vector-two, and autotune off/on, list for each: successes/success rate,
actual steps, both wall clocks, episodes/s, steps/s, mean GPU utilization, mean/peak GPU memory, and
speedups computed the same way. Step counts can change under concurrency, so look at both episode
throughput and steps/s. The throughput signal this entry point gives the existing controller for each
candidate is actual steps/s, while acceptance uses full episodes/s; the controller selecting a candidate
does not by itself mean the 2× target was reached. By default `compare_runs.py` compares native/measured
runs with the same number of environments. Across concurrency configurations pass
`--allow-topology-change` explicitly; it still checks the same profile, source, measurement version, seed,
and layout set, and reports input, action, step, and termination differences by layout ID. It does not
bypass the other consistency checks. Repeat full off/on runs to observe variance.

## Rollback

Wait for every process of the run to exit, roll back parallel first, then roll back measurement as in the
main [patch documentation](patches/README.md). Do not switch source code while model/simulator processes
are still running.

```bash
"$PYTHON" "$RECIPE/apply_runtime_patch.py" rollback "$ROOT" --patch parallel
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch parallel # before
"$PYTHON" "$RECIPE/apply_runtime_patch.py" rollback "$ROOT" --patch measurement
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch measurement # before
```

The patch does not change the user's model and does not publish an official image. The current bounded
configuration and thresholds are experimental settings.

## Measured results by stage

Native 5.1 off/on run of 2026-10-24:
the K=2 candidate reached 2/6, below this run's K=1 candidate at 3/6, so the controller rejected K=2 and
stopped expanding; K=3 was not tested and K=1 was selected. Both off/on formal groups reached 4/6, with
formal windows of 531.049 / 535.972 s and total experiment times including search of 597.953 /
1,863.942 s. This run achieved no speedup and missed the 2× target, and six episodes cannot prove
statistical equivalence. Do not treat this optional entry point as a verified default speedup
configuration.

Later fixed 50-episode comparison, using an earlier version of the parallel patch and the same SDK
autotune: the K1/K2/K3 probes each reached 4/6, and K3/G1 was selected. Formal off/on success went from
27/50 to 29/50, with formal episode throughput 2.1948× and actual steps/s 2.1405×; including the
27.50 minutes of search, overall episode throughput was 1.3426×. The formal run used `--episodes 50` and
probes stayed at 6 episodes; raw evidence is kept outside the repository. The `on` group peaked at
40010 MiB GPU memory, which does not carry over to standard 24 GB cards. System RAM was not sampled for
either group, and per-layout images/actions were not equivalent, so the success rate cannot be claimed to
be unaffected. After the experiment the parallel patch was rolled back; the entry point remains optional.
