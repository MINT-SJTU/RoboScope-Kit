# PhysX CPU worker thread comparison

This candidate only changes the worker thread limit of the PhysX CPU dispatcher
inside a single simulation process. It does not change the number of simulation
processes, model server replicas, or the existing autotune. Off by default; do
not treat the configured thread count as the number of cores actually in use.
Physics dt, backend, and solver, camera sync, video, and model code are all
unchanged.

The change lives in `0007-physics-threads.patch`. It adds explicit Kit arguments
before the existing AppLauncher, and on the first camera capture checks the
effective value and the PhysX dispatcher. Conflicting arguments or a setting
that did not take effect raise an error. The camera wait loop and frame count
logic are unchanged. Only the Isaac Sim 5.1 CPU environment pinned by the recipe
is supported. `--physics-threads` accepts 1, 4, or 8 as a manually fixed value;
there is no search or automatic selection.

The [Kit configuration docs](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/107.0.3/guide/configuring.html)
state that `/persistent` settings can be written back to the user config every
frame. So when thread arguments are enabled explicitly, the patch also passes
`--/app/settings/persistent=false` so that this run's overrides do not persist
into later runs; loadUserConfig is unchanged. In a controlled thread comparison,
both groups specify the thread count explicitly so this setting stays the same,
and the baseline thread value must come from a previous real diagnostic rather
than a guessed default. Changing thread execution order can change
floating-point or nondeterministic trajectories, so check per-layout inputs,
actions, step counts, and termination conditions.

In the original experiments on 2026-10-27, requesting 1 thread hung on the first
reset; the failure was saved and the patch rolled back. **Do not treat 1 as a
usable or accepted setting.** Single episodes with 8 and 4 threads both
completed; per-substep physics time dropped by 7.27%, but execution time per
action dropped by only 0.60%, which is not enough to claim a stable overall
gain. The follow-up 8/4/4/8 runs of six episodes each were cancelled and rolled
back, and thread setting trials are not being continued for now.

## Apply, check, and roll back

First confirm that all processes from this run have exited and the GPU is idle.
Keep measurement and episode-count applied; keep parallel,
async-video, and unpaced-render in the before state. If the measurement version
differs, roll back with the old tools first, then apply and check with the new
tools.

```bash
BASE=/path/to/robodojo
ROOT="$BASE/RoboDojo"
RECIPE=/path/to/committed/examples/robodojo
PY="$HOME/miniconda3/envs/RoboDojo/bin/python"

"$PY" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch physics-threads # before
"$PY" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch physics-threads
"$PY" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch physics-threads # after
"$PY" "$RECIPE/run_eval.py" "$ROOT" --mode measured --seed 1 --num-envs 1 \
  --episodes 1 --label physics-thread-diagnostic --physics-threads 4 --timeout-s 600 \
  --execution-profile --python-profile
"$PY" "$RECIPE/verify_videos.py" /path/to/completed/diagnostic

"$PY" "$RECIPE/apply_runtime_patch.py" rollback "$ROOT" --patch physics-threads
"$PY" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch physics-threads # before
```

Verify complete videos for every baseline and candidate run; repeated layouts
do not count as new independent scenes. Use the existing compare_runs.py to
check result completeness and per-layout differences, and report measured and
end-to-end throughput, resources, success counts and rates, and limitations.
Only a stable gain moves on to 50-episode acceptance; until that acceptance
passes, this is not a default optimization.

The patch tool verifies the pinned sources and the complete before/after state,
and rejects unknown changes and partial application. This patch modifies the
startup/check entry point of render_sync.py and adds a helper; it does not
overlap with the existing measurement or episode-count patches. Keep all raw
JSON, logs, and videos outside the repository.
