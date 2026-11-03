# Disabling render wall-clock rate limiting with a fixed timestep

This is an off-by-default runtime candidate in `0006-unpaced-render.patch`,
supported only on the Isaac Sim 5.1 environment pinned by this recipe. A
single-episode diagnostic found the Kit main loop rate limit enabled at 25 Hz,
with render taking about 40.16 ms on average. Do not call all of this host time
GPU rendering time.

In the original experiments on 2026-10-26, off/on/on/off runs of six episodes
each were completed with profiling off. The two measured steps/s pairs were
1.1174× and 0.9639×, so there is no stable overall gain yet; the candidate stays
off and does not move on to 50-episode acceptance.

The [official Isaac Sim 5.1 examples](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/replicator_tutorials/tutorial_replicator_isaac_snippets.html)
list the main loop FPS rate limit separately from the PhysX frequency and
timeline settings. This candidate only sets `rateLimitEnabled=false` temporarily
inside `BaseEnv.render()`, calls the original render once, and restores the
original value in `finally`. It does not change frequency values or render
parameters.

When enabled, it requires the Isaac loop runner to be in manual mode with a step
size equal to the original rendering dt, and after each call it checks that the
physics step count and rendering dt have not changed. If these conditions do
not hold it raises an error rather than changing the timestep to compensate.
The original camera dispatcher fence, zero-delay setting, physics stepping,
success checks, model code, history/chunk, and three 640×480 / 25 FPS videos are
all kept. Removing the wall-clock wait may affect nondeterministic rendering, so
similar success totals alone do not prove trajectory equivalence.

## Apply, compare, and roll back

First stop the model and simulation processes owned by this run and confirm the
GPU is idle. `measurement`, `isaac51-deps`, and `episode-count` are applied per
the recipe; `parallel` and `async-video` stay in the before state. The native
entry point behaves unchanged when `ROBODOJO_UNPACED_RENDER` is not set.

```bash
BASE=/path/to/robodojo
ROOT="$BASE/RoboDojo"
RECIPE=/path/to/committed/examples/robodojo
PYTHON=/workspace/robodojo-envs/policy/bin/python

"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch unpaced-render # before
"$PYTHON" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch unpaced-render
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch unpaced-render # after
"$PYTHON" "$RECIPE/run_eval.py" "$ROOT" --mode measured --seed 1 \
  --num-envs 1 --episodes 1 --label unpaced-diagnostic \
  --unpaced-render --execution-profile --python-profile
"$PYTHON" "$RECIPE/verify_videos.py" /path/to/completed/diagnostic

# Performance comparison with profiling off, using the same measurement version, seed, and layouts 0–5.
export PYTHONPATH="$BASE/roboscope-kit-2792b6e/packages/roboscope-kit/src:$RECIPE"
"$PYTHON" "$RECIPE/run_series.py" "$ROOT" --rounds 1 --episodes-per-round 6 \
  --num-envs 1 --seed 1 --label unpaced-on --unpaced-render \
  --output "$BASE/evidence/unique-unpaced-on"
# After each group exits, decode all videos before switching; the off group must roll back and omit --unpaced-render.
"$PYTHON" "$RECIPE/apply_runtime_patch.py" rollback "$ROOT" --patch unpaced-render
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch unpaced-render # before
```

The tool verifies the pinned Git revision and the complete before/after content,
and rejects unknown changes and partial application. This patch modifies
different files from the measurement and episode-count patches. It is validated
only on the pinned Isaac Sim 5.1 main path and is not combined with other
candidates.
