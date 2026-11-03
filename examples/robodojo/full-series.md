# RoboDojo single-task 50-episode evaluation

The baseline is **`stack_blocks`, single task, 50 episodes**: ARX-X5, the official Simate-beta checkpoint,
seed 1, fixed layouts 0–49, each layout run once. 50 rounds are no longer run. The Meta-World scope of
50 tasks × 50 episodes does not apply to this RoboDojo experiment. The earlier six-episode runs are a
pilot.

`recipe.toml` sets `full_episodes=50` and `pilot_episodes=6`. Task, robot, checkpoint, the 550-action limit,
FP32, input processing, model history, cameras, and video are unchanged. The model RNG keeps being consumed
across the round; episode reset does not reseed it.

## Running

First complete installation and the minimal native check from the main [README](README.md). `PYTHON` is the
upstream simulator environment (`$HOME/miniconda3/envs/RoboDojo/bin/python`) with
`PYTHONPATH="$SDK/packages/roboscope-kit/src"`; it uses the resource-measurement functions of
roboscope-kit pinned at `2792b6e160f06e24d18499b5ca5f50cca5f3fc31`. The reported results were measured with
an earlier, unreleased version of the kit; they are reference results from the original experiments and
have not been re-verified on roboscope-kit yet.
Source code comes from committed revisions; do not update its directory, the simulator source, or the
environments during a run.

```bash
# apply is idempotent for the same version; an older version must first be rolled back with the old tool.
"$PYTHON" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch episode-count
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch episode-count # after
"$PYTHON" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch measurement
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch measurement # after

# OUT is a result directory outside the repository that does not exist yet; logs also go outside the repository.
setsid nohup "$PYTHON" -u "$RECIPE/run_series.py" "$ROOT" \
  --profile "$RECIPE/recipe.toml" --seed 1 --episodes-per-round 50 --rounds 1 \
  --num-envs 1 --label baseline50 --output "$OUT" \
  >"$BASE/evidence/baseline50-launch.log" 2>&1 < /dev/null &
echo $!
```

Upstream `_task.yml` sets `eval_nums: 25` for `stack_blocks` alone, so even `EVAL_NUM=50` runs only 25
episodes. `0005-episode-count.patch` changes only this value to 50; other tasks keep their values. Both
comparison groups apply this patch, and results are labeled as extended scope. The launcher checks the
effective task limit and layouts 0–49 before loading the GPU. An earlier first run stopped at 25 episodes
because of this limit and is kept as an incomplete record; it is not combined with 25 episodes from a new
process to form the baseline.

`setsid` moves the process into its own process group, detached from the SSH session, and `nohup` ignores
hangup. The runner supports an explicit number of rounds, but this run must use `--rounds 1`. Record
requested=50 and unique_layouts=50; previously cancelled repeat runs do not count toward completion. The
default 14400-second timeout guards against hangs and does not change the per-episode action limit. An error
stops the run and leaves `complete=false`; missing episodes are not skipped or retried automatically.
Restart with a new OUT and keep the interrupted record.

`series.json` is written atomically at the end of the round. While running, read the child run directories
printed in `round-000.log` and their native `_result.json`. No continuous polling is needed. To cancel,
send SIGTERM to the **series main PID**; the runner gives the launcher up to 120 seconds to clean up the
model/simulator process groups it owns, which avoids killing the launcher early and leaving child processes
behind. Do not stop other jobs by process name.

## Timing and completeness

- Formal time runs from evaluation_start to evaluation_end. It includes reset, actual execution, video
  writing/finalization, and result output inside the window; it excludes model loading and simulator
  initialization.
- Total time includes model loading, initialization, child-process preflight, result archiving, and
  cleanup. The layout digest check before the start, downloads, and environment installation are listed
  separately. Autotune is not enabled; later on/off comparisons reuse the existing implementation and list
  search time separately.
- GPU is sampled every 500 ms and cgroup CPU/RAM every second; RAM includes file cache.
- Layouts 0–49 must each appear exactly once, with complete input/action digests and the correct step limit.
  Failed episodes stay in the denominator; unstable, missing, duplicated, or interrupted runs are never
  treated as complete by default.
- measurement records digests of the initial state/vision/instruction/layout, the model input actions, and
  the mapped control sequence, plus reset stability/settling and video finalization time. Timers are nested
  and cannot be summed as exclusive phases. Candidates and the baseline use the same measurement version;
  do not mix in old six-episode timings.
- After completion, outside the timed window, run `verify_videos.py "$OUT"` to check all 150 videos and that
  each decodes to `executed_steps + 1` frames. Upstream can log a finalize error and continue, so exit code 0
  alone is not sufficient for acceptance.

Prepare optimizations in a separate checkout so the running baseline is unaffected. Later candidates use the
same 50-episode scope; when repeated runs are needed, report each group separately and do not add groups
together into a larger suite.
