# Control state reuse candidate

`0008-control-state.patch` only modifies the benchmark's robot manager. Both arms reuse the expanded joint targets within the same control call, and CPU state is built directly as independent NumPy arrays instead of cloning the whole Tensor and then copying it. State is still re-read after every physics step, and the GPU path keeps the original logic. The model, control frequency, action mapping, and physics parameters are unchanged.

Off by default. In the original experiments, a single-episode check showed about 37% less control processing time, but whole-episode actual steps/s rose by only about 1.3%. Both groups failed the task, so this is not a success-rate acceptance result.

First prepare the pinned sources and environments as described in the main [README](README.md), and apply isaac51-deps, episode-count, and measurement; keep the other candidates in the before state. After all evaluation processes exit, use the committed version of the tools:

```bash
RECIPE=/path/to/roboscope-kit/examples/robodojo
BASE=/path/to/robodojo
ROOT="$BASE/RoboDojo"
PYTHON=/workspace/robodojo-envs/controller/bin/python
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch control-state # before
# Run the off group first; the output label must be unused.
"$PYTHON" "$RECIPE/run_eval.py" "$ROOT" --mode measured --seed 1 --episodes 1 --label control-off-NEW
"$PYTHON" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch control-state
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch control-state # after
"$PYTHON" "$RECIPE/run_eval.py" "$ROOT" --mode measured --seed 1 --episodes 1 --label control-on-NEW --control-state-reuse
"$PYTHON" "$RECIPE/apply_runtime_patch.py" rollback "$ROOT" --patch control-state
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch control-state # before
```

`run_series.py` also supports `--control-state-reuse`. The entry point clears any inherited `ROBODOJO_CONTROL_STATE_REUSE`, sets it only from the explicit flag, and verifies the full patch bytes; it does not accept unknown runtime changes that are applied but not declared. For a per-component breakdown, add `--execution-profile` to both groups and report it separately from the profiling-off performance acceptance. Video checks and per-episode comparison use the main recipe tools. When updating the patch, roll back with the original version of the tools; do not force-overwrite the working tree.
