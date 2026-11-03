# Optional overlapped video writing experiment

Status: the code passed CPU and real-ffmpeg validation; **there is no GPU
evaluation speedup result yet**. Run the full baseline from a separate frozen
checkout; do not apply this patch to a baseline directory that is in use.

Upstream already encodes frames as a stream and does not buffer whole episodes.
This patch only moves the blocking ffmpeg stdin write to a separate thread so it
can overlap with the following simulation steps. It does not change camera
rendering, the control loop, model inputs, or encoding quality. Each camera
queues at most two frames plus the one being written. The producer creates an
independent `bytes` object first, so the queue never references an array that
the next capture may overwrite. When the queue is full the producer waits; no
frames are dropped, overwritten, or skipped on failure. For three 640×480 RGB
streams, queued plus in-flight payload is at most about 7.91 MiB, excluding call
stacks, pipe buffers, and the encoder's own buffers; do not read this number as
total process memory.

The FIFO writer thread drains fully before close and uses the original ffmpeg
arguments (libx264, CRF 23, yuv420p, 25 FPS). If the queue makes no progress or
draining takes longer than 120 seconds, it raises an error. Abort terminates the
encoder first and then waits for the writer thread, which avoids a deadlock from
closing a pipe that is still in use. Upstream finalize error handling lets the
evaluation continue, so you must run the video checks below after the run; exit
code 0 alone is not enough to accept the result.

## Apply, check, roll back

Create a separate source checkout, complete the pinned source, asset, dependency,
and measurement patch preparation from the recipe, and use the committed version
of the tools. `async-video` only modifies `utils/save_file.py` and adds
`utils/recipe_video_pipe.py`; it shares no files with the measurement, parallel,
or dependency patches. It does not modify `XPolicyLab/policy/Simate_beta` or the
SDK.

```bash
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch async-video # before
"$PYTHON" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch async-video
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch async-video # after

# Run a matching small-scale check first, then the full rerun; background writing is enabled only by the explicit flag.
"$PYTHON" "$RECIPE/run_eval.py" "$ROOT" \
  --mode measured --seed 1 --episodes 50 --label video-on --async-video

# Run after the evaluation finishes, so its cost is reported outside the measured run; replace with the printed evidence directory.
"$PYTHON" "$RECIPE/verify_videos.py" /path/to/completed/run
"$PYTHON" "$RECIPE/compare_runs.py" /path/to/baseline/run /path/to/video-on/run

# Roll back after all evaluation processes exit; safe to repeat.
"$PYTHON" "$RECIPE/apply_runtime_patch.py" rollback "$ROOT" --patch async-video
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch async-video # before
```

Native baseline runs must roll back this patch; the launcher rejects unknown
runtime changes. `run_eval.py` clears any inherited `ROBODOJO_ASYNC_VIDEO` and
sets it to 1 only for `--async-video`. `run_series.py` supports the same flag;
tasks, layouts, rounds, and concurrency must match the full baseline. Do not
start a candidate measurement or switch its code while a baseline run is using
the same GPU.

## Validation and open measurements

With Python 3.13.9 and ffmpeg 7.1 on a CPU host, the synchronous and asynchronous
paths each encoded 30 640×480 RGB frames at 25 FPS. Inputs were non-contiguous
views overwritten immediately after append; after decoding, the RGB byte digests
of both paths were identical. Also validated: a 200-frame FIFO, the two-frame
queue limit, timeout, broken pipe, short write, abort while blocked, repeated
close/abort, and repeated patch apply/rollback. These are output and cleanup
correctness checks, not RoboDojo GPU performance results.

After the full baseline finishes, run an off/on/on/off small-scale check first,
then the same single-task 50-episode comparison. Each episode must have three
videos with correct success/failure labels, 640×480 at 25 FPS, with a decoded
frame count of `executed_steps + 1`, and no temporary or extra files. The check
tool also accepts a full series directory. Report separately the video wait and
finalize time, measured throughput, actual steps/s, change in success rate, and
total time including loading; report initial-image differences and action
trajectory differences separately. Encoding already partly overlaps with
simulation, and the copy and thread have their own cost, so the gain may be
small or even negative; this page does not claim the 2× throughput goal in
advance.
