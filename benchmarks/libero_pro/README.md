# LIBERO-Pro throughput work

This directory owns the frozen simulator build and throughput work for
[issue #21](https://github.com/MINT-SJTU/RoboScope-Kit/issues/21). The selected draft result is
**3.1006x formal throughput against the historical fixed configuration**, with 8,000 episodes per side
on the available dedicated RTX 4090. See the [fixed-baseline report](FIXED_BASELINE_2026-11-10.md) for
the exact configurations, per-suite scores and remaining validation.

Start code review with the [draft review guide](DRAFT_REVIEW.md). The
[evaluation protocol](../../packages/roboscope-kit/LIBERO_EVALUATION_PROTOCOL.md#11-libero-pro)
defines the frozen task semantics; the [evidence inventory](measurements/README.md) identifies the
committed results and raw archives. This draft comparison changes K/g and other execution settings;
it is not an isolated batching gain or a claim that all issue acceptance criteria are complete.

## Implementation

For the native batch 1/2/4 experiment, see [the batch experiment instructions](BATCH_EXPERIMENT.md).
The runner records actual batch-size distribution, per-batch model/Policy wall times, sampled GPU memory,
formal episode and action-step throughput, scores and paired differences at fixed K/g and MPS mode.

For a matched MPS off/on experiment at fixed K=12/g=3, use
[`run_mps_comparison.py`](run_mps_comparison.py). The [MPS experiment guide](MPS_EXPERIMENT.md)
documents preview, warmup, timing, attachment checks and outputs. It reuses the existing Kit MPS
lifecycle and retains chunk camera rendering in both arms. The driver is an ablation tool; the selected draft result is documented separately above.

- `benchmark-manifest.json` prefers the existing MessagePack transport. A JSON health exchange
  negotiates support; a server without MessagePack support falls back to JSON.
- `defaults.policy_encoding` accepts `json` or `msgpack`. The adapter sets the Worker process's
  `ROBOSCOPE_POLICY_ENCODING` explicitly, so the manifest controls the experiment.
- `defaults.persistent_policy_connection` is a JSON boolean, default `false`. If enabled, the Worker
  reuses its client within an episode, closes the connection at episode boundaries and Worker exit,
  and discards cached actions between episodes. A failed request closes the connection and propagates
  the error; the Worker never replays that request.
- Benchmark metadata records the requested encoding and connection policy. Each completed WorkResult's
  metadata records the negotiated `policy_encoding` and `persistent_policy_connection`; inspect these
  in `queue/completed/*.json` to detect JSON fallback.
- The shared Worker preserves BDDL instructions, initial-state selection, seed-once task environments,
  observation contents, action chunks, physics steps, settle steps, terminal checks and official scores.
  Plain LIBERO and LIBERO-Plus retain their defaults. Native model batching defaults to a cap of one;
  the full-matrix optimized arm used a cap of two.
- `defaults.chunk_camera_rendering` is an independent boolean, default `false`. It skips RGB sensor
  rendering inside action chunks, preserving native sampling clocks. Video runs use full rendering.

The implementation is in `runtime/benchmarks/libero_pro/{factory,adapter}.py` and the shared
`runtime/benchmarks/libero/worker.py` inside the package. The existing automatic K/g search remains in
`runtime/benchmark/autotune.py`; there is no benchmark-specific replacement search.

## Chunk camera rendering

`defaults.chunk_camera_rendering: true` enables the experimental camera optimization independently
of transport and K/g. The shipped default is `false` until real-simulator and closed-loop acceptance.
The Worker consumes the actual returned action queue, so eight actions are not hardcoded. After
popping the final action, it restores RGB sampling for that action's `env.step()`, producing the next
Policy input at the same native physics substep as full rendering. Intermediate steps return cached
images that are never sent to the Policy. State observables and success checks still run every step.

The implementation in `runtime/benchmarks/libero/camera.py` temporarily replaces the two private
robosuite Observable `_sensor` functions and restores them after the episode. The Observable clocks
continue to advance normally; no installed simulator source is patched. The gate assumes the pinned
robosuite 1.4.0 standard two-camera RGB setup without filtering, delay or corruption. It does not
validate these assumptions or other image sensors at runtime; changing the configuration requires
new equivalence checks. Reset/settle steps keep full rendering, and video recording disables the gate.
The original functions are restored on normal completion and exceptions, before the next episode reset.

Create separate manifests without modifying the built runtime; use absolute runtime paths so that
the copies remain valid outside the installation directory:

```bash
python - /sim/libero-pro/benchmark-manifest.json /runs/camera-configs <<'PY'
import json
import sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:])
destination.mkdir(parents=True, exist_ok=True)
config = json.loads(source.read_text())
config['runtime'] = {
    key: str((source.parent / value).resolve())
    for key, value in config['runtime'].items()
}
config['defaults'].update(policy_encoding='msgpack', persistent_policy_connection=False,
                          save_videos=False)
for enabled in (False, True):
    config['defaults']['chunk_camera_rendering'] = enabled
    with (destination / f'camera-{int(enabled)}.json').open('x') as output:
        json.dump(config, output, indent=2)
PY
```

Use the **same new Kit commit** for both arms. The old b30/b31 code does not implement this flag.
After image/state replay validation and a small closed-loop check, run each arm sequentially:

```bash
python -m roboscope_kit.runtime.benchmark \
  --benchmark libero_pro --suite libero_goal_object --episodes 50 \
  --checkpoint-source /models/starvla-libero-pro \
  --policy-config /path/to/policy.json \
  --benchmark-manifest /runs/camera-configs/camera-0.json \
  --concurrency 12 --replicas 3 --output-dir /runs/camera-off

python -m roboscope_kit.runtime.benchmark \
  --benchmark libero_pro --suite libero_goal_object --episodes 50 \
  --checkpoint-source /models/starvla-libero-pro \
  --policy-config /path/to/policy.json \
  --benchmark-manifest /runs/camera-configs/camera-1.json \
  --concurrency 12 --replicas 3 --output-dir /runs/camera-on
```

Each completed WorkResult records effective `chunk_camera_rendering`, `camera_renders` and
`camera_renders_skipped`. The counters count RGB sensor calls during policy action steps only;
installation/cleanup, reset/settle rendering and video/full-render runs are excluded. Both counters
are zero when the gate is inactive, not a claim that the simulator rendered nothing. For full
eight-action chunks, 7/8 of those RGB renders can be skipped; this is not an overall speedup estimate.
Check both 500-episode coverage and paired outcomes, steps, inputs and timing. Then repeat with
independently tuned K/g if needed.

For the current direct-sensor implementation, a 2026-11-10 fixed-action replay on the pinned AutoDL
RTX 4090 runtime compared 12 paired short episodes with camera gating on and off. Consumed RGB images,
physics states, actions, termination, RNG state and video frames matched in the tested cases, including
chunk lengths 1 and 8, mixed lengths 8/2/1 and video fallback. The
[case summary](measurements/camera-replay-2026-11-10.json) records the exact measured camera hash and
scope. This is not a proof for all episodes, sensor configurations or closed-loop model execution.

## Stage timing

Set `ROBOSCOPE_TIMING_DIR` to a fresh absolute directory before launching the controller, and ensure
the Policy environment inherits it. Install the same Kit revision in controller and model environments.
Use separate directories for warmup and formal execution. `worker-<pid>.jsonl` records each Work Item;
`server-<pid>.jsonl` records each decoded request and its request ID/operation. Payloads are not logged.
Summarize a formal timing directory with
`python benchmarks/libero_pro/analyze_timings.py /runs/formal-timing --output /runs/timing-summary.json`.
Failed captures have `ok: false`; a successful transport capture does not imply a successful policy
response, so also inspect normal evaluation error/recovery outputs.

Batch metrics reuse these spans through a request-scoped collector, even when stage JSONL is disabled.
This does not enable model hooks or CUDA events. The Policy stores no last-request duration, and the
Server caches the batch-metrics directory at construction.

| Span | Includes |
| --- | --- |
| `worker.reset`, `worker.set_init_state`, `worker.settle_step` | Initialization and settling, including their normal rendering |
| `worker.action_step` | Full native physics step, observable updates and RGB rendering |
| `worker.rgb_render` | Original RGB sensor call, nested inside action_step; excludes reset, settle and sensor restoration |
| `worker.build_observation` | Worker observation construction |
| `client.act.connect/send/receive` | Connect, serialize/send, or wait/receive/decode respectively; receive includes remote execution |
| `server.lock_wait` | Time after request decoding, waiting for the replica's request lock |
| `server.dispatch`, `server.policy_act`, `server.send` | Dispatch, nested Policy act, and response serialization/send |
| `policy.example_build`, `policy.postprocess` | Image/example preparation and action unnormalization/transforms |
| `policy.predict_total` | Framework prediction including input processing, transfer, GPU work and return-to-CPU synchronization |
| `policy.input_build.wall` | Qwen input preprocessing and upload together |
| `policy.vlm/gather/action_head.wall` | Host-call durations, potentially asynchronous or synchronizing |
| `policy.vlm/gather/action_head.cuda_interval` | Same-stream event intervals, including scheduling/host gaps, not exclusive kernel time |

Each span contains seconds and count. Filter server records to `operation=act` when comparing inference
requests. Never add nested durations (RGB is inside action_step; model stages are inside policy_act)
or treat sums across workers/replicas as elapsed time. Input upload and output copy are not isolated
transfer benchmarks. The framework's prediction residual must not be labelled pure D2H time.
The code does not reset sensor clocks, modify source installations, replay RPCs or change actions.
Timed camera-off runs wrap RGB sensors for measurement, so their render counts are nonzero.

## Stage timing findings, 2026-11-08

On an AutoDL RTX 4090 with a 12-core CPU quota and 96 GiB memory limit, Kit 0.8.0b35 at `d701fc6`
ran `libero_goal_object` at fixed K=12/g=3, bf16/SDPA, MessagePack, video and MPS off, and 500
episodes per camera setting. These diagnostic runs used stage instrumentation and the earlier public
sensor replacement implementation, not the later direct-sensor code. Warmup was separate.

In the camera-on arm, mean server lock wait was 170.100 ms per action request and Policy service
was 109.940 ms; mean full simulator action-step time was 12.933 ms. In the camera-off arm the
corresponding simulator action-step mean was 16.348 ms. The measured client send/serialization
mean was below 1 ms per request in both arms. These observations motivate investigating Policy
queueing and inference as well as camera work. Durations are inclusive and taken across parallel
processes: they cannot be added to obtain wall time, and CUDA event intervals include host gaps.
This was not an Nsight `--profile` run or a measurement of the current head.

## CPU transport reproduction

From the repository root, in Python 3.11 or newer:

```bash
python -m pip install ./packages/roboscope-kit imageio
python benchmarks/libero_pro/measure_policy_io.py --output /tmp/libero-policy-io.json
python benchmarks/libero_pro/measure_policy_io.py --json-only-server \
  --requests 20 --warmup 3 --repeats 2 --output /tmp/libero-policy-io-fallback.json
```

Use new output filenames for each invocation. The script requires permission to bind a local Unix
socket. It starts a separate CPU Policy Server, sends two 256x256 RGB arrays, proprioception and language
through the actual Worker action queue, and verifies dtype, values, action order, episode lifecycle and
encoding negotiation. Both episodes end mid-chunk to check that unused actions do not leak to the next
episode. It compares JSON/MessagePack with per-request/persistent sockets in alternating order. This is
a measurement utility, not a simulator, a model-quality check, or a replacement for Quick Test.

Connection reuse remains opt-in. CPU/socket microbenchmarks do not measure simulator or model
throughput and are not included in this draft's result set. The retained result is the complete
fixed-baseline comparison linked above.

## Local validation

The repository's regression commands are:

```bash
cd packages/roboscope-kit
python -m pip install . pytest ruff matplotlib imageio
PYTHONPATH=src:../.. python -m pytest tests -q
python -m ruff check src tests ../../benchmarks/robotwin
```

## Real-model reproduction on a GPU host

1. Build a fresh runtime using `benchmarks/libero_pro/build-runtime.sh /sim/libero-pro`. Do not patch an
   installed simulator. Install this Kit revision in the controller, simulator and model environments.
2. Follow the [StarVLA setup](../../examples/starvla/README.md) using its `libero_pro` demo and
   `starvla_policy.py`, not the separate technical-report LIBERO Policy. Prepare the snapshot with:

   ```bash
   /path/to/starvla-env/bin/python examples/starvla/prepare_snapshot.py \
     libero_pro /models/starvla-libero-pro
   ```

   The demo pins StarVLA `7e4a47360093d79dc42b883885c04fbe790c5b60`, checkpoint
   `a60fff4173dc3732796df370a15a9a2db7bd7e5e`, bf16, two input cameras and the LIBERO gripper transform.
   Set the absolute Python/script/cwd paths in a copy of `examples/starvla/roboscope-policy.json`.
3. Validate the simulator and run one real episode before any timing comparison:

   ```bash
   python -m roboscope_kit.runtime.benchmark --benchmark libero_pro --self-test \
     --benchmark-manifest /sim/libero-pro/benchmark-manifest.json \
     --output-dir /runs/libero-pro-self-test
   python -m roboscope_kit.runtime.benchmark --benchmark libero_pro --suite libero_goal_object \
     --checkpoint-source /models/starvla-libero-pro \
     --benchmark-manifest /sim/libero-pro/benchmark-manifest.json \
     --policy-config /configs/starvla-policy.json --quick-test \
     --output-dir /runs/libero-pro-quick-test
   ```

4. Create four experiment manifests outside the simulator install. Resolve runtime paths before moving
   the manifest, so its relative paths still address the same frozen runtime:

   ```python
   import json
   from pathlib import Path

   source = Path('/sim/libero-pro/benchmark-manifest.json')
   destination = Path('/runs/libero-pro-configs')
   destination.mkdir(parents=True, exist_ok=True)
   original = json.loads(source.read_text())
   for encoding in ('json', 'msgpack'):
       for persistent in (False, True):
           config = json.loads(json.dumps(original))
           config['runtime'] = {
               key: str((source.parent / value).resolve())
               for key, value in original['runtime'].items()
           }
           config['defaults'].update(policy_encoding=encoding,
                                     persistent_policy_connection=persistent)
           target = destination / f'{encoding}-{int(persistent)}.json'
           with target.open('x') as output:
               json.dump(config, output, indent=2)
   ```

5. Run paired small ablations at fixed K/g before the full matrix. Use the same CPU/GPU allocation,
   checkpoint, episodes, action chunk, video setting and MPS mode. Compare `json-0` against `msgpack-0`
   to isolate encoding, then `msgpack-0` against `msgpack-1` to isolate connection reuse. Use at least
   three repeats and alternate order. Separately run automatic tuning and inspect its `selection.json`;
   do not change K/g in a comparison labeled transport-only.
6. For full coverage, run all 16 suites with 50 initial states per task. This example is one fixed
   configuration and one repeat; choose K/g from measurements and repeat for each comparison arm:

   ```bash
   set -euo pipefail
   mode=msgpack-0
   repeat=1
   workers=1
   replicas=1
   for base in goal spatial 10 object; do
     for perturbation in object swap lan task; do
       suite="libero_${base}_${perturbation}"
       run_dir="/runs/libero-pro/${mode}/repeat-${repeat}/${suite}"
       mkdir -p "$(dirname "$run_dir")"
       /usr/bin/time -f '%e' -o "${run_dir}.wall-seconds" \
         python -m roboscope_kit.runtime.benchmark \
         --benchmark libero_pro --suite "$suite" --episodes 50 \
         --checkpoint-source /models/starvla-libero-pro \
         --benchmark-manifest "/runs/libero-pro-configs/${mode}.json" \
         --policy-config /configs/starvla-policy.json \
         --concurrency "$workers" --replicas "$replicas" --output-dir "$run_dir"
     done
   done
   ```

For bottleneck diagnosis, add `--profile` to a separate fixed-K/g or Quick Test run with Nsight Systems
available. Worker profiling does not capture Policy Server GPU kernels. Record profile findings in this
document and the central profiling index; measure performance again without profiling.

## Draft status and follow-up

- The draft presents one fixed-baseline comparison: 16 suites × 10 tasks × 50 states per side,
  complete unique episode coverage, 3.1006x formal throughput and 3.0979x actual steps/s.
- Keep per-suite/task scores and paired differences visible. The 314 differing success flags and
  2,960 differing step counts need repeat analysis; aggregate score similarity is insufficient.
- Formal throughput uses total completed episodes divided by summed `evaluation_s`, not
  `result.episodes_per_s` or an average of suite ratios. Report loading, preparation, tuning,
  result writing and end-to-end time separately; fixed runs have no tuning search.
- The public issue additionally requires main's best existing configuration, measured using diagonal
  autotune and a fixed K/g sweep on the same dedicated GPU/CPU/memory budget, plus three runs per side
  on `libero_goal_task`, `libero_spatial_lan`, `libero_object_swap` and `libero_10_object`.
  These remain follow-up work; the historical fixed denominator is stated explicitly in this draft.
- The earlier stage-timing findings are recorded above; Worker and Policy Server Nsight traces remain
  outstanding.
- The latest optimized comparison used Kit 0.8.0b38 at `dfebac7` plus the camera patch.
  The current draft also incorporates upstream autotune fixes absent from that run. Before
  merge, use the actual merge date in the changelog, run required checks and clean-build self-test,
  measure the exact head being merged, and publish durable raw-log/reproduction links.
