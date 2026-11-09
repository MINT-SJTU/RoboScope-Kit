# LIBERO-Pro throughput work

This directory owns the frozen simulator build and throughput work for
[issue #21](https://github.com/MINT-SJTU/RoboScope-Kit/issues/21). One real-GPU full-matrix pair
measured 1.3791x formal throughput; the **2x full-evaluation target has not been demonstrated**.
The local 6 GiB GPU cannot load the pinned bf16 baseline with the simulator.

See [the protocol](../../packages/roboscope-kit/LIBERO_EVALUATION_PROTOCOL.md#11-libero-pro) for the
frozen task semantics, [the issue audit](ISSUE_21_AUDIT.md) for completed and outstanding work, and
[the performance review](PERFORMANCE_REVIEW.md) for earlier optimizations and negative results, and
[the full-matrix report](FULL_AUTO_2026-11-09.md) for the results and remaining correctness questions.
The [fixed-baseline report](FIXED_BASELINE_2026-11-09.md) compares a later #19-style fixed-K/g run
against the full optimized matrix and the fastest measured single-suite configuration.

## Implementation

For the native batch 1/2/4 experiment, see [the batch experiment instructions](BATCH_EXPERIMENT.md).
The runner records actual batch-size distribution, per-batch model/Policy wall times, sampled GPU memory,
formal episode and action-step throughput, scores and paired differences at fixed K/g and MPS mode.

For a matched MPS off/on experiment at fixed K=12/g=3, use
[`run_mps_comparison.py`](run_mps_comparison.py). The [MPS experiment guide](MPS_EXPERIMENT.md)
documents preview, warmup, timing, attachment checks and outputs. It reuses the existing Kit MPS
lifecycle and retains chunk camera rendering in both arms. The optimization summary records the
single-suite result and its limitations.

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

The implementation in `runtime/benchmarks/libero/camera.py` uses robosuite's public
`modify_observable(..., "sensor", ...)` API. Observable clocks continue to advance normally. It does
not toggle sensor enablement, alter sampling rates, skip physics, extend action chunks, or force a
new image after stepping. No simulator source or installed package is patched. The optimization
rejects versions other than pinned robosuite 1.4.0, additional image sensors, or RGB sensors with
corruption, filtering or delay. Reset/settle steps keep full rendering, and video recording disables
the gate automatically. Original sensors are restored on normal completion and exceptions, before
the next episode reset. Restoring via the public API incurs two validation renders per episode.

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
Evaluation wall time includes the extra cleanup validation renders. Check both 500-episode coverage
and paired outcomes, steps, inputs and timing. Then repeat with independently tuned K/g if needed.

Local verification uses the existing regression suite plus ad hoc execution of the upstream 1.4.0
Observable implementation: chunk lengths 1/2/8/16 retain identical sample clocks and consumed images;
normal and exceptional exits restore sensors, and delayed camera sensors are rejected. This does not
replace fixed-action image/state replay in the real simulator or StarVLA closed-loop validation.
On 2026-11-08 (Python 3.12.3), all 187 existing regressions, the required Ruff scope and the official
CPython lifecycle check passed. An additional 24 paired Worker checks covered variable chunk lengths,
early termination, a horizon ending inside a chunk, and video fallback with identical consumed images,
steps and video frames. These checks use synthetic camera pixels with the real Observable clock;
they are not measurements of GPU throughput.

## Stage timing

Set `ROBOSCOPE_TIMING_DIR` to a fresh absolute directory before launching the controller, and ensure
the Policy environment inherits it. Install the same Kit revision in controller and model environments.
Use separate directories for warmup and formal execution. `worker-<pid>.jsonl` records each Work Item;
`server-<pid>.jsonl` records each decoded request and its request ID/operation. Payloads are not logged.
Summarize a formal timing directory with
`python benchmarks/libero_pro/analyze_timings.py /runs/formal-timing --output /runs/timing-summary.json`.
Failed captures have `ok: false`; a successful transport capture does not imply a successful policy
response, so also inspect normal evaluation error/recovery outputs.

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

Recorded on 2026-11-07: Intel Core i7-12700H, WSL2 Linux x86_64, Python 3.12.3, Kit 0.8.0b31 working tree,
NumPy 2.5.4 and MessagePack 1.2.4. Each mode has five repeats of 300 measured requests, with 20 warmup
requests and eight actions per chunk. The last chunk of each episode consumes only five actions.
Times include Worker observation construction, transport, server payload verification and action-queue
consumption; they exclude model inference, simulation and startup. Standard deviation is across repeat
means, not a confidence interval.

| Encoding | Connection | Mean ms/request | Standard deviation |
| --- | --- | ---: | ---: |
| JSON | Per request | 7.890 | 0.225 |
| JSON | Per episode | 7.371 | 0.244 |
| MessagePack | Per request | 1.512 | 0.089 |
| MessagePack | Per episode | 1.158 | 0.192 |

All measured payloads/actions matched. The default candidate reduces this local request cost by about
5.2x; enabling connection reuse reduces it further. Neither number establishes full-evaluation speedup.
Raw measurements are in [measurements/policy-io-2026-11-07.json](measurements/policy-io-2026-11-07.json).
Connection reuse stays opt-in pending real-model ablation evidence. A subsequent real-model fixed-K12/g3
pair on an AutoDL RTX 4090 covered 500 `libero_goal_object` episodes per arm: JSON baseline 431.742 s,
MessagePack candidate 424.384 s (1.0173x formal episodes/s), with 84,212 versus 82,948 policy steps.
Nineteen paired successes and 46 step counts differed. See [the issue audit](ISSUE_21_AUDIT.md) for
the experimental settings, data locations and limitations.

Validation on 2026-11-08, Python 3.12.3:

- Existing regression suite: **187 passed**; no new test cases added.
- Required Ruff scope and the measurement utility: passed.
- Package wheel build/install, dependency consistency and import of all 78 installed SDK modules: passed.
- Official CPython lifecycle policy check: passed for declared Python 3.11 through 3.15. Only Python 3.12
  executed locally; the other interpreters and the full distribution matrix require CI.
- JSON-only advertisement: all eight mode/repeat combinations negotiated JSON and preserved payloads,
  action ordering and episode boundaries. See [raw fallback evidence](measurements/json-fallback-2026-11-08.json).
- Ad hoc failure injection: reset, act and end-episode failures, with reuse both off and on, each
  propagated once, closed the client and did not replay the request.
- Ad hoc adapter checks: all 16 suites x four modes produced the same 8000-episode matrix and retained
  BDDL instructions. Worker flags/environment matched each mode; string, integer and null values were
  rejected for the boolean connection setting. This checks task construction, not executed episodes.

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

## Acceptance and delivery status

- Each complete matrix must contain 16 suites x 10 tasks x 50 states = 8000 terminal episode records.
  Match suite/task/episode identities, successes and actual executed steps, and retain failed runs.
  Report per-suite official metrics and paired differences. Aggregate success equality alone is not
  evidence of identical trajectories. Task reset history matters; do not reseed episodes to hide it.
- Formal throughput is total completed episodes divided by the sum of `evaluation_s` in each run's
  `metrics.json` (require the `evaluation_s` completed stage). Do not substitute `result.episodes_per_s`,
  which uses a different wall interval,
  or average suite speedup ratios. Also report total actual steps divided by formal time.
- Report checkpoint preparation, tuning, model loading, runtime preparation and external process wall
  time separately. Fixed-K/g runs measure zero tuning cost; they cannot establish tuning-inclusive gain.
- Compare a frozen baseline and candidate on the same host/resource allocation. The earlier correctness
  baseline used mixed GPUs and K/g settings and cannot establish a fair 2x denominator. Record both Git
  revisions, runtime pins, checkpoint/config hashes, CPU/GPU model, driver, memory, co-tenant load and
  run order. Use repeated paired matrices and report uncertainty.
- The clean AutoDL runtime build, Quick Tests and one pair of complete 8,000-episode matrices passed
  coverage checks. Formal throughput was 1.3791x; full-matrix repeats, analysis of score differences
  and >=2x formal throughput remain pending. No simulator build script is changed by this candidate.
- The tested candidate version is 0.8.0b37. Reconcile the version and dated changelog against main before merge;
  merging publishes through CI. The performance issue must remain open until its acceptance is met.
