# LIBERO-Pro throughput work

This directory owns the frozen simulator build and the transport candidate for
[issue #21](https://github.com/MINT-SJTU/RoboScope-Kit/issues/21). CPU checks and a real-GPU single-suite
comparison have been completed; the **2x full-evaluation target has not been demonstrated**. The local
6 GiB GPU cannot load the pinned bf16 baseline with the simulator.

See [the protocol](../../packages/roboscope-kit/LIBERO_EVALUATION_PROTOCOL.md#11-libero-pro) for the
frozen task semantics, [the issue audit](ISSUE_21_AUDIT.md) for completed and outstanding work, and
[the performance review](PERFORMANCE_REVIEW.md) for earlier optimizations and negative results.

## Implementation

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
  Plain LIBERO and LIBERO-Plus retain their defaults. Rendering and native model batching are unchanged.

The implementation is in `runtime/benchmarks/libero_pro/{factory,adapter}.py` and the shared
`runtime/benchmarks/libero/worker.py` inside the package. The existing automatic K/g search remains in
`runtime/benchmark/autotune.py`; there is no benchmark-specific replacement search.

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
- The clean AutoDL runtime build and real-model Quick Test passed. One 500-episode suite was evaluated
  in both arms. Full-matrix repeats, analysis of score differences and >=2x formal throughput remain
  pending. No simulator build script is changed by this candidate.
- The candidate version is 0.8.0b31. Reconcile the version and dated changelog against main before merge;
  merging publishes through CI. The performance issue must remain open until its acceptance is met.
