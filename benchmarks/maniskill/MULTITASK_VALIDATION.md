# ManiSkill multi-task integration validation

Measured on 2026-11-07, Kit **0.8.0b39**, code snapshot
`b3f6729b1bde2c49144216a3f582cc30745aa63c`, based on main `44fa1ad`.
The submission uses byte-identical SDK, model and simulator-build inputs. The evidence includes
file hashes for checking the submitted source independently of the measured checkout.

The adapter is manifest-driven. Its catalog has 18 tabletop tasks with separate state/RGB profiles;
acceptance uses three official PPO models and representative transport/asset checks, without running
trained-model baselines for every task. Existing Policy APIs, other benchmarks, scheduling and
Policy-only MPS remain unchanged. Runtime optimization is separate work under issue 31.

## Environment and checks

RTX 4090, 49,140 MiB, driver 580.105.08; Xeon Platinum 8470Q with a 20-core container quota.
CPython 3.11.15, Torch 2.6.0/CUDA 12.4, ManiSkill 3.0.0b15, SAPIEN 3.0.0b1, NumPy 1.26.4.
A new prefix at `/root/autodl-tmp/maniskill-multitask-20261107/runtime-final` built in **52 seconds**
using verified source/package caches and a local Python mirror. This is not an uncached download time.

- Committed-builder N1/N2 self-tests pass for PickCube state, PushT's 3D controller and actual CUDA RGB.
- Real-model Quick Test passes 1/1. The complete three-model profile finishes 300 episodes / 23,000 steps.
- K2/G2 routing finishes 30 episodes / 2,300 steps; every record matches its K1/G1 reference.
- The original pinned upstream PushT evaluator passes with N10: 10 episodes / 1,000 steps, evaluation only.
- RGB, dual-Panda and external YCB loading/transport checks each finish two full-horizon episodes with
  a zero-action diagnostic Policy. These check connectivity, not learned task success.
- The real PPO actor also passes a two-episode `state_dict` run, 2/2 success.
- 188 existing pytest cases, Ruff, Bash syntax, ShellCheck and the Python-support check pass.
- Independent normal wheel/sdist builds, Twine, clean installation, pip check, 83 installed-module
  imports and 188 existing tests pass in a separate CPython 3.11 environment without GPU dependencies.
  All 85 installed SDK Python files match accepted source; other declared interpreters remain CI coverage.

## Correctness baseline

Fixed K1/G1, N10, seeds 0-99 per task, full task horizons, deterministic FP32 actors, MPS and autotune off.
Both SDK and native use create/close per vector group.

| Task | Episodes | SDK/native success-once | SDK/native success-at-end | Steps |
| --- | ---: | ---: | ---: | ---: |
| PickCube-v1 | 100 | 100 | 100 | 5000 |
| RollBall-v1 | 100 | 63 | 0 | 8000 |
| PushT-v1 | 100 | 98 | 72 | 10000 |

All **300 paired records** match exactly in initial-state SHA, first action, success-once,
success-at-end and terminal steps; maximum first-action difference is 0.0. Intermediate trajectories
are not compared. Overall success-once is **261/300 (87%)**, with 39 unsuccessful task episodes.
Final runs have zero execution errors, OOMs, retries/recoveries, timeouts, missing/duplicate records or
remaining queue items. Expected horizon truncation is separate from an execution timeout.
RollBall's **63%** success-once rate provides the requested 30-70% task for later correctness checks.

## Timing and resources

One matched integration pair; these are baseline measurements, without a repeated speedup claim.

| Metric | SDK K1/G1 | Native GPU N10 |
| --- | ---: | ---: |
| Complete command s | 108.447 | 99.511 |
| Formal evaluation s | 104.352 | 97.879 |
| Episodes/h | 10349.57 | 11034.07 |
| Control steps/s | 220.41 | 234.98 |
| GPU mean busy % | 27.472 | 28.315 |
| GPU sampled peak busy % | 61.000 | 63.000 |
| Mean VRAM GiB | 4.424 | 4.000 |
| Sampled peak VRAM GiB | 5.105 | 4.631 |
| Waited-process CPU cores | 1.071 | 0.993 |

SDK formal time is `metrics.json.evaluation_s`; native `formal_wall_s` includes its script import/setup
boundary and simulator lifecycle. The boundaries differ, so no cross-implementation speedup is asserted.
The original upstream entry-point check uses its own reset schedule and is a separate GPU-vectorized smoke.
Device NVML totals are sampled every 250 ms; CPU is the waited process tree. Peaks are sampled values.
GPU busy describes temporal activity, not SM saturation.

SDK checkpoint preparation is 0.0002 s, Policy load 1.684 s and runtime preparation 0.604 s.
No tuning probes run. Complete command time includes loading and final output/cleanup; cold uncached
asset installation is accounted separately.

| Worker phase | Sum s | Share of formal time, K1 | N10 group P50/P95 ms |
| --- | ---: | ---: | ---: |
| Environment creation | 19.778 | 19.0% | 564.5 / 654.6 |
| Explicit reset | 13.461 | 12.9% | 441.5 / 535.2 |
| Rollout | 58.999 | 56.5% | 1989.3 / 2464.1 |
| Environment close | 7.446 | 7.1% | 232.6 / 362.1 |

These are CPU wall intervals across 30 sequential vector groups, not independent per-lane latencies.
Rollout contains `env.step` with observation work (51.526 s), RPC waiting (6.550 s), observation staging
(0.214 s) and loop overhead. Mean Policy round trip is **2.848 ms** over 2,300 requests.
Nested rollout intervals must not be added again to the main phase total. CPU timing does not isolate
asynchronous GPU PhysX execution. Pure render time, server GPU inference, server queue wait and
SM/Tensor/RT throughput counters were not collected and remain N/A. Profiling for issue 31 will separate them.

## Assets and reproduction

YCB installation passes through the committed installer: 470 files / 87,216,221 bytes, tree SHA
`f7ac64d414f2b099ef4950ff0809ca342de9b82557c01ced4914532388c0a0cc`.
The immutable official archive retains the original source checksum. ZIP/GZIP cache and regular-file
asset handling were exercised through the real pinned upstream downloader.
AssemblingKits, PickClutterYCB and TurnFaucet asset archives remain unavailable during the
[UCSD upstream outage](https://github.com/mani-skill/ManiSkill/issues/1489); their configurations and
selectable installer are included. The [build guide](README.md) records asset sources and terms.
`/autodl-fs` rejected actual writes on this host, so assets use the explicit root-filesystem fallback.

```bash
ROOT=/root/autodl-tmp/maniskill-multitask-20261107
PY="$ROOT/runtime-final/env/bin/python"
KIT="$ROOT/source-final/kit"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$PY" -m roboscope_kit.runtime.benchmark --benchmark maniskill --suite baseline_state_v1 \
  --benchmark-manifest "$ROOT/runtime-final/benchmark-manifest.json" --checkpoint-source "$ROOT/models" \
  --policy-config "$ROOT/roboscope-policy.json" --concurrency 1 --replicas 1 --output-dir /path/to/fresh-sdk
"$PY" "$KIT/examples/maniskill/native_eval.py" --suite baseline_state_v1 \
  --benchmark-manifest "$ROOT/runtime-final/benchmark-manifest.json" --checkpoint "$ROOT/models" \
  --num-envs 10 --output /path/to/fresh-native/result.json
```

For a real Quick Test replace fixed K/G flags with `--quick-test`. For parallel routing add
`--episodes 10 --concurrency 2 --replicas 2` in place of the fixed K1/G1 flags.
The [model example](../../examples/maniskill/README.md) has portable setup and pinned weight downloads.

Raw results, command argv, resource samples, logs, initial failures and reproduction scripts remain
outside Git. `multitask-final-acceptance-summary.json` records 101 artifact hashes.
`evidence-final.tar.gz` contains 326 independently verified members (556,778 bytes), SHA-256:
`6b3c4a6b2dcf61e14b3347327e626c065f1690bea6df06ee82522b10e8fe1807`.
[Download raw results and logs](https://github.com/QLL-CCDD/RoboScope-Kit/releases/download/maniskill-issue25-evidence-20261107/evidence-final.tar.gz),
[artifact manifest](https://github.com/QLL-CCDD/RoboScope-Kit/releases/download/maniskill-issue25-evidence-20261107/evidence-final-manifest.json),
and [independent verification](https://github.com/QLL-CCDD/RoboScope-Kit/releases/download/maniskill-issue25-evidence-20261107/evidence-final.verification.json).
[Submission source equality](https://github.com/QLL-CCDD/RoboScope-Kit/releases/download/maniskill-issue25-evidence-20261107/submission-source-equality.json)
confirms all 85 SDK Python files and 11 substantive model/build/pin files match the accepted inputs.
The SDK Python tree digest is `9dd04857569e4de82e0becd32aa4f0d94e6def5993972f714f3b0ae94e190104`
(SHA-256 of compact JSON rows `[relative_posix_path, file_sha256]`, sorted by path).
