# LIBERO-Pro throughput draft review

Updated: 2026-11-09, against the revised [issue #21](https://github.com/MINT-SJTU/RoboScope-Kit/issues/21).
Suggested PR title: `perf(libero-pro): reduce policy and rendering overhead`.

This draft implements MessagePack selection, optional camera rendering on demand, and optional native
batching of queued Policy requests. It also includes diagnostic timing and reproducible ablation tools.
The selected draft result is **2.9141x formal throughput against the historical fixed configuration**,
with 8,000 episodes per side. Keep the baseline definition and configuration differences visible.
This is a draft for code review; repeated validation and final delivery remain outstanding.

## Review the execution path first

Paths in this table are relative to the repository root. K means simulation Workers; g means Policy
replicas. Queued batching is local to each Policy Server replica, not a single queue across all replicas.

| Priority | File and entry points | Review focus |
| --- | --- | --- |
| 1 | [`server.py`](../../packages/roboscope-kit/src/roboscope_kit/server.py): `_submit_action`, `_batch_actions`, `_handle_action_batch`, `_close_policy` | Collect only already-queued requests from distinct episodes; no collection timer. Check request/result ordering, per-client encoding, failure propagation, lock ordering and shutdown of waiting futures. Lifecycle calls use the existing request lock. Default cap is one. |
| 2 | [`starvla_policy.py`](../../examples/starvla/starvla_policy.py): `load`, `act`, `act_batch` | LIBERO-Pro QwenOFT alone may opt in above one. One native prediction per batch, per-row action unnormalization and transforms, strict output shape and finite-value validation. Check single-request behavior across the other StarVLA demos and numerical/closed-loop differences for batch sizes above one. |
| 3 | [`libero/worker.py`](../../packages/roboscope-kit/src/roboscope_kit/runtime/benchmarks/libero/worker.py): `PolicyActionQueue`, `run_episode`, `run_chunk` | Episode action-cache and socket lifecycle, no failed-RPC replay, camera demand from the actual remaining action queue, per-step termination and unchanged task reset history. Review shared LIBERO/LIBERO-Plus defaults as well as LIBERO-Pro. |
| 4 | [`libero/camera.py`](../../packages/roboscope-kit/src/roboscope_kit/runtime/benchmarks/libero/camera.py): `ChunkCameraGate.install`, `_sensor` | Skip only unconsumed RGB work while native sensor clocks advance. Preserve the image consumed at the next request. Check pinned-version/sensor guards, restoration on exceptions, settling and video behavior. Uses the public sensor replacement API; no installed simulator source patch. |
| 5 | [`libero_pro/adapter.py`](../../packages/roboscope-kit/src/roboscope_kit/runtime/benchmarks/libero_pro/adapter.py), [`factory.py`](../../packages/roboscope-kit/src/roboscope_kit/runtime/benchmarks/libero_pro/factory.py), [`benchmark-manifest.json`](benchmark-manifest.json) | Wire manifest settings into Worker arguments/environment and result metadata. MessagePack is preferred with JSON fallback; connection reuse and camera gating default to false. |
| 6 | [`client.py`](../../packages/roboscope-kit/src/roboscope_kit/client.py), [`timing.py`](../../packages/roboscope-kit/src/roboscope_kit/timing.py), [`starvla_timing.py`](../../examples/starvla/starvla_timing.py) | Optional diagnostics for RPC, queue/Policy, simulation and model stages. Check disabled-path overhead, nested/parallel timing interpretation, hook restoration and separation of profiled versus formal runs. |

Autotune, task affinity, Worker environment reuse, MessagePack framing and CUDA MPS lifecycle already
exist in the baseline. This branch selects or exercises them; it does not implement a new autotuner,
global scheduler, wire encoding or MPS manager. Persistent connections remain an optional ablation
with no demonstrated full-matrix benefit; all reported full runs leave them disabled.

## Files included in the draft

- **Runtime and configuration:** the execution-path files above belong in the implementation diff.
- **Reproduction tools:** `measure_policy_io.py`, `analyze_timings.py`, `capture_batch_observations.py`,
  `check_native_batch.py`, `run_mps_comparison.py` and `run_batch_comparison.py` belong here because they
  measure or validate the changed behavior. The drivers are fixed-K/g ablations, not the complete
  acceptance runner. Keep their associated [MPS](MPS_EXPERIMENT.md) and [batch](BATCH_EXPERIMENT.md) guides.
- **User documentation:** the benchmark and StarVLA READMEs, package README, evaluation protocol and
  profiling index describe switches, defaults, limits and protocol preservation.
- **Evidence:** retain only the [fixed-baseline comparison](FIXED_BASELINE_2026-11-09.md) and its
  reviewed JSON/CSV files in [`measurements/`](measurements/README.md): both configurations, suite/task
  scores, timing and 8,000 paired outcomes. Other experiment reports and results are archived locally
  under ignored `.local/issue21-draft-archive-20261109/` and excluded from the draft file tree.
- **Release metadata:** the implementation already carries Kit `0.8.0b37` in `pyproject.toml`, `uv.lock`
  and the changelog. This documentation/ignore cleanup does not change executable behavior or bump the
  version. Reconcile version conflicts and the actual merge date before merging.

Keep downloaded models/source snapshots, simulator/virtual environments, raw run directories,
profiles, observation dumps, videos, archives and host-specific launch/configuration files outside Git.
Use `.local/`, `runs/`, `outputs/`, `results/` or `artifacts/` at the repository root, or the corresponding
LIBERO-Pro output directories covered by `.gitignore`. Put private launch/configuration files in
`.local/`; do not put credentials into reproduction commands. Ignore rules deliberately do not match
all JSON, CSV or archives: committed manifests, evidence and RoboTwin's pinned seed archive are inputs.
An ignore rule does not remove an already tracked file; review the staged file list before committing.

## Selected result and follow-up

The [report](FIXED_BASELINE_2026-11-09.md) compares `ed8c9cf`/Kit b29 at historical fixed K/g with
`e63260e`/Kit b37 using per-suite diagonal tuning, MessagePack, camera gating, MPS and batch cap two.
Formal time decreases from 20,853.711 s to 7,156.150 s, giving **2.9141x** episode throughput.
This includes configuration changes and must not be described as the isolated benefit of batching.

Both sides have complete unique episode coverage. There are 288 differing success flags and 2,899
step counts; review per-suite/task results and within-side variation before making a correctness claim.
The report spells out OMP threads, K/g, controller Python, timing boundaries and run-order differences.
The public issue's best-configuration baseline, four-suite repeats and final-head checks remain
follow-up items. Keep this historical-fixed comparison distinct from those acceptance requirements.

Reviewed Worker and Policy profiling findings still need to be recorded with date, machine,
configuration and version in the benchmark docs and central index. Existing instrumentation does not
by itself establish a bottleneck. Before merge, reconcile upstream/version changes, complete required
checks and clean-build self-test, and provide final-head measurements and durable raw-log links.
This cleanup changes documentation, evidence selection and ignore rules, preserving measured
executable files; the selected measurements remain attributed to their original commits.

## Local review commands

```bash
git diff --stat origin/main...HEAD
git diff origin/main...HEAD -- packages/roboscope-kit/src/roboscope_kit/server.py
git diff origin/main...HEAD -- examples/starvla/starvla_policy.py
git diff origin/main...HEAD -- packages/roboscope-kit/src/roboscope_kit/runtime/benchmarks/libero
git status --short --ignored
```

`origin/main...HEAD` uses the locally fetched main merge base. This cleanup does not fetch/rebase main,
launch GPU experiments, push, open a PR or merge. Review the local branch before publishing the draft.
