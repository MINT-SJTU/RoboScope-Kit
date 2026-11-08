# LIBERO-Pro optimization summary

Date: 2026-11-09. Scope: [issue #21](https://github.com/MINT-SJTU/RoboScope-Kit/issues/21).
Baseline: `44fa1adb934456c40df809a7baad27152182488d` (Kit 0.8.0b30).
Latest implementation: `e63260e117fb231706f0190fe6dcfe550afa2d0f` (Kit 0.8.0b37).
The baseline is the original revision used in our previous comparisons, not today's moving upstream main.

## Methods and changed files

Paths below are relative to the repository root. This inventories all 29 files changed between the
two revisions; this summary itself is a subsequent documentation-only addition.

| Method / purpose | Changed files | Behavior |
| --- | --- | --- |
| Policy transport | `packages/roboscope-kit/src/roboscope_kit/client.py`; `benchmarks/libero_pro/benchmark-manifest.json` | Prefer negotiated MessagePack; retain JSON fallback. Optional per-episode socket reuse is off in the experiments. |
| Camera rendering on demand | `packages/roboscope-kit/src/roboscope_kit/runtime/benchmarks/libero/camera.py`, `worker.py`; `packages/roboscope-kit/src/roboscope_kit/runtime/benchmarks/libero_pro/adapter.py`, `factory.py` | Skip unconsumed camera renders inside an action chunk; retain observation sampling points, physics steps and per-step termination checks. Opt-in; video uses the original rendering path. |
| Native model batching | `packages/roboscope-kit/src/roboscope_kit/server.py`; `examples/starvla/starvla_policy.py` | Coalesce already-queued requests from different episodes into one prediction; route each action chunk back to its own request. LIBERO-Pro QwenOFT only, default cap 1, no intentional collection delay. |
| Diagnostic stage timing | `packages/roboscope-kit/src/roboscope_kit/timing.py`; `examples/starvla/starvla_timing.py`; `benchmarks/libero_pro/analyze_timings.py` | Optional JSONL timings for queue/Policy, model processing and simulator stages; also instrument the client, server and worker listed above. |
| Transport experiments | `benchmarks/libero_pro/measure_policy_io.py`; `benchmarks/libero_pro/measurements/policy-io-2026-11-07.json`, `json-fallback-2026-11-08.json` | Transport microbenchmarks and fallback evidence. |
| MPS experiments | `benchmarks/libero_pro/run_mps_comparison.py`, `MPS_EXPERIMENT.md` | Controlled comparison using the Kit's existing private CUDA MPS lifecycle. MPS itself was already implemented upstream. |
| Batch experiments | `benchmarks/libero_pro/capture_batch_observations.py`, `check_native_batch.py`, `run_batch_comparison.py`, `BATCH_EXPERIMENT.md` | Real-observation numerical checks and caps 1/2/4 comparisons with coverage, actual batch-size and timing accounting. |
| Protocol and findings | `benchmarks/libero_pro/README.md`, `ISSUE_21_AUDIT.md`, `PERFORMANCE_REVIEW.md`; `examples/starvla/README.md`; `packages/roboscope-kit/README.md`, `LIBERO_EVALUATION_PROTOCOL.md` | Reproduction, limitations, behavior and comparison methodology. |
| Version history | `packages/roboscope-kit/pyproject.toml`, `uv.lock`, `CHANGELOG.md` | Version bumps and changes through 0.8.0b37. |

Workers, model replicas, task affinity, environment reuse, eight-action chunks and the `diagonal`
K/g heuristic already existed in the baseline. The heuristic implementation was not changed by this
optimization series: `packages/roboscope-kit/src/roboscope_kit/runtime/autotune.py` contains selection
rules; `runtime/benchmark/autotune.py` drives real-model probes; `runtime/benchmark/cli.py` invokes it.
Omitting fixed `--concurrency` / `--replicas` and specifying `--autotune-strategy diagonal` enables it.

## Evidence so far

On the same AutoDL GPU, separate single-suite comparisons measured approximately 1.017x for
MessagePack, 1.054x for camera gating, 1.330x for MPS and 1.120x for batch cap 2 versus cap 1 with
MPS enabled. These are separate experiments; do not multiply their gains. The latest cap-2 run took
269.478 seconds for 500 formal episodes, with 342 successes. The historical original baseline took
431.742 seconds, with 332 successes; that cross-experiment ratio is illustrative, not acceptance.

Existing local regressions passed (187 tests), as did Ruff and real-observation batch checks. The
cap-2 versus cap-1 suite still had 53 differing success flags and 255 differing step counts. Similar
aggregate scores and four numerical observation checks do not establish closed-loop equivalence.
The full matrix, paired task scores and reset-history/batch-numerics differences need review. The
changes have been committed and pushed to the fork; they have not been merged upstream.

## Full comparison launch

Run both revisions sequentially on the same host, each across all 16 released suites with 10 tasks
x 50 episodes: **8,000 episodes per arm, 16,000 total**, excluding Quick Tests and tuning probes.
Each suite independently uses the existing `diagonal` heuristic with unchanged search budgets.

| Setting | Original baseline | Latest implementation |
| --- | --- | --- |
| Kit revision | `44fa1ad` | `e63260e` |
| K/g | Independent built-in diagonal search per suite | Independent built-in diagonal search per suite |
| Transport | JSON | MessagePack |
| Chunk camera gating | Off | On |
| CUDA MPS | Off | On, including tuning |
| Native batch cap | 1 | 2 |
| Persistent connection / video / diagnostic tracing | Off | Off |

Both use the existing clean-built simulator, the same SHA-256 verified StarVLA checkpoint/config,
bf16/SDPA, seed 7, 10 settling steps, primary/wrist cameras, eight-action chunks, native horizons and
per-step termination checks. This is a comparison of the complete optimization package, not an
isolated attribution to batching or MPS. No simulator rebuild, download or fresh-cache timing is claimed.

The host-specific orchestration script is stored beside the remote experiment, outside the package.
It holds the existing experiment lock, verifies inputs, installs the exact Kit revision into both
dedicated controller/model environments, performs a Quick Test per suite, then runs:

```bash
"$EVAL_ROOT/control-env/bin/python" -m roboscope_kit.runtime.benchmark \
  --benchmark libero_pro --suite "$SUITE" --episodes 50 \
  --checkpoint-source "$EVAL_ROOT/model" \
  --benchmark-manifest "$ARM/benchmark-manifest.json" \
  --policy-config "$ARM/policy.json" \
  --autotune-strategy diagonal --output-dir "$CASE/run"
# Optimized arm additionally passes --cuda-mps and sets ROBOSCOPE_STARVLA_MAX_BATCH_SIZE=2.
```

The detached tmux session is `issue21-full-auto`. The latest output directory is recorded in
`/root/autodl-tmp/issue21-20261108/manual/latest-full-auto.txt`. It contains the launcher, exact
commands/configurations, per-suite logs/results, incremental `progress.json`, and final
`comparison.json` / `REPORT.md`. Errors stop the sequence and remain visible in `status.json` and
`launcher.log`; they are never silently dropped. Formal throughput uses 8,000 divided by summed
formal seconds. CLI wall (including tuning), warmup/setup and phase timings are reported separately;
the phase fields must not be double-counted. The final comparison records per-suite/task scores,
paired success/step differences, selected K/g, failures/OOMs and resources.

```bash
ssh -p 46252 root@connect.westd.seetacloud.com
tmux attach -t issue21-full-auto
# Detach without stopping: Ctrl-b, then d.
EVAL_OUT=$(cat /root/autodl-tmp/issue21-20261108/manual/latest-full-auto.txt)
cat "$EVAL_OUT/status.json"
tail -n 30 "$EVAL_OUT/launcher.log"
```

Once detached execution is confirmed, closing SSH or shutting down the local computer does not stop
the run. The AutoDL instance itself must stay running. Completing one full pair does not by itself
prove the issue's 2x target or correctness acceptance; inspect results and repeat key measurements.
