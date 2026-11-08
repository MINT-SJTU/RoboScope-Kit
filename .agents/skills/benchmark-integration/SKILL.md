---
name: benchmark-integration
description: Integrate a new simulation benchmark, or a new suite of an existing one, into RoboScope Kit, from the issue to the merged PR. Use when opening, working on or reviewing a `feat(<benchmark>)` issue or PR.
---

# Benchmark integration

Work through these steps in order. The acceptance gates they reach are in
[`packages/roboscope-kit/BENCHMARK_ACCEPTANCE.md`](../../../packages/roboscope-kit/BENCHMARK_ACCEPTANCE.md).
Throughput work follows in its own `perf(<benchmark>)` issue once the integration is accepted.

1. **Open the issue.** Use the benchmark-integration template, title it `feat(<benchmark>): ...`, and
   assign yourself while you work on it. If you pause, unassign yourself and note where the work stands.
   Get the maintainer's approval for any license or download prerequisite.
2. **Freeze the scope.** Pin the upstream benchmark, simulator, assets and model, each external download
   with its checksum. List every upstream evaluation task and variant in `benchmark-manifest.json`. Write
   the protocol document (robot, observations, actions, cameras, control rate, episodes, seeds or initial
   states, time limits, success and termination rules, aggregation) and link the upstream source for each
   setting. Pick a real policy with a public checkpoint for every task.
3. **Build the environment** with `benchmarks/<benchmark>/build-runtime.sh <install-dir>` from an empty
   directory, keeping every pin and source patch in the script or its pin files. Make `--self-test`
   create one environment, reset it and render a frame.
4. **Write the adapter, worker and factory** under
   `packages/roboscope-kit/src/roboscope_kit/runtime/benchmarks/<benchmark>/`, and register the benchmark
   in `benchmark_specs.py`, `registry.py`, `suite_sizes.py` and `shard_reduce.py`.
   - Call the SDK through its public API; add a public hook to the SDK in the same PR when the worker
     needs one.
   - Make `--quick-test` run the full evaluation path at its smallest size, such as one task with the
     benchmark's native environment grouping.
   - Add simulator-free tests for config validation, `aggregate` record checks and the shard merge.
   - Bump the version, add the CHANGELOG entry, and update the package README and protocol document.
5. **Run every task** with the real policy. The episode count per task is the protocol's, or a smaller
   count the maintainer approves in the issue or PR. Record expected and actual counts per task.
6. **Compare with the native evaluator** on the same workload: same checkpoint, tasks, episode identities
   and initial conditions. Pair outcomes per episode, report per-task success, and locate the first
   observation or action divergence for each disagreement. Repeat a few tasks on the native path to
   measure its run-to-run variation.
7. **Deliver.** Rebase onto `main` with the next free version, get green CI on the exact head, store raw
   records, logs and commands durably (a release asset or the tech report's `data/`), and link them from a
   short PR description. After acceptance, add the benchmark to the top-level and package README tables
   under its upstream name.
