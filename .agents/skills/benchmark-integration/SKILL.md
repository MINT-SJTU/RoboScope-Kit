---
name: benchmark-integration
description: Integrate a new simulation benchmark, or a new suite of an existing one, into RoboScope Kit, from the issue to the merged PR. Use when opening or working on a `feat(<benchmark>)` issue or PR, or when reviewing one.
---

# Benchmark integration

The acceptance gates are in
[`packages/roboscope-kit/BENCHMARK_ACCEPTANCE.md`](../../../packages/roboscope-kit/BENCHMARK_ACCEPTANCE.md).
This skill is the order of work that reaches them, and the mistakes reviews keep finding. Throughput is
separate: open a `perf(<benchmark>)` issue after the integration is accepted.

## Workflow

1. **Issue.** Open `feat(<benchmark>): ...` from the benchmark-integration template and claim it only while
   you are working on it. If you stop, unassign yourself and write where the work stopped. Name any
   license or download prerequisite (for example a proprietary simulator) and get the maintainer's
   approval before depending on it.
2. **Freeze the scope.** Pin the upstream benchmark, simulator, assets and model, with a checksum for every
   external download. List every upstream evaluation task and variant in `benchmark-manifest.json`. Write
   the protocol document (robot, observations, actions, cameras, control rate, episodes, seeds or initial
   states, time limits, success and termination rules, aggregation), and link the upstream source for
   each choice: the paper section, or the official evaluation script and config at the pinned commit.
   Choose a real policy with a public checkpoint for every task.
3. **Build the environment.** Only `benchmarks/<benchmark>/build-runtime.sh <install-dir>`, from an empty
   directory, with every pin and source patch in the script or its pin files. `--self-test` must create
   one environment, reset it and render, not only import modules.
4. **Write the adapter.** Adapter, worker and factory under
   `packages/roboscope-kit/src/roboscope_kit/runtime/benchmarks/<benchmark>/`, registered in
   `benchmark_specs.py`, `registry.py`, `suite_sizes.py` and, for sharding, `shard_reduce.py`.
   - Use only the SDK's public API. If the worker needs something `PolicyClient` or `profiling` does not
     expose, add it to the SDK in the same PR.
   - `--quick-test` runs the same code path as the full evaluation, at the smallest size (for example one
     task with the benchmark's native environment grouping), not a separate smoke script.
   - Add simulator-free tests: config validation, `aggregate` rejecting duplicate, missing, out-of-range
     and malformed records, and the shard merge.
   - Bump the version, add the CHANGELOG entry, and update the package README and protocol document.
5. **Run every task.** A real-policy run that covers every task in the manifest. The maintainer may approve
   fewer episodes per task in the issue or PR (for example 5 instead of 50); the task list is never
   reduced. Check one terminal record per expected episode and record expected and actual counts per task.
6. **Compare with the native evaluator.** Same checkpoint, tasks, episode identities and initial
   conditions, on the same workload as step 5. Pair outcomes per episode, report per-task success, and
   find the first observation or action divergence where outcomes differ. Repeat a few tasks on the
   native path to measure its variation. Close totals are not evidence of agreement.
7. **Deliver.** Rebase onto `main` and resolve conflicts (take the next free version). Get green CI on the
   exact head; a fork PR waits for a maintainer to approve the run. Put raw records, logs and commands
   somewhere durable (a release asset or the tech report's `data/`) and link them from the PR. Keep the PR
   description short and factual.

## Documentation rules

- Use the upstream official benchmark name (for example RoboCasa GR-1 Tabletop, not RoboCasa). Benchmarks
  of the same family with different robots, tasks or protocols are separate rows and are never averaged.
- List the benchmark as supported in the top-level and package READMEs only after it passes acceptance.
  Until then it is partial and labeled so.
- READMEs stay brief; put protocol detail and caveats in the protocol document or the validation record.
- Do not commit scripts that only patch one run's data; run them on the machine and report the result.
- Profiling uses `nsys` injection; do not add `nvtx` or other profiling packages as dependencies.

## Common review findings

- A one-task smoke run presented as a working integration.
- `--quick-test` bypassing the native grouping or termination logic the full run uses.
- A new adapter with no tests.
- A protocol choice (scene split, episode count, termination) without an upstream source.
- Private SDK calls (`client._request`, `profiling._range`) instead of a public hook.
- A draft left with merge conflicts and no CI run.
