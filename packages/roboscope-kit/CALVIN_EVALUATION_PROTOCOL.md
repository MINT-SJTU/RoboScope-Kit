# CALVIN Evaluation Protocol

This document defines the SDK-native RoboScope CALVIN evaluation contract. The kit's benchmark runtime
owns benchmark enumeration, simulator processes, concurrency, recovery, and result aggregation. User
inference code owns only the public `Policy` lifecycle and model-specific preprocessing and
postprocessing.

`roboscope-kit` requires Python 3.11 or newer in every process that imports it, including the Policy and
the simulator workers. Python 3.8–3.10 are unsupported. See the
[Python support policy](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/packages/roboscope-kit/README.md#python-support).

## 1. Benchmark scope

The supported suites are the official CALVIN challenge splits `task_D_D`, `task_ABC_D`, and
`task_ABCD_D`. The selected suite declares the model's training split and all three evaluate on the
official environment D challenge. An official evaluation contains 1,000 ordered
long-horizon sequences. Each sequence contains one initial condition followed by five language subtasks.
The score is the number of consecutive subtasks completed before the first failure.

The runtime reads the ordered sequence list from the benchmark manifest's `runtime.eval_sequences_path`,
requires exactly 1,000 entries and the official SHA-256
`c3689911565037d399d779c655e9e943cbd5bfbfbf44045e2b3b1daa8b1a2317`, records that digest as
`eval_sequences_sha256` in the result metadata, and passes it to every evaluator worker. A worker exits
with an error if the file changes before evaluation starts. Probe runs use a prefix of the same
sequence list, long enough to give every probed worker `probe_items_per_worker` work items (three with the
default diagonal search); an official score uses all 1,000 entries.

### Environment and invocation

The CALVIN simulator is not a dependency of `roboscope-kit`. Build it with
[`benchmarks/calvin/build-runtime.sh`](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/calvin/build-runtime.sh) `<install-dir>`
(Linux x86_64, `uv`, network access). The script creates the virtual environment from
[`runtime-requirements.txt`](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/calvin/runtime-requirements.txt)
and verifies the CPU PyTorch wheel against
[`torch-requirements.txt`](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/calvin/torch-requirements.txt).
It checks out `mees/calvin` at `fa03f01` with `mees/calvin_env` at `797142c` and its `tacto` submodule, downloads
`eval_sequences.json` from starVLA revision `074ce65` and verifies the SHA-256 above, and copies
[`benchmark-manifest.json`](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/calvin/benchmark-manifest.json) into `<install-dir>`.
Every `runtime` path in that manifest is relative to the manifest file:

| `runtime` key | Path under `<install-dir>` | Contents |
| --- | --- | --- |
| `python` | `env/bin/python` | the CALVIN interpreter that runs evaluator workers |
| `benchmark_root` | `calvin` | CALVIN sources; workers read assets from `calvin/calvin_env/data` |
| `config_path` | `calvin/calvin_models/conf` | the official task oracle and validation annotations |
| `dataset_paths.<suite>` | `dataset/minimal_task_D_D` | the environment D evaluation inputs, shared by all three suites |
| `eval_sequences_path` | `eval_sequences.json` | the official 1,000 challenge sequences |

No CALVIN dataset download is needed: the official evaluator hard-codes the initial states and reads no
episode data. The script assembles `dataset/minimal_task_D_D` from the pinned sources: the two
configuration files from `calvin_models/conf` and a `validation/.hydra/merged_config.yaml` composed from
`calvin_env/conf` with the long-finger Panda robot of CALVIN's recorded validation environment. It then
resets and steps the environment once to prove the configuration, assets, and dependencies work together.

Workers use the CALVIN environment's CPython 3.11.15 and import the kit through `PYTHONPATH`.
Use a fresh install directory when migrating a Python 3.8 runtime.

Run an evaluation with the kit installed in any supported Python:

```bash
python -m roboscope_kit.runtime.benchmark \
    --benchmark calvin --suite task_ABC_D --checkpoint-source /models/my-checkpoint \
    --benchmark-manifest <install-dir>/benchmark-manifest.json \
    --policy-config /path/to/roboscope-policy.json \
    --output-dir /runs/calvin-task_ABC_D
```

`--self-test` with the same manifest checks that every runtime path exists. Results, progress, and logs
are written under `--output-dir`.

Use a new or empty output directory for each evaluation or self-test. A directory containing only
`errors.jsonl` can be retried, preserving earlier errors; other run artifacts require a new directory.
Keep shell and scheduler logs outside the output directory.

For the pinned StarVLA QwenGR00T demo, apply the action-noise patch to the Policy's StarVLA checkout
at revision `985c2238627481f25476fbee478083812dd82a21`.

```bash
git -C /path/to/starVLA apply /path/to/RoboScope-Kit/examples/starvla/calvin/starvla-action-noise.patch
```

The default `--eval-mode strict` serves each `act` request individually, even when the Policy
supports coalescing. Choose `--eval-mode fast` to let a Policy with
`coalesce_action_requests=True` group queued requests from distinct episodes. The result records
`dynamic-batching` in `active_fast_optimizations` when this optimization is enabled.

For the StarVLA demo, set `inference.action_noise_seed: 0` and `inference.max_batch_size: 8` in the
snapshot's `config.yaml`. Use the same seed for both modes when making a comparison.

```bash
python -m roboscope_kit.runtime.benchmark \
    --benchmark calvin --suite task_D_D --checkpoint-source /models/starvla-calvin \
    --benchmark-manifest <install-dir>/benchmark-manifest.json \
    --policy-config /path/to/roboscope-policy.json \
    --eval-mode fast --concurrency 16 --replicas 1 --output-dir /runs/calvin-task_D_D-fast
```

The initial action noise is seeded by episode ID and observation step, independent of batch position.
Controlled probes found action differences due to token padding and BF16 batch arithmetic. In the
1,000-sequence `task_D_D` runs at Kit `0.8.0b73` on one cloud RTX 4090 (49,140 MiB reported),
batching achieved **2.840×** the formal evaluation throughput of the fastest unbatched configuration
(16 Workers and two replicas), with **2.825×** end-to-end speedup. Two same-seed serial runs
agreed on all 1,000 scores. A same-seed batched repeat changed 216 scores.
[Throughput records and reproduction commands](https://github.com/Ru-hulu/RoboScope-Kit/releases/tag/calvin-pr138-2026-11-24)

Across seeds 0, 1, and 2, paired 95% confidence intervals for batched minus serial were
[−0.20, +2.13] percentage points for `task_3` and [−0.90, +2.10] for `task_5`, exceeding
the +2-point equivalence margin. The other three task success rates and mean completed subtasks
met their equivalence margins. [Three-seed results and action diagnostics](https://github.com/MINT-SJTU/RoboScope-Kit/pull/138)

## 2. User policy lifecycle

The user entrypoint implements `Policy` from `roboscope_kit` and calls `run(policy)`. The runtime
starts one process per Policy replica with the command from `--policy-config` and injects a Unix socket
through `ROBOSCOPE_INFERENCE_SOCKET`. The runtime then calls:

1. `Policy.load(SessionContext)` once per replica;
2. `Policy.reset(EpisodeContext)` before each language subtask;
3. `Policy.act(Observation)` whenever the evaluator needs a new action chunk;
4. after a successful reset, `Policy.end_episode(episode_id)` when each subtask finishes or aborts;
5. `Policy.unload()` when evaluation ends.

The Policy process owns mutable episode state and should key it by `episode_id`. The SDK does not store or
synchronize user state.

`PolicyCapabilities` must declare CALVIN, the user-selected suite, every required feature, and supported
action-chunk sizes. Evaluator concurrency (`K`) and Policy replica count (`g`) are selected by autotune,
or fixed for one evaluation with `--concurrency` and `--replicas`.

Real model loading can exceed 30 seconds under cold-cache or network-storage conditions. The benchmark CLI
therefore uses the manifest's `defaults.ready_timeout_s` (300 seconds for CALVIN) as the per-replica
`PolicyProcessAdapter.load_timeout_s`; the adapter's 30-second default applies only to callers that
construct it directly.

## 3. Canonical observation

Every `Observation` contains the following benchmark-native features:

| Feature | Type | Meaning |
| --- | --- | --- |
| `images.static` | contiguous `uint8` array | CALVIN static RGB camera image |
| `images.gripper` | contiguous `uint8` array | CALVIN gripper RGB camera image |
| `state.proprioception` | contiguous `float32` array | Native CALVIN robot observation vector |
| `language.instruction` | `str` | Official validation annotation for the current subtask |

The runtime does not resize, normalize, transpose, or otherwise adapt images for a particular model.
Those transformations belong inside the user policy. `episode_id` identifies the sequence and subtask;
`step` starts at zero for each subtask and increases once per simulator action.

## 4. Action contract

`Policy.act()` returns an `ActionChunk` containing either one 7-dimensional action or a two-dimensional
array of 7-dimensional actions. The seven values are the native CALVIN relative end-effector action:

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

The gripper command supplied to the simulator must use CALVIN's native `-1` or `+1` representation. A
model that predicts a continuous gripper value must convert it explicitly in the policy. The evaluator
consumes the returned chunk in order and requests another chunk only when the current one is empty. Cached
actions are discarded at every subtask reset.

## 5. Official rollout behavior

The persistent evaluator worker imports `calvin_agent.evaluation.evaluate_policy.evaluate_sequence` and
uses the official task oracle, validation annotations, environment reset, subtask horizon, action stepping,
and success detection. RoboScope does not reimplement those rules.

Before a sequence starts, the worker selects its initial condition and five-task list by the exact sequence
index. The official evaluator resets the environment to that initial condition. It then evaluates subtasks
in order and stops the sequence at the first failure. The worker wraps `env.step` only to count simulator
steps for throughput reporting; the action and return value are unchanged.

Unexpected simulator, protocol, or policy errors terminate the worker and surface through the generic
runtime. They are not converted into failed tasks.

## 6. Work manifest and recovery

Each `WorkItem` contains a start sequence index and an exact sequence count. It declares the same number of
expected terminal records. Every completed sequence produces exactly one record:

```text
sequence-0000
sequence-0001
...
sequence-0999
```

The record metadata contains the sequence index, ordered task list, and chained success length. Final
aggregation rejects missing indices, duplicate indices, unexpected record IDs, or coverage outside the
requested prefix. Generic queue recovery may rerun an uncommitted work item after a worker failure, but it
cannot silently skip or double-count a sequence.

## 7. Official aggregation

For chained success lengths `L_i` in `[0, 5]`, the result contains:

- `avg_seq_len`: the mean of `L_i`;
- `task_1` through `task_5`: the fraction of sequences with `L_i >= 1` through `L_i >= 5`;
- `success_at_5`: the fraction completing all five subtasks;
- `sequence_success_lengths`: all `L_i` values in sequence-list order.

The generic runtime additionally reports total simulator steps, evaluation wall time, steps per second,
episodes per second, resource samples, and worker-recovery events. These operational metrics do not replace
the official CALVIN aggregation.

## 8. Validation sequence

Before reporting a score or throughput result:

1. Verify that the sequence list contains the official ordered 1,000 entries and record its SHA-256.
2. Run one sequence with K=1 and g=1 through the real policy and simulator.
3. Run a small fixed parallel prefix and compare its per-sequence chained lengths with the established
   execution path.
4. Confirm exact record coverage, an empty pending/running queue, no worker failures, and no OOM recovery.
5. Tune K and g only with the same fixed prefix for every candidate.
6. Run all 1,000 sequences once with the selected fixed K/g configuration.

Image preprocessing and gripper conversion stay inside the Policy; the CALVIN adapter itself contains no
model-specific imports.
