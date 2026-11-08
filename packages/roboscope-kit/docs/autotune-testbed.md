# Autotune Testbed

Updated: 2026-11-08

A real autotune run takes up to 30 minutes on a GPU host, costs money, and gives noisy results, so a search
cannot be judged by rerunning it. The testbed records what real probes returned, once, and replays those
records through the unchanged `BenchmarkAutotuneController`, so any registered strategy runs without a GPU,
deterministically, in milliseconds.

It is a development tool. An evaluation never imports it, it changes no search behavior, and it adds no
dependency. The code is `roboscope_kit/runtime/benchmark/autotune_testbed.py`, the workloads are in
`tests/autotune_workloads/`, and `tests/test_autotune_testbed.py` runs every registered strategy on every
workload.

## Scoring a strategy

```bash
cd packages/roboscope-kit
python -m roboscope_kit.runtime.benchmark.autotune_testbed score tests/autotune_workloads/*.json
```

`--strategy <name>` limits the run to one strategy, `--throughput-estimator steady|elapsed` overrides each
strategy's own estimator as `autotune_throughput_estimator` does in a manifest, and `--repeat <n>` selects
among repeated measurements. Each workload prints what it does not cover, then one line per strategy:

| Field | Meaning |
| --- | --- |
| selected | The K/g the search returned, or `none` if it could not start. |
| relative to best recorded | Selected throughput over the best configuration the record does not show failing, both on one rate, using the median of repeats. |
| probes | Probes the record answered. |
| tuning time | Sum of the replayed wall times against the budget, or `unknown` if a probe has no wall time. |
| end to end | Tuning time plus `evaluation_steps` at the selection's fixed-run rate, also as a multiple of the evaluation alone at the best recorded rate. Left out when the record lacks an input. It is an estimate: a fixed run is shorter than the evaluation. |
| safe | `NO` if any measurement of the selected K/g in the record failed. |
| complete / INCOMPLETE | Whether the search ended by itself or ran off the record, and where. |
| fixed-run stand-ins | K/g answered from a fixed run because no search probed them. |
| unknown signals | Signals of the replayed probes that the record does not hold. |

Each workload also prints a `fixed default` line: `K` at the CPU worker cap with one replica per GPU, and no
tuning, scored on fixed runs. It shows what a search adds over doing nothing. It needs a fixed run at that K/g.

**Read relative throughput together with the coverage.** It compares only against what was measured. On a
record that stops at `g=1`, 1.000 means "best of the `g=1` points", not "best configuration".

## Record format

A record is one JSON file. `schema_version` is 1; change it when a field changes meaning.
`validate_record` checks the structure, the `outcome` rules, `unknown_fields`, and host paths, and `score` refuses
a record that fails it. A test checks the file name of synthetic workloads.

| Field | Content |
| --- | --- |
| `provenance.kind` | `real`, or `synthetic` for numbers nobody measured. A synthetic file's name starts with `synthetic-`. |
| `provenance.source`, `provenance.notes` | Where the numbers come from, and anything approximate or assumed. |
| `workload` | `benchmark`, `suite`, `policy`, `kit_version`, `date`, and `host` with `gpu_model`, `gpu_count`, `vram_per_gpu_bytes`, `cpu_cores`. A value the source does not state is the string `unknown`. |
| `evaluation_steps` | Optional: total steps of the full evaluation a search tunes, for the end-to-end estimate. |
| `bound_hint` | The adapter's bottleneck class. Absent means the controller's default, `gpu_infer`. |
| `replica_vram_bytes` | The replica VRAM footprint `measure` returns. Absent means unknown. |
| `manifest_defaults` | The bounds and limits inputs, with the names of the manifest's `defaults`. With `host`, they go through the real `resolve_bounds` and `resolve_limits`, never through this machine's CPU quota. |
| `observed_searches` | Optional: one entry per recorded search, in `--run` order: `autotune`, the autotune code it ran (such as `main 5f79ce2`); `strategy`; `throughput_estimator`; `path`, every K/g it probed, in order; `selected`; `stop_reason`, with each probe message cut to its code; and `elapsed_s`. |
| `unknown_fields` | Every tracked field that a finished measurement lacks: `cpu_cores`, `gpu_utilization`, `health`, `peak_memory_bytes`, `probe_items`, `queue_wait_p95_ms`, `record_success`, `steady_window` (not required for `no_steady_window`), `steps_per_s`, `throttle_time_ratio`, and `wall_time_s`. A tracked field that is left out must be listed here. |
| `unmeasured` | In words, which K/g nobody measured. |
| `measurements` | One entry per probe. Several entries at one K/g are repeats. |

A measurement holds:

| Field | Content |
| --- | --- |
| `k`, `g` | The configuration. |
| `source` | `probe` from a search, or `fixed-run` from `--concurrency K --replicas G`. |
| `outcome` | `completed`, `no_steady_window`, or a failure: `worker_oom`, `worker_failure`, `worker_loss`, `worker_shortfall`, `incomplete_probe`, `capacity_reached`, `policy_replica_failure`, `correctness_regression`, `candidate_timeout`. |
| `steady_steps_per_s` | Required for `completed`, forbidden for `no_steady_window`. An absent steady rate always means "no window", never "not reported", because that is how the controller reads it. |
| `steps_per_s`, `steady_window` | The elapsed rate and the steady window as in `result.json`. |
| `cpu_cores`, `throttle_time_ratio`, `gpu_utilization`, `peak_memory_bytes`, `queue_wait_p95_ms` | The resource signals the controller reads. |
| `health`, `record_success` | Probe health and per-episode success. |
| `wall_time_s` | The probe's wall time. |
| `search` | For a probe: the index in `observed_searches` of the search that ran it. |
| `probe_items` | Work items in the probe. |

A record must not hold secrets or paths from the recording host. `record` copies named fields only, so
commands, paths, and error messages stay out, and validation rejects anything that looks like a host path.

## Replay rules

`WorkloadReplay` supplies the controller's `measure` and `clock`; strategies are unchanged.

- **A K/g the record does not hold** is not interpolated. The probe stops with `unrecorded_configuration`, as a
  failed probe would, so the search keeps its last accepted configuration. The score lists it under gaps and
  marks the search incomplete. What follows the gap is unknown: the search might have gone further.
- **A needed rate the record does not hold** is treated the same way, with `unrecorded_steps_per_s`. A strategy
  on the elapsed estimator cannot run where the record has no elapsed rates.
- **Other signals the record does not hold** read as "no limit reached": 0 CPU cores, throttle, GPU utilization,
  and peak memory. The controller needs numbers here, so these values exist only inside the replay, never in
  the record, and the score names them under unknown signals. A search with unknown signals has not been
  tested against that resource limit.
- **A failure outcome** raises the same exception the real probe raised. A finished probe is rebuilt with all
  K Workers healthy, which is what finishing means.
- **Correctness** comes from `record_success`, which the controller compares with the baseline as usual, or from
  a recorded `correctness_regression`. Aggregate success in a replayed output is a placeholder.
- **Time** advances by each probe's `wall_time_s`, so the candidate timeout and total budget apply. A probe without
  a wall time advances nothing, so the budget cannot stop a search there, and tuning time is `unknown`.
- **Repeats** are chosen by `--repeat`, wrapping around, so a replay is deterministic. `safe` looks at every
  repeat.
- **Scores do not mix probes and fixed runs.** When the selection has a fixed run, relative throughput compares
  the elapsed rates of fixed runs, which ran one workload. Otherwise it compares probes only.
- **Search probes take precedence over fixed runs** at the same K/g. A fixed run runs more episodes than a probe (the
  full 500-episode evaluation on LIBERO, the probe's 24 work items with 96 episodes on RoboCasa), so its window and
  its share of start-up and drain can differ from a probe's. A fixed run answers a probe only
  where no search probed, and the score lists it as a stand-in.

Wall times also depend on history: a replica probe in a real search reuses the replicas already running, so its
time holds only for the same path. Likewise, a probe's size depends on the estimator: `steady` gives every
Worker three items, while `elapsed` runs a fixed-size probe, so elapsed rates recorded from a steady search are
not exactly what an elapsed search would measure.

## Regression rule

The rule answers one narrow question: is a change worse than today on the workloads we care about? It does not
say whether an algorithm is good in general; only more real records can widen that. How far a selection is from
the recorded best is shown for people to read and is not a gate by itself.

`JUDGED` in `tests/test_autotune_testbed.py` holds each judged workload with its expected relative throughput, on
the default strategy. `regression_failures` applies the rule:

- **Each workload is judged by itself.** A gain on one does not offset a loss on another.
- **A change fails** if, on any judged workload, relative throughput falls below the expected value by more than
  `REGRESSION_TOLERANCE`, the selection is unsafe, or the search is incomplete. A failure stops the change for a
  person to look at; it does not prove the algorithm worse.
- **The tolerance is 2% on every workload,** set by the maintainer. Replay adds no noise, since a record and the code
  give one score, so an unchanged search never fails. The score moves only when a change selects another K/g, and a
  selection more than 2% slower than today's is stopped.
- **The recorded rates are less precise than 2% on RoboCasa.** Repeated fixed runs of one configuration spread
  by (sample standard deviation over mean of their elapsed rates):

  | Workload | K/g | Elapsed rates (steps/s) | Spread |
  |---|---|---|---:|
  | RoboCasa | `K=8/g=1` | 168.7 (2026-11-05), 149.0, 147.8 (2026-11-06) | 7.6% |
  | RoboCasa | `K=12/g=2` | 233.9 (2026-11-05), 226.3, 229.9 (2026-11-06) | 1.7% |
  | RoboCasa | `K=16/g=2` | 223.6 (2026-11-05), 202.2, 215.4 (2026-11-06) | 5.0% |
  | LIBERO | `K=16/g=4` | 209.0, 209.0, 208.4 | 0.17% |
  | LIBERO | `K=12/g=4` | 209.1, 209.7, 208.9 | 0.21% |
  | RoboTwin | `K=8/g=1` | 108.7, 108.9, 109.7 | 0.49% |
  | RoboTwin | `K=10/g=1` | 115.2, 115.8, 116.9 | 0.74% |

  On RoboCasa, `K=8/g=1` is 13% faster on 2026-11-05 than on the next day, when its two runs agree within 1%; `K=16/g=2`
  differs by 6.5% between its two runs on 2026-11-06. A change there that selects a configuration a few percent
  slower fails, though the records cannot tell the two apart; a person decides whether to lower the expected value.
  LIBERO's six repeats ran within about 2.5 hours of one night and hold no drift between days. RoboTwin's ran on an
  evening and the next morning.
- **Lowering an expected value** is allowed only in a PR that says why, and it needs maintainer review. Raise the
  value when a change improves it, so later changes cannot give the gain back.
- **An incomplete search cannot be judged.** Measure the missing point on a GPU and add it to the record.
- **Only real records are judged.** Synthetic workloads cover edge cases in unit tests. A real record that is not
  judged must say why in `NOT_JUDGED`.

`EXPECTED` pins what every strategy does on every workload, so any change of a search shows in the PR's diff.
Update it with the change; it is for review, and the rule above decides.

## Reproducing real searches

A record is only useful if the replay behaves as the real search did. So each judged workload holds a real search
of the current autotune code, named in `REPRODUCED` in `tests/test_autotune_testbed.py`, and the test replays that
code on that search's own probes. It must probe the same K/g in the same order, stop for the same reason, select
the same K/g, and take the same time within 2%. `score` prints, for every recorded search, whether the code being
scored reproduces it.

A change to the search will no longer reproduce searches of the old code; that is expected, and the replay says
where they part. Name in `REPRODUCED` the searches the change does reproduce. Real searches of the new code are
not required to merge it, but a workload whose `REPRODUCED` entry is gone has lost its check, so record one when
the GPU is next free.

Before merging an algorithm change, also run it once for real on a workload or host outside the testbed, and add
that run as a new record, so the algorithm is not fitted to the records alone.

## Adding a workload

1. Run a search on the host, keeping its output directory. Run it again for repeats.

   ```bash
   python -m roboscope_kit.runtime.benchmark --benchmark <benchmark> --suite <suite> ... --output-dir <run>
   ```

2. For configurations no search visits, run them fixed, one output directory each, as many as the budget allows.
   Cover what a search may select, the best configuration you can find, and `K` at the worker cap with `g` at the
   GPU count, for the fixed default. Run the best and the selected configuration three times; the score uses the
   median of repeats.

   ```bash
   python -m roboscope_kit.runtime.benchmark --benchmark <benchmark> --suite <suite> ... \
     --concurrency 16 --replicas 2 --output-dir <sweep-k16-g2>
   ```

3. Build the record. `--run` and `--sweep` repeat; give one `--autotune` per `--run`, in the same order, naming
   the autotune code it ran, such as `main 5f79ce2`. Existing outputs carry everything else; the run outputs do
   not name the Policy, the GPU model, or the Kit version, so pass those. `--evaluation-steps` is `total_steps`
   of a full evaluation's `result.json`, or the steps per episode times its episodes.

   ```bash
   python -m roboscope_kit.runtime.benchmark.autotune_testbed record \
     --run <run> --autotune "main 5f79ce2" --sweep <sweep-k16-g2> \
     --source "RoboCasa sweep, issue #80" --policy "StarVLA Qwen3-VL-OFT-Robocasa" \
     --gpu-model "RTX 4090" --kit-version 0.8.0b29 --date 2026-11-05 --evaluation-steps 864000 \
     --output tests/autotune_workloads/<benchmark>-<suite>-<gpu>.json
   ```

   Only a completed search can be recorded. A probe ended by the total budget or a harness error says nothing
   about its configuration, so it stays unrecorded. A failed fixed run leaves no `result.json`; add its
   measurement by hand with its `outcome`, and no rates.

4. Edit `unmeasured` to say what is missing, and add anything approximate to `provenance.notes`.

5. Check the file for anything that should not be public. Then score it, and add the expected search of every
   strategy to `EXPECTED` in `tests/test_autotune_testbed.py`; a workload or a strategy without an entry there
   fails the test. Add a real workload to `JUDGED` with its expected value, or say why not in `NOT_JUDGED`. A judged
   workload also goes in `REPRODUCED`, with the `autotune` name of its search of the current code.

A new strategy needs only step 5. To record numbers from somewhere other than run outputs, such as a table in an
issue, write the JSON by hand with only the fields the source gives, and list the rest in `unknown_fields`, as
`robocasa-tabletop-rtx4090-issue80.json` does. Never fill a field with a plausible value. A synthetic workload
is for edge cases only, and must say so in `provenance.kind` and in its file name.

## Workloads

| File | Kind | Covers | Missing |
| --- | --- | --- | --- |
| `robocasa-tabletop-rtx4090-issue80.json` | real | `K=1, 2, 4, 8` at `g=1`: steady rate and GPU utilization. `K=16` at `g=1`: no steady window. | Every `g > 1` point; elapsed rates, per-probe wall times, CPU, throttle, peak memory, queue wait, health, per-episode success, steady windows, and probe sizes for every point. |
| `robocasa-tabletop-rtx4090-full-auto.json` | real | Built by `record` from two searches and a fixed-K/g sweep on one host: probes at `K=1, 2, 4, 8, 16` with `g=1` and `K=16/g=2`; fixed runs at 8/1, 12/2, and 16/2 three times each, 9/1, 16/1, 16/3, 20/2, and three that failed (16/4, 20/3, 24/2). | Queue wait; fixed-run health and wall times; fixed runs with CUDA MPS, listed in its notes; `K=32/g=2`, stopped by hand. |
| `libero-10-rtx4090-full-auto.json` | real | Built by `record` from two searches and a fixed-K/g sweep on one host: probes at `K=1, 2, 4, 8` with `g=1`, 8/2, 12/3, and 16/4, in both searches; full 500-episode fixed runs at 16/4 and 12/4 three times each, 8/2, 8/4, 9/1, 12/3, 16/3, 20/4, 24/1, and 24/4. | Queue wait; fixed-run health and wall times; repeats on another day; the K/g `two-phase` asks for next (9/2, 8/5). |
| `robotwin-demo-clean-subset-rtx4090-full-auto.json` | real | Built by `record` from two searches and a fixed-K/g sweep on one host, all on a 500-episode subset of `demo_clean` (episodes 0-9 of each of the 50 tasks) run by a wrapper outside the repository, not Kit's official RoboTwin entry point: probes at `K=1, 2, 4, 8` with `g=1` and `K=16/g=1` (`worker_oom`), in both searches; fixed runs at 8/1 and 10/1 three times each, 6/1, 7/1, 7/2, 8/2, 9/1, 11/1, and two that failed (12/1, 16/1). 8/2 and 11/1 lost one Worker to GPU memory and finished with one fewer, which a fixed run cannot record. | Queue wait; fixed-run health and wall times; the full 5000-episode evaluation; the K/g `two-phase` asks for next (9/2). |
| `synthetic-replica-limits.json` | synthetic | Replica probes, probes without a window, a failing configuration, wall times, repeats, a fixed-run point. | Everything: no number in it was measured. |

Scores on `robocasa-tabletop-rtx4090-full-auto.json`:

| Strategy | Estimator | Selected | Relative to best recorded | Probes | Tuning time | End to end | Safe | Search |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| `diagonal` | steady (default) | `K=8/g=1` | 0.648 of `K=12/g=2` | 5 | 852 s | 6649 s, 1.77x | yes | Complete: stops at `K=16` without a steady window, a known problem (#80) |
| `two-phase` | elapsed (default) | `K=9/g=1` | 0.620 of `K=12/g=2` | 5 | unknown | unknown | yes | Complete, but the `K=9/g=1` probe is answered by a fixed run, which has no probe time |
| `two-phase` | steady | `K=9/g=1` | 0.620 of `K=12/g=2` | 5 | unknown | unknown | yes | As above |

Relative throughput uses the median of each configuration's fixed runs. The fixed default on this host is
`K=9/g=1`, with no tuning: 0.620 of the best, about 6063 s for the full evaluation. `two-phase` selects the same
configuration. `diagonal` selects a faster one, `K=8/g=1` at 0.648, but its tuning time makes it slower end to end.

The record holds two real searches. `main 5f79ce2` stopped at `K=8/g=1` after `K=16/g=1` had no steady window, in
818 s, and the code here reproduces it probe by probe. `#81 81e3824`, a candidate fix for #80, went on to `K=16/g=2`;
the code here does not reproduce it, and the replay shows it stopping after `K=16/g=1`. The scores above use the
first repeat of each K/g, which comes from the `#81` search. Relative throughput compares fixed runs, which ran the same 96 episodes. Default `two-phase` runs a fixed-size probe that this record
does not cover, so its rows only show how it would decide on these rates.

Scores on `libero-10-rtx4090-full-auto.json`:

| Strategy | Estimator | Selected | Relative to best recorded | Probes | Tuning time | End to end | Safe | Search |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| `diagonal` | steady (default) | `K=16/g=4` | 0.999 of `K=12/g=4` | 7 | 501 s | 1156 s, 1.77x | yes | Complete: stops on predicted GPU memory at `K=20/g=5` |
| `two-phase` | elapsed (default) | `K=9/g=1` | 0.549 of `K=12/g=4` | 5 | unknown | unknown | yes | Incomplete: asks for `K=9/g=2`, unrecorded |
| `two-phase` | steady | `K=8/g=4` | 0.974 of `K=12/g=4` | 7 | unknown | unknown | yes | Incomplete: asks for `K=8/g=5`, unrecorded |

`K=16/g=4` and `K=12/g=4` are tied, within the spread of their repeats. The two searches in this record, `main d26a908`
and `#81 81e3824`, probed the same K/g and selected the same one, and the code here reproduces both, so LIBERO guards
against a regression but does not tell them apart. The fixed default is `K=9/g=1`: 0.549 of the best, about 1191 s.

Scores on `robotwin-demo-clean-subset-rtx4090-full-auto.json`:

| Strategy | Estimator | Selected | Relative to best recorded | Probes | Tuning time | End to end | Safe | Search |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| `diagonal` | steady (default) | `K=8/g=1` | 0.940 of `K=10/g=1` | 5 | 608 s | 2729 s, 1.37x | yes | Complete: stops when `K=16/g=1` hits `worker_oom` |
| `two-phase` | elapsed (default) | `K=9/g=1` | 0.976 of `K=10/g=1` | 5 | unknown | unknown | yes | Incomplete: asks for `K=9/g=2`, unrecorded |
| `two-phase` | steady | `K=8/g=1` | 0.940 of `K=10/g=1` | 6 | unknown | unknown | yes | Complete, with fixed-run stand-ins at 9/1 and 8/2 |

`diagonal` doubles K, so after `K=8` it probes `K=16`, which runs out of GPU memory, and it stops without trying
`K=10`. The two searches in this record, `main 44fa1ad` and `#81 5538408`, probed the same K/g and selected the same
one, and the code here reproduces both, so RoboTwin records this gap and guards against a regression but does not tell
them apart. The fixed default is `K=9/g=1`, 0.976 of the best from one run, faster than the tuned selection.
