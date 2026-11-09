# LIBERO-Pro fixed baseline and optimization comparison, 2026-11-09

The #19-style fixed-concurrency baseline completed all 16 released suites on the available AutoDL
RTX 4090 at 17:57 China Standard Time. It used Kit 0.8.0b29 at `ed8c9cf`, the pinned checkpoint,
seed 7 and 10 settle steps. All 8,000 formal episode IDs were present exactly once, and no formal
Worker failures, OOMs or planned recycles were recorded. This is one run; matching episode IDs do
not prove matching initial scenes or trajectories.

This run reproduces #19's version and per-suite K/g on the available host, not its original hardware:
13 of #19's suites used an RTX 3080, and three used a shared RTX 4090. The new run used one idle
4090 with a 12-core CPU quota. The policy used `OMP_NUM_THREADS=8` in all suites, as recorded for
the original three 4090 suites; the setting for the original 13 3080 suites was not recorded. The
prebuilt simulator and cached model were reused, so cold build and download times are excluded.

## Full-matrix throughput

The following are separate single runs on the same AutoDL host with the same checkpoint and frozen
16-suite task catalog. Formal throughput is 8,000 divided by the sum of each suite's `evaluation_s`.
Formal policy-step throughput uses the actual executed action steps, not settle steps.

| Run | Policy K/g and settings | Formal time | Episodes/s | Policy steps/s |
| --- | --- | ---: | ---: | ---: |
| #19-style fixed baseline, b29 | 13 suites 2/1, three 8/2; JSON, MPS off, batch 1, policy OMP 8 | 20,853.711 s | 0.3836 | 95.89 |
| Autotuned original, b30 | Per-suite diagonal, selected 12/3 or 16/4; JSON, MPS off, batch 1, policy OMP 1 | 9,869.044 s | 0.8106 | 202.41 |
| Autotuned optimized, b37 | Per-suite diagonal, selected 8/1, 16/2 or 24/3; MessagePack, camera gate, MPS on, batch cap 2, policy OMP 1 | 7,156.150 s | 1.1179 | 279.78 |

The fixed baseline versus the optimized run gives a descriptive **2.9141x** episode-throughput
ratio and **2.9178x** action-step-throughput ratio. It changes concurrency and policy threads as
well as transport, MPS, camera gating and batching, and neither order nor repeated runs are controlled.
This is not an isolated optimization effect and does not on its own establish issue #21 acceptance.

The original autotuned run is **2.1130x** faster than this fixed baseline. Kit source, LIBERO-Pro
runtime files and the StarVLA example are identical between #19's `ed8c9cf` and b30's `44fa1ad`;
package version metadata and unrelated files differ. K/g, policy thread settings, scheduling and run
order differ, so this ratio cannot be attributed to K/g alone. The [autotuned comparison](FULL_AUTO_2026-11-09.md)
compares two independent diagonal searches on the same host: optimized versus original is **1.3791x**
formal throughput. Even that pair changes MPS and batch settings and still needs correctness and
repeatability review. Do not use the 2.9141x comparison to claim a proven 2x optimization.

The fixed run's controller used Python 3.11.15; both autotuned runs used Python 3.12.3 for the
controller. All three used Python 3.11.15 for the model and the same prebuilt Python 3.11.18
simulator. These controller environments are another difference between the separate runs.

The summed formal CLI process walls were 21,174.533 s for fixed, 17,169.621 s for the autotuned
original, and 13,194.188 s for the optimized run. The latter two include tuning; the fixed run does
not tune. Quick Tests, the fixed-run two-replica smoke and cold preparation are outside those sums.
These numbers do not establish a matched cold end-to-end speedup.

## Per-suite outcome and time

Each success count is out of 500 and is an individual suite result. `#19` is the historical score,
shown for context only because its hardware and execution environment differed. `Changed` counts
matching episode IDs whose success / step counts differ between the fixed and optimized runs. The
16 per-task tables and actual step counts are retained in the JSON evidence.

| Suite | #19 success | Fixed success | Optimized success | Formal seconds fixed / optimized | Ratio | K/g fixed → optimized | Changed success / steps |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `libero_goal_object` | 351 | 339 | 343 | 1,067.20 / 408.67 | 2.611x | 2/1 → 8/1 | 42 / 239 |
| `libero_goal_swap` | 6 | 4 | 6 | 1,817.45 / 415.43 | 4.375x | 2/1 → 24/3 | 8 / 9 |
| `libero_goal_lan` | 443 | 444 | 445 | 869.30 / 324.62 | 2.678x | 2/1 → 8/1 | 23 / 293 |
| `libero_goal_task` | 47 | 45 | 52 | 1,677.93 / 389.28 | 4.310x | 2/1 → 24/3 | 13 / 49 |
| `libero_spatial_object` | 476 | 477 | 484 | 837.06 / 274.50 | 3.049x | 2/1 → 8/1 | 15 / 325 |
| `libero_spatial_swap` | 0 | 0 | 0 | 1,720.62 / 377.21 | 4.561x | 2/1 → 16/2 | 0 / 0 |
| `libero_spatial_lan` | 467 | 470 | 464 | 926.11 / 298.98 | 3.098x | 2/1 → 8/1 | 16 / 332 |
| `libero_spatial_task` | 275 | 278 | 275 | 1,257.81 / 280.58 | 4.483x | 2/1 → 16/2 | 17 / 237 |
| `libero_10_object` | 237 | 239 | 232 | 2,216.16 / 556.69 | 3.981x | 2/1 → 24/3 | 53 / 245 |
| `libero_10_swap` | 0 | 0 | 0 | 1,382.04 / 690.13 | 2.003x | 8/2 → 24/3 | 0 / 0 |
| `libero_10_lan` | 459 | 458 | 447 | 747.65 / 396.17 | 1.887x | 8/2 → 24/3 | 47 / 434 |
| `libero_10_task` | 53 | 51 | 44 | 1,352.51 / 1,171.04 | 1.155x | 8/2 → 8/1 | 21 / 55 |
| `libero_object_object` | 451 | 453 | 450 | 936.34 / 365.97 | 2.558x | 2/1 → 8/1 | 15 / 312 |
| `libero_object_swap` | 0 | 0 | 0 | 1,602.97 / 434.00 | 3.693x | 2/1 → 16/2 | 0 / 0 |
| `libero_object_lan` | 482 | 484 | 481 | 912.87 / 355.45 | 2.568x | 2/1 → 8/1 | 17 / 344 |
| `libero_object_task` | 48 | 48 | 49 | 1,529.67 / 417.43 | 3.664x | 2/1 → 16/2 | 1 / 25 |

Across matched IDs, fixed versus optimized has **288 differing success flags**: 153 fixed-only
successes and 135 optimized-only successes. Step counts differ for **2,899** IDs. For comparison,
fixed versus the autotuned original has 161 differing success flags and 1,126 differing step counts.
These discrepancy counts are not an official pooled success score. They do not establish which
setting caused a difference: worker assignment can change a task environment's reset history, and
the observations/actions were not recorded for this pair.

## Fastest measured single-suite configuration

The prior 500-episode `libero_goal_object` batch experiment at fixed K=12/g=3, MPS on, camera gate
on, MessagePack and batch cap 2 took **269.478 s** (342/500 successes, 82,789 steps). The new fixed
b29 baseline took **1,067.204 s** (339/500, 82,860 steps), a descriptive **3.9603x** ratio. The
same episode IDs have 55 differing success flags and 271 differing step counts. This is not a
matched-concurrency ablation and applies to one suite only; batch metrics logging was enabled in the
prior experiment. A 16-suite matrix was not run with that fixed K/g configuration.

## Evidence and remaining work

- [Fixed-run summary](measurements/issue19-fixed-2026-11-09.json), [plan](measurements/issue19-fixed-2026-11-09-plan.json),
  [per-suite comparison](measurements/issue19-fixed-vs-auto-2026-11-09.json) and
  [paired episode CSV](measurements/issue19-fixed-vs-auto-2026-11-09-paired.csv) preserve the scores,
  actual steps, timings, configuration and identity comparisons. The CSV compares the fixed baseline,
  autotuned original and autotuned optimized records by suite/task/episode index.
- Fixed-run original logs and WorkResults remain at
  `/root/autodl-tmp/issue21-20261108/comparison/issue19-repro-b29-20261109-0QXYVUBi/` on AutoDL.
  The corresponding archive, excluding the copied environments and source tree, has SHA-256
  `c691d1b1ca9b7c3d61a435425d6c794baa45ce8649d0970aef7e39a09cf50d64`.
- A saved #19 model snapshot provenance records SDPA and the same model config and statistics hashes
  used here. It does not establish every original run's full model environment or thread settings.
- Repeat controlled same-host comparisons in alternating order; compare fixed K/g and independently
  tuned runs with matched CPU settings, and investigate starting observations, reset history and
  per-task disagreements before claiming correctness or a 2x acceptance result.
