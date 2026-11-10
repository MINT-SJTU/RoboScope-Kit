# LIBERO-Pro fixed baseline comparison, 2026-11-10

The selected draft result is **3.1006x formal episode throughput** against the historical fixed
configuration reproduced on the available dedicated RTX 4090. Both runs completed the 16 standard
single-perturbation suites, 10 tasks × 50 initial states each: **8,000 episodes per side**.
The optimized arm is the latest complete run; the fixed arm is the previously measured baseline.

The optimized measurement used Kit 0.8.0b38 at `dfebac72a124c2d439707ddd728fc7b92512e6b0`
plus the then-uncommitted camera patch identified below. The current draft subsequently merged
upstream autotune fixes absent from that run. It is not a full measurement of the current draft head.

## Configuration and timing

| Setting / metric | Fixed historical baseline | Latest optimized run |
| --- | --- | --- |
| Source | `ed8c9cfd32baba57c4fbb276c151a70e71edc7ea` | `dfebac72a124c2d439707ddd728fc7b92512e6b0` plus camera patch |
| Kit version | 0.8.0b29 | 0.8.0b38 |
| K/g | 13 suites 2/1; `libero_10_swap`, `libero_10_lan`, `libero_10_task` 8/2 | Diagonal autotune independently per suite; selections in table below |
| Policy `OMP_NUM_THREADS` | 8 | 1 |
| Encoding | JSON | MessagePack |
| Chunk camera rendering | Off | On |
| CUDA MPS | Off | On, including tuning |
| Native batch cap | 1 | 2 |
| Persistent connections / video / diagnostic tracing | Off | Off |
| Formal episodes | 8,000 | 8,000 |
| Formal evaluation, summed | 20,853.711 s | 6,725.749 s |
| Formal episodes/s | 0.3836 | 1.1895 |
| Actual policy action steps | 1,999,574 | 1,997,826 |
| Actual policy steps/s | 95.89 | 297.04 |
| Summed formal CLI wall | 21,174.533 s | 12,669.690 s, including tuning |
| Controller Python | 3.11.15 | 3.12.3 |

`formal speedup = 20853.710727687925 / 6725.748836595565 = 3.1006x`.
Formal throughput is 8,000 divided by the sum of `evaluation_s` across the suites, from the first
episode dispatched to the last result recorded. The actual action-step throughput ratio is 3.0979x;
settling steps are excluded from the policy-step counts. Do not average per-suite speedup factors.
The CLI times additionally include process initialization, loading, tuning and cleanup. The latest
optimized run spent 5,842.389 s in summed autotune phases and 346.046 s in separate summed warmups;
its overall arm elapsed time was 13,026.457 s including warmup, tuning and verification. Cold
build/download are excluded. Per-suite phase timings are retained in the comparison JSON.

Both runs used the same AutoDL host: one reported NVIDIA GeForce RTX 4090 with 49,140 MiB visible
memory, driver 580.105.08, Intel Xeon Gold 6459C, a 12-core CPU quota and a 96 GiB container limit.
The GPU was checked idle before execution. The frozen prebuilt simulator and cached model were reused;
model Python was 3.11.15 and simulator Python was 3.11.18. The fixed run finished at 17:57 China
Standard Time on 2026-11-09; the optimized run ran from 22:02 on 2026-11-09 to 01:39 on 2026-11-10.
Run order was not randomized.

Both preserve the pinned StarVLA checkpoint `a60fff4`, bf16/SDPA, seed 7, ten settle steps, perturbed
BDDL instructions, primary/wrist cameras, eight-action chunks, native horizons and per-step
termination/success checks. Checkpoint SHA-256:
`67d26b27713d44c14af4ddb6eab05cd57e388f10caad153b657c8fdc64ded92f`.
Model config and dataset-statistics SHA-256:
`9e07eb793ff7173ae5d1bf0e5ddebf0421b1c0be46186bb180f0c22d505210db` and
`64bb15c9525653c8efc476496d2a0e43d678401bc4ccc65b61fa6e314c836687`.
All 320 frozen task-file digests were checked before and after the runs.

The optimized run's camera source SHA-256 was
`77419b8f4ff5aa2d8237f0d0b34f25321f8dea233ca3866eebc2cb6898e0dd01`; its patch SHA-256 was
`d6ffb35089f3d6ecef89f8723e415324f4e943d6d02d2350e4241f13e979eed3`. The current camera source
matches the recorded camera hash, but other runtime files and metadata changed after the measurement.

## Per-suite results

Success counts are per-suite official results out of 500. `Changed` counts matching episode IDs
with different success flags / step counts. K means Workers; g means Policy replicas.

| Suite | Fixed successes | Optimized successes | Formal seconds fixed / optimized | K/g fixed → optimized | Changed success / steps |
| --- | ---: | ---: | ---: | --- | ---: |
| `libero_goal_object` | 339 | 343 | 1,067.20 / 391.72 | 2/1 → 8/1 | 44 / 239 |
| `libero_goal_swap` | 4 | 8 | 1,817.45 / 416.85 | 2/1 → 24/3 | 12 / 12 |
| `libero_goal_lan` | 444 | 441 | 869.30 / 223.39 | 2/1 → 16/2 | 33 / 371 |
| `libero_goal_task` | 45 | 49 | 1,677.93 / 389.10 | 2/1 → 24/3 | 10 / 45 |
| `libero_spatial_object` | 477 | 477 | 837.06 / 277.28 | 2/1 → 8/1 | 22 / 348 |
| `libero_spatial_swap` | 0 | 0 | 1,720.62 / 376.98 | 2/1 → 16/2 | 0 / 0 |
| `libero_spatial_lan` | 470 | 469 | 926.11 / 310.03 | 2/1 → 8/1 | 19 / 327 |
| `libero_spatial_task` | 278 | 274 | 1,257.81 / 282.69 | 2/1 → 16/2 | 24 / 224 |
| `libero_10_object` | 239 | 240 | 2,216.16 / 545.97 | 2/1 → 24/3 | 53 / 249 |
| `libero_10_swap` | 0 | 0 | 1,382.04 / 674.49 | 8/2 → 16/4 | 0 / 0 |
| `libero_10_lan` | 458 | 460 | 747.65 / 429.97 | 8/2 → 16/2 | 38 / 422 |
| `libero_10_task` | 51 | 47 | 1,352.51 / 853.28 | 8/2 → 8/2 | 24 / 61 |
| `libero_object_object` | 453 | 446 | 936.34 / 366.46 | 2/1 → 8/1 | 21 / 306 |
| `libero_object_swap` | 0 | 0 | 1,602.97 / 430.61 | 2/1 → 16/2 | 0 / 0 |
| `libero_object_lan` | 484 | 484 | 912.87 / 347.68 | 2/1 → 8/1 | 12 / 330 |
| `libero_object_task` | 48 | 48 | 1,529.67 / 409.25 | 2/1 → 16/2 | 2 / 26 |

All expected terminal episode IDs are present exactly once in both arms. Formal recovery records
contain no Worker failures, OOMs or planned recycles. Across matched IDs, **314 success flags differ**:
159 fixed-only successes and 155 optimized-only successes. **2,960 step counts differ**. These are
comparison counts, not an official pooled success score. Matching IDs do not establish identical
initial scenes: task-environment reuse can make reset history depend on Worker assignment.

## Interpretation and remaining validation

This compares the **historical fixed configuration** with the latest optimized run; it does not show
main's fastest measured configuration. The gain includes changed concurrency, autotuning, Policy
thread settings, MPS, transport, camera rendering and batching; it does not isolate one optimization.
The original integration used 13 suites on an RTX 3080 and three on a shared RTX 4090. This reproduction
uses its per-suite K/g on one dedicated 4090. OMP=8 was documented for those three original 4090 suites
and is an assumption for the other 13 suites. The controller Python versions also differ.

Each side ran once, so this result does not establish repeatability or absence of systematic score
regression. The public [issue #21](https://github.com/MINT-SJTU/RoboScope-Kit/issues/21) also specifies
a best-existing-configuration baseline and three repeats per side on `libero_goal_task`,
`libero_spatial_lan`, `libero_object_swap` and `libero_10_object`. Those acceptance checks and
measurements of the exact head to be merged remain open.

## Evidence

- [Fixed summary](measurements/issue19-fixed-2026-11-09.json) and
  [fixed plan](measurements/issue19-fixed-2026-11-09-plan.json).
- [Optimized plan](measurements/optimized-2026-11-10-plan.json) and
  [comparison JSON](measurements/fixed-vs-optimized-2026-11-10.json), including per-suite probes and timings.
- [Suite results](measurements/fixed-vs-optimized-2026-11-10-suites.csv),
  [task results](measurements/fixed-vs-optimized-2026-11-10-tasks.csv), and
  [8,000 paired episode records](measurements/fixed-vs-optimized-2026-11-10-paired.csv).
- [Camera replay](measurements/camera-replay-2026-11-10.json) and the
  [evidence inventory](measurements/README.md). Raw archives remain local; durable reviewer-accessible
  links are pending.
