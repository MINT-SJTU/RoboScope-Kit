# LIBERO-Pro fixed baseline comparison, 2026-11-09

The selected draft result is **2.9141x formal episode throughput** against the historical fixed
configuration reproduced on the available dedicated RTX 4090. Both runs completed the 16 standard
single-perturbation suites, 10 tasks × 50 initial states each: **8,000 episodes per side**.
This report compares the completed fixed run with the completed optimized run; no new run was
performed while preparing the draft.

## Configuration and timing

| Setting / metric | Fixed historical baseline | Optimized |
| --- | --- | --- |
| Git commit | `ed8c9cfd32baba57c4fbb276c151a70e71edc7ea` | `e63260e117fb231706f0190fe6dcfe550afa2d0f` |
| Kit version | 0.8.0b29 | 0.8.0b37 |
| K/g | 13 suites 2/1; `libero_10_swap`, `libero_10_lan`, `libero_10_task` 8/2 | Per-suite diagonal autotune; selected 8/1, 16/2 or 24/3 |
| Policy `OMP_NUM_THREADS` | 8 | 1 |
| Encoding | JSON | MessagePack |
| Chunk camera rendering | Off | On |
| CUDA MPS | Off | On, including tuning |
| Native batch cap | 1 | 2 |
| Persistent connections / video / diagnostic tracing | Off | Off |
| Formal episodes | 8,000 | 8,000 |
| Formal evaluation, summed | 20,853.711 s | 7,156.150 s |
| Formal episodes/s | 0.3836 | 1.1179 |
| Actual policy action steps | 1,999,574 | 2,002,148 |
| Actual policy steps/s | 95.89 | 279.78 |
| Summed formal CLI wall | 21,174.533 s | 13,194.188 s, including tuning |
| Controller Python | 3.11.15 | 3.12.3 |

`formal speedup = 20853.710727687925 / 7156.150195840746 = 2.9141x`.
Formal throughput is 8,000 divided by the sum of `evaluation_s` across the suites, from the first
episode dispatched to the last result recorded. The actual action-step throughput ratio is 2.9178x;
settling steps are excluded from the policy-step counts. Do not average per-suite speedup factors.
The CLI times additionally include process initialization, loading, tuning and cleanup; separate
Quick Tests and cold build/download are excluded. Per-suite phase timings are retained in the JSON.

Both runs used the same AutoDL host: one reported NVIDIA GeForce RTX 4090 with 49,140 MiB visible
memory, driver 580.105.08, Intel Xeon Gold 6459C, a 12-core CPU quota and a 96 GiB container limit.
The GPU was checked idle before execution. The frozen prebuilt simulator and cached model were reused;
model Python was 3.11.15 and simulator Python was 3.11.18. The optimized run finished at 10:03 and
the fixed run at 17:57 China Standard Time on 2026-11-09. Run order was not randomized.

Both preserve the pinned StarVLA checkpoint `a60fff4`, bf16/SDPA, seed 7, ten settle steps, perturbed
BDDL instructions, primary/wrist cameras, eight-action chunks, native horizons and per-step
termination/success checks. Checkpoint SHA-256:
`67d26b27713d44c14af4ddb6eab05cd57e388f10caad153b657c8fdc64ded92f`.
Model config and dataset-statistics SHA-256:
`9e07eb793ff7173ae5d1bf0e5ddebf0421b1c0be46186bb180f0c22d505210db` and
`64bb15c9525653c8efc476496d2a0e43d678401bc4ccc65b61fa6e314c836687`.
All 320 frozen task-file digests were checked before and after the runs.

## Per-suite results

Success counts are per-suite official results out of 500. `Changed` counts matching episode IDs
with different success flags / step counts. The JSON also retains each arm's 160 per-task scores.

| Suite | Fixed successes | Optimized successes | Formal seconds fixed / optimized | K/g fixed → optimized | Changed success / steps |
| --- | ---: | ---: | ---: | --- | ---: |
| `libero_goal_object` | 339 | 343 | 1,067.20 / 408.67 | 2/1 → 8/1 | 42 / 239 |
| `libero_goal_swap` | 4 | 6 | 1,817.45 / 415.43 | 2/1 → 24/3 | 8 / 9 |
| `libero_goal_lan` | 444 | 445 | 869.30 / 324.62 | 2/1 → 8/1 | 23 / 293 |
| `libero_goal_task` | 45 | 52 | 1,677.93 / 389.28 | 2/1 → 24/3 | 13 / 49 |
| `libero_spatial_object` | 477 | 484 | 837.06 / 274.50 | 2/1 → 8/1 | 15 / 325 |
| `libero_spatial_swap` | 0 | 0 | 1,720.62 / 377.21 | 2/1 → 16/2 | 0 / 0 |
| `libero_spatial_lan` | 470 | 464 | 926.11 / 298.98 | 2/1 → 8/1 | 16 / 332 |
| `libero_spatial_task` | 278 | 275 | 1,257.81 / 280.58 | 2/1 → 16/2 | 17 / 237 |
| `libero_10_object` | 239 | 232 | 2,216.16 / 556.69 | 2/1 → 24/3 | 53 / 245 |
| `libero_10_swap` | 0 | 0 | 1,382.04 / 690.13 | 8/2 → 24/3 | 0 / 0 |
| `libero_10_lan` | 458 | 447 | 747.65 / 396.17 | 8/2 → 24/3 | 47 / 434 |
| `libero_10_task` | 51 | 44 | 1,352.51 / 1,171.04 | 8/2 → 8/1 | 21 / 55 |
| `libero_object_object` | 453 | 450 | 936.34 / 365.97 | 2/1 → 8/1 | 15 / 312 |
| `libero_object_swap` | 0 | 0 | 1,602.97 / 434.00 | 2/1 → 16/2 | 0 / 0 |
| `libero_object_lan` | 484 | 481 | 912.87 / 355.45 | 2/1 → 8/1 | 17 / 344 |
| `libero_object_task` | 48 | 49 | 1,529.67 / 417.43 | 2/1 → 16/2 | 1 / 25 |

All expected terminal episode IDs are present exactly once in both arms. Formal recovery records
contain no Worker failures, OOMs or planned recycles. Across matched IDs, **288 success flags differ**:
153 fixed-only successes and 135 optimized-only successes. **2,899 step counts differ**. These are
comparison counts, not an official pooled success score. Matching IDs do not establish identical
initial scenes: task-environment reuse can make reset history depend on Worker assignment.

## Interpretation and remaining validation

This is a comparison with the **historical fixed configuration**, not with main's fastest measured
configuration. The gain includes changed concurrency, autotuning, Policy thread settings, MPS,
transport, camera rendering and batching; it does not isolate the effect of a single optimization.
The original integration used 13 suites on an RTX 3080 and three on a shared RTX 4090. This reproduction
uses its per-suite K/g on one dedicated 4090. OMP=8 was documented for those three original 4090 suites
and is an assumption for the other 13 suites. The controller Python versions also differ.

This is the result selected for draft review. Each side ran once, so it does not establish repeatability
or absence of systematic score regression. The public [issue #21](https://github.com/MINT-SJTU/RoboScope-Kit/issues/21)
also specifies a best-existing-configuration baseline and three repeats per side on
`libero_goal_task`, `libero_spatial_lan`, `libero_object_swap` and `libero_10_object`.
Those acceptance checks and measurements of the exact head to be merged remain open.

## Evidence

- [Fixed summary](measurements/issue19-fixed-2026-11-09.json) and
  [fixed plan](measurements/issue19-fixed-2026-11-09-plan.json).
- [Optimized summary](measurements/optimized-2026-11-09.json) and
  [optimized plan](measurements/optimized-2026-11-09-plan.json).
- [Comparison JSON](measurements/fixed-vs-optimized-2026-11-09.json) and
  [8,000 paired episode records](measurements/fixed-vs-optimized-2026-11-09-paired.csv).
- [Evidence inventory](measurements/README.md) describes extraction provenance, raw archives and
  reproduction commands. Raw archives remain local/remote; durable reviewer-accessible links are pending.
