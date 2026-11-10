# Selected draft evidence

This directory contains the historical fixed baseline versus the latest complete optimized run:
16 standard suites and 8,000 episodes per side, with 3.1006x formal episode throughput. The
[report](../FIXED_BASELINE_2026-11-10.md) defines the configurations and limitations.

| File | Purpose |
| --- | --- |
| [Fixed plan](issue19-fixed-2026-11-09-plan.json) | Baseline commit, pins, hardware, per-suite K/g, OMP and allocator settings. |
| [Fixed summary](issue19-fixed-2026-11-09.json) | Formal totals, per-suite/task scores, steps, phase timings and recovery/resources. |
| [Optimized plan](optimized-2026-11-10-plan.json) | Latest optimized source snapshot, switches, tuning mode and Policy threads. |
| [Comparison](fixed-vs-optimized-2026-11-10.json) | Both arm totals, per-suite optimized results, tuning probes, phase timings and paired differences. |
| [Suite CSV](fixed-vs-optimized-2026-11-10-suites.csv) | 16 suites with both scores, timings, K/g and discrepancy counts. |
| [Task CSV](fixed-vs-optimized-2026-11-10-tasks.csv) | 160 tasks with both scores and discrepancy counts. |
| [Paired CSV](fixed-vs-optimized-2026-11-10-paired.csv) | Exactly 8,000 rows keyed by suite/task/episode; both success flags and actual action steps. |
| [Camera replay](camera-replay-2026-11-10.json) | Fixed-action camera-on/off image/state comparison using the measured direct-sensor implementation. |

## Extraction provenance

The five 2026-11-10 comparison/plan/CSV files above are byte-for-byte copies of the corresponding
files in the retained local `full8k-250v85vu` result and its suite/task/paired exports. Their SHA-256
digests, in table order, are:

| File | SHA-256 |
| --- | --- |
| Optimized plan | `3c49395a3c5cc2088e84965baaa9387f138f760ef08c51eed981cb866f6c3adf` |
| Comparison | `615fb9f688952e661d6fab001383a3c1d26f36edf655387f2416684e79124f1f` |
| Suite CSV | `9cfa1bc6c29ec712ef24a279a3c62f8cbe48a807165ccb8a81edeb05576b7dcd` |
| Task CSV | `963e2501452c892f8c1b0be563f39e7a340baf556514462c02c4a297539b9f29` |
| Paired CSV | `8ea93650773531a4200e1e43f89c6b3c337581ecb3ab40faa00091df7478ed01` |

The fixed summary and plan were extracted from the earlier source evidence at `e817d18`; they retain
the measured baseline values and omit unrelated historical-reference scores. The optimized run is
`dfebac7` plus its recorded camera patch, not the later draft head. Its source archive SHA-256 is
`90cc1e0838ef8ab7fc9c345234a37e9e32312cf0992f9240067f7d4dc7d6b260`.

## Raw records

The original fixed run directory on the experiment host was
`/root/autodl-tmp/issue21-20261108/comparison/issue19-repro-b29-20261109-0QXYVUBi/`.
Local copies are stored outside the repository at `/home/fan_wenk/RoboScope-Kit-results/`:

| Archive | SHA-256 |
| --- | --- |
| `issue19-b29-fixed-20261109.tar.gz` | `c691d1b1ca9b7c3d61a435425d6c794baa45ce8649d0970aef7e39a09cf50d64` |
| `full8k-250v85vu-results.tar.gz` | `432f21ae0a89fddaa6332ae2734e07f39b7f6203d32088001d535541ac0bb663` |

These filesystem locations are provenance, not publicly accessible artifact links. Publish durable
links to the selected raw records and exact launcher commands before final delivery.

## Reproduction outline

Build the pinned simulator with `benchmarks/libero_pro/build-runtime.sh`, prepare the checkpoint using
the [StarVLA instructions](../../../examples/starvla/README.md), and verify the report's hashes.
Use separate checkouts and controller/model environments for the recorded snapshots, applying the
recorded camera patch to the optimized source. Each Policy configuration must invoke the matching
checkout and set its OMP value in the Policy environment. Use manifest copies with absolute runtime
paths, without changing the built installation.

Run each standard suite with 50 episodes per task and a fresh output directory:

```bash
# Fixed arm: K=2/G=1 except the three libero_10 suites listed in the fixed plan, K=8/G=2.
# Policy OMP_NUM_THREADS=8, JSON, MPS off, batch cap one; allocator setting per fixed plan.
"$FIXED_PYTHON" -m roboscope_kit.runtime.benchmark \
  --benchmark libero_pro --suite "$SUITE" --episodes 50 \
  --checkpoint-source "$CHECKPOINT" --benchmark-manifest "$FIXED_MANIFEST" \
  --policy-config "$FIXED_POLICY" --concurrency "$K" --replicas "$G" \
  --output-dir "$FIXED_OUTPUT"

# Optimized arm: Policy OMP_NUM_THREADS=1 and ROBOSCOPE_STARVLA_MAX_BATCH_SIZE=2;
# MessagePack and chunk_camera_rendering=true in the manifest, persistent connections/video off.
"$OPTIMIZED_PYTHON" -m roboscope_kit.runtime.benchmark \
  --benchmark libero_pro --suite "$SUITE" --episodes 50 \
  --checkpoint-source "$CHECKPOINT" --benchmark-manifest "$OPTIMIZED_MANIFEST" \
  --policy-config "$OPTIMIZED_POLICY" --autotune-strategy diagonal --cuda-mps \
  --output-dir "$OPTIMIZED_OUTPUT"
```

Run Quick Tests separately. Keep the dedicated GPU, CPU/memory limits and cached inputs consistent.
The command templates require host-specific paths; use the archived commands to audit the measured
runs. Recompute coverage and totals from terminal results and formal `evaluation_s`, excluding warmup
and autotune episodes. Preserve failures and differences rather than selecting only successful episodes.
