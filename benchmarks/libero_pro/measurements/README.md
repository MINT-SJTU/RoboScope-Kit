# Selected draft evidence

This directory contains only the **historical fixed baseline versus optimized** comparison selected
for the draft: 16 standard suites and 8,000 episodes per side, with 2.9141x formal throughput.
The [report](../FIXED_BASELINE_2026-11-09.md) defines its configurations and limitations.

| File | Purpose |
| --- | --- |
| [Fixed plan](issue19-fixed-2026-11-09-plan.json) | Baseline commit, pins, hardware, per-suite K/g, OMP and allocator settings. |
| [Fixed summary](issue19-fixed-2026-11-09.json) | Formal totals, per-suite/task scores, steps, phase timings and recovery/resources. |
| [Optimized plan](optimized-2026-11-09-plan.json) | Optimized commit, switches, tuning mode and Policy threads. |
| [Optimized summary](optimized-2026-11-09.json) | Formal totals, per-suite/task scores, selected K/g and probes, timing and recovery/resources. |
| [Comparison](fixed-vs-optimized-2026-11-09.json) | Per-suite two-arm comparison, paired discrepancy counts and overall throughput ratios. |
| [Paired CSV](fixed-vs-optimized-2026-11-09-paired.csv) | Exactly 8,000 rows keyed by suite/task/episode; fixed and optimized success flags and actual action steps. |

## Extraction provenance

The source evidence was committed at `e817d18`. Derived JSON files record the original source name,
SHA-256 and transformation. The optimized summary selects only the optimized arm from the earlier
multi-arm records. The fixed summary omits historical-reference scores, retaining measured results.
The comparison omits the other baseline's fields. The paired CSV selects `suite`, `task_id`,
`episode_index`, `fixed_success`, `optimized_success`, `fixed_steps` and `optimized_steps` from
`issue19-fixed-vs-auto-2026-11-09-paired.csv` at that commit. No retained measured value is changed.
Earlier source files can be recovered with `git show e817d18:benchmarks/libero_pro/measurements/<name>`.
They are also archived under the ignored local `.local/issue21-draft-archive-20261109/` directory.
Archive contents are not part of the draft's final file tree; historical Git commits are not rewritten.

## Raw records

The original run directories on the experiment host are:

- Fixed: `/root/autodl-tmp/issue21-20261108/comparison/issue19-repro-b29-20261109-0QXYVUBi/`.
- Optimized: the optimized arm under
  `/root/autodl-tmp/issue21-20261108/comparison/full-auto-20261109-0125/`.

Local copies are stored outside the repository at `/home/fan_wenk/RoboScope-Kit-results/`:

| Archive | SHA-256 |
| --- | --- |
| `issue19-b29-fixed-20261109.tar.gz` | `c691d1b1ca9b7c3d61a435425d6c794baa45ce8649d0970aef7e39a09cf50d64` |
| `full-auto-20261109-0125.tar.gz` | `3697a18977b46367e9253bf0260e0458e714a5ba067ed21e2b98e718486ad020` |

The second archive also preserves the original experiment's other arm; only its optimized arm is used
in this draft. These filesystem locations are provenance, not publicly accessible artifact links.
Publish durable links to the selected raw records and exact launcher commands before final delivery.

## Reproduction outline

Build the pinned simulator with `benchmarks/libero_pro/build-runtime.sh`, prepare the checkpoint using
the [StarVLA instructions](../../../examples/starvla/README.md), and verify the report's hashes.
Use separate checkouts and controller/model environments for the two recorded commits. Each Policy
configuration must invoke the matching checkout and set its OMP value in the Policy environment.
Use manifest copies with absolute runtime paths, without changing the built installation.

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
