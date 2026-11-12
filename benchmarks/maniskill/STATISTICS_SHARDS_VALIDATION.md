# Statistics and shard correction

On 2026-11-12, CPU-only replay reproduces and fixes the return-denominator and
missing-reducer findings in PR #99. This does not add simulator/task/model
acceptance or change the frozen evaluation inputs. No GPU evaluation is run.

## Actual archived records

The input is the immutable R12/b51 archive at source `42e6b24`, SHA-256
`506a19adb138cf40aa02111b653d4a472645b62202cb87923b4fd4fc079811db`.
It contains complete-horizon records, not five-step prefixes. The replay leaves
their episode identities, actions, returns, horizons and actual terminal masks
unchanged; it only reconstructs summaries and deterministic shard partitions.

| Original batch | Episodes / steps | Old generic success rate | Correct success scoring | Return-only episodes |
|---|---:|---:|---:|---:|
| NEWT mixed profiles | 9 / 5,400 | 2/9 = 22.22% | 2/3 = 66.67%; one unsuccessful scored episode | 6 |
| Humanoid profiles | 5 / 5,000 | 0%; five incorrectly counted failures | N/A; zero binary failures | 5 |
| ANYmal Spin | 1 / 200 | 0%; one incorrectly counted failure | N/A | 1 |
| Unitree standing | 2 / 2,000 | 0%; two incorrectly counted failures | N/A | 2 |
| Official PPO profiles | 11 / 880 | 10/11 = 90.91% | Unchanged | 0 |

These are separate historical evidence batches. They are not combined into a
current-head benchmark score or relabeled as new 100-episode validation. The
current full acceptance gaps and policy-selection concerns remain open.

## Checks and limits

- All 270 existing Linux pytest cases pass; Ruff and the Python-support check pass.
- The actual NEWT, Humanoid and PPO WorkResults replay through the adapter and
  result-only reducer with two and three shards. Per-profile counts, success
  results and sample-weighted returns agree with direct unsharded aggregation.
- The same three families replay through the existing `shard_merge` record-tree
  entry point in six partitions. It verifies completed WorkItems and actual
  profile/index/seed/lane/record identity, horizon, boolean terminal masks and
  agreement between each stored summary and its completed records.
- Thirty record-tree perturbations are rejected, including unknown or duplicate
  records, wrong seed/lane/profile/steps, missing records/masks, integer masks and
  altered summaries. Separate reducer replay rejects 36 inconsistent shard,
  profile, width, model and score mutations. Eight standard-statistics mutations
  are rejected. Unequal-size negative-return and empty-profile cases pass.
- Content-identity fixtures verify equal relative inputs at different paths and
  changed checkpoint/Policy bytes. Replay hashes are explicitly synthetic fixture
  identities; they do not establish new live model/shard provenance.

Integer counts and record identities compare exactly. Floating sufficient-statistic
reduction is checked at `rel_tol=1e-12, abs_tol=1e-10` to allow addition-order
rounding. This tolerance is not applied to model inputs, actions, observations,
success outcomes or terminal masks.

Result-only reduction reports **`coverage="accounted"`**. It cannot inspect
actual records or independently prove episode uniqueness. Record-tree merging
reports **`coverage="complete"`** after actual identity and coverage checks.
Neither label establishes that every upstream task has a trained acceptable
checkpoint or that the full benchmark acceptance protocol has been completed.

No new repository tests are added because the current `AGENTS.md` prohibits them;
the external replay scripts and raw fixtures provide the new regression evidence.
The maintainer review's request for new repository tests remains distinct from
the passing existing suite. Simulator/build scripts and the remote workspace's
older environment/configuration are untouched. No new GPU self-test is claimed
for this arithmetic/merge correction.

Raw replay scripts, original records, perturbation receipts, installed-wheel
checks and the per-profile historical statistics table are stored outside Git
with the [statistics/shard evidence](https://github.com/QLL-CCDD/RoboScope-Kit/releases/tag/maniskill-issue25-statistics-shards-20261112).
The [scoring protocol](PROTOCOL.md#scoring-and-coverage) specifies the two merge paths.
