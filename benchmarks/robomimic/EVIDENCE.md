# Issue 32 delivery evidence

The [review validation](REVIEW_VALIDATION.md) defines the current matched comparisons. The
[historical report](THROUGHPUT.md) preserves the original measurements and explicitly withdraws the
MPS-confounded >=2x acceptance claim. Evidence releases on the contributor fork are reproducibility
assets, not Kit package releases. Issue #32 remains open at the measured **1.745x** combined gain.

## Review revision, 2026-11-09

- [Review evidence release](https://github.com/OopsJoey/RoboScope-Kit/releases/tag/issue32-review-evidence-2026-11-09).
- [Complete raw evidence, including per-step traces](https://github.com/OopsJoey/RoboScope-Kit/releases/download/issue32-review-evidence-2026-11-09/issue32-review-evidence-2026-11-09.tar.gz).
- [Small summary bundle and reproduction scripts](https://github.com/OopsJoey/RoboScope-Kit/releases/download/issue32-review-evidence-2026-11-09/issue32-review-summary-2026-11-09.tar.gz).
- [SHA256SUMS](https://github.com/OopsJoey/RoboScope-Kit/releases/download/issue32-review-evidence-2026-11-09/SHA256SUMS).

Complete review archive: **522,258,952 bytes**, SHA-256
**0158d44249b2eb34ef28c9d1ccf5e2c0de1262ce1afa074b86abfd03d435b6af**.
Its inventory covers 5,759 files. The historical archives below are preserved unchanged; the original
correctness baseline and earlier clean-build logs used by this review are in those archives.

| Evidence | Path in the review archive |
| --- | --- |
| Hardware limits, driver/GPU UUID, checkpoints and simulator binding hashes | review-2026-11-09/host.json |
| Three repeats per Kit configuration, three native repeats, all rates/variance/resources | review-2026-11-09/performance-summary.json |
| Exact run order, resource guards, commands and external drivers | review-2026-11-09/matched-repeats.sh, diagnostics.sh, trace.py, compare-traces.py, native.py, summarize.py; tools/ |
| Every numeric trajectory array, initial XML, episode metadata and cache/context counters | review-2026-11-09/trace-original/, trace-cacheoff/, trace-cacheon/ |
| Three array comparisons, tolerance, first divergence and repeated-reset checks | review-2026-11-09/pair-original-cacheoff.json, pair-original-cacheon.json, pair-cacheoff-cacheon.json |
| Complete Kit output and native per-episode records | runs/review110-*/ |
| CLI/environment, process timing and one-second resource samples | evidence/review110-*/command.json, execution.json, resources.jsonl |
| Kit per-record baseline pairing | evidence/review110-*-pairing.json |
| Native model/import timing, worker stage sums, rollout function hash and pairing | runs/review110-native-k20-off-0{1,2,3}/summary.json |
| Existing checks and new output-directory self-test | review-2026-11-09/checks.sh, checks.log, checks.exit, self-test/self-test.json |
| Measured original/final source snapshots and Policy configurations | baseline-fixed/, final-b43/, policy-baseline-fixed.json, policy-final-b43.json |
| Per-file bytes and SHA-256 | inventory.json |

The small summary bundle contains the completed summaries, comparisons, driver/check scripts and logs,
but omits the NPZ trajectories. Raw traces and all per-episode outputs are in the complete archive.
The release also includes the transfer/inventory verification, executable-source correspondence and
final documentation-revision check log. No checkpoints, datasets, wheelhouse or credentials are included.
All diagnostic scripts are external evidence; no repository tests were added.

## Historical evidence, 2026-11-08

The following assets retain the initial delivery, prior build/check results and the 1.940x fallback.
Their unmatched-MPS timing ratios are historical observations, not #32 acceptance results:

- [Evidence release and assets](https://github.com/OopsJoey/RoboScope-Kit/releases/tag/issue32-evidence-2026-11-08).
- [Raw archive](https://github.com/OopsJoey/RoboScope-Kit/releases/download/issue32-evidence-2026-11-08/issue32-delivery-evidence-2026-11-08.tar.gz).
- [SHA256SUMS](https://github.com/OopsJoey/RoboScope-Kit/releases/download/issue32-evidence-2026-11-08/SHA256SUMS).
- [Final builder self-test result](https://github.com/OopsJoey/RoboScope-Kit/releases/download/issue32-evidence-2026-11-08/final-self-test.json).

Raw archive: 23,956,565 bytes, SHA-256
**3908e6022fca50a5ddf6d7a18c350c36a0b64c079dc724cb684e44d93b7c8747**.

The [complete original remote archive](https://github.com/OopsJoey/RoboScope-Kit/releases/download/issue32-evidence-2026-11-08/issue32-raw-evidence-2026-11-08.tar.gz)
was subsequently resumed and verified: 38,072,005 bytes, SHA-256
**c5462864420d0ee345a110ce26c95a11880ba322578ded20da8b4705f5750207**.
Its entire 9,415-file hash inventory passed verification. It additionally preserves every historical
candidate source tree and the complete remote helper set. Both archives are listed in SHA256SUMS.

### Historical archive map

| Requirement | Path inside the archive |
| --- | --- |
| Hardware, checkpoints, package provenance | evidence/host.json, evidence/checkpoints.json, evidence/wheel-cache.json |
| Exact CLI, environment, process interval, exit status | evidence/&lt;label&gt;/command.json and execution.json |
| One-second CPU/RAM/GPU/VRAM samples | evidence/&lt;label&gt;/resources.jsonl |
| Complete task/seed coverage, aggregate outcomes, resource summaries | evidence/&lt;label&gt;/summary.json |
| Raw episode results, work manifest, logs, metadata | runs/&lt;label&gt;/ |
| Record-by-record comparison | evidence/&lt;label&gt;-pairing.json |
| Upstream native comparisons and original logs | runs/baseline-native/ |
| Worker/Policy CPU profiles | evidence/baseline-cpu-profile/ |
| Nsight profile artifacts | runs/baseline-profile/profile/ |
| Exact source snapshots and configurations | sources/, policy-*.json, tools/ |
| Fresh build and all 250 state digests | evidence/build-final-b43.log, runtime-final-b43/ |
| Final builder self-test result | Separate final-self-test.json release asset (runtime-final-b43/self-test/self-test.json) |
| Existing checks, exit code 0 | evidence/final-checks.log, evidence/final-checks.exit |
| Final full correctness run, exit code 0 | evidence/final-acceptance.exit, runs/final-b43-k20g8-mps-01/ |
| Separate writing diagnostic | output-writing.json, evidence/final-output-profile-01/io/, tools/output-profile/ |
| Hash inventory and delivery/source checks | inventory.json, delivery-validation.json |

Historical completed performance labels are baseline-full-01/02; baseline-k16g4-01;
baseline-k20g8-mps-01; baseline-k32g8-mps-01; candidate-v4/v6/v7 labels listed in the report;
and final-b43-k20g8-mps-01. The output profile is a separate real evaluation, not another timing repeat.
Rejected forward variants and preparation failures are retained and explicitly excluded from accepted gains.

The original-code control is the base commit plus the MessagePack runtime dependency repair.
Candidate v4 and v6 used runtime-baseline-fixed; use their recorded command/runtime, not old unused
builder experiments in their source snapshots. Candidate v7 and final-b43 use their own fresh builds.
The final code matches the measured final-b43 source; subsequent changes are documentation only.

The full final build and self-test passed, 264 existing pytest cases passed, and Ruff, Python support,
uv lock consistency, bash syntax and ShellCheck passed. The final run has 250 unique records,
81,289 steps, 176 successes, all initial-state checks, and no missing/extra records, pairing mismatches,
Worker failures, retries or OOM events. No new tests were added.

Absolute paths record the measurement host and should be adjusted for reproduction. Checkpoints,
datasets, wheelhouse, Git source bundles, SSH session state and credentials are excluded from the archive.
The preserved 1.940x fallback is a separate release asset with its own checksum and source/runtime notes.

## Transfer provenance

The SSH transfer of the original larger archive ended after all run/evidence/runtime records had arrived,
during historical source snapshots. The delivery archive contains only complete downloaded tar members,
plus the final measured source and v6 fallback from previously SHA-256-verified local archives. It is a
new archive with the checksum above, not a claim that the interrupted transfer completed. All 22 full-run
record counts and steps, final repeated per-record comparisons, native results, final check exit codes,
and non-documentation source correspondence to the Git commit were verified before packaging.
Exact source trees in the compact delivery archive are baseline-fixed, candidate-v1/v2/v6, and final-b43.
The complete original archive linked above additionally contains the other historical source trees.
The recovery procedure and file hashes are recorded in delivery-validation.json
and inventory.json. No episode, resource sample, build/check log, or native result was edited.
