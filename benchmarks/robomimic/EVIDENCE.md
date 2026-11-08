# Issue 32 delivery evidence

The [throughput report](THROUGHPUT.md) defines the comparisons, timing boundaries and limitations.
The evidence is hosted on the contributor fork as an evidence-only prerelease, not a Kit release:

- [Evidence release and assets](https://github.com/OopsJoey/RoboScope-Kit/releases/tag/issue32-evidence-2026-11-08).
- [Raw archive](https://github.com/OopsJoey/RoboScope-Kit/releases/download/issue32-evidence-2026-11-08/issue32-delivery-evidence-2026-11-08.tar.gz).
- [SHA256SUMS](https://github.com/OopsJoey/RoboScope-Kit/releases/download/issue32-evidence-2026-11-08/SHA256SUMS).

Raw archive: 23,956,565 bytes, SHA-256
**3908e6022fca50a5ddf6d7a18c350c36a0b64c079dc724cb684e44d93b7c8747**.

## Archive map

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
| Fresh build, 250 state digests and self-test | evidence/build-final-b43.log, runtime-final-b43/ |
| Existing checks, exit code 0 | evidence/final-checks.log, evidence/final-checks.exit |
| Final full acceptance, exit code 0 | evidence/final-acceptance.exit, runs/final-b43-k20g8-mps-01/ |
| Separate writing diagnostic | output-writing.json, evidence/final-output-profile-01/io/, tools/output-profile/ |
| Hash inventory and delivery/source checks | inventory.json, delivery-validation.json |

Accepted performance labels are baseline-full-01/02; baseline-k16g4-01;
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
Exact source trees included are baseline-fixed, candidate-v1/v2/v6, and final-b43; other historical trees
are not claimed complete. The recovery procedure and file hashes are recorded in delivery-validation.json
and inventory.json. No episode, resource sample, build/check log, or native result was edited.
