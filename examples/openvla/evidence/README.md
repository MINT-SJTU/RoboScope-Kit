# OpenVLA evaluation evidence

This directory groups the evidence used by the full OpenVLA evaluation report:
five 200-episode evaluations (Native and Kit 1/1, 4/1, 2/2, 4/2), three completed
Autotune suites (150 formal episodes), and 36 paired diagnostics (72 rollouts).
Probe episodes and action replays are identified separately from formal evaluations.
The four full-suite task sets contain ten tasks each, with initial-state indices 0–4.

## Start with the report's result tables

| Question | Table / analysis | Supporting records |
| --- | --- | --- |
| Which tasks ran, and how often did they succeed? | [Task coverage and success](tables/task-coverage.csv) | [Native](runs/native/README.md), [Kit 1/1](runs/kit-1x1/README.md), [Kit 4/1](runs/kit-4x1/README.md), [Kit 2/2](runs/kit-2x2/README.md), [Kit 4/2](runs/kit-4x2/README.md) |
| Which Native / Kit 2/2 episodes disagree? | [200 paired outcomes and steps](tables/historical-200.csv) | [Native episodes](runs/native/README.md), [Kit completed work items](runs/kit-2x2/README.md) |
| Where do diagnostic trajectories first differ? | [36 diagnostic pairs](tables/diagnostic-36.csv), [analysis](tables/ANALYSIS.md) | [Per-case traces, numerical probes, inputs, and logs](diagnostics/README.md) |
| What are the timing and throughput results? | [Per-suite measurements](tables/performance-by-suite.csv), [aggregate measurements](tables/performance-totals.csv) | [Performance aggregation](runs/performance/README.md) and each run's status and measurement files |
| What did Autotune complete and select? | [Task coverage](tables/task-coverage.csv), [formal timings](tables/performance-by-suite.csv) | [Autotune results, candidate selections, and retry records](runs/autotune/README.md) |
| How were the experiments configured? | [Hardware, runtime, and commands](reproduction/RUNS.md) | [Experiment metadata](reproduction/PROVENANCE.json), per-run manifests and launchers |

## Reading and checking the evidence

CSV table paths are relative to the table's directory. Native records use
`episodes.jsonl` and `summary.json`; Kit records use `result.json` and `metrics.json`.
Kit `completed.jsonl` files consolidate original work-item objects without dropping
fields. They supply the individual action counts for the historical 2/2 comparison
and the diagnostic. The task coverage table leaves action-count cells empty where
only suite totals were retained; it does not infer individual counts from totals.

Diagnostic traces preserve every recorded step's image hashes and actions. Large
JSONL streams are gzip-compressed; seven representative original PNGs are linked
in the [image guide](diagnostics/images/README.md). Paths inside preserved experimental records
and scripts refer to the experiment machine. The directory indexes link to the
corresponding files in this evidence package.

[Checksums](SHA256SUMS) cover every evidence file except the checksum file itself.
[The source index](reproduction/SOURCE_INDEX.json) records the retained inputs,
original byte hashes, compression, and work-item consolidation. Derived tables
are checked against their corresponding run results; [verification results](reproduction/VERIFICATION.json)
record the performed checks.

## Interpretation

The historical Native and Kit 2/2 runs each succeed in 146/200 episodes but disagree
on 36 identities. The diagnostic revisits those identities and retains 18 outcome
disagreements. Execution order and reset history differ between the full runs and
the diagnostic subset, so the outcome changes do not isolate repeat variation
under identical execution conditions. The numerical probes locate differences
in preprocessing, action precision, and initial observations without establishing
a single cause for every outcome difference. This package documents the measured
results and their scope; it does not assert statistical equivalence or completed
benchmark acceptance.
