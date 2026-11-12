# Recorded performance aggregation

- [Timing and outcomes](timing-outcomes.json): Native, Kit 1/1, and Kit 4/1.
- [Latency and resources](latency-resources.json): recorded request and resource aggregation for those three runs.
- [Combined comparison](comparison.json): measurement definitions and the recorded aggregate results.
- [Aggregation script](aggregate_remote.py): the script used on the experiment machine.
- [All configurations by suite](../../tables/performance-by-suite.csv) and [totals](../../tables/performance-totals.csv): recomputed from the included run results and status records.

Kit 2/2 and 4/2 retain their suite latency and resource summaries in their own run
directories. Per-suite percentiles are not averaged to obtain pooled percentiles.
The full report preserves the originally reported pooled latency values; the
per-suite summaries alone do not reconstruct those pooled distributions.
