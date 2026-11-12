# Diagnostic records

The selected workload is 36 historical disagreements, with one rollout per path
for each identity. These 72 rollouts are distinct from the full evaluations.

- [Paired table](../tables/diagnostic-36.csv) · [analysis](../tables/ANALYSIS.md)
- [Full per-step comparison](comparison.json) · [coverage and execution checks](validation.json)
- [Configuration and hardware](expanded-experiment.json) · [selection](selection.json)
- [Preprocessing coverage](preprocess-probe/coverage.json)
- [Action replay](action-replay/summary.json) · [Kit Worker replay](action-replay/kit-worker-float32/summary.json)
- [Reset inspection](reset-probe-all/summary.json)
- [Diagnostic scripts](source/) · [recorded commands](../reproduction/RUNS.md)
- [Representative diagnostic images](images/README.md)

Seven original PNGs illustrate preprocessing differences, unequal initial
observations, and model inputs at an action-divergence milestone. The image guide
identifies each case and links its numerical evidence. All recorded step hashes
and actions remain in the trace streams; large streams use gzip compression.
[The source index](../reproduction/SOURCE_INDEX.json) identifies each retained
image and original stream. Paths to other image/tensor snapshots in original
records describe the experiment outputs; those snapshots are not bundled here.

| Suite | Episode | Native records | Kit records | Native trace | Kit trace |
| --- | --- | --- | --- | --- | --- |
| 10 | task-0000-episode-0002 | [episodes](10-native/episodes.jsonl) | [completed](10-kit/completed.jsonl) | [steps](10-native/trace/steps.jsonl.gz) | [steps](10-kit/policy-trace/steps.jsonl.gz) |
| 10 | task-0001-episode-0001 | [episodes](10-native/episodes.jsonl) | [completed](10-kit/completed.jsonl) | [steps](10-native/trace/steps.jsonl.gz) | [steps](10-kit/policy-trace/steps.jsonl.gz) |
| 10 | task-0001-episode-0002 | [episodes](10-native/episodes.jsonl) | [completed](10-kit/completed.jsonl) | [steps](10-native/trace/steps.jsonl.gz) | [steps](10-kit/policy-trace/steps.jsonl.gz) |
| 10 | task-0002-episode-0000 | [episodes](10-native/episodes.jsonl) | [completed](10-kit/completed.jsonl) | [steps](10-native/trace/steps.jsonl.gz) | [steps](10-kit/policy-trace/steps.jsonl.gz) |
| 10 | task-0002-episode-0001 | [episodes](10-native/episodes.jsonl) | [completed](10-kit/completed.jsonl) | [steps](10-native/trace/steps.jsonl.gz) | [steps](10-kit/policy-trace/steps.jsonl.gz) |
| 10 | task-0002-episode-0002 | [episodes](10-native/episodes.jsonl) | [completed](10-kit/completed.jsonl) | [steps](10-native/trace/steps.jsonl.gz) | [steps](10-kit/policy-trace/steps.jsonl.gz) |
| 10 | task-0004-episode-0003 | [episodes](10-native/episodes.jsonl) | [completed](10-kit/completed.jsonl) | [steps](10-native/trace/steps.jsonl.gz) | [steps](10-kit/policy-trace/steps.jsonl.gz) |
| 10 | task-0006-episode-0001 | [episodes](10-native/episodes.jsonl) | [completed](10-kit/completed.jsonl) | [steps](10-native/trace/steps.jsonl.gz) | [steps](10-kit/policy-trace/steps.jsonl.gz) |
| 10 | task-0007-episode-0000 | [episodes](10-native/episodes.jsonl) | [completed](10-kit/completed.jsonl) | [steps](10-native/trace/steps.jsonl.gz) | [steps](10-kit/policy-trace/steps.jsonl.gz) |
| 10 | task-0007-episode-0001 | [episodes](10-native/episodes.jsonl) | [completed](10-kit/completed.jsonl) | [steps](10-native/trace/steps.jsonl.gz) | [steps](10-kit/policy-trace/steps.jsonl.gz) |
| 10 | task-0008-episode-0001 | [episodes](10-native/episodes.jsonl) | [completed](10-kit/completed.jsonl) | [steps](10-native/trace/steps.jsonl.gz) | [steps](10-kit/policy-trace/steps.jsonl.gz) |
| 10 | task-0009-episode-0003 | [episodes](10-native/episodes.jsonl) | [completed](10-kit/completed.jsonl) | [steps](10-native/trace/steps.jsonl.gz) | [steps](10-kit/policy-trace/steps.jsonl.gz) |
| 10 | task-0009-episode-0004 | [episodes](10-native/episodes.jsonl) | [completed](10-kit/completed.jsonl) | [steps](10-native/trace/steps.jsonl.gz) | [steps](10-kit/policy-trace/steps.jsonl.gz) |
| goal | task-0000-episode-0002 | [episodes](goal-native/episodes.jsonl) | [completed](goal-kit/completed.jsonl) | [steps](goal-native/trace/steps.jsonl.gz) | [steps](goal-kit/policy-trace/steps.jsonl.gz) |
| goal | task-0000-episode-0003 | [episodes](goal-native/episodes.jsonl) | [completed](goal-kit/completed.jsonl) | [steps](goal-native/trace/steps.jsonl.gz) | [steps](goal-kit/policy-trace/steps.jsonl.gz) |
| goal | task-0000-episode-0004 | [episodes](goal-native/episodes.jsonl) | [completed](goal-kit/completed.jsonl) | [steps](goal-native/trace/steps.jsonl.gz) | [steps](goal-kit/policy-trace/steps.jsonl.gz) |
| goal | task-0004-episode-0001 | [episodes](goal-native/episodes.jsonl) | [completed](goal-kit/completed.jsonl) | [steps](goal-native/trace/steps.jsonl.gz) | [steps](goal-kit/policy-trace/steps.jsonl.gz) |
| goal | task-0005-episode-0004 | [episodes](goal-native/episodes.jsonl) | [completed](goal-kit/completed.jsonl) | [steps](goal-native/trace/steps.jsonl.gz) | [steps](goal-kit/policy-trace/steps.jsonl.gz) |
| goal | task-0008-episode-0004 | [episodes](goal-native/episodes.jsonl) | [completed](goal-kit/completed.jsonl) | [steps](goal-native/trace/steps.jsonl.gz) | [steps](goal-kit/policy-trace/steps.jsonl.gz) |
| goal | task-0009-episode-0002 | [episodes](goal-native/episodes.jsonl) | [completed](goal-kit/completed.jsonl) | [steps](goal-native/trace/steps.jsonl.gz) | [steps](goal-kit/policy-trace/steps.jsonl.gz) |
| goal | task-0009-episode-0003 | [episodes](goal-native/episodes.jsonl) | [completed](goal-kit/completed.jsonl) | [steps](goal-native/trace/steps.jsonl.gz) | [steps](goal-kit/policy-trace/steps.jsonl.gz) |
| object | task-0002-episode-0004 | [episodes](object-native/episodes.jsonl) | [completed](object-kit/completed.jsonl) | [steps](object-native/trace/steps.jsonl.gz) | [steps](object-kit/policy-trace/steps.jsonl.gz) |
| object | task-0003-episode-0000 | [episodes](object-native/episodes.jsonl) | [completed](object-kit/completed.jsonl) | [steps](object-native/trace/steps.jsonl.gz) | [steps](object-kit/policy-trace/steps.jsonl.gz) |
| object | task-0003-episode-0002 | [episodes](object-native-extra/episodes.jsonl) | [completed](object-kit/completed.jsonl) | [steps](object-native-extra/trace/steps.jsonl.gz) | [steps](object-kit/policy-trace/steps.jsonl.gz) |
| object | task-0006-episode-0000 | [episodes](object-native-extra/episodes.jsonl) | [completed](object-kit/completed.jsonl) | [steps](object-native-extra/trace/steps.jsonl.gz) | [steps](object-kit/policy-trace/steps.jsonl.gz) |
| object | task-0006-episode-0004 | [episodes](object-native-extra/episodes.jsonl) | [completed](object-kit/completed.jsonl) | [steps](object-native-extra/trace/steps.jsonl.gz) | [steps](object-kit/policy-trace/steps.jsonl.gz) |
| object | task-0008-episode-0004 | [episodes](object-native-extra/episodes.jsonl) | [completed](object-kit/completed.jsonl) | [steps](object-native-extra/trace/steps.jsonl.gz) | [steps](object-kit/policy-trace/steps.jsonl.gz) |
| spatial | task-0001-episode-0003 | [episodes](spatial-native/episodes.jsonl) | [completed](spatial-kit/completed.jsonl) | [steps](spatial-native/trace/steps.jsonl.gz) | [steps](spatial-kit/policy-trace/steps.jsonl.gz) |
| spatial | task-0003-episode-0002 | [episodes](spatial-native-extra/episodes.jsonl) | [completed](spatial-kit/completed.jsonl) | [steps](spatial-native-extra/trace/steps.jsonl.gz) | [steps](spatial-kit/policy-trace/steps.jsonl.gz) |
| spatial | task-0004-episode-0003 | [episodes](spatial-native-extra/episodes.jsonl) | [completed](spatial-kit/completed.jsonl) | [steps](spatial-native-extra/trace/steps.jsonl.gz) | [steps](spatial-kit/policy-trace/steps.jsonl.gz) |
| spatial | task-0005-episode-0001 | [episodes](spatial-native-extra/episodes.jsonl) | [completed](spatial-kit/completed.jsonl) | [steps](spatial-native-extra/trace/steps.jsonl.gz) | [steps](spatial-kit/policy-trace/steps.jsonl.gz) |
| spatial | task-0006-episode-0003 | [episodes](spatial-native-extra/episodes.jsonl) | [completed](spatial-kit/completed.jsonl) | [steps](spatial-native-extra/trace/steps.jsonl.gz) | [steps](spatial-kit/policy-trace/steps.jsonl.gz) |
| spatial | task-0007-episode-0000 | [episodes](spatial-native/episodes.jsonl) | [completed](spatial-kit/completed.jsonl) | [steps](spatial-native/trace/steps.jsonl.gz) | [steps](spatial-kit/policy-trace/steps.jsonl.gz) |
| spatial | task-0008-episode-0001 | [episodes](spatial-native-extra/episodes.jsonl) | [completed](spatial-kit/completed.jsonl) | [steps](spatial-native-extra/trace/steps.jsonl.gz) | [steps](spatial-kit/policy-trace/steps.jsonl.gz) |
| spatial | task-0008-episode-0002 | [episodes](spatial-native-extra/episodes.jsonl) | [completed](spatial-kit/completed.jsonl) | [steps](spatial-native-extra/trace/steps.jsonl.gz) | [steps](spatial-kit/policy-trace/steps.jsonl.gz) |
| spatial | task-0008-episode-0003 | [episodes](spatial-native-extra/episodes.jsonl) | [completed](spatial-kit/completed.jsonl) | [steps](spatial-native-extra/trace/steps.jsonl.gz) | [steps](spatial-kit/policy-trace/steps.jsonl.gz) |

## Batch results and logs

| Batch | Result | Configuration | Logs |
| --- | --- | --- | --- |
| 10-kit | [result](10-kit/result.json) | [manifest](10-kit/manifest.json) | [Workers](10-kit/workers/) · [Policy](10-kit/servers/) |
| 10-native | [result](10-native/summary.json) | [manifest](10-native/run-manifest.json) | [stdout](10-native/upstream.stdout.log) |
| goal-kit | [result](goal-kit/result.json) | [manifest](goal-kit/manifest.json) | [Workers](goal-kit/workers/) · [Policy](goal-kit/servers/) |
| goal-native | [result](goal-native/summary.json) | [manifest](goal-native/run-manifest.json) | [stdout](goal-native/upstream.stdout.log.gz) |
| object-kit | [result](object-kit/result.json) | [manifest](object-kit/manifest.json) | [Workers](object-kit/workers/) · [Policy](object-kit/servers/) |
| object-native | [result](object-native/summary.json) | [manifest](object-native/run-manifest.json) | [stdout](object-native/upstream.stdout.log) |
| object-native-extra | [result](object-native-extra/summary.json) | [manifest](object-native-extra/run-manifest.json) | [stdout](object-native-extra/upstream.stdout.log) |
| spatial-kit | [result](spatial-kit/result.json) | [manifest](spatial-kit/manifest.json) | [Workers](spatial-kit/workers/) · [Policy](spatial-kit/servers/) |
| spatial-native | [result](spatial-native/summary.json) | [manifest](spatial-native/run-manifest.json) | [stdout](spatial-native/upstream.stdout.log) |
| spatial-native-extra | [result](spatial-native-extra/summary.json) | [manifest](spatial-native-extra/run-manifest.json) | [stdout](spatial-native-extra/upstream.stdout.log.gz) |
