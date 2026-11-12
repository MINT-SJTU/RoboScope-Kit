# Representative diagnostic images

These seven PNGs are unmodified snapshots from the historical 36-pair diagnostic
and its preprocessing probe. They are not from the later eight-pair CPU-alignment
check. Task, initial-state, and step indices are zero-based.

| Case | Images | What the records establish |
| --- | --- | --- |
| Spatial task 7 / state 0, step 16 | [Native CPU](spatial-7-0-step-16-native-cpu.png), [Kit CPU](spatial-7-0-step-16-kit-cpu.png), [Kit GPU](spatial-7-0-step-16-kit-gpu.png) | The preprocessing probe uses matching camera inputs. Kit CPU matches Native CPU; the historical preprocessing difference is one color component by one intensity level. See [probe results](../preprocess-probe/summary.json) and [coverage](../preprocess-probe/coverage.json). |
| Long task 9 / state 4, step 0 | [Native observation](long-9-4-step-0-native.png), [Kit observation](long-9-4-step-0-kit.png) | Initial-state arrays match but initial images differ. Both diagnostic rollouts fail. See the [paired table](../../tables/diagnostic-36.csv) and [analysis](../../tables/ANALYSIS.md). |
| Spatial task 7 / state 0, step 43 | [Native model input](spatial-7-0-step-43-native.png), [Kit model input](spatial-7-0-step-43-kit.png) | This is the first model-action difference larger than 1e-6 in this diagnostic pair. Native succeeds after 112 actions; Kit fails after 220. Earlier image differences are recorded in the [paired table](../../tables/diagnostic-36.csv). |

Small pixel differences need numerical comparison and may be invisible when
viewed normally. These snapshots illustrate recorded milestones; they do not
establish that a particular image difference caused the final outcome.

[Source paths and original hashes](../../reproduction/SOURCE_INDEX.json) identify
each retained PNG. Full step hashes, actions, outcomes, and diagnostic summaries
remain in the [diagnostic records](../README.md).
