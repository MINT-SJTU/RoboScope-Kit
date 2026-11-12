# autotune evaluation records

Paths inside preserved records refer to the original experiment machine.

| Suite | Results | Episode records | Configuration | Logs / measurements |
| --- | --- | --- | --- | --- |
| libero_spatial | [result.json](libero_spatial/result.json) · [metrics.json](libero_spatial/metrics.json) | `record_success` in result.json | [autotune/selection.json](libero_spatial/autotune/selection.json) | — |
| libero_object | [result.json](libero_object/result.json) · [metrics.json](libero_object/metrics.json) | `record_success` in result.json | [autotune/selection.json](libero_object/autotune/selection.json) | — |
| libero_goal | [result.json](libero_goal/result.json) · [metrics.json](libero_goal/metrics.json) | `record_success` in result.json | [autotune/selection.json](libero_goal/autotune/selection.json) | — |
| libero_10 | — · [metrics.json](libero_10/metrics.json) | No formal result | [autotune/selection.json](libero_10/autotune/selection.json) | — |

- [Recorded experiment configuration](experiment.json)
- [Run status and complete command timings](status.json)
- [Latency and resource measurements](measurements/)
- [Recorded launcher](launch.py)

The completed formal evaluation covers Spatial, Object, and Goal (150 episodes).
Long selection and [retry records](retries/) are separate from those completed results.
