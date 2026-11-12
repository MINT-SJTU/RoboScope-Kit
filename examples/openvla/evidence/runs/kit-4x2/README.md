# kit-4x2 evaluation records

Paths inside preserved records refer to the original experiment machine.

| Suite | Results | Episode records | Configuration | Logs / measurements |
| --- | --- | --- | --- | --- |
| libero_spatial | [result.json](libero_spatial/result.json) · [metrics.json](libero_spatial/metrics.json) | `record_success` in result.json | [manifest.json](libero_spatial/manifest.json) | [Worker logs](libero_spatial/workers) · [Policy logs](libero_spatial/servers) |
| libero_object | [result.json](libero_object/result.json) · [metrics.json](libero_object/metrics.json) | `record_success` in result.json | [manifest.json](libero_object/manifest.json) | [Worker logs](libero_object/workers) · [Policy logs](libero_object/servers) |
| libero_goal | [result.json](libero_goal/result.json) · [metrics.json](libero_goal/metrics.json) | `record_success` in result.json | [manifest.json](libero_goal/manifest.json) | [Worker logs](libero_goal/workers) · [Policy logs](libero_goal/servers) |
| libero_10 | [result.json](libero_10/result.json) · [metrics.json](libero_10/metrics.json) | `record_success` in result.json | [manifest.json](libero_10/manifest.json) | [Worker logs](libero_10/workers) · [Policy logs](libero_10/servers) |

- [Recorded experiment configuration](experiment.json)
- [Run status and complete command timings](status.json)
- [Latency and resource measurements](measurements/)
- [Recorded launcher](launch.py)
