# robomimic runtime

Build the pinned Linux runtime with `build-runtime.sh <install-dir>`. The builder verifies the source
commits, package versions, patch hashes, and all 250 initial-state digests, then runs the Kit self-test
when `KIT_PYTHON` identifies a Python with roboscope-kit installed. Models are configured through
[`examples/robomimic`](../../examples/robomimic/README.md).

The model-cache patch is specific to the pinned robosuite/MuJoCo runtime and immutable assets. Workers
enable it through `ROBOSCOPE_ROBOMIMIC_MODEL_CACHE=1`; set this variable to `0` to disable it. It caches two
compiled XML models per process and copies each before exposing mutable arrays to a simulation. MuJoCo's
last-compiled XML context is restored when required by `mj_saveLastXML`. It does not skip reset, forward,
physics, control, or observation work. All simulator changes live in the builder and its pinned patch.

See the [evaluation protocol](../../packages/roboscope-kit/ROBOMIMIC_EVALUATION_PROTOCOL.md) for the frozen
scope and the [throughput report](THROUGHPUT.md) for measured configurations, comparisons, and evidence.

## Profiling

On 2026-11-08, the real PH low-dim BC-RNN Policy was profiled on an Ubuntu 22.04 container with a 20-core
CPU quota, 96 GiB memory quota, and one RTX 4090 reporting 48 GiB, using candidate Kit 0.8.0b34 based on
`9174d1b`. The diagnostic used K=5/G=1, two episodes per task, Kit `--profile`, and cProfile reports for
the Worker and the separate Policy process. It overlapped a native diagnostic, so its timings identify
code paths only and are excluded from every speedup calculation.

The aggregated Worker profile recorded 37 XML compilations, 192,580 MuJoCo forward calls, and substantial
Policy socket waits. The separate Policy profile showed time in the upstream recurrent forward pass and
GMM sampling. The complete unprofiled evaluations then isolated parallelism, task ordering, model caching,
and transport. Rendering and image preprocessing are absent from this low-dimensional profile. Skipping
physics forward work changed per-episode results and was rejected. A two-entry compiled-model cache
retained all terminal outcomes and initial-state digests; see the report for the cache-off comparison.
