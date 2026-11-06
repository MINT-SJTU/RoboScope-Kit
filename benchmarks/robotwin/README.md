# RoboTwin benchmark

RoboTwin uses frozen task seeds from its benchmark manifest. The official evaluation runs 50 tasks × 100 episodes.

## Profiling

[Profiling repair and CUDA MPS measurements](profiling-mps.md) contains the configuration, matched
timing/resource/correctness tables, K/G exploration, bottleneck evidence, reproduction commands and attachment identities.

- Built-in profiling captures each Worker's first episode and each Policy Server's lifetime.
- On one 48-GiB RTX 4090, K8/G1 MPS changes repeated 500-episode throughput by −0.0293%; leave it off for that setup.
- Fixed-K follow-ups support retaining the default autotune search and budget.
- [Native RTX 4090 follow-up](profiling-native-mps.md) measures RT/SM/Tensor/VRAM counters and rendering
  synchronization; the earlier container's counter permission failure is recorded separately.

Use [build-runtime.sh](build-runtime.sh) to create or update the simulator environment and standalone bridge.
See [the common profiling guide](../../packages/roboscope-kit/docs/profiling.md) for CLI flags and coverage.
