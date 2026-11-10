# CUDA MPS comparison

This experiment reuses the Kit's existing `--cuda-mps` implementation. It does not add a second MPS
daemon manager, change the model, or enable MPS by default. The driver controls two sequential arms
whose sole intended execution difference is MPS. The current LIBERO-Pro model-service and queue
timings motivate this experiment. This guide describes the ablation procedure; the selected draft
result is the [fixed-baseline comparison](FIXED_BASELINE_2026-11-10.md).

## Configuration

`run_mps_comparison.py` defaults to:

- `libero_goal_object`, 10 tasks x 50 episodes, fixed K=12/g=3 on GPU 0;
- MessagePack, per-request sockets, chunk camera rendering enabled, no videos;
- unchanged checkpoint, precision, attention implementation, eight-action prediction horizon and seeds;
- MPS off first, then on; one separate Quick Test warmup per arm;
- stage timing off in both arms; add `--stage-timing` for a separate diagnostic experiment.

Both arms read the same copied Policy contract and normalized simulator manifest. Relative runtime
paths become absolute in the copy; the built simulator installation is not modified. Both controller
and Policy environments must have the same new Kit version installed, and the Policy command should
point to this checkout's StarVLA example. The runner uses its own Python interpreter for the controller.
It neither installs packages nor downloads/replaces the checkpoint. Preserve and verify the pinned
model, configuration and asset inventories before and after real experiments.

The selected GPU must be idle and `nvidia-cuda-mps-control` must be on PATH. Existing MPS services or GPU
compute clients make the driver stop; it does not kill unrelated processes. These are preflight checks,
not a machine-wide reservation: do not start another GPU job during the pair. The underlying Kit owns
MPS startup, replica attachment and cleanup. Simulator Workers are not attached. The configured active
thread percentage remains 100%; do not simultaneously change it while testing MPS on/off.

## Preview and execute

Install `packages/roboscope-kit` in the controller and model environments first. Preview is read-only:

```bash
python benchmarks/libero_pro/run_mps_comparison.py \
  --benchmark-manifest /sim/libero-pro/benchmark-manifest.json \
  --policy-config /models/starvla-policy.json \
  --checkpoint-source /models/starvla-libero-pro \
  --output-dir /runs/libero-mps-pair-01 \
  --dry-run
```

Remove `--dry-run` to execute; the output directory must not exist. Use tmux for a long run.
`--concurrency 12 --replicas 3 --episodes 50 --gpu 0` makes the defaults explicit.
For a repeated pair, use a new output directory and `--on-first` to reverse the order.
For a diagnostic pair, add `--stage-timing`; warmup/formal records use different directories per arm.
Do not mix instrumented and uninstrumented timing in a speedup denominator.

## Outputs and acceptance

- `plan.json`: Kit version, fixed K/g, GPU, source Policy path, normalized manifest and exact commands.
- `benchmark-manifest.json`, `policy.json`: shared inputs for both arms.
- `mps-{off,on}/warmup` and `run`: original evaluator output; logs/status JSON retain subprocess exits
  and wall times. The formal command's wall time includes its load, preparation, MPS lifecycle and
  cleanup, but excludes the separate warmup and driver preflight/setup.
- `mps-{off,on}/summary.json`: formal timing, scores, executed steps, MPS attachment metadata, resource
  statistics and recovery events. The MPS-on arm must report at least g observed attached clients.
- `comparison.json`: formal episodes/s ratio, actual steps/s ratio, CLI wall ratio and paired outcome/
  step differences. Each arm must contain every expected task/episode exactly once and match its
  official aggregate. Recovery events remain visible; a successful comparison file alone is not a
  no-recovery or correctness-equivalence claim.

If a command fails, the driver retains the partial output and does not start the next arm. After a
completed command it checks that the GPU and MPS services have been released. An interruption or
abnormal process exit needs inspection before a new run; do not manually stop another user's MPS daemon.

Review per-task scores, exceptions, complete coverage and paired differences. Repeat the full suite
before treating a gain as stable. A single-suite MPS comparison does not satisfy issue #21's full-matrix
2x target. The native service and limitations are described in the
[NVIDIA MPS documentation](https://docs.nvidia.com/deploy/mps/).
