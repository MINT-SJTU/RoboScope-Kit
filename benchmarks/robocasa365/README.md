# RoboCasa365 runtime

This runtime is independent of `benchmarks/robocasa`, which owns GR1 Tabletop.
The [protocol](../../packages/roboscope-kit/docs/robocasa365-protocol.md) freezes
the 50-task multi-task workload in **pretrain scenes**, with 50 rollouts per
task (2,500 records), updated task horizons and native five-slot groups.

This is integration-first draft work for
[issue #97](https://github.com/MINT-SJTU/RoboScope-Kit/issues/97). Its complete
2,500-rollout matrix and matched native evaluator comparison remain outstanding.
[Issue #16](https://github.com/MINT-SJTU/RoboScope-Kit/issues/16) concerns the
separate GR1 runtime. Performance work is outside this integration.

## Build and run

Use Linux x86_64, Python 3.11, uv 0.11.14, Git, NVIDIA EGL libraries and a CUDA
GPU. Install the [official Diffusion Policy example](../../examples/robocasa365_diffusion/README.md)
in its separate inference environment. The simulator uses CPU torch for
imported playback utilities and excludes interactive device and training tools.
Allow space for six asset ZIPs (about 10.99 GB), their extracted files,
source checkouts, environments and model weights. Also check the filesystem's
**file count quota** before extraction.

```bash
python -m pip install ./packages/roboscope-kit
KIT_PYTHON="$(command -v python)" \
  ROBOSCOPE_ROBOCASA365_ASSETS_DIR=/autodl-fs/data/robocasa365/downloads \
  bash benchmarks/robocasa365/build-runtime.sh /autodl-fs/data/robocasa365/runtime
```

`ROBOSCOPE_ROBOCASA365_PYTHON` can select an existing absolute Python 3.11
executable instead of downloading one; it must remain available after building.
`ROBOSCOPE_PYPI_INDEX` selects a reachable package mirror without changing pins.
`--environment-only` prepares imports without installing a manifest or claiming
a usable asset-backed runtime. Normal builds verify all asset hashes and finish
with the Kit self-test. Downloads use GET: Box may reject HEAD for valid links.
An existing verified ZIP cache can be transferred from another machine.

The build includes a reviewed EGL cleanup correction: resources are freed in
their owning context, and delayed destruction restores the interrupted context,
API and read/draw surfaces. Original, patch and resulting-file hashes are
checked. Self-test and workers reject runtimes lacking the marker-2 correction.
Seeds, reset rules and observation frequency remain native; every physical step
generates its observations.

Configure the example with absolute interpreter and source paths, and a model
directory containing `model.ckpt` and `clip/`. In an adaptation container the
configuration can be `/workspace/evostudio-policy.json`.

```bash
python -m roboscope_kit.runtime.benchmark \
  --benchmark robocasa365 \
  --benchmark-manifest /autodl-fs/data/robocasa365/runtime/benchmark-manifest.json \
  --self-test --output-dir /results/robocasa365-self-test

python -m roboscope_kit.runtime.benchmark \
  --benchmark robocasa365 --suite target50 \
  --benchmark-manifest /autodl-fs/data/robocasa365/runtime/benchmark-manifest.json \
  --policy-config /workspace/evostudio-policy.json \
  --checkpoint-source /autodl-fs/data/robocasa365/models/official-dp \
  --quick-test --output-dir /results/robocasa365-quick-test
```

The self-test verifies imports and the frozen registry, without creating an
environment. The real Quick Test runs one complete episode of the first task,
retaining its official horizon. It uses one slot and is a smoke test.
For the full matrix remove `--quick-test` and set `--concurrency K --replicas G`.
Each formal worker manages **five** native environments. Use fresh output
directories, verify all records and compare with the pinned native evaluator.
The Policy README provides a captured-input native action comparison helper.

## Measurement and acceptance limits

Worker timers cover construction, resets, native steps/observations, Policy
round trips, video writes and cleanup. Shared five-slot intervals and independent
worker times must not be summed as wall time. `--profile` captures only the first
native group's rollout after environment resets, excluding process/model/
environment startup and the independent Policy Server's GPU execution. Nsight
hardware capture and isolated server GPU timing remain unverified.

This integration claims no speedup, unchanged success rate, full-suite result or
full native equivalence. A self-test, smoke group or exact first action chunk is
insufficient. Keep model/task/scenario streams, horizons, native slot count, K/G
and outputs fixed in later comparisons.

## Validation record

[2026-11-07 integration validation](validation-20261107.md) records the fresh builds,
real one-slot and five-slot smoke runs, exact native first-batch comparison,
the EGL erratum, versioned evidence and remaining full-suite acceptance work.
The fresh b35 build and marker-2 self-test passed; its real TurnOnElectricKettle
group completed five unique 450-step records with 2/5 task successes and no
worker errors. This is integration smoke evidence only. Earlier corrected b33
results are prior evidence, not a matched comparison or proof of this branch.
