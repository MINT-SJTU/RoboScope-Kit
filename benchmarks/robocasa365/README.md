# RoboCasa365 runtime

This runtime is independent of `benchmarks/robocasa`, which owns GR-1 Tabletop.
The [protocol](../../packages/roboscope-kit/docs/robocasa365-protocol.md) freezes
the official 50-task multi-task workload in **pretrain scenes**, with 50 rollouts
per task (2,500 records), the v1.0.1 task horizons and native five-slot groups.
It is tracked in [issue #97](https://github.com/MINT-SJTU/RoboScope-Kit/issues/97).

## Build and run

Use Linux x86_64, Python 3.11, uv 0.11.14, Git, NVIDIA EGL libraries and a CUDA
GPU. Install the [official Diffusion Policy example](../../examples/robocasa365_diffusion/README.md)
in its separate inference environment. The simulator uses CPU torch for
imported playback utilities and excludes interactive device and training tools.
Allow space for six asset ZIPs (about 10.99 GB), their extracted files (about
147,000 files: check the filesystem's file-count quota), source checkouts,
environments and model weights.

```bash
python -m pip install ./packages/roboscope-kit
KIT_PYTHON="$(command -v python)" \
  ROBOSCOPE_ROBOCASA365_ASSETS_DIR=/data/robocasa365-downloads \
  bash benchmarks/robocasa365/build-runtime.sh /sim/robocasa365
```

`ROBOSCOPE_ROBOCASA365_PYTHON` can select an existing absolute Python 3.11
executable instead of downloading one; it must remain available after building.
`ROBOSCOPE_PYPI_INDEX` selects a reachable package mirror without changing pins.
`--environment-only` prepares imports without installing a manifest or claiming
a usable asset-backed runtime. Normal builds verify all asset hashes and finish
with the Kit self-test. Downloads use GET: Box may reject HEAD for valid links.

The builder applies three source patches, checking original, patch and result
hashes:

- **EGL cleanup.** Renderer resources are freed in their owning context, and
  delayed destruction restores the interrupted context. Without it, closing or
  hard-resetting one slot breaks rendering in another slot of the same process.
- **Counter regions in XML order** (`robocasa/models/fixtures/counter.py`).
  Upstream iterates a `set()` of regions, whose order follows object identity,
  so the same constructor seed places objects differently in two processes.
- **Seeded fixture stack bases** (`robocasa/models/fixtures/fixture_stack.py`).
  Upstream samples the stack base without the environment's generator.

The last two are reproducibility corrections: without them two native runs with
the same constructor seeds start from different scenes, and no per-episode
comparison is possible. Results are those of the official evaluator on the
corrected pinned simulator; tasks, seeds and success criteria are unchanged.

```bash
python -m roboscope_kit.runtime.benchmark \
  --benchmark robocasa365 \
  --benchmark-manifest /sim/robocasa365/benchmark-manifest.json \
  --self-test --output-dir /results/robocasa365-self-test

python -m roboscope_kit.runtime.benchmark \
  --benchmark robocasa365 --suite target50 \
  --benchmark-manifest /sim/robocasa365/benchmark-manifest.json \
  --policy-config /workspace/robocasa365-policy.json \
  --checkpoint-source /models/robocasa365-diffusion \
  --quick-test --output-dir /results/robocasa365-quick-test
```

The self-test verifies the frozen registry and resets and renders one
environment twice. The Quick Test runs one native five-slot group of the first
task at its official horizon. For the full matrix remove `--quick-test` and set
`--concurrency K --replicas G`; each Worker runs five simulator environments.

## Native comparison

RESULTS_PLACEHOLDER
