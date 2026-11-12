# RoboCasa365 runtime

This runtime is independent of `benchmarks/robocasa`, which owns GR1 Tabletop.
The [protocol](../../packages/roboscope-kit/docs/robocasa365-protocol.md) freezes
the 50-task multi-task workload in **pretrain scenes**, with 50 rollouts per
task (2,500 records), updated task horizons and native five-slot groups.

This integration contributes to
[issue #97](https://github.com/MINT-SJTU/RoboScope-Kit/issues/97). The
[current-source native comparison](approved250-native-20261109.md) runs all 50
tasks, five episodes each, twice natively and with serial Kit, at the approved
shared 128-step cap. Every path completes 250 unique records, scores 3/250 and
executes 32,000 steps, with zero paired outcome or terminal-step differences.
The formal default remains 2,500; capped agreement does not establish accuracy
at the official task horizons or complete trajectory equality. Exact-head CI remains a
separate acceptance gate. The [historical build/review record](validation-20261107.md),
[earlier all-task collection](native-comparison-20261109.md) and
[focused repeatability controls](correctness-followup-20261109.md) retain their
original sources. Performance is delivered separately in dependent PR #107.
[Issue #16](https://github.com/MINT-SJTU/RoboScope-Kit/issues/16) tracks GR1
Tabletop and supplies measurement guidance; issue #97 owns RoboCasa365.

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

The builder also pins two upstream sampling corrections for paired evaluation:
counter regions retain XML insertion order instead of object-identity set order,
and fixture stack bases receive the environment's seeded generator. Uncorrected
native runs can produce different initial scenes for the same constructor seeds.
The corrected runtime therefore changes those formerly nondeterministic scene
realizations. Run the pinned official evaluator against the same corrected runtime;
do not describe this as an unmodified-upstream comparison or mix its evidence with
earlier runs lacking those corrections. These are reproducibility corrections,
not performance changes.

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

The self-test verifies the frozen registry and validates RGB and state across two
environment resets. The real Quick Test runs one native five-slot group of the
first task with the official horizon.
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

The approved all-task comparison reports exact paired success and terminal-step
agreement at its common 128-step cap. It makes no speed claim or full-horizon
accuracy/trajectory claim. Keep model/task/scenario streams, horizons, native slot
count, K/G and outputs fixed in later comparisons.

## Validation records

The [2026-11-07 startup record and 2026-11-09 review update](validation-20261107.md)
attribute build, self-test, real five-slot Quick Test and 298 passing pytest checks
to their tested revisions. The [approved all-task comparison](approved250-native-20261109.md)
adds two fresh native passes and current-source serial Kit results. Historical
results remain in the [earlier native comparison](native-comparison-20261109.md).
Equal capped outcomes do not establish full-horizon success rates or trajectory parity.
