# Evo-DexBench runtime

The [protocol](../../packages/roboscope-kit/docs/evo-dexbench-protocol.md) freezes Evo-DexBench's
standard `evaluation_v1` workload: 30 dexterous-hand tasks in two suites, 100 episodes per task.
It is tracked in [issue #140](https://github.com/MINT-SJTU/RoboScope-Kit/issues/140).

## Build

Use Linux x86_64, uv 0.11.14, curl, tar and an NVIDIA GPU whose driver provides Vulkan. Physics runs
on the CPU, but ManiSkill hands rendered camera images to CUDA torch, so the environment installs
torch 2.5.1 for CUDA 12.4 (`TORCH_INDEX_URL` selects another index with the same build). Allow about
7 GB for the environment and 2 GB for the assets, about 14,000 files: check the filesystem's
file-count quota.

PartNet-Mobility, which 49 task objects come from, is gated on Hugging Face. Give the builder either
your own download or a token of an account that accepted its terms; the rest of the asset bundle is
public.

```bash
python -m pip install ./packages/roboscope-kit
EVO_DEXBENCH_PARTNET_DIR=/data/partnet-mobility \
  bash benchmarks/evo_dexbench/build-runtime.sh /sim/evo-dexbench
# or: HF_TOKEN=... bash benchmarks/evo_dexbench/build-runtime.sh /sim/evo-dexbench
```

`EVO_DEXBENCH_ASSET_DIR` uses an existing, complete upstream asset root instead of downloading;
`EVO_DEXBENCH_SOURCE_ARCHIVE` an already downloaded source tarball. The build checks the source
checksum, installs the source editable with its revision marker, verifies the assets with upstream's
`assets verify` and writes `/sim/evo-dexbench/benchmark-manifest.json`.

Containers often default to 1,024 open files. A PartNet-heavy scene opens more meshes than that while
loading, so Workers raise their soft limit to the hard limit; raise the hard limit if it is lower.

## Run

```bash
python -m roboscope_kit.runtime.benchmark --benchmark evo_dexbench \
  --benchmark-manifest /sim/evo-dexbench/benchmark-manifest.json \
  --self-test --output-dir /results/evo-dexbench-self-test

python -m roboscope_kit.runtime.benchmark --benchmark evo_dexbench --suite single_hand \
  --benchmark-manifest /sim/evo-dexbench/benchmark-manifest.json \
  --policy-config /workspace/evo-dexbench-policy.json \
  --checkpoint-source /models/evo-dexbench-dp-t \
  --quick-test --output-dir /results/evo-dexbench-quick-test
```

The self-test checks all 30 schedules against `evaluation_v1` and resets and renders one environment.
The Quick Test runs the first wave (four episodes) of the suite's first task. For a full suite remove
`--quick-test` and set `--concurrency K --replicas G`; each Worker runs one wave of four simulator
environments at a time. The [DP-C / DP-T example](../../examples/evo_dexbench_dp/README.md) serves
the upstream Diffusion Policy checkpoints and describes the native comparison.
