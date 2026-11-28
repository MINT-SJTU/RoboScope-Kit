# Evo-DexBench DP-C / DP-T

This example serves Evo-DexBench's Diffusion Policy baselines (DP-C, CNN; DP-T, transformer) through
the RoboScope Kit lifecycle for the [`evo_dexbench`](../../benchmarks/evo_dexbench/README.md)
benchmark. Checkpoints load through the upstream example's own loader (`dp_policy.server`), and
inputs follow its session: two consecutive frames, RGB as CHW float in [0, 1], state as end-effector
pose plus hand joints per role, and the first `execute_steps` (8) actions of each predicted chunk.
Sampling is unseeded, as upstream; on a CUDA device the denoising loop replays as a CUDA graph
(see [CUDA-graph sampling](#cuda-graph-sampling)).

## Checkpoints

Each task group has its own checkpoints: Single1 (`m01`–`m05`), Single2 (`m06`–`m10`), Single3
(`m11`–`m15`), Double1 (`b01`–`b05`), Double2 (`b06`–`b10`) and Double3 (`b11`–`b15`). The group's
last task is its generalization task and uses the group's Gen checkpoint; the other four use Core.
Twelve checkpoints therefore cover all 30 tasks. They come from the Evo-DexBench authors.

Put them in one directory with a `roboscope-checkpoints.json` naming each file relative to it, as in
[`roboscope-checkpoints.example.json`](roboscope-checkpoints.example.json), and pass that directory
as `--checkpoint-source`. `load()` loads the checkpoints of the selected suite and rejects files of
the other variant. Double3 checkpoints were trained on the `interaction` camera instead of `context`;
the observation keys stored in each checkpoint select the cameras.

## Install

```bash
bash examples/evo_dexbench_dp/install-policy.sh /sim/evo-dexbench-dp
```

The installer builds a CPython 3.11 environment with torch 2.5.1 and torchvision 0.20.1 for CUDA 12.4
(`TORCH_INDEX_URL`), the upstream example's pins and the official Diffusion Policy checkout at the
commit the example names, fetched shallowly and checked unmodified. Upstream pins target Python
3.10: `numcodecs` is raised from 0.10.2 to 0.12.1, which has 3.11 wheels and only serves zarr
datasets, and `six`, which the checkpoints' Hydra workspace imports, is added. The installer ends by
importing the loader and that workspace.

Write a policy config with this environment's interpreter, as in
[`roboscope-policy.json`](roboscope-policy.json), and run the commands in the benchmark README.

## CUDA-graph sampling

On a CUDA device, `graph_sampler.py` replaces each checkpoint's `conditional_sample` with a replay of
the same 100-step DDPM loop, captured once per batch shape as a CUDA graph. It launches the same
kernels on the same inputs and draws its noise from the default CUDA generator at the same offsets,
so actions and the generator state after each call are bit-identical to eager sampling: 40 calls of
mixed batch sizes matched exactly for the Single1 and Double3 Core checkpoints. A batch of four takes
71 ms instead of 435 ms on an RTX 4090. Observation encoding stays eager. Set
`EVO_DEXBENCH_CUDA_GRAPHS=0` in the Policy environment to sample eagerly.

## Native comparison

The native path is upstream's evaluator with upstream's server. Install the upstream DP example
(`examples/policies/diffusion_policy` of the pinned source) editable into the simulator environment,
which registers the `dp_t` and `dp_c` client policies, then serve one checkpoint and evaluate its task:

```bash
/sim/evo-dexbench-dp/env/bin/python -m dp_policy.server --checkpoint <single1_core checkpoint> \
  --host 127.0.0.1 --port 8100 --device cuda:0
/sim/evo-dexbench/env/bin/python -m dex_benchmark evaluate --policy dp_t --task-id m02_v1 \
  --control-mode arm_pd_ee_delta_pose_hand_pd_joint_pos --obs-mode rgb \
  --policy-kwargs '{"host":"127.0.0.1","port":8100,"embodiment":"single","execute_steps":8}' \
  --log /results/native-m02.jsonl
```

`run_kit.py` runs the Kit CLI on the same task subset; all arguments pass through unchanged:

```bash
EVO_DEXBENCH_TASKS=m02_v1 python examples/evo_dexbench_dp/run_kit.py \
  --benchmark evo_dexbench --suite single_hand --episodes 100 --concurrency 1 --replicas 1 \
  --benchmark-manifest /sim/evo-dexbench/benchmark-manifest.json \
  --policy-config /workspace/evo-dexbench-policy.json \
  --checkpoint-source /models/evo-dexbench-dp-t --output-dir /results/kit-m02
```

One Worker runs the native four environments, so both paths do the same work. The 2026-11-24 result
is in the [protocol](../../packages/roboscope-kit/docs/evo-dexbench-protocol.md#validation).
