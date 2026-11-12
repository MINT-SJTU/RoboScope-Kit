# Experiment environment and recorded commands

These commands describe the completed experiments, preserving their paths,
environment variables, and case selections. The full evaluations run 50 episodes
per suite on each path. The diagnostic selects 36 task/state identities through
`DIAGNOSTIC_SELECTION`, with one rollout per path for each identity. Paths in the
commands refer to the experiment machine.

## Hardware and runtime

| Experiment | GPU count and model | Driver | CPU |
| --- | --- | --- | --- |
| Historical Native 200 | 1 × NVIDIA GeForce RTX 4090, reported memory 49140 MiB | 595.71.05 | Not recorded |
| Historical Kit 2/2 200 | 1 × NVIDIA GeForce RTX 4090, reported memory 49140 MiB | 580.105.08 | Not recorded |
| Historical Kit 1/1 200 | 1 × NVIDIA GeForce RTX 4090, reported memory 49140 MiB | 595.71.05 | Not recorded |
| Historical Kit 4/1 200 | 1 × NVIDIA GeForce RTX 4090, reported memory 49140 MiB | 595.71.05 | Not recorded |
| Historical Kit 4/2 200 | 1 × NVIDIA GeForce RTX 4090, reported memory 49140 MiB | 580.105.08 | Not recorded |
| Autotune | 1 × NVIDIA GeForce RTX 4090, reported memory 49140 MiB | 595.71.05 | Not recorded |
| Diagnostic 36 pairs | 1 × NVIDIA GeForce RTX 4090, reported memory 49140 MiB | 595.71.05 | Intel Xeon Platinum 8470Q |

Native uses Python 3.10.13 and Kit uses Python 3.11.15. Both simulator environments
use LIBERO revision `8f1084e3132a39270c3a13ebe37270a43ece2a01`, robosuite 1.4.1,
MuJoCo 3.2.3, and NumPy 1.24.4. Inference dependencies are torch 2.2.0,
Transformers 4.40.1, TensorFlow 2.15.0, and FlashAttention 2.5.5.
The Native evaluator revision is `c8f03f48af692657d3060c19588038c7220e9af9`;
the model-code revision is `47a0ec7fc4ec123775a391911046cf33cf9ed83f`.

The paired comparison and diagnostic use Kit with two Workers and two Policy
replicas on one GPU. Other full evaluations use the Worker/replica counts in
their run names and commands. Native uses the `matched-serial` profile. Diagnostic
commands include explicit environment variables and the selected task/state identities.

## Full evaluations: 200 episodes per path

### openvla-kit-200-fixed2g2-seed7-20261110

```bash
/root/autodl-tmp/openvla-kit-20261109/runtime/env/bin/python -m roboscope_kit.runtime.benchmark --benchmark libero --suite libero_10 --checkpoint-source /root/autodl-fs/Kit-Models/openvla/10 --benchmark-manifest /root/autodl-tmp/openvla-kit-200-fixed2g2-seed7-20261110/benchmark-manifest.json --policy-config /root/autodl-tmp/openvla-kit-200-fixed2g2-seed7-20261110/roboscope-policy.json --episodes 5 --concurrency 2 --replicas 2 --output-dir /root/autodl-tmp/openvla-kit-200-fixed2g2-seed7-20261110/libero_10
```

```bash
/root/autodl-tmp/openvla-kit-20261109/runtime/env/bin/python -m roboscope_kit.runtime.benchmark --benchmark libero --suite libero_goal --checkpoint-source /root/autodl-fs/Kit-Models/openvla/goal --benchmark-manifest /root/autodl-tmp/openvla-kit-200-fixed2g2-seed7-20261110/benchmark-manifest.json --policy-config /root/autodl-tmp/openvla-kit-200-fixed2g2-seed7-20261110/roboscope-policy.json --episodes 5 --concurrency 2 --replicas 2 --output-dir /root/autodl-tmp/openvla-kit-200-fixed2g2-seed7-20261110/libero_goal
```

```bash
/root/autodl-tmp/openvla-kit-20261109/runtime/env/bin/python -m roboscope_kit.runtime.benchmark --benchmark libero --suite libero_object --checkpoint-source /root/autodl-fs/Kit-Models/openvla/object --benchmark-manifest /root/autodl-tmp/openvla-kit-200-fixed2g2-seed7-20261110/benchmark-manifest.json --policy-config /root/autodl-tmp/openvla-kit-200-fixed2g2-seed7-20261110/roboscope-policy.json --episodes 5 --concurrency 2 --replicas 2 --output-dir /root/autodl-tmp/openvla-kit-200-fixed2g2-seed7-20261110/libero_object
```

```bash
/root/autodl-tmp/openvla-kit-20261109/runtime/env/bin/python -m roboscope_kit.runtime.benchmark --benchmark libero --suite libero_spatial --checkpoint-source /root/autodl-fs/Kit-Models/openvla/spatial --benchmark-manifest /root/autodl-tmp/openvla-kit-200-fixed2g2-seed7-20261110/benchmark-manifest.json --policy-config /root/autodl-tmp/openvla-kit-200-fixed2g2-seed7-20261110/roboscope-policy.json --episodes 5 --concurrency 2 --replicas 2 --output-dir /root/autodl-tmp/openvla-kit-200-fixed2g2-seed7-20261110/libero_spatial
```


### openvla-native-200-seed7-20261109

```bash
/root/autodl-tmp/openvla-native-runtime/env/bin/python /root/autodl-tmp/openvla-native-200-seed7-20261109/source/run.py --runtime /root/autodl-tmp/openvla-native-runtime --checkpoint /root/autodl-fs/Kit-Models/openvla/10 --checkpoint-inventory /root/autodl-tmp/openvla-libero-quick-20261109/inventories/10.json --suite libero_10 --output-dir /root/autodl-tmp/openvla-native-200-seed7-20261109/libero_10 --episodes-per-task 5 --profile-name matched-serial --seed 7 --env-seed 7 --video off --repeat-id native-200-seed7 --gpu 0 --cpu-threads 1 --sample-interval 0.5
```

```bash
/root/autodl-tmp/openvla-native-runtime/env/bin/python /root/autodl-tmp/openvla-native-200-seed7-20261109/source/run.py --runtime /root/autodl-tmp/openvla-native-runtime --checkpoint /root/autodl-fs/Kit-Models/openvla/goal --checkpoint-inventory /root/autodl-tmp/openvla-libero-quick-20261109/inventories/goal.json --suite libero_goal --output-dir /root/autodl-tmp/openvla-native-200-seed7-20261109/libero_goal --episodes-per-task 5 --profile-name matched-serial --seed 7 --env-seed 7 --video off --repeat-id native-200-seed7 --gpu 0 --cpu-threads 1 --sample-interval 0.5
```

```bash
/root/autodl-tmp/openvla-native-runtime/env/bin/python /root/autodl-tmp/openvla-native-200-seed7-20261109/source/run.py --runtime /root/autodl-tmp/openvla-native-runtime --checkpoint /root/autodl-fs/Kit-Models/openvla/object --checkpoint-inventory /root/autodl-tmp/openvla-libero-quick-20261109/inventories/object.json --suite libero_object --output-dir /root/autodl-tmp/openvla-native-200-seed7-20261109/libero_object --episodes-per-task 5 --profile-name matched-serial --seed 7 --env-seed 7 --video off --repeat-id native-200-seed7 --gpu 0 --cpu-threads 1 --sample-interval 0.5
```

```bash
/root/autodl-tmp/openvla-native-runtime/env/bin/python /root/autodl-tmp/openvla-native-200-seed7-20261109/source/run.py --runtime /root/autodl-tmp/openvla-native-runtime --checkpoint /root/autodl-fs/Kit-Models/openvla/spatial --checkpoint-inventory /root/autodl-tmp/openvla-libero-quick-20261109/inventories/spatial.json --suite libero_spatial --output-dir /root/autodl-tmp/openvla-native-200-seed7-20261109/libero_spatial --episodes-per-task 5 --profile-name matched-serial --seed 7 --env-seed 7 --video off --repeat-id native-200-seed7 --gpu 0 --cpu-threads 1 --sample-interval 0.5
```

## Diagnostic subset: 36 pairs, 72 rollouts

### spatial-native

```bash
env HF_HOME=/root/autodl-tmp/openvla-libero-quick-20261109/hf OPENVLA_SEED=7 PYTHONHASHSEED=7 CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false 'DIAGNOSTIC_SELECTION={"7": [0], "1": [3]}' OPENVLA_TRACE_ROOT=/root/autodl-tmp/openvla-paired-diagnostics-20261111/spatial-native/policy-trace /root/autodl-tmp/openvla-native-runtime/env/bin/python /root/autodl-tmp/openvla-paired-diagnostics-20261111/source/run.py --runtime /root/autodl-tmp/openvla-native-runtime --checkpoint /root/autodl-fs/Kit-Models/openvla/spatial --checkpoint-inventory /root/autodl-tmp/openvla-libero-quick-20261109/inventories/spatial.json --suite libero_spatial --output-dir /root/autodl-tmp/openvla-paired-diagnostics-20261111/spatial-native --episodes-per-task 5 --seed 7 --env-seed 7 --video off --profile-name matched-serial --diagnostic-trace --task-id 7 --task-id 1
```

### object-native

```bash
env HF_HOME=/root/autodl-tmp/openvla-libero-quick-20261109/hf OPENVLA_SEED=7 PYTHONHASHSEED=7 CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false 'DIAGNOSTIC_SELECTION={"2": [4], "3": [0]}' OPENVLA_TRACE_ROOT=/root/autodl-tmp/openvla-paired-diagnostics-20261111/object-native/policy-trace /root/autodl-tmp/openvla-native-runtime/env/bin/python /root/autodl-tmp/openvla-paired-diagnostics-20261111/source/run.py --runtime /root/autodl-tmp/openvla-native-runtime --checkpoint /root/autodl-fs/Kit-Models/openvla/object --checkpoint-inventory /root/autodl-tmp/openvla-libero-quick-20261109/inventories/object.json --suite libero_object --output-dir /root/autodl-tmp/openvla-paired-diagnostics-20261111/object-native --episodes-per-task 5 --seed 7 --env-seed 7 --video off --profile-name matched-serial --diagnostic-trace --task-id 2 --task-id 3
```

### spatial-native-extra

```bash
env HF_HOME=/root/autodl-tmp/openvla-libero-quick-20261109/hf OPENVLA_SEED=7 PYTHONHASHSEED=7 CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false 'DIAGNOSTIC_SELECTION={"3": [2], "4": [3], "5": [1], "6": [3], "8": [1, 2, 3]}' OPENVLA_TRACE_ROOT=/root/autodl-tmp/openvla-paired-diagnostics-20261111/spatial-native-extra/policy-trace /root/autodl-tmp/openvla-native-runtime/env/bin/python /root/autodl-tmp/openvla-paired-diagnostics-20261111/source/run.py --runtime /root/autodl-tmp/openvla-native-runtime --checkpoint /root/autodl-tmp/openvla-diagnostic-model-cache-20261111/spatial --checkpoint-inventory /root/autodl-tmp/openvla-libero-quick-20261109/inventories/spatial.json --suite libero_spatial --output-dir /root/autodl-tmp/openvla-paired-diagnostics-20261111/spatial-native-extra --episodes-per-task 5 --seed 7 --env-seed 7 --video off --profile-name matched-serial --diagnostic-trace --task-id 3 --task-id 4 --task-id 5 --task-id 6 --task-id 8
```

### object-native-extra

```bash
env HF_HOME=/root/autodl-tmp/openvla-libero-quick-20261109/hf OPENVLA_SEED=7 PYTHONHASHSEED=7 CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false 'DIAGNOSTIC_SELECTION={"3": [2], "6": [0, 4], "8": [4]}' OPENVLA_TRACE_ROOT=/root/autodl-tmp/openvla-paired-diagnostics-20261111/object-native-extra/policy-trace /root/autodl-tmp/openvla-native-runtime/env/bin/python /root/autodl-tmp/openvla-paired-diagnostics-20261111/source/run.py --runtime /root/autodl-tmp/openvla-native-runtime --checkpoint /root/autodl-tmp/openvla-diagnostic-model-cache-20261111/object --checkpoint-inventory /root/autodl-tmp/openvla-libero-quick-20261109/inventories/object.json --suite libero_object --output-dir /root/autodl-tmp/openvla-paired-diagnostics-20261111/object-native-extra --episodes-per-task 5 --seed 7 --env-seed 7 --video off --profile-name matched-serial --diagnostic-trace --task-id 3 --task-id 6 --task-id 8
```

### goal-native

```bash
env HF_HOME=/root/autodl-tmp/openvla-libero-quick-20261109/hf OPENVLA_SEED=7 PYTHONHASHSEED=7 CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false 'DIAGNOSTIC_SELECTION={"0": [2, 3, 4], "4": [1], "5": [4], "8": [4], "9": [2, 3]}' OPENVLA_TRACE_ROOT=/root/autodl-tmp/openvla-paired-diagnostics-20261111/goal-native/policy-trace /root/autodl-tmp/openvla-native-runtime/env/bin/python /root/autodl-tmp/openvla-paired-diagnostics-20261111/source/run.py --runtime /root/autodl-tmp/openvla-native-runtime --checkpoint /root/autodl-tmp/openvla-diagnostic-model-cache-20261111/goal --checkpoint-inventory /root/autodl-tmp/openvla-libero-quick-20261109/inventories/goal.json --suite libero_goal --output-dir /root/autodl-tmp/openvla-paired-diagnostics-20261111/goal-native --episodes-per-task 5 --seed 7 --env-seed 7 --video off --profile-name matched-serial --diagnostic-trace --task-id 0 --task-id 4 --task-id 5 --task-id 8 --task-id 9
```

### 10-native

```bash
env HF_HOME=/root/autodl-tmp/openvla-libero-quick-20261109/hf OPENVLA_SEED=7 PYTHONHASHSEED=7 CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false 'DIAGNOSTIC_SELECTION={"0": [2], "1": [1, 2], "2": [0, 1, 2], "4": [3], "6": [1], "7": [0, 1], "8": [1], "9": [3, 4]}' OPENVLA_TRACE_ROOT=/root/autodl-tmp/openvla-paired-diagnostics-20261111/10-native/policy-trace /root/autodl-tmp/openvla-native-runtime/env/bin/python /root/autodl-tmp/openvla-paired-diagnostics-20261111/source/run.py --runtime /root/autodl-tmp/openvla-native-runtime --checkpoint /root/autodl-tmp/openvla-diagnostic-model-cache-20261111/10 --checkpoint-inventory /root/autodl-tmp/openvla-libero-quick-20261109/inventories/10.json --suite libero_10 --output-dir /root/autodl-tmp/openvla-paired-diagnostics-20261111/10-native --episodes-per-task 5 --seed 7 --env-seed 7 --video off --profile-name matched-serial --diagnostic-trace --task-id 0 --task-id 1 --task-id 2 --task-id 4 --task-id 6 --task-id 7 --task-id 8 --task-id 9
```

### spatial-kit

```bash
env HF_HOME=/root/autodl-tmp/openvla-libero-quick-20261109/hf OPENVLA_SEED=7 PYTHONHASHSEED=7 CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false 'DIAGNOSTIC_SELECTION={"1": [3], "3": [2], "4": [3], "5": [1], "6": [3], "7": [0], "8": [1, 2, 3]}' OPENVLA_TRACE_ROOT=/root/autodl-tmp/openvla-paired-diagnostics-20261111/spatial-kit/policy-trace /root/autodl-tmp/openvla-kit-20261109/runtime/env/bin/python /root/autodl-tmp/openvla-paired-diagnostics-20261111/source/kit_entry.py --benchmark libero --suite libero_spatial --checkpoint-source /root/autodl-tmp/openvla-diagnostic-model-cache-20261111/spatial --benchmark-manifest /root/autodl-tmp/openvla-paired-diagnostics-20261111/kit-manifest.json --policy-config /root/autodl-tmp/openvla-paired-diagnostics-20261111/policy.json --episodes 5 --concurrency 2 --replicas 2 --output-dir /root/autodl-tmp/openvla-paired-diagnostics-20261111/spatial-kit
```

### object-kit

```bash
env HF_HOME=/root/autodl-tmp/openvla-libero-quick-20261109/hf OPENVLA_SEED=7 PYTHONHASHSEED=7 CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false 'DIAGNOSTIC_SELECTION={"2": [4], "3": [0, 2], "6": [0, 4], "8": [4]}' OPENVLA_TRACE_ROOT=/root/autodl-tmp/openvla-paired-diagnostics-20261111/object-kit/policy-trace /root/autodl-tmp/openvla-kit-20261109/runtime/env/bin/python /root/autodl-tmp/openvla-paired-diagnostics-20261111/source/kit_entry.py --benchmark libero --suite libero_object --checkpoint-source /root/autodl-tmp/openvla-diagnostic-model-cache-20261111/object --benchmark-manifest /root/autodl-tmp/openvla-paired-diagnostics-20261111/kit-manifest.json --policy-config /root/autodl-tmp/openvla-paired-diagnostics-20261111/policy.json --episodes 5 --concurrency 2 --replicas 2 --output-dir /root/autodl-tmp/openvla-paired-diagnostics-20261111/object-kit
```

### goal-kit

```bash
env HF_HOME=/root/autodl-tmp/openvla-libero-quick-20261109/hf OPENVLA_SEED=7 PYTHONHASHSEED=7 CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false 'DIAGNOSTIC_SELECTION={"0": [2, 3, 4], "4": [1], "5": [4], "8": [4], "9": [2, 3]}' OPENVLA_TRACE_ROOT=/root/autodl-tmp/openvla-paired-diagnostics-20261111/goal-kit/policy-trace /root/autodl-tmp/openvla-kit-20261109/runtime/env/bin/python /root/autodl-tmp/openvla-paired-diagnostics-20261111/source/kit_entry.py --benchmark libero --suite libero_goal --checkpoint-source /root/autodl-tmp/openvla-diagnostic-model-cache-20261111/goal --benchmark-manifest /root/autodl-tmp/openvla-paired-diagnostics-20261111/kit-manifest.json --policy-config /root/autodl-tmp/openvla-paired-diagnostics-20261111/policy.json --episodes 5 --concurrency 2 --replicas 2 --output-dir /root/autodl-tmp/openvla-paired-diagnostics-20261111/goal-kit
```

### 10-kit

```bash
env HF_HOME=/root/autodl-tmp/openvla-libero-quick-20261109/hf OPENVLA_SEED=7 PYTHONHASHSEED=7 CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false 'DIAGNOSTIC_SELECTION={"0": [2], "1": [1, 2], "2": [0, 1, 2], "4": [3], "6": [1], "7": [0, 1], "8": [1], "9": [3, 4]}' OPENVLA_TRACE_ROOT=/root/autodl-tmp/openvla-paired-diagnostics-20261111/10-kit/policy-trace /root/autodl-tmp/openvla-kit-20261109/runtime/env/bin/python /root/autodl-tmp/openvla-paired-diagnostics-20261111/source/kit_entry.py --benchmark libero --suite libero_10 --checkpoint-source /root/autodl-tmp/openvla-diagnostic-model-cache-20261111/10 --benchmark-manifest /root/autodl-tmp/openvla-paired-diagnostics-20261111/kit-manifest.json --policy-config /root/autodl-tmp/openvla-paired-diagnostics-20261111/policy.json --episodes 5 --concurrency 2 --replicas 2 --output-dir /root/autodl-tmp/openvla-paired-diagnostics-20261111/10-kit
```

## Other full evaluations in the report

### openvla-kit-200-fixed4-seed7-20261109

```bash
/root/autodl-tmp/openvla-kit-20261109/runtime/env/bin/python -m roboscope_kit.runtime.benchmark --benchmark libero --suite libero_10 --checkpoint-source /root/autodl-fs/Kit-Models/openvla/10 --benchmark-manifest /root/autodl-tmp/openvla-kit-200-fixed4-seed7-20261109/benchmark-manifest.json --policy-config /root/autodl-tmp/openvla-kit-200-fixed4-seed7-20261109/roboscope-policy.json --episodes 5 --concurrency 4 --replicas 1 --output-dir /root/autodl-tmp/openvla-kit-200-fixed4-seed7-20261109/libero_10
```

```bash
/root/autodl-tmp/openvla-kit-20261109/runtime/env/bin/python -m roboscope_kit.runtime.benchmark --benchmark libero --suite libero_goal --checkpoint-source /root/autodl-fs/Kit-Models/openvla/goal --benchmark-manifest /root/autodl-tmp/openvla-kit-200-fixed4-seed7-20261109/benchmark-manifest.json --policy-config /root/autodl-tmp/openvla-kit-200-fixed4-seed7-20261109/roboscope-policy.json --episodes 5 --concurrency 4 --replicas 1 --output-dir /root/autodl-tmp/openvla-kit-200-fixed4-seed7-20261109/libero_goal
```

```bash
/root/autodl-tmp/openvla-kit-20261109/runtime/env/bin/python -m roboscope_kit.runtime.benchmark --benchmark libero --suite libero_object --checkpoint-source /root/autodl-fs/Kit-Models/openvla/object --benchmark-manifest /root/autodl-tmp/openvla-kit-200-fixed4-seed7-20261109/benchmark-manifest.json --policy-config /root/autodl-tmp/openvla-kit-200-fixed4-seed7-20261109/roboscope-policy.json --episodes 5 --concurrency 4 --replicas 1 --output-dir /root/autodl-tmp/openvla-kit-200-fixed4-seed7-20261109/libero_object
```

```bash
/root/autodl-tmp/openvla-kit-20261109/runtime/env/bin/python -m roboscope_kit.runtime.benchmark --benchmark libero --suite libero_spatial --checkpoint-source /root/autodl-fs/Kit-Models/openvla/spatial --benchmark-manifest /root/autodl-tmp/openvla-kit-200-fixed4-seed7-20261109/benchmark-manifest.json --policy-config /root/autodl-tmp/openvla-kit-200-fixed4-seed7-20261109/roboscope-policy.json --episodes 5 --concurrency 4 --replicas 1 --output-dir /root/autodl-tmp/openvla-kit-200-fixed4-seed7-20261109/libero_spatial
```

### openvla-kit-200-fixed4g2-seed7-20261110

```bash
/root/autodl-tmp/openvla-kit-20261109/runtime/env/bin/python -m roboscope_kit.runtime.benchmark --benchmark libero --suite libero_10 --checkpoint-source /root/autodl-fs/Kit-Models/openvla/10 --benchmark-manifest /root/autodl-tmp/openvla-kit-200-fixed4g2-seed7-20261110/benchmark-manifest.json --policy-config /root/autodl-tmp/openvla-kit-200-fixed4g2-seed7-20261110/roboscope-policy.json --episodes 5 --concurrency 4 --replicas 2 --output-dir /root/autodl-tmp/openvla-kit-200-fixed4g2-seed7-20261110/libero_10
```

```bash
/root/autodl-tmp/openvla-kit-20261109/runtime/env/bin/python -m roboscope_kit.runtime.benchmark --benchmark libero --suite libero_goal --checkpoint-source /root/autodl-fs/Kit-Models/openvla/goal --benchmark-manifest /root/autodl-tmp/openvla-kit-200-fixed4g2-seed7-20261110/benchmark-manifest.json --policy-config /root/autodl-tmp/openvla-kit-200-fixed4g2-seed7-20261110/roboscope-policy.json --episodes 5 --concurrency 4 --replicas 2 --output-dir /root/autodl-tmp/openvla-kit-200-fixed4g2-seed7-20261110/libero_goal
```

```bash
/root/autodl-tmp/openvla-kit-20261109/runtime/env/bin/python -m roboscope_kit.runtime.benchmark --benchmark libero --suite libero_object --checkpoint-source /root/autodl-fs/Kit-Models/openvla/object --benchmark-manifest /root/autodl-tmp/openvla-kit-200-fixed4g2-seed7-20261110/benchmark-manifest.json --policy-config /root/autodl-tmp/openvla-kit-200-fixed4g2-seed7-20261110/roboscope-policy.json --episodes 5 --concurrency 4 --replicas 2 --output-dir /root/autodl-tmp/openvla-kit-200-fixed4g2-seed7-20261110/libero_object
```

```bash
/root/autodl-tmp/openvla-kit-20261109/runtime/env/bin/python -m roboscope_kit.runtime.benchmark --benchmark libero --suite libero_spatial --checkpoint-source /root/autodl-fs/Kit-Models/openvla/spatial --benchmark-manifest /root/autodl-tmp/openvla-kit-200-fixed4g2-seed7-20261110/benchmark-manifest.json --policy-config /root/autodl-tmp/openvla-kit-200-fixed4g2-seed7-20261110/roboscope-policy.json --episodes 5 --concurrency 4 --replicas 2 --output-dir /root/autodl-tmp/openvla-kit-200-fixed4g2-seed7-20261110/libero_spatial
```

### openvla-kit-200-serial-seed7-20261109

```bash
/root/autodl-tmp/openvla-kit-20261109/runtime/env/bin/python -m roboscope_kit.runtime.benchmark --benchmark libero --suite libero_10 --checkpoint-source /root/autodl-fs/Kit-Models/openvla/10 --benchmark-manifest /root/autodl-tmp/openvla-kit-200-serial-seed7-20261109/benchmark-manifest.json --policy-config /root/autodl-tmp/openvla-kit-200-serial-seed7-20261109/roboscope-policy.json --episodes 5 --concurrency 1 --replicas 1 --output-dir /root/autodl-tmp/openvla-kit-200-serial-seed7-20261109/libero_10
```

```bash
/root/autodl-tmp/openvla-kit-20261109/runtime/env/bin/python -m roboscope_kit.runtime.benchmark --benchmark libero --suite libero_goal --checkpoint-source /root/autodl-fs/Kit-Models/openvla/goal --benchmark-manifest /root/autodl-tmp/openvla-kit-200-serial-seed7-20261109/benchmark-manifest.json --policy-config /root/autodl-tmp/openvla-kit-200-serial-seed7-20261109/roboscope-policy.json --episodes 5 --concurrency 1 --replicas 1 --output-dir /root/autodl-tmp/openvla-kit-200-serial-seed7-20261109/libero_goal
```

```bash
/root/autodl-tmp/openvla-kit-20261109/runtime/env/bin/python -m roboscope_kit.runtime.benchmark --benchmark libero --suite libero_object --checkpoint-source /root/autodl-fs/Kit-Models/openvla/object --benchmark-manifest /root/autodl-tmp/openvla-kit-200-serial-seed7-20261109/benchmark-manifest.json --policy-config /root/autodl-tmp/openvla-kit-200-serial-seed7-20261109/roboscope-policy.json --episodes 5 --concurrency 1 --replicas 1 --output-dir /root/autodl-tmp/openvla-kit-200-serial-seed7-20261109/libero_object
```

```bash
/root/autodl-tmp/openvla-kit-20261109/runtime/env/bin/python -m roboscope_kit.runtime.benchmark --benchmark libero --suite libero_spatial --checkpoint-source /root/autodl-fs/Kit-Models/openvla/spatial --benchmark-manifest /root/autodl-tmp/openvla-kit-200-serial-seed7-20261109/benchmark-manifest.json --policy-config /root/autodl-tmp/openvla-kit-200-serial-seed7-20261109/roboscope-policy.json --episodes 5 --concurrency 1 --replicas 1 --output-dir /root/autodl-tmp/openvla-kit-200-serial-seed7-20261109/libero_spatial
```


## Autotune configuration

The [recorded launcher](../runs/autotune/launch.py) defines the suite commands and
environment for the three completed formal evaluations. The [run status](../runs/autotune/status.json)
records completed suites, command durations, and the final command. Each suite's
`autotune/selection.json` records candidate limits, sample counts, and selection.

## Reading the diagnostic records

Diagnostic files ending in `.jsonl.gz` contain the original JSONL bytes compressed
with gzip. Instrumentation patches use `.patch.gz` for the same lossless storage. Read JSONL with `gzip -dc FILE.jsonl.gz`; to use the archived analysis
scripts, unpack them to their original `.jsonl` names in a separate copy of the
diagnostic directory. The [image guide](../diagnostics/images/README.md) links the
seven retained PNGs; the full image/tensor inputs required by image-replay scripts
are not bundled. Commands retain the experiment machine's paths;
map those roots to the prepared environment when reusing the archived scripts.
The [source index](SOURCE_INDEX.json) records each retained file's origin and hash.
