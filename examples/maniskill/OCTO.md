# Octo-small Bridge example

This Policy serves the public Octo-small 1.0 checkpoint on the four ManiSkill Bridge digital-twin
profiles. Octo was trained on Open X-Embodiment data including Bridge real-robot demonstrations.
The checkpoint, metadata and offline T5 tokenizer are pinned by immutable revisions, byte counts
and SHA-256 in [octo-checkpoint.json](octo-checkpoint.json). The checkpoint is about 0.55 GB.
The model card declares MIT. [Official checkpoint](https://huggingface.co/rail-berkeley/octo-small/tree/03d88976c54a58e10480d2043a8c762b35bc2611)
describes the training mixture and architecture.

Build a separate inference environment and download its inputs outside Git.

```bash
UV_BIN=/path/to/uv bash examples/maniskill/build-octo-policy.sh /path/to/octo-policy-runtime
/path/to/octo-policy-runtime/env/bin/python examples/maniskill/download_octo.py /path/to/octo-model
```

Set the absolute interpreter and repository paths in [octo-policy.json](octo-policy.json), then
select `bridge_rgb_v1` with the simulator runtime's manifest.

```bash
/path/to/simulator/env/bin/python -m roboscope_kit.runtime.benchmark \
  --benchmark maniskill --suite bridge_rgb_v1 \
  --benchmark-manifest /path/to/simulator/benchmark-manifest.json \
  --checkpoint-source /path/to/octo-model --policy-config /path/to/octo-policy.json \
  --quick-test --output-dir /path/to/results/octo-quick
```

Quick Test runs the first complete ten-episode group. For the configured 100 episodes per task, replace `--quick-test` with
`--concurrency 1 --replicas 1`. Keep the same vector grouping on the native comparison.

The implementation follows SimplerEnv's `widowx_bridge` preprocessing and action conversion.
It resizes RGB to 256×256 with TensorFlow Lanczos3, retains two observation frames, predicts four
future actions and ensembles the predictions for the current tick with equal weights. Bridge
dataset action statistics restore the translation, rotation and gripper units. The gripper is
thresholded to −1/+1. The environment uses its native seven-dimensional controller and the
original task language instruction.
[Pinned SimplerEnv implementation](https://github.com/simpler-env/SimplerEnv/blob/eddb5690749a1e34267807c0f7b5eede323918d7/simpler_env/policies/octo/octo_model.py)
is the inference reference.

Rotation encoding follows the target simulator's controller. Octo's Bridge rotation deltas use
fixed-axis xyz, equivalent to `Rz @ Ry @ Rx`. The pinned ManiSkill3 `PDEEPoseController` consumes
intrinsic XYZ angles and constructs `Rx @ Ry @ Rz`. The Policy converts through a rotation matrix
to preserve the requested rotation. The legacy SimplerEnv wrapper instead emits an axis-angle
vector. Passing that vector directly into the ManiSkill3 Euler fields would produce a different
rotation. [Pinned controller](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/agents/controllers/pd_ee_pose.py#L212-L235)
defines the required target representation.

The pinned ManiSkill3 controller adds translation in its root frame to its retained target.
Legacy SimplerEnv's `ee_align2` rotates the retained target around the current end-effector
position. This example retains the ManiSkill3 task controller. Legacy SimplerEnv physical
trajectories and reported task scores require their own simulator configuration.

Each episode has independent image history, action history and random state. It starts with
`init_rng=0`, performs the reference's five initial key splits, and splits once per inference.
This gives reproducible episode isolation across workers and replicas. The original SimplerEnv
process can continue one global random stream across episode resets. The Kit example and native
reference both use the explicitly declared independent stream. A batch maps the original
single-episode model call over those independent keys.

The example enables `--xla_gpu_deterministic_ops=true` before JAX initialization. Without it,
the pinned XLA can select different algorithms from startup timings: separate processes produced
approximately `1e-5` action differences on identical images. Two fresh-process diagnostics with
this flag produce bit-identical actions and each matches the unmodified reference inference.
Precision, PRNG and model parameters remain unchanged; GPU autotuning retains its default level.
The [pinned XLA selection code](https://github.com/openxla/xla/blob/ca31652cdbeb6ea187589dea546ff8019274f8b2/xla/service/gpu/stream_executor_util.cc#L537)
defines the repeatable algorithm choice. Exact pairing also requires the same vector width and
history grouping on both evaluation paths.

The native reference constructs Gym environments without the Kit Worker, using the declared
manifest configuration. This is a configured-reference diagnostic, not independently resolved
upstream task-configuration acceptance. It runs the
actor in a persistent subprocess using the same isolated inference interpreter, with lossless RGB
transport. This keeps JAX/TF dependencies out of the simulator environment. Native timing includes
that reference transport and is used for correctness validation.

```bash
/path/to/simulator/env/bin/python examples/maniskill/native_eval_octo.py \
  --checkpoint /path/to/octo-model --policy-python /path/to/octo-policy-runtime/env/bin/python \
  --benchmark-manifest /path/to/simulator/benchmark-manifest.json \
  --suite bridge_rgb_v1 --num-envs 10 --output /path/to/results/octo-native.json
```

Use `--trajectory-dir /path/to/traces` for a separate diagnostic. Compare initial observations,
per-step actions, success-once/end, returns, actual steps, seeds and complete task coverage.
The public SimplerEnv results use ManiSkill2 real2sim task versions. Their reported task scores
are separate from the pinned ManiSkill3 profiles here. This retained example is provisional:
OXE/Bridge training does not establish matched per-task competence, and no final-source
prescribed-count acceptance is claimed. See the
[validation status](../../benchmarks/maniskill/VALIDATION.md) for the remaining requirements.
