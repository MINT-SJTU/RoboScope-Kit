# ManiSkill fine-tuned Octo visual policy

This example loads the real Octo checkpoint released with the Robotics Diffusion Transformer
ManiSkill comparison. It is separate from the Bridge Octo example: it uses the author's ManiSkill
fine-tuning, Octo v1.5 source, and direct normalized end-effector commands. It does not use Bridge
action statistics. The published comparison reports zero success for this Octo baseline on its five
tasks, so successful inference is an integration check, not evidence of task competence.
This is an unvalidated candidate; its published zero-success baseline does not meet
the intended-task competence requirement for acceptance.

## Pinned sources

- [Released checkpoint](https://huggingface.co/robotics-diffusion-transformer/maniskill-model/tree/9622afab2b7ce2312a6cf1febc526928589b77eb/octo/experiment_20241208_112612),
  revision `9622afab2b7ce2312a6cf1febc526928589b77eb`, step `1000000`.
- [Author's evaluator](https://github.com/thu-ml/RoboticsDiffusionTransformer/blob/98e6d7b725797b9640d219a8fd385009e13fa824/eval_sim/eval_octo.py)
  and [reported baseline results](https://github.com/thu-ml/RoboticsDiffusionTransformer/blob/98e6d7b725797b9640d219a8fd385009e13fa824/README.md).
- [Octo v1.5](https://github.com/octo-models/octo/tree/5eaa5c6960398925ae6f52ed072d843d2f3ecb0b)
  and [dlimp](https://github.com/kvablack/dlimp/tree/5edaa4691567873d495633f2708982b42edf1972).

`panda-octo-checkpoint.json` pins every inference checkpoint file, both source archives, and the
offline T5 tokenizer. The checkpoint files total 549,407,955 bytes; optimizer state is not downloaded.
Weights, extracted sources, import overlays, and tokenizer caches stay outside Git.
The release stores eight-device training sharding metadata. The loader supplies explicit Orbax
`ArrayRestoreArgs` with single-device sharding so the original tensor values can be restored on one
evaluation GPU or CPU; it does not change shapes, values, or model operations.

## Prepare

Use the pinned inference Python built by `build-octo-policy.sh`; the simulator uses its own runtime.
The extra packages are installed into the new checkpoint directory, leaving the existing Bridge
environment intact. Octo v1.0 and v1.5 must run in separate Policy processes.

```bash
bash examples/maniskill/build-octo-policy.sh /autodl-fs/roboscope/octo-runtime
/autodl-fs/roboscope/octo-runtime/env/bin/python \
  examples/maniskill/download_panda_octo.py /autodl-fs/roboscope/models/panda-octo
bash examples/maniskill/prepare-panda-octo.sh \
  /autodl-fs/roboscope/octo-runtime/env/bin/python /autodl-fs/roboscope/models/panda-octo
```

`PANDA_OCTO_WHEEL_CACHE` optionally supplies an offline wheel directory for the eight packages in
`panda-octo-requirements.txt`. Preparation checks imports on CPU and writes their provenance into
the checkpoint directory. `PANDA_OCTO_DEPENDENCY_DIR` can place this small import overlay on a local
runtime disk, with a symlink from the checkpoint directory, when the asset volume has a file quota.
Preparation does not execute the model or modify the simulator.

The public Policy is `panda_octo_policy.ManiSkillPandaOctoPolicy`. Pass the downloaded directory as
`SessionContext.checkpoint_path`, use an RGB profile exposing the real `images.primary`, and retain
the manifest's robot, native controller, camera, seeds, success rules, and full episode horizon.
The shipped candidate suite selects only `peg_insertion_side_rgb` and `plug_charger_rgb`,
whose existing controllers supply all seven native EE-pose action values. The author's other
three tasks are PickCube, StackCube and PushCube; their existing Kit RGB profiles use a different
position controller and remain unverified with this checkpoint. Their controllers are unchanged.

## Observation and action contract

The model consumes the actual primary RGB frame and the profile's language instruction. Images use
the author's TensorFlow Lanczos3 resize to 256 by 256 and truncating `uint8` conversion. Its released
configuration contains primary and wrist image tokenizers and **no proprioception tokenizer**.
The wrist view is absent in the author's evaluator and here. Proprioception is omitted.
The checkpoint's unused example `proprio` field
does not establish a consumed state input.

Each episode has two frames of history, an initial `[false, true]` history mask, a PRNG initialized to
zero, and a four-action queue. New real frames enter history on every native step, including steps
which consume queued actions. The source evaluator's random generator is global across episodes;
this adapter instead isolates a reproducible generator per episode. Both use the same split sequence
within an episode. Policy reset clears history, pending actions, instruction, and generator state.

The author sends all seven predicted values directly to ManiSkill's normalized `pd_ee_delta_pose`
controller without applying checkpoint action statistics. This example preserves that decoding:
translation, rotation, and gripper values are native normalized controller commands, not directly
metres or radians. Native controller scaling and clipping remain inside the frozen simulator.

The Policy requires a single native Panda and the complete seven-value `pd_ee_delta_pose`
controller on an author-listed task. Position-only controllers, PandaStick, dual-arm composition
and unrelated tasks are rejected. The benchmark's other original state/RGB profiles remain
selectable with their own model requirements.

`ManiSkillFinetunedOcto.reset(id, instruction)`, `act_batch(ids, images, proprios=None)`, and
`end_episode(id)` expose the learned model's per-episode lifecycle.
`act_batch` returns raw EE7 and ignores the optional proprioception argument by design.

## Acceptance status

Current-source simulation and independent author-configuration validation remain outstanding.
Keeping the public checkpoint and its intended-task inference code does not establish task
competence, full-profile coverage or an accepted visual baseline.
