# Official MS-HAB PPO Fetch transfer

This policy runs the real official MS-HAB PPO Open checkpoints on native ManiSkill Fetch cabinet tasks. It preserves the target robot, controller, reset, task assets, seeds, time limit and success rules. Its inputs and deterministic actor follow the source model. The source policies were trained on ReplicaCAD fridge/kitchen-counter articulations; applying them to PartNet cabinets is an explicit zero-shot transfer, not a claim of task-trained PartNet expertise or source success-rate reproduction.

The model implementation and CPU source comparison are complete. Original processed PartNet cabinet assets remain necessary for canonical cabinet validation. CPU fixtures and unrelated assets do not count as task acceptance.

## Checkpoints and attribution

Models are from [Arth Shukla, Stone Tao and Hao Su's ManiSkill-HAB release](https://huggingface.co/arth-shukla/mshab_checkpoints/tree/91e96be85128df43728a7511355c3fa999bd2c94/rl/set_table/open), immutable revision `91e96be85128df43728a7511355c3fa999bd2c94`. The model card declares **CC-BY-4.0**. Retain this attribution when redistributing the released weights.

| Native task | Official source checkpoint | Size | SHA-256 |
| --- | --- | --- | --- |
| `OpenCabinetDoor-v1` | `rl/set_table/open/fridge/policy.pt` | 68,155,930 B | `d863fa8642584ccdfe2834a6a9b3baa34884d399a5e221fd682639fccdcc879a` |
| `OpenCabinetDrawer-v1` | `rl/set_table/open/kitchen_counter/policy.pt` | 68,155,930 B | `290b22d3c23d12927b6a906e2e54670c9d90605b3e04ef7717c4dd2ad3787de1` |

`fetch-mshab-checkpoints.json` pins each task's original checkpoint. `prepare-fetch-mshab.py` downloads the two original files, verifies published size and SHA-256, and writes `models.json`. Files total 136,311,860 bytes. Models and caches belong outside Git, for example below `/autodl-fs`.

```bash
python prepare-fetch-mshab.py /autodl-fs/data/roboscope/maniskill/models/fetch-mshab
```

An optional `--proxy http://host:port` configures the downloader. Policy loading receives that directory or its `models.json` through `context.checkpoint_path`; it does not fetch weights during evaluation. Inference needs NumPy, PyTorch and RoboScope Kit, using the existing inference environment. No MS-HAB scene dataset, training package or optimizer dependency is required for inference.

The architecture follows the [author PPO agent](https://github.com/arth-shukla/mshab/blob/e9ff3d23496d38e4431c8d913e147ffa007f7f72/mshab/agents/ppo/agent.py) at code revision `e9ff3d23496d38e4431c8d913e147ffa007f7f72`; that project's package metadata declares MIT. Strict loading retains all **27** released model tensors, including the unused critic and learned actor log standard deviation. The optimizer payload is not used for evaluation. Deterministic inference returns the original actor mean; it does not sample or insert an extra tanh.

## Native robot and action contract

The pinned target ManiSkill revision is `baab60ede2e89167c1b7aaed41a9aa8e690a9d1e`. The audited source's recommended ManiSkill `mshab` branch resolved to `17121e3f96e3ee3ed0c03610b17f8bc2864617af`; this is not a claim of the exact historical training commit. Both Fetch URDFs are byte-identical, SHA-256 `7503ea2dc96eb3d1130c81f7ff1e397f30df46459b3dbf10cf5d0d247c5b1256`. All eight shared Fetch methods, including controller and sensor definitions, have identical ASTs.

The 13-dimensional native `pd_joint_delta_pos` action contains arm7, mimic-gripper1, body3 and planar-base2 controls. Arm/body deltas are scaled by 0.1 native joint units, gripper targets by [-0.01,0.05] m, forward velocity by [-1,1] m/s, yaw velocity by [-3.14,3.14] rad/s. The source wrapper zeroes head-pan/head-tilt values at indices 8 and 9; torso and base remain movable. The policy preserves this operation and leaves clipping/scaling to the native controllers. Both source and target use 100 Hz physics and 20 Hz control.

Fifteen controller joint identities correspond to thirteen controls because each mimic-gripper action controls two joints and the two planar velocity controls actuate three virtual base joints. The policy checks this component ordering rather than equating joint-name count with action dimension. State order is established by the actual CPU SAPIEN articulation, not XML ordering: base x/y/yaw, torso, head pan, shoulder pan, head tilt, remaining six arm joints, right finger, left finger. Named mapping preserves that source order if metadata presents another state order.

## Real observation contract

The source's published configuration uses `depth`, three stacked camera frames, concatenated state, and separate depth-camera inputs. The source model takes a **42-dimensional state** and two depth tensors `[B,3,1,128,128]`. The NN preprocesses actual millimeter depth with `1 - tanh(depth / 1000)`.

The SDK must supply the following physical observations. All quaternion arrays are position3 followed by scalar-first quaternion4, in world coordinates. Contact forces are world-frame forces on the named finger from the selected handle link, in newtons.

| Raw SDK feature | Shape | Use |
| --- | --- | --- |
| `state.proprioception` | At least `(30,)` | Native qpos15/qvel15 prefix, mapped by `active_joint_names` |
| `state.robot_base_pose` | `(7,)` | Live `base_link.pose`; the articulation root alone is insufficient |
| `state.robot_tcp_pose` | `(7,)` | `agent.tcp.pose`, matching the source observation's TCP link |
| `state.target_handle_position` | `(3,)` | Actual native task handle marker position |
| `state.left_finger_pose`, `state.right_finger_pose` | Each `(7,)` | Actual finger link poses |
| `state.left_target_contact_force`, `state.right_target_contact_force` | Each `(3,)` | Actual pairwise finger/handle contact forces |
| `state.depth_head`, `state.depth_hand` | Each `(128,128,1)` | Raw native depth in mm, transported as float32 without RGB conversion |

The two existing Fetch cameras are source-identical: head mounted at identity on `head_camera_link`; hand mounted at `[-0.1,0,0.1]` with identity rotation on `gripper_link`; 128x128 pixels, FOV 2 rad, near/far 0.01/100 m. Each episode owns independent three-frame deques. Reset fills each deque with the first real frame three times, matching the author wrapper; every later action appends the latest real depth frame.

The model state is qpos excluding base3 (12), corresponding qvel (12), base-relative TCP pose (7), source-defined absent-object pose (7), base-relative handle position (3), and a target-link grasp flag (1). Open/Close assign no loose object, so the seven zeros are their actual source observation semantics. No robot state, camera or target feature is fabricated. Relative quaternion standardization and rotation arithmetic follow the source. Grasping requires both target-finger force magnitudes >=0.5 N and contact directions within 30 degrees of each finger's native opening direction.

Native visual/depth modes omit cabinet state-only extra fields. Export handle/base/contact quantities from the actual native environment; do not assume that switching observation mode preserves the old flattened state layout. SDK exports raw measurements only; the model wrapper owns pose conversion, grasp calculation, history and inference preprocessing.

## CPU source verification

On 2026-11-09, PyTorch 2.6.0+cu124 ran with `CUDA_VISIBLE_DEVICES` empty. Both official checkpoints strict-loaded all 27 tensors. Against the unmodified author Agent, a shared 32-input batch produced bitwise-identical outputs for each model (maximum absolute error 0). Thirty-two synthetic raw-observation fixtures were compared against the author's unchanged task-observation and native grasp methods: source-state error 0, all grasp flags equal, including 16 positive grasps. These fixtures are a conversion proof, not original cabinet rollouts.

Twenty-four public Policy actions across two interleaved episodes per task preserved exact depth history and reset isolation. The maximum action error against separately evaluated source episodes was 1.43e-6, attributable to the different CPU batch sizes; the same-batch network comparison above was exact. The actual native Fetch active-joint order was verified by building only a CPU PhysX articulation from a temporary visual-free copy of the identical URDF. The installed simulator was not patched.

Raw CPU proof: `fetch-mshab-cpu-source-parity.json` in the external validation artifact directory. The proof explicitly records `canonical_cabinet_executed=false`. A GPU five-step real-model comparison on the original PartNet tasks is still required before these two tasks can be accepted. Preserve native 100-step horizon and success rules; MS-HAB's 200-step horizon, rest requirements and collision-force criteria are training/evaluation differences, not changes to import into the target benchmark.
