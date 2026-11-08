# Pinned ManiSkill task inventory

The source revision is `baab60ede2e89167c1b7aaed41a9aa8e690a9d1e`. Its archive SHA-256 is
`db7f1fb49038a1a11612876022ba31fdb8c9095e4e3f4a408728abf8b821f49e`. An AST scan of
`mani_skill/envs/tasks/**/*.py` finds 58 literal `register_env` decorators. The default registry
also contains nine scene exploration IDs, for 67 registered environments. The minimal template
is a separate source example and is not imported into that registry.

Of the 58 task-directory IDs, 55 define evaluation tasks. Three are helper/reference environments
with source-backed reasons below. `upstream_tasks_v1` declares all 55 evaluation IDs at their
upstream horizons, with 100 episodes and seeds 0–99 per profile. Its 51 state profiles and four
Bridge `rgb+segmentation` profiles are also selectable as `upstream_state_v1` and `bridge_rgb_v1`.
These are explicitly named Kit protocols. ManiSkill does not define one unified official full suite.

## Protocol and validation scope

Each profile freezes the robot, native controller, observation mode, simulation/control frequency,
full time limit, reset behavior, scoring and required asset IDs in `benchmark-manifest.json`.
Defaults retain 100 Hz physics and 20 Hz control. DM-Control Ant/Humanoid use 200/40 Hz,
Hopper uses 100/25 Hz and Bridge uses 500/5 Hz, following their source configurations. Each
episode group performs a complete seeded reconfiguration. Evaluation runs the fixed horizon
and records native terminal masks, success-once/end where defined, and return.

The nine `MS-*` control tasks, both humanoid standing tasks and `AnymalC-Spin-v1` have no native
`success` field. Their primary metric is mean episode return. Their success counts remain absent.
Helper scenes are not converted into artificial success benchmarks.

`official_ppo_state_v1` selects 11 public checkpoint-compatible profiles. The original three-task
`baseline_state_v1` and 18-task state/RGB suites remain selectable with their original profile IDs.
LiftPegUpright and both dual-arm checkpoints use separate profiles for their required controllers.
RGB task definitions do not establish trained RGB-policy acceptance. The four Bridge profiles
also have the separate public [Octo-small visual example](../../examples/maniskill/OCTO.md).

The [review validation](REVIEW_VALIDATION.md) records current-source 100-episode-per-task native
comparisons for eleven state PPO tasks and four Bridge visual tasks. The table's public state PPO
column identifies the state checkpoints; the Bridge tasks use the visual example instead.
The other 40 evaluation tasks lack a selected compatible public checkpoint. Full benchmark
acceptance remains incomplete while other task, observation and controller profiles lack real-policy
runs and matched native comparisons. Asset download failures remain task-level blockers.

The [frozen official demonstration repository](https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations/tree/3558f72cf99d51d04a6ac4779e88b3da4bbf1628)
contains 26 PPO checkpoint files for those same eleven task IDs; the other fifteen files are
alternate controllers for existing tasks. Motion-planning demonstrations are distinct from
trained policy checkpoints. Missing entries below describe this selected model inventory,
without asserting that no other public model exists.

## Evaluation task definitions

| Task | Profile in `upstream_tasks_v1` | Horizon | Metric | Public state PPO | Source |
| --- | --- | ---: | --- | --- | --- |
| `AnymalC-Reach-v1` | `anymal_c_reach_state` | 200 | success_once | Available | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/quadruped/quadruped_reach.py#L161) |
| `AnymalC-Spin-v1` | `anymal_c_spin_state` | 200 | return | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/quadruped/quadruped_spin.py#L129) |
| `AssemblingKits-v1` | `assembling_kits_state` | 200 | success_once | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/tabletop/assembling_kits.py#L25) |
| `FMBAssembly1Easy-v1` | `fmb_assembly1_easy_state` | 500 | success_once | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/fmb/fmb.py#L22) |
| `LiftPegUpright-v1` | `lift_peg_upright_ppo_state` | 50 | success_once | Available | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/tabletop/lift_peg_upright.py#L22) |
| `MS-AntRun-v1` | `ms_ant_run_state` | 1000 | return | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/control/ant.py#L315) |
| `MS-AntWalk-v1` | `ms_ant_walk_state` | 1000 | return | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/control/ant.py#L298) |
| `MS-CartpoleBalance-v1` | `ms_cartpole_balance_state` | 1000 | return | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/control/cartpole.py#L179) |
| `MS-CartpoleSwingUp-v1` | `ms_cartpole_swing_up_state` | 1000 | return | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/control/cartpole.py#L214) |
| `MS-HopperHop-v1` | `ms_hopper_hop_state` | 600 | return | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/control/hopper.py#L245) |
| `MS-HopperStand-v1` | `ms_hopper_stand_state` | 600 | return | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/control/hopper.py#L218) |
| `MS-HumanoidRun-v1` | `ms_humanoid_run_state` | 1000 | return | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/control/humanoid.py#L387) |
| `MS-HumanoidStand-v1` | `ms_humanoid_stand_state` | 1000 | return | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/control/humanoid.py#L285) |
| `MS-HumanoidWalk-v1` | `ms_humanoid_walk_state` | 1000 | return | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/control/humanoid.py#L355) |
| `OpenCabinetDoor-v1` | `open_cabinet_door_state` | 100 | success_once | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/mobile_manipulation/open_cabinet_drawer.py#L362) |
| `OpenCabinetDrawer-v1` | `open_cabinet_drawer_state` | 100 | success_once | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/mobile_manipulation/open_cabinet_drawer.py#L33) |
| `PegInsertionSide-v1` | `peg_insertion_side_state` | 100 | success_once | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/tabletop/peg_insertion_side.py#L51) |
| `PickClutterYCB-v1` | `pick_clutter_ycb_state` | 100 | success_once | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/tabletop/pick_clutter_ycb.py#L195) |
| `PickCube-v1` | `pickcube_state` | 50 | success_once | Available | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/tabletop/pick_cube.py#L20) |
| `PickSingleYCB-v1` | `pick_single_ycb_state` | 50 | success_once | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/tabletop/pick_single_ycb.py#L28) |
| `PlaceSphere-v1` | `place_sphere_state` | 50 | success_once | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/tabletop/place_sphere.py#L24) |
| `PlugCharger-v1` | `plug_charger_state` | 200 | success_once | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/tabletop/plug_charger.py#L21) |
| `PokeCube-v1` | `poke_cube_state` | 50 | success_once | Available | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/tabletop/poke_cube.py#L21) |
| `PullCube-v1` | `pull_cube_state` | 50 | success_once | Available | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/tabletop/pull_cube.py#L21) |
| `PullCubeTool-v1` | `pull_cube_tool_state` | 100 | success_once | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/tabletop/pull_cube_tool.py#L20) |
| `PushCube-v1` | `pushcube_state` | 50 | success_once | Available | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/tabletop/push_cube.py#L38) |
| `PushT-v1` | `pusht_state` | 100 | success_once | Available | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/tabletop/push_t.py#L66) |
| `PutCarrotOnPlateInScene-v1` | `put_carrot_on_plate_in_scene_rgb` | 60 | success_once | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/digital_twins/bridge_dataset_eval/put_on_in_scene.py#L17) |
| `PutEggplantInBasketScene-v1` | `put_eggplant_in_basket_scene_rgb` | 120 | success_once | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/digital_twins/bridge_dataset_eval/put_on_in_scene.py#L75) |
| `PutSpoonOnTableClothInScene-v1` | `put_spoon_on_table_cloth_in_scene_rgb` | 60 | success_once | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/digital_twins/bridge_dataset_eval/put_on_in_scene.py#L215) |
| `RollBall-v1` | `rollball_state` | 80 | success_once | Available | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/tabletop/roll_ball.py#L21) |
| `RotateSingleObjectInHandLevel0-v1` | `rotate_single_object_in_hand_level0_state` | 300 | success_once | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/dexterity/rotate_single_object_in_hand.py#L321) |
| `RotateSingleObjectInHandLevel1-v1` | `rotate_single_object_in_hand_level1_state` | 300 | success_once | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/dexterity/rotate_single_object_in_hand.py#L333) |
| `RotateSingleObjectInHandLevel2-v1` | `rotate_single_object_in_hand_level2_state` | 300 | success_once | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/dexterity/rotate_single_object_in_hand.py#L349) |
| `RotateSingleObjectInHandLevel3-v1` | `rotate_single_object_in_hand_level3_state` | 300 | success_once | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/dexterity/rotate_single_object_in_hand.py#L365) |
| `RotateValveLevel0-v1` | `rotate_valve_level0_state` | 80 | success_once | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/dexterity/rotate_valve.py#L256) |
| `RotateValveLevel1-v1` | `rotate_valve_level1_state` | 150 | success_once | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/dexterity/rotate_valve.py#L268) |
| `RotateValveLevel2-v1` | `rotate_valve_level2_state` | 150 | success_once | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/dexterity/rotate_valve.py#L280) |
| `RotateValveLevel3-v1` | `rotate_valve_level3_state` | 150 | success_once | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/dexterity/rotate_valve.py#L292) |
| `RotateValveLevel4-v1` | `rotate_valve_level4_state` | 300 | success_once | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/dexterity/rotate_valve.py#L304) |
| `StackCube-v1` | `stackcube_state` | 50 | success_once | Available | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/tabletop/stack_cube.py#L19) |
| `StackGreenCubeOnYellowCubeBakedTexInScene-v1` | `stack_green_cube_on_yellow_cube_baked_tex_in_scene_rgb` | 60 | success_once | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/digital_twins/bridge_dataset_eval/put_on_in_scene.py#L155) |
| `TriFingerRotateCubeLevel0-v1` | `tri_finger_rotate_cube_level0_state` | 250 | success_once | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/rotate_cube.py#L352) |
| `TriFingerRotateCubeLevel1-v1` | `tri_finger_rotate_cube_level1_state` | 250 | success_once | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/rotate_cube.py#L363) |
| `TriFingerRotateCubeLevel2-v1` | `tri_finger_rotate_cube_level2_state` | 250 | success_once | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/rotate_cube.py#L374) |
| `TriFingerRotateCubeLevel3-v1` | `tri_finger_rotate_cube_level3_state` | 250 | success_once | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/rotate_cube.py#L385) |
| `TriFingerRotateCubeLevel4-v1` | `tri_finger_rotate_cube_level4_state` | 250 | success_once | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/rotate_cube.py#L396) |
| `TurnFaucet-v1` | `turn_faucet_state` | 200 | success_once | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/tabletop/turn_faucet.py#L24) |
| `TwoRobotPickCube-v1` | `two_robot_pickcube_ppo_state` | 100 | success_once | Available | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/tabletop/two_robot_pick_cube.py#L21) |
| `TwoRobotStackCube-v1` | `two_robot_stackcube_ppo_state` | 100 | success_once | Available | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/tabletop/two_robot_stack_cube.py#L22) |
| `UnitreeG1PlaceAppleInBowl-v1` | `unitree_g1_place_apple_in_bowl_state` | 100 | success_once | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/humanoid/humanoid_pick_place.py#L210) |
| `UnitreeG1Stand-v1` | `unitree_g1_stand_state` | 1000 | return | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/humanoid/humanoid_stand.py#L100) |
| `UnitreeG1TransportBox-v1` | `unitree_g1_transport_box_state` | 100 | success_once | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/humanoid/transport_box.py#L25) |
| `UnitreeGo2-Reach-v1` | `unitree_go2_reach_state` | 200 | success_once | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/quadruped/quadruped_reach.py#L191) |
| `UnitreeH1Stand-v1` | `unitree_h1_stand_state` | 1000 | return | Missing | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/humanoid/humanoid_stand.py#L67) |

## Helpers and templates

| ID | Reason | Source |
| --- | --- | --- |
| `Empty-v1` | Dummy environment for showcasing robots (class docstring). | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/empty_env.py) |
| `RoboCasaKitchen-v1` | Base kitchen scene constructor, reward mode none and evaluate() returns no task metric. | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/mobile_manipulation/robocasa/kitchen.py) |
| `TableTopFreeDraw-v1` | Reference for creating drawing tasks, with no goal or success conditions (class docstring). | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/tasks/drawing/draw.py) |
| `SceneManipulation-v1` | Exploration/visualization scene without task success/failure metrics or rewards (SceneManipulationEnv docstring). | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/scenes/base_env.py) |
| `ArchitecTHOR_SceneManipulation-v1` | Exploration/visualization scene without task success/failure metrics or rewards (SceneManipulationEnv docstring). | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/scenes/__init__.py) |
| `ReplicaCAD_SceneManipulation-v1` | Exploration/visualization scene without task success/failure metrics or rewards (SceneManipulationEnv docstring). | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/scenes/__init__.py) |
| `ReplicaCADPrepareGroceriesTrain_SceneManipulation-v1` | Exploration/visualization scene without task success/failure metrics or rewards (SceneManipulationEnv docstring). | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/scenes/__init__.py) |
| `ReplicaCADPrepareGroceriesVal_SceneManipulation-v1` | Exploration/visualization scene without task success/failure metrics or rewards (SceneManipulationEnv docstring). | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/scenes/__init__.py) |
| `ReplicaCADSetTableTrain_SceneManipulation-v1` | Exploration/visualization scene without task success/failure metrics or rewards (SceneManipulationEnv docstring). | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/scenes/__init__.py) |
| `ReplicaCADSetTableVal_SceneManipulation-v1` | Exploration/visualization scene without task success/failure metrics or rewards (SceneManipulationEnv docstring). | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/scenes/__init__.py) |
| `ReplicaCADTidyHouseTrain_SceneManipulation-v1` | Exploration/visualization scene without task success/failure metrics or rewards (SceneManipulationEnv docstring). | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/scenes/__init__.py) |
| `ReplicaCADTidyHouseVal_SceneManipulation-v1` | Exploration/visualization scene without task success/failure metrics or rewards (SceneManipulationEnv docstring). | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/scenes/__init__.py) |
| `CustomEnv-v1` | Minimal template, not imported into the default registry. | [Definition](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/envs/minimal_template.py) |

The scene helper loop explicitly registers environments for benchmarking/exploration without
designed tasks. `RoboCasaKitchen-v1` is the ManiSkill base scene and is separate from the
RoboScope RoboCasa GR-1 Tabletop benchmark. These exclusions are based on the pinned source,
and can be reviewed independently of asset or model availability.

## Reproduce the source inventory

Read the checksum-verified source built by `build-runtime.sh`. Parse each task module with
`ast.parse`, collect literal string arguments on class `register_env` decorators, and compare
the resulting 58 unique IDs and decorator horizons against `upstream_inventory` in the manifest.
Import `mani_skill.envs` in that runtime and compare `mani_skill.utils.registration.REGISTERED_ENVS`
with those IDs plus the nine documented scene helpers. The source-only `CustomEnv-v1` template
is intentionally absent from that import. No simulator or evaluation output is inferred from
the static inventory.
