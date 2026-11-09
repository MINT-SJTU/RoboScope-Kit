# ManiSkill completion-phase validation

This phase continues issue 25 on November 9, 2026. It starts from the pinned simulator and the
fifteen full-horizon task comparisons in [the prior review report](REVIEW_VALIDATION.md): eleven
official state PPO tasks and four Octo Bridge visual tasks, each with 100 matched SDK/native
episodes. That historical evidence retains its original source and runtime labels. The complete
requested scope remains the [55-task upstream inventory](TASK_INVENTORY.md).

The new policies below cover 27 additional task IDs in real GPU five-tick diagnostics, bringing
selected-profile prefix coverage to 42 of 55 task IDs. They do not add to the fifteen full-horizon
task comparisons. This is an integration-completion phase;
no runtime optimization or speedup is claimed.

## Candidate policies and provenance

Every selected weight stays outside Git and is checked against its pinned byte count and
SHA-256 before loading. Source revisions and licenses are recorded in the corresponding model
metadata or example documentation. The existing Torch/NumPy simulator environment supplies the
new inference paths; the adapters do not install the authors' training simulators.

| Candidate | Native task coverage | Selected inference path and limits | Provenance |
| --- | --- | --- | --- |
| NEWT, nine task-specific weights | MS-AntWalk/Run, MS-CartpoleBalance/SwingUp, MS-HopperStand/Hop, PlaceSphere, PickSingleYCB, PullCubeTool | Deterministic learned policy prior, `mpc=False`, `eval_mode=True`; the upstream 128D padded native-state contract and two-tick action hold. The apple checkpoint is transferred across the complete native YCB object distribution. | [NEWT.md](../../examples/maniskill/NEWT.md), [weight pins](../../examples/maniskill/newt-checkpoints.json) |
| HORA-RMA stage 2 student | RotateSingleObjectInHandLevel0–3 | Isaac Gym Allegro transfer, with measured native joint inputs, source normalization, 30-frame adaptation history and explicit joint permutation. Source target limits and native delta saturation remain visible. | [HORA policy](../../examples/maniskill/hora_policy.md), source `b1f009128f9adad3b07bc8aff91d7916feed09fc` |
| RRC TriFinger push/lift experts | TriFingerRotateCubeLevel0–4 | Real-robot torque-policy transfer. Push handles levels 0/1; lift handles levels 2/3/4. The native PD conversion matches requested torque only at the observation instant when unsaturated. Native PD substeps and the source held-torque trajectory differ. | [TriFinger policy](../../examples/maniskill/trifinger_policy.md), source `e4841da99127a19e319bd06b71d2becd52067e38` |
| ROBEL DClawScrewRandom actor | RotateValveLevel0–4 | Deterministic NumPy dense/ReLU/tanh inference with measured joint/valve inputs. Signed joint permutation, valve-frame conversion and moving policy-side target are explicit. Native valve geometry and task goals remain unchanged. | [ROBEL policy](../../examples/maniskill/robel_policy.md), source `039466811957f8d046213eddaa7d0e764971030d` |
| Official Unitree H1/G1 recurrent actors | UnitreeH1Stand, UnitreeG1Stand | Zero-velocity locomotion-to-standing transfer, using observed root orientation/angular velocity and native joint names. Each episode owns its LSTM, previous action and gait phase. | [Unitree policy](../../examples/maniskill/UNITREE.md), [weight pins](../../examples/maniskill/unitree-checkpoints.json) |
| Community Go2 locomotion actor | UnitreeGo2-Reach | Native-goal velocity-command transfer with observed root kinematics, named joint permutation and published one-tick action latency. The checkpoint publisher does not document its training simulator; the compatible Genesis observation contract is separately pinned. | [Go2 pin and contract](../../examples/maniskill/go2-checkpoint.json), model revision `5e51960de0e2f989f0b0a4a84300ce324a55544c` |
| Anymal locomotion transfer | AnymalC-Spin | Official Reach PPO with an episode-local counterclockwise orbit waypoint planner. Its 30 physical inputs and original actor weights/inference are unchanged; computed goal fields and observed root pose are explicit. It is not a Spin-trained expert. | [Orbit contract](../../examples/maniskill/ANYMAL_SPIN.md), [official PPO pin](../../examples/maniskill/official-models.json). |

The six NEWT control-task models are ManiSkill policies, rather than similarly named DeepMind
Control torque policies. Their source is `1d3fc058b81ddf8d36a5457c29ea407dd374c1b8`, with model
repository revision `7eef11eb63c8ed53d61d739693d7140135ea0876`. The nine checkpoint files total
650,488,995 bytes. The existing PullCubeTool position-controller profile stays unchanged; the
model selects its explicitly named pose-controller profile.

Transfer policies retain the native task's robot, sampling, physics, success criteria, controller
bounds, control frequency and horizon. A real pretrained model can execute without achieving
useful success on the transferred task. Original HORA, RRC, ROBEL and Unitree scores are not
reproduced by changing the controller representation alone. Detailed physical differences and
clipping/saturation diagnostics are retained in the linked policy documents.

## CPU evidence before rollout

The CPU checks establish that released weights load and that their inference conversions have
the documented numerical behavior. They are separate from environment execution and task success.

| Check | Recorded evidence |
| --- | --- |
| NEWT | Nine models, 32 states each, compared with pinned upstream `WorldModel.encode()` / `WorldModel.pi()`; action means are bitwise equal, maximum absolute error 0. |
| TriFinger | Both original exported TorchScript actors, 19 valid inputs each; reduced physical inputs produce bitwise-identical torques to the original full-observation path, maximum error 0. |
| ROBEL | Original six weight arrays, 31 observations; deterministic source dense/ReLU/tanh formula evaluated in float64 differs by at most `1.46e-6`, with finite bounded actions. This is not old TensorFlow graph rollout parity. |
| Unitree | Original recurrent actors with independently assembled deployment observations, two interleaved ten-step episodes per model; maximum native-action differences `2.15e-6` for H1 and `1.35e-6` for G1. Reset and cleanup isolate recurrent history. |
| HORA | Original source student on 17 histories; bitwise-equal action means, maximum error 0. Actual native Allegro state is 105D. |
| Go2 | Two interleaved ten-step episodes with permuted joint orders; maximum actor error 4.77e-7, native-action error 2.38e-7; latency reset/cleanup pass. |
| ANYmal orbit | Two interleaved ten-step episodes; physical prefix error 0, independent waypoint error 2.38e-7 and original-actor action error 7.15e-7; reset/cleanup pass. |

## Five-step integration method

The first GPU batch is a short prefix diagnostic. It loads each real selected model and executes
five native control ticks through the SDK path, with a matching independent Gym path where
available. The environment still receives its full configured `max_episode_steps`; the separate
`diagnostic_step_limit=5` stops the runner. It does not set a five-step environment horizon or
manufacture timeout/termination masks. The final native `terminated` and `truncated` values are
recorded. Since five ticks precede every selected task's native horizon, successful prefix
execution requires both masks to remain false.

Records retain `native_max_episode_steps`, `diagnostic_step_limit`, the original seed/profile,
actual step count and `evaluation_kind="integration_smoke"`. The diagnostic is non-formal.
For example, NEWT Hopper still has its native 600-tick horizon, Ant/Cartpole retain 1,000 ticks,
and the Unitree standing profiles retain 1,000 ticks. NEWT's two-tick hold changes only its own
policy-side action history; physics, observations, rewards and terminal checks still execute on
each native tick.

Named root-state profiles add measured root pose and velocities through the native public robot
API for the locomotion transfer models. Canonical `state.proprioception` remains unchanged.
Root orientation is observed rather than replaced by an upright constant. Joint identities and
native controller order are supplied explicitly to the policy.

The general native reference constructs Gym environments, extracts their native observations
and steps native actions independently of the SDK Worker and RPC transport. Its subprocess runs
the same public Policy implementation. It therefore checks runtime input extraction, episode
isolation, action routing and resulting native trajectories; it is not a second independently
implemented neural network. The CPU upstream-reference checks above supply separate model-side
evidence. Initial full-observation digests, actions and optional complete prefix traces are
compared without silently relaxing tolerances.

For every measured run, retain the complete command, source/manifest and checkpoint digests,
profile, seeds, K/G/vector width, native horizon, diagnostic limit, actual step count and terminal
masks. Record complete and evaluation wall time, episodes/h, control steps/s, inference round
trip, available rendering/wait metrics, GPU busy, mean/peak VRAM, CPU use, task outcomes and
execution errors. Missing independent GPU actor timing, queue timing, render timing or SM/Tensor/RT
counters is reported as N/A, not zero. Prefix throughput is diagnostic throughput and is not
compared with the earlier full-horizon baselines as a speedup.

## GPU results

Hardware: NVIDIA GeForce RTX 4090 (49,140 MiB exposed), driver 580.105.08, Intel Xeon Platinum 8470Q
with a 20-core container quota. Runs use K1/G1/N1 without MPS; no MPS service was active.

The final numerical source was built through the committed builders into fresh `runtime-r5` and
`octo-policy-runtime-r5` prefixes, with Kit 0.8.0b45, Python 3.11.15, ManiSkill 3.0.0b15/SAPIEN 3.0.0b1,
Torch 2.6.0+CUDA12.4 and NumPy 1.26.4. Octo uses its pinned JAX 0.4.20/Flax 0.7.5/TensorFlow-CPU 2.15.1
model environment. New Torch/NumPy actors need no additional inference dependencies.

R1 used `runtime-r4`; R2 reran every task on the final `runtime-r5` source after registering
the five-profile suite size and formatting the native helper. R1 retains its earlier source label.
The five-step SDK/native matrix passed twice, independently starting each of its nine family
commands. Both complete runs matched all 42 paired records, 210 actions and 252 observations per path.
The R1/R2 semantic records and full traces are also identical, including first actions, returns,
success fields, initial full observations and terminal masks. No tolerance was relaxed. Full-horizon
acceptance remains fifteen historical 100-episode task comparisons, not 42.

### Measured prefix matrix

Generated 2026-11-09T04:18:38.190308+00:00; final label `completion-r2`.

These are traced five-native-tick integration prefixes at K1/G1/N1, one seed per profile.
The original native horizon is retained and both terminal masks remain false. These results
are non-formal and do not add full-horizon accepted tasks. Each family is a separate command.
Family command and evaluation sums are sequential command sums, not one persistent-suite wall time.
SDK/native boundaries and their transports differ; no runtime speedup is inferred.

Collector integrity checks: **PASS** (0 errors).

Final coverage is 42 SDK and 42 native records,
with 210 control steps on each path. Native success is available for
33 prefixes (0 success-once, 0 success at prefix end); 9 are return-only.
Recorded worker failures: 0; OOM events: 0; true terminated/truncated masks: 0 / 0.

### SDK clocks and diagnostic throughput

| Family | Records | Steps | Command s | Evaluation s | Episodes/h command / eval | Steps/s command / eval |
| --- | --- | --- | --- | --- | --- | --- |
| newt | 9 | 45 | 27.670 | 19.934 | 1,170.9 / 1,625.4 | 1.626 / 2.257 |
| hora | 4 | 20 | 19.033 | 13.009 | 756.6 / 1,107.0 | 1.051 / 1.537 |
| trifinger | 5 | 25 | 14.842 | 10.972 | 1,212.7 / 1,640.6 | 1.684 / 2.279 |
| robel | 5 | 25 | 14.239 | 11.149 | 1,264.2 / 1,614.5 | 1.756 / 2.242 |
| unitree | 2 | 10 | 13.893 | 8.304 | 518.2 / 867.1 | 0.720 / 1.204 |
| go2 | 1 | 5 | 10.073 | 5.395 | 357.4 / 667.3 | 0.496 / 0.927 |
| spin | 1 | 5 | 10.557 | 5.967 | 341.0 / 603.3 | 0.474 / 0.838 |
| ppo | 11 | 55 | 31.672 | 23.727 | 1,250.3 / 1,669.0 | 1.737 / 2.318 |
| octo | 4 | 20 | 61.738 | 45.324 | 233.2 / 317.7 | 0.324 / 0.441 |
| Sequential sum | 42 | 210 | 203.717 | 143.780 | 742.2 / 1,051.6 | 1.031 / 1.461 |

### Native clocks and diagnostic throughput

| Family | Records | Steps | Command s | Evaluation s | Episodes/h command / eval | Steps/s command / eval |
| --- | --- | --- | --- | --- | --- | --- |
| newt | 9 | 45 | 22.381 | 15.356 | 1,447.6 / 2,109.9 | 2.011 / 2.930 |
| hora | 4 | 20 | 14.759 | 9.881 | 975.7 / 1,457.4 | 1.355 / 2.024 |
| trifinger | 5 | 25 | 13.678 | 9.221 | 1,316.0 / 1,952.0 | 1.828 / 2.711 |
| robel | 5 | 25 | 12.731 | 9.717 | 1,413.9 / 1,852.4 | 1.964 / 2.573 |
| unitree | 2 | 10 | 14.811 | 8.758 | 486.1 / 822.1 | 0.675 / 1.142 |
| go2 | 1 | 5 | 9.039 | 4.303 | 398.3 / 836.7 | 0.553 / 1.162 |
| spin | 1 | 5 | 8.999 | 4.324 | 400.0 / 832.6 | 0.556 / 1.156 |
| ppo | 11 | 55 | 25.201 | 19.969 | 1,571.4 / 1,983.1 | 2.182 / 2.754 |
| octo | 4 | 20 | 58.775 | 43.169 | 245.0 / 333.6 | 0.340 / 0.463 |
| Sequential sum | 42 | 210 | 180.375 | 124.698 | 838.3 / 1,212.5 | 1.164 / 1.684 |

Native evaluation is the reference's `formal_wall_s` field; the field name does not make a
prefix formal. Native command clocks include interpreter imports, model process startup and
lossless pipe transport. SDK evaluation excludes separately timed Policy loading/preparation.

### SDK resources

| Family | GPU busy command / eval % | VRAM mean / peak GiB | CPU accounted mean / sampled peak cores | RSS mean / peak GiB |
| --- | --- | --- | --- | --- |
| newt | 1.844 / 1.260 | 3.021 / 5.376 | 0.866 / 1.098 | 2.496 / 4.693 |
| hora | 5.107 / 7.936 | 3.693 / 8.042 | 0.863 / 1.070 | 2.559 / 5.536 |
| trifinger | 1.966 / 2.591 | 2.278 / 4.758 | 0.920 / 1.064 | 2.015 / 3.913 |
| robel | 5.088 / 6.136 | 2.212 / 4.363 | 0.879 / 1.032 | 1.663 / 3.372 |
| unitree | 3.036 / 4.750 | 2.072 / 5.372 | 0.828 / 1.056 | 1.742 / 5.039 |
| go2 | 0.200 / 0.190 | 1.508 / 3.810 | 0.793 / 1.046 | 1.147 / 3.283 |
| spin | 0.381 / 0.667 | 1.506 / 3.810 | 0.804 / 1.058 | 1.178 / 3.354 |
| ppo | 5.463 / 6.692 | 3.045 / 5.238 | 0.793 / 1.072 | 2.611 / 4.617 |
| octo | 20.550 / 27.710 | 3.804 / 6.575 | 1.288 / 13.175 | 4.330 / 7.254 |

### Native resources

| Family | GPU busy command / eval % | VRAM mean / peak GiB | CPU accounted mean / sampled peak cores | RSS mean / peak GiB |
| --- | --- | --- | --- | --- |
| newt | 1.534 / N/A | 3.271 / 5.376 | 0.929 / 1.051 | 2.614 / 4.618 |
| hora | 8.810 / N/A | 4.328 / 8.042 | 0.915 / 1.034 | 2.869 / 5.539 |
| trifinger | 3.382 / N/A | 2.594 / 4.758 | 0.931 / 1.045 | 2.150 / 3.879 |
| robel | 7.745 / N/A | 2.586 / 4.363 | 0.923 / 1.036 | 1.892 / 3.329 |
| unitree | 0.690 / N/A | 2.531 / 5.372 | 0.883 / 1.047 | 2.243 / 5.008 |
| go2 | 2.333 / N/A | 1.534 / 3.810 | 0.888 / 1.012 | 1.261 / 3.355 |
| spin | 1.000 / N/A | 1.467 / 3.810 | 0.855 / 1.065 | 1.189 / 3.321 |
| ppo | 7.071 / N/A | 3.533 / 5.238 | 0.904 / 1.069 | 2.979 / 4.581 |
| octo | 25.143 / N/A | 4.053 / 6.575 | 1.269 / 12.256 | 4.534 / 7.317 |

NVML means are device-wide sampled busy, not SM/Tensor/RT throughput. Peaks are sampled
maxima; CPU accounted means and sampled peaks use different explicit measurements.
Native stage-aligned resources are N/A because it emits no SDK stage marker.

### SDK lifecycle and inference

| Family | Load s | Create s | Reset s | Step+obs s | Input staging s | RPC wait s | RPC mean ms | Rollout s | Close s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| newt | 3.773 | 8.245 | 4.274 | 0.642 | 0.004 | 0.187 | 4.158 | 0.848 | 2.056 |
| hora | 2.713 | 5.800 | 2.536 | 0.983 | 0.002 | 0.352 | 17.620 | 1.346 | 1.256 |
| trifinger | 1.485 | 5.234 | 2.121 | 0.604 | 0.002 | 0.111 | 4.450 | 0.723 | 1.078 |
| robel | 0.331 | 5.438 | 2.140 | 0.307 | 0.002 | 0.020 | 0.791 | 0.334 | 1.134 |
| unitree | 1.936 | 4.672 | 1.259 | 0.226 | 0.003 | 0.163 | 16.267 | 0.396 | 0.521 |
| go2 | 1.836 | 3.196 | 0.412 | 0.134 | 0.001 | 0.178 | 35.644 | 0.315 | 0.198 |
| spin | 1.746 | 3.356 | 0.482 | 0.172 | 0.001 | 0.109 | 21.746 | 0.284 | 0.209 |
| ppo | 2.357 | 9.453 | 4.832 | 1.179 | 0.005 | 0.190 | 3.452 | 1.392 | 2.245 |
| octo | 13.460 | 10.659 | 6.262 | 5.107 | 0.034 | 19.452 | 972.610 | 24.634 | 1.019 |

Step+observation, input staging and RPC wait are nested within rollout. CPU wall timings
do not isolate asynchronous GPU execution. The trace-writing overhead is included.

### Native phase boundaries

| Family | Actor load s | Create s | Reset s | Rollout s | Actor pipe round trip s | Close s | Reference end-to-end s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| newt | 3.769 | 8.280 | 4.129 | 0.785 | 0.185 | 1.853 | 21.236 |
| hora | 1.543 | 5.358 | 2.410 | 1.183 | 0.274 | 0.860 | 13.677 |
| trifinger | 1.433 | 5.254 | 2.264 | 0.651 | 0.066 | 0.969 | 12.488 |
| robel | 0.153 | 5.739 | 2.298 | 0.329 | 0.014 | 1.100 | 11.530 |
| unitree | 1.694 | 5.873 | 1.501 | 0.420 | 0.168 | 0.897 | 12.598 |
| go2 | 1.585 | 3.275 | 0.415 | 0.235 | 0.094 | 0.362 | 7.869 |
| spin | 1.718 | 3.355 | 0.450 | 0.246 | 0.078 | 0.191 | 7.993 |
| ppo | 1.895 | 10.447 | 5.464 | 1.365 | 0.145 | 2.282 | 24.037 |
| octo | 11.656 | 10.885 | 6.220 | 24.910 | 19.718 | 0.918 | 57.721 |

Independent GPU actor latency, server queue time, isolated render CPU/GPU duration and
SM/Tensor/RT hardware throughput are N/A. Prefix task failures are not execution failures.

### Per-task records

| Family | Task | Profile | Native H / actual steps | Success once / at prefix end | Prefix return | Truncated |
| --- | --- | --- | --- | --- | --- | --- |
| newt | MS-AntRun-v1 | `ms_ant_run_state` | 1000 / 5 | N/A (return only) | 0.064228646 | false |
| newt | MS-AntWalk-v1 | `ms_ant_walk_state` | 1000 / 5 | N/A (return only) | 4.652080059 | false |
| newt | MS-CartpoleBalance-v1 | `ms_cartpole_balance_state` | 1000 / 5 | N/A (return only) | 4.993438244 | false |
| newt | MS-CartpoleSwingUp-v1 | `ms_cartpole_swing_up_state` | 1000 / 5 | N/A (return only) | 0.348500609 | false |
| newt | MS-HopperHop-v1 | `ms_hopper_hop_state` | 600 / 5 | N/A (return only) | 0.000000000 | false |
| newt | MS-HopperStand-v1 | `ms_hopper_stand_state` | 600 / 5 | N/A (return only) | 0.000000000 | false |
| newt | PickSingleYCB-v1 | `pick_single_ycb_state` | 50 / 5 | 0 / 0 | 0.204430893 | false |
| newt | PlaceSphere-v1 | `place_sphere_state` | 50 / 5 | 0 / 0 | 0.666350126 | false |
| newt | PullCubeTool-v1 | `pull_cube_tool_newt_state` | 100 / 5 | 0 / 0 | 0.538895905 | false |
| hora | RotateSingleObjectInHandLevel0-v1 | `rotate_single_object_in_hand_level0_state` | 300 / 5 | 0 / 0 | 0.258283556 | false |
| hora | RotateSingleObjectInHandLevel1-v1 | `rotate_single_object_in_hand_level1_state` | 300 / 5 | 0 / 0 | 0.384920150 | false |
| hora | RotateSingleObjectInHandLevel2-v1 | `rotate_single_object_in_hand_level2_state` | 300 / 5 | 0 / 0 | 0.381053925 | false |
| hora | RotateSingleObjectInHandLevel3-v1 | `rotate_single_object_in_hand_level3_state` | 300 / 5 | 0 / 0 | 0.352225482 | false |
| trifinger | TriFingerRotateCubeLevel0-v1 | `tri_finger_rotate_cube_level0_state` | 250 / 5 | 0 / 0 | 2.966431618 | false |
| trifinger | TriFingerRotateCubeLevel1-v1 | `tri_finger_rotate_cube_level1_state` | 250 / 5 | 0 / 0 | 2.238461018 | false |
| trifinger | TriFingerRotateCubeLevel2-v1 | `tri_finger_rotate_cube_level2_state` | 250 / 5 | 0 / 0 | 2.939495802 | false |
| trifinger | TriFingerRotateCubeLevel3-v1 | `tri_finger_rotate_cube_level3_state` | 250 / 5 | 0 / 0 | 2.945262432 | false |
| trifinger | TriFingerRotateCubeLevel4-v1 | `tri_finger_rotate_cube_level4_state` | 250 / 5 | 0 / 0 | 1.671286225 | false |
| robel | RotateValveLevel0-v1 | `rotate_valve_level0_state` | 80 / 5 | 0 / 0 | 0.049172748 | false |
| robel | RotateValveLevel1-v1 | `rotate_valve_level1_state` | 150 / 5 | 0 / 0 | 0.049172748 | false |
| robel | RotateValveLevel2-v1 | `rotate_valve_level2_state` | 150 / 5 | 0 / 0 | 0.049172748 | false |
| robel | RotateValveLevel3-v1 | `rotate_valve_level3_state` | 150 / 5 | 0 / 0 | 0.046286553 | false |
| robel | RotateValveLevel4-v1 | `rotate_valve_level4_state` | 300 / 5 | 0 / 0 | 0.054438535 | false |
| unitree | UnitreeG1Stand-v1 | `unitree_g1_stand_root_state` | 1000 / 5 | N/A (return only) | 5.000000000 | false |
| unitree | UnitreeH1Stand-v1 | `unitree_h1_stand_root_state` | 1000 / 5 | N/A (return only) | 5.000000000 | false |
| go2 | UnitreeGo2-Reach-v1 | `unitree_go2_reach_root_state` | 200 / 5 | 0 / 0 | 1.548131824 | false |
| spin | AnymalC-Spin-v1 | `anymal_c_spin_root_state` | 200 / 5 | N/A (return only) | -0.251348883 | false |
| ppo | AnymalC-Reach-v1 | `anymal_c_reach_state` | 200 / 5 | 0 / 0 | 1.617087960 | false |
| ppo | LiftPegUpright-v1 | `lift_peg_upright_ppo_state` | 50 / 5 | 0 / 0 | 1.151947737 | false |
| ppo | PickCube-v1 | `pickcube_state` | 50 / 5 | 0 / 0 | 0.925253153 | false |
| ppo | PokeCube-v1 | `poke_cube_state` | 50 / 5 | 0 / 0 | 0.972863138 | false |
| ppo | PullCube-v1 | `pull_cube_state` | 50 / 5 | 0 / 0 | 1.263951659 | false |
| ppo | PushCube-v1 | `pushcube_state` | 50 / 5 | 0 / 0 | 1.076675296 | false |
| ppo | PushT-v1 | `pusht_state` | 100 / 5 | 0 / 0 | 0.573944926 | false |
| ppo | RollBall-v1 | `rollball_state` | 80 / 5 | 0 / 0 | 0.069395043 | false |
| ppo | StackCube-v1 | `stackcube_state` | 50 / 5 | 0 / 0 | 1.116416693 | false |
| ppo | TwoRobotPickCube-v1 | `two_robot_pickcube_ppo_state` | 100 / 5 | 0 / 0 | 0.095330618 | false |
| ppo | TwoRobotStackCube-v1 | `two_robot_stackcube_ppo_state` | 100 / 5 | 0 / 0 | 0.142150789 | false |
| octo | PutCarrotOnPlateInScene-v1 | `put_carrot_on_plate_in_scene_rgb` | 60 / 5 | 0 / 0 | 0.000000000 | false |
| octo | PutEggplantInBasketScene-v1 | `put_eggplant_in_basket_scene_rgb` | 120 / 5 | 0 / 0 | 0.000000000 | false |
| octo | PutSpoonOnTableClothInScene-v1 | `put_spoon_on_table_cloth_in_scene_rgb` | 60 / 5 | 0 / 0 | 0.000000000 | false |
| octo | StackGreenCubeOnYellowCubeBakedTexInScene-v1 | `stack_green_cube_on_yellow_cube_baked_tex_in_scene_rgb` | 60 / 5 | 0 / 0 | 0.000000000 | false |

`return_only` tasks retain null success fields. The SDK structural boolean is not
counted as failure or success for those tasks. Prefix-end success is not native horizon-end success.

### Trace and record comparisons

| Family | SDK/native records exact | Full prefix traces exact | Actions | Observations including terminal |
| --- | --- | --- | --- | --- |
| newt | True | True | 45 | 54 |
| hora | True | True | 20 | 24 |
| trifinger | True | True | 25 | 30 |
| robel | True | True | 25 | 30 |
| unitree | True | True | 10 | 12 |
| go2 | True | True | 5 | 6 |
| spin | True | True | 5 | 6 |
| ppo | True | True | 55 | 66 |
| octo | True | True | 20 | 24 |

### First/second run repeatability

Rows compare semantic records and every observation/action digest across the two runs.
Timing and resource values are intentionally excluded from equality checks.

| Family | SDK records | Native records | SDK traces | Native traces |
| --- | --- | --- | --- | --- |
| newt | True | True | True | True |
| hora | True | True | True | True |
| trifinger | True | True | True | True |
| robel | True | True | True | True |
| unitree | True | True | True | True |
| go2 | True | True | True | True |
| spin | True | True | True | True |
| ppo | True | True | True | True |
| octo | True | True | True | True |

Raw commands, byte hashes, WorkResults, original family results, sampler records and
trace file hashes are preserved in the companion JSON. The collector performs only reads
of original evidence and writes these summary artifacts outside Git.


## Remaining completion work

### Task IDs outside the selected model candidates

Subtracting the fifteen previously paired tasks and the 27 new model candidates from the
55-task inventory leaves these thirteen task IDs. This is a selected-model scope calculation,
not thirteen remaining full-acceptance gaps: the 27 new candidates still need their own
full-horizon comparisons, so the current full-horizon task gap remains forty.

| Remaining native task ID | Outstanding work |
| --- | --- |
| AssemblingKits-v1 | Pinned assets, selected real policy and matched native evaluation |
| PickClutterYCB-v1 | Pinned assets, selected real policy and matched native evaluation |
| TurnFaucet-v1 | Pinned assets, selected real policy and matched native evaluation |
| OpenCabinetDoor-v1 | Pinned assets, selected real policy and matched native evaluation |
| OpenCabinetDrawer-v1 | Pinned assets, selected real policy and matched native evaluation |
| FMBAssembly1Easy-v1 | Selected compatible real policy and matched native evaluation |
| MS-HumanoidStand-v1 | Selected compatible real policy and matched native evaluation |
| MS-HumanoidWalk-v1 | Selected compatible real policy and matched native evaluation |
| MS-HumanoidRun-v1 | Selected compatible real policy and matched native evaluation |
| PegInsertionSide-v1 | Selected compatible real policy and matched native evaluation |
| PlugCharger-v1 | Selected compatible real policy and matched native evaluation |
| UnitreeG1PlaceAppleInBowl-v1 | Selected compatible real policy and matched native evaluation |
| UnitreeG1TransportBox-v1 | Selected compatible real policy and matched native evaluation |

### Observation and controller profiles remain separate

Task-ID coverage does not validate every configured observation/controller profile. All eighteen
original `tabletop_rgb_v1` profiles remain without a matched trained visual-policy comparison:

```text
assembling_kits_rgb
lift_peg_upright_rgb
peg_insertion_side_rgb
pick_clutter_ycb_rgb
pickcube_rgb
pick_single_ycb_rgb
place_sphere_rgb
plug_charger_rgb
poke_cube_rgb
pull_cube_rgb
pull_cube_tool_rgb
pushcube_rgb
pusht_rgb
rollball_rgb
stackcube_rgb
turn_faucet_rgb
two_robot_pickcube_rgb
two_robot_stackcube_rgb
```

The three legacy state-controller profiles below are also distinct from the PPO-compatible
variants used by the earlier full-horizon comparison. Their task IDs are counted once in the
inventory, but acceptance of the PPO sibling does not establish the legacy profile's model parity.

| Unpaired legacy profile | Legacy controller | Previously paired PPO sibling / controller |
| --- | --- | --- |
| `lift_peg_upright_state` | `pd_ee_delta_pos` | `lift_peg_upright_ppo_state` / `pd_ee_delta_pose` |
| `two_robot_pickcube_state` | `pd_ee_delta_pos` | `two_robot_pickcube_ppo_state` / `pd_joint_delta_pos` |
| `two_robot_stackcube_state` | `pd_ee_delta_pos` | `two_robot_stackcube_ppo_state` / `pd_joint_delta_pos` |

Likewise, the NEWT candidate uses `pull_cube_tool_newt_state` with `pd_ee_delta_pose`; the
canonical `pull_cube_tool_state` remains `pd_ee_delta_pos` and is not covered by that candidate.
The H1, G1, Go2 and Spin transfer candidates use named `*_root_state` profiles that add measured
root kinematics. Their canonical counterparts remain separate:

| Candidate root-state profile | Canonical profile without extra root features |
| --- | --- |
| `unitree_h1_stand_root_state` | `unitree_h1_stand_state` |
| `unitree_g1_stand_root_state` | `unitree_g1_stand_state` |
| `unitree_go2_reach_root_state` | `unitree_go2_reach_state` |
| `anymal_c_spin_root_state` | `anymal_c_spin_state` |

Neither the added root features nor a changed controller are silently attributed to the original
profile. A five-step model prefix does not satisfy the repository's full benchmark-acceptance
matrix. Issue 25 full acceptance is still incomplete.

### Asset recovery findings

The five missing-asset tasks above remain blocked. Final probes of the official UCSD kit and
clutter URLs fail with TLS unexpected EOF. The official `sapien-sim/PartNetMobility` archive
probe at revision `ee0aa3ef1df16181d76d83f7415aa8c94ed1da8f` returns HTTP 401; no download gate
was accepted. These are recorded availability failures, not successful asset installation.

The author-recommended NEWT/MMBench `maniskill.tar.gz` was downloaded completely: 83,590,832
bytes, revision `a59d457df617400d3e45a5158c8deac8a52055b4`, SHA-256
`f3a549868af836197ec445d93b269c2c615f9aee275fed7f50f72b87dfea3393`.
Inspection of all 902 archive members finds only robot and YCB assets, with none of the five
missing task packages. This bundle does not close the blocked-task gap.

The Imitator Game's public PartNet archive exposes raw `mobility.urdf` assets. Its publisher's
loader explicitly switches to those raw URDFs, while the benchmark's
[pinned loader](https://github.com/mani-skill/ManiSkill/blob/baab60ede2e89167c1b7aaed41a9aa8e690a9d1e/mani_skill/utils/building/articulations/partnet_mobility.py)
requires processed `mobility_cvx.urdf` or `mobility_fixed.urdf`. A bounded prefix inspection is
enough to establish that mismatch; the remaining 5.6 GB was not downloaded. No verified original
preprocessing recipe or checksum-equivalent reconstruction was found. Renaming raw URDFs or
generating new convex geometry would introduce unverified benchmark assets, so neither was done.

Raw investigations are retained outside Git in `asset-recovery-findings-20261109.md`,
`asset-final-probes-20261109.json` and the associated archive-inspection JSON files. Different
unverified scene/object assets are not substituted into the frozen protocol.

## Final build and delivery checks

Both clean builders passed. The simulator GPU self-test created/reset vector environments and
rendered nonconstant RGB frames. A real `--quick-test` on final `runtime-r5` completed PickCube's
full 50-tick horizon, with 1/1 success, 11.335 s command wall and 6.952 s evaluation wall. It used the
ordinary manifest, without a diagnostic limit.

Additional K1/G1/N2 model checks covered H1, G1 and Go2, two seeds per profile and five ticks per
lane: all 6 SDK/native pairs, 30 actions and 36 observations matched exactly. This exercises actual GPU
vector observation extraction and recurrent/action-history isolation beyond the N1 matrix.

The separate zero-action environment/render check completed all 81 declared profiles at N2 and five
ticks per lane: 73 passed, 8 failed solely on the five missing asset tasks. It covers 50 of 55 task IDs.
This verifies factory/observation/render paths and does not replace learned-policy validation.

Existing checks: 265 pytest passed; Ruff over source, tests, RoboTwin, ManiSkill and examples passed;
both builders passed `bash -n`/ShellCheck; the Python-support check passed. A new-suite size omission
was detected by an existing regression and fixed before the final build. No tests were added.
Final-head GitHub CI is tracked separately by the PR; missing or pending CI is not reported as green.

The final source snapshot is bound to the PR commit by `candidate-git.json` and
`candidate-source-manifest.json`. `runtime-r5-final-source-verification.json` and the Octo policy
verification compare every installed SDK Python module and executable/protocol source against that
snapshot. Documentation can be completed after measurement; numerical-source changes require reruns.

[Durable raw evidence](https://github.com/QLL-CCDD/RoboScope-Kit/releases/tag/maniskill-issue25-completion-20261109)
contains `issue25-completion-evidence-20261109.tar.gz` plus a checksum JSON. The release body provides
its SHA-256. The archive's `EVIDENCE-MANIFEST.json` hashes every member, including command scripts,
per-task records, both prefix matrices, N2 checks, resource samples, logs, asset failures and source
snapshots. Weights/assets/environments remain outside Git under `/autodl-fs`.

Reproduction on the recorded host starts from the pinned builders and downloaders. The archive's
`run_build_r5.sh`, `after_completion_r1.sh`, `run_completion_r5.py`, `run_completion_vector.py`,
`smoke_profiles_final.py` and `after_completion_r5.sh` preserve exact paths/commands. Read-only
`collect_completion.py --root ROOT` recomputes coverage, metrics and strict repeatability from the
raw files. Source/asset caches are reused only after their committed checksums/provenance checks.
