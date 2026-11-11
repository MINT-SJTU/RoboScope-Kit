# RoboCasa365 five-slot concurrency: bounded epoch2 evaluation

Quick-delivery report draft, 2026-11-11. Intended destination: `benchmarks/robocasa365/expanded-kg-epoch2-20261111.md`. The 19-point screening, native reference and all three full 50×5 arms are complete. Immutable full-arm driver files match the completed ledger and supply the timing and paired-record results below. **The complete 1,023,646,072-byte raw archive and all 2,890 manifested payloads have now passed local size/SHA256 verification, including arrays retained in the archive.** Resource and phase values below come from that verified metadata. Offline captured-array checks attribute all 39 initial feature-hash differences (23 screening plus 16 full) to RGB values with a measured maximum uint8 channel difference of 1 in those captures; five state features and language match exactly. No numerical tolerance or whole-trajectory equivalence is adopted. Completion is established from the final ledger and driver records, not the launch or expected duration. At fixed K3/G2, cold CLI gain is **2.213400×** (formal **2.399221×**). Between the separately selected S5/G1 and P3/G2 configurations, cold gain is **1.416654×** (formal **1.556783×**), below the ≥2× target.

## Runtime contribution, comparison and scope

The performance change runs the five native simulation slots inside each Worker in five spawned slot processes retained for the native group's stepping loop and joins at the existing ordered Policy batch barrier. The processes close and are reaped at group end, then are recreated for the next task/group. It overlaps recurring step/observation work and can overlap construction/reset work. Physics, full RGB observations, two-frame history, FP32/100-DDPM inference, real model weights, actions, seed streams, success scoring and termination rules are unchanged. This is a benchmark-local opt-in; serial remains the default for video/profiling and single-slot smoke. It is not a generic SDK N-slot API.

K/G selects Worker/model counts; within-Worker slot concurrency operates at a separate level. MPS is enabled for every compared arm, so this report isolates no MPS on/off effect. Five slots have an ideal 5× ceiling only for the slot-local component. Whole-run gain is limited by inference, barriers, IPC, contention, startup and task imbalance. Overlapping Worker sums cannot establish an Amdahl wall fraction; no such fraction is fabricated here.

[Issue #97](https://github.com/MINT-SJTU/RoboScope-Kit/issues/97) owns RoboCasa365 and its ≥2× target; #16/#31 supply measurement guidance. The target is assessed from the measured fair comparisons, including a result below 2×. Published historical 2.387× fixed-K/G and 2.402× bounded-autotune results retain their old hardware/source/scope and are not new epoch2 results.

| Contribution or control | Treatment |
|---|---|
| Group-lifetime five-slot processes | Runtime change under test; report fixed-K/G and separately selected results |
| Seeded sampling and EGL corrections | Common corrected baseline; no speed credit |
| Full RGB / original MSAA4 | Retained; RGB skipping is not part of this change |
| Policy/model precision and denoising | Unchanged FP32 / 100 DDPM; no inference-optimization credit |
| K/G screening | Bounded external fixed-CLI search; no SDK built-in autotune claim |
| Storage cleanup and verification-cache release | Before timing; no runtime speed credit |

The user selected quick delivery: one healthy 12-task screening sample per point, provisional selections, three unique full arms, and offline resource-range review. Universal three-repeat sampling, an extra built-in-autotune adapter and repeated GPU frontier expansion were canceled. Targeted repeats are reserved for material uncertainty and are reported only if actually run. No global optimum, maximum physical K, native-horizon model accuracy or general equivalence is claimed.

## Frozen hardware, protocol and inputs

All reported episode step counts and steps/s use executed 20 Hz external control/environment steps, not MuJoCo integration substeps. DDPM denoising steps are a separate model setting.

| Input | Epoch2 setting |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 reporting 49,140 MiB; UUID `GPU-8787399e-27fa-9f9c-c41b-0cbdae308942`; driver 595.71.05 |
| CPU and limits | Xeon Platinum 8470Q; 208 logical CPUs visible, 20-core cgroup quota; memory.high 94 GiB, memory.max 96 GiB, swap.max 0 |
| Hardware identity SHA256 | `0acd715dcc2500b5cae39cef1757a848f56522b86774fab221e6aa22e1ac2813` |
| Suite | Frozen target50: 18 Atomic-Seen, 16 Composite-Seen, 16 Composite-Unseen; all use pretrain scenes |
| Approved comparison size | Five episodes/task, one native five-slot group, shared 128-control-step cap; official 50-rollout default and task horizons remain unchanged |
| Episode streams | Constructor seeds 1111111–1111115; stream episode 0; rollout labels 100000–100004 are identifiers, not reset seeds |
| Robot/control | PandaOmron; 20 Hz; 8-action chunks of 12 values; original action conversion and group termination barrier |
| Observations | Left/right agent-view and wrist RGB, 256×256 uint8; two consecutive control-step frames; raw float32 state and task language; no depth |
| Model / training | Official human300 epoch0500 Diffusion Policy; FP32; 100 DDPM steps; pinned CLIP encoder; real downloaded checkpoint; no Policy training in this campaign |
| Checkpoint SHA256 | `e19011f1db25ca75d5047b5b0ab749592d48005e7222c542faadc0c6df88424c` |
| CLIP weights SHA256 | `a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb` |
| Simulator pins | RoboCasa `3443cdf3dd20f9ae8ed41adba442059b506891d0`; robosuite `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`; same corrected builder-produced runtime on both paths |
| Output and transport | Videos off; JSON Policy encoding; original MSAA 4; no profiling in timed runs; SDK raw output device 79; immutable archives on device 78 |
| Process environment | CUDA device 0; PYTHONHASHSEED 42; OMP, OpenBLAS, MKL and NumExpr threads 1; tokenizer parallelism off; pinned control, simulator and Policy environments remain separate |
| Guard | Predicted admission/actual RAM stop 87 GiB; one-second growth projection stop 91 GiB; device stop 44 GiB; nominal 0.25 s sampling cadence; canonical exclusive GPU lease and strict ownership/cleanup checks |

This is a separate hardware campaign. Old Xeon 6459C/12-core/driver 580/GPU 69daf6c8 results remain historical and are not samples, speed denominators or repeat evidence for epoch2. The historical fixed K2/G1 and bounded built-in-autotune results stay in their own report. Speedups from different configurations or stages are not multiplied.

| Source/experiment identity | Recorded value |
|---|---|
| Baseline executable snapshot | `43df866a3c23e83ee6edade8f68615f3eae1be6c` |
| Candidate executable snapshot | `e836f775fb7c86efde7e5b6046b07d5eb3deeee5` |
| Submitted source heads bound by current measured plan | Integration `b05eead2dce031271f4d1f4e17941a847faf2b96`; parallel `8ff21c187904857385900f1cd8614d2202c8b318` |
| Pre-delivery documentation checkpoint; not measured source labels | PR 100 `24badd3a5d08e21935569c2a8115ae033f3447fe`; PR 107 `2bc6ae0df5d9240463c90f7fc18fcabb986f9c5a`; source correspondence must accompany any documentation update |
| Initial plan SHA256 | `42febf31192478f16bab30b90a64deb638b3700b2d9bdf639f93d97d3a506d2d` |
| Controller SHA256 | `fe2ccc6cf170f83ff6fcb06c459f557bbfde98db83ae5fe0dc0f66e1bab8e250` |
| Measured runner SHA256 | `1051b582c4f3ea897389b8da3e29c6b3f65d7b0d70d8d48e1389b5ea0dc7e5e2` |
| Initial resource plan SHA256 | `169586b630db466872a793d6e85b1e4ecc9217c6d783a7400f6143bb5c46c49a` |
| Source proof SHA256 | `9dc8c6620714b237faa12d9ccf006a3b4c979f2f92285afe75c8ad2444b462a0` |

The [frozen protocol](../../packages/roboscope-kit/docs/robocasa365-protocol.md) and [official Policy example](../../examples/robocasa365_diffusion/README.md) specify all state fields, action conversion, RNG/lifecycle and scoring details; this campaign changes only the approved collection count/cap and slot execution mode. The corrected baseline includes both EGL fixes and the seeded counter-region/stack-base patches. The native reference is the pinned official evaluator on that same corrected runtime, not a claim of identical scenes to an unmodified upstream process. Capped success/step agreement does not establish official-horizon model accuracy or complete trajectory equality.

## Completed native reference and supporting calibration

The new native pass covers all 50 tasks and 250 unique selected episodes: 32,000 executed control steps and 3 successes (1.2%). Every selected episode reaches the 128-step cap. `OpenStandMixerHead` scores 2/5 and `TurnOnElectricKettle` 1/5; other tasks score 0/5. These short-horizon rates are not official-horizon accuracy estimates.

The original WaffleReheat attempt produced five episodes but failed strict cleanup and remains preserved and unselected. An explicit infrastructure-only recovery keeps 47 completed tasks unchanged, reruns WaffleReheat, then completes the remaining two tasks. The full raw history therefore contains 255 executed episodes and 32,640 executed control steps. PID 43324 remains unclassified; the failure is not erased or assigned an unsupported root cause. No outcome-based retry is claimed. Separate terminal-reason labels are not exposed by the upstream score output.

The [published native supporting archive](https://github.com/QLL-CCDD/RoboScope-Kit/releases/tag/robocasa365-epoch2-native-reference-20261111) has 1,313 verified files, 229,340,464 uncompressed bytes, and 255 CRC-checked initial-state ZIPs. Archive SHA256: `30393288e03d331f57b5c9a6c57d81d578959b0cbb442872d8c41ab81c3a774a`; manifest SHA256: `9b4e757ddf681ea336bd4cdc270f89f13b209dbbcb58ee4b5f0fdeb13ced39`. Native is a correctness reference only; its sum of separately started task commands is not a performance denominator.

The normal resource calibrations use RAM stop 80 GiB, one-second projection stop 84 GiB and VRAM stop 42 GiB; they precede the separately reviewed 87/91/44 search policy. The two K2/G2 calibration arms each run CategorizeCondiments, GarnishPancake and PortionHotDogs in that order: 15 episodes and 1,920 executed control steps per arm, 30 SDK episodes total. Both match their 15 native pairs on success, steps, seed/index, initial physics/integration state, scenario metadata and initial observation hashes. This is three-task supporting evidence, not the full 50 SDK comparison.

| Calibration only, nonrankable | Serial K2/G2 | Parallel K2/G2 |
|---|---:|---:|
| Episodes; successes; actual control steps | 15; 0; 1,920 | 15; 0; 1,920 |
| Cold CLI wall, s | 307.6444 | 150.0658 |
| Formal evaluation, s | 260.6943 | 90.3105 |
| Policy load, s | 43.9494 | 57.3171 |
| Runtime preparation, s | 0.8035 | 1.0033 |
| Cold / formal steps/s | 6.2410 / 7.3649 | 12.7944 / 21.2600 |
| Cold / formal episodes/h | 175.5273 / 207.1392 | 359.8421 / 597.9369 |
| Formal GPU busy mean, % | 5.9270 | 16.1330 |
| Formal VRAM mean / peak, MiB | 8,883.00 / 13,539 | 11,583.71 / 21,459 |
| Formal GPU CSV sample count / first-to-last timestamp span, s | 1,041 / 260.449 | 361 / 90.110 |
| Maximum recorded formal GPU timestamp gap, s | 0.303 | 0.257 |
| Maximum observed whole-command host-sample monotonic gap, s | 0.250572 | 0.250653 |
| Sampled whole-command host current peak, GiB | 31.3396 | 53.5551 |
| Sampled whole-command host anon peak, GiB | 25.6116 | 47.3383 |
| high/max/OOM/swap event deltas | 0 | 0 |
| Paired outcome/step/initial-hash disagreements | 0 of 15 | 0 of 15 |

Formal GPU/VRAM means above are arithmetic sample means, not time-weighted continuous averages, of the explicitly selected formal-window CSV rows. The configured cadence is nominally 0.25 s: actual timestamps show the larger maximum gaps reported above. First-to-last spans describe observed samples, not guaranteed continuous coverage or exact stage-edge alignment. Host peaks cover the sampled whole command; their observed monotonic gaps are separate from the GPU timestamp gaps. Full-arm gap/coverage diagnostics are reported below. Do not substitute the separate SDK sampler's window. Calibration times are disclosed for reproducibility and are not a repeated ranking/full-workload speedup claim. Their guarded model loading and small task set differ from the final comparison purpose.

The first forced guard calibration remains failed/incomplete: PID 2244 could not be classified after an environ permission error; the known PID 2081 exit-identity race is retained. Attempt 002 passes observed inner/outer/GPU strict cleanup with no rescue and zero high/max/OOM/swap deltas. Its sampled peak is 40.1155 GiB at a 40 GiB forced trigger; overshoot 0.1155 GiB, post-trigger growth 0.0621 GiB and maximum watcher gap 0.1086 s. Three MPS control timeouts remain in the raw evidence. These observations do not prove a universal allocation/quiescence bound. Twenty-one missing anonymous-memory observations remain null (6 observed zombie states, 15 exit-compatible histories), not zero.

The [native per-task report](native50x5-epoch2-20261111.md) retains exact source, recovery and per-episode accounting. The release tag resolves to `c353d658`; it is a publication anchor, not the measured executable head. The current-hardware reference contains one selected native pass. Historical old-hardware native A/A differences are not evidence for a new-hardware A/A run.

## Verified initial19 screening and provisional selection

The same ordered panel contains CategorizeCondiments, GarnishPancake, PortionHotDogs, ScrubCuttingBoard, OpenStandMixerHead, TurnOnMicrowave, TurnOnElectricKettle, PickPlaceDrawerToCounter, PackIdenticalLunches, PreSoakPan, StoreLeftoversInBowl and SeparateFreezerRack. All 19 admitted points (13 serial, six parallel) completed one healthy 60-episode, 7,680-step sample with three successes. These are repeated executions of the same 60 task/slot identities, not 1,140 distinct scenarios. No screening record is missing or duplicated.

The 34,425,358-byte metadata archive has 1,977 manifested payloads, all size/SHA verified, plus its embedded manifest. Independent queue-record versus new-native checks cover 1,140 pairs: success, steps, constructor seed, episode index, initial physics state, integration state and scenario hashes have zero disagreements. **Initial observation/feature hashes differ in 23/1,140 pairs.** This hash includes all sorted `observation.features`, both strings and arrays, so it does not isolate RGB. The metadata archive excludes arrays; the later complete archive supplies the captured-array checks below. The 23 exact hash differences remain disclosed; there is no four-hash equality claim. The old-hardware native A/A observation differences do not substitute for an A/A measurement on this hardware, and no extra GPU test is introduced by this report.

### Offline attribution of captured initial-feature differences

The verified raw archive preserves the native initial observation NPZ and the SDK first Policy batch. An offline CPU comparison reproduces both recorded full-feature SHA256 values for every one of the 39 mismatching native/SDK pairs, covering all nine sorted features (three images, five state fields and language). Native captures hold one reset frame; the evaluator's canonical uint8/float32 conversion and duplicated two-frame reset history reconstruct the compared SDK feature representation. Those two identical history frames are not two independent observations.

| Scope | Mismatching native/SDK pairs diagnosed | Differing wrist / left image fields | Right image differences | Measured maximum absolute uint8 channel difference | State + language fields compared; dtype/shape/value differences |
|---|---:|---:|---:|---:|---:|
| Screening | 23/23 | 19 / 4 | 0 | 1 | 138; 0 / 0 / 0 |
| Three full arms | 16/16 | 13 / 3 | 0 | 1 | 96; 0 / 0 / 0 |
| Combined captured mismatches | 39/39 | 32 / 7 | 0 | 1 | 234; 0 / 0 / 0 |

Every diagnosed pair differs in one image field only; five float32 state features and the language instruction agree exactly in dtype/shape/value. Per-field canonical payload bytes and SHA256 also match for all 234 nonimage field-pairs, including float32 signed-zero representation, so this is stronger than NumPy numerical equality alone. Counts above are feature-pair counts, not counts of independent repeated history frames. The full-arm mismatches total CloseBlenderLid 9, PackIdenticalLunches 2, CloseFridge 1, SlideDishwasherRack 2 and SeparateFreezerRack 2 across the three arms; per-arm/per-task counts remain in the final table.

The maximum difference of 1 describes these exact captured initial mismatches. It is not an accepted tolerance, proof of MSAA or any other root cause, evidence about later frames, or an official-horizon model-accuracy claim. Model XML and numeric simulator-state/integration arrays are absent from these captures; only the recorded initial-state/integration hashes are available for those comparisons. Outcome and 128-control-step agreement remain a separate finite capped check. No new simulation, inference, native A/A or GPU run was used.

Primary quick selection is the highest observed **cold CLI actual control steps/s** in each mode, one healthy sample per configuration. It provisionally selects **serial K5/G1** and **parallel K3/G2**. Formal rates are reported separately; a different-G formal leader is not silently called inferior for the full workload. The original three-sample-median selector was **not executed**, and its legacy eligibility fields are not completion requirements for this quick scope.

The points come from 288 enumerated resource forecasts under the frozen admission model. They are the tested bounded set, not proof that excluded K/G values are physically impossible. The 12-task panel weights startup and packing differently from 50 tasks; the selected full arms are measurements of these provisional choices, not a complete full-workload search.

### Screening timing and resource table

All values below are one sample per point on the 12-task panel. GPU/VRAM means use formal-window samples; host peaks cover the whole command. Every row has 60 episodes, 7,680 executed control steps and 3 successes.

| Mode | K/G | Cold / formal s | Cold / formal episodes/h | Cold / formal steps/s | GPU busy mean % | VRAM mean / peak MiB | Host current peak GiB | Formal CPU mean cores |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| parallel | 1/1 | 593.229 / 575.811 | 364.11 / 375.12 | 12.9461 / 13.3377 | 11.087 | 6961.23 / 10941 | 31.157 | 3.084 |
| parallel | 2/1 | 344.140 / 326.629 | 627.65 / 661.30 | 22.3165 / 23.5129 | 20.125 | 12663.67 / 20400 | 52.252 | 5.454 |
| parallel | 2/2 | 348.661 / 311.094 | 619.51 / 694.32 | 22.0272 / 24.6870 | 20.255 | 13130.52 / 21459 | 54.467 | 5.652 |
| parallel | 3/1 | 272.131 / 250.618 | 793.74 / 861.87 | 28.2218 / 30.6442 | 26.268 | 18241.11 / 29804 | 73.479 | 6.969 |
| parallel | 3/2 | 268.867 / 231.351 | 803.37 / 933.65 | 28.5643 / 33.1963 | 27.030 | 17657.00 / 30863 | 75.613 | 7.588 |
| parallel | 3/3 | 285.121 / 232.405 | 757.57 / 929.41 | 26.9359 / 33.0458 | 27.026 | 18273.55 / 31922 | 77.715 | 7.603 |
| serial | 1/1 | 1520.870 / 1493.038 | 142.02 / 144.67 | 5.0497 / 5.1439 | 4.372 | 5283.33 / 7441 | 21.333 | 1.185 |
| serial | 2/1 | 814.591 / 791.281 | 265.16 / 272.98 | 9.4280 / 9.7058 | 7.941 | 9228.60 / 13178 | 33.212 | 2.111 |
| serial | 2/2 | 789.837 / 753.825 | 273.47 / 286.54 | 9.7235 / 10.1880 | 8.544 | 10374.21 / 14403 | 35.139 | 2.176 |
| serial | 3/1 | 622.271 / 603.654 | 347.12 / 357.82 | 12.3419 / 12.7225 | 10.804 | 12243.32 / 18781 | 43.964 | 2.694 |
| serial | 3/2 | 626.768 / 592.081 | 344.63 / 364.82 | 12.2533 / 12.9712 | 10.625 | 13209.23 / 19840 | 46.692 | 2.734 |
| serial | 3/3 | 653.308 / 586.887 | 330.63 / 368.04 | 11.7556 / 13.0860 | 11.019 | 14153.60 / 20899 | 47.473 | 2.753 |
| serial | 4/1 | 438.442 / 417.428 | 492.65 / 517.45 | 17.5166 / 18.3984 | 15.802 | 17289.30 / 24447 | 55.131 | 3.840 |
| serial | 4/2 | 471.215 / 410.277 | 458.39 / 526.47 | 16.2983 / 18.7190 | 15.432 | 17855.54 / 25474 | 57.064 | 3.891 |
| serial | 4/3 | 537.736 / 412.823 | 401.68 / 523.23 | 14.2821 / 18.6036 | 15.246 | 18588.15 / 26426 | 59.464 | 3.848 |
| serial | 4/4 | 529.230 / 408.142 | 408.14 / 529.23 | 14.5116 / 18.8170 | 15.617 | 19694.84 / 27484 | 60.544 | 3.893 |
| serial | 5/1 | 412.930 / 383.089 | 523.09 / 563.84 | 18.5988 / 20.0475 | 16.925 | 19524.02 / 30783 | 63.744 | 4.141 |
| serial | 5/2 | 431.694 / 366.564 | 500.35 / 589.26 | 17.7904 / 20.9513 | 16.555 | 19935.84 / 31588 | 65.601 | 4.282 |
| serial | 5/3 | 447.193 / 363.483 | 483.01 / 594.25 | 17.1738 / 21.1289 | 17.302 | 21154.75 / 32508 | 66.599 | 4.339 |

| Mode K/G | Outer runner total s | Policy load s | GPU memory busy mean % / power mean W | Host anon / file peak GiB | Maximum observed host gap ms | Initial feature hash differences /60 |
|---|---:|---:|---:|---:|---:|---:|
| parallel 1/1 | 778.375 | 15.287 | 1.142 / 74.60 | 24.374 / 6.119 | 250.672 | 1 |
| parallel 2/1 | 353.067 | 15.518 | 2.088 / 87.35 | 45.337 / 6.165 | 250.977 | 1 |
| parallel 2/2 | 358.341 | 35.145 | 2.302 / 91.41 | 47.280 / 6.443 | 252.372 | 0 |
| parallel 3/1 | 282.804 | 19.116 | 2.882 / 97.90 | 66.168 / 6.449 | 251.157 | 1 |
| parallel 3/2 | 278.791 | 35.060 | 3.095 / 101.73 | 68.038 / 6.745 | 250.714 | 3 |
| parallel 3/3 | 294.327 | 50.099 | 3.030 / 100.82 | 70.047 / 6.847 | 250.516 | 1 |
| serial 1/1 | 1792.659 | 22.952 | 0.508 / 61.78 | 15.167 / 5.959 | 250.822 | 1 |
| serial 2/1 | 823.943 | 20.391 | 0.842 / 71.86 | 26.256 / 6.241 | 250.863 | 2 |
| serial 2/2 | 799.195 | 32.577 | 0.878 / 73.32 | 28.061 / 6.357 | 254.875 | 2 |
| serial 3/1 | 691.050 | 15.479 | 1.184 / 73.33 | 36.662 / 6.558 | 254.049 | 0 |
| serial 3/2 | 635.494 | 31.705 | 1.156 / 75.64 | 39.270 / 6.662 | 250.965 | 1 |
| serial 3/3 | 662.873 | 62.593 | 1.171 / 75.75 | 39.769 / 6.935 | 252.712 | 2 |
| serial 4/1 | 447.492 | 17.718 | 1.637 / 84.01 | 47.352 / 6.983 | 253.936 | 0 |
| serial 4/2 | 480.790 | 57.385 | 1.695 / 86.21 | 49.178 / 7.082 | 250.775 | 2 |
| serial 4/3 | 547.286 | 121.186 | 1.638 / 85.69 | 51.604 / 7.075 | 250.798 | 1 |
| serial 4/4 | 539.749 | 116.904 | 1.694 / 86.55 | 52.588 / 7.131 | 250.487 | 2 |
| serial 5/1 | 422.601 | 26.555 | 1.788 / 86.23 | 55.763 / 7.158 | 251.507 | 0 |
| serial 5/2 | 441.643 | 61.565 | 1.893 / 88.21 | 57.592 / 7.221 | 252.431 | 2 |
| serial 5/3 | 456.815 | 79.804 | 1.893 / 89.99 | 58.520 / 7.251 | 253.029 | 1 |

The independent initial observation/feature hash differences occur only in SeparateFreezerRack (12 pairs) and PackIdenticalLunches (11). The P3G2/S3G2/S5G1 screening arms have 3/1/0 such differences respectively. Initial physics/integration and scenario hashes still align. Captured-array checks below identify the affected image fields and measured values; they assign no rendering root cause or general numerical tolerance. Upstream score records do not expose separate terminal reasons or native stream-episode fields, so those comparisons are unavailable, not proven equal.

These pointwise rates select configurations provisionally; they are not three-run medians or native-horizon accuracy. The summary's 51 generic unknown entries are native supervisory paths, not SDK trial failures. All 19 SDK trials are healthy, and SDK/global-plan diagnostics are empty. Legacy `selection_eligible_three_samples=false` is retained for traceability and imposes no quick-delivery gate.


### Cold selection versus formal-G frontier

| Diagnostic | Serial | Parallel | Interpretation |
|---|---|---|---|
| Observed cold-CLI provisional choice | K5/G1 | K3/G2 | One screening sample per point |
| Formal-rate leader / rate | K5/G3; 21.1289 steps/s | K3/G2; 33.1963 steps/s | Separate diagnostic, not repeated-median selection |
| Maximum K admitted by this model | K5 | K3 | Forecast/test boundary, not a physical K limit |


The serial formal leader K5/G3 reaches 21.1289 steps/s, but its cold rate is 17.1738 versus 18.5988 for K5/G1; Policy load is 79.804 versus 26.555 s. This explains the cold/formal ranking distinction without proving which G is globally best on 50 tasks. Parallel K3/G2 and K3/G1 have close single-sample cold rates (28.5643 versus 28.2218); no repeatability margin is claimed.

For context, the same **screening-only** samples give the following ratios. They are separate from the completed full50 results and do not establish stable repeated gains:

| Twelve-task, N=1 comparison | Cold CLI throughput ratio | Formal throughput ratio |
|---|---:|---:|
| Fixed K3/G2, S→P | 2.331× | 2.559× |
| Provisional choices S5/G1→P3/G2 | 1.536× | 1.656× |

The measured screening GPU means are 10.63% for S3/G2, 27.03% for P3/G2 and 16.93% for S5/G1. These values do not show GPU saturation or a physical K ceiling; the tested resource forecasts and task supply restrict this bounded study. Recorded formal CPU means are 7.588 cores for P3/G2 and 4.141 for S5/G1, below the 20-core quota. These samples do not show that the quota was fully used, but neither GPU nor CPU means alone identify the bottleneck.

## Three unique full arms — completed timing and paired-record results

The actual quick master was PID 127664/start 1039464560 under root execution review `de9cfc2945f1ad9768633cdfc0a6a66ab08d398727017ee8aec816cdfc3343f0`. Its completed ledger records three `complete` / `launched_new` arms, one physical sample each; the master has exited naturally. The three immutable driver SHA256 values match that ledger. This adds 750 unique arm/episode records and 96,000 control steps: three executions of the same 250 task/slot identities, not 750 distinct scenarios. P3G2 is counted once. The quick helper is `17dfc293b6501623e54d137286121de4bfe802a565dd409dae0abd65000df772`, explicitly descended from `95dbb88c0ff915de95a361aab14e501a9782b5bf418098b6f7bc12ba51b233b4`; the measured fe2 controller/1051 runner and Policy remain frozen.

| Arm | Mode / K/G | Roles | Expected episodes | Verified completed samples |
|---|---|---|---:|---|
| P3G2 | Parallel 3/2 | Provisional selected parallel and common-K/G parallel | 250 | 1; complete, 250 unique |
| S3G2 | Serial 3/2 | Common-K/G serial | 250 | 1; complete, 250 unique |
| S5G1 | Serial 5/1 | Provisional selected serial | 250 | 1; complete, 250 unique |

P3G2 is one physical run serving two roles. Do not duplicate it in episode/campaign accounting.

| Required comparison | Baseline → candidate | Cold CLI gain / time saved | Formal gain / time saved | ≥2× target and paired correctness |
|---|---|---|---|---|
| Isolated runtime at fixed K3/G2 | S3G2 → P3G2 | 2.213400× / 54.8206% (1,086.766583 s) | 2.399221× / 58.3198% (1,124.072968 s) | ≥2× in both intervals; 0/250 outcome or step differences for each arm vs native |
| Independently selected provisional configurations | S5G1 → P3G2 | 1.416654× / 29.4111% (373.171118 s) | 1.556783× / 35.7650% (447.295287 s) | **Below 2×** in both intervals; 0/250 outcome or step differences for each arm vs native |

| Arm | Outer runner total s | Cold CLI s | Formal s | Cold episodes/h; steps/s | Formal episodes/h; steps/s |
|---|---|---|---|---|---|
| P3G2 | 905.292681 | 895.637411 | 803.356558 | 1004.8709; 35.728744 | 1120.2996; 39.832873 |
| S3G2 | 1992.391628 | 1982.403994 | 1927.429527 | 453.9942; 16.142018 | 466.9431; 16.602423 |
| S5G1 | 1277.727424 | 1268.808529 | 1250.651845 | 709.3269; 25.220511 | 719.6247; 25.586657 |

| Arm | Formal GPU busy mean % | Formal VRAM mean / peak MiB | Whole-command host current / anon / file peaks GiB | Formal CPU mean cores / 20-core quota | Ownership / cleanup |
|---|---:|---:|---:|---:|---|
| P3G2 | 31.06330 | 19553.77 / 29885 | 72.66095 / 64.20927 / 8.57585 | 8.32007 / 20; throttled 0 µs | Strict clean; no rescue, no remaining/unproven children; GPU identity audit passed |
| S3G2 | 13.22805 | 14626.18 / 19904 | 50.25219 / 40.76505 / 8.74569 | 3.19399 / 20; throttled 0 µs | Strict clean; no rescue, no remaining/unproven children; GPU identity audit passed |
| S5G1 | 21.03283 | 22193.78 / 30190 | 71.91542 / 62.17342 / 8.93170 | 4.73445 / 20; throttled 0 µs | Strict clean; no rescue, no remaining/unproven children; GPU identity audit passed |

| Additional formal-window GPU samples | Memory busy mean % | Power mean / peak W | SM clock mean / range MHz | Memory clock mean / range MHz | Temperature mean / peak °C |
|---|---:|---:|---:|---:|---:|
| P3G2 | 3.70408 | 109.871 / 145.80 | 2729.43 / 2520–2805 | 10501.00 / 10501–10501 | 43.332 / 48 |
| S3G2 | 1.47000 | 82.591 / 140.45 | 2597.38 / 270–2790 | 10422.44 / 405–10501 | 39.769 / 46 |
| S5G1 | 2.28943 | 92.918 / 126.51 | 2644.38 / 225–2790 | 10485.16 / 405–10501 | 41.793 / 46 |

| Arm | Formal GPU samples / first-to-last span s | GPU max timestamp gap formal / whole s | Host samples whole / formal | Host first-to-last span whole / formal s | Host max monotonic gap whole / formal s |
|---|---:|---:|---:|---:|---:|
| P3G2 | 3207 / 803.127 | 0.297 / 0.297 | 3581 / 3212 | 895.384461 / 803.092373 | 0.251469 / 0.251469 |
| S3G2 | 7700 / 1927.099 | 0.328 / 0.343 | 7926 / 7706 | 1982.149189 / 1927.125221 | 0.253979 / 0.253979 |
| S5G1 | 4996 / 1250.514 | 0.267 / 0.352 | 5073 / 5001 | 1268.555988 / 1250.548145 | 0.253093 / 0.253093 |

Formal GPU rows come from each retained `formal-gpu.csv`; host formal rows use `active_stage=evaluation_s`. All three GPU CSVs and host JSONL streams have zero malformed rows and zero invalid numeric GPU fields; GPU timestamps strictly increase. First-to-last spans are observed coverage, not exact stage edges or a guarantee against unseen gaps. The nominal 0.25 s cadence is not a strict delay bound.

Memory busy is the retained `nvidia-smi utilization.memory` sample (`memory_busy_pct`), separate from allocated VRAM and not a measured bandwidth rate. Power, clocks and temperature use the existing GPU CSV and formal-window summary; these retained samples required no new collection or GPU test. Means are arithmetic sample means, not continuously time-weighted measurements.

| Arm | Unique / expected episodes | Successes / rate | Actual control steps | Outcome/step differences | Initial physics/scenario/feature diagnostics | Failed/timeout/guarded attempts | Missing/duplicate records |
|---|---|---|---|---|---|---|---|
| Native selected | 250/250 | 3/250; 1.2% | 32,000 | Reference | Per-episode hashes retained | One original unselected cleanup failure, explicitly recovered | 0/0 selected |
| P3G2 | 250/250 | 3/250; 1.2% | 32,000 | 0/250 vs native | State/integration/scenario: 0; all-feature hash: 4/250 | 0 / 0 / 0 | 0 / 0 |
| S3G2 | 250/250 | 3/250; 1.2% | 32,000 | 0/250 vs native | State/integration/scenario: 0; all-feature hash: 6/250 | 0 / 0 / 0 | 0 / 0 |
| S5G1 | 250/250 | 3/250; 1.2% | 32,000 | 0/250 vs native | State/integration/scenario: 0; all-feature hash: 6/250 | 0 / 0 / 0 | 0 / 0 |

The fixed-K/G result meets ≥2× for this single matched sample; the separately selected comparison does not. They answer different comparison questions and are not multiplied. Each full arm has zero outcome, step, constructor-seed, episode-index, initial-physics, integration-state or scenario-hash disagreements against its 250 native pairs. Initial all-feature hashes differ in P3G2 4/250, S3G2 6/250 and S5G1 6/250; these are separate from outcome/step agreement. The independent captured-array review above attributes all 16 full differences to image values with maximum absolute channel difference 1 in those captures only. All three driver records report exit 0, no errors, no guard events and no initial cleanup failure. The strict offline summary verifies the downloaded measurement-file hashes and all three final SDK samples are healthy with empty diagnostics. The separate archive verification also covers every binary payload. Its 51 generic `unknown` entries are native supervisory paths, not SDK failures; all 22 SDK samples (19 screening plus three full) are healthy.

Cold CLI wall is Popen-to-exit detection, including model/startup, formal work and in-command outputs/cleanup. Here “cold” names the fresh command interval, not proof of empty OS/GPU caches; no per-arm global cache purge is claimed. It excludes outer preparation/hash work, sampler shutdown and external audits; outer total exposes those costs separately. Formal wall uses the recorded evaluation stage. Compute rates from actual completed counts and the named denominator. The nominal 0.25 s poll/sample interval is not a strict exit-detection-lag or coverage bound. GPU/VRAM means are arithmetic sample means; report actual host monotonic and GPU timestamp gaps/windows. GPU busy alone does not prove saturation or isolate asynchronous inference/rendering execution. Retained temperature/clock indicators are reported above. GPU busy means of 31.06%/13.23%/21.03% and formal CPU means of 8.32/3.19/4.73 cores for P3G2/S3G2/S5G1 do not show sustained GPU saturation or full use of the 20-core quota; they do not isolate the remaining bottleneck.

### Recorded screening phase intervals (not full-workload results)

The following are sums of 12 task-group timers, except model load which is a CLI stage. Parallel Worker/slot intervals overlap. Ratios describe the recorded intervals; they are not additive command-wall contributions. No standalone camera-render or server-GPU timer is available.

| Interval | S3G2 s | P3G2 s | S5G1 s | S3G2/P3G2 interval ratio |
|---|---:|---:|---:|---:|
| Model load | 31.705 | 35.060 | 26.555 | 0.904× |
| Environment construction | 473.954 | 165.800 | 481.000 | 2.859× |
| Group reset | 433.539 | 132.756 | 437.418 | 3.266× |
| Recurring step / observation barrier | 401.195 | 103.377 | 401.425 | 3.881× |
| Policy RPC wait | 188.628 | 190.585 | 259.594 | 0.990× |
| Environment close | 5.629 | 10.178 | 6.002 | 0.553× |

| Screening arm | Task-group elapsed median / P95 s | Group reset median / P95 s | Post-reset group elapsed median / P95 s |
|---|---:|---:|---:|
| S3G2 | 115.008 / 185.860 | 32.908 / 52.898 | 48.349 / 59.272 |
| P3G2 | 47.386 / 70.613 | 8.250 / 24.563 | 23.605 / 28.959 |
| S5G1 | 130.169 / 184.145 | 33.192 / 53.642 | 55.310 / 65.993 |

On this single screening panel, fixed-K/G construction, reset and recurring step/barrier interval ratios are 2.859×, 3.266× and 3.881×. Policy RPC wait is essentially unchanged (S/P interval ratio 0.990×), while parallel close takes longer (ratio 0.553×). The cold whole-command ratio is 2.331× at fixed 3/2 and 1.536× for the separate provisional choices. These comparisons show why phase ratios cannot be multiplied or treated as wall-time fractions; they do not establish full 50-task gains.

Each distribution has 12 task groups, not 60 independent episode wall times. The tiny recorded `autotune_s` stage belongs to fixed-CLI execution overhead; no SDK search/probe route ran. Formal/cold rates above exclude or include startup according to their declared denominator; outer total additionally contains input preparation and audit.

## Phase intervals and overlap — full 50-task arms

CLI stages are wall intervals. The six task-phase values below sum one timer per completed task group (50 groups/arm); Worker/slot intervals overlap. They are not additive contributions to command or formal wall time. In particular, their S3G2/P3G2 ratios cannot be multiplied into a whole-run gain. Camera rendering and server Policy GPU execution have no separately isolated timers.

| Recorded interval, s | S3G2 | P3G2 | S5G1 | S3G2/P3G2 interval ratio |
|---|---:|---:|---:|---:|
| Model loading, CLI wall stage | 51.626382 | 89.922456 | 14.719720 | 0.574121× |
| Runtime preparation, CLI wall stage | 0.805932 | 0.805767 | 1.009272 | 1.000205× |
| Environment construction, overlapping task sum | 1672.713574 | 603.319614 | 1653.930448 | 2.772516× |
| Group reset, overlapping task sum | 1666.820922 | 450.304603 | 1656.137451 | 3.701541× |
| Recurring control step / observation barrier, overlapping task sum | 1607.018660 | 401.401916 | 1589.860278 | 4.003515× |
| Policy RPC wait, overlapping task sum | 741.399747 | 848.382925 | 1114.715409 | 0.873898× |
| Environment close, overlapping task sum | 24.424113 | 40.532230 | 22.098994 | 0.602585× |
| Video writing, task sum | 0.000000 | 0.000000 | 0.000000 | 0 / 0; no ratio |
| CLI time outside named CLI stages | 2.541731 | 1.552241 | 2.427296 | Unassigned command remainder |
| Outer runner minus cold CLI | 9.987634 | 9.655270 | 8.918895 | External preparation, sampler shutdown and audit together |

The unassigned CLI remainder is command wall minus the recorded checkpoint-prepare, fixed-CLI `autotune_s`, model-load, runtime-prepare and evaluation stages. It includes unisolated process/import/output/tail costs and is not an import or result-saving measurement. Checkpoint prepare and fixed-CLI `autotune_s` are each below 0.0003 s/arm; no SDK autotune search/probe ran. Environment close remains a separate overlapping task sum. The outer-minus-CLI interval includes external preparation and verification; it is not an inference speed benefit.

| Full arm, 50 task groups | Task elapsed median / P95 s | Group reset median / P95 s | Post-reset group elapsed median / P95 s |
|---|---:|---:|---:|
| S3G2 | 111.823795 / 149.807669 | 31.787199 / 48.346191 | 46.326427 / 55.154708 |
| P3G2 | 45.785715 / 58.487027 | 7.960226 / 18.012775 | 25.579999 / 29.683283 |
| S5G1 | 119.846135 / 151.552523 | 30.946799 / 48.050601 | 54.916454 / 65.801084 |

Each distribution has 50 completed task groups with no missing/conflicting shared group metadata, not 250 independent episode wall samples. The fixed-K/G cold/formal whole-run gains remain 2.213400×/2.399221×; separately selected gains remain 1.416654×/1.556783×. Longer parallel model loading (89.922456 s versus S3G2 51.626382 s and S5G1 14.719720 s) contributes to the smaller cold gains, but no cause for the load-time difference is assigned from one sample. Existing profiling coverage is unchanged; no new Nsight run or profiling acceptance gate was introduced.

## Offline resource-range and uncertainty review

All full-arm high/max/OOM/oom-kill and swap-event deltas are zero, including the driver's cleanup interval. Sampled swap is zero and CPU throttling is zero. The cgroup's historical `memory.peak` file is unavailable in all host rows; sampled current-memory peaks are not substituted for a continuous or historical maximum. Whole-command file-cache peaks (reported above) rise from the screening panel and remain part of cgroup current. Do not sum separately timed current/anon/file maxima as a simultaneous memory decomposition.

| Full arm | Frozen host / VRAM forecast GiB | Sampled whole-command host / device VRAM peak GiB | Margin to 87 GiB host stop / 44 GiB device stop | Margin to 94 GiB memory.high |
|---|---:|---:|---:|---:|
| P3G2 | 82.796176 / 31.900000 | 72.660946 / 29.184570 | 14.339054 / 14.815430 | 21.339054 |
| S3G2 | 58.000000 / 21.400000 | 50.252186 / 19.437500 | 36.747814 / 24.562500 | 43.747814 |
| S5G1 | 81.600000 / 32.700000 | 71.915421 / 29.482422 | 15.084579 / 14.517578 | 22.084579 |

These margins are differences from sampled peaks, not bounds on unobserved bursts, admission safety or future tasks. The 91 GiB one-second projection stop is a retained guard rule, not a measured allocation-rate upper bound.

| Arm | Formal PSI some memory / CPU / I/O, % wall | Largest adjacent positive RAM jump GiB / interval s | Largest adjacent positive RAM rate GiB/s | Largest retained guard projection GiB |
|---|---:|---:|---:|---:|
| P3G2 | 0.00000772 / 0.01565100 / 0.12315980 | 2.699192 / 0.250095 | 10.792687 | 74.786241 |
| S3G2 | 0.00000000 / 0.00759359 / 0.00364159 | 0.802620 / 0.250102 | 3.209175 | 50.799669 |
| S5G1 | 0.00000032 / 0.00695279 / 0.00575268 | 1.136143 / 0.250099 | 4.542769 | 72.494911 |

Ramps above use originally adjacent host samples both labeled evaluation; no excluded stage is bridged. The same largest jumps/rates also occur in the whole-command windows. Guard projection is the original runner field and is not reconstructed from the largest adjacent rate. Observed rates and sampling gaps are empirical diagnostics, not physical limits or allocator/GPU quiescence guarantees.

Task boundaries are matched by exact Worker PID/start identity and task-attempt UID. Each arm has 50 matched completed boundaries and all requested Workers and G model routes; maximum sampled spawned slots are P3G2 15, serial arms 0. All 150 after-task snapshots contain zero spawned slots and zero zombies; the parallel slot group is closed before the next task, while the Worker and its resource tracker can persist.

| Arm | Complete observed Worker-subtree anon peak range GiB | After-task Worker-parent anon range GiB | Host process-anon null observations (R / Z state) |
|---|---:|---:|---:|
| P3G2 | 20.792671–21.976204 | 0.117840–0.122032 | 240 (165 / 75) |
| S3G2 | 12.541889–12.823380 | 7.082359–10.854507 | 27 (26 / 1) |
| S5G1 | 11.900230–12.589489 | 7.097935–10.659859 | 25 (19 / 6) |

Subtree sums include each sampled Worker and all observed descendants only when every anonymous field is known. Shared pages can be counted more than once, so these are not unique cgroup allocations. Missing process-anonymous fields remain null; R/Z labels alone do not explain their cause or assign ownership. Serial retention is visible after tasks; a finite 50-task history still does not prove an unbounded retention plateau. Final +5 s checks are strict-clean in all arms with no remaining or unproven descendants, no rescue and passed GPU identity audits. P3G2’s child subreaper also reaped three exited resource trackers (exact PID/start identities observed earlier under their Workers), each with wait status 0; serial arms record no adopted-child reap. These exited children are preserved in the driver history.

The reviewed model enumerates 288 rows (144 per mode, K/G 1–12). It admits 13 serial and six parallel points, all tested once in the screening panel. The 11.5 GiB base floor, 2 GiB global uncertainty, 13.1 GiB × K serial Worker prior, 2.5/2.6 GiB cold/formal model priors, and serial/parallel graphics priors of 6.25/9.75 GiB per Worker are retained. Parallel simulation memory uses the largest distinct-task envelopes rather than the serial 13.1×K formula; that asymmetric empirical prior is disclosed, not a matched proof of a physical maximum.

| Untested boundary | Frozen host / device VRAM forecast GiB | Bounded disposition |
|---|---:|---|
| Serial K5/G4 | 89.400000 / 36.300000 | Host forecast above 87; untested risk deferral |
| Serial K6/G1 | 94.700000 / 38.950000 | Host forecast above 87; untested risk deferral |
| Parallel K4/G1 | 100.682436 / 40.450000 | Host forecast above 87; untested risk deferral |

For the full 144-row serial grid, 97 rows exceed the host forecast and 81 exceed the VRAM forecast; parallel counts are 118 and 105. Each mode also has 66 G>K rows with unused replicas under the unchanged static routing. Reason counts overlap, so they are not a partition. K≥13 is forecast beyond the 44 GiB device guard from the retained positive graphics coefficients even at G1; those families were not tested. CPU20 is not imposed as K, G or 5K≤20, and K need not equal G.

The completed full-arm observations stay below these retained forecasts but do not justify lowering full-workload priors, prove the excluded frontier physically impossible, establish maximum physical K or establish a global optimum. This finishes the finite offline resource review of the authorized scope. No new predictor, frontier round, built-in-autotune adapter or GPU run is required. There is one full sample per unique arm, so no three-run median or repeatability range exists; old hardware samples are not pooled and gains are not multiplied. A later targeted repeat would require a material, identified ambiguity and a separately recorded identity, not a mandatory universal-repeat gate.

## Preservation, evidence and reproduction

Before screening, one obsolete duplicate asset extraction was removed after byte equality and source-membership checks, freeing 13,146,849,280 bytes. Current assets, source, model and raw evidence remained intact; historical paired-v3 replay needs that extraction rebuilt. A separate leased release of verification-read pages occurred before timing. Neither maintenance step receives runtime speed credit. All new measured arms retain output device79; archives use device78.

The [native reference](native50x5-epoch2-20261111.md) and [public native raw release](https://github.com/QLL-CCDD/RoboScope-Kit/releases/tag/robocasa365-epoch2-native-reference-20261111) are available. The initial metadata and complete quick-R2 raw archives are locally verified. The final quick archive is 1,023,646,072 bytes with 2,890 size/SHA256-verified payloads (3,001,145,341 logical bytes; 2,891 TAR members including the manifest). All 2,348,086,164 binary bytes were streamed and verified in the original archive; only 653,059,177 metadata bytes were extracted for local analysis. Arrays, failed native/guard attempts and all screening/full records remain preserved. The attempted full raw attachment upload returned HTTP 408; the complete archive and manifests remain independently verified locally and in persistent storage, but no public full-archive download is claimed at this checkpoint; captured-array diagnostics are complete as reported above. Metadata-only delivery is not substituted for arrays. Retain the native failure, all screening/full failed or partial records, commands, source/hardware/model identities, arrays and per-file manifests externally; commit necessary Markdown only.


| Verified input / scope | SHA256 |
|---|---|
| Initial metadata archive; arrays omitted | `4f9a7b54329c1a02822eb1aa0087210bc4ce600a8344171d2da985019dcaadfb` |
| Initial per-file manifest | `4699e017a40d89c3c8c08848d3935898d3592ead96ce75d365f4fa52b0267bae` |
| Strictly verified initial19 summary | `01edbb8f48a931883db49d599037d970f1f2232d21cc0b08b0a4a61b68286435` |
| Independent initial19/native diagnostics | `bc2aab1e89342b1cc45fe939ca7af1b30931a52396064892cf3dcf1074e4ccd9` |
| Derived initial19 rates/resources | `07290708ff81cc796381e3a351be34d53d16a4dbff3d96891b97880929da4e01` |
| Actual quick execution review | `de9cfc2945f1ad9768633cdfc0a6a66ab08d398727017ee8aec816cdfc3343f0` |
| Completed P3G2 driver, ledger-bound | `a4aeb92371fbd3f0f1a76427483641102e533661f20c6a791f18240282e8bfb9` |
| Completed S3G2 driver, ledger-bound | `d08c26e791dcb5fe9873ab20c2d3589faa675b7fa9a01ace92617980330d6441` |
| Completed S5G1 driver, ledger-bound | `eea59f1754c46c0dbb6f86c8e8d9a5a2b8f7456cfd6c52def30a94806ddc8a98` |
| Native selected summary | `0566abfba53ff748166646a8c389809208dc5a2b8b6850044462f92d5d1972fa` |
| Complete quick raw archive | `72f83a06e1315d3496c70ef86a6d35fb2b7b9d563198ee000400e2e3edfd50f0` |
| Complete quick raw per-file manifest | `99bf0a77f56e85775946b7c5e337f195f8e4b1185617f9d25ecac8507d7fb7da` |
| Complete quick raw collection receipt | `a239e0f63c33ac5a6ef428b5e8ef330ba08219f7e8d50196f6b069165789097b` |
| Strict offline complete SDK summary | `ec0a84c4cb9e9f1fb43c3ab487df5ad46de545538b28fafec88a19c91a1229fa` |
| Offline full resource/coverage diagnostics | `c9e3a9c9ee6c974244537ea444387a8263cc1d706bcc7293e5edfe8c5f56f63c` |
| Captured initial-feature array diagnostics (all 39 mismatches) | `d1ce3d195cd53b62a17eabe70cc18020dd6cae32484b99dec343d3149ac9258a` |
| Local all-member raw verification | `92e66ee3dcd7145c5b492413bbec39c4aa992284739b36c56fb31a9305d3178b` |

These are external evidence identities, not JSON files to add to Git. The table retains the complete quick raw archive and all-member verification identities; the full raw public transfer failed with HTTP 408 and its public download remains outstanding. An empty release page is not evidence availability.

The initial launched command remains in `epoch2-initial-launch.json`; the quick execution review records actual provisional selection, input hashes and the three-role union. Reproduction pointers must use the final actual launch/quick raw bundle when available; this report does not invent an argv or invoke a duplicate live job. Recorded current PR heads are 24badd3a/2bc6ae0d, with 252/253 non-Markdown blobs unchanged. Their exact-head SDK compatibility runs 43001504530/43001521809 were inspected as action_required, not green. Existing Linux 303-case results retain their tested-source attribution. No code/version change, merge or complete acceptance is claimed by this draft.

## Per-task success and step vectors — completed full arms

Vectors are ordered native slots 0–4; `1` means success. Every selected native and SDK episode executed 128 external control steps, so each task totals 640 per arm. The cells contain the actual five-bit success vector and five control-step counts. Outcome, steps, constructor seed, episode index, initial physics/integration state and scenario hash agree in all 750 SDK/native pairs. The last column counts all-feature hash differences in S3G2/P3G2/S5G1 order, out of five native pairs per arm. A feature hash includes strings and arrays; the separate captured-array checks above attribute all observed mismatches to images, with measured maximum absolute channel difference 1 only in those captures. No general tolerance is adopted.

| Task / split | Native success bits; step vector | S3G2 bits / step vector | P3G2 bits / step vector | S5G1 bits / step vector | Feature hash differences S3/P3/S5, /5 |
|---|---|---|---|---|---|
| CloseBlenderLid / Atomic-Seen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 3 / 3 / 3 |
| CloseFridge / Atomic-Seen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 1 / 0 / 0 |
| CloseToasterOvenDoor / Atomic-Seen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| CoffeeSetupMug / Atomic-Seen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| NavigateKitchen / Atomic-Seen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| OpenCabinet / Atomic-Seen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| OpenDrawer / Atomic-Seen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| OpenStandMixerHead / Atomic-Seen | `00011`; `128,128,128,128,128` | `00011`; `128,128,128,128,128` | `00011`; `128,128,128,128,128` | `00011`; `128,128,128,128,128` | 0 / 0 / 0 |
| PickPlaceCounterToCabinet / Atomic-Seen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| PickPlaceCounterToStove / Atomic-Seen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| PickPlaceDrawerToCounter / Atomic-Seen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| PickPlaceSinkToCounter / Atomic-Seen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| PickPlaceToasterToCounter / Atomic-Seen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| SlideDishwasherRack / Atomic-Seen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 1 / 0 / 1 |
| TurnOffStove / Atomic-Seen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| TurnOnElectricKettle / Atomic-Seen | `00100`; `128,128,128,128,128` | `00100`; `128,128,128,128,128` | `00100`; `128,128,128,128,128` | `00100`; `128,128,128,128,128` | 0 / 0 / 0 |
| TurnOnMicrowave / Atomic-Seen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| TurnOnSinkFaucet / Atomic-Seen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| DeliverStraw / Composite-Seen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| GetToastedBread / Composite-Seen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| KettleBoiling / Composite-Seen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| LoadDishwasher / Composite-Seen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| PackIdenticalLunches / Composite-Seen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 1 / 1 |
| PreSoakPan / Composite-Seen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| PrepareCoffee / Composite-Seen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| RinseSinkBasin / Composite-Seen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| ScrubCuttingBoard / Composite-Seen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| SearingMeat / Composite-Seen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| SetUpCuttingStation / Composite-Seen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| StackBowlsCabinet / Composite-Seen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| SteamInMicrowave / Composite-Seen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| StirVegetables / Composite-Seen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| StoreLeftoversInBowl / Composite-Seen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| WashLettuce / Composite-Seen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| ArrangeBreadBasket / Composite-Unseen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| ArrangeTea / Composite-Unseen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| BreadSelection / Composite-Unseen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| CategorizeCondiments / Composite-Unseen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| CuttingToolSelection / Composite-Unseen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| GarnishPancake / Composite-Unseen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| GatherTableware / Composite-Unseen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| HeatKebabSandwich / Composite-Unseen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| MakeIceLemonade / Composite-Unseen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| PanTransfer / Composite-Unseen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| PortionHotDogs / Composite-Unseen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| RecycleBottlesByType / Composite-Unseen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| SeparateFreezerRack / Composite-Unseen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 1 / 0 / 1 |
| WaffleReheat / Composite-Unseen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| WashFruitColander / Composite-Unseen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
| WeighIngredients / Composite-Unseen | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | `00000`; `128,128,128,128,128` | 0 / 0 / 0 |
