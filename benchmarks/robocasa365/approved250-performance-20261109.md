# RoboCasa365 all-task performance comparison, 2026-11-09

The approved **50-task × 5-episode, 128-control-step-cap** workload completes
with **2.387× formal completed-episode throughput at fixed K=2/G=1** and
**2.402× after independent bounded autotune**, with CUDA MPS enabled throughout.
Every SDK arm completes 250 unique episodes and 32,000 actual physical steps,
with the same three successful episode identities. All paired success and
terminal-step results match both fresh native passes.

This reaches issue #97's **≥2× target on this approved reduced-cap workload**.
The committed official horizons and 50-episode default are unchanged. The result
is not a measurement of native-horizon accuracy or full-horizon speed, nor a
claim of coverage of all 365 tasks. The approved count/cap does not acquire new
2,500-episode, perfect-trajectory-hash or indefinite-memory-plateau gates.
[Native A/B and integration coverage](approved250-native-20261109.md) are reported
separately from the SDK-internal speedup.

## Change and expected scope of benefit

The change runs each Worker's five native simulation slots in persistent spawned
processes. Reset and every physical-step block proceed concurrently, then join
at the existing ordered five-slot Policy batch barrier. Policy code, inference
precision, denoising steps, observations, actions, success checks and termination
rules are unchanged. **All RGB observations remain enabled; RGB skipping is not
part of these timings.** Serial is still the default and is used for video or
profiling runs.

A five-slot-local segment has an ideal concurrency ceiling of 5× before IPC,
barriers, contention and unequal slot cost. That is not a 5× whole-evaluation
prediction: Policy RPC and other serial work remain. The measured phase changes
below show gains in both repeated stepping and reset/construction, rather than
only a one-time startup improvement.

The three concurrency layers are complementary: autotune chooses the number of
Workers K and Policy replicas G; slot concurrency overlaps the five native
environments inside a Worker; CUDA MPS permits CUDA clients to share the GPU.
MPS does not itself parallelize the Python simulator loop or imply faster EGL
rendering. This implementation is benchmark-local, not a generic N-slot SDK API.
All arms have MPS on, so this experiment does not isolate an MPS on/off gain.

## Frozen conditions and provenance

Measurements were collected on 2026-11-09 and reviewed on 2026-11-10.

| Input | Matched setting |
| --- | --- |
| Hardware | RTX 4090 reporting 49,140 MiB; NVIDIA driver 580.105.08; Xeon Gold 6459C |
| Limits | 12-core container quota; memory.max 96 GiB; memory.high 94 GiB; swap.max 0 |
| Workload | Frozen target50: 18 Atomic-Seen, 16 Composite-Seen, 16 Composite-Unseen; pretrain scenes |
| Per task | One native group of five episodes; constructor seeds 1111111–1111115; stream episode zero |
| Model | Official human300 epoch0500 Diffusion Policy; real checkpoint; FP32; 100 DDPM steps |
| Cameras/output | Full RGB, original MSAA=4, videos off; no heavy step tracing in SDK timing runs |
| Fixed configuration | K=2/G=1 for serial and parallel |
| Tuning | Independent diagonal searches, elapsed estimator, K≤2/G≤1, 800 s per candidate, 1800 s total budget |
| Execution order | Native A, Native B, fixed serial, fixed parallel, tuned serial, tuned parallel; exclusive GPU lease |
| SDK output backing | All four arms use the same local filesystem via recorded output-directory symlinks |

The serial executable snapshot is `43df866a3c23e83ee6edade8f68615f3eae1be6c`,
corresponding to submitted integration `0b876848d10f10536a457e1bb499e01a3b9f4532`
(version 0.8.0b34). The parallel snapshot is
`e836f775fb7c86efde7e5b6046b07d5eb3deeee5`, corresponding to submitted
`aeb9c58106816b27e490c75356c9f8f7376445b8` (0.8.0b35). The file-by-file proof
covers all 127/128 runnable SDK, test, benchmark and model files/pins. Final
submitted documentation heads are recorded by the evidence's source
correspondence verification; these snapshot identifiers are not relabelled.

The same clean, checksum-verified runtime built from the committed builder is
used on both paths. It includes the two EGL ownership/finalization corrections
and two seeded-sampling corrections. Native comparisons therefore use the
pinned official evaluator on this same corrected runtime, not an unmodified
upstream runtime. The import proof also verifies 6,599 installed payload files
across eight simulator dependencies byte for byte.

| Artifact | SHA256 |
| --- | --- |
| Human300 checkpoint | `e19011f1db25ca75d5047b5b0ab749592d48005e7222c542faadc0c6df88424c` |
| CLIP weights | `a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb` |
| SDK configuration | `5e64c67750c4a475e266cbc92ad58cf74f417d1eb27bf14bd8262aa83d75c7bf` |
| Scenario plan | `082a66a08f5a066f4903419f381796b811dff88a6121391233f13e5750fc8aab` |

## Formal throughput and total command time

Formal throughput is completed scored episodes divided by the CLI's formal
`evaluation_s`. Actual-step throughput uses recorded physical steps, not
horizon multiplication. Native per-task subprocess startup totals are never a
speedup denominator for this SDK-internal comparison.

| Arm | K/G | Formal (s) | Episodes/h | Actual steps/s | Tuning (s) | Whole command (s) | Successes | Actual steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed-serial | 2/1 | 2,449.99 | 367.3 | 13.06 | 0.00 | 2,493.48 | 3/250 | 32,000 |
| fixed-parallel | 2/1 | 1,026.59 | 876.7 | 31.17 | 0.00 | 1,044.07 | 3/250 | 32,000 |
| tuned-serial | 2/1 | 2,401.50 | 374.8 | 13.33 | 196.14 | 2,601.17 | 3/250 | 32,000 |
| tuned-parallel | 2/1 | 999.67 | 900.3 | 32.01 | 83.46 | 1,085.81 | 3/250 | 32,000 |

| Pair | Formal episode-throughput ratio | Step-rate ratio | Whole-command ratio |
| --- | --- | --- | --- |
| fixed | 2.387× | 2.387× | 2.388× |
| tuned | 2.402× | 2.402× | 2.396× |

Each arm has one successful attempt, no timeout or Worker failure, no retry,
no planned recycle and no OOM recovery. Expected/completed coverage is 250/250;
missing, duplicate and unknown task/episode identities are all zero. The same
32,000 physical steps are retained, so neither ratio comes from dropped work.

Whole-command time starts before sampler/CLI process startup and ends after
process/sampler shutdown. It excludes the separate five-second descendant
cleanup observation, earlier downloads/builds, the native passes and experiment
preparation. Result writing is included in the command; its duration is not
independently isolated.

### Independent bounded autotune

Each search actually evaluates K=1/G=1 and K=2/G=1; G is capped at one, so there
is no G>1 trial. Both choose K=2/G=1 and stop at the identical caps. This measures
what remains after the same bounded search, not a global optimum across larger
K/G or different memory budgets. Probe episodes are additional executed tuning
work and do not count toward the 250 formal identities.

| Arm | Candidate K/G | Decision | Elapsed (s) | Completed/expected probes | Successes | Worker failures / OOM | Requested/final Workers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tuned-serial | 1/1 | accepted | 107.04 | 5/5 | 0 | 0 / 0 | 1/1 |
| tuned-serial | 2/1 | accepted | 89.09 | 10/10 | 0 | 0 / 0 | 2/2 |
| tuned-parallel | 1/1 | accepted | 45.41 | 5/5 | 0 | 0 / 0 | 1/1 |
| tuned-parallel | 2/1 | accepted | 38.05 | 10/10 | 0 | 0 / 0 | 2/2 |

Each search completes 15 probe episodes, with zero probe successes. The
controller's `progress.failures` values 5 and 10 count unsuccessful episodes,
not execution failures; candidate health reports zero Worker failures/OOM.
Probe physical-step counts are not exported by selection metadata and are
left unrecorded rather than reconstructed from the cap. Both selection statuses
are `complete`; both stop reasons are `worker cap 2 reached; replica cap 1 reached`.

## Where the time changes

Worker phase values below are sums of elapsed intervals across 50 task groups.
Workers overlap, so these sums cannot be added to obtain evaluation wall time.
Step/barrier time includes simulation, native observation/rendering and slot
IPC; Policy RPC includes transport, queue waiting and inference completion, not
isolated GPU kernel time.

| Worker phase | Fixed serial sum (s) | Fixed parallel sum (s) | Serial/parallel ratio | Tuned serial sum (s) | Tuned parallel sum (s) |
| --- | --- | --- | --- | --- | --- |
| Environment construction | 1,453.71 | 517.23 | 2.811× | 1,431.18 | 509.27 |
| Episode reset | 1,438.48 | 390.33 | 3.685× | 1,418.58 | 380.79 |
| Native step / observation / barrier | 1,327.96 | 340.34 | 3.902× | 1,296.19 | 327.77 |
| Policy RPC wait | 633.38 | 740.60 | 0.855× | 622.11 | 715.28 |
| Environment close | 20.61 | 35.54 | 0.580× | 20.75 | 34.83 |
| Video write | 0.00 | 0.00 | — | 0.00 | 0.00 |

At fixed K/G, construction is **2.811×**, reset **3.685×**, and the repeated
step/observation/barrier segment **3.902×** faster by these elapsed sums.
Policy RPC grows **16.93%**, and close time grows **72.47%**. These increased
costs are included in the measured overall gain; model inference was not
optimized or counted as a separate contribution. Formal CPU work is
5,367.79→5,773.11 core-seconds (+7.55%): the implementation improves elapsed
completion time by overlapping slot work, while doing slightly more aggregate
CPU work.

| Arm | Checkpoint prep (s) | Runtime prep (s) | Policy load stage (s) | Autotune (s) | Formal (s) | Other command (s) |
| --- | --- | --- | --- | --- | --- | --- |
| fixed-serial | 0.0002 | 0.804 | 39.664 | 0.00 | 2,449.99 | 3.015 |
| fixed-parallel | 0.0002 | 0.817 | 14.752 | 0.00 | 1,026.59 | 1.904 |
| tuned-serial | 0.0003 | 0.803 | 0.001 | 196.14 | 2,401.50 | 2.733 |
| tuned-parallel | 0.0002 | 0.804 | 0.001 | 83.46 | 999.67 | 1.874 |

Checkpoint preparation resolves already-downloaded weights. Policy-load time
varies between the fixed commands; it is outside the formal speedup denominator
and is not attributed to slot concurrency. Autotune loads/probes the Policy and
retains the usable replica, explaining its later near-zero Policy-load stage.
Other time includes interpreter/CLI work, final outputs and sampler shutdown.
These are process-cold starts on a previously prepared machine, not controlled
cold filesystem-cache experiments. Model GPU execution, rendering alone,
serialization alone and output/cleanup alone were not separately isolated.

### Task and group latency, and repeated-configuration variation

A task contains exactly one five-episode native group. All five slots share the
Policy/group barrier, so the following are task/group wall-time distributions,
not independent per-episode latency samples. The reset-inclusive group interval
excludes task environment construction/close; task time includes them. Post-reset
execution is the raw record's `metadata.group_elapsed_s`: it excludes environment
reset/construction and includes the Policy lifecycle, stepping and batch barriers.
Each distribution has 50 group samples, not 250 independent episode samples.
P95 uses nearest rank.

| Arm | Task min / P50 / P95 / max (s) | Reset-inclusive group min / P50 / P95 / max (s) | Post-reset execution P50 / P95 (s) |
| --- | --- | --- | --- |
| fixed-serial | 79.04 / 94.87 / 126.42 / 155.41 | 56.33 / 67.60 / 82.40 / 90.13 | 38.402 / 46.965 |
| fixed-parallel | 31.66 / 39.89 / 52.09 / 61.67 | 23.32 / 29.66 / 37.41 / 42.00 | 21.553 / 25.788 |
| tuned-serial | 79.93 / 93.57 / 124.64 / 154.13 | 56.47 / 65.76 / 81.36 / 89.51 | 37.519 / 45.223 |
| tuned-parallel | 32.60 / 38.48 / 50.45 / 61.13 | 23.10 / 28.34 / 35.40 / 40.97 | 21.086 / 24.425 |

The post-reset median falls **38.402→21.553 s** at fixed K/G; the measured
gain therefore persists after environment construction/reset are excluded. This
segment does not accelerate by the native-step-only 3.902× because Policy/RPC
and barrier costs remain. These short-run timings do not predict a numeric
full-horizon speedup.

There is one full fixed pair and one full independently tuned pair. The formal
serial observations are 2449.99/2401.50 s and the parallel observations are
1026.59/999.67 s; tuning/probes and cache/order context differ. They are not three
independent full-workload repeats or a confidence interval. Earlier atomic
three-pair and four-composite pilots remain separate evidence in
[slot concurrency](slot-concurrency-20261109.md) and the
[composite follow-up](composite-reset-followup-20261109.md); their ratios are not
multiplied into this result.

For the same task, execution path and chosen K/G, the tuned/fixed ratio of
step/barrier time per actual physical step has median/P95/range
**0.976/1.037/0.908–1.057** in serial and
**0.962/1.044/0.856–1.048** in parallel. Task complexity, assignment, cache and
ordering remain confounders. Later complex tasks taking longer is not evidence
that memory caused a slowdown.

## GPU, CPU and host memory

Device-wide GPU samples are taken every 250 ms and include inference and EGL
rendering. Host samples are taken every 0.5 s. CPU cores below come from formal
cgroup CPU-use deltas over the sampled interval. Process counts and RSS peaks
cover the whole command, including tuning. Summed process RSS can double-count
shared pages; cgroup current is the container-budget measure.

| Arm | CPU mean cores | GPU busy mean / P95 / peak (%) | VRAM mean / peak (MiB) | Memory-controller busy mean (%) | Power mean (W) | Peak processes | Peak RSS sum (GiB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed-serial | 2.19 | 10.42 / 38 / 100 | 9,433 / 13,350 | 1.11 | 72.81 | 7 | 28.22 |
| fixed-parallel | 5.63 | 25.81 / 58 / 94 | 12,906 / 20,090 | 2.90 | 96.70 | 23 | 46.20 |
| tuned-serial | 2.19 | 10.81 / 39 / 97 | 9,436 / 13,350 | 1.15 | 72.07 | 7 | 28.56 |
| tuned-parallel | 5.62 | 25.81 / 58 / 94 | 12,846 / 20,090 | 2.91 | 98.44 | 23 | 46.12 |

| Arm | SM clock mean / range (MHz) | Memory clock mean / range (MHz) | Temperature mean / peak (°C) | Formal GPU / host samples |
| --- | --- | --- | --- | --- |
| fixed-serial | 2,498 / 210–2,775 | 10,220 / 405–10,501 | 35.78 / 39 | 9766 / 4791 |
| fixed-parallel | 2,666 / 2,520–2,775 | 10,501 / 10,501–10,501 | 37.99 / 41 | 4094 / 1997 |
| tuned-serial | 2,483 / 210–2,775 | 10,159 / 405–10,501 | 34.65 / 38 | 9588 / 4704 |
| tuned-parallel | 2,679 / 2,520–2,760 | 10,501 / 10,501–10,501 | 38.56 / 41 | 3991 / 1944 |

At fixed K/G, mean GPU busy rises 10.42→25.81%, while peak VRAM rises
13,350→20,090 MiB. This shows more overlapped work, not GPU saturation; a busy
percentage alone cannot identify kernel, renderer or synchronization limits.
All four arms observe one attached MPS inference client, as expected for G=1.
Their compute/graphics ownership records contain no unresolved GPU identities
and no sampler errors. The native ownership interval and its caveats remain
separate from this SDK timing conclusion.

| Arm | Cgroup current mean / peak (GiB) | Anonymous mean / peak (GiB) | File cache mean / peak (GiB) | Anonymous after CLI +5 s (GiB) |
| --- | --- | --- | --- | --- |
| fixed-serial | 33.73 / 36.97 | 23.62 / 26.78 | 9.44 / 9.55 | 0.300 |
| fixed-parallel | 42.73 / 53.82 | 32.30 / 43.40 | 9.73 / 9.86 | 0.303 |
| tuned-serial | 34.47 / 37.90 | 23.87 / 27.41 | 9.92 / 10.08 | 0.303 |
| tuned-parallel | 43.13 / 54.53 | 32.15 / 43.41 | 10.26 / 10.41 | 0.302 |

The largest sampled current value is **54.53 GiB**, below the 94 GiB soft and
96 GiB hard limits. This is the maximum observed current sample, not the
historical cgroup `memory.peak` counter (unavailable on this host). Cgroup memory
includes file cache and any other container tasks. The whole-command versus
formal boundary and all raw samples remain available.

### Pressure, cleanup and task lifetime

For every arm, both the full command and formal sample intervals have **zero
new high/max/OOM/oom-kill events**, zero swap high/max/fail deltas and zero
observed swap use. Historical `high=1462070` is a pre-existing cumulative value,
not an event in this experiment. No counter reset/invalid interval is reported.
CPU throttled periods and throttled microseconds have zero increments.
Pressure is small but is not globally zero:

| Arm | Formal memory PSI some / full (%) | Formal CPU PSI some / full (%) | Formal I/O PSI some / full (%) | Whole-command memory PSI some / full (%) |
| --- | --- | --- | --- | --- |
| fixed-serial | 0.001590 / 0.001590 | 0.013200 / 0.012905 | 0.027125 / 0.027122 | 0.003542 / 0.003542 |
| fixed-parallel | 0.006434 / 0.006434 | 0.017938 / 0.017169 | 0.016590 / 0.016588 | 0.006327 / 0.006327 |
| tuned-serial | 0.001800 / 0.001800 | 0.179919 / 0.179772 | 0.016027 / 0.016024 | 0.151879 / 0.151839 |
| tuned-parallel | 0.000000 / 0.000000 | 0.275592 / 0.275164 | 0.001881 / 0.001880 | 0.000000 / 0.000000 |

Percentages are cumulative PSI-stall deltas divided by the corresponding sampled
wall span, not the instantaneous avg10 value. Full-command memory PSI includes
startup/tuning; tuned serial records 3.949 s of memory-some stall there, while
its formal interval records 0.043 s. Whole-command I/O PSI some/full is
0.909880/0.909563% for fixed serial, 0.047850/0.047678% for fixed parallel,
0.165934/0.165869% for tuned serial and 0.002329/0.002328% for tuned parallel.
These nonzero values are retained rather than converted into an absence-of-
pressure claim.

Each Worker handles 25 tasks. Parallel mode constructs five child slots for a
task and reaps them after its one group, before the next task; it does not keep
those same children across 25 different task environments. All 50 parallel
after-group snapshots in each parallel arm see five live slot children, and every after-task snapshot
sees zero retained slot children and zero zombies. The two parallel parent
Workers' post-task anonymous memory ranges are approximately 0.117–0.121 GiB.
Serial Workers retain approximately 7.09–10.55 GiB after tasks; this is not a
surviving-child claim. All observed descendants are gone at CLI exit and at the
+1/+5 s checks, and cgroup anonymous memory is about 0.300–0.303 GiB afterward.

This directly describes the actual 250-episode, one-group-per-task workload.
The earlier eight-reset experiment is a different lifetime pattern and is not
extrapolated into a per-episode growth rate here. Finite cleanup/pressure evidence
does not establish an indefinite memory plateau, nor is such a plateau an extra
gate for this approved round.

## Correctness and all-task coverage

All six runs (two fresh native passes and four SDK arms) score **3/250 (1.2%)**
and execute **32,000 actual steps**. Every success/step comparison with either
native pass has zero disagreements. SDK serial/parallel pairs also have zero
termination-reason differences. The native evaluator does not export a distinct
per-record terminal reason, so native termination-reason equality is not claimed.
The shared short cap depresses success, particularly on composite tasks; these
are reduced-cap results, not official native-horizon accuracy.

| Split | Tasks / episodes per arm | Fixed serial | Fixed parallel | Tuned serial | Tuned parallel | Actual steps per arm |
| --- | --- | --- | --- | --- | --- | --- |
| atomic_seen | 18 / 90 | 3/90 (3.33%) | 3/90 (3.33%) | 3/90 (3.33%) | 3/90 (3.33%) | 11520 |
| composite_seen | 16 / 80 | 0/80 (0.00%) | 0/80 (0.00%) | 0/80 (0.00%) | 0/80 (0.00%) | 10240 |
| composite_unseen | 16 / 80 | 0/80 (0.00%) | 0/80 (0.00%) | 0/80 (0.00%) | 0/80 (0.00%) | 10240 |

Initial full `mjSTATE_INTEGRATION`, time/qpos/qvel and scene-metadata hashes match
250/250 in every comparison. Only initial-observation hashes vary: 4/250 for
Native A/A, 4–7/250 across the eight Native/SDK comparisons, and 5/250 in each
fixed/tuned SDK pair. There are no missing fingerprint fields. The SDK timing
records do not retain numerical initial RGB arrays, so these new hash differences
are not claimed to have a measured one-level pixel bound. Earlier fixed-state
MSAA diagnostics provide supporting context, not a new numeric comparison.
Native A/A has 208/250 exact recorded traces; the remaining first divergences
begin at observation/feature hashes. SDK heavy trajectory tracing was disabled
for timing, and full trajectory equality is not claimed.

The predefined trigger was any paired outcome or actual-step disagreement;
none occurs, so no discrepancy-triggered replay is required. Five scenarios per
task do not establish arbitrary statistical equivalence beyond this matched
sample.

The vectors below are ordered slots 0–4 (`1` means success). Each arm independently
has five records per row. The steps column is the verified task total **in each
of all four arms**: every constituent episode actually executes 128 steps.

| Task | Split | Fixed serial | Fixed parallel | Tuned serial | Tuned parallel | Actual steps in each arm |
| --- | --- | --- | --- | --- | --- | --- |
| CloseBlenderLid | atomic_seen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| CloseFridge | atomic_seen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| CloseToasterOvenDoor | atomic_seen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| CoffeeSetupMug | atomic_seen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| NavigateKitchen | atomic_seen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| OpenCabinet | atomic_seen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| OpenDrawer | atomic_seen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| OpenStandMixerHead | atomic_seen | `00011` (2/5) | `00011` (2/5) | `00011` (2/5) | `00011` (2/5) | 640 |
| PickPlaceCounterToCabinet | atomic_seen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| PickPlaceCounterToStove | atomic_seen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| PickPlaceDrawerToCounter | atomic_seen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| PickPlaceSinkToCounter | atomic_seen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| PickPlaceToasterToCounter | atomic_seen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| SlideDishwasherRack | atomic_seen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| TurnOffStove | atomic_seen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| TurnOnElectricKettle | atomic_seen | `00100` (1/5) | `00100` (1/5) | `00100` (1/5) | `00100` (1/5) | 640 |
| TurnOnMicrowave | atomic_seen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| TurnOnSinkFaucet | atomic_seen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| DeliverStraw | composite_seen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| GetToastedBread | composite_seen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| KettleBoiling | composite_seen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| LoadDishwasher | composite_seen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| PackIdenticalLunches | composite_seen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| PreSoakPan | composite_seen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| PrepareCoffee | composite_seen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| RinseSinkBasin | composite_seen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| ScrubCuttingBoard | composite_seen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| SearingMeat | composite_seen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| SetUpCuttingStation | composite_seen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| StackBowlsCabinet | composite_seen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| SteamInMicrowave | composite_seen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| StirVegetables | composite_seen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| StoreLeftoversInBowl | composite_seen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| WashLettuce | composite_seen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| ArrangeBreadBasket | composite_unseen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| ArrangeTea | composite_unseen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| BreadSelection | composite_unseen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| CategorizeCondiments | composite_unseen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| CuttingToolSelection | composite_unseen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| GarnishPancake | composite_unseen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| GatherTableware | composite_unseen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| HeatKebabSandwich | composite_unseen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| MakeIceLemonade | composite_unseen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| PanTransfer | composite_unseen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| PortionHotDogs | composite_unseen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| RecycleBottlesByType | composite_unseen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| SeparateFreezerRack | composite_unseen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| WaffleReheat | composite_unseen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| WashFruitColander | composite_unseen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |
| WeighIngredients | composite_unseen | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | `00000` (0/5) | 640 |

## Storage maintenance and measurement boundaries

The shared filesystem's file-count quota required preservation of older raw
experiments. Archives were completely byte/hash-verified before old scattered
files were removed; live outputs, source proofs, models and assets were not
moved. The CPU/import proof and preservation work overlapped parts of the native
passes. Their exact/conservative windows and affected task samples are retained
in `side-load-analysis.json`, with both full and window-excluded resource
summaries; excluded-window peaks are not whole-run no-pressure evidence. Native
wall durations are not rewritten and are never a speedup denominator.

All maintenance, including future output routing, ends at
**2026-11-09 17:18:56.557 +0800**. The exclusive GPU handoff occurs at
**17:33:51.741**, and the first SDK command starts at **17:33:51.905**.
The remaining SDK starts are 18:15:30.601, 18:32:59.854 and 19:16:26.559.
The guard verifies every command starts after maintenance and no preservation
process remains. No bulk evidence verification/archiving runs during SDK formal
timing.

`output-storage-routing.json` records all 13 allowlisted output symlinks:
all four SDK arms share the same fresh local filesystem, plus nine future native
tasks routed before their directories existed. Existing/active files and locks
remain in place. Native storage consequently differs across tasks, another
reason its per-task wall sums are not performance denominators. The four SDK
arms have identical output backing. Disposable bytecode cleanup touches the
model environment resolved on `/root`'s local disk; it contributes zero shared-
filesystem inode release and can change later cold import time. All SDK arms
start after the same cleanup with bytecode writing disabled.

## Reproduction and evidence

Use the fresh pinned runtime from the integration report. The immutable evidence
contains every native/SDK launch command, attempt, manifest, source correspondence,
model/runtime hash, task identity, final record, GPU/host sample, ownership claim,
cleanup check, side-load window and comparison JSON. Weights and installed assets
are excluded. The [approved250 evidence release](https://github.com/QLL-CCDD/RoboScope-Kit/releases/tag/robocasa365-approved250-evidence-20261109)
provides the archive, per-file manifest and checksum; historical archives remain
separate.

The external driver runs all four SDK arms sequentially. It refuses the GPU
handoff until both fresh native passes cover all 250 identities without a final
failure. The command block shows the recorded launch layout; choose a fresh
`OUT` for a new run and first complete the linked Native A/B recipe there.
The recorded `command.json` files retain the exact argv used.
The output-routing script is included for reproducing the quota-limited storage
layout and does not modify simulator/Policy behavior.

```bash
W=/root/autodl-tmp/robocasa365-20261107
E=/autodl-fs/data/robocasa365/acceptance-20261107
C="$W/control-reviewed-native/bin/python"
R="$W/runtime-reviewed-native"
OUT="$E/full250-approved-20261109"

"$C" -B "$W/full250-approved/kit.py" \
  --root "$OUT/kit" \
  --baseline "$W/followup-native-src" \
  --baseline-overlay "$W/followup-native-overlay" \
  --candidate "$W/followup-slots-src" \
  --candidate-overlay "$W/followup-slots-overlay" \
  --runtime "$R" \
  --source-proof "$E/followup-artifact-tools/source-proof.json" \
  --native-summary "$OUT/native/summary.json" \
  --baseline-commit 0b876848d10f10536a457e1bb499e01a3b9f4532 \
  --candidate-commit aeb9c58106816b27e490c75356c9f8f7376445b8 \
  --workers 2 --replicas 1 --max-workers 2 --max-replicas 1 \
  --cap 128 --candidate-timeout-s 800 --tune-budget-s 1800 \
  --arm-timeout-s 7200 --wait-for-gpu-lease

"$C" -B "$W/full250-approved/summarize.py" "$OUT/kit"
"$C" -B "$W/full250-approved/compare_native.py" "$OUT/kit" "$OUT/native"
"$C" -B "$W/full250-approved/memory_analysis.py" \
  --sdk-comparison "$OUT/kit/comparison.json" --native-root "$OUT/native" \
  --gpu-owners "$OUT/gpu-owners.jsonl" --output "$OUT/memory-analysis.json"
"$C" -B "$W/full250-approved/side_load_analysis.py" \
  --root "$OUT" --output "$OUT/side-load-analysis.json"
```

The initial run requires a fresh `--root`; explicit `--resume` validates the
frozen configuration hash and reuses only verified complete attempts. Failed
attempts are never overwritten or silently pooled. The new full-workload archive
supersedes missing-scope claims from earlier pilots without relabelling their
source, count, horizon or measured speed.

The parallel snapshot also includes the complete-message deadline/bounded
shutdown correction verified by eight real Linux spawned-child cases in the
[prior transport follow-up](composite-reset-followup-20261109.md). Those fault
checks are correctness evidence, not additional throughput gains. No inference
algorithm, model precision or simulator-rule optimization is included. Repository
acceptance still requires green CI on the final submitted heads; this performance
report makes no CI or merge claim.

## Restored-hardware follow-up — 2026-11-11

The measurements above retain their original executable revisions, hardware and bounded K≤2/G≤1 search. They are not relabeled as results from the restored GPU/driver/20-core campaign.

The [published epoch2 native supporting reference](https://github.com/QLL-CCDD/RoboScope-Kit/releases/tag/robocasa365-epoch2-native-reference-20261111) covers all 50 tasks × five selected episodes with the shared 128-step cap: 250 unique records, 32,000 steps and three successes. One failed cleanup audit remains preserved and unselected; including its five episodes, actual execution is 255 episodes and 32,640 steps. All three release payloads were downloaded and fully verified. Native command time is a correctness-reference diagnostic, never a speedup denominator.

The [completed bounded epoch2 comparison](expanded-kg-epoch2-20261111.md) uses 19 single-sample external K/G screening points and three full50×5 arms, with MPS, full RGB, FP32/100 DDPM and the corrected runtime unchanged. Fixed K3/G2 formal time improves 1,927.430 to 803.357 seconds (2.399221×); provisional serial K5/G1 versus parallel K3/G2 improves 1,250.652 to 803.357 seconds (1.556783×, below the 2× target). Whole-command gains are 2.213400× and 1.416654×. All three arms finish 250 records/32,000 control steps/three successes and match their native outcomes/steps/initial physics/scenario records. The screening is external fixed-CLI tuning, not new SDK built-in autotune or a global-optimum result. Universal repeats and GPU frontier expansion were canceled for quick delivery. The full raw backup is independently verified locally; its public attachment upload failed with HTTP 408 and remains outstanding.
