# RoboCasa365 composite and reset follow-up, 2026-11-09

This historical composite and eight-reset follow-up remains a separate workload.
The [current all-task comparison](approved250-performance-20261109.md) records
50 tasks with one group per task, its own memory/pressure/cleanup evidence, and
fixed/bounded-tuned throughput. The old reset-growth observations below are not
extrapolated to that lifetime pattern, and the ratios are not pooled.

Baseline SDK `43df866a3c23e83ee6edade8f68615f3eae1be6c` and parallel SDK
`e836f775fb7c86efde7e5b6046b07d5eb3deeee5` ran on the same checksum-verified simulator runtime.
The parallel source includes the complete-message IPC deadline and bounded shutdown fix.
Eight Linux spawned-child transport checks cover a normal 16 MiB round trip, partial
headers/bodies, child death before/during a reply, a blocked 32 MiB request send,
normal close and five non-reading children with full send buffers. These are timeout/
cleanup correctness checks; their timings are not included as performance gains.
The official human300 Diffusion Policy uses the same real weights, FP32, 100 DDPM
steps, actions, full RGB cameras, native success/group termination rules and videos off.
Constructor seeds are 1111111–1111115; subsequent groups continue the same five
environment RNG streams. GPU arms run sequentially, with CUDA MPS enabled on both paths.

The host is an RTX 4090 reporting 49,140 MiB, driver 580.105.08, Xeon Gold 6459C,
12-core container quota and 96 GiB memory limit. The earlier K4/G2 parallel pilot
reached 93.83 GiB container memory. This follow-up bounds Worker/replica counts
rather than extrapolating that setting to larger process counts.

## Complete-message transport validation

The old Linux `multiprocessing.Pipe` reproduction declared a 0.1-second
receive deadline but returned success after 0.603 seconds when only the
message header arrived promptly. The fixed private socket channel uses an
absolute deadline for header/body reads and request sends. An error closes
the channels and reaps the children; close delivery and graceful joins share
one shutdown deadline before a separate bounded kill/join phase.

| Linux spawned-child case | Elapsed including cleanup (s) | Expected result | All children reaped / repeated close |
| --- | ---: | --- | --- |
| Normal 16 MiB round trip | 0.121 | Success | Yes / passed |
| Partial header, child alive | 0.314 | TimeoutError | Yes / passed |
| Partial body, child alive | 0.320 | TimeoutError | Yes / passed |
| Child dies during body | 0.023 | RuntimeError | Yes / passed |
| Child already dead | 0.00024 | RuntimeError | Yes / passed |
| Blocked 32 MiB request send | 0.586 | TimeoutError | Yes / passed |
| Normal close | 0.052 | Success | Yes / passed |
| Five non-reading children, full send buffers | 0.121 | Success | Yes / passed |

Fault tests use a 0.2-second request deadline and 0.1-second shutdown interval;
the normal large-message test uses a 3-second request deadline. Measured time
includes synchronous serialization and cleanup, rather than only the transport
wait. Production settings remain 300 seconds per command and five seconds
for graceful shutdown plus five seconds for forced reaping. Serialization of
trusted slot payloads is synchronous, not preemptible. The raw report records
the exact source hash, Python version, PIDs and old reproduction.

```bash
W=/root/autodl-tmp/robocasa365-20261107
PYTHONPATH="$W/followup-slots-src/packages/roboscope-kit/src" \
  "$W/control-reviewed-native/bin/python" -B "$W/check_slot_ipc_followup.py" \
  --old-source "$W/reviewed-slots-src/packages/roboscope-kit/src/roboscope_kit/runtime/benchmarks/robocasa365/slots.py" \
  --output /tmp/robocasa365-ipc-repeat.json
```

## Composite matched-workload timing

4 tasks × 10 episodes; 2 consecutive native five-slot groups per task; 128-step diagnostic cap.
Fixed K=2/G=1; independent diagonal searches use
the same K≤2/G≤1 bounds, elapsed estimator,
800 s candidate timeout and 1800 s total budget.
The tasks include two Composite-Seen and two Composite-Unseen cases.
The shared cap belongs to the external diagnostic harness; committed official
task horizons and formal rollout counts remain unchanged.

| Arm | K/G | Formal evaluation (s) | Autotune (s) | Whole command (s) | Episodes/h | Steps/s | Successes | Actual steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-serial | 2/1 | 331.99 | 0.02 | 346.68 | 433.7 | 15.42 | 0/40 | 5,120 |
| fixed-parallel | 2/1 | 139.50 | 0.03 | 153.39 | 1032.3 | 36.70 | 0/40 | 5,120 |
| tuned-serial | 2/1 | 319.01 | 211.67 | 536.99 | 451.4 | 16.05 | 0/40 | 5,120 |
| tuned-parallel | 2/1 | 137.23 | 89.56 | 229.72 | 1049.3 | 37.31 | 0/40 | 5,120 |

| Comparison | Formal episode-throughput ratio | Step-rate ratio | End-to-end ratio |
| --- | ---: | ---: | ---: |
| fixed | 2.380× | 2.380× | 2.260× |
| tuned | 2.325× | 2.325× | 2.338× |

All paired physical-step counts match; no arm drops an episode or physical step
from its paired workload. Episode and step throughput are reported together.

These are one fixed pair and one independently tuned pair on this composite
workload. The earlier atomic three-pair median remains a separate result.
Ratios from different workloads or tuning settings are not multiplied or pooled.

### Autotune work and selection

The timing/result table above counts only formal episodes. Tuning performs its
own native-group probes and can repeat episode identities across candidates.
The table below reports candidate progress and health as recorded by autotune;
unrecorded probe step counts are not reconstructed from horizon or elapsed rates.

| Arm | Probe K/G | Decision | Elapsed (s) | Completed / expected probe episodes | Probe successes | Recorded probe steps | Requested / final Workers | Worker failures / OOM | Stop reason |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| tuned-serial | 1/1 | accepted | 99.24 | 5 / 5 | 0 | unrecorded | 1 / 1 | 0 / 0 |  |
| tuned-serial | 2/1 | accepted | 112.23 | 10 / 10 | 0 | unrecorded | 2 / 2 | 0 / 0 |  |
| tuned-parallel | 1/1 | accepted | 44.14 | 5 / 5 | 0 | unrecorded | 1 / 1 | 0 / 0 |  |
| tuned-parallel | 2/1 | accepted | 45.32 | 10 / 10 | 0 | unrecorded | 2 / 2 | 0 / 0 |  |

| Arm | Selection status | Chosen K/G | Probe completed-episode sum | Recorded probe-step sum | Stop reason |
| --- | --- | ---: | ---: | ---: | --- |
| tuned-serial | complete | 2/1 | 15 | unrecorded | worker cap 2 reached; replica cap 1 reached |
| tuned-parallel | complete | 2/1 | 15 | unrecorded | worker cap 2 reached; replica cap 1 reached |

Probe sums count executed candidate work, including repeated identities; they are
not additional formal coverage. Rejected or failed candidates remain visible.
Each selected K/G is the controller decision within the recorded bounds; this
limited search does not establish a global optimum.

## Composite resource use and timing boundaries

| Arm | GPU busy mean / P95 / peak | VRAM mean / peak (MiB) | Memory busy mean | Power mean (W) | CPU mean cores | CPU throttle time / periods | MPS clients |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-serial | 10.66% / 36% / 52% | 10,431 / 14,661 | 1.25% | 73.7 | 1.96 | 0.001 / 0.000 | 1 |
| fixed-parallel | 26.66% / 56% / 79% | 14,529 / 20,355 | 3.48% | 102.4 | 4.78 | 0.000 / 0.000 | 1 |
| tuned-serial | 9.99% / 38% / 67% | 10,596 / 14,661 | 1.18% | 73.9 | 1.99 | 0.000 / 0.000 | 1 |
| tuned-parallel | 24.27% / 48% / 83% | 14,522 / 20,355 | 3.14% | 103.7 | 4.86 | 0.000 / 0.000 | 1 |

| Arm | Cgroup current mean / peak (GiB) | Anonymous mean / peak (GiB) | File cache mean / peak (GiB) | Peak observed processes | Peak process RSS sum (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed-serial | 27.97 / 33.12 | 22.08 / 27.01 | 5.25 / 5.46 | 8 | 28.00 |
| fixed-parallel | 42.06 / 52.01 | 35.91 / 45.83 | 5.47 / 5.48 | 23 | 48.62 |
| tuned-serial | 28.32 / 33.58 | 22.17 / 27.41 | 5.51 / 5.52 | 6 | 28.37 |
| tuned-parallel | 41.78 / 51.76 | 35.55 / 45.49 | 5.55 / 5.56 | 21 | 48.26 |

GPU samples are device-wide at 250 ms; they include rendering and inference,
rather than isolated Policy GPU execution. Clocks and temperature remain in the raw
samples. Host samples are recorded every 0.5 s. Cgroup memory includes file cache
and other processes in the container. **Process RSS sums double-count shared pages**
and must not be interpreted as unique resident memory. Peak process counts include
the command and its observed descendants throughout startup, tuning and evaluation.

| Arm | Initial → after-cleanup current (ΔGiB) | Anonymous (ΔGiB) | File cache (ΔGiB) |
| --- | ---: | ---: | ---: |
| fixed-serial | 5.88 → 6.29 (+0.41) | 0.28 → 0.28 (+0.00) | 5.02 → 5.43 (+0.41) |
| fixed-parallel | 6.29 → 6.32 (+0.02) | 0.28 → 0.28 (-0.00) | 5.43 → 5.46 (+0.02) |
| tuned-serial | 6.32 → 6.36 (+0.04) | 0.28 → 0.28 (+0.00) | 5.46 → 5.50 (+0.04) |
| tuned-parallel | 6.36 → 6.40 (+0.04) | 0.28 → 0.28 (+0.00) | 5.50 → 5.54 (+0.04) |

| Arm | `low` initial → final (Δ) | `high` initial → final (Δ) | `max` initial → final (Δ) | `oom` initial → final (Δ) | `oom_kill` initial → final (Δ) | `oom_group_kill` initial → final (Δ) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-serial | 0 → 0 (+0) | 1462070 → 1462070 (+0) | 0 → 0 (+0) | 0 → 0 (+0) | 0 → 0 (+0) | unrecorded |
| fixed-parallel | 0 → 0 (+0) | 1462070 → 1462070 (+0) | 0 → 0 (+0) | 0 → 0 (+0) | 0 → 0 (+0) | unrecorded |
| tuned-serial | 0 → 0 (+0) | 1462070 → 1462070 (+0) | 0 → 0 (+0) | 0 → 0 (+0) | 0 → 0 (+0) | unrecorded |
| tuned-parallel | 0 → 0 (+0) | 1462070 → 1462070 (+0) | 0 → 0 (+0) | 0 → 0 (+0) | 0 → 0 (+0) | unrecorded |

Initial values are captured before the command and final values after its five-second
descendant cleanup observation. `memory.events` is cumulative for the cgroup: historical
counter values are retained and **only each arm’s delta describes new events**.
The `max` limit-hit counter and `oom`/`oom_kill` counters describe different events.
Container caches and non-descendant processes can affect end-memory differences;
a remaining anonymous-memory delta is not by itself a surviving child process.

| Arm | Checkpoint preparation (s) | Runtime preparation (s) | Policy load stage (s) | Autotune (s) | Formal evaluation (s) | Other command time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-serial | 0.031 | 0.877 | 10.300 | 0.018 | 331.991 | 3.463 |
| fixed-parallel | 0.008 | 0.898 | 10.909 | 0.029 | 139.495 | 2.052 |
| tuned-serial | 0.062 | 0.871 | 0.024 | 211.672 | 319.012 | 5.348 |
| tuned-parallel | 0.009 | 0.869 | 0.011 | 89.564 | 137.229 | 2.037 |

Formal `evaluation_s` includes Worker creation and environment reset. Checkpoint
preparation resolves the already-downloaded weights; earlier downloads and runtime
building are outside these command timings. Autotune includes Policy startup/probes
and retains usable replicas, so its later Policy-load stage can be small. Other
command time includes interpreter/CLI work, sampler finalization and final outputs;
result serialization and cleanup are not independently isolated. The separate
five-second descendant-cleanup observation is excluded from whole-command time.

| Arm | Worker creation sum (s) | Reset sum (s) | Native step/barrier sum (s) | Policy RPC sum (s) | Video write sum (s) | Cleanup sum (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-serial | 113.73 | 218.38 | 192.70 | 98.48 | 0.00 | 1.80 |
| fixed-parallel | 36.97 | 52.01 | 47.42 | 123.53 | 0.00 | 2.65 |
| tuned-serial | 111.41 | 213.57 | 189.52 | 95.27 | 0.00 | 1.84 |
| tuned-parallel | 37.27 | 52.43 | 47.95 | 119.21 | 0.00 | 2.65 |

Worker intervals overlap and cannot be added to obtain wall time. Parallel step
time includes the slot barrier and IPC; RPC includes transport and queue waiting.

## Composite result coverage and paired outcomes

| Pair | Paired episodes | Success differences | Step differences | Termination differences | Initial fingerprint differences |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 40 | 0 | 0 | 0 | 0 |
| tuned | 40 | 0 | 0 | 0 | 0 |

Coverage validates complete unique task/episode IDs, constructor seeds, stream
indices, native slot assignment and actual execution mode. Initial fingerprints
cover the flattened simulator state (time/qpos/qvel), scenario metadata and raw
initial observations. This fingerprint does not cover every `mjSTATE_INTEGRATION`
field. The focused native and raw-reset diagnostics report their fuller checks
separately. These
timed runs do not record complete trajectories or establish native-horizon accuracy.

| Task / split | Pair | Serial success vector | Parallel success vector | Serial / parallel actual steps |
| --- | --- | --- | --- | ---: |
| `KettleBoiling` / composite_seen | fixed | `0000000000` | `0000000000` | 1280 / 1280 |
| `KettleBoiling` / composite_seen | tuned | `0000000000` | `0000000000` | 1280 / 1280 |
| `RinseSinkBasin` / composite_seen | fixed | `0000000000` | `0000000000` | 1280 / 1280 |
| `RinseSinkBasin` / composite_seen | tuned | `0000000000` | `0000000000` | 1280 / 1280 |
| `CuttingToolSelection` / composite_unseen | fixed | `0000000000` | `0000000000` | 1280 / 1280 |
| `CuttingToolSelection` / composite_unseen | tuned | `0000000000` | `0000000000` | 1280 / 1280 |
| `WeighIngredients` / composite_unseen | fixed | `0000000000` | `0000000000` | 1280 / 1280 |
| `WeighIngredients` / composite_unseen | tuned | `0000000000` | `0000000000` | 1280 / 1280 |

Binary digits follow episode-index order. Each block of five is one native group;
each later block continues each constructor RNG stream after another reset.

| Arm | Worker failures | Planned recycles | OOM events | After-task slot leftovers | After-task zombie children | Observed descendants after 5 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-serial | 0 | 0 | 0 | 0 | 0 | 0 |
| fixed-parallel | 0 | 0 | 0 | 0 | 0 | 0 |
| tuned-serial | 0 | 0 | 0 | 0 | 0 | 0 |
| tuned-parallel | 0 | 0 | 0 | 0 | 0 | 0 |

Postprocessing rejects spawned-slot leftovers, unreaped zombie children after
task cleanup, and any observed descendant PID/start-time identity remaining after
the final five-second cleanup observation. Native multiprocessing resource-tracker
helpers may remain between tasks; they must also exit with the command. All raw
checks at exit, one second and five seconds are retained, including process states.

| Arm / task | Groups | Stable slot PID count | First → last anonymous memory (GiB) | Anonymous min–max (GiB) | First → last Worker + slot RSS sum (GiB) | Last four anonymous range (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-serial / `KettleBoiling` | 2 | 0 | 20.77 → 23.90 | 20.77–23.90 | 9.32 → 11.47 | 20.77–23.90 |
| fixed-serial / `RinseSinkBasin` | 2 | 0 | 20.73 → 21.78 | 20.73–21.78 | 9.41 → 10.44 | 20.73–21.78 |
| fixed-serial / `CuttingToolSelection` | 2 | 0 | 26.09 → 24.74 | 24.74–26.09 | 11.93 → 12.80 | 24.74–26.09 |
| fixed-serial / `WeighIngredients` | 2 | 0 | 25.11 → 26.96 | 25.11–26.96 | 11.27 → 12.25 | 25.11–26.96 |
| fixed-parallel / `KettleBoiling` | 2 | 5 | 40.71 → 40.92 | 40.71–40.92 | 20.44 → 20.92 | 40.71–40.92 |
| fixed-parallel / `RinseSinkBasin` | 2 | 5 | 41.81 → 36.85 | 36.85–41.81 | 19.80 → 19.57 | 36.85–41.81 |
| fixed-parallel / `CuttingToolSelection` | 2 | 5 | 42.10 → 24.95 | 24.95–42.10 | 19.50 → 23.19 | 24.95–42.10 |
| fixed-parallel / `WeighIngredients` | 2 | 5 | 40.56 → 45.76 | 40.56–45.76 | 20.47 → 22.06 | 40.56–45.76 |
| tuned-serial / `KettleBoiling` | 2 | 0 | 20.69 → 22.34 | 20.69–22.34 | 9.30 → 11.46 | 20.69–22.34 |
| tuned-serial / `RinseSinkBasin` | 2 | 0 | 20.84 → 23.87 | 20.84–23.87 | 9.39 → 10.41 | 20.84–23.87 |
| tuned-serial / `CuttingToolSelection` | 2 | 0 | 26.02 → 25.17 | 25.17–26.02 | 11.96 → 12.79 | 25.17–26.02 |
| tuned-serial / `WeighIngredients` | 2 | 0 | 25.07 → 27.39 | 25.07–27.39 | 11.18 → 12.69 | 25.07–27.39 |
| tuned-parallel / `KettleBoiling` | 2 | 5 | 40.35 → 40.59 | 40.35–40.59 | 20.38 → 20.91 | 40.35–40.59 |
| tuned-parallel / `RinseSinkBasin` | 2 | 5 | 41.45 → 36.57 | 36.57–41.45 | 19.78 → 19.60 | 36.57–41.45 |
| tuned-parallel / `CuttingToolSelection` | 2 | 5 | 41.73 → 24.59 | 24.59–41.73 | 19.50 → 23.20 | 24.59–41.73 |
| tuned-parallel / `WeighIngredients` | 2 | 5 | 40.17 → 45.42 | 40.17–45.42 | 20.48 → 22.07 | 40.17–45.42 |

Boundary values are sampled after each complete group; they are not peaks.
Cgroup values for concurrent tasks overlap and cannot be attributed to one task.
Each task retains one Worker identity and the same slot PID set across all groups.
Changing random scene assets, allocator retention and caches can change memory
between resets. These first/last values and ranges are descriptive; no fitted
slope or extrapolation labels a finite variation as a leak or proves indefinite
stability. The separate single-Worker run makes the reset sequence easier to inspect.

## Separate consecutive-reset stability diagnostic

1 tasks × 40 episodes; 8 consecutive native five-slot groups per task; 16-step diagnostic cap.
This run uses fixed K=1/G=1 and no autotune.
It spends more of its work on repeated resets and is a finite resource/cleanup
diagnostic, not a new formal-throughput result. Its timing is not combined with
the 128-step composite comparison or the earlier atomic pilot.

| Arm | Command time (s) | Evaluation time (s) | Episodes | Actual steps | Successes |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed-serial | 282.21 | 264.63 | 40 | 640 | 0 |
| fixed-parallel | 96.19 | 81.50 | 40 | 640 | 0 |

| Arm | GPU busy mean / P95 / peak | VRAM mean / peak (MiB) | Memory busy mean | Power mean (W) | CPU mean cores | CPU throttle time / periods | MPS clients |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-serial | 3.30% / 30% / 47% | 6,300 / 8,644 | 0.22% | 62.7 | 1.08 | 0.000 / 0.000 | 1 |
| fixed-parallel | 9.19% / 37% / 54% | 7,428 / 10,929 | 0.49% | 72.5 | 3.34 | 0.000 / 0.000 | 1 |

| Arm | Cgroup current mean / peak (GiB) | Anonymous mean / peak (GiB) | File cache mean / peak (GiB) | Peak observed processes | Peak process RSS sum (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed-serial | 18.91 / 21.57 | 12.65 / 15.21 | 5.64 / 5.74 | 6 | 15.93 |
| fixed-parallel | 27.50 / 32.61 | 21.11 / 26.20 | 5.75 / 5.75 | 11 | 27.92 |

GPU samples are device-wide at 250 ms; they include rendering and inference,
rather than isolated Policy GPU execution. Clocks and temperature remain in the raw
samples. Host samples are recorded every 0.5 s. Cgroup memory includes file cache
and other processes in the container. **Process RSS sums double-count shared pages**
and must not be interpreted as unique resident memory. Peak process counts include
the command and its observed descendants throughout startup, tuning and evaluation.

| Arm | Initial → after-cleanup current (ΔGiB) | Anonymous (ΔGiB) | File cache (ΔGiB) |
| --- | ---: | ---: | ---: |
| fixed-serial | 6.40 → 6.58 (+0.18) | 0.28 → 0.28 (+0.00) | 5.54 → 5.72 (+0.18) |
| fixed-parallel | 6.58 → 6.58 (+0.01) | 0.28 → 0.28 (-0.00) | 5.72 → 5.72 (+0.01) |

| Arm | `low` initial → final (Δ) | `high` initial → final (Δ) | `max` initial → final (Δ) | `oom` initial → final (Δ) | `oom_kill` initial → final (Δ) | `oom_group_kill` initial → final (Δ) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-serial | 0 → 0 (+0) | 1462070 → 1462070 (+0) | 0 → 0 (+0) | 0 → 0 (+0) | 0 → 0 (+0) | unrecorded |
| fixed-parallel | 0 → 0 (+0) | 1462070 → 1462070 (+0) | 0 → 0 (+0) | 0 → 0 (+0) | 0 → 0 (+0) | unrecorded |

Initial values are captured before the command and final values after its five-second
descendant cleanup observation. `memory.events` is cumulative for the cgroup: historical
counter values are retained and **only each arm’s delta describes new events**.
The `max` limit-hit counter and `oom`/`oom_kill` counters describe different events.
Container caches and non-descendant processes can affect end-memory differences;
a remaining anonymous-memory delta is not by itself a surviving child process.

| Pair | Paired episodes | Success differences | Step differences | Termination differences | Initial fingerprint differences |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 40 | 0 | 0 | 0 | 1 |

For `fixed`, the recorded initial simulator-state hashes match 40/40,
scenario hashes 40/40, and initial observation hashes 39/40.
`KettleBoiling` episode index 6, slot 1, stream episode 1 differs in `initial_observation_sha256`.
The hash difference remains part of the result. A separate real-Policy capture
repeated the first two groups (10 episodes per side, cap 16, K=1/G=1 and MPS on),
saving raw reset observations before inference. Both captures' initial input,
simulator-state and scenario hashes exactly equal their corresponding older
episode-6 hashes, linking the numerical comparison to that specific earlier
sample. The difference is 45 left-camera and 144 right-camera RGB channels out
of 196,608 per image, each by one integer intensity level; wrist, state and
language inputs are exact. All ten captured initial integration states and
scenarios match, nine initial observations are exact, and success, actual steps
and termination match in all ten pairs. The raw ZIPs, numerical comparison and
six old/new fingerprint links are retained. This quantifies the prior observation
difference; it adopts no RGB/action/physics tolerance and does not establish
full-trajectory or native-horizon accuracy equivalence. The instrumented capture
is a correctness diagnostic and contributes no performance gain to this report.

Coverage validates complete unique task/episode IDs, constructor seeds, stream
indices, native slot assignment and actual execution mode. Initial fingerprints
cover the flattened simulator state (time/qpos/qvel), scenario metadata and raw
initial observations. This fingerprint does not cover every `mjSTATE_INTEGRATION`
field. The focused native and raw-reset diagnostics report their fuller checks
separately. These
timed runs do not record complete trajectories or establish native-horizon accuracy.

| Task / split | Pair | Serial success vector | Parallel success vector | Serial / parallel actual steps |
| --- | --- | --- | --- | ---: |
| `KettleBoiling` / composite_seen | fixed | `0000000000000000000000000000000000000000` | `0000000000000000000000000000000000000000` | 640 / 640 |

Binary digits follow episode-index order. Each block of five is one native group;
each later block continues each constructor RNG stream after another reset.

| Arm / task | Groups | Stable slot PID count | First → last anonymous memory (GiB) | Anonymous min–max (GiB) | First → last Worker + slot RSS sum (GiB) | Last four anonymous range (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-serial / `KettleBoiling` | 8 | 0 | 11.42 → 14.93 | 11.42–15.11 | 9.29 → 12.81 | 14.00–15.11 |
| fixed-parallel / `KettleBoiling` | 8 | 5 | 21.58 → 26.09 | 21.58–26.09 | 20.34 → 24.97 | 23.09–26.09 |

Boundary values are sampled after each complete group; they are not peaks.
Cgroup values for concurrent tasks overlap and cannot be attributed to one task.
Each task retains one Worker identity and the same slot PID set across all groups.
Changing random scene assets, allocator retention and caches can change memory
between resets. These first/last values and ranges are descriptive; no fitted
slope or extrapolation labels a finite variation as a leak or proves indefinite
stability. The separate single-Worker run makes the reset sequence easier to inspect.

| Completed group | fixed-serial anonymous / current / RSS sum (GiB) | fixed-parallel anonymous / current / RSS sum (GiB) |
| ---: | ---: | ---: |
| 1 | 11.42 / 17.61 / 9.29 | 21.58 / 27.98 / 20.34 |
| 2 | 13.26 / 19.49 / 11.14 | 22.18 / 28.58 / 20.95 |
| 3 | 13.20 / 19.46 / 11.08 | 23.09 / 29.49 / 21.94 |
| 4 | 13.10 / 19.38 / 10.98 | 23.62 / 30.02 / 22.49 |
| 5 | 14.00 / 20.29 / 11.88 | 23.09 / 29.49 / 21.96 |
| 6 | 14.69 / 21.02 / 12.57 | 24.47 / 30.87 / 23.33 |
| 7 | 15.11 / 21.47 / 13.00 | 25.34 / 31.75 / 24.21 |
| 8 | 14.93 / 21.30 / 12.81 | 26.09 / 32.49 / 24.97 |

**The consecutive-reset check does not establish a memory plateau.**
For `fixed-serial`, anonymous cgroup memory after completed groups changed from 11.42 to 14.93 GiB (+3.51 GiB); the last four boundaries span 14.00–15.11 GiB.
For `fixed-parallel`, anonymous cgroup memory after completed groups changed from 21.58 to 26.09 GiB (+4.51 GiB); the last four boundaries span 23.09–26.09 GiB.
The parallel arm continues to grow near the end of the observed sequence.
Both command/process lifetimes end with no observed descendant or zombie child
remaining, and anonymous memory returns close to its recorded pre-command value.
That cleanup evidence does not establish that memory stops growing in a longer-lived
simulator. An execution-pass marker for this diagnostic is not long-term stability acceptance.

| Arm | Worker failures | Planned recycles | OOM events | After-task slot leftovers | After-task zombie children | Observed descendants after 5 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-serial | 0 | 0 | 0 | 0 | 0 | 0 |
| fixed-parallel | 0 | 0 | 0 | 0 | 0 | 0 |

Postprocessing rejects spawned-slot leftovers, unreaped zombie children after
task cleanup, and any observed descendant PID/start-time identity remaining after
the final five-second cleanup observation. Native multiprocessing resource-tracker
helpers may remain between tasks; they must also exit with the command. All raw
checks at exit, one second and five seconds are retained, including process states.

### Native reset object-lifetime control

A separate single-environment `KettleBoiling` diagnostic ran eight resets with
sixteen fixed zero actions per reset through the official converter (128 steps).
It uses no Policy and makes no performance or accuracy claim. Original MSAA
remained at four samples; the diagnostic did not alter the installed runtime or its source.

Weak references found one previous `MjSim` and renderer still alive before
explicit collection in seven of eight resets. `gc.collect()` made all previous
simulator/context weak references dead; the other reset had already collected
them automatically. GC reduced anonymous process memory by 0–690 MiB, and a
subsequent diagnostic glibc `malloc_trim(0)` released another 375–1373 MiB.
For example, the first reset measured 3987.88 → 3461.85 → 2803.57 MiB before GC,
after GC and after trimming. Current `mjSTATE_INTEGRATION`, scenario metadata
and read-only framebuffer hashes remained exact at every manual cleanup stage.
Raw arrays, weak-reference counts and process RSS/RssAnon/PSS are retained.

The source explains a concrete lifetime mechanism: the
[pinned reset implementation](https://github.com/ARISE-Initiative/robosuite/blob/5ce6643f3092639d08f7b0f90ed1c6a84f50552c/robosuite/environments/base.py)
only explicitly destroys the previous simulation when `renderer == "mujoco"`.
This runtime reports the default `renderer="mjviewer"` with hard reset enabled,
so it replaces the simulation without that explicit free. The
[simulator/context ownership and free implementation](https://github.com/ARISE-Initiative/robosuite/blob/5ce6643f3092639d08f7b0f90ed1c6a84f50552c/robosuite/utils/binding_utils.py)
contains reciprocal sim/context references; explicit `MjSim.free()` breaks its
references and collects garbage. The diagnostic therefore demonstrates delayed
cyclic disposal and allocator retention in this process. It neither attributes
every byte of the multi-slot growth nor proves an unbounded leak or a plateau.
The extra per-reset GC and trim operations were not added to evaluation or
included in the reported speed gains.

The actual container limits are `memory.max=96 GiB` and `memory.high=94 GiB`;
the earlier 93.83 GiB result was close to the soft threshold. Explicit K≤2/G≤1
limits address the measured resource budget, but do not establish indefinite
stability or capacity for every task. Reproduce this diagnostic separately:

```bash
W=/root/autodl-tmp/robocasa365-20261107
E=/autodl-fs/data/robocasa365/acceptance-20261107
"$W/runtime-reviewed-native/env/bin/python" -B "$W/followup-reset-memory.py" \
  --output "$E/reset-memory-gc-repeat" --mode collect --trim
```

`--trim` is optional; it was enabled in this diagnostic to separate object
collection from allocator retention. `--mode normal` defers manual collection
to the final control and is available for further investigation, but was not run.

## Reproduction and limits

Extract the frozen `followup-review` harness and sources from the evidence release.
Set the baseline/candidate source paths, SDK-only overlays and shared runtime to
the recorded paths; keep the virtual-environment Python path without resolving its symlink.
Run all GPU arms sequentially, in fresh output directories.

```bash
W=/root/autodl-tmp/robocasa365-20261107
E=/autodl-fs/data/robocasa365/acceptance-20261107
C="$W/control-reviewed-native/bin/python"
COMMON=(
  --baseline "$W/followup-native-src" --baseline-overlay "$W/followup-native-overlay"
  --baseline-commit 43df866a3c23e83ee6edade8f68615f3eae1be6c
  --candidate "$W/followup-slots-src" --candidate-overlay "$W/followup-slots-overlay"
  --candidate-commit e836f775fb7c86efde7e5b6046b07d5eb3deeee5
  --runtime "/root/autodl-tmp/robocasa365-20261107/runtime-reviewed-native"
)
"$C" "$W/followup-review/run.py" "${COMMON[@]}" \
  --root "$E/composite-followup-repeat" --cap 128 --episodes 10 \
  --tasks KettleBoiling RinseSinkBasin CuttingToolSelection WeighIngredients \
  --workers 2 --replicas 1 \
  --max-workers 2 --max-replicas 1 \
  --candidate-timeout-s 800 --tune-budget-s 1800
"$C" "$W/followup-review/summarize.py" "$E/composite-followup-repeat"
"$C" "$W/followup-review/run.py" "${COMMON[@]}" \
  --root "$E/slot-stability-followup-repeat" --tasks KettleBoiling \
  --cap 16 --episodes 40 \
  --workers 1 --replicas 1 --max-workers 1 --max-replicas 1 \
  --arms fixed-serial fixed-parallel
"$C" "$W/followup-review/summarize.py" "$E/slot-stability-followup-repeat"
"$C" "$W/followup-reset-trace/run.py" "${COMMON[@]}" \
  --root "$E/reset-rgb-capture-repeat" --tasks KettleBoiling \
  --cap 16 --episodes 10 \
  --workers 1 --replicas 1 --max-workers 1 --max-replicas 1 \
  --arms fixed-serial fixed-parallel
"$C" "$W/followup-reset-trace/numeric_summary.py" "$E/reset-rgb-capture-repeat" \
  --previous-root "$E/slot-stability-followup-repeat"
```

The released command.json and plan.json files contain the exact executed arguments.
Both comparisons use reduced diagnostic horizons. They do not establish native-horizon
accuracy equivalence, indefinite memory stability or the full-workload ≥2× target.
The approved 50×5 native correctness count is distinct from the official default
50×50 matrix; this follow-up does not reinstate 2,500 as a required diagnostic count.

[Raw results, samples, cleanup checks, frozen sources and reproduction harness](https://github.com/QLL-CCDD/RoboScope-Kit/releases/tag/robocasa365-review-followup-evidence-20261109) · [Earlier atomic pilot](slot-concurrency-20261109.md) · [Native comparison](native-comparison-20261109.md)
