# Performance work reviewed for LIBERO-Pro

Reviewed: 2026-11-07; PR states and our own results updated 2026-11-08. Target:
[issue #21](https://github.com/MINT-SJTU/RoboScope-Kit/issues/21). The candidate and its tested baseline
are based on `44fa1ad`, not the latest upstream main. See [the issue audit](ISSUE_21_AUDIT.md) for current
implementation and experiment status.
This is a review of existing evidence and candidate priorities, not acceptance of LIBERO-Pro's 2x target.

## Evidence overview

| Work | Mechanism | Relevant measured result | Delivery and limitation |
| --- | --- | --- | --- |
| [RoboDojo #93](https://github.com/MINT-SJTU/RoboScope-Kit/pull/93) | MessagePack arrays instead of base64 JSON | CPU/socket round trip 52.6 to 3.0 ms; a three-episode real run 637 to 577 s, about 1.10x | Merged. The real run is one task and three layouts, not a full benchmark acceptance. |
| [RoboCasa #39](https://github.com/MINT-SJTU/RoboScope-Kit/pull/39) | Defer unconsumed intermediate RGB observations | On the reported PR head, 48 episodes at K=8/g=1: evaluation 254.2 to 236.8 s, about 1.07x; summed simulator time 1077.9 to 890.1 s, about 1.21x | Draft, unmerged. Outcomes, seeds and executed steps pair up on all 48 episodes. |
| [CALVIN #22](https://github.com/MINT-SJTU/RoboScope-Kit/pull/22) | Native batching of queued requests from distinct episodes | Historical concurrent/batched run: 4344.8 to 1860.8 s, 2.33x | Draft, unmerged. Historical comparison changes concurrency and model source; 2.33x cannot be attributed to batching. No real-model validation on the new Kit. |
| [Autotune #81](https://github.com/MINT-SJTU/RoboScope-Kit/pull/81) | Continue replica search after a probe has no steady window | RoboCasa, 1200 episodes: K=8/g=1 at 5457.1 s to K=16/g=2 at 3750.0 s, 1.455x formal throughput | Merged upstream after our tested baseline; absent from `44fa1ad` and `8ca22d1`. No guarantee of a gain on LIBERO-Pro. |
| [Historical MetaWorld #9](https://github.com/suica/evostudio-evaluation-kit/pull/9) | Joint K/g tuning, then optional CUDA MPS | Autotune: 2.202x formal, 1.486x including tuning. Separate fixed-K3/g3 MPS experiment: 1.308x including loading/preparation/cleanup | Merged in the former repository. Different historical environments; not current Kit acceptance. These ratios must not be multiplied. |

K is the number of simulation Workers; g is the number of Policy replicas. Replica conventions must also
record GPU count. A higher number of Workers is not the same as multiple environments inside one
simulator process or a larger native inference batch.

## What each solution actually does

### Binary Policy transport: measured, reusable, limited by its share of runtime

RoboDojo sends three 640x480 RGB frames on each required observation update. #93 measured JSON/base64
encoding as a substantial extra cost. Its MessagePack representation preserves array dtype, shape and
contents. The receiver still owns writable arrays. A JSON `health` exchange negotiates support, older
servers stay on JSON, and the server replies using the request's encoding.

Only RoboDojo enables it by default on the reviewed main. LIBERO-Pro can reuse this implementation;
there is no need to design a second binary protocol. Its two 256x256 cameras and the StarVLA demo's
eight-action chunk produce a different communication workload, so RoboDojo's measured gain is not
transferable numerically.

The 17.5x socket microbenchmark gain became approximately 1.10x on the reported real run. This is the
important distinction between reducing one operation's latency and accelerating formal evaluation.

### RoboCasa: remove observations that are provably discarded

The native wrapper retains only the last observation from each 12-action chunk. The optimization skips
the other 11 steps' RGB acquisition and discarded crop/resize/copy work, while still running every
physics step, reward calculation and success check. Images consumed by the Policy are freshly rendered.

The implementation keeps native sensor timing and falls back when it cannot preserve semantics: videos,
multiple-frame histories, native early termination, depth/segmentation, or temporal sensor processing.
It also isolates mutable distractor configuration and fixes scene seeding so paired comparisons are
meaningful. Those fixes apply to both sides of the comparison.

The current-head 48-episode comparison demonstrates a simulator improvement, but inference limits the
overall gain at K=8/g=1. Large headlines in this work use different comparisons:

- Historical Kit-internal 3.93x changed K=1 to K=4 and enabled deferral together.
- The [matched independent StarVLA control](https://github.com/suica/evostudio-evaluation-kit/pull/6)
  applied equivalent parallelism and deferral to both sides: 1200 paired episodes, identical successes
  and steps, and only 1.000545x formal throughput. That is parity, not a material Kit advantage.
- The newer 10.6x to 16.6x figures versus the serial official entry point also include concurrency,
  video and transport differences. They do not isolate deferral or establish an equally optimized
  native comparison. Full runs were performed once per configuration; shorter repeats show day-to-day drift.

### CALVIN: keep native batching; discard low-yield additions

#22 collects only already-queued `act` requests from distinct episodes and sends one native `act_batch`
call. It adds no collection timer, maintains request/response identity, serializes lifecycle calls, and
propagates failures to waiting clients. StarVLA must perform a genuinely batched `predict_action` call;
raising `max_batch_size` while retaining the default loop over `act` does not provide GPU batching.

The [historical ablation](https://github.com/suica/evostudio-evaluation-kit/pull/7) removed:

| Addition | Historical change from the preceding group | Decision |
| --- | ---: | --- |
| Persistent connections plus FIFO/late collection | +2.78% sequence throughput | Removed |
| CPU pipeline and lifecycle concurrency | -1.05% | Removed |
| Whole-batch CPU processing and image-copy shortcuts | +1.33%; only +0.27% over FIFO group | Removed |

These were grouped, single full-run measurements, so they neither isolate connection reuse nor prove
a stable ordering of small gains. The useful lesson is to require reproducible end-to-end benefit before
retaining lifecycle complexity. Native batching itself remains a candidate pending a controlled,
same-source comparison. The current default batch size is still one.

#22's server patch predates #93's MessagePack implementation. Integrating batching now requires
preserving the newer framing and encoding negotiation; copying its old server file would lose them.

### Autotune: distinguish lack of measurement from lack of capacity

#81 addresses a search failure on RoboCasa. A task-sized work catalog can stop supplying enough work
for a steady-state window at larger K. The earlier main treated this as a resource ceiling and skipped replica
exploration. The patch ends only the K phase for `no_steady_window`, continues replica probes, and
compares both candidates using elapsed rates whenever either lacks a steady window. OOM, failures,
timeouts and budget exhaustion still stop unsafe exploration.

This is now merged upstream, along with the separate [dead-replica probe fix #104](https://github.com/MINT-SJTU/RoboScope-Kit/pull/104);
neither is in our tested branch. This changed the RoboCasa real run's selection from 8/1 to 16/2. A fixed sweep still found 12/2 slightly faster,
so autotune is a measured heuristic, not proof of the global optimum. Its replay evidence does not
change the search result on the recorded plain LIBERO workload. Do not assume that LIBERO-Pro has
RoboCasa's work-item limitation; its factory already grows per-task probe episodes with K.

### MPS and simulator concurrency: separate mechanisms

The historical MetaWorld work first made probes representative and searched K jointly with g. It then
compared MPS off/on at fixed K3/g3, without rerunning tuning. The MPS comparison had equal total success
counts but 278 different per-episode outcomes and 64 different initial image hashes. Equal aggregate
scores therefore did not establish trajectory identity or statistical equivalence.

Current Kit already has `--cuda-mps` and an owned MPS lifecycle. It connects Policy replicas, not simulator
Workers. Reuse it only when profiling and hardware justify a multi-replica experiment; it does not
automatically vectorize MuJoCo or make an oversized model fit in memory.

RoboDojo's [tracking discussion](https://github.com/MINT-SJTU/RoboScope-Kit/issues/18#issuecomment-6656099481)
also distinguishes multiple single-environment Workers from native `num_envs` vectorization. CUDA Graphs
for PhysX and native vectorization remain investigations, not demonstrated current-Kit speedups.
The placeholder-Policy trace's approximately 11% CUDA-kernel busy fraction excludes Policy Server work
and graphics; it is not whole-device occupancy or proof that all remaining time is removable launch overhead.
Earlier async-video, render-pacing and physics-thread experiments found no stable gain. The original
[recipe PR #14](https://github.com/MINT-SJTU/RoboScope-Kit/pull/14) is closed without merging; retained
candidate switches in main are experimental and off by default.

## Application to LIBERO-Pro #21

1. **Reproduce a fair baseline first.** #19's 8000 episodes used mixed GPUs and K/g settings. Re-run the
   full 16-suite matrix on a controlled resource budget before using a baseline as the 2x denominator.
   Reuse the frozen checkpoint, BDDL/init files, instructions, seed semantics and output configuration.
2. **Use small ablations to identify the dominant cost.** Separate model request wait from actual
   inference, simulator/rendering, initialization/reset, scheduling and I/O. Profile the Policy Server
   separately; Worker `--profile` does not include its GPU kernels. Time performance without profiling.
3. **Measure K/g before adding a new scheduler.** The existing queue already prefers the current task
   group, and the LIBERO Worker already retains its current task environment. Compare fixed settings
   with the selected configuration, inspect probe termination reasons, and keep CPU/VRAM budgets matched.
4. **Treat MessagePack as an isolated, reusable candidate.** Measure JSON versus MessagePack at the
   same K/g and connection policy. Record the negotiated encoding, including fallback. A fast socket
   measurement is useful evidence about transport, not the 2x acceptance result.
5. **Keep connection reuse as a separate ablation.** The current implementation draft allows it to be
   disabled. Historical CALVIN evidence argues against treating it as the principal route to 2x.
   Close episode resources and do not replay state-changing requests after a transport failure.
6. **Investigate consumed-frame rendering only if rendering dominates.** LIBERO-Pro's demo requests
   actions in eight-step chunks, but the Worker still steps and obtains observations each action.
   Audit LIBERO's own camera timing, termination and observation dependencies before porting RoboCasa's
   idea. Preserve every physics step and required observation, including videos and settle steps.
7. **Consider native batching only if inference dominates.** Coordinate with #22 instead of introducing
   another queue. Validate bf16 outputs, action decoding, memory and state isolation with the real model.
8. **Do not import another benchmark's seeding fix as a performance optimization.** LIBERO-Pro currently
   seeds a task environment once; reset history affects starts. Per-episode reseeding would change the
   established protocol. Any necessary protocol correction requires correcting both comparison groups
   and rebuilding their baseline.

For a fixed-concurrency illustration, if transport is 10% of total runtime, making it seven times faster
gives only `1 / (0.9 + 0.1 / 7) = 1.094x` overall, assuming the other costs are unchanged. Measurements of
the full pipeline are required to establish a 2x target.

## Acceptance discipline shared by the useful examples

- Distinguish Kit before/after, comparison against serial native code, and comparison against equally
  optimized native code. Name the denominator of every speedup.
- Keep complete episode identities and actual executed steps, even for failures, retries and short
  trajectories. Report per-task and per-suite official metrics, not only an aggregate success count.
- Separate formal evaluation from loading, preparation, tuning, teardown and output writing. For the
  full matrix, divide total completed episodes by the corresponding total formal time rather than
  averaging suite speedup factors.
- Use paired/repeated runs and document uncontrolled randomness and co-tenant load. Equal totals or
  overlapping confidence intervals alone do not establish equivalence.
- Preserve failed experiments and negative results. Remove low-yield complexity when the end-to-end
  benefit is not reproducible.
- Freeze runtime changes in build scripts and pins. If assets or protocol change, rebuild both baselines.
  Current PR #39 explicitly notes uncommitted mirror edits in its measured build; clean reproduction is
  a separate acceptance item and must not be inferred from those results.

The local machine's 6 GiB GPU cannot run the pinned bf16 baseline. An AutoDL 48 GiB GPU was subsequently
used for a clean simulator build, real-model Quick Tests and a full matrix per comparison arm. The
[full-matrix report](FULL_AUTO_2026-11-09.md) records 1.3791x formal throughput and the observed
correctness differences. Repeated runs and the 2x result remain outstanding.
