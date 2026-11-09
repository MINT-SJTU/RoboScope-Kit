# LIBERO-Pro issue #21 audit

Reviewed 2026-11-08 against [issue #21](https://github.com/MINT-SJTU/RoboScope-Kit/issues/21), its
[integration baseline #19](https://github.com/MINT-SJTU/RoboScope-Kit/pull/85), the local candidate
`8ca22d139827916d440673029bc213cb2a0773ff`, and the recorded AutoDL experiment. This is an audit and a
prioritized investigation plan; issue #21 remains open and the 2x target is not met. The later
[full-matrix report](FULL_AUTO_2026-11-09.md) records the 2026-11-09 results and supersedes the
single-suite status below where noted.

## Acceptance boundary

The standard supported matrix is 16 released single-perturbation suites, each 10 tasks x 50 initial
states: 8,000 episodes per comparison arm. Environment perturbations and combined perturbations are not
supported by the frozen #19 integration; they must be separately defined and scored if supported later,
not counted as a completed part of this standard profile. Preserve the base-suite horizons, seed 7,
10 settle steps, camera observations, eight-action StarVLA chunk, per-step termination/success checks,
and official per-suite/task scores. The frozen single-perturbation files already encode the chosen
perturbations; the runtime does not use `evaluation_config.yaml` to generate them at evaluation time.

The target compares the sum of completed episodes divided by the sum of **formal evaluation time**
across the complete matrix, under equivalent hardware and protocol: candidate / baseline >= 2.0.
Actual executed steps/s, preparation, model loading, tuning, output and total elapsed time are reported
separately. A single suite or a CPU communication benchmark cannot establish full-matrix acceptance.

## What has been done

| Requirement | Evidence and limit |
| --- | --- |
| Baseline protocol and pins | #19 is merged. The released suites, LIBERO-Pro source `eafdb809`, task-file revision `c86fc3b`, 320 task-file SHA-256 digests, Python 3.11.18 runtime and StarVLA checkpoint `a60fff4` are pinned. The #19 8,000-episode correctness run used 13 suites on an RTX 3080 at K2/g1 and three suites on a shared RTX 4090 at K8/g2; it is not an equivalent-hardware throughput denominator. |
| Existing runtime capabilities | Both tested commits already have Workers, Policy replicas, a durable queue, task-group affinity, task-environment reuse, eight-action chunks and `diagonal` K/g autotune. Those are not newly implemented speedups in this candidate. |
| Candidate code | The LIBERO-Pro manifest prefers the existing negotiated MessagePack encoding; JSON fallback remains available. Per-episode socket reuse is separately configurable and off by default. Worker results record the negotiated encoding. Failed requests are not automatically replayed. |
| Clean runtime build | On AutoDL, the committed `build-runtime.sh` built the candidate runtime into a new directory, using a local source mirror whose Git blobs were checked against the pinned commit, checksum-verified task files, and dependency mirrors/offline wheelhouse. Its Python 3.11.18 checks, all-suite checks and `--self-test` passed. This is not a default-URL or measured cold-download reproduction. |
| Real-model smoke and baseline pair | The pinned 3B checkpoint was SHA-256 checked, bf16 model inference passed and Quick Test completed. One 500-episode `libero_goal_object` pair was run on the same AutoDL RTX 4090 instance, sequentially at fixed K12/g3: old Kit `44fa1ad`/0.8.0b30 JSON versus candidate `8ca22d1`/0.8.0b31 MessagePack, both with per-request sockets. Both used the same model snapshot, simulator, initial-state indices and output configuration. |
| Accounting and diagnostics | All 500 expected task/episode IDs in each arm occur once; the candidate reported MessagePack in all 500 WorkResults; no Worker failures or OOMs occurred in the final pair. All 320 frozen task-file digests were rechecked. CLI wall, formal time, step count, CPU/GPU statistics, container memory samples, per-task results and paired results were recorded. |
| Local checks and documentation | Python 3.12 existing regressions passed 187 tests, required Ruff passed, and CPU transport/fallback checks passed. Version 0.8.0b31, lock, changelog and evaluation docs were updated. The candidate has a local commit and was pushed to the user's fork; it is not merged upstream. |

The AutoDL pair's report, raw results and commands are under
`/root/autodl-tmp/issue21-20261108/comparison/paired-20261108-195313/` on that machine. The original
experiment `r1` candidate stopped after 392 episodes and left three orphaned Policy processes; the
later `r2` attempts failed at model loading with OOM while those processes still held VRAM. Those
attempts were retained and excluded from the final pair. The orphaned processes were verified and
stopped before that pair. This observation warrants investigating interrupted-run process cleanup; it
does not establish that the candidate itself caused the interruption.

## The measured pair

| Metric | `44fa1ad` JSON baseline | `8ca22d1` MessagePack candidate |
| --- | ---: | ---: |
| Completed episodes | 500 | 500 |
| Successes | 332 (66.4%) | 339 (67.8%) |
| Actual policy action steps | 84,212 | 82,948 |
| Formal evaluation | 431.742 s | 424.384 s |
| Formal throughput | 1.1581 episodes/s | 1.1782 episodes/s |
| Actual policy steps/s | 195.052 | 195.455 |
| CLI wall, excluding warmup/install | 464.570 s | 457.864 s |
| Selected K/g | fixed 12/3 | fixed 12/3 |

This one pair gives **1.0173x formal episodes/s**, **1.0021x actual policy steps/s**, and **1.0146x
CLI wall speedup**. The candidate executed about 1.50% fewer policy steps; its shorter evaluation
cannot be attributed entirely to communication. The reported action steps exclude 5,000 settle steps
per arm. Nineteen paired episode success flags and 46 step counts differ; their causes and possible
systematic regressions are not established. This is not statistical equivalence.

#19 reported 351/500 (70.2%) for this suite on a different RTX 3080 and K2/g1. Our 332/500 and 339/500
are useful correctness signals to investigate, not a matched comparison with that run. The AutoDL pair
uses SDPA and `OMP_NUM_THREADS=1`; confirm the #19 baseline's precise model environment and attention
configuration before making a controlled reproduction claim. Matching episode IDs and init-state indices
alone does not prove identical starting scenes: #19 documents reset-history dependence when task
environments move between Workers.

The two arms' sampled mean GPU utilization was about 96.5% and 96.9%, and peak device memory about
31.6 and 31.7 GiB respectively. These device-wide figures include inference and rendering; they do
not identify the bottleneck. The local CPU-only socket experiment (7.890 to 1.512 ms/request) measures
the communication path, not full model or simulator speed.

## Still required or only partially done

1. **Full matrix:** one baseline/candidate pair of 8,000 episodes per arm is complete with episode and
   score coverage verified. Repeat it on a controlled resource budget and investigate its correctness
   differences against matched starts and the historical #19 baseline.
2. **Repeated evidence:** repeat paired same-hardware comparisons, alternate run order, report spread,
   and investigate 19 success/46 step differences in the original pair and the later full-matrix
   differences. Include per-task results and the
   difference from the historical #19 correctness baseline. Match caches, output settings and mode.
3. **Bottleneck breakdown:** measure environment construction, reset/settle, physics, image render and
   postprocessing, Worker-side communication, server queue/lock wait, actual inference, output and
   GPU memory. The current `inference_wait_s` combines socket/queue/server/model waiting and is not
   pure GPU inference. Sums across concurrent Workers are not elapsed wall time. Profile Worker and
   Policy Server separately, then time unprofiled runs.
4. **Tuning and end-to-end:** the original transport pair fixed K/g, so it measures no autotune benefit. Compare
   representative fixed K/g settings against actual `diagonal` choices under the same host budget;
   measure tuning and cold preparation separately. CLI wall covers the evaluation process, not the
   original runtime build/download and warmup. Do not mix those timing boundaries.
5. **Optimization and correctness:** no evidence yet demonstrates >=2x. Real-GPU MessagePack versus
   JSON was not isolated within the same candidate commit. Separate single-suite comparisons of
   render deferral, MPS and native batching exist, but their correctness differences need review;
   connection reuse and simulator vectorization still need matched real-workload evidence.
6. **Delivery:** the full-matrix report and machine-readable scores are in the repository; complete
   raw logs and the launcher remain on AutoDL. Preserve them and run the full supported-Python
   build/install/import/CI matrix before merging. Reconcile the tested 0.8.0b37 version and
   merge-date changelog against main. No upstream merge has occurred.

The #19 reference included a 200-episode, four-suite native correctness comparison, not a matched native
throughput claim. A new native comparison at equivalent concurrency is needed only for a new claim
that the candidate beats the native evaluator; it is distinct from Kit-before/after acceptance.

## What to borrow, and when

| Reference | Proven result and relevant mechanism | Application to LIBERO-Pro |
| --- | --- | --- |
| [RoboDojo #93](https://github.com/MINT-SJTU/RoboScope-Kit/pull/93), merged | MessagePack reduced a 3-frame 640x480 socket round trip from 52.6 to 3.0 ms, while its three-episode real run improved about 1.10x. | Reuse the protocol, as done. On our workload the real pair shows only 1.017x episodes/s, with differing steps. Record and retain the ablation, but do not assume transport can achieve 2x. |
| [RoboCasa #39](https://github.com/MINT-SJTU/RoboScope-Kit/pull/39), draft | At K8/g1, 48 episodes: render deferral changed formal evaluation 254.2 to 236.8 s, about 1.07x; simulator time improved about 1.21x. | If profiling shows discarded image rendering is material, investigate skipping only intermediate camera observations in an eight-action chunk. LIBERO's early termination, sensor timing, videos, settling and resets need an independent audit and fallback. Do not port RoboCasa's per-episode reseeding into LIBERO-Pro. |
| [CALVIN #22](https://github.com/MINT-SJTU/RoboScope-Kit/pull/22), draft | Queue already-waiting requests from distinct episodes into a genuinely batched StarVLA call; no batch timer. The historical 2.33x changed concurrency and model source and is not an isolated batching result. | If Policy inference/queueing dominates, prototype native `predict_action` batching and measure latency, batch distribution, VRAM and correctness. Keep per-episode identity and output decoding; preserve the newer MessagePack protocol when adapting the old PR. |
| [Autotune #81](https://github.com/MINT-SJTU/RoboScope-Kit/pull/81) and [#104](https://github.com/MINT-SJTU/RoboScope-Kit/pull/104), merged | #81 continues replica exploration after `no_steady_window`; #104 rejects a dead replica during probes. A RoboCasa full run changed K8/g1 to K16/g2, 1.455x formal throughput. | Upstream main contains these fixes, but neither tested commit does. Sync/rebase before future changes and rerun matched measurements; our single LIBERO-Pro search did reach K12/g3, so no evidence shows #81 would improve it. The fix does not predict host memory limits. |
| [Autotune testbed #83](https://github.com/MINT-SJTU/RoboScope-Kit/pull/83), merged | Records real probe/fixed-run results and replays search strategies offline; missing configurations are reported as unrecorded, not invented. | Record a representative LIBERO-Pro workload after collecting matched probes and fixed runs. Replay helps diagnose selection changes without repeatedly spending GPU time, but does not replace real-model validation. |
| [MetaWorld historical #9](https://github.com/suica/evostudio-evaluation-kit/pull/9), merged in former repository | A separate fixed-K3/g3 CUDA MPS off/on experiment reported 1.308x including loading; 278 paired episode outcomes and 64 initial image hashes differed. | Try the existing opt-in `--cuda-mps` only as a controlled same-K/g ablation when profiling supports it. Report cost and outcome differences. The historical number is not a LIBERO-Pro estimate. |
| [RoboDojo #102](https://github.com/MINT-SJTU/RoboScope-Kit/pull/102), draft; [RoboCasa365 #107](https://github.com/MINT-SJTU/RoboScope-Kit/pull/107), draft | Investigate multiple simulator environments or parallel native slots; both retain important correctness limitations. | Longer-term research only. Current K means separate one-environment Workers, not native vectorized MuJoCo. Independent resets, termination, seed history and memory must be proven before using such a path. |

Suggested order: (1) reconcile exact baseline inputs and instrument a small real-model workload; (2)
repeat paired JSON/MessagePack and fixed-K/g sweeps; (3) choose batching or render work from the measured
bottleneck; (4) validate a small paired set at consumed-observation/action/physics boundaries; and
(5) rerun the full matched matrix and timing/score audit. Avoid changing seed semantics or accepting
weaker coverage just to improve throughput.
