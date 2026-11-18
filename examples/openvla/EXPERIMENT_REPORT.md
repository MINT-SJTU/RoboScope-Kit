# OpenVLA experiment report

Updated: 2026-11-14.

Current implementation: **Kit `0.8.0b65`, code commit `14d2f50`**.
The completed Native validation and README update are recorded in documentation
commit `84fcc95`. The performance campaign used `0.8.0b58` / `da280f0`; those
measured versions remain identified separately below.

This report summarizes the completed evaluation of the updated RoboScope Kit
integration and the subsequent Goal investigations. It contains settings, key
measurements and conclusions; raw logs, images and experiment archives are not
included. The existing detailed report is retained separately.

The main performance measurements predate the final review fixes to the Native
entry point and normalization-key fallback. The additional 200-episode Native
validation is complete and recorded in section 6. Kit was not rerun: all four
checkpoints contain the original suite key, so the fallback leaves their inference
key unchanged. Existing performance results retain their original provenance.

## 1. Evaluation scope and settings

The main comparison covers LIBERO-Spatial, Object, Goal and Long (`libero_10`).
Each suite uses its corresponding fine-tuned OpenVLA-7B checkpoint and all ten
tasks, with initial-state indices 0–4: 50 episodes per suite, 200 per arm. This
covers the complete agreed workload, not the standard 500 episodes per suite. Kit was
rerun after updating its Policy-inference and simulation environments to follow
Kit's standard runtime layout. The official OpenVLA Native environment is not
changed to follow Kit's dependency versions; this accounts for the retained
runtime-version differences.

| Setting | Kit | Native |
| --- | --- | --- |
| Evaluated implementation | Kit `0.8.0b58`, source `da280f0` | Pinned official OpenVLA evaluator, source `c8f03f4`; configuration unchanged |
| Execution | Two simulator Workers / two Policy replicas (K/g=2/2) | Serial evaluation |
| Runtime layout | Separate Policy/CLI and standard LIBERO Worker environments | Original environment retained |
| Python | Policy 3.11.15; Worker 3.10.20 | 3.10.13 |
| Simulator | robosuite **1.4.0**, MuJoCo 3.2.3 | robosuite **1.4.1**, MuJoCo 3.2.3 |
| Image preprocessing | TensorFlow on CPU | TensorFlow on CPU |
| Model inference | GPU BF16, FlashAttention2, greedy decoding | GPU BF16, FlashAttention2, greedy decoding |
| Executed action representation | float32 | float64-valued actions |

Both paths use model/environment seed 7, ten settling steps and one generated
action per inference call. Maximum policy steps are 220 for Spatial, 280 for
Object, 300 for Goal and 520 for Long. Videos and MPS are disabled. Formal timing
runs do not collect extra observation/action trajectories.

### Current K/g performance and configuration selection

We reran the configuration study to follow the updated Kit implementation.
Completed November 14: four fixed configurations and Autotune, 200 formal episodes each
(1,000 total). K denotes simulator Workers and g denotes Policy replicas.
All use the settings above on one RTX 4090, with checkpoints and outputs on the
AutoDL data disk. Native is unchanged by Kit updates, so its existing results
remain the comparison reference.

The conclusion matches the historical Kit configuration study despite changes
to the Policy inference path, including CPU preprocessing: retain **2/2**.
It has the highest fixed-configuration throughput and shortest formal evaluation
time, while success totals remain close at 150–152/200. Compared with 1/1,
throughput increases **22.71%** and formal time decreases **18.94%**.
4/2 is nearly tied in throughput (0.22% lower), but 2/2 has lower service latency
and memory use, supporting the same practical configuration choice.

Success rates vary little across the tested K/g configurations, so subsequent
experiments use K/g=2/2 as the representative Kit configuration.

#### Successes

| K/g | Spatial | Object | Goal | Long | Total / 200 | Success rate |
| --- | --- | --- | --- | --- | --- | --- |
| 1/1 | 41 | 43 | 36 | 30 | 150 | 75.0% |
| 4/1 | 40 | 44 | 39 | 29 | 152 | 76.0% |
| 2/2 | 40 | 43 | 38 | 30 | 151 | 75.5% |
| 4/2 | 40 | 44 | 39 | 29 | 152 | 76.0% |
| Autotune | 40 | 43 | 38 | 30 | 151 | 75.5% |

Each suite column is out of 50. Autotune probe episodes are excluded.

#### Timing and throughput

| K/g | Policy steps | Formal min | Steps/s | Episodes/min | Policy load min | Tuning min | Command min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1/1 | 42106 | 152.57 | 4.600 | 1.311 | 4.74 | 0.00 | 157.47 |
| 4/1 | 41742 | 134.69 | 5.165 | 1.485 | 0.52 | 0.00 | 135.39 |
| 2/2 | 41886 | 123.68 | 5.644 | 1.617 | 0.93 | 0.00 | 124.77 |
| 4/2 | 41860 | 123.88 | 5.632 | 1.615 | 0.96 | 0.00 | 125.02 |
| Autotune | 41886 | 123.52 | 5.652 | 1.619 | 0.01 | 85.83 | 209.53 |

Aggregate throughput is summed policy steps / summed formal wall time, not averaged suite rates. Formal time excludes loading/tuning. Policy load is the SDK stage wall time, not summed parallel replica load durations. For Autotune, model loads inside probes are charged to tuning; the near-zero later Policy load field is not total model-loading cost. Command time already includes tuning, formal execution and overhead.

The first 1/1 arm loaded for 4.74 min, compared with 0.52–0.96 min for later fixed arms. Runs were sequential with shared OS cache, without deliberate warming or cache clearing. Storage/cache effects confound complete-duration comparisons; do not attribute loading reductions to inference or K/g alone. Different success outcomes also change executed step counts.

#### Latency and GPU sampling

| K/g | Mean service ms | P95 ms | P99 ms | Preprocess mean ms | Predict mean ms | GPU mean % | GPU peak GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1/1 | 176.55 | 180.89 | 186.79 | 7.52 | 166.27 | 71.48 | 15.82 |
| 4/1 | 192.54 | 207.10 | 218.32 | 8.54 | 180.88 | 80.43 | 17.57 |
| 2/2 | 307.82 | 348.02 | 350.74 | 7.58 | 295.79 | 93.09 | 31.50 |
| 4/2 | 349.35 | 369.97 | 377.24 | 8.67 | 336.73 | 95.12 | 32.54 |
| Autotune | 307.67 | 347.98 | 351.21 | 7.60 | 295.45 | 93.09 | 31.50 |

Latency statistics pool every formal request across all suites, including first requests; P95/P99 use linear interpolation, not averages of suite percentiles. Service timing includes preprocessing, inference and action conversion but excludes queue/socket time; Native uses a different boundary.

GPU mean is weighted by valid formal sample counts. Sampling waits 0.5 s between queries; actual cadence also includes query time. Peaks are observed samples, not guaranteed absolute maxima. Missing/failed samples are excluded, not zero-filled. Sampler NoSuchProcess counts were 11/25/27/61/37 across the five arms; these are process disappearance during resource inspection, separate from simulator Worker failure counters. The source sampling audit records phase attribution. No formal inference errors, Worker failures or OOM were recorded.

#### Unlimited Autotune

| Suite | Selected K/g | Candidate path | Tuning min | Formal min |
| --- | --- | --- | --- | --- |
| spatial | 2/2 | 1/1 → 2/1 → 2/2 | 12.29 | 19.24 |
| object | 2/2 | 1/1 → 2/1 → 2/2 → 3/3 | 20.83 | 24.24 |
| goal | 2/2 | 1/1 → 2/1 → 2/2 → 3/3 | 19.54 | 22.72 |
| 10 | 2/2 | 1/1 → 2/1 → 2/2 | 33.16 | 57.32 |

Both total and candidate time limits were infinite. Fourteen probes completed, each with ten episodes (one state per task), for 140 probe episodes in addition to 200 formal episodes. Spatial/Long stopped before 3/3 on predicted memory; Object/Goal completed 3/3 and stopped on GPU-utilization limits. Search ended on algorithm/resource rules, not budget. Total tuning cost: 85.83 min; full Autotune command time: 209.53 min.

#### Per-suite measurements

| K/g | Suite | Steps | Formal min | Command min | Steps/s | GPU mean % | GPU peak GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1/1 | spatial | 6386 | 24.04 | 25.38 | 4.427 | 69.22 | 15.51 |
| 1/1 | object | 8249 | 29.99 | 31.20 | 4.585 | 70.99 | 15.45 |
| 1/1 | goal | 7963 | 28.79 | 29.96 | 4.610 | 71.56 | 15.40 |
| 1/1 | 10 | 19508 | 69.75 | 70.92 | 4.661 | 72.45 | 15.82 |
| 4/1 | spatial | 6563 | 21.43 | 21.63 | 5.103 | 79.58 | 17.11 |
| 4/1 | object | 8092 | 26.25 | 26.41 | 5.137 | 80.01 | 16.87 |
| 4/1 | goal | 7524 | 24.31 | 24.48 | 5.158 | 80.13 | 16.77 |
| 4/1 | 10 | 19563 | 62.70 | 62.87 | 5.200 | 81.01 | 17.57 |
| 2/2 | spatial | 6476 | 19.22 | 19.50 | 5.617 | 92.26 | 31.02 |
| 2/2 | object | 8249 | 24.19 | 24.46 | 5.684 | 93.71 | 30.90 |
| 2/2 | goal | 7653 | 22.66 | 22.93 | 5.629 | 92.82 | 30.79 |
| 2/2 | 10 | 19508 | 57.62 | 57.88 | 5.643 | 93.20 | 31.50 |
| 4/2 | spatial | 6563 | 19.35 | 19.68 | 5.654 | 95.39 | 32.09 |
| 4/2 | object | 8210 | 24.38 | 24.66 | 5.611 | 94.67 | 31.84 |
| 4/2 | goal | 7524 | 22.26 | 22.53 | 5.634 | 95.15 | 31.71 |
| 4/2 | 10 | 19563 | 57.89 | 58.16 | 5.633 | 95.21 | 32.54 |
| Autotune | spatial | 6476 | 19.24 | 31.57 | 5.611 | 91.99 | 31.02 |
| Autotune | object | 8249 | 24.24 | 45.12 | 5.671 | 93.30 | 30.90 |
| Autotune | goal | 7653 | 22.72 | 42.31 | 5.613 | 92.49 | 30.79 |
| Autotune | 10 | 19508 | 57.32 | 90.53 | 5.672 | 93.62 | 31.50 |

All 1,000 formal episodes passed unique task/state coverage and request/step-count checks. Autotune selected 2/2 in every suite; its formal episode outcomes and step counts match fixed 2/2 exactly.

## 2. Main four-suite results and performance

The Kit column uses the latest fixed 2/2 run from the configuration campaign.
Native retains the previous four-suite reference; no Native rerun was needed
for the Kit-only update.

| Suite | Kit success | Native success | Kit eval (min) | Native eval (min) |
| --- | ---: | ---: | ---: | ---: |
| Spatial | 40/50 | 43/50 | 19.22 | 22.96 |
| Object | 43/50 | 42/50 | 24.19 | 28.66 |
| Goal | 38/50 | 33/50 | 22.66 | 29.11 |
| Long | 30/50 | 28/50 | 57.62 | 64.20 |
| Total | 151/200 | 146/200 | 123.68 | 144.93 |

Kit executes 41,886 policy steps and Native 42,708. Aggregate throughput is
**5.644 versus 4.911 steps/s** (summed steps / summed formal time).
Kit formal evaluation time is **14.66% shorter**. These are evaluation
measurements including concurrency and trajectory differences.
Formal timing excludes loading, tuning and diagnostic recording. Kit's four
Policy-load stages total **55.79 seconds**. All configuration arms use local
data-disk checkpoints; sequential runs share the OS cache, so loading differences
are reported separately from formal evaluation speed.

The existing consistency campaign below is reported separately from these
configuration runs; its repetition counts are unchanged.

## 3. Repeatability and Long comparison

| Workload, states 0–4 | Completed non-recording rounds | Successes per round | Policy steps per round |
| --- | --- | --- | --- |
| Long, Kit | 3 | 30/50 in each | 19,508 in each |
| Long, Native | 3 | 28/50 in each | 19,595 in each |
| Goal, Kit | 5 | 38/50 in each | 7,653 in each |
| Goal, Native | 3 | 33/50 in each | 8,516 in each |

Within each arm and suite, every episode repeats its outcome and step count.
Baseline rounds are included in these counts. Repeated cases are not independent
new evaluation identities; the extra recorded Goal run is accounted for separately.

The earlier Long deficit favoring Native is no longer observed after the update.
Kit moves from 23/50 to 30/50, while Native remains at 28/50. The directional
outcome split changes from four Kit-only / nine Native-only to **six Kit-only /
four Native-only** successes. Each current comparison also has 24 both-success
and 16 both-failure cases. Thus the earlier pronounced deficit is removed, but
ten per-case disagreements remain. Multiple integration/runtime factors changed;
the experiments do not isolate a single cause for the improvement.

## 4. Goal investigation: motivation, design and results

An earlier 36-pair diagnostic across the four suites motivated closer examination
of observations, actions and execution conditions. It also motivated comparing
selected-case diagnostics with complete-workload execution. Its detailed results
are not repeated here; the focus is the November 13 Goal experiments.

OpenVLA has no cross-episode recurrent memory: each episode starts inference
afresh. Nevertheless, matching task/state labels alone does not ensure identical
complete scene conditions. To make the comparison more precise and reproducible,
we retained the full 50-case workload while recording selected cases, then added
another 50-case block per arm to examine broader initial-state coverage.

| Experiment | Execution and recording | Kit | Native | Main purpose |
| --- | --- | ---: | ---: | --- |
| Goal seven-case subset | Run only the seven selected identities; record every step | 7/7 | 7/7 | Inspect observations, model inputs and actions for the targeted cases |
| Goal full workload, selective recording | Run all 50 states0–4 cases per arm; record only the same seven identities | 38/50 | 33/50 | Reproduce the complete evaluation while capturing the targeted trajectories |
| Goal additional states | Run ten tasks × states5–9, 50 cases per arm; no extra trajectory recording | 36/50 | 35/50 | Compare both paths on a new set of initial-state identities |

### Full execution versus selected-case execution

The recorded full run reproduces every episode's outcome and step count from
its respective non-recording baseline. All seven original outcome disagreements
reappear, and the selected observation/action records pass validation. Selected
Kit traces contain six successes; Native traces contain one. This differs from
running only the subset, where both arms complete all seven cases successfully.

| Task / state | Full-run Kit outcome / steps | Full-run Native outcome / steps | Subset Kit / Native steps; both successful |
| --- | --- | --- | --- |
| 0/2 | Success / 109 | Failure / 300 | 109 / 180 |
| 0/4 | Success / 174 | Failure / 300 | 120 / 123 |
| 5/3 | Success / 121 | Failure / 300 | 143 / 129 |
| 8/4 | Success / 84 | Failure / 300 | 77 / 77 |
| 9/1 | Failure / 300 | Success / 133 | 171 / 120 |
| 9/2 | Success / 148 | Failure / 300 | 153 / 178 |
| 9/3 | Success / 142 | Failure / 300 | 142 / 142 |

Indices are zero-based; policy steps exclude settling. The subset changes prior
execution/reset histories. Recording during the full run preserves the case set
and scheduler rules, although recording overhead can affect Worker assignment.
Equal final outcomes or step counts alone do not imply equal trajectories.

### What the recorded inference steps show

The full-workload traces contain two observable patterns:

- **Five pairs begin with identical images.** Small observation differences arise
  during simulation before generated actions diverge. The first visual difference
  affects one pixel in each case. These cases are 0/2, 0/4, 5/3, 8/4 and 9/1.
- **Two pairs begin with different scene/input conditions.** Cases 9/2 and 9/3
  have different images from the first inference step. Their actual Kit Worker
  assignment and environment reuse history differ from serial Native execution.

Across all 911 aligned steps, **173 have identical saved processor inputs and
identical inputs after BF16 conversion; all 173 generate identical raw model
actions**. No compared step with an identical raw camera image has a different
saved model image. The earlier seven-case subset separately contains 440
identical-input steps, also with identical generated actions. No observed instance
establishes that inference randomness produces different actions for equal inputs.

Kit converts actions to float32 after generation. Wherever raw generated actions
match, converting Native's executed action to float32 exactly matches Kit's action.
These observations do not establish precision loss as a cause of different model
outputs. They also do not exclude an indirect effect through subsequent simulation,
changed observations and later actions; that effect has not been isolated here.

Each episode is new to the memoryless model. This does not imply that all simulator
state is statistically independent: reset can change scene properties not restored
by set_init_state. The experiments have not established a causal relationship or
quantified statistical correlation between reset-history differences and behavioral
divergence. Neither a universal reset explanation nor absence of reset influence
is demonstrated. The two patterns above describe observed divergence, not its
fully isolated cause.

### Additional Goal states 5–9

| Measure | Kit | Native |
| --- | ---: | ---: |
| Successes | 36/50 (72%) | 35/50 (70%) |
| Policy steps | 8,187 | 8,309 |
| Formal evaluation time | 24.38 min | 27.90 min |
| Evaluation throughput | 5.596 steps/s | 4.963 steps/s |

Both arms use the same new identities and fixed settings. Pairwise counts are
32 both-success, 11 both-failure, **four Kit-only and three Native-only**. The seven
outcome differences are Kit-only 0/6, 4/5, 9/8, 9/9 and Native-only 0/7, 3/8, 9/7.

This new block is more balanced than the states0–4 split of six Kit-only versus
one Native-only success. Within the observed cases, the strength of the directional
difference does not persist across the two blocks. This does **not** establish
absence of bias, equivalence, or random variation as the cause; the original
states0–4 differences remain repeatable.

Taking one result per unique identity across the two blocks gives Kit 74/100 and
Native 68/100, with ten Kit-only and four Native-only successes. States5–9 began
in fresh task environments without replaying states0–4. This is a descriptive
union of separately initialized blocks, not one continuous states0–9 evaluation;
repeats and recorded copies are not added again to its denominator.

## 5. Conclusions and limits

The updated Kit completes the agreed four-suite workload using separate Policy
and standard simulator environments. The earlier pronounced Native-favoring Long
deficit is no longer observed. Long and Goal full-workload repetitions are stable
within each arm; the new Goal state block shows a more balanced directional split.

The recorded Goal cases show that some comparisons start from different scene
conditions, while others develop small observation differences during simulation,
followed by different actions and trajectories. No equal-input/different-action
instance attributable to inference randomness has been observed.

These conclusions apply only to the examined cases and conditions. The evaluation
covers a limited number of initial states, repeats reuse the same identities,
runtime versions and action precision differ, and timing comparisons have the
boundaries described above. The experiments do not establish statistical equivalence,
a general directional preference, universal model determinism, or a unique cause
for the remaining discrepancies. Data collection for the described Goal experiments
is complete; further causal attribution may remain inconclusive. Overall, the
updated Kit reruns improve consistency, while the performance conclusions remain
aligned with the previous experiments.

### Known implementation differences identified during review

Kit Worker assignment can give the same task/state identity a different reset
history. LIBERO samples fixture placement from the global NumPy RNG at reset;
`set_init_state` does not restore those model properties. This accounts for the
initial-scene differences in Goal 9/2 and 9/3. The recorded action replay also
shows that float32 versus upstream float64 execution changes trajectories.
Native's robosuite 1.4.1 additionally includes a Panda `link7` collision geom and
resets the gripper's previous action, unlike Kit's 1.4.0 runtime. These known
differences are separate follow-up work; their contributions to every final
success/failure disagreement have not been fully isolated.

## 6. Corrected Native entry-point validation

Completed November 14 at 15:57 +08:00. The corrected `run_native.py` calls
`eval_libero.__wrapped__(cfg)` and runs the pinned official OpenVLA evaluator
`c8f03f48af692657d3060c19588038c7220e9af9`. This validation covers all four suites,
ten tasks and initial-state indices 0–4 per suite: **200 episodes**. Model and
environment seeds are 7, settling is ten steps, and the suite horizons remain
unchanged. Native runs serially on the RTX 4090 with its existing robosuite 1.4.1
environment, CPU TensorFlow preprocessing and local data-disk checkpoints. Videos
and additional observation/action traces are disabled.

The run used the review-fix working-tree snapshot based on `ca37420` on
`fix/openvla-review-followup`; it was not yet a committed revision when launched.
The uploaded runner and comparison script match commit `14d2f501370be9b8a702936236a40e33516e6cdd`
byte for byte. This identifies the tested source without relabeling the
pre-commit launch as a checkout of that later commit:

| File | SHA-256 |
| --- | --- |
| `run_native.py` | `304c3eab0c125579b15a13a461c911f9a7c2e6feec06cd595041e6182a150ce6` |
| `compare_native.py` | `644363cdddad2569a965a7118d1d218f8722be9947da95c1f2b766ee63e682a6` |

The Native process uses its existing LIBERO source/configuration paths. An isolated
offline model-code cache resolves the official loader's default reference to
`47a0ec7fc4ec123775a391911046cf33cf9ed83f`, matching the frozen model code used by Kit.
Unlike the preceding single-case smoke check, this run uses the full entry point
without task-subset modifications.

| Suite | Native successes | Native policy steps | Runner elapsed (min) | Kit-only / Native-only successes |
| --- | ---: | ---: | ---: | ---: |
| Spatial | 43/50 | 6,181 | 22.11 | 1 / 4 |
| Object | 42/50 | 8,416 | 28.02 | 1 / 0 |
| Goal | 33/50 | 8,516 | 28.31 | 6 / 1 |
| Long | 28/50 | 19,595 | 63.25 | 6 / 4 |
| Total | **146/200** | **42,708** | **141.69** | **14 / 9** |

Runner elapsed time includes model loading and setup; total queue wall time is
143.06 minutes, including process startup and comparison work. These timings do
not replace the evaluation-only times in the main performance table.

`compare_native.py` completed for all four suites against the existing fixed
Kit 2/2 configuration-campaign results (`da280f0`). Every expected task/state pair
is present exactly once, and no caught evaluator execution errors were found.
**All 200 Native episodes reproduce the previous Native reference's success/failure
outcome and executed policy-step count exactly.** The Kit/Native outcome split
therefore also remains unchanged: 14 Kit-only and nine Native-only successes.

This validates the corrected Native entry point through the complete requested
workload and supports retaining the existing Native reference. It is an additional
Native validation, not a new same-final-head Kit/Native performance campaign;
Kit was not rerun for this follow-up.
