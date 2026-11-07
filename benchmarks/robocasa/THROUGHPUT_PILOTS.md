# RoboCasa GR1 Tabletop throughput study — 2026-11-07

The real StarVLA Quick Test completed one 720-step episode successfully. At
fixed K=1/G=1, the site Jacobian candidate improved formal throughput by
**1.019×**, below the 5% screening threshold. Using the existing scheduler to
increase K from 1 to 4 at G=1 delivered **3.323×** throughput over four episodes
and **3.190×** over eight episodes. Seeds, outcomes and steps matched within
each pair. The full-suite 2× acceptance target remains open.

[Issue #16](https://github.com/MINT-SJTU/RoboScope-Kit/issues/16) covers
`robocasa` / `tabletop`, the legacy GR1 suite. The acceptance workload is 24
tasks × 50 episodes, or 1,200 episodes. The target is at least 2× completed
episodes per formal evaluation second, with complete records and no systematic
score regression. RoboCasa365 measurements belong to issue #97.

## Source, environment and evaluation contract

| Component | Actual pilot configuration |
| --- | --- |
| Upstream Kit main used for the rebase | `03371ed7b8aa5b4f117eb742d591a3204a7772a6` |
| Baseline and executed Kit SDK | `0.8.0b34`, `bcc60844258e29a388c88dd252387a96689884f5`, tree `d4b3e01351abd5c5165f0f71f962927366ae6bc0` |
| Baseline content | [PR #39](https://github.com/MINT-SJTU/RoboScope-Kit/pull/39) rebased onto main. Main advanced by a documentation commit relative to PR #39's base. Both pilot arms enable its existing guarded observation deferral. |
| Jacobian candidate | Separate runtime built through the candidate builder from the b40 draft source archive, SHA-256 `3c3770ab99c4a2c9e01c63f5fd90b16da2283f298e2c228a558a263d55f3aae4`. The actual evaluation and both replay interpreters imported the same b34 SDK. |
| Simulator sources | StarVLA environment `2e5f239bc0b1661d7d556bdba5071f3041544cc6`, RoboCasa 0.2.0 `4840e671596f93ca03651524b9f72ffb1aadfeff`, RoboSuite 1.5.1 `a071383d53568ab798eb315c0e95357911be922d` |
| Simulator packages | CPython 3.11.14, MuJoCo 3.2.6, Gymnasium 1.0.0, EGL |
| Assets | Common PR #39 checksummed asset lock. The four replacement kitchen archives are pinned at `Livfour/robocasa-assets@69f7446f5f0f763971cabeb8b77026fcecced17c`. They differ from the old Box archives. |
| Policy | `StarVLA/Qwen3-VL-OFT-Robocasa@30ec4b5c5c8ffbb3ad651b2ab53b30714c8c2fd4`, step 90,000. Verified 9,785,286,394-byte weight SHA-256 `66a114db658b4b2dda8fd5530bfe3c7cd44b3c37e49c6fb68be22bab5e8f4bf5`. |
| Model code and runtime | StarVLA `985c2238627481f25476fbee478083812dd82a21`, CPython 3.11.15, PyTorch 2.6.0+cu124, Transformers 4.57.0, BF16, DDIM 10 steps, batch one |
| Hardware | One RTX 4090 reported as 49,140 MiB, NVIDIA driver 580.105.08, Xeon Gold 6459C, 12-CPU quota and 96 GiB memory limit |
| Rules held fixed | Manifest and Python hash seed 42, task-derived constructor and episode seeds, native cameras and observations, 29-dimensional actions, first 12 of 16 predicted actions executed per chunk, 720 control steps per episode, 25 physics substeps per control step, videos and MPS off, JSON transport and one CPU math thread per worker |

The common runtime includes PR #39's Python 3.11 repair, seeded construction
and resets, actual step accounting and the distractor configuration copy.
The baseline and candidate simulator builds passed their self-tests. The
candidate used the same verified extracted assets in a separate runtime
prefix. The b34 wheel build and installation passed, as did 187 existing
pytest cases, Ruff and Bash syntax. The Policy preflight verified weights and
metadata and passed CPU imports and normalization. Its package check returned
status 1 on a Decord wheel platform diagnostic. The native Decord library
import and the real CUDA model calls passed.

The full suite consists of these task groups. The suite definition supplies
the exact native environment IDs.

| Group | Tasks |
| --- | --- |
| Cabinet, drawer and microwave | `PnPBottleToCabinetClose`, `PnPCanToDrawerClose`, `PnPCupToDrawerClose`, `PnPMilkToMicrowaveClose`, `PnPPotatoToMicrowaveClose`, `PnPWineToCabinetClose` |
| Cutting board | `PosttrainCuttingboardToBasket`, `PosttrainCuttingboardToCardboardbox`, `PosttrainCuttingboardToPan`, `PosttrainCuttingboardToPot`, `PosttrainCuttingboardToTieredbasket` |
| Placemat | `PosttrainPlacematToBasket`, `PosttrainPlacematToBowl`, `PosttrainPlacematToPlate`, `PosttrainPlacematToTieredshelf` |
| Plate | `PosttrainPlateToBowl`, `PosttrainPlateToCardboardbox`, `PosttrainPlateToPan`, `PosttrainPlateToPlate` |
| Tray | `PosttrainTrayToCardboardbox`, `PosttrainTrayToPlate`, `PosttrainTrayToPot`, `PosttrainTrayToTieredbasket`, `PosttrainTrayToTieredshelf` |

## Real-model runs at K=1/G=1

Quick Test used `PnPBottleToCabinetClose`. The paired pilot used
`PosttrainCuttingboardToBasket`, a novel-object task that scored **27/50 (54%)**
in the [historical GR1 study](https://github.com/suica/evostudio-evaluation-kit/blob/codex/robocasa-throughput-issue-3/docs/experiments/robocasa-throughput/README.md).
In the new pair, seed `3178861460` succeeded and seed `3178861461` failed in
both arms. Each episode ran all 720 steps and deferred 660 intermediate
observations in both arms.

| Metric | Quick Test | Basket A, baseline | Basket B, Jacobian |
| --- | ---: | ---: | ---: |
| End-to-end command wall, s | 52.546 | 71.378 | 67.060 |
| Local checkpoint preparation within outer residual, s | 0.009 | 0.031 | 0.008 |
| Policy startup and load, s | 25.425 | 20.765 | 17.181 |
| Runtime preparation, s | 0.237 | 0.254 | 0.262 |
| Formal evaluation wall, s | 25.125 | 48.4398 | 47.5368 |
| End-to-end time outside those phases, s | 1.759 | 1.919 | 2.081 |
| Environment construction, s | 2.858 | 3.800 | 4.376 |
| Episode resets, s | 2.322 | 6.963 | 6.261 |
| Simulator steps with controllers, observations and rendering, s | 14.940 | 29.4664 | 28.9236 |
| Policy round trips, s | 3.777 | 6.833 | 6.603 |
| Formal time outside the worker timers, s | 1.228 | 1.377 | 1.373 |
| First episode wall after reset, s | 18.719 | 19.000 | 18.482 |
| Second episode wall after reset, s | — | 17.305 | 17.054 |
| Completed episodes / expected | 1/1 | 2/2 | 2/2 |
| Task successes | 1/1 (100%) | 1/2 (50%) | 1/2 (50%) |
| Actual control steps | 720 | 1,440 | 1,440 |
| Formal episodes/h | 143.29 | 148.64 | 151.46 |
| Formal control steps/s | 28.66 | 29.73 | 30.29 |
| SDK worker-window mean GPU busy, % | 10.78 | 7.07 | 7.13 |
| SDK worker-window mean active CPU cores | 1.184 | 1.181 | 1.181 |
| Whole-command mean GPU busy, % | 6.23 | 5.81 | 5.85 |
| Whole-command mean / sampled peak VRAM, MiB | 5,484 / 10,696 | 7,732 / 11,120 | 8,067 / 11,120 |
| Missing / duplicate records | 0 / 0 | 0 / 0 | 0 / 0 |
| Worker failures / retries / timeouts / OOMs | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

Throughput uses `phase_timings.evaluation_s`. This timer starts at queue
release and ends when the monitor reads the drained results. It includes
in-task environment close and precedes Worker and Policy teardown. The SDK
resource monitor covers Worker preparation through exit, with CPU usage from
the container cgroup. Whole-command GPU statistics are device-wide samples
at 250 ms intervals. GPU busy combines rendering and model work. Residuals
subtract the recorded phase timers from their enclosing wall intervals and
include checkpoint preparation in the outer residual, and scheduling and
bookkeeping in the formal residual. Policy round trips include preprocessing,
inference, encoding, transport and waiting. Simulator steps include native
action conversion, controllers, observations and rendering. The live runs
do not isolate rendering, model GPU kernels or Policy queue wait. At K>1,
summed Worker durations overlap and can exceed formal wall time.

The simulator saved **0.543 s** and formal evaluation saved **0.903 s** in one
pair. The larger end-to-end difference includes **3.584 s** of Policy startup
variation even though the Policy was identical. The two episodes per arm
show matching outcomes for the measured seeds. A repeated timing distribution
and a whole-suite score comparison remain outstanding.

## Fixed-action diagnosis and attempted changes

### Simulator and controller

The fixed-action diagnostic uses one fresh Basket environment at seed
`3178861460`. It holds the reset arm and waist positions with neutral hand
commands. Eight chunks execute 96 control steps and 2,400 physics substeps
with observation deferral active. cProfile covers only the `env.step` calls. The
profiled step interval was 2.344 s. An unprofiled replay spent 1.885 s stepping,
4.065 s constructing, 2.882 s resetting and 0.133 s closing. The fresh A/A
comparison, one profiled run and one unprofiled replay, matched the initial
state and all eight recorded chunk boundaries. This probe uses fixed actions
without Policy inference.

| Profile scope | Calls | Time, s | Share of profiled step wall |
| --- | ---: | ---: | ---: |
| MuJoCo `mj_step1` + `mj_step2`, self | 2,400 each | 1.0273 | 43.8% |
| `action_limits`, inclusive | 7,296 | 0.1407 | 6.0% |
| `_update_single_reference`, inclusive | 4,800 | 0.2282 | 9.7% |
| `mj_jacSite`, self, nested in reference work | 19,200 | 0.0169 | 0.72% |

The candidate computes the translational and rotational Jacobians in one
native call and derives site velocities from those same Jacobians. It applies
to the two absolute-position GR1 arm controllers with fixed impedance and
one reference site. The candidate A/B replay matched hashes for consumed RGB,
proprioception, instruction and MuJoCo integration state at reset and every
recorded chunk boundary. The eight fixed action chunks, cumulative success
flags and termination flags also matched. Unprofiled step wall fell from
1.885 s to 1.783 s (**1.057×**) in this one replay pair. The live comparison
covers full-horizon outcomes and step counts. Hash checks cover the replay's
recorded boundaries.

The native Jacobian calculation itself consumed only 0.72% of profiled step
time. Even eliminating the complete reference update would bound this
diagnostic's step-only improvement at **1.108×**, assuming its other costs
stay fixed. If that profile share applied throughout the serial live pilot,
where stepping used 60.8% of formal time, eliminating that reference work in
the step phase alone would yield at most **1.063×** formal throughput. The
candidate also runs during reset settling, which this bound holds fixed. The
measured **1.019×** live formal ratio falls below the 5% retention threshold
used for this screening pass. The baseline controller remains the reference
for scheduling. Reduced simulation substeps and soft reset would
change the protocol, including the scene construction and sampling performed
by the native hard reset.

The profile already ran with guarded observation deferral active. The remaining
`action_limits` work accounted for 6.0% of step time. Eliminating all of it would
save about **3.65% of serial formal wall** if its measured share carried over
to the two-episode pilot. The baseline task constructed its environment once
in 3.800 s. With the same construction cost, a 50-episode work item would
amortize it to **0.076 s per episode**. Episode resets still run separately.

### Wire transport and model service

A Unix socketpair carried the actual GR1 payload shapes, a 256×256×3 uint8
image and a 16×29 float32 action chunk, through the SDK codecs. Three
300-request runs per codec gave median run means of **3.613 ms for JSON**
and **0.341 ms for MessagePack**. Image, state and action round-trip values
were checked. The difference projects to **0.196 s per episode** at 60
requests if the microbenchmark cost carries over, about 0.8% of the serial
Basket formal wall per episode. This diagnostic excludes the real Policy
Server and simulator. The timed evaluation arms retain JSON.

An independent direct real-checkpoint smoke on synthetic input measured a
**37.686 ms** warm median wall interval, with matching action hashes across
the repeated calls. The first call took 1.508 s. Its CUDA-event interval can
include CPU gaps, so it cannot isolate kernel execution or estimate
multi-worker queue latency.

## Scheduling pilots and full-suite acceptance

Both pairs used the baseline controller, deferral on, JSON, the same model,
seed policy, one CPU math thread per Worker and the same output settings.
Only K changed within each pair. The first pair ran one full-horizon episode
for each of four tasks from shard 0 of 6, with K=1 first. The expanded pair
ran two episodes per task, with K=4 first. G=1 throughout.

| Metric | 4 episodes, K=1 | 4 episodes, K=4 | 8 episodes, K=1 | 8 episodes, K=4 |
| --- | ---: | ---: | ---: | ---: |
| End-to-end command wall, s | 175.035 | 52.699 | 204.400 | 79.887 |
| Local checkpoint preparation within outer residual, s | 0.006 | 0.031 | 0.017 | 0.052 |
| Policy startup and load, s | 72.924 | 20.525 | 16.294 | 18.913 |
| Runtime preparation, s | 0.219 | 0.259 | 0.216 | 0.281 |
| Formal evaluation wall, s | 100.086 | 30.121 | 185.969 | 58.296 |
| End-to-end time outside those phases, s | 1.807 | 1.795 | 1.921 | 2.397 |
| Summed environment construction, s | 15.794 | 18.996 | 14.503 | 17.115 |
| Summed episode resets, s | 11.219 | 12.888 | 25.728 | 26.040 |
| Summed simulator steps, s | 58.053 | 57.517 | 119.754 | 120.390 |
| Summed Policy round trips, s | 12.521 | 19.711 | 23.843 | 37.600 |
| Formal time outside Worker timers, s | 2.499 | Overlap | 2.141 | Overlap |
| Episode wall after reset, min–max s | 16.317–18.563 | 19.022–19.567 | 15.661–21.699 | 18.323–23.433 |
| Completed episodes / expected | 4/4 | 4/4 | 8/8 | 8/8 |
| Task successes / failures | 3 / 1 (75.00%) | 3 / 1 (75.00%) | 5 / 3 (62.50%) | 5 / 3 (62.50%) |
| Actual control steps | 2,880 | 2,880 | 5,760 | 5,760 |
| Deferred native observations | 2,640 | 2,640 | 5,280 | 5,280 |
| Formal episodes/h | 143.88 | 478.07 | 154.86 | 494.03 |
| Formal control steps/s | 28.78 | 95.61 | 30.97 | 98.81 |
| SDK worker-window mean GPU busy, % | 7.64 | 23.54 | 9.19 | 27.07 |
| SDK worker-window mean active CPU cores | 1.195 | 4.030 | 1.195 | 3.776 |
| Whole-command mean GPU busy, % | 4.94 | 15.21 | 8.57 | 20.03 |
| Whole-command mean / sampled peak VRAM, MiB | 6,320 / 11,105 | 7,476 / 13,228 | 9,953 / 11,126 | 9,403 / 13,411 |
| Missing / duplicate records | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Worker failures / retries / timeouts / OOMs | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

The first pilot used episode index 0. The expanded pilot used indices 0 and
1. Both arms returned the same outcome for each task and seed at 720 steps.

| Task | Index 0 seed and outcome | Index 1 seed and outcome | Expanded successes per arm |
| --- | --- | --- | ---: |
| `PnPBottleToCabinetClose` | `2118723578`, success | `2118723579`, success | 2/2 |
| `PosttrainCuttingboardToBasket` | `3178861460`, success | `3178861461`, failure | 1/2 |
| `PosttrainPlacematToBowl` | `146481877`, failure | `146481878`, failure | 0/2 |
| `PosttrainPlateToPlate` | `573781297`, success | `573781298`, success | 2/2 |

Formal throughput increased **3.323×** and **3.190×** in the two workloads.
Summed simulator time stayed within 1% in both pairs. Overlapping Worker
activity accounts for the wall-time gain. At K=4, summed Policy round-trip
time rose by 7.190 s and 13.757 s. These timings combine service, contention
and queueing. The first pair's end-to-end wall also reflects a 52.398 s
startup difference. In the expanded pair, K=4 took 2.619 s longer to start
the same Policy. A line-ending error affected only the expanded wrapper's
post-run comparison. The comparison was regenerated from the completed runs.

K=4/G=1 is an existing setting with measured gains on both small workloads.
Each has one task per Worker at K=4, so neither exercises a sustained task
queue. The second workload changes both order and episode count relative to
the first. A repeated same-workload distribution and a matched 1,200-episode
baseline and optimized evaluation remain outstanding.

## Historical measurements

[PR #39](https://github.com/MINT-SJTU/RoboScope-Kit/pull/39) compared deferral
off and on for 24×2 real episodes at K=8/G=1. Formal wall went from 254.2 s to
236.8 s (**1.074×**) and summed simulator time went from 1,077.9 s to 890.1 s.
All 48 seeds, outcomes and step counts matched, with 26 successes in each
arm. The current experiments already use that deferral in both arms.

PR #39 also reports a historical same-host RTX 4090 24×50 comparison using
the same runtime and checkpoint. Its K=8/G=1 autotuned run took **5,457.1 s**.
The best fixed configuration in its sweep, K=12/G=2, took **3,465.7 s**,
a **1.575×** formal throughput ratio. Both completed 1,200 episodes with
681 successes. These were single runs at different K and G. PR #39 labels
them as measurements from before its last rebase.

The [older Kit study](https://github.com/suica/evostudio-evaluation-kit/pull/6)
reported 3.926× across 1,200 episodes when it changed both deferral and K=1
to K=4. At equal K=4/G=1 with deferral active, the [independent StarVLA control](https://github.com/suica/evostudio-evaluation-kit/blob/codex/robocasa-throughput-issue-3/docs/experiments/robocasa-throughput/STANDALONE.md)
matched Kit's results and had approximately equal throughput. PR #39 also
reports serial upstream comparisons that include different video and transport
settings. These studies have their own scope and do not supply a timing arm
for the new pilot.

## Reproduction and evidence

To reproduce the baseline tree, check out `03371ed7b8aa5b4f117eb742d591a3204a7772a6`,
fetch PR #39's commit `d0dee1bbdfb9c27032713218ae209b85b22d551a` from the
upstream repo, and cherry-pick it onto the checkout. Run
`benchmarks/robocasa/build-runtime.sh` from that tree with the pinned
dependencies and verified asset and source caches. The
candidate uses its separate builder with
`ROBOSCOPE_ROBOCASA_COMBINE_SITE_JACOBIANS=1`. Keep the candidate build files
and the baseline runtime separate. The current measured SDK is b34. On the
adaptation host, the paths below resolve to the original frozen source,
isolated Policy config and verified checkpoint snapshot. The pilot driver wrote
a run manifest next to each built runtime's manifest, retaining relative
runtime paths and selecting deferral on and videos off.

```bash
WORK=/root/autodl-tmp/robocasa-issue16-20261107
STORE=/autodl-fs/data/robocasa-issue16-20261107
CONTROL="$WORK/control/bin/python"
CHECKPOINT=/autodl-fs/data/evostudio/mr77-acceptance-20261013/models/robocasa/snapshot
POLICY="$WORK/policy-gr1/roboscope-policy.json"
OUT="$(mktemp -d /tmp/gr1-reproduce.XXXXXX)"
export PYTHONPATH="$WORK/kit/packages/roboscope-kit/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export ROBOSCOPE_POLICY_ENCODING=json

"$CONTROL" -m roboscope_kit.runtime.benchmark --benchmark robocasa \
  --benchmark-manifest "$WORK/runtime/benchmark-manifest.json" \
  --self-test --output-dir "$OUT/self-test"

"$CONTROL" -m roboscope_kit.runtime.benchmark --benchmark robocasa --suite tabletop \
  --benchmark-manifest "$WORK/runtime/issue16-quick-01-manifest.json" \
  --policy-config "$POLICY" --checkpoint-source "$CHECKPOINT" \
  --quick-test --output-dir "$OUT/quick"

for arm in a1 b1; do
  RUNTIME="$WORK/runtime"
  if [ "$arm" = b1 ]; then RUNTIME="$WORK/runtime-jacobian"; fi
  "$CONTROL" -m roboscope_kit.runtime.benchmark --benchmark robocasa --suite tabletop \
    --benchmark-manifest "$RUNTIME/issue16-basket-$arm-manifest.json" \
    --policy-config "$POLICY" --checkpoint-source "$CHECKPOINT" \
    --concurrency 1 --replicas 1 --episodes 2 --shard-count 24 --shard-index 6 \
    --output-dir "$OUT/basket-$arm"
done
```

For the scheduling pairs use the baseline runtime, `--shard-count 6
--shard-index 0 --replicas 1`, and select `--concurrency 1` and `--concurrency 4`
in separate fresh output directories. Use `--episodes 1` for the first pair
and `--episodes 2` in reverse order for the second. For the complete fixed-K/G
workload omit the sharding flags and use the 50-episode manifest default.

The archive contains the exact run commands, manifests, per-episode queue
records and resource samples in `pilot/`, the fixed actions and replay hashes
in `profile/`, plus the Policy preflight, build checks and reproduction
scripts. The adaptation host retains the original logs under `$STORE`.
[GR1 pilot evidence archive](https://github.com/QLL-CCDD/RoboScope-Kit/releases/download/robocasa-issue16-pilots-evidence-20261107/robocasa-issue16-pilots-evidence.tar.gz)
