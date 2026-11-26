# robomimic throughput

## Hardware and workload

Measured on 2026-11-09 in a Linux container with a 12-core CPU quota, 96 GiB RAM and one RTX 4090
(49,140 MiB, driver 580.105.08). Each run evaluates the five `ph_low_dim` BC-RNN tasks (Lift, Can, Square,
Transport, Tool Hang) on seeds 0-49: 250 episodes and 81,289 steps. Runs use FP32, no cameras or video, MPS
off, fixed K/G, one CPU thread per process, and fresh Worker and Policy processes.

## Matched comparison

Three interleaved runs per row. Formal time is `result.json` `phase_timings.evaluation_s`.

| Code, K/G | Formal runs (s) | Formal mean ± SD (s) | End-to-end mean (s) | Steps/s |
| --- | --- | ---: | ---: | ---: |
| Original, 16/1 | 354.18, 354.45, 386.21 | 364.95 ± 18.42 | 371.42 | 222.74 |
| Original, 20/8 | 225.56, 220.32, 221.14 | 222.34 ± 2.82 | 243.07 | 365.61 |
| This change, 20/8 | 209.19, 210.33, 208.04 | 209.19 ± 1.14 | 230.40 | 388.59 |

The combined gain is **1.745x** in formal time and 1.612x end to end. Configuration alone gives 1.641x, and
the code gives 1.063x at the same K20/G8. All nine runs complete the same 250 episodes with 176 successes
and match the original per-episode records.

## Compiled-model cache

The pinned builder patch keeps two compiled MuJoCo models per process and gives each simulation its own copy.
`ROBOSCOPE_ROBOMIMIC_MODEL_CACHE=0` disables it. The original binding, the patched binding with the cache off
and with it on each ran 260 episodes, covering all 250 task/seed pairs plus repeated resets and task
switches. All 18,356 retained observation, action and simulator-state arrays are byte-identical in every pair.

## Native comparison

The pinned upstream `run_trained_agent.rollout`, run by 20 persistent processes under the same resource
limits, takes 239.28 ± 1.03 s formal and 247.66 s end to end over three runs. Kit at K20/G8 is 1.144x faster
formal and 1.075x end to end. Native holds 20 model replicas and Kit holds eight.

## Limitations

- The combined gain is below 2x, and most of it comes from the K/G configuration.
- The measurements were taken on Kit `0.8.0b43`; the robomimic code is unchanged since, but the shared runtime
  has moved on.
- The cache check covers this workload and the pinned simulator, not arbitrary tasks, assets or versions.
- The cache assumes all compilation goes through `MjSim.from_xml_string`; `MjModel.from_xml_path` does not
  update its last-compiled XML record.

## Raw evidence

The [evidence release](https://github.com/OopsJoey/RoboScope-Kit/releases/tag/issue32-review-evidence-2026-11-09)
holds the per-run records, per-step traces, scripts and checksums.
