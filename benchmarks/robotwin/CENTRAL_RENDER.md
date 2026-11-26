# RoboTwin central rendering

`render_mode: central` moves RoboTwin's ray tracing out of the evaluator processes. The evaluators simulate
on the CPU, and two render services per GPU render the observations of all evaluators on that GPU. It is
off by default (`render_mode: native`).

## Why

In native mode every evaluator holds its own Vulkan context, ray-tracing pipeline and OIDN denoiser on the
GPU, and builds a new ray-traced scene for every episode. On a 49 GB RTX 4090 that caps a single GPU at about
eight evaluators; at that point the GPU still has idle time, but there is no memory left for more evaluators.
Central rendering keeps one renderer per service, so the same GPU memory holds more evaluators and Policy
replicas, and the GPU stays busy.

It does not make one image faster, and it does not let rendering and Policy inference run at the same time:
measured on one RTX 4090, ray tracing and CUDA inference running together reach 0.95–1.12 of one GPU's
standalone throughput. Throughput per GPU is bounded by the render time plus the inference time of each
step, so the gain is limited to the GPU time the native mode leaves idle.

## How it works

- **Evaluators.** The launcher replaces the `sapien.render` classes that describe visuals (materials,
  textures, render shapes, render bodies) with recorders. Task code, actor builders and the URDF loader run
  unchanged, physics is created as usual, and lights and cameras are recorded with the same random draws.
  Each episode's scene setup becomes an ordered event log that is sent to the service once; every
  observation then sends only the poses of the recorded entities and the cameras. The evaluator runs with
  `CUDA_VISIBLE_DEVICES` empty.
- **Render services** (`roboscope_robotwin_render.py`). Each episode's events are replayed with the real
  classes into a new render-only scene on the service's one renderer, exactly as a native evaluator builds a
  new scene per episode. The service renders the head and wrist cameras (the cameras the Policy reads; the
  launcher's lean-camera and skip-unread patches select the same ones natively), returns RGB through shared
  memory, and clears SAPIEN's resource cache every five episodes per connected evaluator, as the native
  evaluator does per process.
- **Kit.** The RoboTwin adapter starts a GPU's services when its first Worker starts and spreads that GPU's
  Workers over them. An evaluator that loses its service (the service failed, was killed, or closed the
  connection) reports it, and its Worker exits instead of recording the remaining episodes as failures. The
  runtime requeues the Worker's task and restarts the Worker, which starts a new service because the old one
  has exited. Services stop when the pass ends, and exit on their own if the Kit process exits.

Images match native rendering only with a new scene per episode. Re-posing the entities of a reused
ray-traced scene, even one built from an identical event log, changed about 7% of the first frame's pixels
by one intensity level and changed the actions of about 3.5% of episodes, so the service never reuses one.

## Enabling it

In the installed `<install-dir>/benchmark-manifest.json`, set:

```json
"defaults": {
  "render_mode": "central",
  "render_services_per_gpu": 2
}
```

Central rendering supports `demo_clean` with qpos actions, a seed cache (formal runs always have one) and
videos off. `demo_randomized` changes light colors during episodes, which the services do not replay; EE
actions and runs without a seed cache need cuRobo on the GPU in the evaluator. Other combinations are
rejected before the run starts. Service logs are written to `<output-dir>/render-services/`.

Autotune selects `K`/`g` for the configured mode. Use `--cuda-mps` with it: with only the services' renderers
on the GPU, MPS lets Policy replicas overlap (+4–7%), while in native mode it added 0.9%.

## Measurements

### 500 episodes (50 tasks × 10)

One idle RTX 4090 (49,140 MiB, driver 595.71.05), Xeon Platinum 8470Q with a 20-core CPU quota, `build-runtime.sh`
RoboTwin runtime (CPython 3.8.20, SAPIEN 3.0.0b1, cuRobo, full pinned assets), StarVLA `Qwen3-VL-OFT-Robotwin2`
(revision `e9cdfc38`), RoboScope Kit `0.8.0b72` (`96957cf`), `--cuda-mps` and `--run-cache off` (default task
order) on every arm. Workload: 50 tasks × 10 episodes (500 episodes) of `demo_clean` with
`instruction_seed_limit: 10`. Two render services per GPU for all central runs.

Arms ran back to back in execution order. `Eval. throughput` divides `total_steps` by `phase_timings.evaluation_s`
(excluding model loading and teardown), `Steady` is Kit's `result.steady_steps_per_s` (while all `K` Workers are
busy), and `Wall time` is `phase_timings.policy_load_s + result.elapsed_s` (including loading, evaluation, and
teardown):

| Run | Successes / steps | Eval. time (s) | Eval. throughput (steps/s) | Steady (steps/s) | Wall time (s) | Peak GPU mem (MiB) | GPU util | OIDN / OOM errors | Match vs. run 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. native K9/G1 | 232 / 211,058 | 1,702.3 | 124.0 | 127.3 | 1,721.3 | 48,167 | 60.2% | 0 / 0 | baseline |
| 2. central K20/G2 | 232 / 211,058 | 1,188.7 | 177.6 | 184.9 | 1,395.1 | 38,479 | 82.9% | 0 / 0 | 500 / 500 |
| 3. central K20/G2 | 232 / 211,058 | 1,179.9 | 178.9 | 185.8 | 1,280.5 | 38,483 | 83.5% | 0 / 0 | 500 / 500 |
| 4. native K9/G1 | 233 / 210,767 | 1,682.4 | 125.3 | 128.4 | 1,783.4 | 48,507 | 60.7% | 17 OIDN\* / 0 | 499 / 500\* |
| 5. central K16/G3 | 232 / 211,058 | **1,165.2** | **181.1** | **189.1** | **1,249.4** | 46,725 | 83.0% | 0 / 0 | 500 / 500 |

- **Speedup (evaluation time):** against the clean `native K9/G1` run (1,702.3 s, `n=1`), `central K20/G2`
  averages 1,184.3 s (19.7 min, 178.2 steps/s, **1.44×** faster) while using 9.5 GiB less GPU memory
  (37.6 vs. 47.0 GiB), and `central K16/G3` finishes in 1,165.2 s (19.4 min, 181.1 steps/s, **1.46×** faster).
  Wall time on run 2 includes 180.0 s of cold checkpoint loading from network storage (74.1 s on run 3, 77.1 s on
  run 5), and runs 2 and 3 include ~26 s of post-evaluation teardown because `gpu0-service-0` hit the 20 s
  SIGTERM-to-SIGKILL shutdown timeout in `Scene.__del__` (vs. 7.2 s teardown on run 5 and 2.9 s on native).
- **\*Silent OIDN OOM at native K9/G1:** on run 4 (48,507 MiB / 47.4 GiB peak), `worker-007` logged 17 OIDN error
  lines (3 `OIDN Error: out of memory` and 14 `changes to the filter are not committed`) during
  `place_object_stand` episode 7 (`seed 1034009`). The worker did not crash, so Kit recorded `oom_events: 0`, but
  undenoised observations flipped that episode from failure at 400 steps to success at 109 steps (291 fewer steps
  total, so run 4 is excluded from the clean baseline ratio). On a 48 GiB RTX 4090, `native K8/G1` (40.9 GiB in
  the sweep below; not run at 500 episodes) is the largest native configuration with safe memory headroom.
- **Agreement:** all three central runs matched run 1 (`native K9/G1`) on 500/500 success outcomes and 500/500
  terminal steps, with zero OOM or OIDN errors and no render-service restarts.

### Concurrency sweep (50 tasks × 2)

Same host, runtime, Policy, and flags on 50 tasks × 2 episodes (100 episodes) of `demo_clean`
(`instruction_seed_limit: 100`), used to pick the configurations above. Every arm matched `native K8/G1` on
100/100 outcomes and terminal steps (47/100 successes, 42,330 total steps). `native K9/G1`, `central K16/G2`, and
`central K20/G2` loaded the checkpoint from network storage (affecting `Wall time` only, not `Eval. time`), while
the other six arms loaded from `/dev/shm`. On `central K16/G2`, `gpu0-service-0` also hit the 20 s shutdown
timeout (25.6 s teardown):

| Configuration | Eval. time (s) | Eval. throughput (steps/s) | Wall time (s) | Peak GPU mem (MiB) | GPU util | OOM / worker failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native K6/G2 | 438.2 | 96.6 | 488.9 | 45,063 | 43.0% | 0 / 0 |
| native K8/G1 | 386.0 | 109.7 | 405.5 | 41,861 | 52.3% | 0 / 0 |
| native K9/G1 | 379.3 | 111.6 | 481.4 | 45,156 | 52.7% | 0 / 0 |
| native K8/G2 | 381.8 | 110.9 | 417.3 | 48,491 | 49.0% | 1 / 1 (scaled to K7; 66 OIDN errors) |
| central K12/G2 (2 svc) | 316.1 | 133.9 | 359.3 | 34,192 | 59.4% | 0 / 0 |
| central K12/G3 (2 svc) | 314.4 | 134.6 | 382.3 | 44,335 | 59.4% | 0 / 0 |
| central K16/G2 (2 svc) | 299.2 | 141.5 | 368.2 | 36,333 | 64.5% | 0 / 0 |
| central K20/G2 (2 svc) | 292.4 | 144.8 | 337.5 | 38,287 | 66.8% | 0 / 0 |
| central K16/G3 (2 svc) | 289.2 | 146.4 | 369.8 | 46,471 | 64.9% | 0 / 0 |

### Image/action hashing and service recovery

Per-episode hashing of every Policy input image and executed action matched `native` on 100/100 episodes
(`50 × 2`) and 50/50 episodes (`50 × 1`, `build-runtime.sh`, `central K12/G3` vs. `native K6/G2`). In a
fault-injection repeat of that 50-episode `build-runtime.sh` run, killing one of the two render services with
`SIGKILL` 150 s into the run caused its six connected Workers to exit, requeue their tasks, start a replacement
service on restart, and complete matching native on all 50/50 episodes, including both interrupted episodes rerun
from the start.

## Limitations

- Multi-GPU runs and full 5,000-episode runs have not been measured.
- A service failure interrupts every Worker connected to it; their tasks are requeued and rerun. Like any
  other Worker failure, a Worker restarts once; a second failure of the same Worker stops the run.
- Results match native rendering on the same machine; RoboTwin rollouts on different machines are not
  bit-identical in either mode.
