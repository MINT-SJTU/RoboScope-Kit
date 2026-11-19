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

### This implementation

One RTX 4090 (49 GB, driver 595.71.05), Xeon Platinum 8470Q with a 20-core CPU quota, StarVLA
`Qwen3-VL-OFT-Robotwin2` (revision `e9cdfc38`), 50 tasks × 2 episodes of `demo_clean`, CUDA MPS on both modes.
The arms ran back to back in the order native, central, central, native. The RoboTwin side was the
prototype's environment (CPython 3.10, SAPIEN 3.0.0b1, RoboTwin at the runtime commit, no cuRobo, so both modes
loaded the planner without it), not one built by `build-runtime.sh`. Another job was installing packages on the
host's CPU during the runs, which likely explains the spread between repeats.

| | native K6/G2 | central K12/G3, 2 services per GPU |
| --- | ---: | ---: |
| Successes / total steps (each arm) | 47 / 42,347 | 47 / 42,347 |
| Evaluation throughput, steps/s (two arms) | 99.3, 104.8 | 137.6, 160.5 |
| Mean | 102.1 | 149.1 (+46%) |
| Wall time including startup, s (mean) | 447.8 | 335.3 (1.34× faster) |
| Peak GPU memory, MiB | 44,697 | 44,587 |
| Episodes with identical actions and Policy images | — | 100 / 100 in every pair |

Every image the Policy received and every executed action were hashed per episode; all four arms matched
episode by episode. No render service restarted.

On a `build-runtime.sh` runtime (CPython 3.8, cuRobo, the pinned embodiment and object archives; the 11 GB
background textures, which `demo_clean` does not use, were left out for disk space), same host and Policy,
50 tasks × 1 episode, CUDA MPS, one arm each:

| | native K6/G2 | central K12/G3 | central K12/G3, one service killed |
| --- | ---: | ---: | ---: |
| Successes / total steps | 24 / 20,768 | 24 / 20,768 | 24 / 20,768 |
| Evaluation throughput, steps/s | 87.3 | 116.4 | 139.0 |
| Episodes with identical actions and Policy images | — | 50 / 50 | 50 / 50 |

In the third arm one of the two services was killed with SIGKILL 150 s into the run. The six Workers connected
to it reported the lost service and exited, their six tasks were requeued, and the restarted Workers started
a new service; the run completed, and the two interrupted episodes, rerun from the start, matched native.

### Prototype

The measurements below come from the prototype that this mode was ported from: the same worker recorders,
replay, per-episode scenes and cache clearing, run through the Kit 0.8.0b31 SDK and a launcher of that
time, with StarVLA `Qwen3-VL-OFT-Robotwin2` (revision `e9cdfc38`), action horizon and observation stride 16,
320×240 RT at 32 spp with OIDN, and `front_camera` not rendered. One RTX 4090 (49 GB, driver 580.105.08),
12-core CPU quota; the two sides of each comparison ran back to back on the same GPU. `K` is the number of
Workers and `G` the number of Policy replicas; the native side uses the largest configuration that did not run
out of GPU memory.

50 tasks × 10 episodes of `demo_clean`, CUDA MPS on both sides:

| | native K6/G2 | central K12/G3, 2 services per GPU |
| --- | ---: | ---: |
| Successes / total steps | 233 / 208,789 | 233 / 208,789 |
| Evaluation throughput, steps/s | 123.5 | 154.5 (+25.1%) |
| Steady throughput, steps/s (until K−1 Workers remain) | 125.0 | 171.1 (+36.9%) |
| Wall time including startup, s | 1731.0 | 1407.5 (1.23× faster) |
| Peak GPU memory, MiB | 43,686 | 44,044 |
| Episodes with identical action sequences | — | 500 / 500 |

Single task (`adjust_bottle`, 96 episodes):

| Configuration | steps/s | Peak GPU memory |
| --- | ---: | ---: |
| native K8/G2 | 106.4 | 48.0 GB (at the limit) |
| native K8/G2, MPS | 107.4 | 48.0 GB |
| central K8/G2 | 128.6 (+20.9%) | 29.0 GB |
| central K12/G2 | 145.8 | 31.9 GB |
| central K12/G3, MPS | 157.7 (+46.9% vs. native with MPS) | 42.3 GB |

A reversed-order repeat of the last comparison gave 107.8 → 158.8 (+47.3%), and a second RTX 4090 host gave
103.1 → 145.4 (+41.0%); all 96 episodes had identical actions in each pair. On 50 tasks × 4 episodes, one
service per GPU gave 143.8 steps/s and two gave 158.3; a third added about 1%. The first observation of an
episode took 1.5–2.4 s on average in native mode (scene rebuild in the evaluator) and 0.3–0.9 s with a service.

Native K8/G2 does not fit on 50 tasks: OIDN runs out of memory, denoising fails and 3 of 200 episodes changed
actions, so native K6/G2 is the baseline above. Adding Workers beyond K12–16 per GPU did not help, because the
GPU is saturated.

## Limitations

- Multi-GPU runs and full 5,000-episode runs have not been measured; on four GPUs the single-GPU ratio
  suggests about 1.2–1.3×.
- A service failure interrupts every Worker connected to it; their tasks are requeued and rerun. Like any
  other Worker failure, a Worker restarts once; a second failure of the same Worker stops the run.
- Results match native rendering on the same machine; RoboTwin rollouts on different machines are not
  bit-identical in either mode.
