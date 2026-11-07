# RoboScope Kit Profiling

Updated: 2026-11-07

This document explains how to use `--profile` and optional device-wide GPU metrics to see where an
evaluation's time goes, and records what profiling has found so far. After every profiling run, add its
findings to the benchmark's own documentation and link them from
[Findings](#findings), so results accumulate instead of being measured again from scratch.

## Requirements

- **Nsight Systems CLI.** The machine that runs the evaluation needs
  [Nsight Systems](https://developer.nvidia.com/nsight-systems/get-started) with `nsys` on `PATH`. The Kit
  does not install it. Version 2026.5.1 is verified (installed under `/opt/nvidia/nsight-systems-cli/2026.5.1`
  in the AutoDL container); 2026.6.1 is verified for native RTX 4090 device counters.
  Without `nsys`, `--profile` stops before any Worker starts.
- **No extra Python package.** SDK ranges use Nsight Systems' injection library through `ctypes`.
  RoboTwin's standalone simulator bridge uses the Torch NVTX API already supplied by its pinned
  simulator environment; it does not import the Kit. No environment needs the separate `nvtx` package.
- **A fixed configuration.** `--profile` works only with `--quick-test` or a fixed
  `--concurrency K --replicas G`; autotune probes are never captured by its Worker/Policy traces.
  The separate `--profile-gpu-metrics` collector also supports autotune.
- **Disk space.** Each Worker writes one report per attempt, roughly proportional to its first episode's steps: on RoboDojo,
  about 27 MB for a 20-step episode and about 150 MB for a 105-step episode. `nsys` also writes temporary
  files under `/tmp` while capturing. Each Policy replica also writes a full-lifetime report, including
  model loading and every request. Keep profiling runs short; a full evaluation can produce large reports.

## Running

```bash
python -m roboscope_kit.runtime.benchmark --benchmark robodojo --suite arx_x5 \
  --checkpoint-source <ckpt> --benchmark-manifest <install-dir>/benchmark-manifest.json \
  --policy-config <policy.json> --output-dir <out> \
  --quick-test --profile
```

To look at concurrent Workers, replace `--quick-test` with `--concurrency K --replicas G`.

## Device-wide GPU metrics

`--profile-gpu-metrics METRIC_SET` starts one evaluation-level Nsight counter collector for the selected
GPUs, independently of `--profile`, fixed K/g, autotune or CUDA MPS. For example, add
`--profile-gpu-metrics ad10x-gfxt --profile-gpu-metrics-frequency 10000` where the installed Nsight
version supports that metric set. Check the installed tool's metric-set help for the actual GPU.
The optional frequency accepts 10–200000 Hz; its default is 10000 Hz.

The collector uses one uniquely named, targetless `nsys start` / `nsys stop` session with CPU sampling
and context-switch tracing disabled, the selected `CUDA_VISIBLE_DEVICES` mask and
`--gpu-metrics-devices=cuda-visible`. It spans tuning and
evaluation cleanup instead of starting competing samplers inside every Worker or Policy process.
Counters cover all processes using those devices; they are not per-PID attribution or a pure-inference
measurement. Sampling may affect timing, so use short matched diagnostics and retain the formal
unprofiled throughput baseline separately. The generic `--cuda-mps` switch is unchanged.

The collector writes `<out>/profile/gpu-metrics.json`; command attempts also write `gpu-metrics.log`.
When collection succeeds, the same directory holds `gpu-metrics.nsys-rep` and its exported
`gpu-metrics.sqlite`. A completed `result.json` embeds the metadata as `gpu_metrics`. Metadata preserves
the requested set, device mask,
frequency, actual tool version, start/stop/export commands, return codes and timestamps, diagnostics,
sample count, and `report_created` / `sqlite_created` flags. Do not assume the report or database exists
on an unavailable path. The run manifest separately records requested `gpu_metrics` settings and
`diagnostic_profile: true`, even if Worker/Policy tracing is off. Autotune candidate and formal manifests
retain that request; it is not the final availability result.

Statuses are `pending`, `collecting`, `available`, `partial` or `unavailable`. `available: true` requires
an exported SQLite `GPU_METRICS` table with a positive row count; retained permission diagnostics can
mark such data `partial`. An accepted CLI option, successful start or empty report is not evidence of
measured counters. Tool, metric-set or permission failures warn and record unavailable data while
normal evaluation continues. The collector does not change host/container counter permissions.

On the 2026-11-06 RoboTwin host, a newer Nsight Systems 2026.6.1 still encounters counter permission
denial under the current container capabilities, including absent `CAP_SYS_ADMIN` / `CAP_PERFMON`.
Container root alone does not grant those permissions. A subsequent native RTX 4090 run obtains real
RT/SM/Tensor/VRAM counters with administrator permissions and the same optional collector; see the
[native follow-up](../../../benchmarks/robotwin/profiling-native-mps.md). CUDA traces, host observation
durations and GPU busy percentage are not replacement hardware-throughput metrics.

## Output

- `<out>/profile/worker-<id>-attempt-<n>.nsys-rep`: the first episode of each Worker attempt, with captured CUDA kernels
  and API calls of the Worker and its child processes (such as the Isaac Sim evaluator), and NVTX ranges:
  - `episode`: from reset to end-of-episode; this is the capture range. Standard SDK clients use domain
    `roboscope`; RoboTwin uses Torch's default NVTX domain in its independent socket client.
  - `policy.<operation>`: each Policy round trip, such as `policy.act`, `policy.observe`, and
    `policy.reset`. RoboTwin also marks inclusive `simulation.observation` and `simulation.action` calls.
- `<out>/profile/policy-<id>-attempt-<n>.nsys-rep`: each Policy process's complete lifetime, including
  loading, inference CUDA work and these host spans:
  - `server.queue.<operation>`: waiting to acquire the existing server request lock.
  - `server.<operation>`: request handling after lock acquisition, including validation and response building.
  - `inference.act` / `inference.act_batch`: the actual Policy method, including its preprocessing and
    output processing. These are not pure GPU-kernel durations.
  A restarted replica gets a separate attempt number. Normal unload lets Nsight finish writing its report.
- `<out>/manifest.json` records `"profile": true`. Tracing slows the run, so its throughput is not a
  production throughput measurement. Policy capture spans all requests while Worker capture spans only
  its first episode; compare matching intervals rather than summing entire reports. Ranges are inclusive,
  nested and parallel, and profiling does not add per-request GPU synchronization.

## Reading a report

- **GUI:** open the `.nsys-rep` in Nsight Systems and inspect the timeline.
- **Command line:** `nsys stats <report>.nsys-rep`.
- **Custom metrics:** export to SQLite and query it. This script prints NVTX range totals and the
  union of **captured CUDA kernel** intervals. It is not whole-GPU busy time: graphics rendering and
  untraced processes are absent. Some simulator reports have valid NVTX events and no CUDA kernel table.

```bash
nsys export --type=sqlite -o report.sqlite worker-000-attempt-000.nsys-rep
python3 - <<'PY'
import sqlite3
c = sqlite3.connect("report.sqlite")
tables = {row[0] for row in c.execute("select name from sqlite_master where type='table'")}
if "NVTX_EVENTS" in tables:
    for row in c.execute("""select coalesce(n.text, s.value), count(*), sum(n.end - n.start) / 1e9
                            from NVTX_EVENTS n left join StringIds s on n.textId = s.id
                            where n.end is not null group by 1 order by 3 desc"""):
        print(row)
busy, covered = 0, 0
kernels = (c.execute("select start, end from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
           if "CUPTI_ACTIVITY_KIND_KERNEL" in tables else [])
for start, end in kernels:
    start = max(start, covered)
    if end > start:
        busy, covered = busy + end - start, end
window = (max(end for _, end in kernels) - kernels[0][0]) / 1e9 if kernels else None
print("captured kernels", len(kernels), "CUDA kernel union s", busy / 1e9, "window s", window)
for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET", "GPU_METRICS"):
    print(table, "present" if table in tables else "not captured")
PY
```

## Coverage

The built-in benchmarks have a Worker episode trigger. RoboTwin retains its independent socket
client and emits equivalent markers directly; switching to `PolicyClient` is unnecessary. Policy processes
are captured separately for every benchmark. Code-path support and completed hardware validation are
different: do not infer that an untested simulator or graphics backend was validated from the shared code.

| Benchmark | Worker capture trigger | Hardware verification |
|---|---|---|
| RoboDojo | SDK `PolicyClient`, first episode | Worker capture verified with placeholder Policy (2026-11-05, K=1 and K=3); new Policy capture not yet revalidated here |
| LIBERO | SDK `PolicyClient`, first episode | Code path checked; not yet hardware-verified |
| LIBERO-Plus | Same Worker module as LIBERO | Code path checked; not yet hardware-verified |
| LIBERO-Pro | Same Worker module as LIBERO | Code path checked; not yet hardware-verified |
| MetaWorld | SDK `PolicyClient`, first episode | Code path checked; not yet hardware-verified |
| robomimic | SDK `PolicyClient`, first episode | Code path checked; not yet hardware-verified |
| RoboCasa | SDK `PolicyClient`, first episode | Code path checked; not yet hardware-verified |
| CALVIN | SDK `PolicyClient`, first subtask | Code path checked; not yet hardware-verified; capture is not the complete five-task sequence |
| RoboTwin | Independent client, first episode | Real StarVLA, K4/g2, paired 20 episodes plus CLI QuickTest, MPS off/on (2026-11-06); Worker/Policy reports verified. Native RTX 4090 counter QuickTest and renderer hardware/CPU profiles verified (2026-11-07) |

The original RoboTwin MPS measurements used external instrumentation. The subsequent built-in integration
adds Worker and Policy coverage; see its [findings](../../../benchmarks/robotwin/README.md#profiling).

## Scope and limits

- Default Worker/Policy tracing is CUDA/NVTX only. Vulkan/OpenGL GPU rendering, CPU sampling/call stacks,
  and SM/Tensor/RT throughput counters are not included in those reports. Device-wide counters require
  the separate explicit collector and actual supported, permitted samples. No kernel rows in a simulator report do not mean
  no rendering work. Missing copy or hardware-metric tables mean unavailable data, not zero usage.
- Hardware-counter permissions are separate from trace support. A successfully written report does not
  establish that counters are present. Inspect its exported tables and diagnostics before reporting them.
  Driver/tool compatibility warnings must also be retained. The RoboTwin validation used Nsight 2023.1
  with a CUDA 13.0 driver interface fallback; final captures report no CUPTI drops after graceful
  shutdown was ordered before MPS client termination.
- Worker captures start at the first episode marker, not model/environment startup. CALVIN resets per
  subtask. SDK reset/end RPC exceptions close their ranges; abrupt crashes can still leave incomplete
  ranges. Capture-end tails are not normal episode durations. Profiles are diagnostics, not a timing
  baseline for unprofiled evaluation.
- The CLI supports `--quick-test` or fixed K/g, not autotune profiling. Custom low-level callers supplying
  an existing `server_pool` must create it with `profile_dir`; a pool cannot retroactively profile an
  already running Policy. Inference adapters without a separate process cannot supply a Policy report
  and are rejected in profiling mode.

## How it works

- The runtime wraps each Worker's command in `nsys profile` with `--trace=cuda,nvtx --sample=none
  --cpuctxsw=none --capture-range=nvtx --nvtx-capture=episode@roboscope --capture-range-end=stop`, so only
  the first episode is captured. RoboTwin matches the `episode` name without restricting its NVTX domain.
- Each Policy process runs under a separate CUDA/NVTX Nsight capture from launch to exit. It has no
  episode capture trigger because several Workers share the same Policy and may overlap their episodes.
- When `nsys` launches a process, it names its injection library in `NVTX_INJECTION64_PATH`.
  `roboscope_kit.profiling` loads that library with `ctypes`, follows the same initialization as NVTX's
  header-only C API (`InitializeInjectionNvtx2` and the export tables) to obtain `nvtxDomainCreateA`,
  `nvtxDomainRangeStartEx`, and `nvtxDomainRangeEnd`, and writes the ranges with them.
- RoboTwin's standalone launcher uses its existing Torch NVTX API only under profiling; its simulator
  remains independent of the Kit package. Ranges are written only when `ROBOSCOPE_PROFILE` is set and
  the process actually runs under `nsys`. Neither path synchronizes CUDA or changes Policy actions.

## Findings

Each benchmark keeps its profiling findings in its own documentation, newest first, with the date, machine,
configuration, and Kit version:

- [RoboTwin native counters](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/robotwin/profiling-native-mps.md):
  Nsight 2026.6.1 on a 24-GB RTX 4090 obtains RT/SM/Tensor/VRAM samples. One/two-scene static controls
  show RT throughput means of 10.44%/8.10%, despite GPU busy means of 83.35%/94.55%; CPU stacks resolve
  rendering-path CUDA synchronization. Repeated b31 K2/G1 ten-episode throughput changes by −0.602%;
  valid fixed-K2 RT throughput is 1.389%/1.396% off/on. Serial active rollout spends 62.26% in action-side
  simulation/control/wait, 30.09% in observations and 7.53% in Policy RPC. Four broader counter runs
  recover K2→1 after camera OOM and are reported separately (2026-11-07).
- [RoboTwin](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/robotwin/profiling-mps.md):
  Worker/Policy capture and MPS client lifecycle repaired; repeated 500-episode K8/G1 throughput changes
  by −0.0293% on one 48-GiB RTX 4090. Fixed-K 100-episode checks do not justify extra default autotune probes.
  CUDA/Vulkan diagnostics identify material observation rendering and request queueing; hardware
  RT/SM/Tensor/DRAM counters were permission-blocked in that container (2026-11-06).
- [RoboDojo](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/robodojo/README.md#profiling):
  captured CUDA kernels cover about 11% of an `arx_x5` episode; the capture does not account for graphics
  GPU work (2026-11-05).
