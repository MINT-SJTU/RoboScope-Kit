# RoboScope Kit Profiling

Updated: 2026-11-11

This document explains how to use `--profile` to see where an evaluation's time goes, and records what
profiling has found so far. After every profiling run, add its findings to the benchmark's own documentation and link them from
[Findings](#findings), so results accumulate instead of being measured again from scratch.

## Requirements

- **Nsight Systems CLI.** The machine that runs the evaluation needs
  [Nsight Systems](https://developer.nvidia.com/nsight-systems/get-started) with `nsys` on `PATH`. The Kit
  does not install it. Version 2026.5.1 is verified (installed under `/opt/nvidia/nsight-systems-cli/2026.5.1`
  in the AutoDL container). Without `nsys`, `--profile` stops before any Worker starts.
- **No extra Python package.** NVTX ranges are written with `ctypes` through Nsight Systems' own injection
  library (see [How it works](#how-it-works)), so neither the simulator environment nor the Policy
  environment needs `nvtx`. Without `--profile`, nothing is loaded.
- **A fixed configuration.** `--profile` works only with `--quick-test` or a fixed
  `--concurrency K --replicas G`; autotune probes are never captured.
- **Disk space.** Each Worker writes one report, roughly proportional to the episode's steps: on RoboDojo,
  about 27 MB for a 20-step episode and about 150 MB for a 105-step episode. `nsys` also writes temporary
  files under `/tmp` while capturing.

## Running

```bash
python -m roboscope_kit.runtime.benchmark --benchmark robodojo --suite arx_x5 \
  --checkpoint-source <ckpt> --benchmark-manifest <install-dir>/benchmark-manifest.json \
  --policy-config <policy.json> --output-dir <out> \
  --quick-test --profile
```

To look at concurrent Workers, replace `--quick-test` with `--concurrency K --replicas G`.

## Output

- `<out>/profile/worker-<id>-attempt-<n>.nsys-rep`: the first episode of each Worker, with the CUDA kernels
  and API calls of the Worker and its child processes (such as the Isaac Sim evaluator), and NVTX ranges:
  - `episode` (domain `roboscope`): from `PolicyClient.reset` to `end_episode`; this is the capture range.
  - `policy.<operation>`: each Policy round trip, such as `policy.act`, `policy.observe`, and
    `policy.reset`.
- `<out>/manifest.json` records `"profile": true`. Tracing slows the run, so its throughput is not a
  measurement.

## Reading a report

- **GUI:** open the `.nsys-rep` in Nsight Systems and inspect the timeline.
- **Command line:** `nsys stats <report>.nsys-rep`.
- **Custom metrics:** export to SQLite and query it. This script prints the total time of each NVTX range
  and the GPU busy time (the union of all kernel intervals):

```bash
nsys export --type=sqlite -o report.sqlite worker-000-attempt-000.nsys-rep
python3 - <<'PY'
import sqlite3
c = sqlite3.connect("report.sqlite")
for row in c.execute("""select coalesce(n.text, s.value), count(*), sum(n.end - n.start) / 1e9
                        from NVTX_EVENTS n left join StringIds s on n.textId = s.id
                        where n.end is not null group by 1 order by 3 desc"""):
    print(row)
busy, covered = 0, 0
kernels = c.execute("select start, end from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
for start, end in kernels:
    start = max(start, covered)
    if end > start:
        busy, covered = busy + end - start, end
print("kernels", len(kernels), "busy s", busy / 1e9, "window s", (kernels[-1][1] - kernels[0][0]) / 1e9)
PY
```

## Coverage

Capture starts when the Worker opens the `episode` range. Most workers do this through
`PolicyClient`; RoboCasa365 keeps one range open around its complete native episode group. The Policy Server does not run under `nsys`: its own GPU time
is not in the report, only the round-trip time of each request.
RoboCasa365 opens that range after its group's environment resets and before
Policy reset calls. Construction/reset timing uses separate worker wall timers;
these timers do not establish verified Nsight hardware coverage.

| Benchmark | Uses `PolicyClient` | Verified on hardware |
|---|---|---|
| RoboDojo | Yes | Yes (2026-11-05, K=1 and K=3) |
| LIBERO, LIBERO-Plus | Yes | Not yet |
| MetaWorld | Yes | Not yet |
| RLBench | Yes | Not yet |
| robomimic | Yes | Not yet |
| RoboCasa GR-1 Tabletop | Yes | Not yet |
| RoboCasa365 | Yes; first native group after environment resets | Nsight not verified; worker wall timers recorded separately |
| CALVIN | Yes | Not yet |
| RoboTwin | No, its launcher has its own client | Records nothing |

The [RoboCasa365 epoch2 native correctness reference](https://github.com/QLL-CCDD/RoboScope-Kit/releases/tag/robocasa365-epoch2-native-reference-20261111) completed on the restored RTX 4090 (driver 595.71.05, Xeon 8470Q with a 20-core quota) on 2026-11-11. Its 50 tasks × five selected episodes use a shared 128-step cap, full RGB and FP32/100-step DDPM. This is supporting correctness evidence, not a new profiling capture or throughput result. It adds no verified Nsight coverage and is never a speed denominator. The [completed bounded epoch2 comparison](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/robocasa365/expanded-kg-epoch2-20261111.md) adds three full50×5 SDK arms with external phase/resource telemetry: fixed K3/G2 formal gain 2.399× and provisionally selected S5/G1 versus P3/G2 gain 1.557×, below 2×. It is external fixed-CLI screening, not new SDK built-in autotune. It establishes no new Nsight coverage or isolated Policy-GPU timing. The historical Findings retain their original hardware and source attribution.

## How it works

- The runtime wraps each Worker's command in `nsys profile` with `--trace=cuda,nvtx --sample=none
  --cpuctxsw=none --capture-range=nvtx --nvtx-capture=episode@roboscope --capture-range-end=stop`, so only
  the first episode is captured and no time window has to be guessed.
- When `nsys` launches a process, it names its injection library in `NVTX_INJECTION64_PATH`.
  `roboscope_kit.profiling` loads that library with `ctypes`, follows the same initialization as NVTX's
  header-only C API (`InitializeInjectionNvtx2` and the export tables) to obtain `nvtxDomainCreateA`,
  `nvtxDomainRangeStartEx`, and `nvtxDomainRangeEnd`, and writes the ranges with them.
- Ranges are written only when `ROBOSCOPE_PROFILE` is set and the process actually runs under `nsys`.

## Findings

Each benchmark keeps its profiling findings in its own documentation, newest first, with the date, machine,
configuration, and Kit version:

- [RoboCasa365 bounded epoch2 comparison](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/robocasa365/expanded-kg-epoch2-20261111.md): RTX 4090/driver595, CPU quota20, Kit b36/b37, MPS; full50×5/cap128, common K3/G2 and provisional selected S5/G1 vs P3/G2 (2026-11-11). Formal gain 2.399× fixed and 1.557× selected; recurring step/barrier Worker intervals improve about 4× while Policy RPC wait does not improve. Intervals overlap and are not additive wall-time contributions. Formal GPU busy means are 13.23%/31.06% for the common serial/parallel pair, not saturation or isolated kernel time. Full raw remains verified locally; public full-archive upload returned HTTP408.
- [RoboCasa365 all-task performance](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/robocasa365/approved250-performance-20261109.md): RTX 4090, Kit b34/b35, MPS, K=2/G=1 (2026-11-09);
  slot concurrency improves capped formal throughput 2.387× and Worker step/barrier intervals 3.902×; Policy RPC wait grows.
- [RoboDojo](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/benchmarks/robodojo/README.md#profiling):
  `arx_x5` keeps the GPU busy only about 11% of an episode (2026-11-05).
