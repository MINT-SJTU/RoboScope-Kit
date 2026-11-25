# Evo-DexBench evaluation protocol

The `evo_dexbench` benchmark runs Evo-DexBench's standard `evaluation_v1` protocol: dexterous
single-hand (Panda + Shadow hand) and bimanual tasks on ManiSkill 3.0.1 / SAPIEN 3.0.3, physics
stepped on the CPU at 120 Hz and controlled at 30 Hz. It is tracked in
[issue #140](https://github.com/MINT-SJTU/RoboScope-Kit/issues/140).

## Frozen workload

Evo-DexBench is pinned to
[`3013f21f3c483e0fa748b55fc56e00a671f4068c`](https://github.com/MINT-SJTU/Evo-DexBench/tree/3013f21f3c483e0fa748b55fc56e00a671f4068c)
and its public asset bundle to revision `09868871eafe6fffdbd8e37b053b6b1f7e9ea6ec`. The 30 tasks,
their horizons (1,000 to 5,000 control steps) and splits are frozen in
[`protocol.py`](../src/roboscope_kit/runtime/benchmarks/evo_dexbench/protocol.py); every Worker
checks the installed upstream protocol fields and source revision before it runs.

| Suite | Tasks | Episodes per task | Expected records |
|---|---|---:|---:|
| `single_hand` | `m01_v1`–`m15_v1` | 100 | 1,500 |
| `bimanual` | `b01_v1`–`b15_v1` | 100 | 1,500 |

Each task's 100 episodes are upstream's schedule: 20 per condition (`nominal`, `spatial`, `camera`,
`lighting`, `physical`), 34/33/33 over the task's three layouts, generalization seeds 1001–1005, in
upstream's execution order. `--episodes` accepts multiples of four; `--quick-test` runs the first
four episodes of the suite's first task. Runs with fewer tasks or episodes are labeled
`full_suite: false` or `standard_protocol: false`.

## Execution

One work item is one upstream wave: four consecutive episodes of the execution order, stepped by the
upstream `CPUParallelEnv` (four simulator subprocesses) and `CPUProtocolEvaluator` with the frozen
settings (`obs_mode=rgb`, `action_bounds_policy=clip`, observation elision allowed). Only the policy
session is replaced: it is upstream's `BatchedPolicySession` contract with a Kit Policy behind it. A
finished environment freezes until its wave ends, so every outcome depends only on its own
environment and the Policy, as in the native evaluator. Upstream's native log for each wave is kept
under `native-logs/`.

## Host memory

Each simulator subprocess holds a whole scene, and the size is set by the task's `scene_class`, not
by the task. Measured on 2026-11-25 with the upstream evaluator and the Kit (same environment code,
same numbers), one RTX 4090 host, glibc 2.35, 128 visible cores:

| Scene | Tasks | Per environment after reset | Four environments (one wave) |
|---|---|---:|---:|
| `studio` | m01 m03 m05 m07 m13–m15, b02 b04 b06 b09 b11–b15 | 20–22 GiB | 79–89 GiB |
| `kitchen` | the other 14 | 7–9 GiB | 27–36 GiB |

Memory grows about 35% further during a rollout (a `kitchen` environment reaches about 12 GiB). A
host needs this much per Worker; a Worker that exceeds its container's memory limit loses
environment subprocesses (`BrokenPipeError`/`EOFError`), usually with no OOM kill reported.

Much of this is allocator overhead, not scene data. While a scene loads, SAPIEN's native layer
starts about 1,100 threads per environment (independent of CPU affinity), and glibc gives
concurrently allocating threads their own malloc arenas, up to eight per visible core. After loading,
a `kitchen` environment holds about 480 arenas of 64 MiB and a `studio` one about 150; freed memory
stays in them. Capping the arenas with `MALLOC_ARENA_MAX=2` in the environment of the evaluating
process (its simulator subprocesses inherit it) measured:

| Scene | Default | `MALLOC_ARENA_MAX=2` |
|---|---:|---:|
| `kitchen` (m04) | 9.5 GiB | 3.3–5.3 GiB |
| `studio` (m01) | 22.6 GiB | 17.4 GiB |

Environment creation and reset took the same time either way. The setting changes only where memory
comes from, not any computation; its effect on step time and results is being checked in the
acceptance matrix, which runs with it on both paths for all but its first jobs and records the
setting per job. Hosts with fewer visible
cores get fewer arenas by default and see less of the overhead.

## Policy interface

Per slot the Policy receives the two most recent control-step frames:

| Feature | Shape | Content |
|---|---|---|
| `images.<camera>` | `(2, H, W, 3)` uint8 | single hand: `context`, `interaction`, `wrist_right`; bimanual: `context`, `wrist_left`, `wrist_right`, and `interaction` when the scene has one |
| `state.<role>.end_effector_pose` | `(2, 7)` | upstream TCP pose: scene-frame position, then quaternion in ManiSkill wxyz order |
| `state.<role>.joint_position` | `(2, 31)` | arm and hand joint positions |
| `language.instruction` | string | the task instruction |

It returns an action chunk of `evo-dexbench.semantic-v1` actions, 30 dimensions per role (6D
end-effector delta and 24 hand joints; 60 for bimanual), which upstream's `decode_semantic_action`
turns into controller commands. As in the upstream DP session, frames are rendered only when the next
request needs them.

## Validation

Status: code candidate, partial. Two of the 30 tasks have been compared with the native
evaluator so far; the full matrix of every task, its native comparison and native repeatability are
still to run, and the integration is not accepted until they pass.

On 2026-11-24 upstream DP-T checkpoints ran the standard 100 episodes of one single-hand and one
bimanual task on one RTX 4090 host, once through the native evaluator (`dexbench evaluate` with the
upstream `dp_policy.server`) and once through the Kit (one Worker, the native four environments).
`b11_v1` uses Double3's `interaction` camera.

| Task (checkpoint) | Path | SR | StageScore | Wall time |
|---|---|---:|---:|---:|
| `m02_v1` (Single1 Core) | Native | 0.37 | 0.586 | 32 min 39 s |
| | Kit | 0.34 | 0.553 | 32 min 2 s |
| `b11_v1` (Double3 Core) | Native | 0.54 | 0.627 | 62 min 8 s |
| | Kit | 0.56 | 0.617 | 59 min 45 s |

DP sampling is unseeded upstream, so runs agree statistically, not per episode; the SR differences
are 0.4 and 0.3 standard errors. Both used the same pinned packages on CPython 3.11 (upstream's own environments
use 3.10) with the open-file limit raised to the hard limit.
