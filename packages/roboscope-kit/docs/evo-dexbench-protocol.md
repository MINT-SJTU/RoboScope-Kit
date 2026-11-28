# Evo-DexBench evaluation protocol

The `evo_dexbench` benchmark runs Evo-DexBench's standard `evaluation_v1` protocol: dexterous
single-hand (Panda + Shadow hand) and bimanual tasks on ManiSkill 3.0.1 / SAPIEN 3.0.3, physics
stepped on the CPU at 120 Hz and controlled at 30 Hz. It is tracked in
[issue #140](https://github.com/MINT-SJTU/RoboScope-Kit/issues/140).

## Frozen workload

Evo-DexBench is pinned to
[`818ac28e42e9199f0cbccce8617cb686519d8abf`](https://github.com/MINT-SJTU/Evo-DexBench/tree/818ac28e42e9199f0cbccce8617cb686519d8abf)
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
upstream's execution order. `--episodes N` accepts multiples of four and runs upstream's own N-episode
schedule, balanced over the conditions; `--quick-test` runs one wave of the suite's first task, one
episode in each of four of the five conditions. Runs with fewer tasks or episodes are labeled
`full_suite: false` or `standard_protocol: false`.

## Execution

One work item is one upstream wave: four consecutive episodes of the execution order, stepped by the
upstream `CPUParallelEnv` (four simulator subprocesses) and `CPUProtocolEvaluator` with the frozen
settings (`obs_mode=rgb`, `action_bounds_policy=clip`, observation elision allowed). Only the policy
session is replaced: it is upstream's `BatchedPolicySession` contract with a Kit Policy behind it. A
finished environment freezes until its wave ends, so every outcome depends only on its own
environment and the Policy, as in the native evaluator. Upstream's native log for each wave is kept
under `native-logs/<task>-wave-<NN>.jsonl`; a retried wave keeps the failed attempt's log as
`.failed-N.jsonl`.

## Host memory

A simulator subprocess needs much more memory while its scene loads than afterwards. While the
render resources load, SAPIEN's native layer runs about 1,100 (`kitchen` scenes) to 2,700
(`studio` scenes) threads per environment, independent of CPU affinity, and their working memory
peaks at 8–22 GiB per environment; once reset, a `studio` environment needs about 1.3 GiB. Until
[Evo-DexBench #2](https://github.com/MINT-SJTU/Evo-DexBench/pull/2), the four environments of a
wave loaded at once and glibc kept most of the freed peak, so a Worker held 27–36 GiB (`kitchen`)
or 79–89 GiB (`studio`) for its whole run.

The pinned revision includes that fix: the environments load one at a time and return freed heap
memory to the system after every build. Measured with four environments on 2026-11-25, one RTX 4090
host, glibc 2.35:

| Scene | Tasks | Peak while loading | After loading | First load |
|---|---|---:|---:|---:|
| `studio` (m01) | m01 m03 m05 m07 m13–m15, b02 b04 b06 b09 b11–b15 | 26.7 GiB | 6.1 GiB | 177 s |
| `kitchen` (m04) | the other 14 | 13.3 GiB | 5.7 GiB | 92 s |

Memory grows about 35% during a rollout. Size a host for the load peak per Worker that may load at
the same time; a Worker that exceeds its container's memory limit loses environment subprocesses
(`BrokenPipeError`/`EOFError`), usually with no OOM kill reported. Loading one at a time changes no
result: the upstream pull request compared observations, rewards and episode records bit for bit.

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

Status: code candidate, partial. Ten of the 30 tasks have been compared with the native
evaluator so far; the other tasks, and native repeatability, are still to run, and the integration
is not accepted until they pass. The tolerance is a per-task SR and StageScore difference within two
standard errors.

From 2026-11-24 to 2026-11-26, at the previous pin
[`3013f21`](https://github.com/MINT-SJTU/Evo-DexBench/tree/3013f21f3c483e0fa748b55fc56e00a671f4068c)
(the current pin changes only how scenes are built, not any result), upstream DP-T checkpoints ran the
standard 100 episodes of each task on one RTX 4090 host, once through the native evaluator
(`dexbench evaluate` with the upstream `dp_policy.server`) and once through the Kit (one Worker, the
native four environments). Each task uses its group's checkpoint (Core, or Gen for `m10_v1`);
`b11_v1` uses Double3's `interaction` camera.

| Task | Native SR | Kit SR | Native StageScore | Kit StageScore |
|---|---:|---:|---:|---:|
| `m01_v1` | 0.25 | 0.18 | 0.313 | 0.222 |
| `m02_v1` | 0.37 | 0.34 | 0.586 | 0.553 |
| `m04_v1` | 0.48 | 0.50 | 0.792 | 0.800 |
| `m06_v1` | 0.16 | 0.21 | 0.235 | 0.270 |
| `m08_v1` | 0.14 | 0.11 | 0.525 | 0.515 |
| `m09_v1` | 0.14 | 0.12 | 0.483 | 0.478 |
| `m10_v1` | 0.32 | 0.26 | 0.491 | 0.461 |
| `m11_v1` | 0.00 | 0.00 | 0.115 | 0.125 |
| `b07_v1` | 0.01 | 0.00 | 0.158 | 0.152 |
| `b11_v1` | 0.54 | 0.56 | 0.627 | 0.617 |

DP sampling is not seeded upstream, so runs agree statistically, not per episode; every difference
is within 1.6 standard errors. Both paths used the same pinned packages on CPython 3.11 (upstream's
own environments use 3.10) with the open-file limit raised to the hard limit. `m02_v1` took 32 min 39 s
natively and 32 min 2 s through the Kit, `b11_v1` 62 min 8 s and 59 min 45 s. The runs of `m04_v1`–`m11_v1`
and `b07_v1` set `MALLOC_ARENA_MAX=2` (except the Kit run of `m04_v1`), which changes only where memory
comes from.
