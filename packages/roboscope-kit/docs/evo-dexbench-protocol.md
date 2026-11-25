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
