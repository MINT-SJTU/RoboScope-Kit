# RoboDojo / Simate-beta simulation evaluation recipe

Originally developed by [@QLL-CCDD](https://github.com/QLL-CCDD).

This directory contains a standalone recipe and runnable example that connect
the upstream RoboDojo simulator directly to the official XPolicyLab model
server. It pins Isaac Sim 5.1.0, the official Simate-beta checkpoint, and
`stack_blocks / arx_x5 / seed 1`. The current full scope is 50 layouts, one
episode each. It is not MT50 or the full RoboDojo task suite.

Performance figures in these docs are reference results from the original
experiments, measured with an earlier, unreleased version of the kit. They have
not yet been re-verified on roboscope-kit. The parallel entry point reuses the
kit's existing autotune. Observation FIFO, control-state reuse, video, render,
and thread candidates are all off by default. The success-rate drop seen with
asynchronous observation over 50 episodes is not fully explained yet; short runs
are not a substitute for full validation. This recipe does not register a
platform benchmark, a kit `Policy`, or an official image.

## 1. Scope and pinned inputs

| Item | Scope declared by this recipe |
| --- | --- |
| Model | Official Simate-beta multi-task FP32 AE checkpoint; no training required |
| Simulation task | `stack_blocks` only: stack three blocks with different textures; color order is not required |
| Robot / action | `arx_x5` dual-arm, 14-dim joint actions |
| Minimal validation | seed 1, layout 0, 1 episode |
| Full validation of the declared scope | seed 1, layouts 0–49, 50 episodes, single process, single environment |
| Earlier pilot | seed 1, layouts 0–5, 6 episodes |
| Limit / success | Native limit of 550 executed actions per episode; three blocks stacked and arms returned to home |
| Simulation / observation | physics dt=0.004 s, nominal observation/control rate 25 Hz |
| Model settings | FP32, 10 flow steps; predicts 50 actions, executes the first 10 |
| Memory | Model keeps 25 frames of history, sampled every 20 actions, AE grid 4×4 |

The 50-episode run is still a single-task evaluation, not MT50, all RoboDojo
tasks, or the full official leaderboard protocol. The six-episode result is kept
as a pilot and does not count as the current full validation.

[recipe.toml](recipe.toml) is the configuration the preparation and run tools
actually read: download sources, digests, and evaluation parameters. It is a
run input, not an experiment archive. Per-file checksum manifests, run outputs,
and raw experiment material are not committed; see the
[patch docs](patches/README.md) for how patches are checked.

Source revisions. The upstream installer moves the submodules to their main
branches, so they are recorded per run rather than enforced:

| Source | Original experiments | Re-validated 2026-11-04 (upstream `install.sh -i`) |
| --- | --- | --- |
| [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) | `f57e1633` | `f57e1633` |
| [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) | `5d0be149` | `6baccd57` |
| [IsaacLab fork](https://github.com/yuechen0614/IsaacLab) | `afca7b09` | `afca7b09` |
| [CuRobo fork](https://github.com/yuechen0614/curobo) | `d17b54ce` | `895c6517` |

Pinned data: [official assets and checkpoint](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/tree/35efbc7dedfdbeeb6e95fb749bd885d73d483e41)
at `35efbc7dedfdbeeb6e95fb749bd885d73d483e41`.

Checkpoint relative path:
`ckpt/RoboDojo/Simate-beta/RoboDojo-sim-arx_x5-joint-0/`, four files totaling
13,502,053,373 bytes. The model also matches the authors'
[AE release](https://huggingface.co/SIPAILab/sipai-robodojo-eval-ae/tree/0738aab54e0c873db76d4d2aec7a467ca1b04120);
do not substitute the pooling-only weights.

| File | SHA-256 |
| --- | --- |
| `model.safetensors` | `11424ad091c37d6192d8fb18228c11b5f5d1fff30436ad8d20bbfebc89d00e3a` |
| `tokenizer.model` | `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6` |
| `LICENSE` | `43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1` |
| `README.md` | `42100c3470106a33c481f2eba0ca43feb95baf36e416178b5b393562ceab23ab` |

The 15,365 asset files total 41,269,513,111 bytes. The configuration also pins
digests of the asset and checkpoint file manifests. The downloader first
verifies the official revision and manifest, then checks each file's size and
SHA-256 or Git blob SHA-1; files that fail verification are rejected. Upstream
licenses are preserved.

## 2. Environment and installation

Requirements: Linux x86_64, working NVIDIA RTX/Vulkan rendering, Git, a CUDA
compiler toolchain, and root access to install system packages. The official
Isaac Sim 5.1 minimum includes an RTX 4080/16 GB and 32 GB RAM, and the
validated Linux driver is **580.65.06**; see the
[official requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html).
In the original experiments, the native six-episode run peaked at 23,594 MiB
and the upstream two-environment run at 23,934 MiB, so a 24 GB GPU has little
headroom. Tested hardware: a 49,140 MiB GPU with a 20-CPU quota and 96 GiB RAM,
and a 24,564 MiB GPU with 32 GB RAM (native single-episode peak 22,757 MiB).
Reserve at least 150 GB of free disk for assets, weights, simulator packages,
and caches. This is an engineering capacity estimate, not a guarantee for every
task.

Both environments come from the upstream installers, unchanged: RoboDojo's
`scripts/install.sh -i` creates the conda environment `RoboDojo` (Python 3.11,
torch 2.7.0+cu128, Isaac Sim 5.1.0.0, IsaacLab, CuRobo), and XPolicyLab's
`policy/Simate_beta/install.sh` creates `simate-beta-eval` for the model
server. `install.sh -i` moves the three submodules to their main branches, so
the exact source revisions depend on the install date; every run records the
revisions it used, and the patch tool checks each patch against the checkout
you have. The recipe tools only need Python 3.11+ (the `RoboDojo` environment
works).

Isaac Sim is proprietary NVIDIA software governed by the
[NVIDIA Omniverse License Agreement](https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html).
This recipe does not accept the EULA for you: after reading it, run
`export OMNI_KIT_ACCEPT_EULA=YES` yourself. `run_eval.py` and the parallel
entry point exit immediately if it is not set. Whether to set
`PRIVACY_CONSENT` is your decision; the recipe does not set it.

```bash
export OMNI_KIT_ACCEPT_EULA=YES   # only after reading the EULA
KIT=/path/to/roboscope-kit
RECIPE="$KIT/examples/robodojo"
BASE=/path/to/robodojo
ROOT="$BASE/RoboDojo"

git clone https://github.com/RoboDojo-Benchmark/RoboDojo.git "$ROOT"
cd "$ROOT"
bash scripts/install.sh -i
bash XPolicyLab/policy/Simate_beta/install.sh simate-beta-eval
PYTHON="$HOME/miniconda3/envs/RoboDojo/bin/python"

# Official assets and the Simate-beta checkpoint, hash-verified while downloading.
"$PYTHON" "$RECIPE/prepare.py" assets "$ROOT" --workers 8
"$PYTHON" "$RECIPE/prepare.py" model "$ROOT" --workers 4

# Change the sources only with the committed patch tool; never edit the runtime directory by hand.
"$PYTHON" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch episode-count
"$PYTHON" "$RECIPE/prepare_runtime_assets.py" apply "$ROOT"
"$PYTHON" "$RECIPE/prepare_runtime_assets.py" check "$ROOT"
```

After the upstream install, `pip check` in `RoboDojo` reports conflicts that
come from the upstream pins (IsaacLab declares `starlette==0.49.1` while the
installer pins 0.45.3; pytest/packaging; httpx2/idna). They do not affect
evaluation; the validated runs below used exactly this environment.
`prepare.py verify` re-hashes all files and is optional.

At the pinned upstream revision, `stack_blocks` defaults to a limit of
**25 episodes**, and `EVAL_NUM` can only lower it. This recipe uses
`episode-count` to extend that task's configuration to **50 episodes**; the
native and measured entry points use the same configuration. No other tasks,
layouts, or per-episode parameters change. This is a declared single-task
extension, not the upstream default suite. Before launch, the tools check the
effective limit and layout files and refuse silent truncation.

Optional download settings: `--endpoint https://hf-mirror.com`; with aria2c
installed, add `--downloader aria2` to the model command; `--transport
modelscope` only swaps the transport and still verifies against the locked
official hashes. The raw manifest can be supplied offline with
`--inventory-file`, and its digest must match. Download receipts and run
evidence are written to `BASE/evidence/` and are not tracked in Git.

On hosts in mainland China, pip follows your `pip.conf` mirror, but the
installers fetch torch from `download.pytorch.org` and the submodules from
GitHub; if large GitHub clones break off, route GitHub through a mirror for the
install only (for example with `GIT_CONFIG_COUNT=1
GIT_CONFIG_KEY_0=url.<mirror>/https://github.com/.insteadOf
GIT_CONFIG_VALUE_0=https://github.com/`) rather than editing the scripts.

## 3. Check the renderer first

A real RTX startup check must pass first; working CUDA tensors do not mean the
renderer works. Adjust the NVIDIA ICD path for your machine; the example below
uses the path on one tested system. The check passes only when both markers
appear and the exit code is 0. Do not bypass it by disabling image rendering.

```bash
env -u CUDA_VISIBLE_DEVICES \
  ACCEPT_EULA=Y \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 \
  VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  "$PYTHON" -u - --allow-root <<'PY'
from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'multi_gpu': False, 'active_gpu': 0})
print('RENDERER_READY', flush=True)
for _ in range(3):
    app.update()
print('RENDERER_UPDATES_COMPLETE', flush=True)
app.close()
PY
```

The two markers are `RENDERER_READY` and `RENDERER_UPDATES_COMPLETE`. The 5.1
default fast shutdown ends the process directly, so do not expect Python output
after `app.close()`. On a standard Ubuntu host the ICD is often at
`/usr/share/vulkan/icd.d/nvidia_icd.json`; adjust the command to the actual
file. The launcher looks for the NVIDIA ICD in both directories, or you can pass
`--vulkan-icd /absolute/path.json` explicitly without changing the system Vulkan
configuration.

After switching machines, repeat the RTX check before starting a real model
evaluation; CUDA availability does not guarantee a working renderer.
Environment troubleshooting is not counted as a runtime speedup.

## 4. Native entry point, measured entry point, and results

`run_eval.py` is a runnable serial example. It starts the upstream Simate
server and the RoboDojo client, connected over XPolicyLab WebSocket.
`--mode native` applies no measurement patch; `measured` uses the same model,
task, and control loop and only adds timing and result fields. The full command
and configuration of each are written to its `invocation.json`. The tool records
source revisions and checks patch state and unexpected code changes; results from
different profiles or simulators cannot be treated as paired runs.

```bash
# Minimal validation: native layout 0, with exactly the same settings as the measured entry point.
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch measurement # before
"$PYTHON" "$RECIPE/run_eval.py" "$ROOT" \
  --mode native --seed 1 --episodes 1 --label native-smoke
"$PYTHON" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch measurement
"$PYTHON" "$RECIPE/run_eval.py" "$ROOT" \
  --mode measured --seed 1 --episodes 1 --label measured-smoke
"$PYTHON" "$RECIPE/compare_runs.py" /path/to/native-smoke /path/to/measured-smoke

# Current full scope: same profile, checkpoint, seed, and 50 layouts.
"$PYTHON" "$RECIPE/apply_runtime_patch.py" rollback "$ROOT" --patch measurement
"$PYTHON" "$RECIPE/run_eval.py" "$ROOT" \
  --mode native --seed 1 --episodes 50 --label native-fifty
"$PYTHON" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch measurement
"$PYTHON" "$RECIPE/run_eval.py" "$ROOT" \
  --mode measured --seed 1 --episodes 50 --label measured-fifty
"$PYTHON" "$RECIPE/compare_runs.py" /path/to/native-fifty /path/to/measured-fifty
```

Replace `/path/to/...` with the `BASE/evidence/evaluation/<run_id>/` printed by
the launcher. `--timeout-s` sets the client wait limit; a timeout counts as
incomplete and does not change the per-episode action limit. By default only
the visible GPU 0 is used. This is a single-task, single-environment reference
and does not claim to beat upstream parallel modes. See
[model-contract.md](model-contract.md) for the full description of inputs,
outputs, and reset.

| Output | Contents |
| --- | --- |
| `summary.json` | `complete`, requested/completed episodes, successes/success rate, layout IDs, per-phase/total time, and GPU metrics |
| `result.json` | Full upstream result: per-episode `success` and partial `score`; measured mode also records executed actions and termination reason |
| `events/*.jsonl` | Reset, execution, and model-call timings, used by the parallel autotune |
| `client.log` / `server.log` | Errors, model loading, native action progress |
| Upstream task run directory | `ROOT/eval_result/RoboDojo/stack_blocks/Simate_beta/arx_x5/<seed>_<label>/<run_id>/`, including videos |

A complete result requires exit code 0, `complete=true`, completed count equal
to requested count, and layout IDs exactly 0 through N−1 with no duplicates.
Missing, duplicated, or substituted layouts, crashes, and timeouts all fail the
run. Failed episodes still count in the denominator; unstable or abandoned
records are kept, and missing episodes are never assumed complete. SIGINT/SIGTERM
cleans up child processes created by the run and keeps the failure record.

With a single task, the overall success count/rate equals that task's.
`score` is a partial-completion score and does not replace the success rate.
Upstream `eval_time` is the number of completed episodes, not seconds. For
example, `completed_episodes=50, successes=29, success_rate=0.58` means 50
episodes completed and 29 succeeded; it is not a guaranteed value for later
runs.

### Smoke validation on the upstream install (2026-11-04)

A fresh RoboDojo clone installed with the two upstream scripts above (sources as
in the table in section 1), on a 49,140 MiB RTX 4090 with a 12-CPU quota:

| Run | Result |
| --- | --- |
| All 7 runtime patches applied stacked, then rolled back in reverse | clean checkout |
| RTX renderer check | both markers, exit 0 |
| `run_eval.py --mode native`, layout 0 | complete, 1/1 success, peak 22,351 MiB |
| `run_eval.py --mode measured`, layout 0 | complete, 1/1 success, 269 actions, 69.6 s formal window, peak 22,479 MiB; `compare_runs.py` passes |
| parallel `--mode fixed`, K2/G1, layouts 0–1 | complete, 1/2 success |
| parallel `--mode on`, `--maximum-workers 2`, 2 probe + 2 evaluation episodes | SDK `two-phase` selected K2/G1 (6.19 vs 4.35 actions/s for K1), complete, 1/2 success |

These are smoke runs, not a success-rate or throughput claim.

## 5. Optional runtime options and limitations

- [Full 50-episode run](full-series.md), [model input/action/reset contract](model-contract.md).
- [Parallelism and existing autotune](autotune.md), [observation RPC/FIFO](observation-runtime.md), [control-state reuse](control-state.md).
- [Per-component profiling](execution-profile.md), [video writing](async-video.md), [render pacing](unpaced-render.md), [PhysX threads](physics-threads.md).
- [Applying, checking, and rolling back patches](patches/README.md). Apply every candidate only after the relevant processes have exited; do not hand-edit the shared runtime or save images.

Identical or similar single-run scores do not prove functional equivalence.
The model RNG is consumed continuously across episodes, and native
rendering/physics also varies between runs; keep per-layout success/failure,
action counts, termination, and input differences. Model code and
precision/history/chunk settings are outside the scope of these optimizations.

After changing the task, robot, or model, re-pin the configuration and run a
native baseline; do not inherit this task's success-rate conclusions.
Installation constraints and the recipe configuration are read by the tools;
results and raw evidence live outside the repository.
