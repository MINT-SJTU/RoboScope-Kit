# BEHAVIOR2026VLA BEHAVIOR Policy

A `roboscope-kit` `Policy` that serves [`IliaLarchenko/BEHAVIOR2026VLA`](https://huggingface.co/IliaLarchenko/BEHAVIOR2026VLA),
the Robot Learning Collective's single checkpoint for all 100 tasks of the 2026 BEHAVIOR Challenge, on the
`behavior` benchmark's `challenge_2026_rgb224` suite. The benchmark contract is the
[BEHAVIOR protocol](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/packages/roboscope-kit/BEHAVIOR_EVALUATION_PROTOCOL.md).

The model is a π0.5 (PaliGemma plus a flow-matching action expert, 2.9B parameters) with task and stage
embeddings instead of prompts, trained on the challenge's 20,000 demonstrations at 224x224 RGB
([technical report](https://arxiv.org/abs/2611.14432),
[solution code](https://github.com/Robot-Learning-Collective/behavior-1k-solution-2026)). Its authors
report a q-score of 0.392 over the 100 tasks and instances 301-310 for the learned policy alone; their
leaderboard entry (0.446) also hands 14 tasks to an LLM agent whose practice sessions are not public, and
this example does not include it. It is the strongest public checkpoint found (November 2026) that covers
all 100 tasks; [`examples/gr00t`](../gr00t/README.md) serves a GR00T N1.7 fine-tune on the `challenge_2026`
suite, with no published score.

| File | Purpose |
| --- | --- |
| `behavior2026vla_policy.py` | The Policy: runs the solution's evaluation code in process |
| `build-model-env.sh` | Builds the solution's environment (Python 3.11, JAX) and fetches the checkpoint, all pinned |
| `run_native.py` | Runs BEHAVIOR's own `omnigibson.eval.eval`, one process per instance, against the solution's server |
| `roboscope-policy.json` | Policy process contract template |

## Observation

The model was trained and evaluated on the evaluator's `DefaultWrapper` observation: three 224x224 RGB
cameras. `challenge_2026_rgb224` is that observation; the `challenge_2026` suite renders the
`RGBDFullResWrapper` cameras (720x720 and 480x480 RGB-D) instead, which the model was not evaluated on, so
the Policy accepts only `challenge_2026_rgb224`. Both suites run the same tasks, instances, step limits and
metric.

## Process

The solution needs Python 3.11, which the kit supports, so the Policy runs in the solution's own
environment with the kit source on `PYTHONPATH` and no model server. For each episode it creates the
solution's rollout state (`B1KPolicyWrapper`) around the loaded JAX policy and steps it with
`act_batched`, as the solution's `scripts/serve_b1k.py` does for the native evaluator's request of one
environment (the native observation keys with a batch dimension of one: the three RGB images, the 61-dim
proprioception and `task_id`). All model-specific behaviour is the solution's: 30 predicted actions with 20
denoising steps, the first 26 executed over 13 control steps, rolling inpainting, stage voting and the
gripper rule. A model call executes 13 actions, and the steps in between read no observation, so
`act` returns those 13 actions at once and the worker replays them, as the native client does with
`--replay-action-chunk-size 13`. The Policy keeps the rollout state of each open episode, so one replica
serves several workers. Each replica loads the model (6.3 GB); JAX may take up to half of the GPU memory
(`XLA_PYTHON_CLIENT_MEM_FRACTION`, as `serve_b1k.py` sets it), so lower that fraction in the Policy
environment for several replicas on one GPU.

## Model environment

`build-model-env.sh` checks out the solution at commit `68c7f912` with its submodules at the commits it
records (openpi `01177e02` with the solution's two openpi patches, BEHAVIOR-1K `v3.9.3-post1`), installs
its `uv.lock` exactly, then installs BEHAVIOR-1K's `bddl3` and `OmniGibson[eval]`, `av<16` and
`warp-lang` the way the solution's `setup_remote.sh` does. OmniGibson's requirements are resolved by uv
outside the lock, which, as in the solution's own setup, downgrades pillow and cffi to OmniGibson's pins;
the resolved environment is recorded in `b1k-pip-freeze.txt`. OmniGibson asks for LeRobot from the
`wensi-ai/lerobot` `release/b1k` branch, which has since moved to a release that needs Python 3.12, so
the build holds LeRobot at the commit the solution's lock records. Isaac Sim is not installed: the model never
starts the simulator, which runs in the BEHAVIOR runtime. The checkpoint (10.7 GB) is fetched at revision
`dc8c443f` and every file is checked against its Hugging Face digest.

## Run

1. Build the simulator runtime (`benchmarks/behavior/build-runtime.sh --accept-licenses /sim/behavior`)
   and the model environment:

   ```bash
   examples/behavior2026vla/build-model-env.sh /models/behavior2026vla
   ```

   It prints `B1K_SOLUTION_ROOT`, `B1K_PYTHON` and the checkpoint directory. The checkpoint and the
   solution are Apache-2.0; the π0.5 base model's PaliGemma backbone is under the Gemma terms of use.

2. Copy `roboscope-policy.json` and replace the paths. The Policy starts in the solution checkout (`cwd`),
   as `serve_b1k.py` does. `OMNIGIBSON_DATA_PATH` is the runtime's `data/`: OmniGibson's evaluation
   utilities, which the solution imports, read the challenge task table from it.

3. Evaluate from any Python 3.11+ with `roboscope-kit` (`--quick-test` runs one worker and one replica;
   a full run passes `--concurrency K --replicas G`; on one 48 GB RTX 4090 with 20 CPU cores, K=4 workers
   and G=2 replicas):

   ```bash
   python -m roboscope_kit.runtime.benchmark \
       --benchmark behavior --suite challenge_2026_rgb224 --quick-test \
       --checkpoint-source /models/behavior2026vla/checkpoints/BEHAVIOR2026VLA \
       --benchmark-manifest /sim/behavior/benchmark-manifest.json \
       --policy-config /path/to/roboscope-policy.json \
       --output-dir /runs/behavior-rgb224-quick
   ```

4. Run the same workload natively and compare:

   ```bash
   B1K_SOLUTION_ROOT=... B1K_PYTHON=... /sim/behavior/env/bin/python examples/behavior2026vla/run_native.py \
       --behavior-runtime /sim/behavior --task turning_on_radio --instances 1 \
       --checkpoint /models/behavior2026vla/checkpoints/BEHAVIOR2026VLA \
       --output-dir /runs/behavior-rgb224-native-quick
   python examples/gr00t/compare_native.py --kit-run /runs/behavior-rgb224-quick \
       --native /runs/behavior-rgb224-native-quick
   ```

## Results

Instance 301 of all 100 tasks on one RTX 4090 with two workers and two replicas (2026-11-25): 100 of 100
episodes, 23 successes, mean Q 0.410, 16.5 actions/s over 24.1 hours. That is one episode per task, not
the formal 1,000, and has no paired native run yet; the
[protocol document](https://github.com/MINT-SJTU/RoboScope-Kit/blob/main/packages/roboscope-kit/BEHAVIOR_EVALUATION_PROTOCOL.md)
(sections 9 and 10) has the details and the time breakdown. With K=4 and G=2, the chunked actions and
the scene texture caches of Kit `0.8.0b92`, the same machine evaluates about 1.6 times as many actions per
second (section 10).
