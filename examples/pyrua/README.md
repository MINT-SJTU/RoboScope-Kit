# PyRUA-Lean Policy example

This experimental Policy runs [PyRUA-Lean](https://github.com/DAGroup-PKU/PyRUA-Lean)'s RoboDojo agent on the
Kit's RoboDojo Workers. The agent keeps PyRUA's own `Bridge`, episode runner, robot primitives and SAM3
perception; the Policy only translates the Kit transport.

The adapter targets PyRUA-Lean commit `c6861200bfb190169953d1827260ccf71320eea1`, using `robodojo_xpl_rgb`,
`cells`, `no-vla`, on-demand images and pure feedback. It uses the Kit's interactive Policy contract
([package README](../../packages/roboscope-kit/README.md#episode-state)): the agent can wait without advancing
the simulator, an early finish asks RoboDojo to judge the current scene, and the simulator's verdict is
reported back to PyRUA.

## Set up

1. Build the RoboDojo runtime with [`build-runtime.sh`](../../benchmarks/robodojo/build-runtime.sh).
2. Create a separate Python 3.11 Policy environment containing that PyRUA revision, its dependencies, SAM3, and
   this version of `roboscope-kit`. Follow the upstream PyRUA and SAM3 installation and model-license
   instructions. Keep the Policy dependencies out of the simulator environment.
3. Obtain the official SAM3 checkpoint and set `SAM3_CHECKPOINT_PATH` in the Policy environment.
4. Copy [`roboscope-policy.json`](roboscope-policy.json), replace its paths, and supply an available Codex
   executable and model.

## Run it

A small integration check, with a new output directory:

```bash
/sim/robodojo/env/bin/python -m roboscope_kit.runtime.benchmark \
    --benchmark robodojo --suite arx_x5 \
    --checkpoint-source /path/to/sam3.pt \
    --benchmark-manifest /sim/robodojo/benchmark-manifest.json \
    --policy-config /path/to/roboscope-policy.json \
    --output-dir /runs/pyrua-smoke --quick-test
```

This issues real model requests unless the configured Codex executable is a fake service.

PyRUA changes the process-wide working directory and output streams, so each Policy replica accepts one active
episode. Run with `--concurrency K --replicas K`; all replicas may be assigned to the same GPU if its memory
permits.

The adapter records the Kit episode and native run IDs in `kit_episode.json`, alongside the agent result. After
the simulator reports its verdict, the adapter waits up to twelve minutes for the agent to finish.

## Optional shared SAM3

On a trusted single host, Policy replicas can use one local SAM3 service instead of one model copy each, which
saves GPU memory. Start the service in the Policy environment and wait for its ready message:

```bash
/path/to/pyrua-policy-env/bin/python \
    /path/to/RoboScope-Kit/examples/pyrua/pyrua-shared-sam3/sam3_http.py \
    --checkpoint /path/to/sam3.pt --port 8790
```

Copy [`roboscope-policy-shared-sam3.json`](roboscope-policy-shared-sam3.json), replace the paths, and use it as
the Policy config. Keep the service running until evaluation finishes. The wrapper requires the pinned PyRUA
local SAM3 and this exact `pyrua_policy.py`, and fails the Policy if the service is unavailable. The vendored
service and its attribution are described in [`pyrua-shared-sam3/README.md`](pyrua-shared-sam3/README.md).

## Results

On one RTX 4090 48 GB with GPT-6.1 Terra, two runs of 8 episodes from four tasks took 3,438.6 s of evaluation
with four Workers and shared SAM3, against 10,855.4 s with one Worker: 3.16× the episode throughput
(2026-11-08). Every measured configuration, with its table and limits, is in
[RoboDojo's throughput notes](../../benchmarks/robodojo/README.md#throughput).

## Limits

- Each Worker uses one environment. Lockstep batches, asynchronous observation overlap and `act_batch` are not
  supported for this Policy.
- The SAM3 service listens on loopback by default and has no authentication. It serializes SAM3 requests, so
  replica throughput and memory headroom depend on the workload.
- The example enforces no request budget for the model service.
- The published PyRUA-Lean leaderboard entry used a different model from this example's Terra configuration, so
  a result here is not a reproduction of that leaderboard score.
- When comparing runs, record the model, reasoning effort, task, layout, seed, replica count, memory usage, and
  the tool and simulator versions.
