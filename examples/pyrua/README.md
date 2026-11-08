# PyRUA-Lean Policy example

This experimental Policy connects [PyRUA-Lean](https://github.com/DAGroup-PKU/PyRUA-Lean)'s
RoboDojo agent to the Kit's RoboDojo Workers. It uses PyRUA's own `Bridge`, episode runner,
robot primitives and SAM3 perception. The Policy translates the Kit transport; it does not
replace the agent with a direct image-to-joint policy.

The adapter targets PyRUA-Lean commit `c6861200bfb190169953d1827260ccf71320eea1`, using
`robodojo_xpl_rgb`, `cells`, `no-vla`, on-demand images and pure feedback. The agent can wait
without advancing the simulator, and early finish asks RoboDojo to judge the current scene.
The actual simulator verdict is reported back to PyRUA. Each Worker uses one environment;
lockstep batches and asynchronous observation overlap are not supported for this Policy.
Interactive Policies use `act` and `poll`; `act_batch` is not supported.
The adapter records the Kit episode and native run IDs in `kit_episode.json`, alongside the agent result.
After the simulator reports its verdict, the adapter waits up to twelve minutes for the agent to finish.
PyRUA changes process-wide working directory and output streams, so each Policy replica accepts
one active episode. Use `--concurrency K --replicas K`; all replicas may be assigned to the same GPU
if its memory permits. Compare configurations with replica count and memory usage explicitly reported.

Build the RoboDojo runtime with [`build-runtime.sh`](../../benchmarks/robodojo/build-runtime.sh).
Create a separate Python 3.11 Policy environment containing that PyRUA revision, its dependencies,
SAM3, and this version of `roboscope-kit`. Obtain the official SAM3 checkpoint and set
`SAM3_CHECKPOINT_PATH` in that environment. Follow the upstream PyRUA and SAM3 installation and
model-license instructions; do not install Policy dependencies into the simulator environment.

Copy [`roboscope-policy.json`](roboscope-policy.json), replace its paths, and supply an available
Codex executable and model. A small integration check can be run with a new output directory:

```bash
/sim/robodojo/env/bin/python -m roboscope_kit.runtime.benchmark \
    --benchmark robodojo --suite arx_x5 \
    --checkpoint-source /path/to/sam3.pt \
    --benchmark-manifest /sim/robodojo/benchmark-manifest.json \
    --policy-config /path/to/roboscope-policy.json \
    --output-dir /runs/pyrua-smoke --quick-test
```

This issues real model requests unless the configured Codex executable is a fake service. The
example itself does not enforce an upstream request budget. Record model, reasoning effort, task,
layout, seed, Tool and simulator versions when comparing runs. The published PyRUA-Lean leaderboard
entry used a different model from the example's Terra configuration; a result with another model is
not a reproduction of that leaderboard score. A completed run may still fail its robot task.

## Optional shared SAM3

On a trusted single host, Policy replicas can use one local SAM3 service to reduce the memory used
by multiple model copies. Start the service in the Policy environment and wait for its ready message:

```bash
/path/to/pyrua-policy-env/bin/python \
    /path/to/RoboScope-Kit/examples/pyrua/pyrua-shared-sam3/sam3_http.py \
    --checkpoint /path/to/sam3.pt --port 8790
```

Copy [`roboscope-policy-shared-sam3.json`](roboscope-policy-shared-sam3.json), replace the paths,
and use it as the Policy config. Keep the service running until evaluation finishes. Its HTTP server
listens on loopback by default and has no authentication. The wrapper requires the pinned PyRUA
local SAM3 and this exact `pyrua_policy.py`, and fails the Policy if the service is unavailable.
The service serializes SAM3 requests, so replica throughput and memory headroom depend on the workload.
The vendored service and its attribution are described in
[`pyrua-shared-sam3/README.md`](pyrua-shared-sam3/README.md).
