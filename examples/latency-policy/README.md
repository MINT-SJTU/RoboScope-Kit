# Latency-bound Policy example

[`latency_policy.py`](latency_policy.py) is a synthetic stand-in for an agent-based Policy on RoboDojo. It uses
no GPU, holds the current joint positions, and sleeps a fixed time in every `act()` call, so it measures how
throughput scales with concurrent episodes when the Policy, not the simulator or the GPU, sets the pace. It is
not a real agent: nothing is planned, and holding still never solves a task, so every episode runs to its step
limit and the success rate is meaningless.

## Run it

Copy [`roboscope-policy.json`](roboscope-policy.json) and set its paths. It runs in the simulator's Python, which
already has `roboscope-kit`. A replica serves one request at a time, so give each Worker its own replica:

```bash
/sim/robodojo/env/bin/python -m roboscope_kit.runtime.benchmark \
    --benchmark robodojo --suite arx_x5 \
    --checkpoint-source /sim/robodojo \
    --benchmark-manifest /sim/robodojo/benchmark-manifest.json \
    --policy-config /path/to/roboscope-policy.json \
    --episodes 1 --concurrency 5 --replicas 5 \
    --output-dir /runs/latency-k5
```

`--checkpoint-source` must exist but is not read. `--latency-s` sets the time per `act()` call and `--chunk` the
actions it returns. The default is 3 s and 10
actions, which is 0.3 s of Policy time per control step.

## Choosing K

Set `K` by GPU memory. Each simulator process needs 8.0 to 10.3 GB depending on the task, and nothing limits
how many start. A simulator that runs out of GPU memory is stopped and counted in `oom_events`, and the Kit
continues with one Worker fewer on that GPU. Results are in [RoboDojo's throughput notes](../../benchmarks/robodojo/README.md#throughput).
