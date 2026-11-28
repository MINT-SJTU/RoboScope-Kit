# RoboInF policies

Use `policy.py` in the inference environment, with the Kit and the pinned
`<runtime>/RoboInF` checkout on its Python path. This wrapper delegates image
preprocessing, normalization, rotation conventions and action queues to the
selected model's own adapter. The runner executes at most 20 returned actions.

| Argument | Historical local method | Released client |
| --- | --- | --- |
| `pi_05` | ALOHA joint14, 50 targets | OpenPI WebSocket |
| `openwam` | EEF16, 32 targets; retain the checkpoint's pose convention | OpenWAM ModelClient, EEF by default or `qpos` |
| `gr00t` | N1.7 bimanual joint14, 40 targets | Isaac-GR00T PolicyClient |
| `lingbot-va` | EEF16, four-action feedback slices and per-episode KV cache | LingBot-VA WebSocket, one action per call |

For a **historical checkpoint** use its trusted `benchmarking` checkout,
original method manifest, model-specific imports and model environment. Set
the Kit checkpoint source to the method's `config.checkpoint`, or
`config.model_bundle` for LingBot. This keeps the fine-tuned preprocessing
and stateful inference path, rather than silently substituting a stock model.
Example Kit policy JSON (replace paths with absolute paths):

```json
{"schema_version":2,"cwd":"/path/to/experiment",
 "command":["/usr/bin/env","PYTHONPATH=/path/to/runtime/RoboInF:/path/to/experiment:/path/to/benchmarking-parent",
 "/path/to/model-env/bin/python","/path/to/Kit/examples/roboinf/policy.py",
 "gr00t","--method","/path/to/experiment/method.json"]}
```

For a **public upstream server**, start the correct checkpoint separately,
install the model's upstream client, and replace `--method ...` with
`--client-config /path/to/client.json` (e.g. `{"host":"localhost","port":8000}`
for pi_05). See the [pinned client documentation](https://github.com/SJTU-DeepVisionLab/RoboInF/blob/af9d8d583885df48b300ae97f1a45bd07d318b6a/docs/policies.md)
for the other ports, layouts and dependencies. The remote API does not
attest its checkpoint: verify its identity independently. `--chunk N` must
match the actual returned horizon if different from the defaults above.

Start stateful endpoints with `--concurrency 1 --replicas 1` and a fixed
concurrency, not autotuning. Do not share a stateful server across Kit replicas.
Use `--suite test` or `--suite test_random` with the benchmark manifest,
policy configuration and authorized domain-specific judge as described in
the [benchmark guide](../../benchmarks/roboinf/README.md). The only full
Kit correctness comparison is the historical GR00T clear run.
