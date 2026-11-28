# RoboInF test and test_random

This integration exposes the `test` and `test_random` domains of
[RoboInF](https://github.com/SJTU-DeepVisionLab/RoboInF), with `joint14_absolute`
and `eef16_absolute_wxyz` actions. Both domains use the same 861 case IDs and
115 scenes; only `test_random` applies the released per-scene textures and
lighting. This is not a custom appearance variant. It does not cover val or training.
RoboInF contributions are CC BY 4.0 and third-party assets retain their
licenses; obtain RoboTwin assets separately. Only the public evaluation
tables are downloaded, not demonstrations or judgments.

Point `ROBOINF_SIM_ENV` at a provisioned RoboInF-compatible simulator
environment (NumPy 1.26.4, MPLib 0.2.1, CuRobo, SAPIEN and MessagePack)
and `ROBOINF_ROBOTWIN_ROOT` at the upstream simulator checkout with assets.
Run `benchmarks/roboinf/build-runtime.sh /path/to/new/runtime`.
This assembles the evaluator, **not** the licensed assets or GPU environment.
Verify the result with `--benchmark roboinf --self-test --benchmark-manifest
/path/to/new/runtime/benchmark-manifest.json --output-dir /path/to/new/check`.

For policy evaluation use `--benchmark roboinf --suite test` or `--suite test_random`, a Kit policy
config, and the checkpoint used by that policy. Set `ROBOSCOPE_ROBOINF_METHOD`
to its identifier and `ROBOSCOPE_ROBOINF_JUDGE` to an authorized executable.
The latter receives one argument (a native prediction JSONL path), and writes
one JSON object on stdout containing `case_id`, `domain`, `task_success`,
`instruction_success`, and `prediction_sha256` binding its decision to the
exact executed actions. Send diagnostics to stderr. The public source does
not distribute the held-out evaluator. Without it a scored run is unavailable.

Each task uses the pinned native horizon and 250 passive steps, even on early
success. Reset supplies the instruction at the first observation. The worker
transmits three native RGB views and the dual-arm joint/gripper state; the
opaque transport episode identifier has no reference information. The native
loop executes at most 20 actions per chunk. Official results report TR and JR
separately, macro-averaged over tasks. Generic Kit success is TR.

[Policy examples](../../examples/roboinf/README.md) cover pi_05, OpenWAM,
GR00T and LingBot-VA. EEF models receive the measured 16D state including
both grippers. Use separate output directories and a judge supporting the
requested domain; clear and randomized cases share IDs, not decisions.

For author-side validation, run `compare_clear.py --cases test-cases.jsonl
--baseline reference.jsonl --candidate candidate.jsonl`. Use judgments of
the reference and candidate policy rollouts, never demonstration labels.
Both exports must contain every expected case once and agree on both TR and JR
for each case. Use `--domain test_random` to compare randomized predictions'
judgments. This compares outcomes; it does not verify model identity
or replace simulation, executed-action validation or grader receipts.

### Clear test correctness baseline

The GR00T N1.7 C2C 30K SFT checkpoint (training batch size 64) was
evaluated with `--eval-mode strict --replicas 8 --concurrency 24` at Kit
commit `1dfce0b`. The comparison uses the historical native run, not a
new estimate of native repeatability. Counts in the last two columns
were identical on the native and Kit paths.

| Task | Cases | TR successes | JR successes |
| --- | ---: | ---: | ---: |
| open_laptop | 117 | 10 | 9 |
| place_bottle | 104 | 28 | 28 |
| place_fan | 104 | 0 | 0 |
| place_via_divider | 114 | 44 | 16 |
| press_buttons | 108 | 15 | 0 |
| shake_bottle | 104 | 31 | 0 |
| stack_blocks | 108 | 2 | 1 |
| turn_switch | 102 | 45 | 0 |
| Total | 861 | 175 | 54 |

All 861 paired outcomes matched for both metrics. The official eight-task
macro averages were TR 20.4666% and JR 6.1970% on both paths. This comparison
does not establish throughput, other profiles, or full benchmark acceptance.
