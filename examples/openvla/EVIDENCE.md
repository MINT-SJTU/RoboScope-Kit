# Historical OpenVLA evaluation evidence

These records describe completed historical runs, not validation of this PR head.
Kit used source `b432786` and package `0.8.0b33`. Native used OpenVLA
`c8f03f48af692657d3060c19588038c7220e9af9` with the preserved instrumentation.
No experiment was rerun for this documentation update. Paired divergence analysis,
repeatability runs, fresh-head GPU validation and passing fork CI remain outstanding.

## Runtime and hardware

Native used CPython 3.10.13; Kit used CPython 3.11.15. Both recorded LIBERO 0.1.0
at `8f1084e3132a39270c3a13ebe37270a43ece2a01`, robosuite 1.4.1,
MuJoCo 3.2.3, NumPy 1.24.4, torch 2.2.0, Transformers 4.40.1,
TensorFlow 2.15.0 and FlashAttention 2.5.5.
Each run used one GPU reported as NVIDIA GeForce RTX 4090 with 49,140 MiB.
Replica count is the number of model processes, not GPU count.
The server inspected on 2026-11-11 reports Intel Xeon Platinum 8470Q;
this is a current observation, not verified historical CPU identity for every run.
Native manifests retain CPU affinity and a cgroup quota of 20 CPU equivalents.
Historical CPU model provenance remains incomplete.

| Run and records | Path / concurrency/replicas | Recorded driver |
| --- | --- | --- |
| [openvla-kit-200-fixed2g2-seed7-20261110](evidence/openvla-kit-200-fixed2g2-seed7-20261110/) | 2/2 | 580.105.08 |
| [openvla-kit-200-fixed4-seed7-20261109](evidence/openvla-kit-200-fixed4-seed7-20261109/) | 4/1 | 595.71.05 |
| [openvla-kit-200-fixed4g2-seed7-20261110](evidence/openvla-kit-200-fixed4g2-seed7-20261110/) | 4/2 | 580.105.08 |
| [openvla-kit-200-serial-seed7-20261109](evidence/openvla-kit-200-serial-seed7-20261109/) | 1/1 | 595.71.05 |
| [openvla-native-200-seed7-20261109](evidence/openvla-native-200-seed7-20261109/) | Native | 595.71.05 |

## Commands and result files

[Exact recorded commands](evidence/COMMANDS.md) cover all four suites for each run.
The committed launchers retain environment setup and absolute historical paths.
All complete arms used tasks 0–9, initial-state indices 0–4 and seed 7:
50 episodes per suite, 200 per arm. Video, full diagnostic trace, MPS and SDK
profiling were disabled. These different configurations are not repeat runs.

The evidence directories preserve the downloaded records without rewriting paths:

- Native: `libero_*/episodes.jsonl`, `summary.json`, `run-manifest.json`,
  `code-provenance.json`, `native-timings.json`, upstream logs and instrumentation;
  `source/` and `launch.py` retain the native runner. Native does not emit Kit's
  `result.json` / `metrics.json`; its summary and episode records are preserved instead.
- Kit: `libero_*/result.json`, `metrics.json`, `events.jsonl`, Worker and Policy logs,
  root `experiment.json`, command arrays, manifests, Policy configuration and launchers.
- `SHA256SUMS` identifies the archived files. This is a partial evidence archive:
  per-request timing streams, resource samples and queue files remain on the server
  under `/root/autodl-tmp/<run>/` and are not included here. Full step-by-step
  observations/actions were not recorded in these runs.

Repository-relative links become durable review links when this commit is pushed.
The server paths alone are not a public evidence archive. Autotune runs are outside
this five-arm archive and must not be represented as fully archived evidence.
