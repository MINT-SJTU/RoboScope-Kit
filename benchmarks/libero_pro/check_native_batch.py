"""Compare real StarVLA single and batched predictions on frozen simulator observations."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from roboscope_kit import Observation, SessionContext
from roboscope_kit.benchmark_specs import BenchmarkName, get_benchmark_spec

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--observations', type=Path, required=True)
parser.add_argument('--checkpoint-source', type=Path, required=True)
parser.add_argument('--starvla-root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--rtol', type=float, default=0.01)
parser.add_argument('--atol', type=float, default=0.005)
args = parser.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'examples/starvla'))
sys.path.insert(0, str(args.starvla_root.resolve()))
os.chdir(args.starvla_root.resolve())
os.environ['ROBOSCOPE_STARVLA_MAX_BATCH_SIZE'] = '4'
from starvla_policy import StarVLAPolicy  # noqa: E402

with np.load(args.observations, allow_pickle=False) as source:
    observations = tuple(Observation(f'batch-check-{i}', 0, {
        'images.primary': source['primary'][i], 'images.wrist': source['wrist'][i],
        'state.proprioception': source['proprioception'][i],
        'language.instruction': str(source['instructions'][i]),
    }) for i in range(4))
policy = StarVLAPolicy()
results = []
try:
    capabilities = policy.load(SessionContext('batch-check', str(args.checkpoint_source.resolve()),
                               get_benchmark_spec(BenchmarkName.LIBERO_PRO, 'libero_goal_object')))
    if capabilities.max_batch_size != 4:
        raise RuntimeError('Policy did not enable batch cap 4')
    single = [policy.act(o).actions for o in observations]
    for size in (2, 4):
        actual = []
        for start in range(0, 4, size):
            actual.extend(c.actions for c in policy.act_batch(observations[start:start + size]))
        max_abs = max(float(np.max(np.abs(x - y))) for x, y in zip(single, actual, strict=True))
        gripper_equal = all(np.array_equal(x[:, 6], y[:, 6]) for x, y in zip(single, actual, strict=True))
        close = all(np.allclose(x, y, rtol=args.rtol, atol=args.atol)
                    for x, y in zip(single, actual, strict=True))
        results.append({'batch_size': size, 'max_absolute_action_difference': max_abs,
                        'gripper_exact': gripper_equal, 'allclose': close})
    reverse = [c.actions for c in policy.act_batch(tuple(reversed(observations)))][::-1]
    results.append({'batch_size': 4, 'order': 'reversed',
                    'max_absolute_action_difference': max(float(np.max(np.abs(x - y)))
                                                          for x, y in zip(single, reverse, strict=True)),
                    'gripper_exact': all(np.array_equal(x[:, 6], y[:, 6])
                                         for x, y in zip(single, reverse, strict=True)),
                    'allclose': all(np.allclose(x, y, rtol=args.rtol, atol=args.atol)
                                    for x, y in zip(single, reverse, strict=True))})
finally:
    policy.unload()
report = {'rtol': args.rtol, 'atol': args.atol, 'comparisons': results}
with args.output.open('x') as stream:
    json.dump(report, stream, indent=2)
print(json.dumps(report))
if not all(x['allclose'] and x['gripper_exact'] for x in results):
    raise SystemExit('single vs batched action comparison failed; do not start full evaluation')
