"""Capture four real LIBERO-Pro initial observations for batch correctness checks."""

import argparse
import json
from pathlib import Path

import numpy as np
from libero.libero import benchmark
from roboscope_kit.runtime.benchmarks.libero.worker import (
    LIBERO_DUMMY_ACTION,
    build_observation,
    create_task_runtime,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--suite', default='libero_goal_object')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
suite = benchmark.get_benchmark_dict()[args.suite]()
rows = []
for task in range(4):
    runtime = create_task_runtime(suite, task, 7, 'bddl')
    try:
        runtime.env.reset()
        observation = runtime.env.set_init_state(runtime.initial_states[0])
        for _ in range(10):
            observation, _, done, _ = runtime.env.step(LIBERO_DUMMY_ACTION)
            if done or runtime.env.env.done:
                raise RuntimeError(f'task {task} terminated during settling')
        rows.append(build_observation(f'batch-check-{task}', 0, observation, runtime.task_description))
    finally:
        runtime.env.close()
with args.output.open('xb') as stream:
    np.savez_compressed(stream,
                        primary=np.stack([r.features['images.primary'] for r in rows]),
                        wrist=np.stack([r.features['images.wrist'] for r in rows]),
                        proprioception=np.stack([r.features['state.proprioception'] for r in rows]),
                        instructions=np.asarray([r.features['language.instruction'] for r in rows]))
print(json.dumps({'output': str(args.output), 'observations': len(rows), 'tasks': list(range(4))}))
