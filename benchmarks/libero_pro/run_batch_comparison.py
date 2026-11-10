"""Run sequential fixed-K/g native batch experiments with CUDA MPS enabled."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

from roboscope_kit.policy_config import load_policy_runtime_config
from roboscope_kit.runtime.benchmarks.libero_pro.adapter import LIBERO_PRO_SUITES
from run_mps_comparison import require_idle_gpu, validate_arm, write_json


def summarize_batches(directory: Path, cap: int, expected: int) -> dict:
    rows = [json.loads(line) for path in sorted(directory.glob('batch-*.jsonl'))
            for line in path.read_text().splitlines()]
    if not rows or any(not x['ok'] or not 1 <= x['size'] <= cap or x['policy_wall_s'] < 0 for x in rows):
        raise RuntimeError(f'invalid/missing batch metrics in {directory}')
    if sum(x['size'] for x in rows) != expected:
        raise RuntimeError(f'batch metrics do not account for all Policy calls in {directory}')
    if any(not isinstance(x.get('prediction_wall_s'), (int, float))
           or not math.isfinite(x['prediction_wall_s']) or x['prediction_wall_s'] < 0 for x in rows):
        raise RuntimeError(f'missing/invalid model prediction timings in {directory}')
    times = defaultdict(list)
    predictions = defaultdict(list)
    for row in rows:
        times[row['size']].append(1000 * row['policy_wall_s'])
        predictions[row['size']].append(1000 * row['prediction_wall_s'])
    return {
        'batches': len(rows), 'requests': expected,
        'actual_batch_sizes': {str(k): v for k, v in sorted(Counter(x['size'] for x in rows).items())},
        'mean_batch_size': expected / len(rows),
        'policy_wall_ms_by_batch_size': {
            str(k): {'count': len(v), 'mean': statistics.mean(v), 'median': statistics.median(v),
                     'p95': sorted(v)[math.ceil(.95 * len(v)) - 1]}
            for k, v in sorted(times.items())
        },
        'prediction_wall_ms_by_batch_size': {
            str(k): {'count': len(v), 'mean': statistics.mean(v), 'median': statistics.median(v),
                     'p95': sorted(v)[math.ceil(.95 * len(v)) - 1]}
            for k, v in sorted(predictions.items())
        },
        'note': 'Prediction includes framework input processing, GPU work and output synchronization.',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-manifest', type=Path, required=True)
    parser.add_argument('--policy-config', type=Path, required=True)
    parser.add_argument('--checkpoint-source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--suite', choices=sorted(LIBERO_PRO_SUITES), default='libero_goal_object')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--concurrency', type=int, default=12)
    parser.add_argument('--replicas', type=int, default=3)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.episodes <= 50 or not 1 <= args.replicas <= args.concurrency:
        parser.error('require 1..50 episodes and 1 <= replicas <= concurrency')
    if (not args.batch_sizes or len(set(args.batch_sizes)) != len(args.batch_sizes)
            or min(args.batch_sizes) < 1):
        parser.error('batch sizes must be distinct positive integers')
    if not args.gpu or ',' in args.gpu or args.gpu in {'-1', 'all'}:
        parser.error('select one GPU index or UUID')
    for name in ('benchmark_manifest', 'policy_config', 'checkpoint_source'):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.output_dir = args.output_dir.resolve()
    policy = load_policy_runtime_config(args.policy_config)
    manifest = json.loads(args.benchmark_manifest.read_text())
    if args.suite not in manifest['supported_suites'].get('libero_pro', []):
        parser.error('manifest does not support the selected suite')
    manifest['runtime'] = {k: str((args.benchmark_manifest.parent / v).resolve(strict=True))
                           for k, v in manifest['runtime'].items()}
    manifest['defaults'].update(policy_encoding='msgpack', persistent_policy_connection=False,
                                chunk_camera_rendering=True, save_videos=False)
    policy_config = policy.summary()
    policy_config.pop('path')
    plan = {'kit_version': version('roboscope-kit'), 'suite': args.suite,
            'episodes_per_task': args.episodes, 'k': args.concurrency, 'g': args.replicas,
            'gpu': args.gpu, 'cuda_mps': True, 'batch_caps': args.batch_sizes,
            'checkpoint_source': str(args.checkpoint_source), 'policy_source': str(args.policy_config),
            'manifest_source': str(args.benchmark_manifest), 'manifest': manifest,
            'policy': policy_config, 'commands': {}}
    for cap in args.batch_sizes:
        folder = args.output_dir / f'batch-{cap}'
        common = [sys.executable, '-m', 'roboscope_kit.runtime.benchmark', '--benchmark', 'libero_pro',
                  '--suite', args.suite, '--checkpoint-source', str(args.checkpoint_source),
                  '--benchmark-manifest', str(args.output_dir / 'benchmark-manifest.json'),
                  '--policy-config', str(args.output_dir / 'policy.json'), '--cuda-mps']
        plan['commands'][str(cap)] = {
            'warmup': common + ['--quick-test', '--output-dir', str(folder / 'warmup')],
            'formal': common + ['--episodes', str(args.episodes), '--concurrency', str(args.concurrency),
                                '--replicas', str(args.replicas), '--output-dir', str(folder / 'run')],
        }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    require_idle_gpu(args.gpu)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / 'plan.json', plan)
    write_json(args.output_dir / 'benchmark-manifest.json', manifest)
    write_json(args.output_dir / 'policy.json', policy_config)
    env = {k: v for k, v in os.environ.items() if not k.startswith('CUDA_MPS_')
           and k not in {'ROBOSCOPE_TIMING_DIR', 'ROBOSCOPE_PROFILE'}}
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED='1')
    results = {}
    record_maps = {}
    for cap in args.batch_sizes:
        folder = args.output_dir / f'batch-{cap}'
        folder.mkdir()
        env['ROBOSCOPE_STARVLA_MAX_BATCH_SIZE'] = str(cap)
        walls = {}
        for phase in ('warmup', 'formal'):
            require_idle_gpu(args.gpu)
            env['ROBOSCOPE_BATCH_METRICS_DIR'] = str(folder / f'batches-{phase}')
            print(f'Starting batch cap {cap} {phase}', flush=True)
            started = time.perf_counter()
            with (folder / f'{phase}.log').open('x') as log:
                status = subprocess.run(plan['commands'][str(cap)][phase], env=env,
                                        stdout=log, stderr=subprocess.STDOUT, check=False)
            walls[phase] = {'wall_s': time.perf_counter() - started, 'exit_code': status.returncode}
            write_json(folder / f'{phase}-status.json', walls[phase])
            if status.returncode:
                raise RuntimeError(f'batch-{cap} {phase} failed; see {folder / (phase + ".log")}')
            require_idle_gpu(args.gpu)
        summary, records = validate_arm(folder, episodes=args.episodes, concurrency=args.concurrency,
                                        replicas=args.replicas, mps=True)
        result = json.loads((folder / 'run/result.json').read_text())
        if any(v['max_batch_size'] != cap for v in result['inference']['replicas'].values()):
            raise RuntimeError(f'batch-{cap} Policy capabilities do not match the requested cap')
        expected = sum(math.ceil(r['steps'] / 8) for r in records.values())
        if summary['recovery']['worker_failures'] or summary['recovery']['oom_events']:
            raise RuntimeError(f'batch-{cap} had recovery; inspect logs before comparing request counts')
        summary['batch_metrics'] = summarize_batches(folder / 'batches-formal', cap, expected)
        summary['CLI_wall_s'] = walls['formal']['wall_s']
        summary['task_results'] = result['official_result']['tasks']
        record_maps[str(cap)] = records
        write_json(folder / 'summary.json', summary)
        results[str(cap)] = summary
    paired = {}
    if '1' in record_maps:
        baseline = record_maps['1']
        for cap, records in record_maps.items():
            if cap != '1':
                paired[cap] = {
                    'success_differences': sum(
                        baseline[k]['success'] != records[k]['success'] for k in baseline
                    ),
                    'step_differences': sum(baseline[k]['steps'] != records[k]['steps'] for k in baseline),
                    'episode_throughput_ratio': (
                        results[cap]['episodes_per_s'] / results['1']['episodes_per_s']
                    ),
                    'step_throughput_ratio': results[cap]['steps_per_s'] / results['1']['steps_per_s'],
                }
    write_json(args.output_dir / 'comparison.json', {'arms': results, 'paired_vs_batch_1': paired,
        'note': 'One order with batch logging; review paired episodes and repeat before conclusions.'})
    print(f'Completed: {args.output_dir / "comparison.json"}', flush=True)


if __name__ == '__main__':
    main()
