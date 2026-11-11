"""Launch the existing SDK, sequentially, without video or diagnostic traces."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import threading
import types
import sys
import inspect

BASE = Path('/root/autodl-tmp/openvla-kit-20261109')
OUT = Path('/root/autodl-tmp/openvla-kit-200-fixed4-seed7-20261109')
RUNTIME = BASE / 'runtime'
PYTHON = RUNTIME / 'env/bin/python'
SOURCE = BASE / 'source-final/examples/openvla'
MODELS = Path('/root/autodl-fs/Kit-Models/openvla')

def write(name, value):
    path = OUT / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)

def gpu_pids():
    return subprocess.check_output(['nvidia-smi', '--id=0', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()

def main():
    if (OUT / 'status.json').exists():
        raise RuntimeError('Refusing to overwrite an existing experiment')
    previous = Path('/root/autodl-tmp/openvla-kit-200-serial-seed7-20261109')
    prior = json.loads((previous / 'status.json').read_text())
    if prior['stage'] != 'complete' or len(prior['completed']) != 4:
        raise RuntimeError('Previous arm is not complete')
    for done in prior['completed']:
        if done['episodes'] != 50 or done['recovery']['worker_failures'] or done['recovery']['oom_events']:
            raise RuntimeError('Previous arm has invalid coverage or recovery')
    native = Path('/root/autodl-tmp/openvla-native-200-seed7-20261109')
    status = json.loads((native / 'status.json').read_text())
    if status['stage'] != 'complete':
        raise RuntimeError('Native must finish before Kit starts')
    for suite in ('spatial', 'object', 'goal', '10'):
        summary = json.loads((native / ('libero_' + suite) / 'summary.json').read_text())
        if not (summary['execution_valid'] and summary['complete_unique_coverage']
                and summary['completed_episodes'] == summary['unique_episodes'] == summary['expected_episodes'] == 50
                and not summary['errors'] and not summary['oom_errors']):
            raise RuntimeError('Invalid native coverage: ' + suite)
    if gpu_pids():
        raise RuntimeError('GPU still occupied before Kit startup')
    plan = json.loads((native / 'sample-plan.json').read_text())
    expected_samples = [{'suite': 'libero_' + suite, 'task_id': t, 'initial_state_index': e}
                        for suite in ('spatial', 'object', 'goal', '10') for t in range(10) for e in range(5)]
    if plan['samples'] != expected_samples:
        raise RuntimeError('Native sample plan differs from Kit plan')
    plan.update(arm='kit', concurrency=4, replicas=1,
                order='SDK scheduling; terminal identities validated per suite')
    write('sample-plan.json', plan)
    (OUT / 'logs').mkdir()
    manifest = json.loads((RUNTIME / 'benchmark-manifest.json').read_text())
    for key, value in manifest['runtime'].items():
        manifest['runtime'][key] = os.path.relpath(RUNTIME / value, OUT)
    manifest['defaults'].update(seed=7, sample_seed=7, save_videos=False, episodes=5)
    write('benchmark-manifest.json', manifest)
    write('roboscope-policy.json', {'schema_version': 2, 'command': [str(PYTHON), str(OUT / 'measured_policy.py')], 'cwd': str(SOURCE)})
    env = dict(os.environ)
    for key in ('OPENVLA_TRACE_ROOT', 'CUDA_MPS_PIPE_DIRECTORY', 'CUDA_MPS_LOG_DIRECTORY'):
        env.pop(key, None)
    env.update(OPENVLA_SEED='7', PYTHONHASHSEED='7', CUDA_VISIBLE_DEVICES='0', TF_FORCE_GPU_ALLOW_GROWTH='true', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', HF_HOME='/root/autodl-tmp/openvla-libero-quick-20261109/hf')
    write('experiment.json', {
        'arm': 'kit', 'profile': 'fixed4', 'global_seed': 7, 'environment_seed': 7,
        'episodes_per_suite': 50, 'total_episodes': 200, 'tasks': list(range(10)),
        'initial_states': list(range(5)), 'concurrency': 4, 'replicas': 1,
        'video': False, 'diagnostic_trace': False, 'mps': False, 'autotune': False,
        'source_commit': 'b432786', 'policy_sha256': hashlib.sha256((SOURCE / 'openvla_policy.py').read_bytes()).hexdigest(),
        'build': json.loads((RUNTIME / 'build-manifest.json').read_text()),
        'hardware': subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version,memory.total', '--format=csv'], text=True),
        'timing_note': 'SDK phases plus buffered numeric request timings, allocator peaks and native resource sampler with SDK stage labels. Service boundaries remain distinct from native. No images, actions, video or page warming.', 'measurement_sha256': {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest() for name in ('launch.py', 'measured_policy.py', 'resource_sampler.py')}
    })
    state = {'stage': 'starting', 'completed': [], 'started_unix_s': time.time(), 'pid': os.getpid()}
    write('status.json', state)
    try:
        for suite in ('spatial', 'object', 'goal', '10'):
            state.update(stage='waiting-for-idle-gpu', suite=suite, updated_unix_s=time.time())
            write('status.json', state)
            while gpu_pids():
                time.sleep(30)
            run = OUT / ('libero_' + suite)
            command = [str(PYTHON), '-m', 'roboscope_kit.runtime.benchmark', '--benchmark', 'libero', '--suite', 'libero_' + suite, '--checkpoint-source', str(MODELS / suite), '--benchmark-manifest', str(OUT / 'benchmark-manifest.json'), '--policy-config', str(OUT / 'roboscope-policy.json'), '--episodes', '5', '--concurrency', '4', '--replicas', '1', '--output-dir', str(run)]
            write(suite + '-command.json', command)
            state.update(stage='evaluate', command=command, updated_unix_s=time.time())
            write('status.json', state)
            measurements = OUT / 'measurements' / ('libero_' + suite)
            measurements.mkdir(parents=True)
            env['OPENVLA_MEASUREMENTS'] = str(measurements)
            env['OPENVLA_SDK_RUN'] = str(run)
            env['OPENVLA_RUN_MODE'] = 'fixed4'
            sys.path.insert(0, str(OUT))
            import resource_sampler as native_runner
            # Preserve the native sampler's fields, query and interval. Only its
            # phase lookup changes to the SDK's actual metrics heartbeat.
            sampler_source = inspect.getsource(native_runner.sample_resources)
            sampler_source = sampler_source.replace('args.output_dir / "phase.json"', 'args.sdk_run / "metrics.json"').replace('["phase"]\n', '["active_stage"]\n')
            (measurements / 'resource-sampler.py').write_text(sampler_source)
            namespace = dict(vars(native_runner))
            exec(compile(sampler_source, 'resource-sampler.py', 'exec'), namespace)
            sampler_args = types.SimpleNamespace(output_dir=measurements, sdk_run=run, gpu='0', sample_interval=0.5)
            stop = threading.Event()
            samples = []
            start = time.monotonic()
            with (OUT / 'logs' / (suite + '.log')).open('x') as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                sampler = threading.Thread(target=namespace['sample_resources'], args=(sampler_args, child, stop, samples), daemon=True)
                sampler.start()
                try:
                    returncode = child.wait()
                finally:
                    stop.set()
                    sampler.join(timeout=10)
                if returncode:
                    raise subprocess.CalledProcessError(returncode, command)
            elapsed = time.monotonic() - start
            rows = []
            for path in (run / 'queue/completed').glob('item-*.json'):
                rows.extend(json.loads(path.read_text())['records'])
            identities = [(r['metadata']['task_id'], r['metadata']['episode_index']) for r in rows]
            expected = {(t, e) for t in range(10) for e in range(5)}
            if len(rows) != 50 or set(identities) != expected:
                raise RuntimeError('Invalid terminal coverage for ' + suite)
            result = json.loads((run / 'result.json').read_text())
            import numpy as np
            all_requests = [json.loads(line) for path in measurements.glob('replica-*/requests.jsonl') for line in path.read_text().splitlines()]
            requests = [r for r in all_requests if r['phase'] == 'formal']
            probes = [r for r in all_requests if r['phase'] == 'probe']
            write(str(measurements.relative_to(OUT) / 'probe-summary.json'), {'flushed_requests': len(probes), 'errors': sum(not r['success'] for r in probes), 'note': 'Buffered requests from forcibly terminated probes may be incomplete; candidate terminal records and selection.json are authoritative for coverage.'})
            def distribution(values):
                return None if not values else {'count': len(values), 'mean': float(np.mean(values)), 'p50': float(np.percentile(values, 50)), 'p95': float(np.percentile(values, 95)), 'p99': float(np.percentile(values, 99)), 'max': max(values)}
            latency = {'requests': len(requests), 'errors': sum(not r['success'] for r in requests), 'boundary': 'See per-replica model.json; native service_ms has a different boundary', 'metrics_ms': {key: {group: distribution([r[key] for r in requests if r['success'] and key in r and (group == 'all' or r['first_request'] == (group == 'first'))]) for group in ('all', 'first', 'subsequent')} for key in ('service_ms', 'image_preprocess_ms', 'predict_action_ms', 'action_postprocess_ms')}}
            write(str(measurements.relative_to(OUT) / 'latency.json'), latency)
            write(str(measurements.relative_to(OUT) / 'sample-summary.json'), {'samples': len(samples), 'errors': sum(bool(r.get('error')) for r in samples), 'sample_gap_s': distribution([b['monotonic_s'] - a['monotonic_s'] for a, b in zip(samples, samples[1:])])})
            from collections import Counter
            request_counts = Counter(r['episode_id'] for r in requests)
            if any(request_counts[r['record_id']] != r['steps'] for r in rows):
                raise RuntimeError('Per-episode formal request/step mismatch')
            if latency['errors'] or len(requests) != sum(r['steps'] for r in rows):
                raise RuntimeError('Request errors or request/step coverage mismatch; retain evidence and investigate')
            recovery = result.get('recovery', {})
            if recovery.get('worker_failures') or recovery.get('oom_events'):
                raise RuntimeError('Formal recovery requires review before advancing')
            selection = run / 'autotune/selection.json'
            if selection.exists():
                tuning = json.loads(selection.read_text())
                if tuning['status'] != 'complete':
                    raise RuntimeError('Autotune did not complete')
                write(str(measurements.relative_to(OUT) / 'selection.json'), tuning)
            state['completed'].append({'suite': suite, 'episodes': len(rows), 'successes': sum(bool(r['success']) for r in rows), 'steps': sum(r['steps'] for r in rows), 'command_wall_s': elapsed, 'phase_timings': result.get('phase_timings'), 'recovery': result.get('recovery'), 'run': str(run)})
            write('status.json', state)
        state.update(stage='complete', updated_unix_s=time.time())
        write('status.json', state)
    except BaseException as error:
        state.update(stage='failed', error=repr(error), updated_unix_s=time.time())
        write('status.json', state)
        raise

if __name__ == '__main__':
    main()
