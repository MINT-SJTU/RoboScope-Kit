"""Run 200 paired-input native episodes using the committed official evaluator observer."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/root/autodl-tmp/openvla-native-200-seed7-20261109')
RUNTIME = Path('/root/autodl-tmp/openvla-native-runtime')

def save(name, data):
    path = ROOT / name
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)

def main():
    if (ROOT / 'status.json').exists():
        raise RuntimeError('Refusing to overwrite an existing experiment')
    (ROOT / 'logs').mkdir()
    save('sample-plan.json', {
        'arm': 'native', 'profile': 'matched-serial', 'global_seed': 7, 'environment_seed': 7,
        'episodes_per_task': 5, 'episodes_per_suite': 50, 'total_episodes': 200,
        'selection': 'Fixed initial-state indices 0..4 for all ten tasks; not a random sample.',
        'order': 'spatial, object, goal, 10; task 0..9; initial state 0..4 within task',
        'video': False, 'diagnostic_trace': False,
        'samples': [{'suite': 'libero_' + suite, 'task_id': task, 'initial_state_index': episode}
                    for suite in ('spatial', 'object', 'goal', '10') for task in range(10) for episode in range(5)]
    })
    env = dict(os.environ)
    for key in ('CUDA_MPS_PIPE_DIRECTORY', 'CUDA_MPS_LOG_DIRECTORY', 'OPENVLA_TRACE_ROOT'):
        env.pop(key, None)
    env.update(HF_HOME='/root/autodl-tmp/openvla-libero-quick-20261109/hf', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    state = {'stage': 'starting', 'pid': os.getpid(), 'completed': [], 'started_unix_s': time.time()}
    try:
        for suite in ('spatial', 'object', 'goal', '10'):
            state.update(stage='waiting-for-idle-gpu', suite=suite, updated_unix_s=time.time())
            save('status.json', state)
            while subprocess.check_output(['nvidia-smi', '--id=0', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                time.sleep(30)
            output = ROOT / ('libero_' + suite)
            command = [str(RUNTIME / 'env/bin/python'), str(ROOT / 'source/run.py'),
                       '--runtime', str(RUNTIME), '--checkpoint', '/root/autodl-fs/Kit-Models/openvla/' + suite,
                       '--checkpoint-inventory', '/root/autodl-tmp/openvla-libero-quick-20261109/inventories/' + suite + '.json',
                       '--suite', 'libero_' + suite, '--output-dir', str(output),
                       '--episodes-per-task', '5', '--profile-name', 'matched-serial',
                       '--seed', '7', '--env-seed', '7', '--video', 'off', '--repeat-id', 'native-200-seed7',
                       '--gpu', '0', '--cpu-threads', '1', '--sample-interval', '0.5']
            state.update(stage='evaluate', command=command, updated_unix_s=time.time())
            save('status.json', state)
            with (ROOT / 'logs' / (suite + '.log')).open('x') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            summary = json.loads((output / 'summary.json').read_text())
            if not summary['execution_valid'] or summary['expected_episodes'] != 50 or summary['completed_episodes'] != 50:
                raise RuntimeError('Invalid native coverage/execution: ' + suite)
            state['completed'].append(summary)
            save('status.json', state)
        state.update(stage='complete', updated_unix_s=time.time())
        save('status.json', state)
    except BaseException as error:
        state.update(stage='failed', error=repr(error), updated_unix_s=time.time())
        save('status.json', state)
        raise

if __name__ == '__main__':
    main()
