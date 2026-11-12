"""Experiment-only timing observer; delegates inference to the committed Policy."""
import inspect
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, '/root/autodl-tmp/openvla-kit-20261109/source-final/examples/openvla')
import openvla_policy as implementation
from roboscope_kit.server import run

class MeasuredPolicy(implementation.OpenVLAPolicy):
    def __init__(self):
        super().__init__()
        self.output = Path(os.environ['OPENVLA_MEASUREMENTS']) / ('replica-' + os.environ['ROBOSCOPE_REPLICA_ID'] + '-pid-' + str(os.getpid()))
        self.labels = {}
        self.generation = 0
        self.output.mkdir(parents=True, exist_ok=True)
        self.pending = []
        self.parts = {}
        self.seen = set()

    def write(self, name, value):
        target = self.output / name
        temp = target.with_suffix('.tmp')
        temp.write_text(json.dumps(value, indent=2) + '\n')
        temp.replace(target)

    def flush(self):
        if self.pending:
            with (self.output / 'requests.jsonl').open('a') as stream:
                for row in self.pending:
                    stream.write(json.dumps(row) + '\n')
            self.pending.clear()

    def timed(self, name, function):
        def observed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                self.parts[name] = (time.perf_counter() - start) * 1000
        return observed

    def load(self, context):
        import torch
        started = time.perf_counter()
        result = super().load(context)
        self.write('model.json', {
            'policy_load_s': time.perf_counter() - started,
            'checkpoint': str(context.checkpoint_path), 'suite': self.suite,
            'code_revision': implementation.CODE_REVISION,
            'dtype': str(next(self.model.parameters()).dtype),
            'loading_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'loading_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'model_class_file': inspect.getfile(type(self.model)),
            'boundary': 'PyTorch allocator only; excludes TensorFlow and EGL allocations',
            'service_boundary': 'Policy.act including JPEG/resize/crop, prompt, processor, H2D, prediction, CPU action, gripper conversion; excludes socket/queue and measurement writes',
            'native_service_boundary_difference': 'Native get_action excludes JPEG/resize and environment action conversion; service_ms is not directly comparable.',
            'synchronization': 'No added CUDA synchronization; predict_action returns a CPU NumPy action',
        })
        torch.cuda.reset_peak_memory_stats()
        implementation.prepare_image = self.timed('image_preprocess_ms', implementation.prepare_image)
        implementation.environment_action = self.timed('action_postprocess_ms', implementation.environment_action)
        self.model.predict_action = self.timed('predict_action_ms', self.model.predict_action)
        return result

    def reset(self, context):
        self.generation += 1
        selection = Path(os.environ['OPENVLA_SDK_RUN']) / 'autotune/selection.json'
        label = {'phase': 'formal', 'candidate': None,
                 'replica_id': os.environ['ROBOSCOPE_REPLICA_ID'],
                 'pid': os.getpid(), 'generation': self.generation}
        if os.environ['OPENVLA_RUN_MODE'] == 'autotune':
            state = json.loads(selection.read_text())
            if state['status'] == 'running':
                candidates = [c for c in state['candidates'] if c['status'] == 'running']
                if len(candidates) != 1:
                    raise RuntimeError('Ambiguous autotune candidate at reset')
                c = candidates[0]
                label.update(phase='probe', candidate={'k': c['k'], 'g': c['g']})
            elif state['status'] != 'complete':
                raise RuntimeError('Invalid autotune state at reset')
        self.labels[context.episode_id] = label
        self.seen.discard(context.episode_id)
        with (self.output / 'resets.jsonl').open('a') as stream:
            stream.write(json.dumps({'episode_id': context.episode_id, 'unix_s': time.time(), **label}) + '\n')
        return super().reset(context)

    def act(self, observation):
        self.parts = {}
        started = time.perf_counter()
        success = False
        error = None
        try:
            result = super().act(observation)
            success = True
            return result
        except BaseException as exc:
            error = type(exc).__name__ + ': ' + str(exc)
            raise
        finally:
            duration = (time.perf_counter() - started) * 1000
            self.pending.append({'episode_id': observation.episode_id, 'step': observation.step,
                                 'first_request': observation.episode_id not in self.seen,
                                 'unix_s': time.time(), 'service_ms': duration,
                                 'success': success, 'error': error, **self.parts, **self.labels[observation.episode_id]})
            self.seen.add(observation.episode_id)
            if not success:
                self.flush()

    def end_episode(self, episode_id):
        super().end_episode(episode_id)
        self.flush()
        self.peaks()

    def peaks(self):
        import torch
        self.write('allocator.json', {'scope': 'Since model load; pooled Autotune process includes probe and formal phases',
                                     'evaluation_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                     'evaluation_peak_reserved_bytes': torch.cuda.max_memory_reserved()})

    def unload(self):
        self.flush()
        self.peaks()
        super().unload()

if __name__ == '__main__':
    run(MeasuredPolicy())
