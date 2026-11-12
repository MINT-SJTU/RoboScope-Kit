"""Observe the pinned official evaluator without replacing its rollout or model code."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import sys
import time
import traceback
import types
from pathlib import Path

EVALUATOR_SHA256 = "369d9fc3d59067998ebfadafbd516872c7ff6f9f06a0a48cda8b6ca48051de77"
CODE_REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def array_identity(value) -> dict:
    import numpy as np

    array = np.ascontiguousarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


class ObservedEnvironment:
    def __init__(self, env, observer):
        self.env = env
        self.observer = observer
        self.reset_count = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        self.reset_count += 1
        return self.env.reset()

    def snapshot_scene(self, name, observation):
        if self.observer.trace is None:
            return
        import numpy as np
        from PIL import Image
        folder = self.observer.trace.root / self.observer.current["episode_id"]
        folder.mkdir(parents=True, exist_ok=True)
        sim = self.env.sim
        arrays = {"body_pos": np.asarray(sim.model.body_pos).copy(), "body_quat": np.asarray(sim.model.body_quat).copy()}
        for key in ["qpos", "qvel", "ctrl", "qacc_warmstart", "act"]:
            if hasattr(sim.data, key): arrays[key] = np.asarray(getattr(sim.data, key)).copy()
        np.savez_compressed(folder / (name + ".npz"), **arrays)
        Image.fromarray(observation["agentview_image"][::-1, ::-1]).save(folder / (name + ".png"))

    def set_init_state(self, state):
        self.observer.current["initial_state"] = array_identity(state)
        observation = self.env.set_init_state(state)
        self.observer.current["initial_primary_image"] = array_identity(observation["agentview_image"])
        self.observer.current["reset_count_in_task_runtime"] = self.reset_count
        self.snapshot_scene("initial", observation)
        return observation

    def step(self, action):
        current = self.observer.current
        settling = current["settling_steps"] < 10
        if not settling and "first_environment_action" not in current:
            current["first_environment_action"] = list(action)
        result = self.env.step(action)
        current["settling_steps" if settling else "policy_steps"] += 1
        if settling and current["settling_steps"] == 10:
            self.snapshot_scene("after-settling", result[0])
        if settling and result[2]:
            current["termination_during_settling"] = True
        if not settling:
            current["success"] = bool(result[2])
        current["last_step_monotonic_s"] = time.monotonic()
        if not settling and self.observer.trace is not None:
            self.observer.trace.executed(action, result[1], result[2])
        return result


class ObservedProcessor:
    def __init__(self, processor, observer):
        self.processor, self.observer = processor, observer

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def __call__(self, prompt, image, *args, **kwargs):
        inputs = self.processor(prompt, image, *args, **kwargs)
        if self.observer.trace is not None:
            self.observer.trace.processor(prompt, image, inputs)
        current = self.observer.current
        if "first_processor_inputs" not in current:
            current["first_prompt"] = prompt
            current["first_cropped_image"] = array_identity(image)
            current["first_processor_inputs"] = {
                name: array_identity(value.detach().cpu().numpy()) for name, value in inputs.items()
            }
        return inputs


class Observer:
    def __init__(self, args):
        self.args = args
        self.output = args.output_dir
        self.current = None
        self.records = []
        self.pending_requests = []
        self.timings = {}
        self.formal_started = None
        self.task = None
        self.torch = None
        self.trace = None
        if args.diagnostic_trace:
            from trace_artifacts import DiagnosticTrace

            self.trace = DiagnosticTrace(self.output / "trace")
        self.phase("imports")

    def phase(self, name):
        write_json(
            self.output / "phase.json",
            {
                "phase": name,
                "unix_s": time.time(),
                "monotonic_s": time.monotonic(),
            },
        )

    def task_ids(self, count):
        selected = self.args.task_id if self.args.task_id is not None else list(range(count))
        if len(set(selected)) != len(selected) or any(i < 0 or i >= count for i in selected):
            raise ValueError(f"Invalid task identities: {selected}")
        return selected

    def formal_start(self):
        self.loading_peak = self.torch.cuda.max_memory_allocated()
        self.loading_reserved_peak = self.torch.cuda.max_memory_reserved()
        self.torch.cuda.reset_peak_memory_stats()
        self.phase("evaluation")
        self.formal_started = time.monotonic()

    def formal_finish(self):
        self.timings["evaluation_s"] = time.monotonic() - self.formal_started
        self.torch_peak = {
            "loading_peak_allocated_bytes": self.loading_peak,
            "loading_peak_reserved_bytes": self.loading_reserved_peak,
            "evaluation_peak_allocated_bytes": self.torch.cuda.max_memory_allocated(),
            "evaluation_peak_reserved_bytes": self.torch.cuda.max_memory_reserved(),
            "scope": "PyTorch allocator on cuda:0; excludes TensorFlow and EGL allocations",
        }
        self.phase("cleanup")

    def task_env(self, task_id, task, env, description):
        from libero.libero import get_libero_path

        if self.args.env_seed is not None:
            env.seed(self.args.env_seed)
        identity = task._asdict()
        files = {
            "bddl": Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file,
            "initial_states": Path(get_libero_path("init_states"))
            / task.problem_folder
            / task.init_states_file,
        }
        self.task = {
            "task_id": task_id,
            "task": identity,
            "instruction": description,
            "input_files": {
                name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for name, path in files.items()
            },
        }
        return ObservedEnvironment(env, self)

    def episode_start(self, task_id, episode_index, description):
        self.current = {
            **self.task,
            "suite": self.args.suite,
            "initial_state_index": episode_index,
            "episode_id": f"task-{task_id:04d}-episode-{episode_index:04d}",
            "global_seed": self.args.seed,
            "environment_seed": self.args.env_seed if self.args.env_seed is not None else 0,
            "policy_steps": 0,
            "settling_steps": 0,
            "success": False,
            "errors": [],
            "started_monotonic_s": time.monotonic(),
            "started_unix_s": time.time(),
            "elapsed_boundary": "before env.reset through upstream video/log writes",
        }

    def error(self, error):
        details = {"class": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        if self.current is not None:
            self.current["errors"].append(details)
        with (self.output / "errors.jsonl").open("a") as file:
            file.write(
                json.dumps(
                    {"unix_s": time.time(), "episode_id": (self.current or {}).get("episode_id"), **details}
                )
                + "\n"
            )

    def episode_end(self, upstream_logged_success):
        current = self.current
        current["elapsed_s"] = time.monotonic() - current["started_monotonic_s"]
        current["upstream_logged_success"] = bool(upstream_logged_success)
        current["end_reason"] = (
            "error" if current["errors"] else "success" if current["success"] else "horizon"
        )
        current["completed"] = True
        with (self.output / "episodes.jsonl").open("a") as file:
            file.write(json.dumps(current, allow_nan=False) + "\n")
        with (self.output / "requests.jsonl").open("a") as file:
            file.writelines(json.dumps(row, allow_nan=False) + "\n" for row in self.pending_requests)
        self.pending_requests.clear()
        self.records.append(current)
        self.current = None

    def timed_loader(self, original, name, processor=False):
        def load(cfg):
            self.phase("policy_load")
            started = time.monotonic()
            loaded = original(cfg)
            self.timings[name] = time.monotonic() - started
            self.torch = sys.modules["torch"]
            if processor:
                loaded = ObservedProcessor(loaded, self)
            return loaded

        return load

    def observe_action(self, original):
        def action(*args, **kwargs):
            current = self.current
            if "first_resized_image" not in current:
                current["first_resized_image"] = array_identity(args[2]["full_image"])
                current["unnorm_key"] = args[0].unnorm_key
            started = time.perf_counter()
            success = False
            try:
                result = original(*args, **kwargs)
                if self.trace is not None:
                    self.trace.model_action(result, args[0].unnorm_key)
                success = True
                return result
            finally:
                self.pending_requests.append(
                    {
                        "episode_id": current["episode_id"],
                        "policy_step": current["policy_steps"],
                        "service_ms": (time.perf_counter() - started) * 1000,
                        "success": success,
                    }
                )

        return action

    def observe_image(self, original):
        def image(observation, resize_size):
            result = original(observation, resize_size)
            if self.trace is not None:
                self.trace.observation(
                    self.current["episode_id"], self.current["policy_steps"], observation, result
                )
            return result

        return image


def instrument_source(source: str, save_video: bool) -> str:
    replacements = [
        ("range(cfg.num_trials_per_task)", "[int(i) for i in __import__('json').loads(__import__('os').environ['DIAGNOSTIC_SELECTION'])[str(task_id)]]"),
        ("@draccus.wrap()\n", ""),
        ("    # Start evaluation\n", "    _observer.formal_start()\n    # Start evaluation\n"),
        ("range(num_tasks_in_suite)", "_observer.task_ids(num_tasks_in_suite)"),
        (
            "        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)",
            "        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)\n"
            "        env = _observer.task_env(task_id, task, env, task_description)",
        ),
        (
            "            # Reset environment\n",
            "            _observer.episode_start(task_id, episode_idx, task_description)\n"
            "            # Reset environment\n",
        ),
        (
            "                except Exception as e:\n",
            "                except Exception as e:\n                    _observer.error(e)\n",
        ),
        (
            "            log_file.flush()\n",
            "            log_file.flush()\n            _observer.episode_end(done)\n",
        ),
        ("    # Save local log file\n", "    _observer.formal_finish()\n    # Save local log file\n"),
    ]
    if not save_video:
        replacements.append(("                    replay_images.append(img)\n", ""))
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError(f"Pinned evaluator instrumentation anchor is not unique: {before!r}")
        source = source.replace(before, after, 1)
    compile(source, "instrumented-upstream", "exec")
    return source


def pin_remote_code():
    from experiments.robot import openvla_utils

    def pin_code(cls):
        original = cls.from_pretrained

        def from_pretrained(*load_args, **kwargs):
            kwargs["code_revision"] = CODE_REVISION
            return original(*load_args, **kwargs)

        cls.from_pretrained = from_pretrained

    pin_code(openvla_utils.AutoModelForVision2Seq)
    pin_code(openvla_utils.AutoProcessor)
    # ProcessorMixin does not forward code_revision to every nested auto loader.
    pin_code(openvla_utils.AutoImageProcessor)
    pin_code(openvla_utils.AutoConfig)


def run_child(args) -> int:
    observer = Observer(args)
    started = time.monotonic()
    source_path = args.runtime / "openvla/experiments/robot/libero/run_libero_eval.py"
    source = source_path.read_text()
    if hashlib.sha256(source.encode()).hexdigest() != EVALUATOR_SHA256:
        raise ValueError("Official evaluator does not match the pinned source SHA-256")
    patched = instrument_source(source, args.video == "on")
    generated = args.output_dir / "instrumented-upstream.py"
    generated.write_text(patched)
    (args.output_dir / "instrumentation.patch").write_text(
        "".join(
            difflib.unified_diff(
                source.splitlines(keepends=True),
                patched.splitlines(keepends=True),
                fromfile="upstream/run_libero_eval.py",
                tofile="instrumented/run_libero_eval.py",
            )
        )
    )
    module = types.ModuleType("openvla_native_instrumented")
    module.__file__ = str(generated)
    module._observer = observer
    sys.modules[module.__name__] = module
    exec(compile(patched, str(generated), "exec"), module.__dict__)
    observer.timings["import_s"] = time.monotonic() - started
    pin_remote_code()
    module.get_model = observer.timed_loader(module.get_model, "model_load_s")
    module.get_processor = observer.timed_loader(module.get_processor, "processor_load_s", processor=True)
    module.get_action = observer.observe_action(module.get_action)
    module.get_libero_image = observer.observe_image(module.get_libero_image)
    if args.video == "off":
        module.save_rollout_video = lambda *a, **kw: None
    cfg = module.GenerateConfig(
        pretrained_checkpoint=str(args.checkpoint),
        task_suite_name=args.suite,
        num_trials_per_task=args.episodes_per_task,
        seed=args.seed,
        center_crop=True,
        local_log_dir=str(args.output_dir / "upstream-logs"),
        use_wandb=False,
    )
    status = 0
    try:
        module.eval_libero(cfg)
    except BaseException as error:
        observer.error(error)
        if observer.current is not None:
            observer.current["completed"] = False
            write_json(args.output_dir / "interrupted-episode.json", observer.current)
        status = 1
    finally:
        if observer.pending_requests:
            with (args.output_dir / "requests.jsonl").open("a") as file:
                file.writelines(json.dumps(row) + "\n" for row in observer.pending_requests)
        modules = {}
        for name, loaded in list(sys.modules.items()):
            path = getattr(loaded, "__file__", None)
            if path and name.startswith(("transformers_modules.", "prismatic.extern.hf")):
                p = Path(path)
                if p.is_file():
                    modules[name] = {"file": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        write_json(args.output_dir / "code-provenance.json", modules)
        write_json(
            args.output_dir / "native-timings.json",
            {
                **observer.timings,
                "torch_allocator": getattr(observer, "torch_peak", None),
                "cuda_runtime_version": observer.torch.version.cuda if observer.torch is not None else None,
                "cudnn_version": observer.torch.backends.cudnn.version()
                if observer.torch is not None
                else None,
                "exit_code": status,
            },
        )
        observer.phase("exiting")
    return status
