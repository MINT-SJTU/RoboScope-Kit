"""Lossless diagnostic observations and model/environment actions for small runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class DiagnosticTrace:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.pending = None

    def observation(self, episode_id, step, observation, resized):
        import numpy as np

        self.pending = {
            "episode_id": episode_id,
            "step": step,
            "raw_primary": np.asarray(observation["agentview_image"]).copy(),
            "resized_primary": np.asarray(resized).copy(),
            "state": {
                key: np.asarray(observation[key]).copy()
                for key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot0_joint_pos")
                if key in observation
            },
        }

    def processor(self, prompt, image, inputs):
        import numpy as np

        self.pending.update(
            prompt=prompt,
            model_image=np.asarray(image).copy(),
            processor_inputs={key: value.detach().cpu().numpy().copy() for key, value in inputs.items()},
        )

    def model_action(self, action, unnorm_key):
        import numpy as np

        # Upstream modifies the returned action's gripper in place after get_action.
        self.pending.update(model_action=np.asarray(action).copy(), unnorm_key=unnorm_key)

    def executed(self, action, reward, done):
        import numpy as np
        from PIL import Image

        pending = self.pending
        if pending is None or "model_action" not in pending:
            raise RuntimeError("Environment action has no corresponding captured model request")
        directory = self.root / pending["episode_id"] / f"step-{pending['step']:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        images = {
            "raw_primary": pending["raw_primary"],
            "canonical_primary": pending["raw_primary"][::-1, ::-1],
            "resized_primary": pending["resized_primary"],
            "model_image": pending["model_image"],
        }
        identities = {}
        for name, value in images.items():
            Image.fromarray(value).save(directory / f"{name}.png")
            identities[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
            }
        np.savez_compressed(directory / "processor-inputs.npz", **pending["processor_inputs"])
        np.savez_compressed(directory / "simulator-state.npz", **pending["state"])
        raw = pending["model_action"]
        executed = np.asarray(action)
        expected = raw.copy()
        expected[-1] = -np.sign(2 * expected[-1] - 1)
        record = {
            "episode_id": pending["episode_id"],
            "step": pending["step"],
            "prompt": pending["prompt"],
            "unnorm_key": pending["unnorm_key"],
            "files": str(directory.relative_to(self.root)),
            "images": identities,
            "model_action": raw.tolist(),
            "environment_action": executed.tolist(),
            "expected_environment_action": expected.tolist(),
            "action_shape_valid": raw.shape == (7,) and executed.shape == (7,),
            "actions_finite": bool(np.isfinite(raw).all() and np.isfinite(executed).all()),
            "action_conversion_matches": bool(np.array_equal(expected, executed)),
            "reward": float(reward),
            "done": bool(done),
        }
        with (self.root / "steps.jsonl").open("a") as file:
            file.write(json.dumps(record, allow_nan=False) + "\n")
        self.pending = None
