"""Run a public Policy through a persistent JSON pipe, independently of SDK RPCs."""

from __future__ import annotations

import argparse
import base64
import contextlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np


def decode_features(features: dict) -> dict:
    return {name: np.frombuffer(base64.b64decode(value["data"]), dtype=value["dtype"])
            .reshape(value["shape"]).copy() if isinstance(value, dict) and "data" in value else value
            for name, value in features.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-module", required=True)
    parser.add_argument("--policy-class", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--suite", required=True)
    args = parser.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        from roboscope_kit import EpisodeContext, Observation, SessionContext, get_benchmark_spec

        module = args.policy_module
        if module.endswith(".py"):
            path = Path(module).resolve(strict=True)
            sys.path.insert(0, str(path.parent))
            module = path.stem
        policy = getattr(importlib.import_module(module), args.policy_class)()
        policy.load(SessionContext("native-reference", args.checkpoint,
                                   get_benchmark_spec("maniskill", args.suite)))
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        with contextlib.redirect_stdout(sys.stderr):
            operation = request["op"]
            if operation == "reset":
                for episode in request["episodes"]:
                    policy.reset(EpisodeContext.from_dict(episode))
                reply = {"ok": True}
            elif operation == "act":
                observations = [Observation(
                    episode_id=row["episode_id"], step=row["step"],
                    features=decode_features(row["features"]),
                ) for row in request["observations"]]
                chunks = policy.act_batch(observations)
                actions = []
                for chunk in chunks:
                    action = np.asarray(chunk.actions, dtype=np.float32)
                    if action.ndim == 2:
                        if action.shape[0] != 1:
                            raise ValueError("The native reference requires one action per observation")
                        action = action[0]
                    actions.append(action.tolist())
                reply = {"actions": actions}
            elif operation == "end":
                for episode_id in request["episode_ids"]:
                    policy.end_episode(episode_id)
                reply = {"ok": True}
            elif operation == "quit":
                policy.unload()
                reply = {"closed": True}
            else:
                raise ValueError(f"Unknown actor operation {operation!r}")
        print(json.dumps(reply), flush=True)
        if operation == "quit":
            break


if __name__ == "__main__":
    main()
