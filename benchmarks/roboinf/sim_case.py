"""Simulator subprocess: run one native case; communicate only policy inputs.

The simulator interpreter does not import the Kit.
"""

import argparse
import json
import socket
from multiprocessing.connection import Connection
from pathlib import Path

import msgpack
import numpy as np
from roboinf_benchmark.contracts import ACTION_SPECS, ActionChunk, PolicyDescription
from roboinf_benchmark.robotwin import environment_factory
from roboinf_benchmark.runner import run_case
from roboinf_benchmark.suite import load_suite


def main():
    parser = argparse.ArgumentParser()
    for name in ("data", "robotwin", "case"):
        parser.add_argument("--" + name, required=True)
    for name in ("output", "method"):
        parser.add_argument("--" + name)
    parser.add_argument("--fd", type=int)
    parser.add_argument("--domain", choices=("test", "test_random"), required=True)
    parser.add_argument("--action-spec", choices=ACTION_SPECS, default="joint14_absolute")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    scenes, cases = load_suite(args.data, args.domain, verify_release=True)
    case = cases[args.case]
    root = Path(args.robotwin).resolve()
    if args.self_test:
        env = environment_factory(root)(
            scenes[case["task"], case["scene_id"]], case, args.domain
        )
        try:
            env.observe().checked_copy()
        finally:
            env.close()
        return
    out = Path(args.output).resolve()
    with Connection(socket.socket(fileno=args.fd).detach()) as channel:

        class Policy:
            def describe(self):
                return PolicyDescription(args.method, args.action_spec)

            def reset(self, ctx):
                channel.send_bytes(msgpack.packb(["reset", ctx.instruction, ctx.seed]))
                if channel.recv_bytes() != b"ok":
                    raise RuntimeError("Policy reset failed")

            def predict(self, obs):
                images = {k: (v.shape, v.tobytes()) for k, v in obs.images.items()}
                channel.send_bytes(
                    msgpack.packb(
                        [
                            "act",
                            obs.action_step,
                            obs.time_sec,
                            images,
                            obs.joint_state.tolist(),
                            None if obs.eef_state is None else obs.eef_state.tolist(),
                        ]
                    )
                )
                return ActionChunk(
                    np.asarray(msgpack.unpackb(channel.recv_bytes())), args.action_spec
                )

        row = run_case(
            case,
            scenes[case["task"], case["scene_id"]],
            args.domain,
            environment_factory(root),
            Policy(),
            seed=0,
        )
        with out.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
        channel.send_bytes(msgpack.packb(["done", row["action_count"]]))


if __name__ == "__main__":
    main()
