"""Bridge one isolated native RoboInF episode to a Kit PolicyClient."""

import socket
import subprocess
from multiprocessing.connection import Connection
from uuid import uuid4

import msgpack
import numpy as np

from roboscope_kit import EpisodeContext, Observation


def run_case(client, command, task):
    handle = uuid4().hex
    active = False
    parent, child = socket.socketpair()
    try:
        with child:
            proc = subprocess.Popen([*command, "--fd", str(child.fileno())], pass_fds=(child.fileno(),))
        try:
            with Connection(parent.detach()) as channel:
                instruction = None
                while True:
                    message = msgpack.unpackb(channel.recv_bytes(), raw=False)
                    if message[0] == "reset" and not active:
                        _, instruction, seed = message
                        client.reset(EpisodeContext(handle, task, {"seed": seed}))
                        active = True
                        channel.send_bytes(b"ok")
                    elif message[0] == "act" and active:
                        _, step, timestamp, images, joints = message
                        features = {
                            f"images.{key}": np.frombuffer(data, dtype=np.uint8).reshape(shape)
                            for key, (shape, data) in images.items()
                        }
                        features.update(
                            {
                                "state.joints": np.asarray(joints, dtype=np.float32),
                                "language.instruction": instruction,
                            }
                        )
                        chunk = client.act(Observation(handle, step, features, timestamp))
                        channel.send_bytes(msgpack.packb(np.asarray(chunk.actions).tolist()))
                    elif message[0] == "done" and active:
                        if proc.wait() != 0:
                            raise RuntimeError("Simulator exited unsuccessfully")
                        return message[1]
                    else:
                        raise ValueError("Unexpected simulator message")
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
    finally:
        parent.close()
        child.close()
        if active:
            client.end_episode(handle)
