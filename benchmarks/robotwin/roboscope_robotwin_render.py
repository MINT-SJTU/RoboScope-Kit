#!/usr/bin/env python3
"""Centralized RoboTwin rendering: CPU-physics evaluators share one render service per GPU slot.

In the default (native) mode every evaluator process owns a Vulkan context, a ray-tracing pipeline and
an OIDN denoiser on the GPU. With ``render_mode: central`` the evaluators run without a render device:

* Worker side (``install_worker``, called by ``roboscope_robotwin_launcher.py``). The ``sapien.render``
  classes that describe visuals (materials, textures, render shapes and render bodies) become recorders,
  so task code, actor builders and the URDF loader run unchanged while physics is created as usual.
  Each episode's scene setup records an ordered event log (lights, every entity's render body and the
  camera setup), which is sent to the service once per episode; afterwards each observation sends only
  the poses of the recorded entities and the cameras.
* Service side (``python roboscope_robotwin_render.py --socket PATH``). It replays each episode's events
  with the real ``sapien.render`` classes into a new render-only scene on one shared renderer, exactly
  as a native evaluator builds a new scene per episode, renders the head and wrist cameras and returns
  their RGB through shared memory.

Images are identical to native rendering only with a new scene per episode: re-posing the entities of a
reused ray-tracing scene changes some pixels by one intensity level. The service therefore never reuses
a scene, and like the native evaluator it clears SAPIEN's resource cache periodically.

This file runs under the RoboTwin runtime interpreter (CPython 3.8) and does not import the Kit.
"""

# ruff: noqa: B905 - this file runs on CPython 3.8, whose zip() has no strict= argument
from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import hashlib
import json
import os
import signal
import sys
import time
import types
from multiprocessing.connection import Client, wait
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import numpy as np

READY_MARKER = "ROBOSCOPE_ROBOTWIN_RENDER_READY"
# Printed by an evaluator that lost its render service; the Kit's RoboTwin worker then restarts the Worker,
# which gets a new service, instead of recording the remaining episodes as failures.
SERVICE_LOST_MARKER = "ROBOSCOPE_ROBOTWIN_RENDER_SERVICE_LOST"
# The Kit starts a service and waits for its socket before it starts the Worker, so the socket normally
# exists already; a missing one means the service has exited.
CONNECT_TIMEOUT_S = 15.0
# Native evaluations with video, the viewer, depth, point clouds, segmentation and third-view data off read
# these cameras only; the launcher's lean-camera and skip-unread patches apply the same selection natively.
UNCREATED_CAMERAS = ("observer_camera", "world_camera1")
RENDERED_STATIC_CAMERAS = ("head_camera",)
# Native evaluators call sapien.render.clear_cache() every clear_cache_freq (5) episodes per process.
CLEAR_CACHE_EPISODES_PER_CLIENT = 5
RECORDED_CLASSES = (
    "RenderMaterial", "RenderTexture2D", "RenderTexture", "RenderCubemap",
    "RenderShapeBox", "RenderShapeSphere", "RenderShapeCapsule", "RenderShapeCylinder",
    "RenderShapePlane", "RenderShapeTriangleMesh", "RenderBodyComponent",
)
SHARED_CLASSES = ("RenderMaterial", "RenderTexture2D", "RenderTexture", "RenderCubemap")
CAMERA_KWARGS = ("camera", "left_embodiment_config", "pcd_crop", "pcd_down_sample_num", "bbox")
UNSUPPORTED_DATA = (
    "depth", "pointcloud", "observer", "third_view", "mesh_segmentation", "actor_segmentation",
)


# --------------------------------------------------------------------------- recorded visual events


def encode(value):
    """Serialize recorded arguments into JSON data (poses, arrays, tuples, nested recorders)."""

    import sapien

    if isinstance(value, Recorder):
        return {"__stub__": type(value).__name__, "ops": [encode(op) for op in value._ops]}
    if isinstance(value, sapien.Pose):
        return {"__pose__": [np.asarray(value.p, dtype=np.float64).tolist(),
                             np.asarray(value.q, dtype=np.float64).tolist()]}
    if isinstance(value, np.ndarray):
        return {"__array__": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {"__tuple__": [encode(item) for item in value]}
    if isinstance(value, list):
        return [encode(item) for item in value]
    if isinstance(value, dict):
        return {"__dict__": [[encode(key), encode(item)] for key, item in value.items()]}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"cannot record visual argument of type {type(value).__name__}")


class Recorder:
    """Stand-in for a ``sapien.render`` value class: records construction, attribute writes and calls."""

    def __init__(self, *args, **kwargs):
        object.__setattr__(self, "_ops", [("init", list(args), dict(kwargs))])
        object.__setattr__(self, "_values", dict(kwargs))

    def __setattr__(self, name, value):
        self._ops.append(("set", name, value))
        self._values[name] = value

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        values = object.__getattribute__(self, "_values")
        if name in values:
            return values[name]
        if name.startswith("get_") and name[4:] in values:
            return lambda: values[name[4:]]

        def call(*args, **kwargs):
            self._ops.append(("call", name, list(args), dict(kwargs)))
            if name.startswith("set_") and len(args) == 1 and not kwargs:
                values[name[4:]] = args[0]
            if name == "attach":
                values.setdefault("render_shapes", []).append(args[0])
            return self

        return call


class _Registry:
    """Ordered visual events of one CPU scene."""

    def __init__(self):
        self.events = []
        self.entities = []  # (entity, recorded render body) in scene-add order
        self.removed = set()  # indices of entities removed after the events were sent
        self.sent = False


def _registry(scene):
    registry = getattr(scene, "_roboscope_visual_registry", None)
    if registry is None:
        registry = _Registry()
        scene._roboscope_visual_registry = registry
    return registry


class _RecordedLight:
    def __init__(self, kind):
        self.kind = kind

    def __getattr__(self, name):
        raise RuntimeError(f"central rendering does not support changing a {self.kind} after scene setup")


# Render bodies added to entities that are not in a scene yet, by entity id. A scene setup that fails before
# adding them (an unstable seed) leaves them behind, so each new scene and each closed task clears them.
_PENDING_BODIES = {}


def install_recorders():
    """Replace the visual description classes of ``sapien.render`` in this CPU-only process."""

    import sapien
    from sapien import pysapien

    stubs = {name: type(name, (Recorder,), {}) for name in RECORDED_CLASSES}
    for module in {id(sapien.render): sapien.render, id(pysapien.render): pysapien.render}.values():
        for name, cls in stubs.items():
            setattr(module, name, cls)
    body_class = stubs["RenderBodyComponent"]
    pending = _PENDING_BODIES

    fresh_add_component = sapien.Entity.add_component

    def add_component(self, component):
        if isinstance(component, body_class):
            pending.setdefault(id(self), (self, []))[1].append(component)
            return self
        return fresh_add_component(self, component)

    fresh_add_entity = sapien.Scene.add_entity
    fresh_remove_entity = sapien.Scene.remove_entity

    def add_entity(self, entity):
        result = fresh_add_entity(self, entity)
        bodies = pending.pop(id(entity), None)
        if bodies is not None:
            registry = _registry(self)
            for body in bodies[1]:
                index = len(registry.entities)
                event = {"kind": "entity", "index": index, "name": entity.name, "body": encode(body)}
                registry.events.append(event)
                registry.entities.append((entity, body))
        return result

    def remove_entity(self, entity):
        registry = _registry(self)
        indices = [index for index, (known, _) in enumerate(registry.entities) if known is entity]
        if indices and not registry.sent:
            raise RuntimeError("central rendering does not support removing a rendered entity during setup")
        registry.removed.update(indices)
        return fresh_remove_entity(self, entity)

    def light(kind):
        def record(self, *args, **kwargs):
            event = {"kind": kind, "args": encode(list(args)), "kwargs": encode(dict(kwargs))}
            _registry(self).events.append(event)
            return _RecordedLight(kind)

        return record

    sapien.Entity.add_component = add_component
    sapien.Scene.add_entity = add_entity
    sapien.Scene.remove_entity = remove_entity
    sapien.Scene.set_ambient_light = light("set_ambient_light")
    lights = ("add_directional_light", "add_point_light", "add_spot_light", "add_area_light_for_ray_tracing")
    for kind in lights:
        if hasattr(sapien.Scene, kind):
            setattr(sapien.Scene, kind, light(kind))
    sapien.Scene.update_render = lambda self: None


def scene_events(scene):
    registry = _registry(scene)
    registry.sent = True
    events = registry.events
    text = json.dumps(events, sort_keys=True, separators=(",", ":"))
    return {"events": events, "digest": hashlib.sha256(text.encode()).hexdigest()}


def entity_poses(scene):
    """Poses of the entities in the sent events; an entity removed since is moved far below the scene."""

    registry = _registry(scene)
    poses = np.empty((len(registry.entities), 7), dtype=np.float32)
    for index, (entity, _) in enumerate(registry.entities):
        if index in registry.removed:
            poses[index] = (0, 0, -1000, 1, 0, 0, 0)
        else:
            pose = entity.pose
            poses[index, :3], poses[index, 3:] = pose.p, pose.q
    return poses


def pack_pose(pose):
    return np.r_[pose.p, pose.q].astype(np.float32)


def state_digest(body_poses, camera_poses):
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(body_poses, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(camera_poses, dtype=np.float32).tobytes())
    return digest.hexdigest()


# --------------------------------------------------------------------------- worker side


class RenderServiceLost(RuntimeError):
    pass


def service_lost(detail):
    """Report a lost render service on stdout, where the Kit's worker reads it, and return the error."""

    print(SERVICE_LOST_MARKER, json.dumps({"pid": os.getpid(), "detail": detail}), flush=True)
    return RenderServiceLost(f"RoboTwin render service lost: {detail}")


class RenderClient:
    """One evaluator's connection to its render service."""

    def __init__(self, address, timeout_s=CONNECT_TIMEOUT_S):
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                self.connection = Client(address, family="AF_UNIX")
                break
            except (FileNotFoundError, ConnectionRefusedError) as error:
                if time.monotonic() > deadline:
                    raise service_lost(f"cannot connect to {address}: {error!r}") from error
                time.sleep(0.2)
        self.memory = None
        self.images = None
        self.camera_names = []
        self.episode = -1
        self.frame = -1

    def _request(self, message, kind):
        try:
            self.connection.send(message)
            reply = self.connection.recv()
        except (EOFError, OSError) as error:
            raise service_lost(f"connection closed: {error!r}") from error
        if reply.get("kind") == "error":
            # The service reports a failure to every client and then exits.
            raise service_lost("the service failed:\n" + reply["error"])
        if reply.get("kind") != kind:
            raise RuntimeError(f"unexpected RoboTwin render service reply {reply.get('kind')!r}")
        return reply

    def reset(self, spec):
        self.episode += 1
        self.frame = -1
        reply = self._request({"kind": "reset", "episode": self.episode, "spec": spec}, "reset")
        self.camera_names = list(reply["camera_names"])
        if self.memory is None or self.memory.name != reply["memory"]:
            self._release_memory()
            self.memory = SharedMemory(name=reply["memory"])
            # The service owns the segment; this process's resource tracker must not unlink it on exit.
            from multiprocessing import resource_tracker

            resource_tracker.unregister(self.memory._name, "shared_memory")
        self.images = np.ndarray(reply["shape"], dtype=np.uint8, buffer=self.memory.buf)
        return reply

    def render(self, body_poses, camera_poses):
        self.frame += 1
        digest = state_digest(body_poses, camera_poses)
        reply = self._request({"kind": "render", "episode": self.episode, "frame": self.frame,
                               "body_poses": body_poses, "camera_poses": camera_poses, "digest": digest},
                              "rendered")
        if reply["episode"] != self.episode or reply["frame"] != self.frame or reply["digest"] != digest:
            raise RuntimeError("RoboTwin render service answered a different frame")
        images = {name: self.images[index].copy() for index, name in enumerate(self.camera_names)}
        return images, reply["camera_config"]

    def _release_memory(self):
        self.images = None
        if self.memory is not None:
            self.memory.close()
            self.memory = None

    def close(self):
        try:
            self.connection.send({"kind": "close"})
            self.connection.recv()
        except (EOFError, OSError):
            pass
        self.connection.close()
        self._release_memory()


class RemoteCameras:
    """Replaces ``task.cameras``: ``get_obs`` packs observations as natively, with RGB from the service."""

    def __init__(self, task, client, cameras):
        self.task, self.client = task, client
        poses = {name: camera.entity.pose
                 for name, camera in zip(cameras.static_camera_name, cameras.static_camera_list)}
        poses.update(left_camera=cameras.left_camera.entity.pose,
                     right_camera=cameras.right_camera.entity.pose)
        self.names = list(client.camera_names)
        self.poses = np.stack([pack_pose(poses[name]) for name in self.names])
        self.images, self.config = None, None

    def update_wrist_camera(self, left_pose, right_pose):
        for name, pose in (("left_camera", left_pose), ("right_camera", right_pose)):
            if name in self.names:
                self.poses[self.names.index(name)] = pack_pose(pose)

    def update_picture(self):
        self.images, self.config = self.client.render(entity_poses(self.task.scene), self.poses.copy())

    def get_rgb(self):
        if self.images is None:
            raise RuntimeError("update_picture must precede get_rgb")
        return {name: {"rgb": image} for name, image in self.images.items()}

    def get_config(self):
        if self.config is None:
            raise RuntimeError("update_picture must precede get_config")
        # Native get_config returns new dictionaries that get_obs then extends with the images.
        return copy.deepcopy(self.config)


class _MetadataCamera:
    """Camera created by ``Camera.load_camera`` in a worker: it keeps poses and never renders."""

    def __init__(self, name, **_kwargs):
        import sapien

        self.name = name
        self.entity = self
        self.pose = sapien.Pose()

    def set_pose(self, pose):
        self.pose = pose

    def get_pose(self):
        return self.pose

    def take_picture(self):
        raise RuntimeError("a central-render worker camera cannot render")


class _MetadataCameraScene:
    def add_camera(self, name, *_args, **kwargs):
        return _MetadataCamera(name, **kwargs)


_client = None


def _client_for(address):
    global _client
    if _client is None:
        _client = RenderClient(address)
    return _client


def close_worker_client():
    global _client
    if _client is not None:
        client, _client = _client, None
        client.close()


def _install_expert_planner_stub(robotwin_root):
    """Load ``envs.robot.planner`` without cuRobo: workers only use MplibPlanner and the gripper planner."""

    import ast

    path = Path(robotwin_root) / "envs" / "robot" / "planner.py"
    module = types.ModuleType("envs.robot.planner")
    module.__package__ = "envs.robot"
    module.__file__ = str(path)

    class CuroboPlanner:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("central-render workers do not run cuRobo expert planning")

    tree = ast.parse(path.read_text())
    body = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
            or isinstance(node, ast.ClassDef) and node.name == "MplibPlanner"]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), module.__dict__)
    module.CuroboPlanner = CuroboPlanner
    sys.modules[module.__name__] = module


def install_worker(robotwin_root, address):
    """Make this evaluator a CPU-physics worker of the render service at ``address``."""

    import numpy as _np
    import sapien

    _install_expert_planner_stub(robotwin_root)
    install_recorders()
    from envs._base_task import Base_Task
    from envs.camera.camera import Camera

    def setup_scene(self, **kwargs):
        # The native setup_scene on a CPU system; lights are recorded, with the same random draws.
        _PENDING_BODIES.clear()
        self.engine, self.renderer = sapien.Engine(), None
        sapien.physx.set_scene_config(sapien.SceneConfig())
        self.scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
        self.scene.set_timestep(kwargs.get("timestep", 1 / 250))
        self.scene.add_ground(kwargs.get("ground_height", 0))
        self.scene.default_physical_material = self.scene.create_physical_material(
            kwargs.get("static_friction", 0.5),
            kwargs.get("dynamic_friction", 0.5),
            kwargs.get("restitution", 0),
        )
        self.scene.set_ambient_light(kwargs.get("ambient_light", [0.5, 0.5, 0.5]))
        shadow = kwargs.get("shadow", True)
        direction_lights = kwargs.get("direction_lights", [[[0, 0.5, -1], [0.5, 0.5, 0.5]]])
        self.direction_light_lst = []
        for direction_light in direction_lights:
            if self.random_light:
                direction_light[1] = [_np.random.rand(), _np.random.rand(), _np.random.rand()]
            self.direction_light_lst.append(
                self.scene.add_directional_light(direction_light[0], direction_light[1], shadow=shadow)
            )
        point_lights = kwargs.get("point_lights", [[[1, 0, 1.8], [1, 1, 1]], [[-1, 0, 1.8], [1, 1, 1]]])
        self.point_light_lst = []
        for point_light in point_lights:
            if self.random_light:
                point_light[1] = [_np.random.rand(), _np.random.rand(), _np.random.rand()]
            self.point_light_lst.append(
                self.scene.add_point_light(point_light[0], point_light[1], shadow=shadow)
            )

    def load_camera(self, **kwargs):
        # The native camera setup, including its random draws, without render cameras.
        self.cameras = Camera(
            bias=self.table_z_bias, random_head_camera_dis=self.random_head_camera_dis, **kwargs
        )
        self.cameras.load_camera(_MetadataCameraScene())
        recorded = {key: kwargs[key] for key in CAMERA_KWARGS if key in kwargs}
        _registry(self.scene).events.append({"kind": "cameras", "kwargs": encode(dict(
            bias=self.table_z_bias, random_head_camera_dis=self.random_head_camera_dis, **recorded))})
        self.scene.step()

    def update_render(self):
        if self.crazy_random_light:
            raise RuntimeError("central rendering does not support per-frame light changes")
        self.cameras.update_wrist_camera(
            self.robot.left_camera.get_pose(), self.robot.right_camera.get_pose()
        )

    def close_env(self, clear_cache=False):
        # The service owns every GPU resource and clears its cache itself.
        _PENDING_BODIES.clear()
        self.close()

    fresh_init = Base_Task._init_task_env_

    def init_task_env(self, *args, **kwargs):
        data = kwargs.get("data_type") or {}
        if any(data.get(key) for key in UNSUPPORTED_DATA) or kwargs.get("render_freq"):
            raise RuntimeError("central rendering serves RGB observations only (no depth, point clouds, "
                               "segmentation, observer or third-view data, or viewer)")
        result = fresh_init(self, *args, **kwargs)
        cameras = self.cameras
        client = _client_for(address)
        client.reset(scene_events(self.scene))
        self.cameras = RemoteCameras(self, client, cameras)
        return result

    Base_Task.setup_scene = setup_scene
    Base_Task.load_camera = load_camera
    Base_Task._update_render = update_render
    Base_Task.close_env = close_env
    Base_Task._init_task_env_ = init_task_env

    # The official evaluator checks rendering with a local renderer before evaluating; the service has
    # already created its renderer, and this process must not create one.
    preflight = types.ModuleType("test_render")
    preflight.Sapien_TEST = lambda: print(
        "ROBOSCOPE_RENDER_PREFLIGHT delegated to the render service", flush=True
    )
    sys.modules["test_render"] = preflight


# --------------------------------------------------------------------------- service side


class Realizer:
    """Builds recorded visuals with the real classes; identical materials and textures are shared."""

    def __init__(self):
        self.cache = {}

    def decode(self, value):
        import sapien

        if isinstance(value, list):
            return [self.decode(item) for item in value]
        if isinstance(value, dict):
            if "__stub__" in value:
                return self.realize(value)
            if "__pose__" in value:
                position, quaternion = value["__pose__"]
                return sapien.Pose(position, quaternion)
            if "__array__" in value:
                return np.asarray(value["__array__"], dtype=value["dtype"])
            if "__tuple__" in value:
                return tuple(self.decode(item) for item in value["__tuple__"])
            if "__dict__" in value:
                return {self.decode(key): self.decode(item) for key, item in value["__dict__"]}
            raise ValueError("unknown recorded value")
        return value

    def realize(self, node):
        import sapien

        name = node["__stub__"]
        key = json.dumps(node, sort_keys=True) if name in SHARED_CLASSES else None
        if key is not None and key in self.cache:
            return self.cache[key]
        built = None
        for op in node["ops"]:
            op = op["__tuple__"]
            if op[0] == "init":
                built = getattr(sapien.render, name)(*self.decode(op[1]), **self.decode(op[2]))
            elif op[0] == "set":
                setattr(built, op[1], self.decode(op[2]))
            else:
                getattr(built, op[1])(*self.decode(op[2]), **self.decode(op[3]))
        if key is not None:
            self.cache[key] = built
        return built


class _LeanCameraScene:
    """Like the launcher's lean cameras: the observer and world cameras are not created."""

    def __init__(self, scene):
        self._scene = scene

    def add_camera(self, name, *args, **kwargs):
        if name in UNCREATED_CAMERAS:
            return _MetadataCamera(name)
        return self._scene.add_camera(name, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._scene, name)


class EpisodeScene:
    """Render-only replica of one worker episode, built by the same calls as the native scene."""

    def __init__(self, spec, realizer):
        import sapien
        from envs.camera.camera import Camera

        self.scene = sapien.Scene([sapien.render.RenderSystem()])
        self.entities = []
        cameras = None
        for event in spec["events"]:
            kind = event["kind"]
            if kind == "entity":
                entity = sapien.Entity()
                entity.name = event["name"]
                entity.add_component(realizer.decode(event["body"]))
                self.scene.add_entity(entity)
                self.entities.append(entity)
            elif kind == "cameras":
                cameras = Camera(**realizer.decode(event["kwargs"]))
                cameras.load_camera(_LeanCameraScene(self.scene))
            else:
                getattr(self.scene, kind)(*realizer.decode(event["args"]), **realizer.decode(event["kwargs"]))
        if cameras is None:
            raise RuntimeError("the recorded scene never loaded its cameras")
        # The front camera exists, as in the native scene, but is not rendered (see the launcher's
        # skip-unread patch); each camera renders independently, so the others are unchanged.
        rendered = [
            (name, camera)
            for name, camera in zip(cameras.static_camera_name, cameras.static_camera_list)
            if name in RENDERED_STATIC_CAMERAS
        ]
        cameras.static_camera_name = [name for name, _ in rendered]
        cameras.static_camera_list = [camera for _, camera in rendered]
        self.cameras = cameras
        mapping = dict(rendered)
        if cameras.collect_wrist_camera:
            mapping.update(left_camera=cameras.left_camera, right_camera=cameras.right_camera)
        self.camera_names = list(cameras.get_config())
        self.camera_entities = [mapping[name].entity for name in self.camera_names]
        sizes = {(int(mapping[name].height), int(mapping[name].width)) for name in self.camera_names}
        if len(sizes) != 1:
            raise RuntimeError(f"central rendering needs one image size for all cameras; got {sorted(sizes)}")
        self.image_shape = (len(self.camera_names), *sizes.pop(), 3)

    def render(self, body_poses, camera_poses):
        import sapien

        if len(body_poses) != len(self.entities) or len(camera_poses) != len(self.camera_entities):
            raise RuntimeError("render request does not match the episode scene")
        for entity, pose in zip(self.entities, body_poses):
            entity.pose = sapien.Pose(pose[:3], pose[3:])
        for entity, pose in zip(self.camera_entities, camera_poses):
            entity.pose = sapien.Pose(pose[:3], pose[3:])
        self.scene.update_render()
        self.cameras.update_picture()
        images = {name: value["rgb"] for name, value in self.cameras.get_rgb().items()}
        return images, self.cameras.get_config()

    def close(self):
        self.entities.clear()
        self.camera_entities.clear()
        self.cameras = None
        self.scene = None


class _ClientState:
    def __init__(self):
        self.scene = None
        self.memory = None
        self.images = None
        self.episode = -1
        self.frame = -1

    def allocate(self, shape):
        size = int(np.prod(shape))
        if self.memory is None or self.memory.size < size:
            self.release_memory()
            self.memory = SharedMemory(create=True, size=size)
        self.images = np.ndarray(shape, dtype=np.uint8, buffer=self.memory.buf)

    def release_memory(self):
        self.images = None
        if self.memory is not None:
            self.memory.close()
            self.memory.unlink()
            self.memory = None

    def release(self):
        if self.scene is not None:
            self.scene.close()
            self.scene = None
        self.release_memory()


def serve(address, parent_pid):
    """Serve render requests on ``address`` until the parent exits.

    Any failure is reported to every connected worker and ends the service; the Kit starts a new one
    for the next Worker it starts on this GPU slot.
    """

    import socket
    import traceback
    from multiprocessing.connection import Connection

    import sapien

    # The native evaluator's renderer settings (setup_scene and the launcher's shared engine).
    sapien.render.set_global_config(max_num_materials=50000, max_num_textures=50000)
    sapien.render.set_camera_shader_dir("rt")
    sapien.render.set_ray_tracing_samples_per_pixel(32)
    sapien.render.set_ray_tracing_path_depth(8)
    sapien.render.set_ray_tracing_denoiser("oidn")
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)
    realizer = Realizer()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(address)
    server.listen(64)
    clients = {}
    stats = {"episodes": 0, "renders": 0, "cache_clears": 0, "reset_s": 0.0, "render_s": 0.0}
    print(READY_MARKER, json.dumps({"pid": os.getpid(), "address": address}), flush=True)

    def drop(connection):
        clients.pop(connection).release()
        connection.close()

    def handle(connection):
        state = clients[connection]
        try:
            message = connection.recv()
        except (EOFError, OSError):
            drop(connection)
            return
        kind = message["kind"]
        if kind == "reset":
            began = time.perf_counter()
            if state.scene is not None:
                state.scene.close()
                state.scene = None
                gc.collect()
            stats["episodes"] += 1
            if stats["episodes"] % (CLEAR_CACHE_EPISODES_PER_CLIENT * len(clients)) == 0:
                # Drop the realizer's references too, or clear_cache cannot free the shared materials and
                # textures; scenes that are still open keep their own references.
                realizer.cache.clear()
                sapien.render.clear_cache()
                stats["cache_clears"] += 1
            state.scene = EpisodeScene(message["spec"], realizer)
            shape = state.scene.image_shape
            state.allocate(shape)
            state.episode, state.frame = message["episode"], -1
            stats["reset_s"] += time.perf_counter() - began
            connection.send({"kind": "reset", "camera_names": state.scene.camera_names,
                             "memory": state.memory.name, "shape": shape})
        elif kind == "render":
            expected = (state.episode, state.frame + 1)
            if state.scene is None or (message["episode"], message["frame"]) != expected:
                raise RuntimeError("out-of-order render request")
            if state_digest(message["body_poses"], message["camera_poses"]) != message["digest"]:
                raise RuntimeError("render request digest mismatch")
            began = time.perf_counter()
            images, config = state.scene.render(message["body_poses"], message["camera_poses"])
            for index, name in enumerate(state.scene.camera_names):
                state.images[index] = images[name]
            state.frame = message["frame"]
            stats["renders"] += 1
            stats["render_s"] += time.perf_counter() - began
            connection.send({"kind": "rendered", "episode": state.episode, "frame": state.frame,
                             "digest": message["digest"], "camera_config": config})
        elif kind == "close":
            connection.send({"kind": "closed"})
            drop(connection)
        else:
            raise RuntimeError(f"unknown render request {kind!r}")

    try:
        while os.getppid() == parent_pid:
            for ready in wait([server, *clients], timeout=1.0):
                if ready is server:
                    accepted, _ = server.accept()
                    clients[Connection(accepted.detach())] = _ClientState()
                else:
                    handle(ready)
    except Exception:
        error = traceback.format_exc()
        for connection in clients:
            with contextlib.suppress(OSError):
                connection.send({"kind": "error", "error": error})
        raise
    finally:
        print("ROBOSCOPE_ROBOTWIN_RENDER_STATS", json.dumps(stats), flush=True)
        for connection in list(clients):
            drop(connection)
        server.close()
        if os.path.exists(address):
            os.unlink(address)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True, help="Unix socket path to serve on")
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, required=True, help="exit when this process exits")
    args = parser.parse_args()
    root = args.robotwin_root.resolve()
    os.chdir(root)  # recorded asset paths are relative to the RoboTwin root
    for path in (root, root / "script", root / "policy"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    # The Kit stops services with SIGTERM; exiting through serve's cleanup prints the stats and removes
    # the socket. A second SIGTERM during that cleanup is ignored.
    def stop(*_):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    serve(args.socket, args.parent_pid)


if __name__ == "__main__":
    main()
