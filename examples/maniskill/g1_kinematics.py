"""Model-side FK and Cartesian-to-joint conversion for the pinned native G1 URDF."""

from __future__ import annotations

import hashlib
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

URDF_COMMIT = "baab60ede2e89167c1b7aaed41a9aa8e690a9d1e"
URDF_SHA256 = "3ab8835fb251c8313974c13cb9e943e09fbeed9575ec86c36002ee6436868344"
URDF_NAME = "g1_simplified_upper_body.urdf"
URDF_URL = (
    f"https://raw.githubusercontent.com/mani-skill/ManiSkill/{URDF_COMMIT}/"
    f"mani_skill/assets/robots/g1_humanoid/{URDF_NAME}"
)


def rotation(vector: np.ndarray) -> np.ndarray:
    """Convert an axis-angle vector to a matrix, in the articulation root frame."""
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = vector / angle
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def rotation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    delta = target @ current.T
    angle = float(np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1)))
    skew = np.array([delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0],
                     delta[1, 0] - delta[0, 1]])
    if angle < 1e-7:
        return skew / 2
    if np.pi - angle < 1e-5:
        _, vectors = np.linalg.eigh((delta + delta.T) / 2)
        axis = vectors[:, -1]
        return axis * angle * (1 if axis[np.argmax(np.abs(axis))] >= 0 else -1)
    return skew * (angle / (2 * np.sin(angle)))


def origin_matrix(element) -> np.ndarray:
    result = np.eye(4)
    if element is not None:
        result[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        roll, pitch, yaw = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        result[:3, :3] = (
            rotation(np.array([0, 0, yaw])) @ rotation(np.array([0, pitch, 0]))
            @ rotation(np.array([roll, 0, 0]))
        )
    return result


def download_urdf(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / URDF_NAME
    data = path.read_bytes() if path.is_file() else urllib.request.urlopen(URDF_URL, timeout=60).read()
    if hashlib.sha256(data).hexdigest() != URDF_SHA256:
        raise ValueError("Native G1 kinematics differ from the pinned URDF")
    path.write_bytes(data)
    return path


class G1Kinematics:
    def __init__(self, path: Path, joint_names: list[str]) -> None:
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != URDF_SHA256:
            raise ValueError("Expected the unmodified native G1 upper-body URDF")
        tree = ET.fromstring(data)
        self.names = list(joint_names)
        self.indices = {name: index for index, name in enumerate(self.names)}
        joints = {node.get("name"): node for node in tree.findall("joint")}
        moving = {name for name, node in joints.items() if node.get("type") != "fixed"}
        if len(self.names) != 25 or len(self.indices) != 25 or moving != set(self.names):
            raise ValueError("The actual native joint names must match all 25 URDF joints")
        self.lower = np.array([float(joints[name].find("limit").get("lower")) for name in self.names])
        self.upper = np.array([float(joints[name].find("limit").get("upper")) for name in self.names])
        children = {node.find("child").get("link") for node in joints.values()}
        roots = {node.get("name") for node in tree.findall("link")} - children
        if len(roots) != 1:
            raise ValueError("Expected one native articulation root link")
        self.root = roots.pop()
        self.joints = []
        self.ancestors = {self.root: ()}
        pending = list(joints.items())
        while pending:
            ready = [(name, node) for name, node in pending
                     if node.find("parent").get("link") in self.ancestors]
            if not ready:
                raise ValueError("The URDF has disconnected or cyclic joints")
            for name, node in ready:
                parent, child = node.find("parent").get("link"), node.find("child").get("link")
                axis_node = node.find("axis")
                axis = np.fromstring(axis_node.get("xyz") if axis_node is not None else "1 0 0", sep=" ")
                kind = node.get("type")
                index = self.indices.get(name)
                if index is not None:
                    axis /= np.linalg.norm(axis)
                self.joints.append((parent, child, origin_matrix(node.find("origin")), axis, kind, index))
                self.ancestors[child] = self.ancestors[parent] + (() if index is None else (index,))
                pending.remove((name, node))

    def forward(self, qpos: np.ndarray) -> tuple[dict, dict]:
        transforms, axes = {self.root: np.eye(4)}, {}
        for parent, child, origin, axis, kind, index in self.joints:
            before = transforms[parent] @ origin
            motion = np.eye(4)
            if index is not None:
                axes[index] = (before[:3, 3], before[:3, :3] @ axis, kind)
                if kind in {"revolute", "continuous"}:
                    motion[:3, :3] = rotation(axis * qpos[index])
                elif kind == "prismatic":
                    motion[:3, 3] = axis * qpos[index]
                else:
                    raise ValueError(f"Unsupported native joint type {kind}")
            transforms[child] = before @ motion
        return transforms, axes

    def jacobian(self, link: str, transforms: dict, axes: dict, indices: list[int]) -> np.ndarray:
        jacobian = np.zeros((6, len(indices)))
        for column, index in enumerate(indices):
            if index not in self.ancestors[link]:
                continue
            position, axis, kind = axes[index]
            if kind == "prismatic":
                jacobian[:3, column] = axis
            else:
                jacobian[:3, column] = np.cross(axis, transforms[link][:3, 3] - position)
                jacobian[3:, column] = axis
        return jacobian

    def inverse(self, qpos: np.ndarray, goals: dict[str, np.ndarray]) -> np.ndarray:
        """Track observed-frame Cartesian goals; no simulator state is changed."""
        selected = {index for link in goals for index in self.ancestors[link]}
        indices = sorted(selected)
        values = np.asarray(qpos, dtype=np.float64).copy()
        weights = np.tile([1, 1, 1, 0.25, 0.25, 0.25], len(goals))
        for _ in range(12):
            transforms, axes = self.forward(values)
            jacobian = np.concatenate([self.jacobian(link, transforms, axes, indices) for link in goals])
            errors = np.concatenate([
                np.concatenate((target[:3, 3] - transforms[link][:3, 3],
                                rotation_error(target[:3, :3], transforms[link][:3, :3])))
                for link, target in goals.items()
            ])
            if np.linalg.norm(errors[:3]) < 1e-5 and np.linalg.norm(errors) < 1e-4:
                break
            jacobian *= weights[:, None]
            errors *= weights
            change = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + np.eye(len(errors)) * 0.0025, errors,
            )
            values[indices] = np.clip(values[indices] + np.clip(change, -0.1, 0.1),
                                      self.lower[indices], self.upper[indices])
        return values


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download the checksum-pinned native G1 kinematics")
    parser.add_argument("destination", type=Path)
    print(download_urdf(parser.parse_args().destination))
