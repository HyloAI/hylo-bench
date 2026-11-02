"""Frames and cameras for the robot contract on MuJoCo-based simulators (pure numpy; shared with the RoboCasa
adapter): 4x4 poses, pinhole intrinsics from a vertical field of view, camera poses in the OpenCV optical frame and
MuJoCo's depth buffer in metres."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def pose(R: Any, p: Any) -> np.ndarray:
    """4x4 homogeneous transform from a rotation matrix and a translation."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(p, dtype=np.float64).reshape(3)
    return T


def body_pose(d: Any, body_id: int) -> np.ndarray:
    """A body's pose in the simulator's world frame (MjData)."""
    return pose(d.xmat[body_id].reshape(3, 3), d.xpos[body_id])


def intrinsics(fovy_deg: float, height: int, width: int) -> list[list[float]]:
    """Pinhole K for a MuJoCo camera (square pixels: the horizontal focal length equals the vertical one)."""
    f = 0.5 * height / math.tan(math.radians(fovy_deg) / 2)
    return [[f, 0.0, (width - 1) / 2], [0.0, f, (height - 1) / 2], [0.0, 0.0, 1.0]]


# MuJoCo cameras look along their -z with +y up; OpenCV's optical frame looks along +z with +y down
GL_TO_CV = np.diag([1.0, -1.0, -1.0])


def camera_pose(d: Any, cam_id: int, T_world_inv: np.ndarray) -> np.ndarray:
    """The camera in the contract's world frame (`T_world_inv` maps the simulator's world into it), OpenCV axes
    (x right, y down, z forward)."""
    return T_world_inv @ pose(d.cam_xmat[cam_id].reshape(3, 3) @ GL_TO_CV, d.cam_xpos[cam_id])


def depth_metres(m: Any, z: np.ndarray) -> np.ndarray:
    """MuJoCo's normalised depth buffer -> metres along the optical axis; 0 where nothing was hit."""
    ext = m.stat.extent
    near, far = m.vis.map.znear * ext, m.vis.map.zfar * ext
    out = (near / (1.0 - z * (1.0 - near / far))).astype(np.float32)
    out[z >= 1.0 - 1e-7] = 0.0
    return out


def project(K: Any, T_world_cam: Any, p_world: Any) -> tuple[float, float, float]:
    """A world point -> (u, v, depth) in an OpenCV camera."""
    p = np.linalg.inv(np.asarray(T_world_cam, float)) @ np.r_[np.asarray(p_world, float).reshape(3), 1.0]
    uvw = np.asarray(K, float) @ p[:3]
    return float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2]), float(p[2])


__all__ = ["GL_TO_CV", "body_pose", "camera_pose", "depth_metres", "intrinsics", "pose", "project"]
