# hylo: worker-side (runs in a simulator venv: mujoco, a robosuite MjSim)
"""Cameras of a robosuite simulation in the contract's conventions: upright RGB, depth in metres along the optical
axis, intrinsics, the OpenCV camera pose in the contract's world frame, and instance segmentation decoded straight
from MuJoCo's segmentation render (independent of the robosuite version's own read-out)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .geometry import camera_pose, depth_metres, intrinsics


def render_rgbd(sim: Any, camera: str, size: int, T_world_inv: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """(rgb (H, W, 3) uint8, depth (H, W) float32 metres, {"K", "T_world_cam"}) of a square camera; `T_world_inv`
    maps robosuite's world frame into the contract's."""
    m, d = sim.model._model, sim.data._data
    cid = m.camera(camera).id
    rgb, z = sim.render(camera_name=camera, width=size, height=size, depth=True)
    return (np.ascontiguousarray(rgb[::-1]), depth_metres(m, np.ascontiguousarray(z[::-1])),   # GL rows bottom-up
            {"K": intrinsics(float(m.cam_fovy[cid]), size, size), "T_world_cam": camera_pose(d, cid, T_world_inv).tolist()})


def render_instances(sim: Any, camera: str, size: int, geom_ids: np.ndarray) -> np.ndarray:
    """(H, W) int32 instance ids seen by a camera, upright: `geom_ids[g]` for every pixel showing geom g; geoms made
    invisible (alpha 0, e.g. wiped dirt) and non-geom pixels are 0."""
    import mujoco

    m = sim.model._model
    ctx = sim._render_context_offscreen
    ctx.render(size, size, camera_id=m.camera(camera).id, segmentation=True)
    rgb = np.empty((size, size, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(rgb=rgb, depth=None, viewport=mujoco.MjrRect(0, 0, size, size), con=ctx.con)
    code = rgb.astype(np.int32)
    idx = code[..., 0] + (code[..., 1] << 8) + (code[..., 2] << 16)          # scene geom index + 1; 0: none
    n = ctx.scn.ngeom
    lut = np.zeros(n + 1, dtype=np.int32)
    geom = int(mujoco.mjtObj.mjOBJ_GEOM)
    for i in range(n):
        g = ctx.scn.geoms[i]
        if g.segid >= 0 and g.objtype == geom and 0 <= g.objid < len(geom_ids) and m.geom_rgba[g.objid, 3] > 0.0:
            lut[g.segid + 1] = geom_ids[g.objid]
    return np.ascontiguousarray(lut[np.where(idx <= n, idx, 0)][::-1])


def no_reflections(sim: Any) -> None:
    """Turn off planar reflections in the offscreen scene: with software rendering (OSMesa) they double the cost of
    every camera; the images lose a table's mirror sheen and nothing else."""
    import mujoco

    ctx = sim._render_context_offscreen
    if ctx is not None:
        ctx.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0


__all__ = ["no_reflections", "render_instances", "render_rgbd"]
