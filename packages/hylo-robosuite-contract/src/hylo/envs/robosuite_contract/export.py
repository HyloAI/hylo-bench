# hylo: worker-side (runs in a simulator venv: mujoco)
"""An arm's MJCF exported for hylo-manip's kinematics, and its TCP measured on it.

`export_model` takes the XML a simulator builds for one arm with its gripper (each adapter produces it with its own
robosuite version) and writes it once into the asset cache, content-addressed:
`$HYLO_ASSETS/<namespace>/<digest>/<file>` with the meshes copied into `assets/`. The root body goes to the origin
(the contract places it with ArmSpec.base_in_world), the given finger joints are flipped (axis and range) so every
finger opens towards the upper end of its range (how hylo-manip maps a hand command to finger joints), and
everything that is not kinematics or geometry is removed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


def cache_root() -> Path:
    """The asset cache: $HYLO_ASSETS, else ~/.cache/hylo/assets."""
    return Path(os.environ.get("HYLO_ASSETS") or Path.home() / ".cache" / "hylo" / "assets").expanduser()


def rewrite_model(xml: str, assets: dict[str, Path], root_body: str | None = None,
                  flip_joints: Sequence[str] = ()) -> str:
    """The XML with asset files renamed into assets/ (collected in `assets`), `root_body` at the origin, the joints in
    `flip_joints` mirrored, and actuators, sensors, tendons, equality and contact sections removed."""
    root = ET.fromstring(xml)
    comp = root.find("compiler")
    if comp is None:
        comp = ET.Element("compiler")
        root.insert(0, comp)
    comp.set("angle", "radian")
    for k in ("meshdir", "texturedir", "assetdir"):
        comp.attrib.pop(k, None)
    for node in root.iter():
        f = node.get("file")
        if f:
            src = Path(f)
            name = f"{src.parent.name}_{src.name}"
            if name in assets and assets[name] != src:
                name = f"{hashlib.sha256(str(src).encode()).hexdigest()[:8]}_{src.name}"
            assets[name] = src
            node.set("file", f"assets/{name}")
    world = root.find("worldbody")
    if root_body and world is not None:
        for body in world.findall("body"):
            if body.get("name") == root_body:
                body.set("pos", "0 0 0")
                for k in ("quat", "euler", "axisangle", "xyaxes", "zaxis"):
                    body.attrib.pop(k, None)
    flip = set(flip_joints)
    for j in root.iter("joint"):
        if j.get("name") in flip:
            axis = [float(v) for v in (j.get("axis") or "0 0 1").split()]
            lo, hi = [float(v) for v in (j.get("range") or "0 0").split()]
            j.set("axis", " ".join(f"{-v:g}" for v in axis))
            j.set("range", f"{-hi:g} {-lo:g}")
    for tag in ("actuator", "sensor", "tendon", "equality", "contact"):
        for node in root.findall(tag):
            root.remove(node)
    return ET.tostring(root, encoding="unicode")


def export_model(xml: str, namespace: str, root_body: str | None = None, flip_joints: Sequence[str] = (),
                 source_note: str = "", file_name: str = "model.xml", version: str = "1") -> Path:
    """Write the arm's model once into `$HYLO_ASSETS/<namespace>/<digest>/<file_name>` (digest of the rewritten XML
    and every asset's bytes); returns the XML path. Safe with several workers exporting at once."""
    assets: dict[str, Path] = {}
    text = rewrite_model(xml, assets, root_body, flip_joints)
    h = hashlib.sha256((version + text).encode())
    for name in sorted(assets):
        h.update(name.encode())
        h.update(assets[name].read_bytes())
    folder = cache_root() / namespace / h.hexdigest()[:16]
    path = folder / file_name
    if path.is_file():
        return path
    tmp = folder.with_name(folder.name + f".tmp{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "assets").mkdir(parents=True)
    for name, src in assets.items():
        shutil.copyfile(src, tmp / "assets" / name)
    (tmp / file_name).write_text(text)
    (tmp / "SOURCE.txt").write_text(source_note + " Root body at the origin; flipped joints: "
                                    f"{', '.join(flip_joints) or 'none'}; actuators removed. Derived data, not "
                                    "redistributed.\n")
    try:
        os.replace(tmp, folder)
    except OSError:                                    # another worker wrote it first
        shutil.rmtree(tmp, ignore_errors=True)
    return path


def _frames(model_path: str | Path, ee_body: str, finger_joints: Sequence[str], travel: float):
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    for j in finger_joints:                            # fingers fully open (after the flip: the top of the range)
        d.qpos[m.jnt_qposadr[m.joint(j).id]] = travel
    mujoco.mj_kinematics(m, d)
    ee = m.body(ee_body).id
    return m, d, ee, d.xmat[ee].reshape(3, 3), d.xpos[ee].copy()


def _tcp(origin: np.ndarray, y: np.ndarray) -> list[list[float]]:
    x = np.array([0.0, 0.0, 1.0])                      # approach: the hand's +z
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, np.cross(x, y), origin
    return np.round(T, 6).tolist()


def parallel_hand_geometry(model_path: str | Path, ee_body: str, finger_joints: Sequence[str], pads: Sequence[str],
                           travel: float) -> dict[str, Any]:
    """Parallel gripper: TCP midway between the two finger pads (fully open), +x the hand's approach axis (its +z), +y
    the closing axis (pad 2 towards pad 1); `max_width`: the gap between the pad faces when open; `tcp_offset`: the
    TCP's distance along the approach axis."""
    m, d, ee, R, p = _frames(model_path, ee_body, finger_joints, travel)
    pp = [R.T @ (d.geom_xpos[m.geom(g).id] - p) for g in pads]
    origin = (pp[0] + pp[1]) / 2
    y = pp[0] - pp[1]
    y[2] = 0.0
    gap = float(np.linalg.norm(y))
    half = float(m.geom_size[m.geom(pads[0]).id][1])          # pad half thickness along the closing axis
    return {"tcp_in_ee": _tcp(origin, y / gap), "max_width": round(gap - 2 * half, 4),
            "tcp_offset": round(float(origin[2]), 4)}


def face_hand_geometry(model_path: str | Path, ee_body: str) -> dict[str, Any]:
    """A fingerless tool (a sponge): TCP at the centre of its contact face, the deepest point of the hand body's
    geoms along the approach axis (+z of the hand)."""
    import mujoco

    m, d, ee, R, p = _frames(model_path, ee_body, (), 0.0)
    reach = 0.0
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != ee:
            continue
        c = R.T @ (d.geom_xpos[g] - p)
        if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_BOX):
            extent = float(np.abs((R.T @ d.geom_xmat[g].reshape(3, 3))[2] * m.geom_size[g]).sum())
        else:
            extent = float(m.geom_rbound[g])
        reach = max(reach, float(c[2]) + extent)
    return {"tcp_in_ee": _tcp(np.array([0.0, 0.0, reach]), np.array([0.0, 1.0, 0.0])), "max_width": 0.0,
            "tcp_offset": round(reach, 4)}


def point_hand_geometry(model_path: str | Path, ee_body: str, point_body: str) -> dict[str, Any]:
    """No gripper: TCP at a body of the hand (robosuite's grip point), +x along the hand's +z."""
    m, d, ee, R, p = _frames(model_path, ee_body, (), 0.0)
    origin = R.T @ (d.xpos[m.body(point_body).id] - p)
    return {"tcp_in_ee": _tcp(origin, np.array([0.0, 1.0, 0.0])), "max_width": 0.0,
            "tcp_offset": round(float(origin[2]), 4)}


def cached(key: str, namespace: str, build: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """`build()`'s JSON result, kept in `$HYLO_ASSETS/<namespace>/index-<key>.json` while the "model" file it names
    exists: lets `spec()` skip importing the simulator and re-measuring the hand after the first export."""
    path = cache_root() / namespace / f"index-{key}.json"
    try:
        info = json.loads(path.read_text())
        if Path(info["model"]).is_file():
            return info
    except (OSError, ValueError, KeyError):
        pass
    info = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(info))
    os.replace(tmp, path)
    return info


__all__ = ["cache_root", "cached", "export_model", "face_hand_geometry", "parallel_hand_geometry",
           "point_hand_geometry", "rewrite_model"]
