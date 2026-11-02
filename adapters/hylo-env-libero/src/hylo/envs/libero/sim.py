# hylo: worker-side (runs in the LIBERO simulator venv: robosuite 1.4, mujoco)
"""Simulator side of the LIBERO robot contract: the Panda's MJCF exported from robosuite into the asset cache for
hylo-manip's kinematics, the TCP measured on it, and instance segmentation with labels from the BDDL objects."""

from __future__ import annotations

import hashlib
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .robot import EE_BODY, FINGER_JOINTS, FINGER_TRAVEL, PADS, PARTS, object_label

EXPORT_VERSION = "1"


def cache_root() -> Path:
    return Path(os.environ.get("HYLO_ASSETS") or Path.home() / ".cache" / "hylo" / "assets").expanduser()


def _robot_xml() -> str:
    """robosuite's MountedPanda with its PandaGripper as one MJCF string (the arm LIBERO simulates; the mount is
    left out: it never moves and sits below the support surface)."""
    import libero.libero.envs  # noqa: F401 - registers LIBERO's robots
    from libero.libero.envs.robots import MountedPanda
    from robosuite.models.grippers import gripper_factory

    robot = MountedPanda(idn=0)
    robot.add_gripper(gripper_factory("PandaGripper", idn=0))
    return robot.get_xml()


def _rewrite(xml: str, assets: dict[str, Path]) -> str:
    """Asset files renamed into assets/, and finger_joint2 flipped (axis and range) so both fingers open towards the
    upper end of their range, which is how hylo-manip maps a hand command to finger joints."""
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
            name = f"{src.parent.parent.name}_{src.parent.name}_{src.name}" if src.parent.name == "meshes" else \
                f"{src.parent.name}_{src.name}"
            if name in assets and assets[name] != src:
                name = f"{hashlib.sha256(str(src).encode()).hexdigest()[:8]}_{src.name}"
            assets[name] = src
            node.set("file", f"assets/{name}")
    for j in root.iter("joint"):
        if j.get("name") == FINGER_JOINTS[1]:
            axis = [float(v) for v in (j.get("axis") or "0 1 0").split()]
            lo, hi = [float(v) for v in (j.get("range") or f"-{FINGER_TRAVEL} 0").split()]
            j.set("axis", " ".join(f"{-v:g}" for v in axis))
            j.set("range", f"{-hi:g} {-lo:g}")
    for tag in ("actuator", "sensor", "tendon", "equality"):   # kinematics only: no drives, nothing that refers to
        for node in root.findall(tag):                          # the flipped finger by its old sign
            root.remove(node)
    return ET.tostring(root, encoding="unicode")


def export_panda_model() -> Path:
    """The Panda MJCF for hylo-manip's kinematics, written once into the asset cache (content-addressed:
    `$HYLO_ASSETS/libero/panda/<digest>/panda.xml` with its meshes in assets/). Returns the XML path."""
    assets: dict[str, Path] = {}
    xml = _rewrite(_robot_xml(), assets)
    h = hashlib.sha256((EXPORT_VERSION + xml).encode())
    for name in sorted(assets):
        h.update(name.encode())
        h.update(assets[name].read_bytes())
    folder = cache_root() / "libero" / "panda" / h.hexdigest()[:16]
    path = folder / "panda.xml"
    if path.is_file():
        return path
    tmp = folder.with_name(folder.name + f".tmp{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "assets").mkdir(parents=True)
    for name, src in assets.items():
        shutil.copyfile(src, tmp / "assets" / name)
    (tmp / "panda.xml").write_text(xml)
    (tmp / "SOURCE.txt").write_text("Exported by hylo.envs.libero.sim from robosuite 1.4 (MIT): robots/panda/robot.xml "
                                    "+ grippers/panda_gripper.xml, as LIBERO's MountedPanda builds it. finger_joint2 "
                                    "flipped (axis, range); actuators removed. Derived data, not redistributed.\n")
    try:
        os.replace(tmp, folder)
    except OSError:                                     # another worker wrote it first
        shutil.rmtree(tmp, ignore_errors=True)
    return path


def hand_geometry(model_path: str | Path) -> dict[str, Any]:
    """TCP in the ee frame (`gripper0_right_gripper`) and the opening, measured on the exported model: origin
    midway between the two finger pads (fully open), +x along the hand's approach axis (its +z), +y the closing
    axis (from pad 2 towards pad 1), +z = x cross y. max_width: the gap between the pad faces when open."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    for j in FINGER_JOINTS:
        d.qpos[m.jnt_qposadr[m.joint(j).id]] = FINGER_TRAVEL
    mujoco.mj_kinematics(m, d)
    ee = m.body(EE_BODY).id
    R_ee, p_ee = d.xmat[ee].reshape(3, 3), d.xpos[ee]
    pads = [R_ee.T @ (d.geom_xpos[m.geom(g).id] - p_ee) for g in PADS]
    origin = (pads[0] + pads[1]) / 2
    x = np.array([0.0, 0.0, 1.0])
    y = pads[0] - pads[1]
    y -= (y @ x) * x
    gap = float(np.linalg.norm(y))
    y /= gap
    z = np.cross(x, y)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, origin
    half = float(m.geom_size[m.geom(PADS[0]).id][1])      # pad half thickness along the closing axis
    return {"tcp_in_ee": np.round(T, 6).tolist(), "max_width": round(gap - 2 * half, 4),
            "tcp_offset": round(float(origin[2]), 4)}


class Segments:
    """Geom -> instance id lookup for one LIBERO scene: 1 support (table, or the floor in floor scenes), 2 robot
    (arm, gripper, mount), then one id per BDDL object and fixture, then one per articulated fixture part; walls and
    other scenery are 0 (background)."""

    def __init__(self, m: Any, objects: Sequence[str]) -> None:
        import mujoco

        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)]
        support = {b for b, n in enumerate(names) if n.endswith("table")}
        floor = not support
        if floor:                                        # floor scenes: the ground is the support
            support = {b for b, n in enumerate(names) if n == "floor"}
        self.labels: dict[int, str] = {1: "floor" if floor else "table", 2: "robot"}
        body_id = np.zeros(m.nbody, dtype=np.int32)
        for b, n in enumerate(names):
            if n.startswith(("robot0_", "gripper0_", "mount0_")):
                body_id[b] = 2
            elif b in support:
                body_id[b] = 1
        nxt = 3
        for obj in sorted(objects, key=len, reverse=True):          # longest first: "plate_1" vs "plate_10"
            own = [b for b, n in enumerate(names) if n.startswith(obj + "_") and body_id[b] == 0]
            if not own:
                continue
            label = object_label(obj)
            self.labels[nxt] = label
            body_id[own] = nxt
            nxt += 1
            for b in own:
                suffix = names[b][len(obj) + 1:]
                if suffix in PARTS:
                    self.labels[nxt] = PARTS[suffix].format(o=label)
                    # the part and every body below it
                    sub = {b}
                    for c in range(b + 1, m.nbody):
                        if m.body_parentid[c] in sub:
                            sub.add(c)
                    body_id[list(sub)] = nxt
                    nxt += 1
        self.geom = body_id[m.geom_bodyid].copy()

    def render(self, sim: Any, cam_id: int, width: int, height: int) -> np.ndarray:
        """(H, W) instance ids seen by a camera, rows bottom-up like MuJoCo's buffers (robosuite's own segmentation
        read-out overflows uint8 arithmetic under numpy 2, so the id colours are decoded here)."""
        import mujoco

        ctx = sim._render_context_offscreen
        ctx.render(width, height, camera_id=cam_id, segmentation=True)
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb=rgb, depth=None, viewport=mujoco.MjrRect(0, 0, width, height), con=ctx.con)
        code = rgb.astype(np.int32)
        idx = code[..., 0] + (code[..., 1] << 8) + (code[..., 2] << 16)        # scene geom index + 1, 0: none
        n = ctx.scn.ngeom
        lut = np.zeros(n + 1, dtype=np.int32)
        geom = int(mujoco.mjtObj.mjOBJ_GEOM)
        for i in range(n):
            g = ctx.scn.geoms[i]
            if g.segid >= 0 and g.objtype == geom and 0 <= g.objid < len(self.geom):
                lut[g.segid + 1] = self.geom[g.objid]
        return lut[np.where(idx <= n, idx, 0)]


__all__ = ["Segments", "cache_root", "export_panda_model", "hand_geometry"]
