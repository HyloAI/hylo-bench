"""MuJoCo Franka Panda at a table: the manipulation environment of RFC-0006 part 3 (`mujoco.panda_table`).

The arm and its parallel hand are the MuJoCo Menagerie model (Apache-2.0, fetched at a pinned commit into the asset
cache on first use, see `assets.py`); the table, the objects and the cameras are ours. One model holds a pool of
objects (blocks, cylinders, plates, bowls, trays, a large box, zones, numbered buttons); each episode activates the
ones its task needs and parks the rest inside the table, so the model never changes shape and snapshots are cheap.

Control: joint-position targets at `control_hz` (20 Hz; 25 physics steps of 2 ms underneath). Action = 7 joint
targets (rad) + 1 hand command in [0, 1] (0 closed, 1 open). Observation layout follows `hylo.core.robot`:
`q.arm`, `hand.arm` (measured, command units), RGB-D images per camera, `<cam>.seg` with `segmentation=True`, and
camera intrinsics and poses in `extra["cameras"]`. With `privileged_obs=True` the observation also carries object
poses and the task goal in `extra["privileged"]` (for the oracle agent; a privileged run is never ranked with an
unprivileged one because env params are part of the experiment digest).

Rendering needs an OpenGL backend: MUJOCO_GL=egl, or on CPU MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa with
LD_LIBRARY_PATH pointing at an OSMesa build (bench/scripts/install-osmesa-user.sh). `cameras=[]` needs none.
"""

from __future__ import annotations

import json
import math
import os
import struct
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from hylo.core.robot import ArmSpec, CameraSpec, HandSpec, RobotSpec
from hylo.core.task import TaskInstance
from hylo.core.types import (Action, ActionSemantics, BoxSpace, EnvCapabilities, EnvSpec, Observation,
                             ObservationSpec, StepResult)

from .assets import panda_assets, panda_assets_dir

# -- robot ------------------------------------------------------------------------------------------------------
PREFIX = "panda/"                                  # names inside the scene; ArmSpec names are panda.xml's own
JOINTS = tuple(f"joint{i}" for i in range(1, 8))
LOWER = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973)
UPPER = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973)
LIMIT_MARGIN = 0.02                                # ArmSpec limits sit this far inside the model's
HOME = (0.0, -0.9, 0.0, -2.7, 0.0, 1.9, 0.7854)    # hand raised above the base, clear of the front view
BASE_POS = (-0.5, 0.0, 0.0)                        # link0 on the table top
TCP_OFFSET = 0.1029                                # hand frame to the centre of the finger pads, along hand +z
FINGER_TRAVEL = 0.04                               # each finger, metres, at command 1.0
HAND_CTRL_MAX = 255.0                              # Menagerie actuator8 range
GRIP_GAIN = 5.0                                    # scale of the finger servo gains
# TCP in the hand frame: +x approach = hand +z, +y closing = hand +y, +z = x cross y = hand -x
TCP_IN_EE = [[0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, TCP_OFFSET], [0.0, 0.0, 0.0, 1.0]]

# -- scene ------------------------------------------------------------------------------------------------------
TABLE_TOP = 0.0
TABLE_X, TABLE_Y = (-0.65, 0.45), (-0.65, 0.65)
WORKSPACE = ((-0.17, 0.14), (-0.27, 0.27))         # where object centres are placed (x, y in metres)
PARK = (-0.1, 0.0, -0.4)                           # unused objects wait inside the table body, out of every view
CONTROL_DT_PHYS = 0.002
SETTLE_S = 0.2
MESH_CELL = 0.009                                  # grid of the simplified visual meshes (m)

BLOCK_HALF = 0.0225
CYL_R, CYL_HALF = 0.021, 0.03
PLATE_R, PLATE_HALF = 0.075, 0.006
BOWL_BASE_R, BOWL_R0, BOWL_R1, BOWL_H, BOWL_SEGS = 0.06, 0.068, 0.085, 0.05, 14
TRAY_IN, TRAY_WALL, TRAY_H, TRAY_BASE = 0.055, 0.005, 0.03, 0.006
BIG_HALF = (0.055, 0.055, 0.03)
ZONE_HALF = 0.085
BTN_HOUSE, BTN_HOUSE_H, BTN_CAP, BTN_CAP_H, BTN_TRAVEL = 0.04, 0.01, 0.034, 0.006, 0.008
BTN_PRESSED = 0.005                                # cap displacement that counts as a press

POOL: dict[str, int] = {"block": 6, "cyl": 4, "plate": 2, "bowl": 2, "tray": 3, "bigbox": 1, "zone": 2, "button": 4}
FREE_KINDS = ("block", "cyl", "bigbox")            # objects with a free joint; the rest are fixtures (mocap)
POOL_NAMES = tuple(f"{k}{i}" for k, n in POOL.items() for i in range(n))

COLOURS: dict[str, tuple[float, float, float]] = {
    "red": (0.85, 0.12, 0.10), "green": (0.15, 0.65, 0.20), "blue": (0.15, 0.35, 0.90),
    "yellow": (0.95, 0.82, 0.10), "purple": (0.55, 0.22, 0.75), "orange": (0.97, 0.50, 0.08),
}
FIXTURE_COLOURS: dict[str, tuple[float, float, float]] = {
    "white": (0.93, 0.93, 0.91), "pink": (0.95, 0.62, 0.72), "cyan": (0.35, 0.80, 0.85), "grey": (0.55, 0.56, 0.58),
}
ALL_COLOURS = {**COLOURS, **FIXTURE_COLOURS}
CAP_RGB, DIGIT_RGB = (0.94, 0.94, 0.92), (0.06, 0.06, 0.07)

# seven-segment digits on the button caps (segment: centre (right, up) and half extents (right, up), metres)
_DH, _DW, _DT = 0.05, 0.028, 0.008
SEGMENTS = {
    "a": ((0, _DH / 2), (_DW / 2, _DT / 2)), "g": ((0, 0), (_DW / 2, _DT / 2)), "d": ((0, -_DH / 2), (_DW / 2, _DT / 2)),
    "b": ((_DW / 2, _DH / 4), (_DT / 2, _DH / 4)), "c": ((_DW / 2, -_DH / 4), (_DT / 2, _DH / 4)),
    "e": ((-_DW / 2, -_DH / 4), (_DT / 2, _DH / 4)), "f": ((-_DW / 2, _DH / 4), (_DT / 2, _DH / 4)),
}
SEG_HIDDEN_Z = -0.002                              # never 0: a geom at its body's origin ignores geom_pos edits
DIGITS = {0: "abcdef", 1: "bc", 2: "abged", 3: "abgcd", 4: "fgbc", 5: "afgcd", 6: "afgedc", 7: "abc", 8: "abcdefg",
          9: "abcdfg"}

# -- cameras: name -> (position, look-at point, fovy degrees); the wrist camera rides on the hand -----------------
CAMERAS = {
    "front": ((0.40, 0.0, 0.56), (-0.07, 0.0, -0.02), 50.0),
    "side": ((-0.04, -0.78, 0.58), (-0.04, 0.0, -0.02), 55.0),
}
WRIST_CAM = ((0.05, 0.0, 0.07), (-0.4226, 0.0, 0.9063), (0.9063, 0.0, 0.4226), 70.0)  # pos, forward, up (hand frame), fovy
CAMERA_NAMES = ("front", "side", "wrist")


def _xyaxes(forward: Sequence[float], up: Sequence[float]) -> str:
    f = np.asarray(forward, float) / np.linalg.norm(forward)
    x = np.cross(f, np.asarray(up, float))
    x /= np.linalg.norm(x)
    y = np.cross(-f, x)
    return " ".join(f"{v:.6f}" for v in (*x, *y))


def _f(v: Sequence[float]) -> str:
    return " ".join(f"{x:.5g}" for x in v)


def _items_xml() -> str:
    out = []
    park = _f(PARK)
    for name in POOL_NAMES:
        kind = name.rstrip("0123456789")
        g = f'name="{name}/g'
        if kind == "block":
            out.append(f'<body name="{name}" pos="{park}"><freejoint/>'
                       f'<geom {g}" type="box" size="{BLOCK_HALF} {BLOCK_HALF} {BLOCK_HALF}" mass="0.06" class="obj"/></body>')
        elif kind == "cyl":
            out.append(f'<body name="{name}" pos="{park}"><freejoint/>'
                       f'<geom {g}" type="cylinder" size="{CYL_R} {CYL_HALF}" mass="0.06" class="obj"/></body>')
        elif kind == "bigbox":
            out.append(f'<body name="{name}" pos="{park}"><freejoint/>'
                       f'<geom {g}" type="box" size="{_f(BIG_HALF)}" mass="0.3" class="obj"/></body>')
        elif kind == "plate":
            out.append(f'<body name="{name}" mocap="true" pos="{park}">'
                       f'<geom {g}" type="cylinder" size="{PLATE_R} {PLATE_HALF}" pos="0 0 {PLATE_HALF}" class="fix"/></body>')
        elif kind == "zone":
            out.append(f'<body name="{name}" mocap="true" pos="{park}">'
                       f'<geom {g}" type="box" size="{ZONE_HALF} {ZONE_HALF} 0.0006" pos="0 0 0.0007" '
                       f'contype="0" conaffinity="0" class="fix"/></body>')
        elif kind == "tray":
            w, h, b = TRAY_WALL, TRAY_H / 2, TRAY_BASE / 2
            o = TRAY_IN + w
            walls = [(o, 0, w, o + w), (-o, 0, w, o + w), (0, o, TRAY_IN, w), (0, -o, TRAY_IN, w)]
            geoms = [f'<geom {g}0" type="box" size="{o + w} {o + w} {b}" pos="0 0 {b}" class="fix"/>']
            geoms += [f'<geom {g}{i + 1}" type="box" size="{hx} {hy} {h}" pos="{x} {y} {h}" class="fix"/>'
                      for i, (x, y, hx, hy) in enumerate(walls)]
            out.append(f'<body name="{name}" mocap="true" pos="{park}">{"".join(geoms)}</body>')
        elif kind == "bowl":
            geoms = [f'<geom {g}0" type="cylinder" size="{BOWL_BASE_R + 0.012} 0.003" pos="0 0 0.003" class="fix"/>']
            tilt = math.atan2(BOWL_R1 - BOWL_R0, BOWL_H)
            rm, seg = (BOWL_R0 + BOWL_R1) / 2, 2 * math.pi * BOWL_R1 / BOWL_SEGS / 2 + 0.002
            for i in range(BOWL_SEGS):
                a = 2 * math.pi * i / BOWL_SEGS
                # wall panel: outward normal along (cos a, sin a), leaning out by `tilt`
                q = _quat_mul(_quat_z(a), _quat_y(tilt))
                geoms.append(f'<geom {g}{i + 1}" type="box" size="0.004 {seg:.4f} {BOWL_H / 2 + 0.003}" '
                             f'pos="{rm * math.cos(a):.5f} {rm * math.sin(a):.5f} {BOWL_H / 2 + 0.003}" '
                             f'quat="{_f(q)}" class="fix"/>')
            out.append(f'<body name="{name}" mocap="true" pos="{park}">{"".join(geoms)}</body>')
        elif kind == "button":
            segs = "".join(
                f'<geom {g}s{s}" type="box" size="{hu} {hr} 0.0005" pos="{-cu} {cr} {SEG_HIDDEN_Z}" rgba="{_f(DIGIT_RGB)} 1" '
                f'contype="0" conaffinity="0" mass="0"/>' for s, ((cr, cu), (hr, hu)) in SEGMENTS.items())
            top = 2 * BTN_HOUSE_H
            out.append(
                f'<body name="{name}" mocap="true" pos="{park}">'
                f'<geom {g}h" type="box" size="{BTN_HOUSE} {BTN_HOUSE} {BTN_HOUSE_H}" pos="0 0 {BTN_HOUSE_H}" '
                f'rgba="0.22 0.23 0.25 1" class="fix"/>'
                f'<body name="{name}/cap" pos="0 0 {top + BTN_CAP_H}">'
                f'<joint name="{name}/press" type="slide" axis="0 0 1" range="-{BTN_TRAVEL} 0" stiffness="300" '
                f'springref="0.006" damping="3"/>'
                f'<geom {g}c" type="box" size="{BTN_CAP} {BTN_CAP} {BTN_CAP_H}" mass="0.02" rgba="{_f(CAP_RGB)} 1" '
                f'class="obj"/>{segs}</body></body>')
    return "\n      ".join(out)


def _quat_z(a: float) -> tuple[float, float, float, float]:
    return (math.cos(a / 2), 0.0, 0.0, math.sin(a / 2))


def _quat_y(a: float) -> tuple[float, float, float, float]:
    return (math.cos(a / 2), 0.0, math.sin(a / 2), 0.0)


def _quat_mul(p: Sequence[float], q: Sequence[float]) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = p
    w2, x2, y2, z2 = q
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def scene_xml(width: int, height: int, items: str | None = None) -> str:
    """The table, cameras, lights and defaults; `items` replaces the object pool (mujoco.panda_scene passes "")."""
    cams = "\n    ".join(f'<camera name="{n}" pos="{_f(p)}" xyaxes="{_xyaxes(np.subtract(t, p), (0, 0, 1))}" fovy="{fov}"/>'
                         for n, (p, t, fov) in CAMERAS.items())
    tx, ty = (TABLE_X[0] + TABLE_X[1]) / 2, (TABLE_Y[0] + TABLE_Y[1]) / 2
    hx, hy = (TABLE_X[1] - TABLE_X[0]) / 2, (TABLE_Y[1] - TABLE_Y[0]) / 2
    return f"""
<mujoco model="panda_table">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{CONTROL_DT_PHYS}" integrator="implicitfast" cone="elliptic" impratio="5">
    <flag multiccd="enable"/>
  </option>
  <size memory="20M"/>
  <visual>
    <global offwidth="{max(width, 64)}" offheight="{max(height, 64)}"/>
    <headlight ambient="0.32 0.32 0.32" diffuse="0.45 0.45 0.45" specular="0.05 0.05 0.05"/>
    <map znear="0.008" zfar="3"/>
    <quality shadowsize="0"/>
  </visual>
  <default>
    <default class="obj"><geom condim="4" friction="1.0 0.01 0.001" solref="0.005 1" rgba="0.6 0.6 0.6 1"/></default>
    <default class="fix"><geom condim="3" friction="0.8 0.01 0.001" rgba="0.8 0.8 0.8 1"/></default>
  </default>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.82 0.87 0.92" rgb2="0.35 0.40 0.47" width="256" height="1536"/>
    <texture name="wood" type="2d" builtin="flat" rgb1="0.78 0.70 0.58" rgb2="0.72 0.64 0.52" mark="random" random="0.02" width="256" height="256"/>
    <material name="table" texture="wood" texrepeat="4 4" reflectance="0.02" specular="0.1"/>
    <material name="floor" rgba="0.50 0.52 0.55 1"/>
  </asset>
  <worldbody>
    <light name="key" directional="true" pos="0.5 -0.6 2.5" dir="-0.2 0.3 -1" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1" castshadow="false"/>
    {cams}
    <geom name="floor" type="plane" size="4 4 0.1" pos="0 0 -0.75" material="floor" contype="0" conaffinity="0"/>
    <geom name="table" type="box" size="{hx} {hy} 0.375" pos="{tx} {ty} {TABLE_TOP - 0.375}" material="table"
          friction="0.9 0.01 0.001"/>
      {_items_xml() if items is None else items}
  </worldbody>
</mujoco>
"""


def _read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vs: list[list[float]] = []
    fs: list[list[int]] = []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            vs.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
            idx = [i - 1 if i > 0 else len(vs) + i for i in idx]
            fs += [[idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1)]
    return np.asarray(vs, dtype=np.float64), np.asarray(fs, dtype=np.int64)


def _cluster(v: np.ndarray, f: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Vertex-clustering simplification: vertices in one grid cell merge into their mean; collapsed and repeated
    triangles are dropped (winding kept)."""
    _, inv = np.unique(np.floor(v / cell).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    n = int(inv.max()) + 1
    nv = np.zeros((n, 3))
    np.add.at(nv, inv, v)
    nv /= np.bincount(inv, minlength=n)[:, None]
    nf = inv[f]
    nf = nf[(nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])]
    keep = np.sort(np.unique(np.sort(nf, axis=1), axis=0, return_index=True)[1])
    return nv, nf[keep]


def _simplified_meshes(folder: Path, cell: float, min_faces: int = 800) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Simplified copies of the large visual meshes, cached next to the assets (derived data, never the assets)."""
    out_path = folder.parent / f"{folder.name}.visual-{cell * 1000:.0f}mm.npz"
    if out_path.is_file():
        z = np.load(out_path)
        return {k[:-2]: (z[k], z[k[:-2] + ".f"]) for k in z.files if k.endswith(".v")}
    meshes = {}
    for obj in sorted((folder / "assets").glob("*.obj")):
        v, f = _read_obj(obj)
        if len(f) >= min_faces:
            meshes[obj.name] = _cluster(v, f, cell)
    tmp = out_path.with_name(out_path.name + f".{os.getpid()}.npz")
    np.savez(tmp, **{f"{k}.v": v for k, (v, _) in meshes.items()}, **{f"{k}.f": f for k, (_, f) in meshes.items()})
    os.replace(tmp, out_path)
    return meshes


def build_model(width: int = 320, height: int = 240, mesh_detail: str = "fast"):
    """Compile the scene with the Menagerie Panda attached at BASE_POS (names prefixed with PREFIX). mesh_detail
    "fast" draws the arm with simplified visual meshes (about 6x fewer triangles; collision geometry unchanged),
    "full" with the original ones."""
    import mujoco

    spec = mujoco.MjSpec.from_string(scene_xml(width, height))
    attach_panda(spec, mesh_detail)
    for name in POOL_NAMES:                    # button caps never collide with their own housing
        if name.startswith("button"):
            ex = spec.add_exclude()
            ex.bodyname1, ex.bodyname2 = name, f"{name}/cap"
    return spec.compile()


def attach_panda(spec: Any, mesh_detail: str = "fast") -> None:
    """Attach the Menagerie Panda (wrist camera, gravity compensation, finger pads, stiffer grip) to an MjSpec at
    BASE_POS, names prefixed with PREFIX (shared with mujoco.panda_scene)."""
    import mujoco

    folder = panda_assets()
    panda = mujoco.MjSpec.from_file(str(folder / "panda.xml"))
    if mesh_detail == "fast":
        simple = _simplified_meshes(folder, MESH_CELL)
        for mesh in panda.meshes:
            vf = simple.get(Path(mesh.file).name)
            if vf is not None:
                mesh.file = ""
                mesh.uservert = vf[0].astype(np.float32).ravel().tolist()
                mesh.userface = vf[1].astype(np.int32).ravel().tolist()
    elif mesh_detail != "full":
        raise ValueError(f"mesh_detail must be 'fast' or 'full', got {mesh_detail!r}")
    hand = panda.body("hand")
    pos, fwd, up, fov = WRIST_CAM
    cam = hand.add_camera(name="wrist", pos=list(pos), fovy=fov)
    cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    cam.alt.xyaxes = [float(v) for v in _xyaxes(fwd, up).split()]
    for body in panda.bodies:
        body.gravcomp = 1.0                    # the real arm compensates gravity; position servos then track well
    for finger in ("left_finger", "right_finger"):
        # one flat pad face over the Menagerie's small fingertip boxes (0.5 mm proud of them, collision only):
        # round objects otherwise settle in the gap between those boxes and get squeezed out sideways; stiff
        # like the objects, so friction holds instead of creeping under the hand's accelerations
        panda.body(finger).add_geom(name=f"{finger}_pad", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.0088, 0.0012, 0.0105],
                                    pos=[0.0, 0.0007, 0.0445], group=3, contype=1, conaffinity=1, condim=4,
                                    friction=[1.0, 0.01, 0.001], solref=[0.005, 1.0])
    grip = panda.actuator("actuator8")         # stiffer finger servo: about 10 N on a 4 cm object instead of 2 N
    grip.gainprm[0] *= GRIP_GAIN
    grip.biasprm[1] *= GRIP_GAIN
    grip.biasprm[2] *= GRIP_GAIN
    frame = spec.worldbody.add_frame(pos=list(BASE_POS))
    frame.attach_body(panda.body("link0"), PREFIX, "")


# -- frames -----------------------------------------------------------------------------------------------------
def _pose(R: np.ndarray, p: np.ndarray) -> list[list[float]]:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T.tolist()


def _yaw_of(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return float(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class _Cameras:
    """One offscreen GL context; each camera rendered once for RGB and depth (one read), once more for ids."""

    def __init__(self, model: Any, width: int, height: int, segmentation: bool) -> None:
        import mujoco

        self._mj = mujoco
        self.w, self.h = width, height
        try:
            self.gl = mujoco.GLContext(width, height)
            self.gl.make_current()
        except Exception as exc:  # noqa: BLE001 - re-raised with what to do about it
            raise RuntimeError("mujoco.panda_table cameras need an OpenGL backend: set MUJOCO_GL=egl, or "
                               "MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa (and LD_LIBRARY_PATH to OSMesa), "
                               f"or pass cameras=[] ({exc})") from exc
        self.ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.ctx)
        self.scene = mujoco.MjvScene(model, maxgeom=3000)
        self.opt = mujoco.MjvOption()
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.rect = mujoco.MjrRect(0, 0, width, height)
        self.segmentation = segmentation
        ext = model.stat.extent
        self.near, self.far = model.vis.map.znear * ext, model.vis.map.zfar * ext
        self._rgb = np.empty((height, width, 3), dtype=np.uint8)
        self._depth = np.empty((height, width), dtype=np.float32)

    def render(self, model: Any, data: Any, cam_id: int, geom_instance: np.ndarray) -> tuple[np.ndarray, ...]:
        mj = self._mj
        self.gl.make_current()
        self.cam.fixedcamid = cam_id
        mj.mjv_updateScene(model, data, self.opt, None, self.cam, mj.mjtCatBit.mjCAT_ALL, self.scene)
        mj.mjr_render(self.rect, self.scene, self.ctx)
        mj.mjr_readPixels(self._rgb, self._depth, self.rect, self.ctx)
        rgb = np.flipud(self._rgb).copy()
        z = np.flipud(self._depth)
        depth = (self.near / (1.0 - z * (1.0 - self.near / self.far))).astype(np.float32)
        depth[z >= 1.0 - 1e-7] = 0.0                       # nothing hit within the far plane: no return
        if not self.segmentation:
            return rgb, depth, None
        flags = self.scene.flags
        flags[mj.mjtRndFlag.mjRND_SEGMENT] = flags[mj.mjtRndFlag.mjRND_IDCOLOR] = 1
        try:
            mj.mjr_render(self.rect, self.scene, self.ctx)
            mj.mjr_readPixels(self._rgb, None, self.rect, self.ctx)
        finally:
            flags[mj.mjtRndFlag.mjRND_SEGMENT] = flags[mj.mjtRndFlag.mjRND_IDCOLOR] = 0
        code = np.flipud(self._rgb).astype(np.int32)
        idx = code[..., 0] + (code[..., 1] << 8) + (code[..., 2] << 16) - 1   # scene geom index, -1: background
        n = self.scene.ngeom
        lut = np.zeros(n + 1, dtype=np.int32)
        for i in range(n):
            g = self.scene.geoms[i]
            if g.objtype == mj.mjtObj.mjOBJ_GEOM and 0 <= g.objid < len(geom_instance):
                lut[i + 1] = geom_instance[g.objid]
        seg = lut[np.clip(idx + 1, 0, n)]
        return rgb, depth, seg

    def close(self) -> None:
        try:
            self.gl.make_current()
            self.ctx.free()
            self.gl.free()
        except Exception:  # noqa: BLE001 - closing is best effort
            pass


class PandaTableEnv:
    """`mujoco.panda_table`. Params: cameras (subset of front, side, wrist; [] for none), width, height (pixels,
    default 320x240), segmentation (instance ids per camera), privileged_obs (object poses and the goal in
    extra["privileged"], for the oracle), control_hz, depth (depth images), mesh_detail ("fast": simplified visual
    meshes of the arm, about 4x faster to render on CPU; "full": the original meshes)."""

    def __init__(self, cameras: Sequence[str] | None = CAMERA_NAMES, width: int = 320, height: int = 240,
                 segmentation: bool = False, privileged_obs: bool = False, control_hz: float = 20.0,
                 depth: bool = True, mesh_detail: str = "fast") -> None:
        cams = tuple(CAMERA_NAMES if cameras is None else cameras)
        unknown = [c for c in cams if c not in CAMERA_NAMES]
        if unknown:
            raise ValueError(f"unknown cameras {unknown}; available: {list(CAMERA_NAMES)}")
        self.camera_names, self.width, self.height = cams, int(width), int(height)
        self.segmentation, self.privileged_obs, self.depth = bool(segmentation), bool(privileged_obs), bool(depth)
        self.control_hz = float(control_hz)
        if mesh_detail not in ("fast", "full"):
            raise ValueError(f"mesh_detail must be 'fast' or 'full', got {mesh_detail!r}")
        self.mesh_detail = mesh_detail
        self._model = None
        self._cams: _Cameras | None = None
        self.task = None
        self.ep: dict[str, Any] = {}

    # -- spec (cheap: no simulator, no download) ----------------------------------------------------------------
    def robot_spec(self) -> RobotSpec:
        lo = tuple(round(v + LIMIT_MARGIN, 4) for v in LOWER)
        hi = tuple(round(v - LIMIT_MARGIN, 4) for v in UPPER)
        hand = HandSpec(kind="parallel", tcp_in_ee=TCP_IN_EE, open=(1.0,), closed=(0.0,),
                        max_width=2 * FINGER_TRAVEL, joints=("finger_joint1", "finger_joint2"))
        arm = ArmSpec(name="arm", model=str(panda_assets_dir() / "panda.xml"), joints=JOINTS, ee_frame="hand",
                      base_in_world=_pose(np.eye(3), np.asarray(BASE_POS)), lower=lo, upper=hi, home=HOME, hand=hand,
                      action_slice=(0, 7), hand_slice=(7, 8))
        cams = tuple(CameraSpec(name=c, height=self.height, width=self.width, depth=self.depth,
                                mounted_on="arm" if c == "wrist" else None) for c in self.camera_names)
        (x0, x1), (y0, y1) = WORKSPACE
        notes = (
            f"Franka Panda (MuJoCo Menagerie) on a table; world frame: +z up, table top at z={TABLE_TOP}, arm base "
            f"(link0) at {list(BASE_POS)}, +x points from the base towards the front camera. Objects start with "
            f"centres in x [{x0}, {x1}], y [{y0}, {y1}] (metres); the arm reaches top-down grasps over that area. "
            f"Blocks are {2 * BLOCK_HALF:.3f} m cubes; the large box ({2 * BIG_HALF[0]:.2f} m) is too wide to grasp "
            f"(max opening {2 * FINGER_TRAVEL:.2f} m) and must be pushed. Buttons show a seven-segment digit on the "
            f"cap (upright as seen from the front camera) and register a press when pushed down "
            f"{BTN_PRESSED * 1000:.0f} mm of {BTN_TRAVEL * 1000:.0f} mm. Action: 7 joint-position targets (rad) + "
            f"hand command (0 closed, 1 open) at {self.control_hz:g} Hz; hand.arm is the measured opening in command "
            f"units (about {BLOCK_HALF / FINGER_TRAVEL:.2f} while holding a block). TCP: between the finger pads, "
            f"{TCP_OFFSET} m along the hand's z axis; +x approach, +y closing. Home keeps the hand above the base, "
            f"out of the front camera's view of the workspace. Success needs objects at rest and released.")
        return RobotSpec(arms=(arm,), cameras=cams, table_height=TABLE_TOP, notes=notes)

    def spec(self) -> EnvSpec:
        lo = (*LOWER, 0.0)
        hi = (*UPPER, 1.0)
        cams = {c: (self.height, self.width, 3) for c in self.camera_names}
        return EnvSpec(
            name="mujoco.panda_table", version="0.1",
            observation=ObservationSpec(state={"q.arm": (7,), "hand.arm": (1,)}, cameras=cams,
                                        units={"q.arm": "rad", "hand.arm": "command (0 closed, 1 open)"}),
            action=BoxSpace(shape=(8,), low=lo, high=hi,
                            semantics=ActionSemantics(control_mode="joint_pos", frame="base", gripper="0_close_1_open",
                                                      dim_labels=(*JOINTS, "hand"))),
            control_hz=self.control_hz,
            capabilities=EnvCapabilities(seedable=True, snapshot=True, render=True, privileged_state=True),
            robot=self.robot_spec())

    # -- simulator ----------------------------------------------------------------------------------------------
    def _build(self) -> None:
        import mujoco

        self._mj = mujoco
        m = build_model(self.width, self.height, self.mesh_detail)
        self._bind_robot(m)
        self.items: dict[str, dict[str, Any]] = {}
        for name in POOL_NAMES:
            kind = name.rstrip("0123456789")
            b = m.body(name)
            geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b.id or
                     (kind == "button" and m.geom_bodyid[g] == m.body(f"{name}/cap").id)]
            info: dict[str, Any] = {"kind": kind, "body": b.id, "geoms": geoms,
                                    "contype": m.geom_contype[geoms].copy(), "conaffinity": m.geom_conaffinity[geoms].copy()}
            if kind in FREE_KINDS:
                j = m.body_jntadr[b.id]
                info["qadr"], info["vadr"] = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
            else:
                info["mocap"] = int(m.body_mocapid[b.id])
            if kind == "button":
                j = m.joint(f"{name}/press").id
                info["press_qadr"] = int(m.jnt_qposadr[j])
                info["cap"] = m.body(f"{name}/cap").id
                info["segs"] = {s: m.geom(f"{name}/gs{s}").id for s in SEGMENTS}
                info["seg_z"] = BTN_CAP_H + 0.0004
            self.items[name] = info
        # instance ids for segmentation: 1 table, 2 robot, then one per pool item
        self.instance = {"table": 1, "robot": 2, **{n: 3 + i for i, n in enumerate(POOL_NAMES)}}
        gi = np.zeros(m.ngeom, dtype=np.int32)
        robot_bodies = {b for b in range(m.nbody) if m.body(b).name.startswith(PREFIX)}
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in robot_bodies:
                gi[g] = 2
        gi[m.geom("table").id] = 1
        for name, info in self.items.items():
            gi[info["geoms"]] = self.instance[name]
        self.geom_instance = gi
        self._sig = mujoco.mjtState.mjSTATE_INTEGRATION
        self._warn = 0

    def _bind_robot(self, m: Any) -> None:
        """Model, data and the arm's indices (shared with mujoco.panda_scene)."""
        import mujoco

        self._model, self.data = m, mujoco.MjData(m)
        self.n_substeps = max(1, round(1.0 / (self.control_hz * m.opt.timestep)))
        jid = [m.joint(PREFIX + j).id for j in JOINTS]
        self._qadr = np.array([m.jnt_qposadr[j] for j in jid])
        self._vadr = np.array([m.jnt_dofadr[j] for j in jid])
        self._finger_qadr = np.array([m.jnt_qposadr[m.joint(PREFIX + f).id] for f in ("finger_joint1", "finger_joint2")])
        self._hand = m.body(PREFIX + "hand").id
        self._fingers = {m.body(PREFIX + f).id for f in ("left_finger", "right_finger")}
        self._cam_ids = {c: m.camera(c if c != "wrist" else PREFIX + "wrist").id for c in CAMERA_NAMES}

    @property
    def model(self):
        if self._model is None:
            self._build()
        return self._model

    # -- episode ------------------------------------------------------------------------------------------------
    def reset(self, task: TaskInstance) -> Observation:
        from .manip_tasks import sample_episode

        m = self.model
        mj = self._mj
        self.task = task
        rng = np.random.default_rng(task.seed)
        self.ep = sample_episode(task.spec.params, rng)
        mj.mj_resetData(m, self.data)
        self._apply_episode()
        d = self.data
        for name, it in self.ep["items"].items():
            self._place(name, it["pos"], it.get("yaw", 0.0))
        d.qpos[self._qadr] = HOME
        d.qpos[self._finger_qadr] = FINGER_TRAVEL
        d.ctrl[:7] = HOME
        d.ctrl[7] = HAND_CTRL_MAX
        mj.mj_forward(m, d)
        mj.mj_step(m, d, nstep=int(SETTLE_S / m.opt.timestep))
        d.time = 0.0
        self._warn = self._divergence_count()
        self.t = 0
        self.ep["pressed"] = []
        self.ep["latched"] = []
        return self._observe()

    def _apply_episode(self) -> None:
        """Model-side state of the episode (colours, digits, which pool items take part): re-applied by restore."""
        m = self._model
        active = self.ep["items"]
        for name, info in self.items.items():
            it = active.get(name)
            g = info["geoms"]
            if it is None:                              # parked: no contacts, no gravity, out of sight
                m.geom_contype[g] = 0
                m.geom_conaffinity[g] = 0
                m.body_gravcomp[info["body"]] = 1.0
                if info["kind"] == "button":
                    m.body_gravcomp[info["cap"]] = 1.0
                continue
            m.geom_contype[g] = info["contype"]
            m.geom_conaffinity[g] = info["conaffinity"]
            m.body_gravcomp[info["body"]] = 0.0
            rgb = ALL_COLOURS.get(it.get("colour", ""), None)
            if info["kind"] == "button":
                m.body_gravcomp[info["cap"]] = 0.0
                on = set(DIGITS[int(it["digit"])])
                for s, gid in info["segs"].items():
                    m.geom_pos[gid, 2] = info["seg_z"] if s in on else SEG_HIDDEN_Z   # unlit ones hide inside the cap
            elif rgb is not None:
                for gid in g:
                    m.geom_rgba[gid, :3] = rgb

    def _place(self, name: str, pos: Sequence[float], yaw: float) -> None:
        info, d = self.items[name], self.data
        q = np.array(_quat_z(yaw))
        if "qadr" in info:
            a = info["qadr"]
            d.qpos[a:a + 3], d.qpos[a + 3:a + 7] = pos, q
            d.qvel[info["vadr"]:info["vadr"] + 6] = 0.0
        else:
            d.mocap_pos[info["mocap"]], d.mocap_quat[info["mocap"]] = pos, q

    def _divergence_count(self) -> int:
        w = self._mj.mjtWarning
        return sum(int(self.data.warning[k].number) for k in (w.mjWARN_BADQPOS, w.mjWARN_BADQVEL, w.mjWARN_BADQACC))

    # -- step ---------------------------------------------------------------------------------------------------
    def step(self, action: Action) -> StepResult:
        from .manip_tasks import evaluate

        mj, m, d = self._mj, self._model, self.data
        a = np.asarray(action.data, dtype=np.float64).reshape(-1)
        d.ctrl[:7] = np.clip(a[:7], LOWER, UPPER)
        d.ctrl[7] = HAND_CTRL_MAX * float(np.clip(a[7], 0.0, 1.0))
        mj.mj_step(m, d, nstep=self.n_substeps)
        self.t += 1
        bad = self._divergence_count()
        if bad != self._warn:
            self._warn = bad
            return StepResult(observation=self._observe(), reward=None, terminated=False, truncated=True,
                              success=None, info={"sim_error": "mujoco divergence warning (BADQPOS/QVEL/QACC)"})
        self._update_buttons()
        success, fail, info = evaluate(self)
        if fail:
            info["fail_reason"] = fail
        return StepResult(observation=self._observe(), reward=1.0 if success else 0.0,
                          terminated=bool(success or fail), truncated=False, success=bool(success), info=info)

    def hold_action(self, last: np.ndarray | None) -> np.ndarray:
        if last is not None:
            return np.asarray(last, dtype=float).copy()
        return np.concatenate([self.data.ctrl[:7], [self.data.ctrl[7] / HAND_CTRL_MAX]])

    def _update_buttons(self) -> None:
        latched = set(self.ep["latched"])
        for name in self.ep["items"]:
            if not name.startswith("button"):
                continue
            depth = self.button_depth(name)
            if depth >= BTN_PRESSED and name not in latched:
                latched.add(name)
                if name not in self.ep["pressed"]:
                    self.ep["pressed"].append(name)
            elif depth < BTN_PRESSED / 2 and name in latched:
                latched.discard(name)
        self.ep["latched"] = sorted(latched)

    # -- scene queries (used by task predicates) ----------------------------------------------------------------
    def pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        info, d = self.items[name], self.data
        if "qadr" in info:
            a = info["qadr"]
            return d.qpos[a:a + 3].copy(), d.qpos[a + 3:a + 7].copy()
        return d.mocap_pos[info["mocap"]].copy(), d.mocap_quat[info["mocap"]].copy()

    def speed(self, name: str) -> tuple[float, float]:
        info = self.items[name]
        if "vadr" not in info:
            return 0.0, 0.0
        v = self.data.qvel[info["vadr"]:info["vadr"] + 6]
        return float(np.linalg.norm(v[:3])), float(np.linalg.norm(v[3:]))

    def touching_hand(self, name: str) -> bool:
        body = self.items[name]["body"]
        m, d = self._model, self.data
        n = d.ncon
        b1, b2 = m.geom_bodyid[d.contact.geom1[:n]], m.geom_bodyid[d.contact.geom2[:n]]
        fingers = np.fromiter(self._fingers, dtype=b1.dtype)
        return bool(np.any(((b1 == body) & np.isin(b2, fingers)) | ((b2 == body) & np.isin(b1, fingers))))

    def button_depth(self, name: str) -> float:
        return float(-self.data.qpos[self.items[name]["press_qadr"]])

    def hand_state(self) -> float:
        return float(np.clip(self.data.qpos[self._finger_qadr].mean() / FINGER_TRAVEL, 0.0, 1.0))

    def tcp_pose(self) -> np.ndarray:
        d = self.data
        T = np.eye(4)
        T[:3, :3] = d.xmat[self._hand].reshape(3, 3)
        T[:3, 3] = d.xpos[self._hand]
        return T @ np.asarray(TCP_IN_EE)

    # -- observation --------------------------------------------------------------------------------------------
    def camera_info(self, name: str) -> dict[str, Any]:
        m, d = self._model, self.data
        cid = self._cam_ids[name]
        f = 0.5 * self.height / math.tan(math.radians(float(m.cam_fovy[cid])) / 2)
        K = [[f, 0.0, (self.width - 1) / 2], [0.0, f, (self.height - 1) / 2], [0.0, 0.0, 1.0]]
        R = d.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])   # MuJoCo camera -> OpenCV optical frame
        return {"K": K, "T_world_cam": _pose(R, d.cam_xpos[cid])}

    def _observe(self) -> Observation:
        d = self.data
        images: dict[str, np.ndarray] = {}
        extra: dict[str, Any] = {"task_family": self.ep.get("task", "")}
        if self.camera_names:
            if self._cams is None:
                self._cams = _Cameras(self._model, self.width, self.height, self.segmentation)
            extra["cameras"] = {}
            for c in self.camera_names:
                rgb, depth, seg = self._cams.render(self._model, d, self._cam_ids[c], self.geom_instance)
                images[c] = rgb
                if self.depth:
                    images[f"{c}.depth"] = depth
                if seg is not None:
                    images[f"{c}.seg"] = seg
                extra["cameras"][c] = self.camera_info(c)
            if self.segmentation:
                extra["segments"] = self.segments()
        if self.privileged_obs:
            extra["privileged"] = self.privileged_state(json_ready=True)
        return Observation(state={"q.arm": d.qpos[self._qadr].copy(), "hand.arm": np.array([self.hand_state()])},
                           images=images, instruction=self.ep.get("instruction"), time=float(d.time), extra=extra)

    def segments(self) -> dict[str, str]:
        out = {str(self.instance["table"]): "table", str(self.instance["robot"]): "robot"}
        for name, it in self.ep.get("items", {}).items():
            out[str(self.instance[name])] = it["label"]
        return out

    def privileged_state(self, json_ready: bool = False) -> dict[str, Any]:
        objects = {}
        for name, it in self.ep.get("items", {}).items():
            p, q = self.pose(name)
            o = {"label": it["label"], "kind": self.items[name]["kind"], "pos": p.tolist(), "quat": q.tolist(),
                 "yaw": _yaw_of(q), "half_size": list(it["half_size"]), "instance": self.instance[name]}
            if name.startswith("button"):
                o["digit"], o["depth"] = it["digit"], self.button_depth(name)
            objects[name] = o
        out = {"task": self.ep.get("task"), "goal": self.ep.get("goal"), "objects": objects,
               "pressed": list(self.ep.get("pressed", [])), "q": self.data.qpos[self._qadr].tolist(),
               "hand": self.hand_state(), "tcp": self.tcp_pose().tolist(), "instruction": self.ep.get("instruction")}
        return json.loads(json.dumps(out)) if json_ready else out

    def render(self, camera: str = "front", width: int | None = None, height: int | None = None) -> np.ndarray:
        """An RGB frame from one camera at the configured resolution (width/height must match when given)."""
        if (width or self.width) != self.width or (height or self.height) != self.height:
            raise ValueError(f"this environment renders at {self.width}x{self.height}")
        if self._cams is None:
            self._cams = _Cameras(self.model, self.width, self.height, self.segmentation)
        return self._cams.render(self._model, self.data, self._cam_ids[camera], self.geom_instance)[0]

    # -- snapshot -----------------------------------------------------------------------------------------------
    def snapshot(self) -> bytes:
        mj = self._mj
        state = np.empty(mj.mj_stateSize(self._model, self._sig))
        mj.mj_getState(self._model, self.data, state, self._sig)
        head = json.dumps({"ep": self.ep, "t": self.t, "warn": self._warn,
                           "task": self.task.model_dump(mode="json") if self.task else None}).encode()
        return struct.pack("<I", len(head)) + head + state.tobytes()

    def restore(self, blob: bytes) -> None:
        mj = self._mj
        (n,) = struct.unpack("<I", blob[:4])
        head = json.loads(blob[4:4 + n].decode())
        self.ep, self.t, self._warn = head["ep"], int(head["t"]), int(head["warn"])
        self.task = TaskInstance.model_validate(head["task"]) if head["task"] else None
        self._apply_episode()
        state = np.frombuffer(blob[4 + n:], dtype=np.float64).copy()
        mj.mj_setState(self._model, self.data, state, self._sig)
        mj.mj_forward(self._model, self.data)

    def close(self) -> None:
        if self._cams is not None:
            self._cams.close()
            self._cams = None


__all__ = ["CAMERA_NAMES", "PandaTableEnv", "build_model", "scene_xml"]
