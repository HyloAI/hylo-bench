"""A floating dexterous hand above a table with objects of several shapes: `mujoco.dex_table`.

The robot is one hand from MuJoCo Menagerie (`hands.py`: the Unitree G1 three-finger hand or the Allegro hand, left
or right) on a six-joint base (x, y, z slides, then yaw, pitch and roll hinges), every joint position-controlled and
gravity-compensated, so the toolkit plans and runs the base like an arm (one ArmSpec named "hand" with the hand's
dexterous HandSpec). At zero base joints the TCP points straight down. The table, objects and cameras are ours.

Objects (one of each in the pool, all free): box (5 cm cube), cylinder (r 2.5 cm, 10 cm tall), sphere (r 3 cm), mug
(r 3.5 cm, 9 cm tall, a handle: concave, a cup of wall segments) and rock (an irregular solid of three overlapping
lumps, seeded), each a set of convex mesh parts (`shape(kind)`, also what a planner can be given as ground truth).

World frame: table top at z = 0, object centres within 6 cm of the origin, the hand's home 30 cm above it.
Observation: `q.hand` (base joints), `hand.hand` (finger joints), RGB-D from `front`, `side` and `back`, segmentation
and privileged state on request. Params: hand (default allegro_right), cameras, width, height, segmentation,
privileged_obs, control_hz, depth, finger_gain (multiplies the finger servos' stiffness).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.spatial import ConvexHull

from hylo.core.robot import ArmSpec, CameraSpec, RobotSpec

from .arm_table import ArmIO, ArmTableEnv, camera_xml
from .hands import BASE_JOINTS, BASE_RANGE, base_joint_names, floating_hand_file, floating_hand_spec, hand_entry, resolve

TABLE_TOP = 0.0
WORKSPACE = ((-0.06, 0.06), (-0.06, 0.06))
HOME = (0.0, 0.0, 0.30, 0.0, 0.0, 0.0)
CAMERAS = {"front": ((0.55, 0.0, 0.42), (0.0, 0.0, 0.04), 45.0),
           "side": ((0.05, -0.55, 0.42), (0.0, 0.0, 0.04), 45.0),
           "back": ((-0.45, 0.25, 0.50), (0.0, 0.0, 0.04), 45.0)}
CAMERA_NAMES = tuple(CAMERAS)
SHAPES = ("box", "cylinder", "sphere", "mug", "rock")
POOL = {k: 1 for k in SHAPES}


# ---------------------------------------------------------------------------------------------------- shapes
@dataclass
class Shape:
    """Convex parts (V, F) in the object frame (origin on the table under its centre), mass, half size (box)."""
    kind: str
    parts: list[tuple[np.ndarray, np.ndarray]]
    mass: float
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def half_size(self) -> list[float]:
        V = np.vstack([v for v, _ in self.parts])
        lo, hi = V.min(0), V.max(0)
        return [float(x) for x in (hi - lo) / 2]

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        Vs, Fs, k = [], [], 0
        for V, F in self.parts:
            Vs.append(V)
            Fs.append(F + k)
            k += len(V)
        return np.vstack(Vs), np.vstack(Fs)


def hull_mesh(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convex hull of points as (V, F) with outward winding."""
    h = ConvexHull(P)
    V = P[h.vertices]
    remap = {v: i for i, v in enumerate(h.vertices)}
    F = np.array([[remap[i] for i in s] for s in h.simplices])
    c = V.mean(0)
    for k, f in enumerate(F):
        a, b, cc = V[f]
        if np.dot(np.cross(b - a, cc - a), a - c) < 0:
            F[k] = f[[0, 2, 1]]
    return V, F


def _box(half: Sequence[float], centre: Sequence[float] = (0, 0, 0)):
    V = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * np.asarray(half) + centre
    return hull_mesh(V)


def _cyl(r: float, z0: float, z1: float, n: int = 24):
    t = 2 * np.pi * np.arange(n) / n
    ring = np.stack([r * np.cos(t), r * np.sin(t)], 1)
    return hull_mesh(np.vstack([np.c_[ring, np.full(n, z0)], np.c_[ring, np.full(n, z1)]]))


def _ball(r: float, n: int, centre, scale=(1.0, 1.0, 1.0)):
    i = np.arange(n) + 0.5
    phi, th = np.arccos(1 - 2 * i / n), np.pi * (1 + 5 ** 0.5) * i
    P = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1) * r * np.asarray(scale)
    return P + np.asarray(centre, float)


def shape(kind: str, seed: int = 0) -> Shape:
    if kind == "box":
        return Shape(kind, [_box([0.025] * 3, [0, 0, 0.025])], 0.10)
    if kind == "cylinder":
        return Shape(kind, [_cyl(0.025, 0.0, 0.10)], 0.12)
    if kind == "sphere":
        V, F = hull_mesh(_ball(0.03, 160, (0, 0, 0.03)))
        return Shape(kind, [(V - [0, 0, V[:, 2].min()], F)], 0.08)
    if kind == "mug":
        r, wall, h, base = 0.035, 0.006, 0.09, 0.008
        parts = [_cyl(r, 0.0, base)]
        n = 12
        for k in range(n):
            ts = np.linspace(2 * np.pi * k / n, 2 * np.pi * (k + 1) / n, 3)
            pts = [[rr * np.cos(t), rr * np.sin(t), z] for t in ts for rr in (r - wall, r) for z in (base, h)]
            parts.append(hull_mesh(np.asarray(pts)))
        parts += [_box([0.012, 0.006, 0.005], [r + 0.010, 0, 0.072]), _box([0.012, 0.006, 0.005], [r + 0.010, 0, 0.022]),
                  _box([0.005, 0.006, 0.030], [r + 0.020, 0, 0.047])]
        return Shape(kind, parts, 0.15, {"handle": "+x"})
    if kind == "rock":
        rng = np.random.default_rng(seed)
        parts = []
        for c, sc in (((0.0, 0.0, 0.022), (1.0, 0.85, 0.75)), ((0.017, 0.008, 0.02), (0.65, 0.6, 0.6)),
                      ((-0.014, -0.01, 0.018), (0.6, 0.7, 0.55))):
            P = _ball(0.03, 60, c, sc)
            parts.append(hull_mesh(P + rng.normal(0, 0.003, P.shape)))
        lo = min(V[:, 2].min() for V, _ in parts)
        return Shape(kind, [(V - [0, 0, lo], F) for V, F in parts], 0.10, {"seed": seed})
    raise KeyError(f"no shape {kind!r}; shapes: {list(SHAPES)}")


# ---------------------------------------------------------------------------------------------------- model
def scene_xml(width: int, height: int) -> str:
    return f"""
<mujoco model="dex_table">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"><flag multiccd="enable"/></option>
  <size memory="64M"/>
  <visual>
    <global offwidth="{max(width, 64)}" offheight="{max(height, 64)}"/>
    <headlight ambient="0.34 0.34 0.34" diffuse="0.45 0.45 0.45" specular="0.05 0.05 0.05"/>
    <map znear="0.01" zfar="5"/>
    <quality shadowsize="0"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.82 0.87 0.92" rgb2="0.35 0.40 0.47" width="256" height="1536"/>
    <texture name="wood" type="2d" builtin="flat" rgb1="0.62 0.56 0.48" rgb2="0.58 0.52 0.44" mark="random" random="0.02" width="256" height="256"/>
    <material name="table" texture="wood" texrepeat="4 4" reflectance="0.02" specular="0.1"/>
  </asset>
  <worldbody>
    <light name="key" directional="true" pos="0.8 -0.4 3" dir="-0.3 0.2 -1" diffuse="0.5 0.5 0.5" castshadow="false"/>
    {camera_xml(CAMERAS)}
    <geom name="table" type="box" size="0.45 0.45 0.02" pos="0 0 {TABLE_TOP - 0.02}" material="table" friction="0.9 0.01 0.001"/>
  </worldbody>
</mujoco>
"""


def build_model(hand: str, width: int = 320, height: int = 240, finger_gain: float = 1.0):
    import mujoco

    spec = mujoco.MjSpec.from_string(scene_xml(width, height))
    for kind in SHAPES:
        shp = shape(kind)
        b = spec.worldbody.add_body(name=f"{kind}0", pos=[0, 0, -0.45])
        b.add_freejoint(name=f"{kind}0/free")
        vols = [ConvexHull(V).volume for V, _ in shp.parts]
        for k, (V, F) in enumerate(shp.parts):
            spec.add_mesh(name=f"{kind}0/m{k}", uservert=np.asarray(V, np.float32).ravel().tolist(),
                          userface=np.asarray(F, np.int32).ravel().tolist())
            b.add_geom(name=f"{kind}0/g{k}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{kind}0/m{k}",
                       mass=shp.mass * vols[k] / sum(vols), condim=4, friction=[1.0, 0.01, 0.001],
                       solref=[0.004, 1.0], rgba=[0.6, 0.6, 0.6, 1.0])
    robot = floating_hand_spec(hand, finger_gain)
    for g in robot.geoms:                              # one contact model for the fingers of every hand
        if g.contype or g.conaffinity:
            g.friction = [1.0, 0.01, 0.001]
            g.condim = 4
            g.solref = [0.004, 1.0]
    hand_bodies = [b.name for b in robot.bodies if b.name not in ("world",) and not b.name.startswith("dex_")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec.worldbody.add_frame().attach_body(robot.body("dex_x_link"), "", "")
    for i, a in enumerate(hand_bodies):               # the hand's links do not collide with each other
        for b in hand_bodies[i + 1:]:
            ex = spec.add_exclude()
            ex.bodyname1, ex.bodyname2 = a, b
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return spec.compile()


class DexTableEnv(ArmTableEnv):
    """`mujoco.dex_table`; see the module docstring."""

    NAME = "mujoco.dex_table"
    CAMERA_NAMES = CAMERA_NAMES
    TASK_MODULE = ".dex_tasks"
    POOL = POOL
    FREE_KINDS = SHAPES
    TABLE_TOP = TABLE_TOP
    PARK_XY = (0.3, 0.3)

    def __init__(self, hand: str = "allegro_right", cameras=CAMERA_NAMES, width: int = 320, height: int = 240,
                 segmentation: bool = False, privileged_obs: bool = False, control_hz: float = 20.0,
                 depth: bool = True, finger_gain: float = 1.0) -> None:
        self.hand = resolve(hand)
        self.entry = hand_entry(self.hand)
        self.finger_gain = float(finger_gain)
        super().__init__(cameras, width, height, segmentation, privileged_obs, control_hz, depth, "full")

    def _arms(self) -> list[ArmIO]:
        e = self.entry
        lo = np.array([r[0] for r in e.ranges])
        hi = np.array([r[1] for r in e.ranges])
        joints = base_joint_names()
        return [ArmIO(name="hand", joints=joints, actuators=tuple(f"{j}_servo" for j in joints),
                      hand_actuators=tuple(f"{j}_servo" for j in e.spec.joints), finger_joints=e.spec.joints,
                      hand_to_ctrl=lambda c, lo_=lo, hi_=hi: np.clip(np.asarray(c, float), lo_, hi_),
                      hand_state=lambda q: np.asarray(q, float).copy(), hand_open=e.spec.open, ee_body=e.ee,
                      tcp_in_ee=[list(r) for r in e.spec.tcp_in_ee], finger_bodies=(), home=HOME,
                      lower=tuple(BASE_RANGE[j][0] for j, _, _ in BASE_JOINTS),
                      upper=tuple(BASE_RANGE[j][1] for j, _, _ in BASE_JOINTS),
                      hand_lower=tuple(lo), hand_upper=tuple(hi))]

    def _build_model(self):
        m = build_model(self.hand, self.width, self.height, self.finger_gain)
        a = self.arms[0]
        root = m.body("dex_x_link").id
        a.finger_bodies = tuple(m.body(b).name for b in range(m.nbody)
                                if self._descends(m, b, root) and not m.body(b).name.startswith("dex_"))
        return m

    def _is_robot_body(self, m: Any, b: int) -> bool:
        root = m.body("dex_x_link").id
        return b == root or self._descends(m, b, root)

    def _set_fingers_open(self, a: ArmIO) -> None:
        self.data.qpos[a.finger_qadr] = a.hand_to_ctrl(np.asarray(a.hand_open, float))

    def robot_spec(self) -> RobotSpec:
        e = self.entry
        a = self.arms[0]
        arm = ArmSpec(name="hand", model=str(floating_hand_file(self.hand)), joints=a.joints, ee_frame=e.ee,
                      lower=a.lower, upper=a.upper, home=a.home, hand=e.spec, action_slice=(0, 6),
                      hand_slice=(6, 6 + len(e.spec.open)))
        cams = tuple(CameraSpec(name=c, height=self.height, width=self.width, depth=self.depth) for c in self.camera_names)
        notes = (f"{e.title} (MuJoCo Menagerie, {e.licence}) on a floating base: six joint-position targets (x, y, z "
                 "slides in metres, then yaw, pitch, roll hinges in radians; at zero the TCP points straight down), "
                 f"then {len(e.spec.open)} finger joint targets (rad, HandSpec.joints order). Table top at z=0, "
                 "objects within 6 cm of the origin, home 30 cm above it. Presets: " + ", ".join(sorted(e.spec.presets)) +
                 f". Control at {self.control_hz:g} Hz. Use dex_grasps for multi-finger grasps.")
        return RobotSpec(arms=(arm,), cameras=cams, table_height=TABLE_TOP, notes=notes)


__all__ = ["CAMERA_NAMES", "DexTableEnv", "SHAPES", "Shape", "build_model", "hull_mesh", "shape"]
