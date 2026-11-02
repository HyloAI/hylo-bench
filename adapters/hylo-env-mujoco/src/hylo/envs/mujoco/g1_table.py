"""The Unitree G1 humanoid standing at a table, working with both arms and three-finger hands: `mujoco.g1_table`.

The robot is the MuJoCo Menagerie `unitree_g1` model with hands (`g1_with_hands.xml`, BSD-3-Clause, fetched at a
pinned commit into the asset cache on first use, see `assets.py`). Upper-body manipulation only: the pelvis is
fixed in place (no floating base, `mobile_base` false) and the leg and waist joints are made rigid at zero, so the
torso, head and shoulders never move; each arm (7 joints) with its hand (7 finger joints) is one ArmSpec with a
dexterous HandSpec. Arm bodies compensate gravity, so the position servos track planned motions closely. The table,
objects and cameras are ours.

World frame: the floor at z = 0, the robot's pelvis at the origin's vertical (x forward, y to the robot's left), the
table top at z = TABLE_TOP in front of it. Hand command: the seven finger joint targets in radians, in HandSpec.joints
order (thumb 0-2, index 0-1, middle 0-1); open is the claw ready to grasp from above; presets name a power grasp, a
pinch, a pointing index and a flat hand.

Observation: `q.left`, `hand.left`, `q.right`, `hand.right` (measured finger joints), RGB-D from `head` (on the torso,
looking down at the table) and `front` (across the table), segmentation and privileged state on request.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

from hylo.core.robot import ArmSpec, CameraSpec, HandSpec, RobotSpec

from .arm_table import ArmIO, ArmTableEnv, attach_quietly, camera_xml, simplify_visual_meshes
from .assets import G1_HANDS, assets, assets_dir
from .panda_table import _f, _xyaxes

ARM_NAMES = ("left", "right")
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
HAND_JOINTS = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
LIMIT_MARGIN = 0.02
# hands raised above the table, claw open and pointing down, above the head camera's view of the table
HOME = {"left": (-0.344, 1.866, 0.565, -0.387, -0.224, 0.608, -0.798),
        "right": (-0.357, -1.867, -0.551, -0.386, 0.226, 0.61, 0.799)}
# Finger joint targets of the left hand (the right hand mirrors the z-axis joints). The hand grasps like a claw from
# above: the thumb hangs down behind, the index and middle fingers curl down in front, and closing swings them
# towards each other under the palm. open: the claw ready (8-9 cm between thumb and fingertips); power: closed on
# anything up to about 6 cm; pinch: fingertips meet the thumb tip; point: index straight, the rest curled; flat: all
# straight.
_OPEN = (0.0, -0.4, 0.0, -0.9, -0.6, -0.9, -0.6)
_POWER = (0.0, 0.6, 0.7, -1.3, -1.2, -1.3, -1.2)
_PINCH = (0.0, 0.5, 0.5, -1.1, -0.9, -1.1, -0.9)
_POINT = (0.0, 0.9, 1.0, 0.0, 0.0, -1.5, -1.6)
_FLAT = (0.0,) * 7
_MIRROR = (1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
# TCP in the wrist_yaw_link frame: about 3 cm in front of the palm, between the thumb and the fingertips of the open
# claw; +x approach (into the palm), +y closing (thumb towards the fingers), +z = x cross y
TCP_POS = {"left": (0.097, -0.055, 0.0), "right": (0.097, 0.055, 0.0)}
TCP_ROT = {"left": ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),          # rows: the TCP's x, y, z axes
           "right": ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0))}
FINGERTIP_BODIES = ("thumb_2", "index_1", "middle_1")
FINGER_TORQUE = 0.7                                # N m limit of the index and middle finger servos (model: 1.4)

# -- scene ------------------------------------------------------------------------------------------------------
TABLE_TOP = 0.86
TABLE_X = (0.16, 0.76)                             # table extent in front of the robot (x)
TABLE_Y = (-0.55, 0.55)
WORKSPACE = ((0.22, 0.32), (-0.25, 0.25))          # where object centres are placed (x, y)
MESH_CELL = 0.004

BLOCK_RANGE = (0.022, 0.026)                       # half size of the cubes
PLATE_R, PLATE_HALF = 0.065, 0.005
BTN_HOUSE, BTN_HOUSE_H, BTN_CAP, BTN_CAP_H, BTN_TRAVEL = 0.035, 0.012, 0.024, 0.006, 0.008
BTN_PRESSED = 0.005
POOL: dict[str, int] = {"block": 2, "plate": 2, "button": 3}
FREE_KINDS = ("block",)
CAP_RGB = (0.94, 0.94, 0.92)

CAMERAS = {"front": ((1.15, 0.0, 1.40), (0.28, 0.0, 0.86), 50.0)}
HEAD_CAM = ((0.10, 0.0, 0.47), (0.30, 0.0, 0.86), 75.0)      # on the torso (pos in torso frame; target in world)
CAMERA_NAMES = ("head", "front")


def _items_xml() -> str:
    out = []
    for kind, n in POOL.items():
        for i in range(n):
            name = f"{kind}{i}"
            g = f'name="{name}/g'
            if kind == "block":
                h = BLOCK_RANGE[1]
                out.append(f'<body name="{name}"><freejoint/><geom {g}" type="box" size="{h} {h} {h}" mass="0.06" '
                           f'class="obj"/></body>')
            elif kind == "plate":
                out.append(f'<body name="{name}" mocap="true"><geom {g}" type="cylinder" size="{PLATE_R} {PLATE_HALF}" '
                           f'pos="0 0 {PLATE_HALF}" class="fix"/></body>')
            elif kind == "button":
                top = 2 * BTN_HOUSE_H
                out.append(
                    f'<body name="{name}" mocap="true">'
                    f'<geom {g}h" type="box" size="{BTN_HOUSE} {BTN_HOUSE} {BTN_HOUSE_H}" pos="0 0 {BTN_HOUSE_H}" '
                    f'rgba="0.22 0.23 0.25 1" class="fix"/>'
                    f'<body name="{name}/cap" pos="0 0 {top + BTN_CAP_H}">'
                    f'<joint name="{name}/press" type="slide" axis="0 0 1" range="-{BTN_TRAVEL} 0" stiffness="150" '
                    f'springref="0.006" damping="2"/>'
                    f'<geom {g}c" type="box" size="{BTN_CAP} {BTN_CAP} {BTN_CAP_H}" mass="0.02" rgba="{_f(CAP_RGB)} 1" '
                    f'class="obj"/></body></body>')
    return "\n      ".join(out)


def scene_xml(width: int, height: int) -> str:
    tx, ty = (TABLE_X[0] + TABLE_X[1]) / 2, (TABLE_Y[0] + TABLE_Y[1]) / 2
    hx, hy = (TABLE_X[1] - TABLE_X[0]) / 2, (TABLE_Y[1] - TABLE_Y[0]) / 2
    return f"""
<mujoco model="g1_table">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10">
    <flag multiccd="enable"/>
  </option>
  <size memory="60M"/>
  <visual>
    <global offwidth="{max(width, 64)}" offheight="{max(height, 64)}"/>
    <headlight ambient="0.34 0.34 0.34" diffuse="0.45 0.45 0.45" specular="0.05 0.05 0.05"/>
    <map znear="0.01" zfar="5"/>
    <quality shadowsize="0"/>
  </visual>
  <default>
    <default class="obj"><geom condim="4" friction="1.0 0.02 0.001" solref="0.005 1" rgba="0.6 0.6 0.6 1"/></default>
    <default class="fix"><geom condim="3" friction="0.8 0.01 0.001" rgba="0.8 0.8 0.8 1"/></default>
  </default>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.82 0.87 0.92" rgb2="0.35 0.40 0.47" width="256" height="1536"/>
    <texture name="wood" type="2d" builtin="flat" rgb1="0.62 0.56 0.48" rgb2="0.58 0.52 0.44" mark="random" random="0.02" width="256" height="256"/>
    <material name="table" texture="wood" texrepeat="4 4" reflectance="0.02" specular="0.1"/>
    <material name="floor" rgba="0.48 0.50 0.53 1"/>
  </asset>
  <worldbody>
    <light name="key" directional="true" pos="0.8 -0.4 3" dir="-0.3 0.2 -1" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1" castshadow="false"/>
    {camera_xml(CAMERAS)}
    <geom name="floor" type="plane" size="4 4 0.1" material="floor" contype="0" conaffinity="0"/>
    <geom name="table" type="box" size="{hx} {hy} {TABLE_TOP / 2}" pos="{tx} {ty} {TABLE_TOP / 2}" material="table"
          friction="0.9 0.01 0.001"/>
      {_items_xml()}
  </worldbody>
</mujoco>
"""


def build_model(width: int = 320, height: int = 240, mesh_detail: str = "fast", spec=None):
    """Compile the scene with the Menagerie G1 (hands) attached: pelvis fixed, legs and waist rigid, names as in
    g1_with_hands.xml."""
    import mujoco

    if spec is None:                                 # `spec`: a scene_xml(width, height) spec to build on
        spec = mujoco.MjSpec.from_string(scene_xml(width, height))   # (mujoco.free_scene adds its objects)
    folder = assets(G1_HANDS)
    g1 = mujoco.MjSpec.from_file(str(folder / "g1_with_hands.xml"))
    for k in list(g1.keys):
        g1.delete(k)
    for s in list(g1.sensors):
        g1.delete(s)
    for light in list(g1.lights):
        g1.delete(light)
    rigid = [j for j in g1.joints if not any(j.name.startswith(f"{s}_") and j.name[len(s) + 1:].split("_joint")[0] in
                                             ARM_JOINTS + tuple(f"hand_{h}" for h in HAND_JOINTS) for s in ARM_NAMES)]
    names = {j.name for j in rigid}
    for a in list(g1.actuators):
        if a.target in names:
            g1.delete(a)
    for j in rigid:                                  # the free joint and the leg and waist joints: rigid at zero
        g1.delete(j)
    for body in g1.bodies:
        body.gravcomp = 1.0
    for a in g1.actuators:
        # the index and middle fingers (1.4 N m each) outpush the thumb (1.4 N m on a longer lever): an object in the
        # claw gets shoved against the thumb's stop. Halving the two fingers' torque balances the claw.
        if "_hand_index_" in a.target or "_hand_middle_" in a.target:
            a.forcerange = [-FINGER_TORQUE, FINGER_TORQUE]
            a.forcelimited = 1
    torso = g1.body("torso_link")
    pos, target, fov = HEAD_CAM
    ref = g1.copy().compile()
    d = mujoco.MjData(ref)
    mujoco.mj_forward(ref, d)
    tid = ref.body("torso_link").id
    R_t, p_t = d.xmat[tid].reshape(3, 3), d.xpos[tid]
    cam_world = p_t + R_t @ np.asarray(pos)
    fwd = R_t.T @ (np.asarray(target) - cam_world)
    up = R_t.T @ np.array([0.0, 0.0, 1.0])
    cam = torso.add_camera(name="head_cam", pos=list(pos), fovy=fov)
    cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    cam.alt.xyaxes = [float(v) for v in _xyaxes(fwd, up).split()]
    if mesh_detail == "fast":
        simplify_visual_meshes(g1, folder, MESH_CELL, min_faces=3000, cache_tag="-g1")
    elif mesh_detail != "full":
        raise ValueError(f"mesh_detail must be 'fast' or 'full', got {mesh_detail!r}")
    frame = spec.worldbody.add_frame()
    attach_quietly(frame, g1.body("pelvis"))
    for side in ARM_NAMES:
        # the palm's collision mesh is convex, so it fills the hollow the fingers fold into: fingers and palm of one
        # hand do not collide with each other (as on the real hand, where they fold into the palm)
        parts = [f"{side}_wrist_yaw_link"] + [f"{side}_hand_{b}_link" for b in
                                              ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")]
        for i, a in enumerate(parts):
            for b in parts[i + 1:]:
                ex = spec.add_exclude()
                ex.bodyname1, ex.bodyname2 = a, b
    for name in (f"button{i}" for i in range(POOL["button"])):
        ex = spec.add_exclude()
        ex.bodyname1, ex.bodyname2 = name, f"{name}/cap"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return spec.compile()


def _side_sign(side: str) -> np.ndarray:
    return np.ones(7) if side == "left" else np.asarray(_MIRROR)


def preset(side: str, closure) -> tuple[float, ...]:
    return tuple(round(float(v), 4) + 0.0 for v in np.asarray(closure) * _side_sign(side))


def tcp_in_ee(side: str) -> list[list[float]]:
    T = np.eye(4)
    T[:3, :3] = np.asarray(TCP_ROT[side]).T                  # columns of the rotation: the TCP axes
    T[:3, 3] = TCP_POS[side]
    return T.tolist()


class G1TableEnv(ArmTableEnv):
    """`mujoco.g1_table`. Params: cameras (subset of head, front; [] for none), width, height (default 320x240),
    segmentation, privileged_obs, control_hz, depth, mesh_detail ("fast": simplified visual meshes; "full")."""

    NAME = "mujoco.g1_table"
    CAMERA_NAMES = CAMERA_NAMES
    TASK_MODULE = ".humanoid_tasks"
    POOL = POOL
    FREE_KINDS = FREE_KINDS
    RESIZABLE = ("block",)
    TABLE_TOP = TABLE_TOP
    PARK_XY = (0.5, 0.0)

    def __init__(self, cameras=CAMERA_NAMES, width: int = 320, height: int = 240, segmentation: bool = False,
                 privileged_obs: bool = False, control_hz: float = 20.0, depth: bool = True,
                 mesh_detail: str = "fast") -> None:
        super().__init__(cameras, width, height, segmentation, privileged_obs, control_hz, depth, mesh_detail)

    def _arms(self) -> list[ArmIO]:
        out = []
        lim = _joint_limits()
        for s in ARM_NAMES:
            joints = tuple(f"{s}_{j}_joint" for j in ARM_JOINTS)
            fingers = tuple(f"{s}_hand_{h}_joint" for h in HAND_JOINTS)
            lo = tuple(round(lim[j][0] + LIMIT_MARGIN, 4) for j in joints)
            hi = tuple(round(lim[j][1] - LIMIT_MARGIN, 4) for j in joints)
            flo = np.array([lim[f][0] for f in fingers])
            fhi = np.array([lim[f][1] for f in fingers])
            out.append(ArmIO(name=s, joints=joints, actuators=joints, hand_actuators=fingers, finger_joints=fingers,
                             hand_to_ctrl=lambda c, lo_=flo, hi_=fhi: np.clip(np.asarray(c, float), lo_, hi_),
                             hand_state=lambda q: np.asarray(q, float).copy(), hand_open=preset(s, _OPEN),
                             ee_body=f"{s}_wrist_yaw_link", tcp_in_ee=tcp_in_ee(s),
                             finger_bodies=tuple(f"{s}_hand_{b}_link" for b in
                                                 ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0",
                                                  "middle_1")) + (f"{s}_wrist_yaw_link",),
                             home=HOME[s], lower=lo, upper=hi, hand_lower=tuple(flo), hand_upper=tuple(fhi)))
        return out

    def _build_model(self):
        return build_model(self.width, self.height, self.mesh_detail)

    def _camera_model_name(self, cam: str) -> str:
        return "head_cam" if cam == "head" else cam

    def _is_robot_body(self, m: Any, b: int) -> bool:
        root = m.body("pelvis").id
        return b == root or self._descends(m, b, root)

    def _item_geoms(self, m: Any, name: str) -> list[int]:
        return super()._item_geoms(m, name)

    def _coloured_geoms(self, name: str, info: dict[str, Any]) -> list[int]:
        if info["kind"] == "button":                    # the housing keeps its grey, the cap takes the colour
            return [g for g in info["geoms"] if self._model.geom(g).name.endswith("/gc")]
        return list(info["geoms"])

    def _bind_item(self, m: Any, name: str, info: dict[str, Any]) -> None:
        if info["kind"] == "button":
            info["press_qadr"] = int(m.jnt_qposadr[m.joint(f"{name}/press").id])

    def button_depth(self, name: str) -> float:
        return float(-self.data.qpos[self.items[name]["press_qadr"]])

    def _after_step(self) -> None:
        state = self.ep.setdefault("state", {})
        pressed = list(state.get("pressed", []))
        latched = set(state.get("latched", []))
        for name in self.ep["items"]:
            if not name.startswith("button"):
                continue
            depth = self.button_depth(name)
            if depth >= BTN_PRESSED and name not in latched:
                latched.add(name)
                if name not in pressed:
                    pressed.append(name)
            elif depth < BTN_PRESSED / 2 and name in latched:
                latched.discard(name)
        state["pressed"], state["latched"] = pressed, sorted(latched)

    def _privileged_item(self, name: str, it: dict[str, Any]) -> dict[str, Any]:
        return {"depth": self.button_depth(name)} if name.startswith("button") else {}

    def robot_spec(self) -> RobotSpec:
        model = str(assets_dir(G1_HANDS) / "g1_with_hands.xml")
        arms, i = [], 0
        for a in self.arms:
            s = a.name
            hand = HandSpec(kind="dexterous", tcp_in_ee=a.tcp_in_ee, open=preset(s, _OPEN), closed=preset(s, _POWER),
                            max_width=0.08, joints=a.finger_joints,
                            presets={"power": preset(s, _POWER), "pinch": preset(s, _PINCH), "point": preset(s, _POINT),
                                     "flat": preset(s, _FLAT)})
            arms.append(ArmSpec(name=s, model=model, joints=a.joints, ee_frame=a.ee_body,
                                base_in_world=np.eye(4).tolist(), lower=a.lower, upper=a.upper, home=a.home,
                                hand=hand, action_slice=(i, i + 7), hand_slice=(i + 7, i + 14)))
            i += 14
        cams = tuple(CameraSpec(name=c, height=self.height, width=self.width, depth=self.depth) for c in self.camera_names)
        (x0, x1), (y0, y1) = WORKSPACE
        notes = (
            "Unitree G1 humanoid (MuJoCo Menagerie, three-finger hands) standing at a table with its pelvis fixed; only "
            "the arms and fingers move (torso, head and legs are rigid). World frame: floor at z=0, +x forward from "
            f"the robot, +y to its left; table top at z={TABLE_TOP}, its near edge at x={TABLE_X[0]}. Objects start "
            f"with centres in x [{x0}, {x1}], y [{y0}, {y1}]; the left hand reaches best at y > -0.1, the right at "
            "y < 0.1; reach falls off quickly beyond x = 0.33. Each arm: 7 joint-position targets (rad) then 7 "
            "finger joint targets (rad, HandSpec.joints order: thumb 0-2, index 0-1, middle 0-1); the hand grasps like a claw from above (thumb behind, two "
            "fingers in front, closing under the palm): open is that claw ready (about 8 cm between thumb and "
            "fingertips), closed the power grasp; presets power, pinch, point (index straight, others curled), "
            f"flat (all straight). Control at {self.control_hz:g} Hz; hand.<arm> is the measured finger joints. "
            "TCP: about 3 cm in front of the palm between the thumb and the fingertips; +x approach (into the palm), "
            "+y closing (from the thumb towards the fingers). Home holds both hands high over the near edge of the "
            "table, claws pointing down, out of the head camera's view. Buttons register a press when pushed down "
            f"{BTN_PRESSED * 1000:.0f} mm. Success needs objects at rest and released unless the task says to hold.")
        return RobotSpec(arms=tuple(arms), cameras=cams, table_height=TABLE_TOP, notes=notes)


def _joint_limits() -> dict[str, tuple[float, float]]:
    """Joint ranges of the arm and finger joints as in g1_with_hands.xml (kept here so spec() needs no download)."""
    arm = {"shoulder_pitch": (-3.0892, 2.6704), "shoulder_yaw": (-2.618, 2.618), "elbow": (-1.0472, 2.0944),
           "wrist_roll": (-1.97222, 1.97222), "wrist_pitch": (-1.61443, 1.61443), "wrist_yaw": (-1.61443, 1.61443)}
    out: dict[str, tuple[float, float]] = {}
    for s, roll in (("left", (-1.5882, 2.2515)), ("right", (-2.2515, 1.5882))):
        for j, r in {**arm, "shoulder_roll": roll}.items():
            out[f"{s}_{j}_joint"] = r
    left = {"thumb_0": (-1.0472, 1.0472), "thumb_1": (-0.724312, 1.0472), "thumb_2": (0.0, 1.74533),
            "index_0": (-1.5708, 0.0), "index_1": (-1.74533, 0.0), "middle_0": (-1.5708, 0.0), "middle_1": (-1.74533, 0.0)}
    for h, (lo, hi) in left.items():
        out[f"left_hand_{h}_joint"] = (lo, hi)
        out[f"right_hand_{h}_joint"] = (lo, hi) if h == "thumb_0" else (-hi, -lo)
    return out


__all__ = ["CAMERA_NAMES", "G1TableEnv", "build_model", "preset", "scene_xml", "tcp_in_ee"]
