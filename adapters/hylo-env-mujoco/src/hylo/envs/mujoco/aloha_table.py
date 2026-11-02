"""Two ViperX 300 arms facing each other across a table: `mujoco.aloha_table` (RFC-0006 part 3, two-arm tasks).

The arms, their parallel grippers and their joint-position servos are the MuJoCo Menagerie `aloha` model
(BSD-3-Clause, fetched at a pinned commit into the asset cache on first use, see `assets.py`); the table, the objects,
the fixed cameras and the pinhole wrist cameras are ours. Arm bodies compensate gravity (as the real servos do with
their integral terms), so the position servos track planned motions closely.

World frame: +z up, table top at z = 0, the left arm's base at x = -0.469 and the right arm's at x = +0.469 (both at
y = -0.019), facing each other along x; the operator stands at -y. Each arm reaches top-down grasps over
x in [-0.5, 0.1] (left) or [-0.1, 0.5] (right): only the middle strip is reachable by both.

Control: joint-position targets at `control_hz` (20 Hz; 25 physics steps of 2 ms). Action = left arm 6 joint
targets (rad), left gripper command (0 closed .. 1 open), right arm 6 joint targets, right gripper command.
Observation: `q.left`, `hand.left`, `q.right`, `hand.right` (measured opening in command units: 0 when the pads
touch), RGB-D from `overhead`, `front`, `left_wrist`, `right_wrist`, segmentation and privileged state on request.

Cameras need an OpenGL backend (MUJOCO_GL=egl, or osmesa on CPU); `cameras=[]` needs none.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

from hylo.core.robot import ArmSpec, CameraSpec, HandSpec, RobotSpec

from .arm_table import ArmIO, ArmTableEnv, attach_quietly, camera_xml, simplify_visual_meshes
from .assets import ALOHA, assets, assets_dir
from .panda_table import _f

# -- robot ------------------------------------------------------------------------------------------------------
ARM_NAMES = ("left", "right")
JOINT_SUFFIX = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")
MODEL_LOWER = (-3.14158, -1.85005, -1.76278, -3.14158, -1.8675, -3.14158)
MODEL_UPPER = (3.14158, 1.25664, 1.6057, 3.14158, 2.23402, 3.14158)
LIMIT_MARGIN = 0.02
HOME = {"left": (1.57, -0.96, 1.16, 0.0, -0.3, 0.0),  # the Menagerie neutral pose turned towards the back
        "right": (-1.57, -0.96, 1.16, 0.0, -0.3, 0.0)}  # corners: grippers out of the cameras' view
BASES = {"left": (-0.469, -0.019, 0.02), "right": (0.469, -0.019, 0.02)}
FINGER_OPEN = 0.037                                # finger slide at the open command (Menagerie control range top)
FINGER_TOUCH = 0.0083                              # finger slide where the two fingertips meet (mean of both)
FINGER_SQUEEZE = 0.004                             # command 0 aims this far past the touch point (grip force)
PAD_OFFSET = 0.0079                              # pad face distance from the centre = slide - PAD_OFFSET
TCP = (0.135, 0.0, -0.003)                         # TCP in the gripper_link frame: between the fingertip pads
MAX_WIDTH = 2 * (FINGER_OPEN - PAD_OFFSET)         # about 0.058 m between the pads when open
TCP_IN_EE = [[1.0, 0.0, 0.0, TCP[0]], [0.0, 1.0, 0.0, TCP[1]], [0.0, 0.0, 1.0, TCP[2]], [0.0, 0.0, 0.0, 1.0]]

# -- scene ------------------------------------------------------------------------------------------------------
TABLE_TOP = 0.0
TABLE_HALF = (0.61, 0.38)
WORKSPACE = ((-0.46, 0.46), (-0.16, 0.24))         # where object centres are placed (x, y)
REACH_X = 0.10                                     # each arm's top-down reach past the middle (x)
MESH_CELL = 0.004

BLOCK_RANGE = (0.018, 0.023)                       # half size of the cubes
BAR_HALF = (0.016, 0.016)                          # cross-section half sizes of the handover bar
BAR_LEN_RANGE = (0.08, 0.10)                       # its half length
LIFT_BODY = (0.10, 0.05, 0.03)                     # half extents of the two-handed box (too wide for one gripper)
LIFT_FIN = (0.006, 0.03, 0.04)                     # half extents of its end handles
LIFT_FIN_Z = 0.03                                  # handle centre above the box centre: 4 cm proud of the top
LIFT_MASS = 0.5
ZONE_HALF = 0.085
BIN_IN, BIN_WALL, BIN_H, BIN_BASE = 0.06, 0.006, 0.04, 0.006

POOL: dict[str, int] = {"block": 2, "bar": 1, "liftbox": 1, "zone": 2, "bin": 2}
FREE_KINDS = ("block", "bar", "liftbox")

CAMERAS = {
    "overhead": ((0.0, -0.30, 0.72), (0.0, 0.04, 0.0), 60.0),
    "front": ((0.0, -0.86, 0.40), (0.0, 0.04, 0.04), 52.0),
}
WRIST_FOVY = 75.0
CAMERA_NAMES = ("overhead", "front", "left_wrist", "right_wrist")


def _items_xml() -> str:
    out = []
    for kind, n in POOL.items():
        for i in range(n):
            name = f"{kind}{i}"
            g = f'name="{name}/g'
            if kind == "block":
                h = BLOCK_RANGE[1]
                out.append(f'<body name="{name}"><freejoint/><geom {g}" type="box" size="{h} {h} {h}" mass="0.05" '
                           f'class="obj"/></body>')
            elif kind == "bar":
                out.append(f'<body name="{name}"><freejoint/><geom {g}" type="box" '
                           f'size="{BAR_HALF[0]} {BAR_LEN_RANGE[1]} {BAR_HALF[1]}" mass="0.08" class="obj"/></body>')
            elif kind == "liftbox":
                fx = LIFT_BODY[0] + LIFT_FIN[0]
                out.append(f'<body name="{name}"><freejoint/>'
                           f'<geom {g}0" type="box" size="{_f(LIFT_BODY)}" mass="{LIFT_MASS - 0.06}" class="obj"/>'
                           f'<geom {g}1" type="box" size="{_f(LIFT_FIN)}" pos="{-fx} 0 {LIFT_FIN_Z}" mass="0.03" class="obj"/>'
                           f'<geom {g}2" type="box" size="{_f(LIFT_FIN)}" pos="{fx} 0 {LIFT_FIN_Z}" mass="0.03" class="obj"/>'
                           f'</body>')
            elif kind == "zone":
                out.append(f'<body name="{name}" mocap="true"><geom {g}" type="box" size="{ZONE_HALF} {ZONE_HALF} 0.0006" '
                           f'pos="0 0 0.0007" contype="0" conaffinity="0" class="fix"/></body>')
            elif kind == "bin":
                w, h, b = BIN_WALL, BIN_H / 2, BIN_BASE / 2
                o = BIN_IN + w
                walls = [(o, 0, w, o + w), (-o, 0, w, o + w), (0, o, BIN_IN, w), (0, -o, BIN_IN, w)]
                geoms = [f'<geom {g}0" type="box" size="{o + w} {o + w} {b}" pos="0 0 {b}" class="fix"/>']
                geoms += [f'<geom {g}{k + 1}" type="box" size="{hx} {hy} {h}" pos="{x} {y} {h}" class="fix"/>'
                          for k, (x, y, hx, hy) in enumerate(walls)]
                out.append(f'<body name="{name}" mocap="true">{"".join(geoms)}</body>')
    return "\n      ".join(out)


def scene_xml(width: int, height: int) -> str:
    hx, hy = TABLE_HALF
    return f"""
<mujoco model="aloha_table">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10">
    <flag multiccd="enable"/>
  </option>
  <size memory="40M"/>
  <visual>
    <global offwidth="{max(width, 64)}" offheight="{max(height, 64)}"/>
    <headlight ambient="0.34 0.34 0.34" diffuse="0.45 0.45 0.45" specular="0.05 0.05 0.05"/>
    <map znear="0.01" zfar="4"/>
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
    <light name="key" directional="true" pos="0.3 -0.6 2.5" dir="-0.1 0.3 -1" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1" castshadow="false"/>
    {camera_xml(CAMERAS)}
    <geom name="floor" type="plane" size="4 4 0.1" pos="0 0 -0.75" material="floor" contype="0" conaffinity="0"/>
    <geom name="table" type="box" size="{hx} {hy} 0.375" pos="0 0 {TABLE_TOP - 0.375}" material="table"
          friction="0.9 0.01 0.001"/>
      {_items_xml()}
  </worldbody>
</mujoco>
"""


def build_model(width: int = 320, height: int = 240, mesh_detail: str = "fast", spec=None):
    """Compile the scene with the Menagerie ALOHA arms at their own frame (names as in aloha.xml)."""
    import mujoco

    if spec is None:                                 # `spec`: a scene_xml(width, height) spec to build on
        spec = mujoco.MjSpec.from_string(scene_xml(width, height))   # (mujoco.free_scene adds its objects)
    folder = assets(ALOHA)
    aloha = mujoco.MjSpec.from_file(str(folder / "aloha.xml"))
    for k in list(aloha.keys):
        aloha.delete(k)
    ref = aloha.copy().compile()
    for side in ARM_NAMES:
        old = aloha.camera(f"wrist_cam_{side}")           # an intrinsic (focal, sensor size) camera: replaced by a
        cid = ref.camera(f"wrist_cam_{side}").id          # pinhole one at the same pose, so K follows from fovy
        pos, quat = ref.cam_pos[cid].tolist(), ref.cam_quat[cid].tolist()
        aloha.delete(old)
        aloha.body(f"{side}/gripper_base").add_camera(name=f"{side}/wrist", pos=pos, quat=quat, fovy=WRIST_FOVY)
    for light in list(aloha.lights):
        aloha.delete(light)
    for body in aloha.bodies:
        body.gravcomp = 1.0
    if mesh_detail == "fast":
        simplify_visual_meshes(aloha, folder, MESH_CELL, min_faces=3000)
    elif mesh_detail != "full":
        raise ValueError(f"mesh_detail must be 'fast' or 'full', got {mesh_detail!r}")
    frame = spec.worldbody.add_frame()
    for side in ARM_NAMES:
        attach_quietly(frame, aloha.body(f"{side}/base_link"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return spec.compile()


def _hand_to_ctrl(cmd: np.ndarray) -> np.ndarray:
    c = float(np.clip(np.asarray(cmd, float).ravel()[0], 0.0, 1.0))
    lo = FINGER_TOUCH - FINGER_SQUEEZE
    return np.array([lo + c * (FINGER_OPEN - lo)])


def _hand_state(q: np.ndarray) -> np.ndarray:
    return np.array([float(np.clip((float(np.mean(q)) - FINGER_TOUCH) / (FINGER_OPEN - FINGER_TOUCH), 0.0, 1.0))])


class AlohaTableEnv(ArmTableEnv):
    """`mujoco.aloha_table`. Params: cameras (subset of overhead, front, left_wrist, right_wrist; [] for none),
    width, height (default 320x240), segmentation, privileged_obs, control_hz, depth, mesh_detail ("fast":
    simplified visual meshes of the grippers, much faster to render on CPU; "full": the original meshes)."""

    NAME = "mujoco.aloha_table"
    CAMERA_NAMES = CAMERA_NAMES
    TASK_MODULE = ".bimanual_tasks"
    POOL = POOL
    FREE_KINDS = FREE_KINDS
    RESIZABLE = ("block", "bar")
    TABLE_TOP = TABLE_TOP

    def __init__(self, cameras=CAMERA_NAMES, width: int = 320, height: int = 240, segmentation: bool = False,
                 privileged_obs: bool = False, control_hz: float = 20.0, depth: bool = True,
                 mesh_detail: str = "fast") -> None:
        super().__init__(cameras, width, height, segmentation, privileged_obs, control_hz, depth, mesh_detail)

    def _arms(self) -> list[ArmIO]:
        lo = tuple(round(v + LIMIT_MARGIN, 4) for v in MODEL_LOWER)
        hi = tuple(round(v - LIMIT_MARGIN, 4) for v in MODEL_UPPER)
        return [ArmIO(name=s, joints=tuple(f"{s}/{j}" for j in JOINT_SUFFIX),
                      actuators=tuple(f"{s}/{j}" for j in JOINT_SUFFIX), hand_actuators=(f"{s}/gripper",),
                      finger_joints=(f"{s}/left_finger", f"{s}/right_finger"), hand_to_ctrl=_hand_to_ctrl, hand_state=_hand_state,
                      hand_open=(1.0,), ee_body=f"{s}/gripper_link", tcp_in_ee=TCP_IN_EE,
                      finger_bodies=(f"{s}/left_finger_link", f"{s}/right_finger_link"), home=HOME[s], lower=lo, upper=hi)
                for s in ARM_NAMES]

    def _build_model(self):
        return build_model(self.width, self.height, self.mesh_detail)

    def _camera_model_name(self, cam: str) -> str:
        return f"{cam.split('_')[0]}/wrist" if cam.endswith("_wrist") else cam

    def _is_robot_body(self, m: Any, b: int) -> bool:
        return m.body(b).name.startswith(("left/", "right/"))

    def _set_fingers_open(self, a: ArmIO) -> None:
        m, d = self._model, self.data
        for j in (f"{a.name}/left_finger", f"{a.name}/right_finger"):
            d.qpos[m.jnt_qposadr[m.joint(j).id]] = FINGER_OPEN

    def _coloured_geoms(self, name: str, info: dict[str, Any]) -> list[int]:
        return list(info["geoms"])

    def robot_spec(self) -> RobotSpec:
        model = str(assets_dir(ALOHA) / "aloha.xml")
        arms, i = [], 0
        for a in self.arms:
            hand = HandSpec(kind="parallel", tcp_in_ee=TCP_IN_EE, open=(1.0,), closed=(0.0,), max_width=MAX_WIDTH,
                            joints=(f"{a.name}/left_finger", f"{a.name}/right_finger"))
            arms.append(ArmSpec(name=a.name, model=model, joints=a.joints, ee_frame=a.ee_body, lower=a.lower,
                                upper=a.upper, home=a.home, hand=hand, action_slice=(i, i + 6), hand_slice=(i + 6, i + 7)))
            i += 7
        cams = tuple(CameraSpec(name=c, height=self.height, width=self.width, depth=self.depth,
                                mounted_on=c.split("_")[0] if c.endswith("_wrist") else None) for c in self.camera_names)
        (x0, x1), (y0, y1) = WORKSPACE
        notes = (
            "Two ViperX 300 arms (MuJoCo Menagerie ALOHA) facing each other across a table; world frame: +z up, table "
            f"top at z={TABLE_TOP}, left arm base at {list(BASES['left'])}, right arm base at {list(BASES['right'])}; "
            "the operator side is -y. Each arm reaches top-down grasps only up to about "
            f"{REACH_X:.2f} m past the middle (left: x <= {REACH_X}, right: x >= {-REACH_X}): objects on the far side "
            "must be handed over. Objects start with centres in "
            f"x [{x0}, {x1}], y [{y0}, {y1}]. Grippers open to {MAX_WIDTH:.3f} m between the pads; the two-handed box "
            f"({2 * LIFT_BODY[0]:.2f} x {2 * LIFT_BODY[1]:.2f} m) is too wide for one gripper and has a thin handle "
            f"({2 * LIFT_FIN[0] * 100:.1f} cm) standing {LIFT_FIN_Z + LIFT_FIN[2] - LIFT_BODY[2]:.2f} m above its top "
            "at each end. Action: per arm 6 joint-position targets (rad) then a gripper command (0 closed, 1 open), "
            f"left first, at {self.control_hz:g} Hz; hand.<arm> is the measured opening (0 when the pads touch). TCP: "
            "between the fingertip pads, +x approach, +y closing; each gripper's wrist camera mount sticks out about "
            "11 cm on the TCP's +z side and its finger rail 7.5 cm either way along +y (keep them clear of the other "
            "gripper when both hold one object). Home raises each gripper above its own base. "
            "Success needs objects at rest and released unless the task says to hold them.")
        return RobotSpec(arms=tuple(arms), cameras=cams, table_height=TABLE_TOP, notes=notes)


__all__ = ["AlohaTableEnv", "CAMERA_NAMES", "build_model", "scene_xml"]
