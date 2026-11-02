"""The robot contract (hylo.core.robot) for LIBERO: the RobotSpec, human labels for the BDDL objects, camera
intrinsics and poses (pure Python: numpy only). The simulator side (the Panda model exported from robosuite, the
segmentation render) is in `sim.py`.

World frame of the robot contract: the robot's base frame (robosuite body `robot0_base`), +x forward, +z up. LIBERO
places the robot differently per scene (kitchen and study tables: base 0.91 m above the floor; living room: 0.42 m;
floor scenes: on the floor) while `EnvSpec.robot` is fixed for an environment, so every pose the contract reports
(camera poses, TCP, privileged object poses) is relative to the base; `ArmSpec.base_in_world` is then the identity
and the support surface is always within a few centimetres of z = 0. `extra["libero"]["base_in_sim"]` gives the
base in LIBERO's own world frame.
"""

from __future__ import annotations

import math
import re
from typing import Any, Sequence

import numpy as np

# -- the arm (robosuite's Panda: robots/panda/robot.xml + grippers/panda_gripper.xml) ------------------------------
JOINTS = tuple(f"robot0_joint{i}" for i in range(1, 8))
FINGER_JOINTS = ("gripper0_finger_joint1", "gripper0_finger_joint2")
EE_BODY = "gripper0_right_gripper"
PADS = ("gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision")
LOWER = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973)
UPPER = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973)
LIMIT_MARGIN = 0.02                                # ArmSpec limits sit this far inside the model's
FINGER_TRAVEL = 0.04                               # each finger, metres, at the joint limit
GAP_CLOSED, GAP_OPEN = 0.001, 0.0788               # finger travel sum where robosuite's finger servos settle
# hand raised above the base and pulled back: out of agentview's view of the table (LIBERO's start posture,
# robosuite init_qpos (0, -0.161, 0, -2.445, 0, 2.227, pi/4), holds the hand over the back of the table)
HOME = (0.0, -0.9, 0.0, -2.7, 0.0, 1.9, 0.7854)
CAMERAS = {"agentview": "agentview", "wrist": "robot0_eye_in_hand"}   # contract name -> robosuite camera


def robot_spec(model_path: str, geometry: dict[str, Any], cameras: Sequence[str], height: int, width: int,
               control_hz: float, table_height: float = 0.0, segmentation: bool = False):
    from hylo.core.robot import IDENTITY, ArmSpec, CameraSpec, HandSpec, RobotSpec

    lo = tuple(round(v + LIMIT_MARGIN, 4) for v in LOWER)
    hi = tuple(round(v - LIMIT_MARGIN, 4) for v in UPPER)
    hand = HandSpec(kind="parallel", tcp_in_ee=geometry["tcp_in_ee"], open=(1.0,), closed=(0.0,),
                    max_width=geometry["max_width"], joints=FINGER_JOINTS)
    arm = ArmSpec(name="arm", model=str(model_path), joints=JOINTS, ee_frame=EE_BODY,
                  base_in_world=[r[:] for r in IDENTITY], lower=lo, upper=hi, home=HOME, hand=hand,
                  action_slice=(0, 7), hand_slice=(7, 8))
    cams = tuple(CameraSpec(name=c, height=height, width=width, depth=True, mounted_on="arm" if c == "wrist" else None)
                 for c in cameras)
    notes = (
        "LIBERO (robosuite 1.4) Franka Panda with its parallel gripper. World frame: the robot's base frame, +z up, "
        "+x pointing from the base across the table towards the agentview camera; the support surface (a table or the "
        "floor, depending on the scene) is within about 4 cm of z=0 (measure it with robot.table). Objects lie about "
        "0.3-0.9 m in front of the base. Action: 7 joint-position targets (rad) + hand command (0 closed, 1 open) at "
        f"{control_hz:g} Hz; the hand command is a target opening, reached at about 0.4 s for the full stroke; "
        f"hand.arm is the measured opening (max {geometry['max_width']:.3f} m between the pads). TCP: "
        f"between the finger pads, {geometry['tcp_offset']:.3f} m along the hand's axis; +x approach, +y closing. "
        "Objects are scanned kitchen items (bowls, plates, boxes, bottles, mugs, a basket) and articulated fixtures "
        "(a cabinet drawer slides out along its front's normal when pulled by its handle, the stove turns on when its "
        "knob is twisted, a microwave door swings); bowls and plates are wider than the gripper: grasp a bowl by its rim. Episodes have a tight "
        "step budget (LIBERO's: 280-520 steps); the episode ends as soon as the task's goal holds."
        + (" Segment labels name the BDDL objects (e.g. 'black bowl', 'plate', 'basket') and fixture parts "
           "(e.g. 'top drawer of the wooden cabinet', 'stove knob')." if segmentation else ""))
    return RobotSpec(arms=(arm,), cameras=cams, table_height=table_height, notes=notes)


# ---------------------------------------------------------------------------------------------- labels
# BDDL categories whose words differ from how instructions name them; others: underscores to spaces
LABELS = {
    "akita_black_bowl": "black bowl", "glazed_rim_porcelain_ramekin": "ramekin", "cookies": "cookie box",
    "flat_stove": "stove", "chefmate_8_frypan": "frying pan", "porcelain_mug": "white mug",
    "white_yellow_mug": "yellow and white mug", "red_coffee_mug": "red mug", "black_book": "book",
    "desk_caddy": "caddy", "new_salad_dressing": "salad dressing", "cream_cheese": "cream cheese box",
    "main_table": "table", "kitchen_table": "table", "living_room_table": "table", "study_table": "table",
    "bowl_drainer": "bowl drainer", "wooden_tray": "wooden tray", "white_storage_box": "storage box",
}
# fixture parts (body suffix -> label pattern, {o} is the fixture's own label)
PARTS = {
    "cabinet_top": "top drawer of the {o}", "cabinet_middle": "middle drawer of the {o}",
    "cabinet_bottom": "bottom drawer of the {o}", "button": "{o} knob", "microdoorroot": "{o} door",
}


def category(name: str) -> str:
    """BDDL object name without its instance number ("akita_black_bowl_1" -> "akita_black_bowl")."""
    return re.sub(r"_\d+$", "", name)


def object_label(name: str) -> str:
    cat = category(name)
    return LABELS.get(cat, cat.replace("_", " "))


# ---------------------------------------------------------------------------------------------- cameras and frames
def pose(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def intrinsics(fovy_deg: float, height: int, width: int) -> list[list[float]]:
    f = 0.5 * height / math.tan(math.radians(fovy_deg) / 2)
    return [[f, 0.0, (width - 1) / 2], [0.0, f, (height - 1) / 2], [0.0, 0.0, 1.0]]


def camera_pose(m: Any, d: Any, cam_id: int, T_base_inv: np.ndarray) -> np.ndarray:
    """The camera in the contract's world frame (the robot base), OpenCV optical axes (x right, y down, z forward)."""
    R = d.cam_xmat[cam_id].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    return T_base_inv @ pose(R, d.cam_xpos[cam_id])


def depth_metres(m: Any, z: np.ndarray) -> np.ndarray:
    """MuJoCo's normalised depth buffer -> metres along the optical axis; 0 where nothing was hit."""
    ext = m.stat.extent
    near, far = m.vis.map.znear * ext, m.vis.map.zfar * ext
    out = (near / (1.0 - z * (1.0 - near / far))).astype(np.float32)
    out[z >= 1.0 - 1e-7] = 0.0
    return out


__all__ = ["CAMERAS", "HOME", "JOINTS", "LABELS", "PARTS", "camera_pose", "category", "depth_metres", "intrinsics",
           "object_label", "pose", "robot_spec"]
