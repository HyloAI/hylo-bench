"""Pure tests of the shared robot-contract machinery (no simulator); the model export runs on a small synthetic MJCF
with MuJoCo (installed in the workspace through hylo-manip)."""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hylo.envs.robosuite_contract.control import hand_opening, split_action, step_gripper_setpoint
from hylo.envs.robosuite_contract.env import IDENTITY, ArmLayout, RobosuiteContractEnv
from hylo.envs.robosuite_contract.geometry import camera_pose, depth_metres, intrinsics, pose, project
from hylo.envs.robosuite_contract.labels import LabelRules, label_geoms, top_ancestors
from hylo.envs.robosuite_contract.panda import HOME, LIMIT_MARGIN, LOWER, ArmFacts, PandaNames, robot_spec


# ---------------------------------------------------------------- geometry
def test_intrinsics_and_projection_follow_the_pinhole_model():
    K = np.asarray(intrinsics(90.0, 128, 128))
    assert K[0, 0] == pytest.approx(64.0) and K[0, 2] == pytest.approx(63.5)
    T = pose(np.eye(3), [0.0, 0.0, 0.0])                     # OpenCV camera at the origin looking along +z
    u, v, z = project(K, T, [0.5, 0.0, 1.0])
    assert (u, v, z) == pytest.approx((63.5 + 32.0, 63.5, 1.0))


def test_camera_pose_turns_mujoco_axes_into_opencv_axes():
    d = SimpleNamespace(cam_xmat=np.eye(3).reshape(1, 9), cam_xpos=np.array([[1.0, 2.0, 3.0]]))
    T = camera_pose(d, 0, np.eye(4))
    assert np.allclose(T[:3, 2], [0.0, 0.0, -1.0])           # MuJoCo looks along -z: OpenCV's +z
    assert np.allclose(T[:3, 1], [0.0, -1.0, 0.0]) and np.allclose(T[:3, 3], [1.0, 2.0, 3.0])


def test_depth_buffer_becomes_metres_and_misses_become_zero():
    m = SimpleNamespace(stat=SimpleNamespace(extent=2.0), vis=SimpleNamespace(map=SimpleNamespace(znear=0.01, zfar=50.0)))
    near, far = 0.02, 100.0
    z_mid = (1 - near / 1.0) / (1 - near / far)              # buffer value of a point 1 m away
    out = depth_metres(m, np.array([[0.0, z_mid, 1.0]]))
    assert out[0, 0] == pytest.approx(near) and out[0, 1] == pytest.approx(1.0, rel=1e-5) and out[0, 2] == 0.0


# ---------------------------------------------------------------- panda and RobotSpec
def test_panda_names_follow_robosuite_15():
    n = PandaNames(1, "PandaGripper")
    assert n.joints[0] == "robot1_joint1" and n.ee_body == "gripper1_right_right_gripper"
    assert n.finger_joints == ("gripper1_right_finger_joint1", "gripper1_right_finger_joint2")
    assert PandaNames(0, "WipingGripper").ee_body == "gripper0_right_wiping_gripper"
    assert PandaNames(0, None).finger_joints == () and PandaNames(0, None).ee_body == "gripper0_right_null_gripper"


def test_robot_spec_lays_out_one_block_of_eight_per_arm():
    facts = [ArmFacts(f"robot{i}", PandaNames(i), "/m.xml", IDENTITY, IDENTITY, 0.079) for i in range(2)]
    spec = robot_spec(facts, [{"name": "front", "height": 64, "width": 64, "mounted_on": None}], -0.1, "notes")
    assert spec.action_dim == 16
    assert spec.arm("robot1").action_slice == (8, 15) and spec.arm("robot1").hand_slice == (15, 16)
    assert spec.arm("robot0").lower[0] == pytest.approx(LOWER[0] + LIMIT_MARGIN) and spec.arm("robot0").home == HOME


# ---------------------------------------------------------------- control
def test_gripper_setpoint_moves_at_the_gripper_speed_towards_the_command():
    g = SimpleNamespace(current_action=np.array([1.0, -1.0]), speed=0.2)       # fully open
    step_gripper_setpoint(g, 0.0)
    assert np.allclose(g.current_action, [0.8, -0.8])
    for _ in range(20):
        step_gripper_setpoint(g, 0.0)
    assert np.allclose(g.current_action, [-1.0, 1.0])                          # closed, and it stays there
    step_gripper_setpoint(g, 0.5, speed=5.0)
    assert np.allclose(g.current_action, [0.0, 0.0])


def test_hand_opening_and_action_split():
    assert hand_opening([0.0, 0.0], 0.0, 0.08) == 0.0 and hand_opening([0.04, -0.04], 0.0, 0.08) == 1.0
    parts = split_action(np.arange(16.0), 2)
    assert np.allclose(parts[1][0], np.arange(8.0, 15.0)) and parts[1][1] == 15.0
    with pytest.raises(ValueError):
        split_action(np.zeros(7), 1)


# ---------------------------------------------------------------- labels
def test_label_rules_give_stable_ids_in_rule_order():
    bodies = ["world", "table", "robot0_base", "robot0_link1", "peg_main", "pot_root", "bin2", "contact3_main"]
    parent = [0, 0, 0, 2, 3, 0, 0, 0]
    geoms = ["floor", "table_top", "robot0_g0", "robot0_g1", "peg_g0", "pot_base", "pot_handle0_c", "bin2_wall",
             "contact3_g0"]
    gbody = [0, 1, 2, 3, 4, 5, 5, 6, 7]
    rules = LabelRules(objects={"pot": "pot"}, parts=((r"^pot_handle", "pot handle"),), bodies={"bin2": "target bin"},
                       markers={"contact": "dirt"}, attached=((r"^peg_", "peg"),))
    out = label_geoms(bodies, parent, geoms, gbody, rules)
    lab = {g: out.labels.get(int(i), "-") for g, i in zip(geoms, out.geom)}
    assert lab == {"floor": "-", "table_top": "table", "robot0_g0": "robot", "robot0_g1": "robot", "peg_g0": "peg",
                   "pot_base": "pot", "pot_handle0_c": "pot handle", "bin2_wall": "target bin", "contact3_g0": "dirt"}
    assert out.labels[1] == "table" and out.labels[2] == "robot" and out.segments()["3"] == "peg"
    assert top_ancestors(bodies, parent)[3] == "robot0_base"


def test_support_prefixes_label_a_kitchen_counter():
    out = label_geoms(["world", "counter_1_main"], [0, 0], ["c"], [1],
                      LabelRules(support=(), support_prefixes=("counter_1_",), support_label="counter"))
    assert out.labels[int(out.geom[0])] == "counter"


# ---------------------------------------------------------------- the base environment (its pure side)
class _Env(RobosuiteContractEnv):
    def arm_layout(self):
        return [ArmLayout("arm", PandaNames(0), IDENTITY)]

    def camera_map(self):
        return {"front": ("frontview", None), "wrist": ("robot0_eye_in_hand", "arm")}

    def panda_model(self, names):
        return {"model": "/nowhere/panda.xml", "geometry": {"tcp_in_ee": IDENTITY, "max_width": 0.079,
                                                             "tcp_offset": 0.093}}


def test_base_env_publishes_the_contract_without_a_simulator():
    env = _Env(cameras=("front", "wrist"), image_size=64)
    spec = env.spec()
    assert spec.action.shape == (8,) and spec.action.semantics.control_mode == "joint_pos"
    assert set(spec.observation.state) == {"q.arm", "hand.arm"} and spec.observation.cameras["front"] == (64, 64, 3)
    assert [c.mounted_on for c in spec.robot.cameras] == [None, "arm"] and "0.079" in spec.robot.notes
    with pytest.raises(ValueError):
        _Env(cameras=("nope",))


# ---------------------------------------------------------------- model export (MuJoCo, synthetic arm)
ARM_XML = """<mujoco model="toy"><compiler angle="radian"/><worldbody>
<body name="robot0_base" pos="0.3 0.1 0.9"><joint name="robot0_joint1" type="hinge" axis="0 0 1"/>
 <geom type="box" size="0.05 0.05 0.05"/>
 <body name="gripper0_right_right_gripper" pos="0 0 0.2">
  <body name="f1" pos="0 0.01 0.05"><joint name="gripper0_right_finger_joint1" type="slide" axis="0 1 0" range="0 0.04"/>
   <geom name="gripper0_right_finger1_pad_collision" type="box" size="0.01 0.002 0.01" pos="0 0 0.05"/></body>
  <body name="f2" pos="0 -0.01 0.05"><joint name="gripper0_right_finger_joint2" type="slide" axis="0 1 0" range="-0.04 0"/>
   <geom name="gripper0_right_finger2_pad_collision" type="box" size="0.01 0.002 0.01" pos="0 0 0.05"/></body>
 </body></body></worldbody>
<actuator><motor joint="robot0_joint1"/></actuator></mujoco>"""


def test_export_flips_the_second_finger_and_measures_the_tcp(tmp_path, monkeypatch):
    pytest.importorskip("mujoco")
    from hylo.envs.robosuite_contract.export import export_model, parallel_hand_geometry

    monkeypatch.setenv("HYLO_ASSETS", str(tmp_path))
    names = PandaNames(0)
    path = export_model(ARM_XML, "toy/arm", root_body="robot0_base", flip_joints=names.finger_joints[1:])
    assert path.is_file() and path.parent.parent == tmp_path / "toy" / "arm"
    text = path.read_text()
    assert "<actuator" not in text and 'range="0 0.04"' in text and 'axis="-0 -1 -0"' in text
    assert export_model(ARM_XML, "toy/arm", root_body="robot0_base", flip_joints=names.finger_joints[1:]) == path
    geo = parallel_hand_geometry(path, names.ee_body, names.finger_joints, names.pads, 0.04)
    T = np.asarray(geo["tcp_in_ee"])
    assert np.allclose(T[:3, 3], [0.0, 0.0, 0.1]) and np.allclose(T[:3, 0], [0.0, 0.0, 1.0])
    assert np.allclose(T[:3, 1], [0.0, 1.0, 0.0]) and geo["max_width"] == pytest.approx(2 * 0.05 - 2 * 0.002)


LIBERO_EXPORTS = sorted((Path.home() / ".cache" / "hylo" / "assets" / "libero" / "panda").glob("*/panda.xml"))


@pytest.mark.skipif(not LIBERO_EXPORTS, reason="no LIBERO Panda export in the asset cache")
def test_shared_tcp_measurement_reproduces_the_libero_adapters():
    """The LIBERO adapter (robosuite 1.4) could use this module: on its exported Panda the shared measurement gives
    the TCP its README documents (0.0934 m along the hand, 0.079 m between the open pads)."""
    from hylo.envs.robosuite_contract.export import parallel_hand_geometry

    geo = parallel_hand_geometry(LIBERO_EXPORTS[-1], "gripper0_right_gripper",
                                 ("gripper0_finger_joint1", "gripper0_finger_joint2"),
                                 ("gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision"), 0.04)
    assert geo["tcp_offset"] == pytest.approx(0.0934, abs=2e-4) and geo["max_width"] == pytest.approx(0.079, abs=1e-3)
    assert math.isclose(np.linalg.det(np.asarray(geo["tcp_in_ee"])[:3, :3]), 1.0, abs_tol=1e-6)
