import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from hylo.core.kernel import EpisodeRunner  # noqa: E402
from hylo.core.robot import RobotSpec  # noqa: E402
from hylo.core.task import TaskInstance  # noqa: E402
from hylo.core.types import Action, EnvSpec  # noqa: E402
from hylo.envs.mujoco import g1_table as gt  # noqa: E402
from hylo.envs.mujoco.g1_oracle import G1Oracle  # noqa: E402
from hylo.envs.mujoco.humanoid_tasks import MAX_STEPS, TASK_KINDS, humanoid_benchmark, humanoid_tasks  # noqa: E402
from hylo.envs.mujoco.oracle import Kinematics  # noqa: E402

TASKS = {t.params["task"]: t for t in humanoid_tasks()}
USER_MESA = Path.home() / ".local" / "mesa" / "root" / "usr" / "lib" / "x86_64-linux-gnu"


@pytest.fixture(scope="module")
def assets():
    from hylo.envs.mujoco.assets import G1_HANDS, assets

    try:
        return assets(G1_HANDS)
    except RuntimeError as exc:
        pytest.skip(str(exc))


@pytest.fixture
def env(assets):
    e = gt.G1TableEnv(cameras=[], privileged_obs=True)
    yield e
    e.close()


def test_spec_is_cheap_and_carries_two_dexterous_arms():
    spec = gt.G1TableEnv().spec()                 # no simulator, no download
    assert EnvSpec.model_validate_json(spec.model_dump_json()) == spec
    robot = spec.robot
    assert isinstance(robot, RobotSpec) and robot.action_dim == 28 == spec.action.shape[0] and not robot.mobile_base
    for i, side in enumerate(("left", "right")):
        a = robot.arm(side)
        assert len(a.joints) == 7 and a.ee_frame == f"{side}_wrist_yaw_link"
        assert a.action_slice == (14 * i, 14 * i + 7) and a.hand_slice == (14 * i + 7, 14 * i + 14)
        assert a.model.endswith("unitree_g1/g1_with_hands.xml")
        h = a.hand
        assert h.kind == "dexterous" and len(h.open) == len(h.closed) == len(h.joints) == 7
        assert {"power", "pinch", "point", "flat"} <= set(h.presets)
        lo, hi = spec.action.low[a.hand_slice[0]:a.hand_slice[1]], spec.action.high[a.hand_slice[0]:a.hand_slice[1]]
        for cmd in (h.open, *h.presets.values()):
            assert all(lo_ - 1e-9 <= c <= hi_ + 1e-9 for c, lo_, hi_ in zip(cmd, lo, hi))
        R = np.asarray(h.tcp_in_ee)[:3, :3]
        assert np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
    assert [c.name for c in robot.cameras] == ["head", "front"]
    assert "pelvis fixed" in robot.notes


def test_hands_mirror_each_other(env):
    """The right hand's presets mirror the left's, and so does the TCP (thumb towards the body's midline)."""
    robot = env.spec().robot
    left, right = robot.arm("left").hand, robot.arm("right").hand
    assert left.presets["power"][3] < 0 < right.presets["power"][3]           # index curls the other way
    Tl, Tr = np.asarray(left.tcp_in_ee), np.asarray(right.tcp_in_ee)
    assert np.allclose(Tl[:3, 1], Tr[:3, 1])                                  # closing: along the fingers (+x)
    assert np.allclose(Tl[:3, 0], -Tr[:3, 0])                                 # approach: into each palm


def test_tcp_between_thumb_and_fingers_and_kinematics_agree(env):
    env.reset(TaskInstance(spec=TASKS["pick_place"], seed=0))
    m, d = env.model, env.data
    robot = env.spec().robot
    for side in ("left", "right"):
        T = env.tcp_pose(side)
        thumb = d.xpos[m.body(f"{side}_hand_thumb_2_link").id]
        index = d.xpos[m.body(f"{side}_hand_index_1_link").id]
        palm = d.xpos[m.body(f"{side}_wrist_yaw_link").id]
        assert (index - thumb) @ T[:3, 1] > 0.05                              # +y from the thumb to the fingers
        assert (T[:3, 3] - palm) @ T[:3, 0] > 0.0                             # +x away from the palm
        q = d.qpos[env.arm_io(side).qadr]
        assert np.allclose(Kinematics(robot.arm(side)).fk(q), T, atol=1e-4)


def test_pelvis_fixed_and_only_arms_and_fingers_move(env):
    m = env.model
    robot_joints = [m.joint(j).name for j in range(m.njnt) if env._is_robot_body(m, int(m.jnt_bodyid[j]))]
    assert len(robot_joints) == 28 and all(("shoulder" in n or "elbow" in n or "wrist" in n or "hand" in n)
                                           for n in robot_joints)
    obs = env.reset(TaskInstance(spec=TASKS["press_button"], seed=1))
    pelvis = env.data.xpos[m.body("pelvis").id].copy()
    a = env.hold_action(None)
    for _ in range(10):
        env.step(Action(data=a))
    assert np.allclose(env.data.xpos[m.body("pelvis").id], pelvis)
    assert set(obs.state) == {"q.left", "hand.left", "q.right", "hand.right"} and obs.state["hand.left"].shape == (7,)
    np.testing.assert_allclose(obs.state["q.right"], gt.HOME["right"], atol=0.03)
    json.dumps(obs.extra["privileged"])


def test_layouts_vary_and_two_hands_split_the_table(env):
    texts = set()
    for s in range(10):
        obs = env.reset(TaskInstance(spec=TASKS["two_hands"], seed=s))
        objs = obs.extra["privileged"]["objects"]
        for blk, plate in env.ep["goal"]["assign"].items():
            assert np.sign(objs[blk]["pos"][1]) == np.sign(objs[plate]["pos"][1])
        texts.add(obs.instruction)
        texts.add(env.reset(TaskInstance(spec=TASKS["pick_place"], seed=s)).instruction)
    assert len(texts) >= 12


def test_snapshot_restores_episode_and_model_state(env):
    env.reset(TaskInstance(spec=TASKS["pick_place"], seed=2))
    a = env.hold_action(None)
    for _ in range(3):
        env.step(Action(data=a))
    blob = env.snapshot()
    ra = [env.step(Action(data=a)).observation.state["hand.left"] for _ in range(4)]
    env.reset(TaskInstance(spec=TASKS["press_button"], seed=7))
    env.restore(blob)
    rb = [env.step(Action(data=a)).observation.state["hand.left"] for _ in range(4)]
    np.testing.assert_array_equal(np.array(ra), np.array(rb))


def test_benchmark_splits():
    b = humanoid_benchmark()
    assert b.id == "mujoco/humanoid-v0" and {t.params["task"] for t in b.tasks} == set(TASK_KINDS)
    seen = set()
    for name, seeds in b.splits.items():
        assert not (seen & set(seeds)), name
        seen |= set(seeds)
    assert len(b.splits["validation"]) == 20 and all(t.max_steps == MAX_STEPS[t.params["task"]] for t in b.tasks)


@pytest.mark.parametrize("kind", TASK_KINDS)
def test_oracle_solves_each_task(env, kind):
    runner = EpisodeRunner(env, G1Oracle())
    wins = 0
    seeds = humanoid_benchmark().splits["search"][:3]
    for seed in seeds:
        wins += runner.run(TaskInstance(spec=TASKS[kind], seed=seed)).outcome.status == "success"
    assert wins >= 2, (kind, wins)


def test_one_hand_after_the_other_misses_the_two_hands_step_limit(env):
    out = EpisodeRunner(env, G1Oracle(sequential=True)).run(
        TaskInstance(spec=TASKS["two_hands"], seed=humanoid_benchmark().splits["search"][0])).outcome
    assert out.status == "failure" and out.steps == MAX_STEPS["two_hands"]


def test_wrong_button_fails_at_once(env):
    env.reset(TaskInstance(spec=TASKS["press_button"], seed=3))
    target = env.ep["goal"]["button"]
    other = next(b for b in env.ep["items"] if b.startswith("button") and b != target)
    from hylo.envs.mujoco.humanoid_tasks import evaluate

    env.data.qpos[env.items[other]["press_qadr"]] = -0.007                # pushed down
    env._after_step()
    ok, fail, _ = evaluate(env)
    assert not ok and "pressed" in fail and env.ep["state"]["pressed"] == [other]


def test_conformance_without_cameras(assets):
    from hylo.testing.conformance import check_environment

    rep = check_environment(lambda: gt.G1TableEnv(cameras=[], privileged_obs=True), TASKS["two_hands"])
    assert rep.ok, rep.summary()


def test_cameras_depth_and_segmentation(assets):
    env = dict(os.environ)
    if env.get("MUJOCO_GL") not in ("egl", "osmesa"):
        if not USER_MESA.is_dir():
            pytest.skip("no OpenGL backend")
        env.update(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa",
                   LD_LIBRARY_PATH=f"{USER_MESA}:{env.get('LD_LIBRARY_PATH', '')}")
    code = r"""
import numpy as np
from hylo.core.task import TaskInstance
from hylo.envs.mujoco.g1_table import G1TableEnv
from hylo.envs.mujoco.humanoid_tasks import humanoid_tasks
spec = {t.params["task"]: t for t in humanoid_tasks()}["two_hands"]
e = G1TableEnv(segmentation=True, width=160, height=120)
obs = e.reset(TaskInstance(spec=spec, seed=1))
out = {}
for c in e.camera_names:
    seg = obs.images[c + ".seg"]
    out[c] = sorted(obs.extra["segments"].get(str(i), "?") for i in np.unique(seg) if i)
    assert (obs.images[c + ".depth"] > 0).any()
print("OUT", out)
"""
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    seen = eval(next(ln for ln in r.stdout.splitlines() if ln.startswith("OUT"))[4:])
    for cam in ("head", "front"):
        assert "table" in seen[cam] and sum("block" in s for s in seen[cam]) == 2 and sum("plate" in s for s in seen[cam]) == 2
    assert "robot" in seen["front"]
