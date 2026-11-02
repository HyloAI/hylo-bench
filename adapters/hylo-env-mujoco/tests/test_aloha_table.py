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
from hylo.envs.mujoco import aloha_table as at  # noqa: E402
from hylo.envs.mujoco.aloha_oracle import AlohaOracle  # noqa: E402
from hylo.envs.mujoco.bimanual_tasks import (MAX_STEPS, TASK_KINDS, bimanual_benchmark, bimanual_tasks,  # noqa: E402
                                             tilt)
from hylo.envs.mujoco.oracle import Kinematics  # noqa: E402

TASKS = {t.params["task"]: t for t in bimanual_tasks()}
USER_MESA = Path.home() / ".local" / "mesa" / "root" / "usr" / "lib" / "x86_64-linux-gnu"


@pytest.fixture(scope="module")
def assets():
    from hylo.envs.mujoco.assets import ALOHA, assets

    try:
        return assets(ALOHA)
    except RuntimeError as exc:                     # offline and not cached
        pytest.skip(str(exc))


@pytest.fixture
def env(assets):
    e = at.AlohaTableEnv(cameras=[], privileged_obs=True)
    yield e
    e.close()


def test_spec_is_cheap_and_carries_two_arms():
    spec = at.AlohaTableEnv().spec()               # no simulator, no download
    assert EnvSpec.model_validate_json(spec.model_dump_json()) == spec
    robot = spec.robot
    assert isinstance(robot, RobotSpec) and robot.action_dim == 14 == spec.action.shape[0]
    left, right = robot.arm("left"), robot.arm("right")
    assert left.joints[0] == "left/waist" and right.ee_frame == "right/gripper_link"
    assert left.action_slice == (0, 6) and left.hand_slice == (6, 7)
    assert right.action_slice == (7, 13) and right.hand_slice == (13, 14)
    assert left.model == right.model and left.model.endswith("aloha/aloha.xml")
    for a in (left, right):
        assert all(lo < h < hi for lo, h, hi in zip(a.lower, a.home, a.upper))
        assert a.hand.kind == "parallel" and 0.05 < a.hand.max_width < 0.065
    assert [c.name for c in robot.cameras] == ["overhead", "front", "left_wrist", "right_wrist"]
    assert robot.cameras[2].mounted_on == "left" and robot.cameras[3].mounted_on == "right"
    assert "handed over" in robot.notes
    with pytest.raises(ValueError):
        at.AlohaTableEnv(cameras=["side"])


def test_assets_match_the_recorded_digests(assets):
    from hylo.envs.mujoco.assets import ALOHA, manifest, verify

    assert (assets / "LICENSE").is_file() and verify(assets, ALOHA) == []
    assert manifest(ALOHA)["commit"] == manifest("menagerie_panda")["commit"]


def test_tcp_between_the_pads_and_kinematics_agree(env):
    env.reset(TaskInstance(spec=TASKS["two_bins"], seed=0))
    m, d = env.model, env.data
    robot = env.spec().robot
    for side in at.ARM_NAMES:
        tips = [d.geom_xpos[m.geom(f"{side}/{f}_g0").id] for f in ("left", "right")]
        T = env.tcp_pose(side)
        mid = (tips[0] + tips[1]) / 2
        assert np.linalg.norm((T[:3, 3] - mid) - ((T[:3, 3] - mid) @ T[:3, 0]) * T[:3, 0]) < 0.006
        closing = (tips[0] - tips[1]) / np.linalg.norm(tips[0] - tips[1])
        assert abs(abs(closing @ T[:3, 1]) - 1.0) < 1e-2                 # +y: closing axis
        kin = Kinematics(robot.arm(side))
        q = d.qpos[env.arm_io(side).qadr]
        assert np.allclose(kin.fk(q), T, atol=1e-6)                      # the tools see the same TCP


def test_observation_layout_hands_and_privileged_state(env):
    obs = env.reset(TaskInstance(spec=TASKS["handover"], seed=3))
    assert set(obs.state) == {"q.left", "hand.left", "q.right", "hand.right"}
    np.testing.assert_allclose(obs.state["q.left"], at.HOME["left"], atol=0.02)
    assert obs.state["hand.left"][0] > 0.95 and obs.instruction
    priv = obs.extra["privileged"]
    assert priv["task"] == "handover" and priv["goal"]["object"] in priv["objects"]
    json.dumps(priv)
    a = env.hold_action(None)
    a[6] = a[13] = 0.0
    for _ in range(15):
        obs = env.step(Action(data=a)).observation
    assert obs.state["hand.left"][0] < 0.05 and obs.state["hand.right"][0] < 0.05   # pads touch: 0


def test_layouts_need_two_arms_and_vary(env):
    texts = set()
    for s in range(12):
        obs = env.reset(TaskInstance(spec=TASKS["handover"], seed=s))
        g, objs = env.ep["goal"], obs.extra["privileged"]["objects"]
        bar, zone = objs[g["object"]], objs[g["zone"]]
        assert np.sign(bar["pos"][0]) != np.sign(zone["pos"][0])                 # opposite sides
        assert abs(bar["pos"][0]) > at.REACH_X + 0.1 and abs(zone["pos"][0]) > at.REACH_X + 0.15
        assert (g["from"] == "left") == (bar["pos"][0] < 0)
        texts.add(obs.instruction)
        obs = env.reset(TaskInstance(spec=TASKS["two_bins"], seed=s))
        for blk, b in env.ep["goal"]["assign"].items():
            p, q = obs.extra["privileged"]["objects"][blk]["pos"], obs.extra["privileged"]["objects"][b]["pos"]
            assert np.sign(p[0]) == np.sign(q[0]) and abs(p[0]) < 0.36       # same side, clear of the bases
            assert abs(p[2] - env.ep["items"][blk]["half_size"][2]) < 0.003   # resting on the table
        texts.add(obs.instruction)
    assert len(texts) >= 15
    sizes = {round(env.reset(TaskInstance(spec=TASKS["handover"], seed=s)) and env.ep["items"]["bar0"]["size"][1], 3)
             for s in range(6)}
    assert len(sizes) > 3


def test_box_is_too_wide_for_one_gripper(env):
    env.reset(TaskInstance(spec=TASKS["lift_box"], seed=0))
    half = env.ep["items"]["liftbox0"]["half_size"]
    assert 2 * min(half[0], half[1]) > at.MAX_WIDTH + 0.03


def test_snapshot_restores_episode_and_model_state(env):
    env.reset(TaskInstance(spec=TASKS["handover"], seed=2))
    a = env.hold_action(None)
    for _ in range(3):
        env.step(Action(data=a))
    blob = env.snapshot()
    size = env.model.geom_size[env.items["bar0"]["geoms"][0]].copy()
    ra = [env.step(Action(data=a)).observation.state["q.left"] for _ in range(4)]
    env.reset(TaskInstance(spec=TASKS["two_bins"], seed=9))         # different items, colours and sizes
    env.restore(blob)
    assert np.allclose(env.model.geom_size[env.items["bar0"]["geoms"][0]], size)
    rb = [env.step(Action(data=a)).observation.state["q.left"] for _ in range(4)]
    np.testing.assert_array_equal(np.array(ra), np.array(rb))


def test_benchmark_splits_are_disjoint_and_seeded():
    b = bimanual_benchmark()
    assert b.id == "mujoco/bimanual-v0" and {t.params["task"] for t in b.tasks} == set(TASK_KINDS)
    seen = set()
    for name, seeds in b.splits.items():
        assert not (seen & set(seeds)), name
        seen |= set(seeds)
    assert len(b.splits["validation"]) == 20
    assert all(t.max_steps == MAX_STEPS[t.params["task"]] for t in b.tasks)
    with pytest.raises(ValueError):
        bimanual_tasks(["juggle"])


@pytest.mark.parametrize("kind", TASK_KINDS)
def test_oracle_solves_each_task(env, kind):
    runner = EpisodeRunner(env, AlohaOracle())
    for seed in bimanual_benchmark().splits["search"][:2]:
        out = runner.run(TaskInstance(spec=TASKS[kind], seed=seed)).outcome
        assert out.status == "success", (kind, seed, out.termination_reason, env.ep["instruction"])


def test_one_arm_after_the_other_misses_the_two_bins_step_limit(env):
    runner = EpisodeRunner(env, AlohaOracle(sequential=True))
    out = runner.run(TaskInstance(spec=TASKS["two_bins"], seed=bimanual_benchmark().splits["search"][0])).outcome
    assert out.status == "failure" and out.steps == MAX_STEPS["two_bins"]


def test_lifting_one_handle_tips_the_box(env):
    """One arm alone: grasp one handle and lift; the box tips past the level limit (and never succeeds)."""
    ag = AlohaOracle()

    class Ctx:
        env_spec = env.spec()

        def emit(self, *a):
            pass

    ag.reset(Ctx())
    obs = env.reset(TaskInstance(spec=TASKS["lift_box"], seed=4))
    ag._obs = obs
    worst, success = 0.0, False

    def execute(rows):
        nonlocal worst, success, obs
        for a in ag._merge({"left": rows}):
            res = env.step(Action(data=a))
            obs = ag._obs = res.observation
            success |= bool(res.success)
            worst = max(worst, tilt(env, "liftbox0"))

    from hylo.envs.mujoco.multi_oracle import topdown

    o = obs.extra["privileged"]["objects"]["liftbox0"]
    c, R = np.asarray(o["pos"]), np.asarray(o["R"])
    fx = at.LIFT_BODY[0] + at.LIFT_FIN[0]
    fin = min((c + R @ np.array([k * fx, 0.0, at.LIFT_FIN_Z]) for k in (-1.0, 1.0)), key=lambda f: f[0])
    p = np.array([fin[0], fin[1], fin[2] + at.LIFT_FIN[2] - 0.016])
    Rt = topdown(ag.closing_yaw("left", "liftbox0", fin[:2]))
    left = ag.arms["left"]
    for rows in left.travel(p + [0.0, 0.0, 0.08], Rt, hand=1.0, lift_to=0.16):
        execute(rows)
    execute(left.line(p, Rt, speed=0.08, settle=6))
    s = ag.arms["left"]
    T = s.tcp()                                        # planned at the grasp pose: lift 8 cm straight up
    up = s.line(T[:3, 3] + [0.0, 0.0, 0.08], T[:3, :3], speed=0.06, hand=0.0)
    s.cmd = None
    s.cmd = np.concatenate([s.q(), [1.0]])
    execute(s.gripper(0.0, 16))
    execute(up)
    execute(np.repeat(up[-1:], 20, axis=0))
    assert not success and worst > 0.2


def test_conformance_without_cameras(assets):
    from hylo.testing.conformance import check_environment

    rep = check_environment(lambda: at.AlohaTableEnv(cameras=[], privileged_obs=True), TASKS["lift_box"])
    assert rep.ok, rep.summary()


def _gl_env() -> dict | None:
    env = dict(os.environ)
    if env.get("MUJOCO_GL") in ("egl", "osmesa"):
        return env
    if USER_MESA.is_dir():
        env.update(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa",
                   LD_LIBRARY_PATH=f"{USER_MESA}:{env.get('LD_LIBRARY_PATH', '')}")
        return env
    return None


def test_cameras_depth_and_segmentation(assets):
    """Rendering runs in a subprocess: the GL backend is chosen when mujoco is first imported."""
    env = _gl_env()
    if env is None:
        pytest.skip("no OpenGL backend (MUJOCO_GL=egl/osmesa or ~/.local/mesa)")
    code = r"""
import numpy as np
from hylo.core.task import TaskInstance
from hylo.envs.mujoco.aloha_table import AlohaTableEnv
from hylo.envs.mujoco.bimanual_tasks import bimanual_tasks
spec = {t.params["task"]: t for t in bimanual_tasks()}["two_bins"]
e = AlohaTableEnv(segmentation=True, width=160, height=120)
obs = e.reset(TaskInstance(spec=spec, seed=1))
out = {}
for c in e.camera_names:
    seg, depth = obs.images[c + ".seg"], obs.images[c + ".depth"]
    out[c] = sorted(obs.extra["segments"].get(str(i), "?") for i in np.unique(seg) if i)
    assert obs.images[c].shape == (120, 160, 3) and depth.dtype == np.float32 and (depth > 0).any()
    K, T = np.array(obs.extra["cameras"][c]["K"]), np.array(obs.extra["cameras"][c]["T_world_cam"])
    assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6)
# a table pixel of the overhead camera back-projects to the table plane
d = obs.images["overhead.depth"]; s = obs.images["overhead.seg"]
v, u = np.argwhere(s == 1)[len(np.argwhere(s == 1)) // 2]
K = np.array(obs.extra["cameras"]["overhead"]["K"]); z = d[v, u]; p = np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
T = np.array(obs.extra["cameras"]["overhead"]["T_world_cam"])
print("OUT", out, round(float((T[:3, :3] @ p + T[:3, 3])[2]), 4))
"""
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("OUT"))
    seen = eval(line[4:line.rindex("}") + 1])
    assert {"table", "robot"} <= set(seen["overhead"]) and any("block" in s for s in seen["overhead"])
    assert any("bin" in s for s in seen["front"])
    assert abs(float(line.split()[-1])) < 0.005
