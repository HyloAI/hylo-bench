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
from hylo.envs.mujoco import panda_table as pt  # noqa: E402
from hylo.envs.mujoco.manip_tasks import TASK_KINDS, manip_benchmark, manip_tasks  # noqa: E402
from hylo.envs.mujoco.oracle import Kinematics, PandaOracle  # noqa: E402

TASKS = {t.params["task"]: t for t in manip_tasks()}
USER_MESA = Path.home() / ".local" / "mesa" / "root" / "usr" / "lib" / "x86_64-linux-gnu"


@pytest.fixture(scope="module")
def assets():
    from hylo.envs.mujoco.assets import panda_assets

    try:
        return panda_assets()
    except RuntimeError as exc:                     # offline and not cached
        pytest.skip(str(exc))


@pytest.fixture
def env(assets):
    e = pt.PandaTableEnv(cameras=[], privileged_obs=True)
    yield e
    e.close()


def home_action() -> np.ndarray:
    return np.array([*pt.HOME, 1.0])


def test_spec_is_cheap_and_carries_the_robot_contract():
    spec = pt.PandaTableEnv().spec()                # no simulator, no download
    assert EnvSpec.model_validate_json(spec.model_dump_json()) == spec
    robot = spec.robot
    assert isinstance(robot, RobotSpec) and robot.action_dim == 8 == spec.action.shape[0]
    arm = robot.arm("arm")
    assert arm.joints == tuple(f"joint{i}" for i in range(1, 8)) and arm.ee_frame == "hand"
    assert arm.model.endswith("franka_emika_panda/panda.xml") and Path(arm.model).is_absolute()
    assert arm.action_slice == (0, 7) and arm.hand_slice == (7, 8)
    assert all(lo < h < hi for lo, h, hi in zip(arm.lower, arm.home, arm.upper))
    assert arm.hand.kind == "parallel" and arm.hand.open == (1.0,) and arm.hand.closed == (0.0,)
    R = np.asarray(arm.hand.tcp_in_ee)[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
    assert [c.name for c in robot.cameras] == ["front", "side", "wrist"] and robot.cameras[2].mounted_on == "arm"
    assert (robot.cameras[0].width, robot.cameras[0].height) == (320, 240)
    assert spec.observation.cameras["front"] == (240, 320, 3)
    assert "table top" in robot.notes and "TCP" in robot.notes
    assert pt.PandaTableEnv(cameras=["wrist"], width=160, height=120).spec().observation.cameras == {"wrist": (120, 160, 3)}
    with pytest.raises(ValueError):
        pt.PandaTableEnv(cameras=["top"])


def test_assets_match_the_recorded_digests(assets):
    from hylo.envs.mujoco.assets import verify

    assert (assets / "LICENSE").is_file() and verify(assets) == []


def test_tcp_sits_between_the_pads_with_the_convention_axes(env):
    env.reset(TaskInstance(spec=TASKS["pick_place"], seed=0))
    m, d = env.model, env.data
    pads = [d.geom_xpos[m.geom(f"{pt.PREFIX}{f}_finger_pad").id] for f in ("left", "right")]
    T = env.tcp_pose()
    assert np.linalg.norm(T[:3, 3] - (pads[0] + pads[1]) / 2) < 0.003
    closing = (pads[0] - pads[1]) / np.linalg.norm(pads[0] - pads[1])
    assert abs(abs(closing @ T[:3, 1]) - 1.0) < 1e-3                       # +y: closing axis
    hand = d.xpos[env._hand]
    assert (T[:3, 3] - hand) @ T[:3, 0] > 0.09                             # +x: from the hand towards the pads
    # the kinematics tools (ArmSpec.model + base_in_world + tcp_in_ee) agree with the simulator
    kin = Kinematics(env.spec().robot.arms[0])
    assert np.allclose(kin.fk(env.data.qpos[env._qadr]), T, atol=1e-6)


def test_observation_layout_and_measured_hand(env):
    obs = env.reset(TaskInstance(spec=TASKS["stack"], seed=3))
    assert set(obs.state) == {"q.arm", "hand.arm"} and obs.state["q.arm"].shape == (7,)
    np.testing.assert_allclose(obs.state["q.arm"], pt.HOME, atol=0.02)
    assert obs.state["hand.arm"][0] > 0.95 and obs.images == {} and obs.instruction
    priv = obs.extra["privileged"]
    assert priv["task"] == "stack" and set(priv["goal"]["order"]) <= set(priv["objects"])
    json.dumps(priv)                                                        # JSON-ready
    closed = home_action()
    closed[7] = 0.0
    for _ in range(15):
        obs = env.step(Action(data=closed)).observation
    assert obs.state["hand.arm"][0] < 0.05                                  # nothing between the pads


def test_seeded_layouts_are_deterministic_and_vary(env):
    a = env.reset(TaskInstance(spec=TASKS["pick_place"], seed=11))
    pa = a.extra["privileged"]["objects"]
    b = env.reset(TaskInstance(spec=TASKS["pick_place"], seed=11))
    assert a.instruction == b.instruction and pa == b.extra["privileged"]["objects"]
    for _ in range(5):
        ra = env.step(Action(data=home_action()))
    env.reset(TaskInstance(spec=TASKS["pick_place"], seed=11))
    for _ in range(5):
        rb = env.step(Action(data=home_action()))
    np.testing.assert_array_equal(ra.observation.state["q.arm"], rb.observation.state["q.arm"])
    texts = {env.reset(TaskInstance(spec=TASKS[k], seed=s)).instruction for k in TASK_KINDS for s in range(5)}
    assert len(texts) >= 20


def test_snapshot_restores_episode_and_model_state(env):
    env.reset(TaskInstance(spec=TASKS["press_by_number"], seed=2))
    for _ in range(3):
        env.step(Action(data=home_action()))
    blob, before = env.snapshot(), env.privileged_state(json_ready=True)
    env.reset(TaskInstance(spec=TASKS["sort_by_colour"], seed=5))         # other objects, colours, digits
    env.restore(blob)
    after = env.privileged_state(json_ready=True)
    np.testing.assert_allclose(after.pop("tcp"), before.pop("tcp"), atol=1e-12)
    assert after == before and env.t == 3
    segs = env.items[env.ep["goal"]["sequence"][0]]["segs"]
    lit = {s for s, g in segs.items() if env.model.geom_pos[g, 2] > 0}
    assert lit == set(pt.DIGITS[env.ep["goal"]["digits"][0]])


def test_predicates_need_objects_at_rest_and_presses_in_order(env):
    env.reset(TaskInstance(spec=TASKS["pick_place"], seed=4))
    obj, plate = env.ep["goal"]["object"], env.ep["goal"]["target"]
    p = env.pose(plate)[0]
    env._place(obj, [p[0], p[1], 2 * pt.PLATE_HALF + pt.BLOCK_HALF + 0.03], 0.0)   # dropped from 3 cm: moving first
    r = env.step(Action(data=home_action()))
    assert not r.success
    for _ in range(20):
        r = env.step(Action(data=home_action()))
        if r.terminated:
            break
    assert r.success and r.terminated

    env.reset(TaskInstance(spec=TASKS["press_by_number"], seed=1))
    wrong = next(b for b in env.ep["items"] if b != env.ep["goal"]["sequence"][0])
    env.data.qpos[env.items[wrong]["press_qadr"]] = -pt.BTN_TRAVEL
    env.model.jnt_stiffness[env.model.joint(f"{wrong}/press").id] = 0.0    # held down (no spring) for the step
    r = env.step(Action(data=home_action()))
    assert r.terminated and r.success is False and "out of order" in r.info["fail_reason"]


def test_benchmark_has_six_tasks_and_disjoint_rfc0002_splits():
    b = manip_benchmark()
    assert b.id == "mujoco/manip-v0" and len(b.tasks) == 6
    assert {"search", "search_val", "gate", "heldout", "validation"} <= set(b.splits)
    seeds = [s for v in b.splits.values() for s in v]
    assert len(seeds) == len(set(seeds))
    assert set(manip_benchmark(protocol="standard").splits) == {"train", "validation", "heldout"}
    assert [t.id for t in manip_benchmark(tasks=["stack"]).tasks] == ["mujoco/manip_stack"]


@pytest.mark.parametrize("kind", TASK_KINDS)
def test_privileged_oracle_solves_each_task(env, kind):
    runner = EpisodeRunner(env, PandaOracle())
    for seed in manip_benchmark().splits["search"][:3]:
        out = runner.run(TaskInstance(spec=TASKS[kind], seed=seed)).outcome
        assert out.status == "success", (kind, seed, out.termination_reason, env.ep["instruction"])


def test_oracle_needs_privileged_observations(assets):
    e = pt.PandaTableEnv(cameras=[])
    out = EpisodeRunner(e, PandaOracle()).run(TaskInstance(spec=TASKS["pick_place"], seed=0)).outcome
    assert out.status == "error" and "privileged" in out.termination_reason


def test_conformance_without_cameras(assets):
    from hylo.testing.conformance import check_agent, check_environment

    def factory():
        return pt.PandaTableEnv(cameras=[], privileged_obs=True)

    rep = check_environment(factory, TASKS["put_in_bowl"])
    assert rep.ok, rep.summary()
    e = factory()
    rep = check_agent(PandaOracle(), e, TASKS["pick_place"])
    e.close()
    assert rep.ok, rep.summary()


CAMERA_CHECK = r"""
import json, numpy as np
from hylo.core.task import TaskInstance
from hylo.core.types import Action
from hylo.envs.mujoco.manip_tasks import manip_tasks
from hylo.envs.mujoco.panda_table import PandaTableEnv, HOME
from hylo.testing.conformance import check_environment
env = PandaTableEnv(segmentation=True, privileged_obs=True)
obs = env.reset(TaskInstance(spec=manip_tasks(["stack"])[0], seed=4))
obs = env.step(Action(data=np.array([*HOME, 1.0]))).observation
out = {"keys": sorted(obs.images), "hits": 0, "probes": 0}
for cam in ("front", "side", "wrist"):
    rgb, depth, seg = obs.images[cam], obs.images[cam + ".depth"], obs.images[cam + ".seg"]
    out[cam] = [list(rgb.shape), str(rgb.dtype), list(depth.shape), str(depth.dtype), str(seg.dtype),
                float(depth[depth > 0].min()), float(depth.max())]
    assert set(np.unique(seg).tolist()) - {0} <= {int(k) for k in obs.extra["segments"]}
    K, T = np.array(obs.extra["cameras"][cam]["K"]), np.array(obs.extra["cameras"][cam]["T_world_cam"])
    for o in obs.extra["privileged"]["objects"].values():
        top = np.array(o["pos"]) + [0, 0, o["half_size"][2]]
        pc = np.linalg.inv(T) @ np.append(top, 1)
        u, v, w = K @ pc[:3]
        u, v = int(round(u / w)), int(round(v / w))
        if 0 <= u < 320 and 0 <= v < 240:
            out["probes"] += 1
            out["hits"] += int(seg[v, u] == o["instance"] and abs(depth[v, u] - pc[2]) < 0.01)
out["labels"] = sorted(obs.extra["segments"].values())
rep = check_environment(lambda: PandaTableEnv(segmentation=True), manip_tasks(["press_by_number"])[0])
out["conformance"] = rep.ok
out["summary"] = rep.summary()
env.close()
print("RESULT " + json.dumps(out))
"""


def _render_env() -> dict[str, str] | None:
    if os.environ.get("MUJOCO_GL") in ("egl", "osmesa"):
        return dict(os.environ)
    if (USER_MESA / "libOSMesa.so.8").exists():
        return {**os.environ, "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa",
                "LD_LIBRARY_PATH": os.pathsep.join(filter(None, [str(USER_MESA), os.environ.get("LD_LIBRARY_PATH")]))}
    return None


def test_rgbd_cameras_segmentation_and_poses(assets):
    render_env = _render_env()
    if render_env is None:
        pytest.skip("needs MuJoCo rendering (MUJOCO_GL=egl or osmesa)")
    proc = subprocess.run([sys.executable, "-c", CAMERA_CHECK], env=render_env, capture_output=True, text=True,
                          timeout=600)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
    assert line, proc.stderr[-2000:]
    out = json.loads(line[len("RESULT "):])
    assert out["conformance"], out["summary"]
    for cam in ("front", "side", "wrist"):
        assert {cam, f"{cam}.depth", f"{cam}.seg"} <= set(out["keys"])
        shape, dtype, dshape, ddtype, sdtype, dmin, dmax = out[cam]
        assert shape == [240, 320, 3] and dtype == "uint8" and dshape == [240, 320] and ddtype == "float32"
        assert sdtype == "int32" and 0.02 < dmin < dmax < 10.0
    assert "red block" in out["labels"] or any(lab.endswith("block") for lab in out["labels"])
    assert {"table", "robot"} <= set(out["labels"])
    assert out["probes"] >= 8 and out["hits"] >= out["probes"] - 1     # K and T_world_cam land on the objects
