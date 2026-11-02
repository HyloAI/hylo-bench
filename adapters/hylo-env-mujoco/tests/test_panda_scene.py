import json
import math

import numpy as np
import pytest
from pydantic import TypeAdapter

pytest.importorskip("mujoco")

from hylo.core.scene import Predicate  # noqa: E402
from hylo.core.task import TaskInstance  # noqa: E402
from hylo.core.types import Action, EnvSpec  # noqa: E402
from hylo.envs.mujoco import panda_table as pt  # noqa: E402
from hylo.envs.mujoco.panda_scene import PandaSceneEnv, load_scene, scene_benchmark, scene_tasks  # noqa: E402

PRED = TypeAdapter(Predicate)


def box(name, pos=None, region=None, **kw):
    o = {"name": name, "label": kw.pop("label", name.replace("_", " ")), "shape": "box", "size": [0.04, 0.04, 0.04],
         "mass": 0.05, **kw}
    if region is not None:
        o["region"] = region
    else:
        o["pose"] = {"pos": pos}
    return o


OPS_SCENE = {
    "name": "ops", "instruction": "move things around", "max_steps": 300,
    "objects": [
        box("a", [-0.1, -0.2], colour="red"), box("b", [0.0, -0.2], colour="blue"), box("c", [0.1, -0.2], colour="green"),
        {"name": "tray", "label": "tray", "shape": "tray", "size": [0.15, 0.15, 0.04], "fixed": True, "pose": {"pos": [0.05, 0.15]}},
        {"name": "rail", "label": "rail", "shape": "box", "size": [0.2, 0.04, 0.02], "fixed": True, "pose": {"pos": [-0.15, 0.15]}},
        {"name": "slider", "label": "slider", "parent": "rail", "shape": "box", "size": [0.04, 0.04, 0.03],
         "pose": {"pos": [-0.06, 0, 0.025]}, "joint": {"type": "slide", "axis": [1, 0, 0], "range": [0, 0.1], "damping": 5}},
        box("loose", region={"x": [-0.2, -0.05], "y": [0.25, 0.32], "yaw_deg": [-40, 40]}, colour="yellow"),
    ],
    "success": {"op": "sequence", "terms": [{"op": "joint_at_least", "object": "slider", "value": 0.05},
                                            {"op": "inside", "a": "a", "b": "tray"}]},
}


@pytest.fixture(scope="module")
def assets():
    from hylo.envs.mujoco.assets import panda_assets

    try:
        return panda_assets()
    except RuntimeError as exc:
        pytest.skip(str(exc))


@pytest.fixture
def scene_file(tmp_path):
    p = tmp_path / "ops.json"
    p.write_text(json.dumps(OPS_SCENE))
    return p


@pytest.fixture
def env(assets, scene_file):
    e = PandaSceneEnv(scene=str(scene_file), cameras=[], privileged_obs=True)
    yield e
    e.close()


def task(path, seed=0):
    return TaskInstance(spec=scene_tasks(path)[0], seed=seed)


def hold(env, n=10):
    a = np.array([*pt.HOME, 1.0])
    r = None
    for _ in range(n):
        r = env.step(Action(data=a))
    return r


def holds(env, pred) -> bool:
    return env._eval(PRED.validate_python(pred), "t", {})


def put(env, name, xyz, yaw_deg=0.0, roll_deg=0.0):
    from hylo.envs.mujoco.panda_scene import _quat_euler

    env._set_pose(name, xyz, _quat_euler(math.radians(roll_deg), 0.0, math.radians(yaw_deg)))


def test_spec_is_cheap_and_shares_the_panda_contract():
    spec = PandaSceneEnv().spec()                                  # no scene, no simulator
    assert spec.name == "mujoco.panda_scene" and EnvSpec.model_validate_json(spec.model_dump_json()) == spec
    base = pt.PandaTableEnv().spec()
    assert spec.robot.arms == base.robot.arms and spec.robot.cameras == base.robot.cameras
    assert spec.action == base.action and spec.observation == base.observation
    assert "scene file" in spec.robot.notes and "serves object centres" in spec.robot.notes
    with pytest.raises(ValueError, match="needs a scene"):
        PandaSceneEnv(cameras=[]).model


def test_loads_steps_and_reports_privileged_state(env, scene_file):
    obs = env.reset(task(scene_file))
    assert set(obs.state) == {"q.arm", "hand.arm"} and obs.instruction == "move things around"
    np.testing.assert_allclose(obs.state["q.arm"], pt.HOME, atol=0.02)
    priv = obs.extra["privileged"]
    json.dumps(priv)
    assert set(priv["objects"]) == {"a", "b", "c", "tray", "rail", "slider", "loose"}
    a = priv["objects"]["a"]
    assert abs(a["pos"][2] - 0.02) < 0.003 and a["movable"] and np.allclose(a["extent"], 0.04, atol=1e-6)
    assert priv["objects"]["tray"]["interior"]["half"][0] == pytest.approx(0.075 - 0.006)
    assert priv["objects"]["slider"]["joint"] == {"type": "slide", "value": pytest.approx(0, abs=1e-4),
                                                   "range": [0.0, 0.1], "units": "m"}
    assert priv["penetration"] == [] and priv["start_contacts"] == [] and priv["placement_problems"] == []
    assert env.segments()[str(env.instance["slider"])] == "slider"
    r = hold(env, 20)
    assert not r.terminated and r.success is False and r.info == {"done": 0, "of": 2}
    assert np.linalg.norm(np.subtract(env.pose("a")[0], a["pos"])) < 0.002          # everything rests


def test_regions_vary_with_the_seed_and_fixed_poses_do_not(env, scene_file):
    p1 = env.reset(task(scene_file, 1)).extra["privileged"]["objects"]
    p1b = env.reset(task(scene_file, 1)).extra["privileged"]["objects"]
    p2 = env.reset(task(scene_file, 2)).extra["privileged"]["objects"]
    assert p1 == p1b
    assert p1["loose"]["pos"] != p2["loose"]["pos"] and p1["tray"]["pos"] == p2["tray"]["pos"]
    for p in (p1, p2):
        x, y = p["loose"]["pos"][:2]
        assert -0.2 <= x <= -0.05 and 0.25 <= y <= 0.32


def test_snapshot_restores_state_and_sequence_progress_across_scenes(env, scene_file, tmp_path):
    env.reset(task(scene_file))
    env.data.qpos[env.items["slider"]["jqadr"]] = 0.08
    r = hold(env, 3)
    assert r.info["done"] == 1 and env.ep["latch"] == {"0": 1}                # first term latched
    blob, before = env.snapshot(), env.privileged_state(json_ready=True)
    other = tmp_path / "other.json"
    other.write_text(json.dumps({**OPS_SCENE, "name": "other", "objects": OPS_SCENE["objects"][:3],
                                 "success": {"op": "on", "a": "a", "b": "b"}}))
    env.reset(task(other))
    assert set(env.privileged_state()["objects"]) == {"a", "b", "c"}
    env.restore(blob)
    after = env.privileged_state(json_ready=True)
    np.testing.assert_allclose(after.pop("tcp"), before.pop("tcp"), atol=1e-12)
    assert after == before and env.t == 3
    env.data.qpos[env.items["slider"]["jqadr"]] = 0.0                           # closing again keeps the latch
    put(env, "a", [0.05, 0.15, 0.03])
    r = hold(env, 25)
    assert r.success and r.terminated and r.info == {"done": 2, "of": 2}


def test_predicate_operators(env, scene_file):
    env.reset(task(scene_file))
    hold(env, 2)
    assert holds(env, {"op": "on", "a": "a", "b": "table"})
    assert not holds(env, {"op": "on", "a": "a", "b": "b"})
    put(env, "a", [0.0, -0.2, 0.062])                                           # a on b
    put(env, "c", [0.05, 0.15, 0.03], yaw_deg=20)                               # c in the tray
    hold(env, 25)
    assert holds(env, {"op": "on", "a": "a", "b": "b"}) and not holds(env, {"op": "on", "a": "b", "b": "a"})
    assert holds(env, {"op": "above", "a": "a", "b": "b"}) and not holds(env, {"op": "above", "a": "b", "b": "a"})
    assert holds(env, {"op": "lifted", "a": "a", "height": 0.03}) and not holds(env, {"op": "lifted", "a": "b"})
    assert holds(env, {"op": "inside", "a": "c", "b": "tray"}) and not holds(env, {"op": "inside", "a": "b", "b": "tray"})
    assert not holds(env, {"op": "on", "a": "c", "b": "tray"})                  # in it, not on top of it
    assert holds(env, {"op": "inside", "a": "c", "region": {"x": [0.0, 0.1], "y": [0.1, 0.2]}})
    assert holds(env, {"op": "inside", "a": "a", "b": "b"}) is False            # not inside b's bounding box
    assert holds(env, {"op": "near", "a": "c", "b": "tray", "max_dist": 0.02})
    assert not holds(env, {"op": "near", "a": "c", "point": [0.3, 0.3], "max_dist": 0.05})
    assert holds(env, {"op": "upright", "a": "c"})
    put(env, "b", [0.15, -0.25, 0.021], roll_deg=90)                            # a block on its side ...
    put(env, "a", [0.15, -0.15, 0.021], roll_deg=45)                            # ... and one tipping over
    hold(env, 30)
    assert holds(env, {"op": "upright", "a": "b", "axis": "y"}) and not holds(env, {"op": "upright", "a": "b", "axis": "x"})
    put(env, "a", [0.0, -0.25, 0.021]), put(env, "b", [0.01, -0.15, 0.021]), put(env, "c", [-0.01, -0.05, 0.021])
    hold(env, 15)
    assert holds(env, {"op": "row", "objects": ["a", "b", "c"], "axis": "y"})
    assert not holds(env, {"op": "row", "objects": ["c", "b", "a"], "axis": "y"})
    assert not holds(env, {"op": "row", "objects": ["a", "b", "c"], "axis": "y", "max_offset": 0.005})
    assert not holds(env, {"op": "row", "objects": ["a", "b", "c"], "axis": "x"})
    env.data.qpos[env.items["slider"]["jqadr"]] = 0.08
    hold(env, 2)
    assert holds(env, {"op": "joint_at_least", "object": "slider", "fraction": 0.7})
    assert not holds(env, {"op": "joint_at_most", "object": "slider", "value": 0.05})
    assert holds(env, {"op": "not", "term": {"op": "joint_at_most", "object": "slider", "value": 0.05}})
    assert holds(env, {"op": "any_of", "terms": [{"op": "lifted", "a": "a"}, {"op": "joint_at_least", "object": "slider", "value": 0.07}]})
    assert not holds(env, {"op": "all_of", "terms": [{"op": "lifted", "a": "a"}, {"op": "joint_at_least", "object": "slider", "value": 0.07}]})
    env.ep["latch"] = {}
    seq = {"op": "sequence", "terms": [{"op": "joint_at_least", "object": "slider", "value": 0.05},
                                       {"op": "joint_at_most", "object": "slider", "value": 0.01}]}
    assert not holds(env, seq) and env.ep["latch"] == {"t": 1}
    env.data.qpos[env.items["slider"]["jqadr"]] = 0.0
    hold(env, 2)
    assert holds(env, seq)


def test_released_objects_only_and_falling_off_fails(env, scene_file):
    env.reset(task(scene_file))
    put(env, "a", [0.05, 0.15, 0.04])                                           # falling into the tray: moving
    env.data.qvel[env.items["a"]["vadr"] + 2] = -1.0
    env._mj.mj_forward(env.model, env.data)
    assert not holds(env, {"op": "inside", "a": "a", "b": "tray"})
    assert holds(env, {"op": "inside", "a": "a", "b": "tray", "released": False})
    put(env, "b", [0.0, -0.66, 0.03])                                           # over the edge
    r = hold(env, 30)
    assert r.terminated and r.success is False and "fell off" in r.info["fail_reason"]


def test_benchmark_turns_a_directory_into_tasks_with_disjoint_splits(tmp_path):
    for name in ("one", "two"):
        (tmp_path / f"{name}.json").write_text(json.dumps({**OPS_SCENE, "name": name, "max_steps": 123}))
    b = scene_benchmark(str(tmp_path))
    assert b.id == "mujoco/scenes-v0" and [t.id for t in b.tasks] == ["mujoco/scene_one", "mujoco/scene_two"]
    assert b.tasks[0].max_steps == 123 and b.tasks[0].params["scene"].endswith("one.json")
    seeds = [s for v in b.splits.values() for s in v]
    assert len(seeds) == len(set(seeds)) and {"search", "gate", "heldout", "validation"} <= set(b.splits)
    assert set(scene_benchmark(str(tmp_path), protocol="standard").splits) == {"train", "validation", "heldout"}
    (tmp_path / "three.json").write_text(json.dumps({**OPS_SCENE, "name": "one"}))
    with pytest.raises(ValueError, match="unique"):
        scene_tasks(tmp_path)
    assert load_scene(tmp_path / "one.json").name == "one"


def test_conformance_without_cameras(assets, scene_file):
    from hylo.testing.conformance import check_environment

    rep = check_environment(lambda: PandaSceneEnv(scene=str(scene_file), cameras=[], privileged_obs=True),
                            scene_tasks(scene_file)[0])
    assert rep.ok, rep.summary()
