"""The stage world (RFC-0008): composition, actions with receipts, controllers per robot family, checks, people and
the world tool surface. Robots come from MuJoCo Menagerie (fetched at the pinned commit; skipped offline)."""

import json
import math

import numpy as np
import pytest

pytest.importorskip("mujoco")

from hylo.core.stage import Body, Camera, Entity, Mount, Pose, Robot, Shape, Stage  # noqa: E402
from hylo.envs.mujoco.checks import check, gate  # noqa: E402
from hylo.envs.mujoco.compose import look_at  # noqa: E402
from hylo.envs.mujoco.crowd import astar, shorten  # noqa: E402
from hylo.envs.mujoco.packs import PACKS, pack  # noqa: E402
from hylo.envs.mujoco.world import World  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def menagerie():
    try:
        for name in ("panda", "go1", "go2", "spot", "a1", "x2", "crazyflie", "tidybot", "ur5e", "robotiq_2f85"):
            pack(name).model_path()
    except RuntimeError as exc:
        pytest.skip(str(exc))


def table_stage(*extra):
    return Stage(name="lab", entities=[
        Entity(id="table", shape=Shape(type="box", size=(0.4, 0.4, 0.2)), body=Body(dynamic=False),
               pose=Pose(pos=(0.5, 0, 0.2))),
        Entity(id="arm", robot=Robot(pack="panda"), pose=Pose(pos=(0, 0, 0.4))),
        Entity(id="cube", shape=Shape(type="box", size=(0.02,) * 3, rgba=(0.9, 0.1, 0.1, 1)), body=Body(mass=0.05),
               pose=Pose(pos=(0.5, 0, 0.42))), *extra])


def act(w, robot, kind, **kw):
    return w.act({"action_id": f"{robot}-{kind}-{w.step}", "expected_generation": w.generation, "robot": robot,
                  "kind": kind, **kw})


def test_pick_up_a_cube_with_receipts_and_checks():
    w = World(table_stage())
    assert w.overlaps() == [] and set(w.robots) == {"arm"}
    down = [0, 1, 0, 0]
    goal = {"op": "all", "terms": [{"op": "lifted", "entity": "cube", "min_z": 0.5},
                                   {"op": "near", "a": "cube", "b": "arm", "max_m": 0.05}]}
    g = gate(w, goal)
    assert g["ok"] and {f["reason"] for f in g["failures"]} == {"NOT_LIFTED", "DISTANCE_NOT_MET"}
    r = act(w, "arm", "tcp", position=[0.5, 0, 0.53], quat_wxyz=down, duration_s=1.5)
    assert r["status"] == "completed" and r["measured"]["max_joint_error"] < 0.02
    act(w, "arm", "tcp", position=[0.5, 0, 0.43], quat_wxyz=down)
    r = act(w, "arm", "gripper", command="close")
    assert r["measured"]["holding"] and 0.3 < r["measured"]["opening"] < 0.7      # a 4 cm cube in an 8 cm hand
    act(w, "arm", "tcp", position=[0.5, 0, 0.58], quat_wxyz=down, duration_s=1.5)
    res = check(w, goal)
    assert res["satisfied"], res
    tcp = w.state()["robots"]["arm"]["tcp"]["position"]
    assert abs(tcp[2] - 0.58) < 0.01
    assert gate(w, goal)["reason"] == "SPEC_ALREADY_SATISFIED"


def test_generation_idempotency_and_refusals():
    w = World(table_stage())
    g = w.generation
    a = {"action_id": "x1", "expected_generation": g, "robot": "arm", "kind": "hold", "duration_s": 0.1}
    r1 = w.act(a)
    r2 = w.act(a)
    assert r2["repeated"] and r2["end_step"] == r1["end_step"] and w.step == r1["end_step"]
    assert w.act({**a, "action_id": "x2", "expected_generation": g - 1})["reason"].startswith("stale")
    assert w.act({**a, "action_id": "x3", "kind": "walk"})["reason"].startswith("unsupported")
    assert w.act({**a, "action_id": "x4", "robot": "ghost"})["reason"].startswith("unknown_robot")
    r = w.act({**a, "action_id": "x5", "kind": "tcp", "position": [3.0, 0, 0.4]})
    assert r["status"] == "refused" and r["reason"].startswith("unreachable") and r["steps"] == 0
    assert w.reset() == g + 1 and w.step == 0
    assert w.act({**a, "action_id": "x6"})["reason"].startswith("stale")


@pytest.mark.parametrize("name", ["go1", "go2", "a1", "spot"])
def test_quadrupeds_trot_forward_and_stay_upright(name):
    w = World(Stage(name="dog", entities=[Entity(id="dog", robot=Robot(pack=name, base="free"))]))
    m = act(w, "dog", "walk", forward=0.8, duration_s=3)["measured"]
    assert m["upright"] and m["forward_m"] > 0.4 and abs(m["turned_deg"]) < 15, m
    m = act(w, "dog", "walk", forward=-0.6, duration_s=2)["measured"]
    assert m["upright"] and m["forward_m"] < -0.1, m


@pytest.mark.parametrize("name", ["x2", "crazyflie"])
def test_drones_fly_to_points_and_hover(name):
    w = World(Stage(name="air", entities=[Entity(id="d", robot=Robot(pack=name, base="free"),
                                                 pose=Pose(pos=(0, 0, 0.1)))]))
    r = act(w, "d", "fly", position=[0.4, 0.3, 1.0], duration_s=8)
    assert r["measured"]["error_m"] < 0.05, r
    r = act(w, "d", "hold", duration_s=2)
    assert r["measured"]["error_m"] < 0.05 and w.state()["robots"]["d"]["base"]["upright"] > 0.95


def test_mobile_base_and_a_gripper_mounted_on_an_arm():
    w = World(Stage(name="mix", entities=[
        Entity(id="tb", robot=Robot(pack="tidybot")),
        Entity(id="ur", robot=Robot(pack="ur5e"), pose=Pose(pos=(0, 2, 0))),
        Entity(id="hand", robot=Robot(pack="robotiq_2f85", mount=Mount(entity="ur", body="wrist_3_link")))]))
    assert w.overlaps() == []
    r = act(w, "tb", "base", x=0.5, y=0.2, yaw_deg=30, duration_s=3)
    j = w.state()["robots"]["tb"]["joints"]
    assert abs(j["joint_x"] - 0.5) < 0.02 and abs(math.degrees(j["joint_th"]) - 30) < 2, r
    hand_body = w.model.body("hand/base_mount").id if "hand/base_mount" in [w.model.body(i).name for i in
                                                                            range(w.model.nbody)] else None
    if hand_body is not None:                                   # the hand rides on the arm's wrist
        assert w.entity_of_body(hand_body) == "hand"
    before = w.data.xpos[w.robots["hand"].root_body].copy()
    act(w, "ur", "joints", targets={"shoulder_pan": 0.8}, duration_s=1.5)
    assert np.linalg.norm(w.data.xpos[w.robots["hand"].root_body] - before) > 0.1
    r = act(w, "hand", "gripper", command="close")
    assert r["measured"]["opening"] < 0.1 and not r["measured"]["holding"]


def test_batch_runs_robots_in_the_same_steps():
    w = World(table_stage(Entity(id="dog", robot=Robot(pack="go1", base="free"), pose=Pose(pos=(0, -1.5, 0)))))
    g = w.generation
    rs = w.act_batch([{"action_id": "p", "expected_generation": g, "robot": "arm", "kind": "hold", "duration_s": 0.5},
                      {"action_id": "q", "expected_generation": g, "robot": "dog", "kind": "walk", "duration_s": 1.0},
                      {"action_id": "r", "expected_generation": g, "robot": "dog", "kind": "hold"}])
    assert [r["status"] for r in rs] == ["completed", "completed", "refused"] and rs[2]["reason"].startswith("busy")
    assert rs[0]["start_step"] == rs[1]["start_step"] == 0 and rs[0]["end_step"] < rs[1]["end_step"]


def test_people_walk_routes_and_stop_for_obstacles():
    w = World(Stage(name="hall", entities=[
        Entity(id="crate", shape=Shape(type="box", size=(0.3, 0.3, 0.4)), body=Body(dynamic=False),
               pose=Pose(pos=(1.0, 0.0, 0.4))),
        Entity(id="ann", human={"route": [[2.5, 0.0]], "loop": False, "speed": 1.2}, pose=Pose(pos=(-0.5, 0, 0)))]))
    w.advance(5.0)
    st = w.humans.state()["ann"]
    assert st["status"] == "arrived" and abs(st["position"][0] - 2.5) < 0.1, st
    assert st["walked_m"] > 3.0                                    # walked around the crate, not through it
    w.humans.goto("ann", [-0.5, 0.0])
    w.advance(0.5)
    w.humans.stop("ann")
    p = w.humans.state()["ann"]["position"]
    w.advance(1.0)
    assert w.humans.state()["ann"]["position"] == p and w.humans.state()["ann"]["status"] == "stopped"


def test_grid_planner():
    free = np.ones((10, 10), bool)
    free[5, 0:8] = False
    path = astar(free, (0, 0), (9, 0))
    assert path[0] == (0, 0) and path[-1] == (9, 0) and all(free[c] for c in path)
    assert not any(c[0] == 5 and c[1] < 8 for c in path)
    pts = [np.array([i + 0.5, j + 0.5]) for i, j in path]
    short = shorten(pts, free, np.zeros(2), 1.0)
    assert len(short) < len(pts) and np.allclose(short[-1], pts[-1])


def test_every_pack_is_described_and_points_at_a_manifest():
    for name, p in PACKS.items():
        assert p.license and p.source.startswith("https://github.com/") and "/tree/" in p.source
        assert p.status in ("usable", "partial", "unverified")
        if p.family == "quadruped":
            assert len(p.legs) == 4 and "walk" in p.controls
        if "tcp" in p.controls:
            assert p.tcp is not None


def renders():
    try:
        import mujoco
        r = mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco/>"), 8, 8)
        r.close()
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not renders(), reason="no OpenGL (set MUJOCO_GL=egl or osmesa)")
def test_world_surface_end_to_end(tmp_path):
    from hylo.envs.mujoco.world_tools import Workspace

    ws = Workspace(tmp_path / "stages", "lab")
    s = ws.surface
    eye = (1.8, -0.6, 1.4)
    ops = [{"op": "add", "entity": e.model_dump(mode="json", exclude_none=True)} for e in table_stage().entities]
    ops.append({"op": "add", "entity": {"id": "cam", "camera": {"width": 160, "height": 120},
                                        "pose": {"pos": list(eye), "quat": list(look_at(eye, (0.4, 0, 0.4)))}}})
    assert s.call("stage_edit", {"expected_revision": 0, "ops": ops}).result["revision"] == 1
    assert s.call("stage_edit", {"expected_revision": 0, "ops": ops}).error["code"] == "conflict"
    assert s.call("act", {"actions": [{"robot": "arm", "kind": "hold"}]}).error["code"] == "no_world"
    b = s.call("world_build", {}).result
    assert b["revision"] == 1 and b["cameras"] == ["cam"] and b["overlaps"] == []
    r = s.call("act", {"actions": [{"robot": "arm", "kind": "tcp", "expected_generation": b["generation"],
                                    "position": [0.5, 0, 0.55], "quat_wxyz": [0, 1, 0, 0]}],
                       "record": {"camera": "cam", "fps": 10}})
    assert r.ok and r.result["receipts"][0]["status"] == "completed" and r.result["video"]["frames"] > 5
    cap = s.call("capture", {})
    assert cap.ok and cap.images[0].shape == (120, 160, 3)
    assert {"arm", "cube", "table"} <= set(cap.result["cameras"]["cam"]["entities_seen"])
    p = s.call("pixel_to_world", {"capture": cap.result["capture"], "camera": "cam", "pixels": [[80, 60]]})
    assert p.ok and len(p.result["points"][0]["world"]) == 3
    ex = s.call("export", {"frames": 2, "every_s": 0.1, "name": "ds"}).result
    rows = [json.loads(x) for x in (tmp_path / "stages" / "lab" / "outputs" / "ds" / "samples.jsonl").read_text().splitlines()]
    assert ex["samples"] == len(rows) == 2 and rows[1]["step"] > rows[0]["step"] and "state" in rows[0]
    assert (tmp_path / "stages" / "lab" / "outputs" / "ds" / rows[0]["depth"]).is_file()
    assert s.call("check", {"goal": {"op": "lifted", "entity": "cube", "min_z": 1.0}}).result["reasons"] == ["NOT_LIFTED"]
    assert s.call("stage_show", {}).result["world_built_from_revision"] == 1
    scan = s.call("lidar", {"camera": "cam", "rows": 8, "cols": 16}).result
    assert scan["hits"] > 0 and {"table", "arm"} & set(scan["hits_by_entity"])
    # calibrate the camera back from points and the pixels the capture shows them at
    c = ws.world.capture(["cam"])["cam"]
    pts = np.array([[0.3, -0.2, 0.4], [0.7, 0.2, 0.4], [0.5, 0.0, 0.6], [0.2, 0.3, 0.1], [0.8, -0.3, 0.2],
                    [0.4, 0.1, 0.8], [0.6, -0.1, 0.3]])
    Tc = np.linalg.inv(c["T_world_cam"])
    pc = pts @ Tc[:3, :3].T + Tc[:3, 3]
    uv = (c["K"] @ pc.T).T[:, :2] / pc[:, 2:3]
    fit = s.call("camera_fit", {"points": pts.tolist(), "pixels": uv.tolist(), "focal_px": float(c["K"][0, 0]),
                                "cx": float(c["K"][0, 2]), "cy": float(c["K"][1, 2])}).result
    assert fit["reprojection_rms_px"] < 0.01 and np.allclose(fit["position"], eye, atol=1e-3)
    assert len(s.specs()) == 21 and all(t["parameters"]["type"] == "object" for t in s.specs())


@pytest.mark.parametrize("name,pol", [("g1_legs", "g1_walk"), ("h1_legs", "h1_walk")])
def test_pretrained_walking_policies_run_on_numpy(name, pol):
    from hylo.envs.mujoco.policies import network, policy, sniff

    try:
        net, info = network(pol)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    assert info["format"] == "torchscript" and info["recurrent"] and net.outputs == len(policy(pol).joints)
    assert sniff(__file__)["format"] == "unknown"
    w = World(Stage(name="h", entities=[Entity(id="bot", robot=Robot(pack=name, base="free"))]))
    m = act(w, "bot", "policy", policy=pol, command=[0.5, 0, 0], duration_s=4)["measured"]
    assert m["upright"] and m["forward_m"] > 1.2 and abs(m["turned_deg"]) < 20, m
    m = act(w, "bot", "policy", policy=pol, command=[0, 0, 0], duration_s=3)["measured"]
    assert m["upright"] and m["moved_m"] < 0.5, m


def test_policy_refused_on_the_wrong_robot():
    w = World(Stage(name="h", entities=[Entity(id="bot", robot=Robot(pack="h1_legs", base="free"))]))
    r = act(w, "bot", "policy", policy="g1_walk")
    assert r["status"] == "refused" and r["reason"].startswith("mismatch")


def test_restricted_reader_refuses_foreign_objects(tmp_path):
    import pickle
    import zipfile

    from hylo.envs.mujoco.policies import read_torchscript

    p = tmp_path / "evil.pt"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("m/data.pkl", pickle.dumps({"x": __import__("os").getcwd}))     # a global outside the whitelist
        z.writestr("m/code/__torch__/m.py", "")
    with pytest.raises(pickle.UnpicklingError, match="refused"):
        read_torchscript(p)
