"""The free scene's library modules on their own, without a robot: the program loader (free_program), the world
index, sim views and demo hands on a bare MJCF model (free_world), and the model helpers (free_model)."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from hylo.envs.mujoco.free_model import (DEMO_HAND, SceneHandle, add_demo_hands, flex_name,  # noqa: E402
                                         robot_facts)
from hylo.envs.mujoco.free_program import (DEFAULT_MAX_STEPS, SceneProgram, SceneProgramError,  # noqa: E402
                                           program_problems, short_error)
from hylo.envs.mujoco.free_world import DemoHands, SceneWorld, SimEdit, SimView  # noqa: E402

WORLD = """
<mujoco>
  <extension><plugin plugin="mujoco.elasticity.cable"/></extension>
  <worldbody>
    <geom name="table" type="box" size="0.6 0.6 0.05" pos="0 0 -0.05"/>
    <body name="cube" pos="0.1 0 0.02"><freejoint/><geom type="box" size="0.02 0.02 0.02"/></body>
    <body name="chest" pos="-0.2 0.2 0.03"><geom type="box" size="0.05 0.05 0.03"/>
      <body name="lid" pos="0 0 0.035"><joint name="hinge" type="hinge" axis="0 1 0" range="0 90"/>
        <geom type="box" size="0.05 0.05 0.005"/><site name="knob" pos="0.04 0 0.01"/></body></body>
    <composite prefix="rope" type="cable" curve="s" count="6 1 1" size="0.25" offset="-0.1 -0.2 0.012" initial="free">
      <plugin plugin="mujoco.elasticity.cable"><config key="bend" value="1e5"/></plugin>
      <joint kind="main" damping="0.02"/><geom type="capsule" size="0.01" density="300"/>
    </composite>
    <flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.03 0.03 0.03" pos="0.2 0.2 0.005" radius="0.004"
              dim="2" mass="0.01"><edge equality="true"/><contact selfcollide="none"/></flexcomp>
    <body name="hand" pos="0.1 0 0.3"><geom name="finger" type="box" size="0.01 0.01 0.01"/></body>
  </worldbody>
  <contact><exclude body1="chest" body2="lid"/></contact>
</mujoco>"""


@pytest.fixture
def world():
    spec = mujoco.MjSpec.from_string(WORLD)
    base = {"world"}
    add_demo_hands(spec, 2)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    names = {m.body(b).name for b in range(m.nbody)} - base - {"hand", f"{DEMO_HAND}0", f"{DEMO_HAND}1"}
    hand = m.body("hand").id
    w = SceneWorld(m, names, robot_bodies={hand}, finger_bodies={hand}, labels={"rope": "red rope"})
    return w, m, d


def test_the_world_index_finds_cables_flexes_joints_and_groups(world):
    w, m, d = world
    assert list(w.cables) == ["rope"] and len(w.cables["rope"]) == 5
    assert [m.body(b).name for b in w.cables["rope"]][[0, -1][0]] == "ropeB_first"
    assert m.body(w.cables["rope"][-1]).name == "ropeB_last"
    assert list(w.flexes) == ["cloth"] and w.flexes["cloth"]["dim"] == 2 and len(w.flexes["cloth"]["bodies"]) == 9
    assert list(w.joints) == ["hinge"] and w.joints["hinge"]["type"] == "hinge"
    assert {m.body(b).name for b in w.rigid} == {"cube", "chest", "lid"}
    assert set(w.groups) == {"rope", "cloth", "cube", "chest"} and len(w.groups["chest"]) == 2
    assert w.label("rope") == "red rope" and w.label("cube") == "cube"
    assert w.flex_topology("cloth").shape == (8, 3) and flex_name(m, 0) == "cloth"
    gi, ids = w.instances(np.zeros(m.ngeom, np.int32), base=3)
    assert sorted(ids.values()) == [3, 4, 5, 6] and gi[m.geom("finger").id] == 0
    assert gi[m.geom_bodyid.tolist().index(m.body("lid").id)] == ids["chest"]   # a child joins its root's group


def test_state_is_json_ready_and_complete(world):
    w, m, d = world
    s = w.state(d)
    assert set(s) == {"bodies", "joints", "cables", "flexes", "sites"}
    cube = s["bodies"]["cube"]
    assert cube["free"] and not cube["mocap"] and abs(cube["top"] - 0.04) < 1e-6
    assert s["bodies"]["lid"]["parent"] == "chest" and not s["bodies"]["chest"]["free"]
    assert s["joints"]["hinge"]["range"] == pytest.approx([0, np.pi / 2]) and s["joints"]["hinge"]["units"] == "rad"
    assert len(s["cables"]["rope"]["nodes"]) == 5 and s["cables"]["rope"]["radius"] == pytest.approx(0.01)
    assert not s["cables"]["rope"]["anchored"] and len(s["flexes"]["cloth"]["vertices"]) == 9
    assert s["sites"]["knob"] == pytest.approx([-0.16, 0.2, 0.075])


def test_views_read_and_edit_and_touching_follows_contacts(world):
    w, m, d = world
    view = SimView(w, d, table_top=0.0)
    assert view.pos("cube") == pytest.approx([0.1, 0, 0.02]) and view.nodes("rope").shape == (5, 3)
    assert view.flex("cloth").shape == (9, 3) and view.joint("hinge") == 0.0 and view.table_top == 0.0
    lo, hi = view.bounds("chest")
    assert hi[2] == pytest.approx(0.07) and view.bodies("cloth_")[:1] == ["cloth_0"]
    for _ in range(50):
        mujoco.mj_step(m, d)
    assert view.touching("cube", "table") and view.touching("rope", "table") and view.touching("cloth", "table")
    assert not view.held("cube") and not view.touching("cube", "robot")
    with pytest.raises(KeyError, match="no body 'ghost'"):
        view.pos("ghost")
    edit = SimEdit(w, d, 0.0, forward=lambda: mujoco.mj_forward(m, d))
    edit.set_body("cube", pos=[0.1, 0.0, 0.3])
    edit.set_joint("hinge", 0.5)
    assert view.pos("cube")[2] == pytest.approx(0.3) and view.joint("hinge") == pytest.approx(0.5)
    with pytest.raises(SceneProgramError, match="no free joint"):
        edit.set_body("chest", pos=[0, 0, 1])
    with pytest.raises(SceneProgramError, match="only in demo"):
        edit.grab("cube")


def test_demo_hands_carry_a_body_and_record_success(world):
    w, m, d = world
    hands = DemoHands(w, 2)
    hands.reset(d)

    def step() -> bool:
        mujoco.mj_step(m, d, nstep=25)
        mujoco.mj_forward(m, d)
        return bool(d.xpos[m.body("ropeB_last").id][2] > 0.1)

    edit = SimEdit(w, d, 0.0, lambda: mujoco.mj_forward(m, d), hands, step, control_hz=20)
    end = edit.pos("ropeB_last")
    edit.grab("ropeB_last", hand=1)
    edit.move(1, end + [0, 0, 0.15], seconds=1.0)
    assert edit.pos("ropeB_last")[2] > 0.12 and hands.success_at is not None and hands.steps == 20
    assert hands.log == [{"grab": "ropeB_last", "hand": 1, "point": pytest.approx(end.tolist())}]
    edit.release(1)
    edit.wait(1.0)
    assert edit.pos("ropeB_last")[2] < 0.05
    with pytest.raises(SceneProgramError, match="hand 5 does not exist"):
        edit.grab("cube", hand=5)


def test_the_program_loader(tmp_path):
    p = tmp_path / "scene.py"
    p.write_text("import math\nROBOT = 'G1'\nINSTRUCTION = ' x '\ndef build(scene):\n    return 1 / 0\n"
                 "def success(sim): return math.pi > 3\nprint('top level runs')\n")
    prog = SceneProgram(tmp_path)
    assert prog.robot == "g1" and prog.instruction == "x" and prog.max_steps == DEFAULT_MAX_STEPS
    assert prog.success(None) is True and prog.demo is None and prog.output == ["top level runs\n"]
    with pytest.raises(SceneProgramError, match=r"build\(\) raised line 5 in build: return 1 / 0 -> ZeroDivisionError"):
        prog.call(prog.build, None)
    p.write_text("ROBOT = 'panda'\nINSTRUCTION = 'x'\ndef build(scene): pass\ndef success(sim): pass\nx = 1 / 0\n")
    with pytest.raises(SceneProgramError, match="ZeroDivisionError"):
        SceneProgram(p)
    assert program_problems("def f(_x): pass")[0].endswith("starts with an underscore")
    try:
        raise ValueError("boom")
    except ValueError as exc:
        assert short_error(exc) == "ValueError: boom"


def test_robot_facts_and_the_scene_handle():
    for robot, arms in (("panda", ["arm"]), ("aloha", ["left", "right"]), ("g1", ["left", "right"])):
        f = robot_facts(robot)
        assert list(f["arms"]) == arms and f["opening"] > 0.05
        assert f["table_x"][0] < f["arms"][arms[0]]["x"][0] < f["arms"][arms[0]]["x"][1] < f["table_x"][1]
    spec = mujoco.MjSpec.from_string('<mujoco><worldbody><geom name="table" type="box" size="1 1 .1"/></worldbody>'
                                     '</mujoco>')
    h = SceneHandle(spec, "g1")
    assert h.table_top == 0.86 and set(h.arms) == {"left", "right"} and h.rgba("red")[3] == 1.0
    h.add_mjcf('<mujoco><worldbody><body name="b"><geom size=".02"/></body></worldbody></mujoco>', pos=[0, 0, 1])
    assert spec.compile().body("b").id > 0
    with pytest.raises(SceneProgramError, match="does not parse"):
        h.add_mjcf("<mujoco><worldbody><body></mujoco>")
