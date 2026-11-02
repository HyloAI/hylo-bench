"""mujoco.free_scene: scene programs with each robot, deformables (cable, cloth), articulated parts, the success
function's view of the state, demo hands, the program guard, and snapshots."""

import textwrap

import numpy as np
import pytest

pytest.importorskip("mujoco")

from hylo.core.registry import registry  # noqa: E402
from hylo.core.task import TaskInstance, TaskSpec  # noqa: E402
from hylo.core.types import Action  # noqa: E402
from hylo.envs.mujoco.free_program import SceneProgram, SceneProgramError, program_problems  # noqa: E402
from hylo.envs.mujoco.free_scene import FreeSceneEnv  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def assets():
    from hylo.envs.mujoco.assets import ALOHA, G1_HANDS, PANDA, assets

    try:
        for name in (PANDA, ALOHA, G1_HANDS):
            assets(name)
    except RuntimeError as exc:                     # offline and not cached
        pytest.skip(str(exc))


CABLE = """
<mujoco>
  <extension><plugin plugin="mujoco.elasticity.cable"/></extension>
  <worldbody>
    <composite prefix="rope" type="cable" curve="s" count="16 1 1" size="0.3" offset="{x} {y} {z}" initial="free">
      <plugin plugin="mujoco.elasticity.cable"><config key="twist" value="1e5"/><config key="bend" value="1e5"/>
        <config key="vmax" value="0"/></plugin>
      <joint kind="main" damping="0.02"/>
      <geom type="capsule" size="0.012" rgba="0.8 0.2 0.1 1" density="300" condim="4"/>
    </composite>
  </worldbody>
</mujoco>"""

CLOTH = """
<mujoco>
  <worldbody>
    <body name="box" pos="{bx} {by} {bz}"><geom type="box" size="0.04 0.04 0.03" rgba="0.2 0.4 0.8 1"/></body>
    <flexcomp name="cloth" type="grid" count="6 6 1" spacing="0.03 0.03 0.03" pos="{x} {y} {z}" radius="0.004"
              dim="2" mass="0.03"><edge equality="true" damping="0.01"/><contact selfcollide="none" condim="3"/>
    </flexcomp>
  </worldbody>
</mujoco>"""

LID = """
<mujoco>
  <worldbody>
    <body name="chest" pos="{x} {y} {z}">
      <geom type="box" size="0.05 0.05 0.004" pos="0 0 -0.036"/>
      <geom type="box" size="0.004 0.05 0.04" pos="0.046 0 0"/><geom type="box" size="0.004 0.05 0.04" pos="-0.046 0 0"/>
      <geom type="box" size="0.05 0.004 0.04" pos="0 0.046 0"/><geom type="box" size="0.05 0.004 0.04" pos="0 -0.046 0"/>
      <body name="lid" pos="-0.05 0 0.044">
        <joint name="hinge" type="hinge" axis="0 -1 0" range="0 110" damping="0.05" armature="0.001"/>
        <geom type="box" size="0.05 0.052 0.004" pos="0.05 0 0" mass="0.05" rgba="0.7 0.5 0.2 1"/>
      </body>
    </body>
  </worldbody>
  <contact><exclude body1="chest" body2="lid"/></contact>
</mujoco>"""


def program(tmp_path, robot, build, success="return False", extra="", name="s"):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    body = textwrap.indent(textwrap.dedent(build).strip(), "    ")
    src = (f'ROBOT = "{robot}"\nINSTRUCTION = "do the thing"\nMAX_STEPS = 300\nLABELS = {{"rope": "red rope"}}\n\n'
           f"def build(scene):\n    t = scene.table_top\n{body}\n\n"
           f"def success(sim):\n    {success}\n" + textwrap.dedent(extra))
    (d / "scene.py").write_text(src)
    return d


def task(d, max_steps=300):
    return TaskInstance(spec=TaskSpec(id="free/t", version="1", instruction="x", params={}, max_steps=max_steps), seed=0)


def make(d, **kw):
    return registry.build("envs", {"type": "mujoco.free_scene",
                                   "params": {"scene": str(d), "cameras": [], "privileged_obs": True, **kw}})


def settle(env, n=20):
    hold = env.hold_action(None)
    r = None
    for _ in range(n):
        r = env.step(Action(data=hold))
    return r


@pytest.mark.parametrize("robot,where", [("panda", (-0.1, 0.0)), ("aloha", (-0.15, 0.05)), ("g1", (0.18, -0.15))])
def test_each_robot_loads_a_cable_with_its_own_robot_and_cameras(tmp_path, robot, where):
    d = program(tmp_path, robot, f'scene.add_mjcf("""{CABLE}""".format(x={where[0]}, y={where[1]}, z=t + 0.012))',
                success='return bool(sim.nodes("rope")[-1][2] > sim.table_top + 0.1)')
    env = make(d)
    try:
        spec = env.spec()
        assert spec.name == "mujoco.free_scene" and spec.robot is not None
        arms = {"panda": ["arm"], "aloha": ["left", "right"], "g1": ["left", "right"]}[robot]
        assert [a.name for a in spec.robot.arms] == arms
        assert "free scene program" in spec.robot.notes and "block" not in spec.robot.notes.lower()
        obs = env.reset(task(d))
        priv = obs.extra["privileged"]
        assert priv["robot"] == robot and list(priv["cables"]) == ["rope"]
        nodes = np.asarray(priv["cables"]["rope"]["nodes"])
        assert nodes.shape == (15, 3) and not priv["cables"]["rope"]["anchored"]
        assert np.all(np.diff(nodes[:, 0]) > 0)                     # chain order along +x
        assert abs(nodes[0, 2] - (priv["table"]["height"] + 0.012)) < 0.004
        r = settle(env)
        assert not r.truncated and r.success is False and r.info == {}
        assert set(obs.state) == {f"{k}.{a}" for a in arms for k in ("q", "hand")}
    finally:
        env.close()


def test_cloth_and_box_privileged_state_success_and_segments(tmp_path):
    d = program(tmp_path, "panda", f'scene.add_mjcf("""{CLOTH}""".format(bx=0.08, by=0.15, bz=t + 0.03, x=-0.05, '
                                    'y=-0.12, z=t + 0.005))',
                success=("v = sim.flex('cloth'); lo, hi = sim.bounds('box')\n"
                         "    return bool(((v[:, 0] > lo[0]) & (v[:, 0] < hi[0]) & (v[:, 1] > lo[1]) & (v[:, 1] < hi[1]) "
                         "& (v[:, 2] > hi[2])).sum() >= 4)"),
                extra="""
                def demo(sim):
                    v = sim.flex("cloth")
                    lo, hi = sim.bounds("box")
                    c = (lo + hi) / 2
                    sim.grab("cloth_0", hand=0); sim.grab("cloth_5", hand=1)
                    sim.move_together({0: v[0] + [0, 0, 0.12], 1: v[5] + [0, 0, 0.12]}, seconds=1.0)
                    sim.move_together({0: [c[0] - 0.08, c[1] + 0.03, hi[2] + 0.08], 1: [c[0] + 0.08, c[1] + 0.03, hi[2] + 0.08]}, seconds=1.5)
                    sim.move_together({0: [c[0] - 0.08, c[1] + 0.05, hi[2] - 0.02], 1: [c[0] + 0.08, c[1] + 0.05, hi[2] - 0.02]}, seconds=1.0)
                """)
    env = make(d, demo_hands=2, segmentation=True)
    try:
        obs = env.reset(task(d))
        p = obs.extra["privileged"]
        assert set(p["bodies"]) == {"box"} and not p["bodies"]["box"]["free"]
        f = p["flexes"]["cloth"]
        assert len(f["vertices"]) == 36 and f["bodies"][:2] == ["cloth_0", "cloth_1"] and f["dim"] == 2
        assert env.segments()[str(env.group_instance["cloth"])] == "cloth"
        view = env.view()
        assert view.touching("cloth", "table") and not view.touching("cloth", "box") and not view.held("cloth")
        assert view.bodies("cloth_")[:2] == ["cloth_0", "cloth_1"] and view.bodies()[0] == "box"
        assert settle(env).success is False
        res = env.run_demo()
        assert res["ran"] and res["error"] is None and res["success_seen"], res
        assert [g["grab"] for g in res["grabs"]] == ["cloth_0", "cloth_5"]
    finally:
        env.close()


def test_hinged_lid_joint_state_targets_and_snapshot(tmp_path):
    d = program(tmp_path, "aloha", f'scene.add_mjcf("""{LID}""".format(x=-0.15, y=0.05, z=t + 0.04))',
                success="return bool(sim.joint('hinge') > 1.2)",
                extra="""
                def targets(sim):
                    lo, hi = sim.bounds("lid")
                    return {"lid edge": [hi[0] - 0.01, (lo[1] + hi[1]) / 2, hi[2]]}

                def randomize(sim, rng):
                    sim.set_joint("hinge", 0.0)
                """)
    env = make(d)
    try:
        obs = env.reset(task(d))
        j = obs.extra["privileged"]["joints"]["hinge"]
        assert j["type"] == "hinge" and j["body"] == "lid" and j["units"] == "rad"
        assert abs(j["range"][1] - np.radians(110)) < 1e-6        # MJCF angles in degrees by default
        assert list(env.program_targets()) == ["lid edge"]
        settle(env, 5)
        snap = env.snapshot()
        env.editor().set_joint("hinge", 1.5)
        assert env.step(Action(data=env.hold_action(None))).success   # the program's success function judges
        env.restore(snap)
        assert abs(env.view().joint("hinge")) < 0.05
    finally:
        env.close()


def test_the_program_guard_refuses_files_imports_and_underscores(tmp_path):
    assert program_problems("import os\nROBOT='panda'") and program_problems("x = ().__class__")
    assert any("open" in p for p in program_problems("open('/etc/passwd')"))
    assert any("from_file" in p for p in program_problems("import mujoco\nmujoco.MjSpec.from_file('x')"))
    assert program_problems("import numpy as np\nnp.load('x')")
    assert not program_problems("import math, numpy as np, mujoco\nROBOT = 'panda'")
    d = tmp_path / "bad"
    d.mkdir()
    (d / "scene.py").write_text("import subprocess\nROBOT='panda'")
    with pytest.raises(SceneProgramError, match="import subprocess"):
        SceneProgram(d)
    (d / "scene.py").write_text("ROBOT = 'ur5'\nINSTRUCTION = 'x'\ndef build(scene): pass\ndef success(sim): return 1")
    with pytest.raises(SceneProgramError, match="ROBOT must be one of"):
        SceneProgram(d)
    (d / "scene.py").write_text("ROBOT = 'panda'\nINSTRUCTION = 'x'\ndef build(scene): pass")
    with pytest.raises(SceneProgramError, match="must define success"):
        SceneProgram(d)


def test_actuators_keyframes_compile_errors_and_build_errors_are_the_programs_problem(tmp_path):
    cases = [
        ('scene.add_mjcf("<mujoco><worldbody><body name=\'b\'><joint name=\'j\'/><geom size=\'.02\'/></body>'
         '</worldbody><actuator><motor joint=\'j\'/></actuator></mujoco>")', "must not add actuators"),
        ('scene.add_mjcf("<mujoco><worldbody><body name=\'b\'><geom size=\'.02\'/></body></worldbody>'
         '<keyframe><key qpos=\'\'/></keyframe></mujoco>")', "keyframes"),
        ('scene.add_mjcf("<mujoco><worldbody><body name=\'b\'><geom type=\'mesh\' mesh=\'nope\'/></body>'
         '</worldbody></mujoco>")', "does not"),
        ("x = [1, 2][5]", "IndexError"),
    ]
    for i, (build, words) in enumerate(cases):
        d = program(tmp_path, "panda", build, name=f"c{i}")
        env = FreeSceneEnv(scene=str(d), cameras=[])
        with pytest.raises(SceneProgramError, match=words):
            env.reset(task(d))
        env.close()


def test_success_errors_and_non_bools_are_reported_not_raised(tmp_path):
    d = program(tmp_path, "panda", "pass", success="return sim.pos('nothing')[2] > 0", name="e")
    env = make(d)
    try:
        env.reset(task(d))
        r = env.step(Action(data=env.hold_action(None)))
        assert r.success is False and "no body 'nothing'" in r.info["success_error"]
    finally:
        env.close()
    d = program(tmp_path, "panda", "pass", success="return 0", name="t")
    env = make(d)
    try:
        env.reset(task(d))
        r = env.step(Action(data=env.hold_action(None)))
        assert r.success is False and r.info["success_type"] == "int"
    finally:
        env.close()


def test_a_task_scene_must_keep_the_robot(tmp_path):
    a = program(tmp_path, "panda", "pass", name="a")
    b = program(tmp_path, "aloha", "pass", name="b")
    env = make(a)
    try:
        bad = TaskInstance(spec=TaskSpec(id="t", version="1", instruction="x", params={"scene": str(b)}), seed=0)
        with pytest.raises(ValueError, match="uses robot aloha"):
            env.reset(bad)
    finally:
        env.close()


@pytest.mark.parametrize("robot", ["panda", "aloha"])
def test_the_whole_world_exports_as_mjcf_that_loads_alone(tmp_path, robot):
    import mujoco

    d = program(tmp_path, robot, 'scene.add_mjcf("""{}""".format(x=-0.1, y=0.0, z=t + 0.012))'.replace("{}", CABLE))
    env = make(d)
    try:
        xml = env.model_xml()
        m = mujoco.MjModel.from_xml_string(xml)
        assert (m.nq, m.nbody, m.ngeom) == (env.model.nq, env.model.nbody, env.model.ngeom)
    finally:
        env.close()
