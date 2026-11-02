"""mujoco.dex_table: the hand library against the Menagerie models, the floating-hand robot, the lift tasks, and a
physics lift of a box planned with the toolkit's dex_grasps and executed through plan / run (no cameras: the planner
gets the box's mesh at its true pose, as a perception step would give its points)."""

from typing import Any

import numpy as np
import pytest

pytest.importorskip("mujoco")

from hylo.core.task import TaskInstance  # noqa: E402
from hylo.core.types import Action  # noqa: E402
from hylo.envs.mujoco import dex_table as dt  # noqa: E402
from hylo.envs.mujoco import hands  # noqa: E402
from hylo.envs.mujoco.dex_tasks import TASK_KINDS, dex_tasks, dexterous_benchmark, sample_episode  # noqa: E402


@pytest.fixture(scope="module")
def fetched():
    from hylo.envs.mujoco.assets import ALLEGRO, G1_HANDS, assets

    try:
        return assets(G1_HANDS), assets(ALLEGRO)
    except RuntimeError as exc:
        pytest.skip(str(exc))


def test_library_names_and_sides():
    assert hands.hand_names() == ["allegro_left", "allegro_right", "g1_left", "g1_right"]
    assert hands.resolve("g1", "left") == "g1_left" and hands.resolve("allegro") == "allegro_right"
    with pytest.raises(KeyError):
        hands.resolve("shadow")
    for name in hands.hand_names():
        e = hands.hand_entry(name)
        lo, hi = np.array(e.ranges).T
        for cmd in [e.spec.open, e.spec.closed, *e.spec.presets.values()]:
            assert len(cmd) == len(e.spec.joints) and np.all(np.asarray(cmd) >= lo - 1e-6) and np.all(np.asarray(cmd) <= hi + 1e-6)


def test_shapes_are_closed_convex_parts():
    for kind in dt.SHAPES:
        s = dt.shape(kind)
        assert s.parts and s.mass > 0
        V = np.vstack([v for v, _ in s.parts])
        assert V[:, 2].min() == pytest.approx(0.0, abs=1e-9)          # resting on the table at the origin
        for v, f in s.parts:
            a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
            vol = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6     # outward winding: positive volume
            assert vol > 0
    assert len(dt.shape("mug").parts) > 10                              # the cup is concave: many parts


def test_tasks_sample_one_object():
    assert len(dex_tasks()) == len(TASK_KINDS) == 5
    ep = sample_episode({"task": "lift_mug", "object": "mug"}, np.random.default_rng(3))
    assert list(ep["items"]) == ["mug0"] and ep["goal"]["object"] == "mug0" and "mug" in ep["instruction"]
    assert dexterous_benchmark(episodes=3).splits["validation"] == (0, 1, 2)


def test_library_matches_the_models(fetched):
    import mujoco

    for name in hands.hand_names():
        e = hands.hand_entry(name)
        path = hands.floating_hand_file(name)
        m = mujoco.MjModel.from_xml_path(str(path))
        rng = np.array([m.jnt_range[m.joint(j).id] for j in e.spec.joints])
        assert np.allclose(rng, np.array(e.ranges), atol=1e-4), name
        assert m.site("wrist").bodyid == m.body(e.ee).id
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)                                       # base joints at zero: TCP straight down
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = d.xmat[m.body(e.ee).id].reshape(3, 3), d.xpos[m.body(e.ee).id]
        tcp = T @ np.asarray(e.spec.tcp_in_ee)
        assert np.allclose(tcp[:3, 0], [0, 0, -1], atol=1e-6), name
        assert {m.actuator(i).name for i in range(m.nu)} == {f"{j}_servo" for j in (*e.spec.joints,
                                                                                    *hands.base_joint_names())}


class EnvExecutor:
    """The toolkit's Executor over an environment, stepping it directly (tests and scripts)."""

    def __init__(self, env: Any, obs: Any, max_steps: int) -> None:
        self.env, self.obs, self.max_steps, self.steps, self.ended, self.success = env, obs, max_steps, 0, False, None

    def observation(self):
        return self.obs

    def run(self, actions: np.ndarray) -> dict[str, Any]:
        n = 0
        for a in np.asarray(actions, float):
            if self.ended or self.steps >= self.max_steps:
                break
            r = self.env.step(Action(data=a.tolist()))
            self.obs, n, self.steps = r.observation, n + 1, self.steps + 1
            self.success = r.success
            self.ended = bool(r.terminated or r.truncated)
        return {"steps": n, "ended": self.ended, "success": self.success, "reason": None}

    def status(self) -> dict[str, Any]:
        return {"step": self.steps, "max_steps": self.max_steps, "instruction": self.obs.instruction, "ended": self.ended}


def test_dex_grasps_lift_a_box_in_physics(fetched):
    from hylo.manip import Goal, Tools
    from hylo.manip.datatypes import Table
    from hylo.manip.geometry import apply

    env = dt.DexTableEnv(hand="g1_right", cameras=[], privileged_obs=True)
    task = next(t for t in dex_tasks(["lift_box"]))
    obs = env.reset(TaskInstance(spec=task, seed=1))
    robot = Tools(env.spec().robot, EnvExecutor(env, obs, task.max_steps))
    o = obs.extra["privileged"]["objects"]["box0"]
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.asarray(o["R"]), np.asarray(o["pos"])
    V, F = dt.shape("box").mesh()
    table = Table(height=0.0, normal=np.array([0.0, 0.0, 1.0]))
    goals = robot.dex_grasps(None, table, mesh=(apply(T, V), F), top=4)
    assert goals
    plan = robot.plan(goals, table)
    assert plan is not None, robot.why()
    robot.run(plan)
    up = robot.status()["tcp"]["hand"].copy()
    up[2, 3] += 0.12
    out = robot.run(robot.plan(Goal("lift", "hand", tcp=up), table))
    assert out["success"], env.privileged_state()["objects"]["box0"]["pos"]
    env.close()
