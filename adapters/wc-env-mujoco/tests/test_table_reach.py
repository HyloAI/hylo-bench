import numpy as np
import pytest

pytest.importorskip("mujoco")

from worldcurrent.agents.composable import make_object_reach_agent  # noqa: E402
from worldcurrent.core.kernel import EpisodeRunner  # noqa: E402
from worldcurrent.core.task import TaskInstance  # noqa: E402
from worldcurrent.core.types import Action  # noqa: E402
from worldcurrent.envs.mujoco.table_reach import INDIRECT, TableReachEnv, table_tasks  # noqa: E402

TASKS = {t.id.split("/")[1]: t for t in table_tasks()}


@pytest.fixture
def env():
    e = TableReachEnv()
    yield e
    e.close()


def test_episode_sampling_is_seeded(env):
    a = env.reset(TaskInstance(spec=TASKS["table_direct"], seed=5))
    colors, instr = list(env.colors), a.instruction
    b = env.reset(TaskInstance(spec=TASKS["table_direct"], seed=5))
    assert colors == list(env.colors) and instr == b.instruction
    np.testing.assert_array_equal(a.state["obj_pos"], b.state["obj_pos"])


def test_objects_are_occluded_after_visible_steps(env):
    obs = env.reset(TaskInstance(spec=TASKS["table_direct"], seed=1))
    assert obs.state["objects_visible"][0] == 1.0 and np.any(obs.state["obj_pos"])
    for _ in range(5):
        obs = env.step(Action(data=np.zeros(2))).observation
    assert obs.state["objects_visible"][0] == 0.0 and not np.any(obs.state["obj_pos"])


def test_indirect_instructions_never_name_the_colour(env):
    obs = env.reset(TaskInstance(spec=TASKS["table_indirect"], seed=3))
    assert obs.instruction in INDIRECT.values()
    assert env.colors[env.target] not in obs.instruction.split()


def test_dwelling_on_the_wrong_pad_fails(env):
    env.reset(TaskInstance(spec=TASKS["table_direct"], seed=2))
    wrong = next(i for i in range(3) if i != env.target)
    from worldcurrent.agents.composable import PlanarIKGoalPolicy

    pol = PlanarIKGoalPolicy(kp=0.5)
    for _ in range(150):
        obs = env._observe()
        r = env.step(Action(data=pol.act(obs, env.pad_pos[wrong]).actions[0]))
        if r.terminated:
            break
    assert r.terminated and r.success is False and "touched" in r.info["fail_reason"]


def test_memory_is_what_makes_the_agent_work(env):
    inst = TaskInstance(spec=TASKS["table_direct"], seed=7)
    with_mem = EpisodeRunner(env, make_object_reach_agent()).run(inst).outcome
    without = EpisodeRunner(env, make_object_reach_agent(memory=None)).run(inst).outcome
    assert with_mem.status == "success" and without.status == "failure"


@pytest.mark.skipif(__import__("os").environ.get("MUJOCO_GL") not in ("egl", "osmesa"), reason="needs MuJoCo rendering")
def test_vision_variant_adds_camera_images_only_while_visible(env):
    from worldcurrent.envs.mujoco.table_reach import table_benchmark

    spec = table_benchmark(vision=True).tasks[0]
    obs = env.reset(TaskInstance(spec=spec, seed=0))
    assert obs.images["top"].shape == (384, 384, 3) and "top" in obs.extra["camera_params"]
    for _ in range(5):
        obs = env.step(Action(data=np.zeros(2))).observation
    assert obs.images == {}


def test_mujoco_envs_conform():
    from worldcurrent.envs.mujoco.planar_reach import PlanarReachEnv, mujoco_tasks
    from worldcurrent.testing.conformance import check_environment

    for factory, task in ((PlanarReachEnv, mujoco_tasks()[1]), (TableReachEnv, TASKS["table_direct"])):
        rep = check_environment(factory, task)
        assert rep.ok, rep.summary()
