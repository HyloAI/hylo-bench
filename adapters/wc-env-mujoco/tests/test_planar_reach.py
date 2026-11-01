import numpy as np
import pytest

from worldcurrent.agents.baseline import PlanarIKAgent
from worldcurrent.core.kernel import EpisodeRunner
from worldcurrent.core.task import TaskInstance
from worldcurrent.core.types import Action

mujoco = pytest.importorskip("mujoco")


@pytest.fixture
def mj():
    from worldcurrent.envs.mujoco.planar_reach import PlanarReachEnv, mujoco_tasks
    return PlanarReachEnv(), mujoco_tasks()


def test_mujoco_reach_succeeds(mj):
    env, tasks = mj
    rec = EpisodeRunner(env, PlanarIKAgent(kp=0.3)).run(TaskInstance(spec=tasks[1], seed=4))
    assert rec.outcome.status == "success"


def test_mujoco_snapshot_restore_is_exact(mj):
    env, tasks = mj
    env.reset(TaskInstance(spec=tasks[0], seed=0))
    for _ in range(5):
        env.step(Action(data=np.array([0.5, -0.5])))
    blob = env.snapshot()
    a = [env.step(Action(data=np.array([0.3, 0.3]))).observation.state["qpos"] for _ in range(5)]
    env.restore(blob)
    b = [env.step(Action(data=np.array([0.3, 0.3]))).observation.state["qpos"] for _ in range(5)]
    np.testing.assert_array_equal(np.stack(a), np.stack(b))


def test_mujoco_divergence_is_reported_not_hidden(mj):
    env, tasks = mj
    env.reset(TaskInstance(spec=tasks[0], seed=0))
    env.data.qvel[:] = np.nan
    r = env.step(Action(data=np.zeros(2)))
    assert r.truncated and "sim_error" in r.info
