from pathlib import Path

import numpy as np
import pytest

from hylo.envs.libero.env import quat2axisangle
from hylo.envs.libero.suites import MAX_STEPS, libero_benchmark, libero_tasks

from hylo.core.profile import load_profile

ROOT = Path(__file__).resolve().parents[3]
SIM_PY = ROOT / "envs" / "libero" / ".venv" / "bin" / "python"
MESA = Path.home() / ".local" / "mesa" / "root" / "usr" / "lib" / "x86_64-linux-gnu"
LOCAL = {"python": str(SIM_PY), "env": {"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa", "LD_LIBRARY_PATH": str(MESA),
                                        "LIBERO_CONFIG_PATH": str(ROOT / "envs" / "libero" / "config")}}
# the simulator runtime registered in this machine's profile (envs/libero/setup.sh), else a venv in this checkout
SIM = {"runtime": "libero-sim"} if "libero-sim" in load_profile().runtimes else (LOCAL if SIM_PY.exists() else None)


def test_task_index_covers_the_standard_suites():
    for suite, n in [("libero_spatial", 10), ("libero_object", 10), ("libero_goal", 10), ("libero_10", 10), ("libero_90", 90)]:
        tasks = libero_tasks(suite)
        assert len(tasks) == n and all(t.max_steps == MAX_STEPS[suite] for t in tasks)
        assert all(t.init.count == 50 and t.success.expr.startswith("(And") for t in tasks)


def test_splits_partition_the_official_50_init_states():
    b = libero_benchmark("libero_goal", n_validation=25, n_heldout=25)
    assert set(b.splits["validation"]) | set(b.splits["heldout"]) == set(range(50))
    with pytest.raises(ValueError):
        libero_benchmark("libero_goal", n_validation=40, n_heldout=20)


def test_quat2axisangle_matches_the_robosuite_formula():
    assert np.allclose(quat2axisangle(np.array([0.0, 0.0, 0.0, 1.0])), 0.0)
    half = np.pi / 4  # 90 degrees about z
    assert np.allclose(quat2axisangle(np.array([0.0, 0.0, np.sin(half), np.cos(half)])), [0.0, 0.0, np.pi / 2])


@pytest.mark.skipif(SIM is None or not (MESA / "libOSMesa.so.8").exists(),
                    reason="needs the libero-sim runtime (envs/libero/setup.sh) and user-space OSMesa")
def test_remote_libero_episode_runs_end_to_end(tmp_path):
    from hylo.agents.baseline import RandomAgent
    from hylo.core.kernel import EpisodeRunner
    from hylo.core.remote.client import RemoteEnvironment
    from hylo.core.task import TaskInstance
    from hylo.core.trace import TraceStore

    env = RemoteEnvironment("hylo.envs.libero.env:LiberoEnv", {"image_size": 128}, **SIM)
    try:
        spec = libero_tasks("libero_spatial", [0])[0].model_copy(update={"max_steps": 5})
        store = TraceStore(tmp_path)
        rec = EpisodeRunner(env, RandomAgent(dim=7), trace=store).run(TaskInstance(spec=spec, seed=3), episode_id="ep")
        assert rec.outcome.status == "failure" and rec.outcome.steps == 5
        obs = store.read("ep").first("observation")["payload"]
        assert obs["images"]["agentview"].shape == (128, 128, 3) and obs["state"]["eef_axis_angle"].shape == (3,)
        assert "black bowl" in obs["instruction"]
    finally:
        env.close()
