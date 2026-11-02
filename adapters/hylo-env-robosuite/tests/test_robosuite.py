"""robosuite adapter: pure tests always run; simulator tests need the robosuite-sim runtime (bench/envs/robosuite/
setup.sh) and user-space OSMesa, and skip otherwise."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from hylo.core.profile import load_profile
from hylo.envs.robosuite.suites import robosuite_benchmark, robosuite_tasks
from hylo.envs.robosuite.tasks import TASKS, base_in_world, task

ROOT = Path(__file__).resolve().parents[3]                     # the bench repository
EXAMPLES = ROOT / "adapters" / "hylo-env-robosuite" / "examples"
SIM_PY = ROOT / "envs" / "robosuite" / ".venv" / "bin" / "python"
MESA = Path.home() / ".local" / "mesa" / "root" / "usr" / "lib" / "x86_64-linux-gnu"
LOCAL = {"python": str(SIM_PY), "env": {"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa", "LD_LIBRARY_PATH": str(MESA)}}
SIM = {"runtime": "robosuite-sim"} if "robosuite-sim" in load_profile().runtimes else (LOCAL if SIM_PY.exists() else None)
needs_sim = pytest.mark.skipif(SIM is None or not (MESA / "libOSMesa.so.8").exists(),
                               reason="needs the robosuite-sim runtime (bench/envs/robosuite/setup.sh) and OSMesa")


def test_task_table_covers_robosuites_standard_tasks():
    envs = {t.env for t in TASKS.values()}
    assert {"Lift", "Stack", "PickPlaceMilk", "PickPlaceBread", "PickPlaceCereal", "PickPlaceCan", "NutAssemblySquare",
            "NutAssemblyRound", "Door", "Wipe", "TwoArmLift", "TwoArmHandover", "TwoArmPegInHole"} == envs
    assert all(t.arms == 2 for t in TASKS.values() if t.env.startswith("TwoArm"))
    with pytest.raises(ValueError):
        task("juggle")


def test_benchmark_splits_are_disjoint_and_seeded():
    b = robosuite_benchmark(["lift", "two_arm_lift"])
    assert set(b.splits) == {"search", "search_val", "gate", "heldout", "validation"}
    assert len(b.splits["validation"]) == 20 and b.splits["validation"][0] == 10_000
    assert sum(len(v) for k, v in b.splits.items() if k != "validation") == 50
    std = robosuite_benchmark(protocol="standard")
    assert set(std.splits) == {"train", "validation", "heldout"} and len(std.tasks) == len(TASKS)
    spec = robosuite_tasks(["two_arm_lift"])[0]
    assert spec.params == {"task": "two_arm_lift", "env": "TwoArmLift", "arms": 2} and "bimanual" in spec.tags


def test_second_arm_stands_opposite_the_first():
    T = np.asarray(base_in_world(task("two_arm_lift"), 1))
    assert np.allclose(T[:3, 3], [1.12, 0.0, 0.0]) and np.allclose(T[:3, 0], [-1.0, 0.0, 0.0])


def _env(key, **kw):
    from hylo.envs.robosuite.env import RobosuiteEnv

    env = RobosuiteEnv(task=key, **kw)
    env.panda_model = lambda names: {"model": f"/x/{names.index}.xml", "geometry": {
        "tcp_in_ee": np.eye(4).tolist(), "max_width": 0.079 if names.fingered else 0.0, "tcp_offset": 0.09}}
    return env


def test_env_spec_follows_the_task_without_a_simulator():
    one = _env("lift").spec()
    assert one.action.shape == (8,) and [a.name for a in one.robot.arms] == ["arm"]
    assert [c.name for c in one.robot.cameras] == ["agentview", "sideview", "wrist"] and one.robot.table_height == -0.112
    two = _env("two_arm_handover", segmentation=True).spec()
    assert two.action.shape == (16,) and [a.name for a in two.robot.arms] == ["robot0", "robot1"]
    assert two.robot.arm("robot1").base_in_world[0][3] == 1.62 and "hammer handle" in two.robot.notes
    peg = _env("two_arm_peg_in_hole").spec()
    assert peg.robot.table_height is None and peg.robot.arm("robot0").hand.joints == ()
    with pytest.raises(ValueError):
        _env("pick_place_can", cameras=["sideview"])            # PickPlace scenes have no sideview
    with pytest.raises(ValueError):
        _env("two_arm_lift", cameras=["wrist"])                 # two arms: robot0_wrist / robot1_wrist
    with pytest.raises(ValueError):
        _env("")


def test_programs_pass_the_sandbox_and_cover_every_task():
    pytest.importorskip("hylo.codeagents")
    from hylo.codeagents.sandbox import validate

    spec = importlib.util.spec_from_file_location("robosuite_run", EXAMPLES / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    assert set(run.SUITE.programs) == set(TASKS)
    for name in set(run.SUITE.programs.values()):
        validate((EXAMPLES / "programs" / f"{name}.py").read_text())


def test_runner_writes_one_task_per_experiment(tmp_path):
    from hylo.envs.robosuite_contract.runs import code_llm_config, program_config

    spec = importlib.util.spec_from_file_location("robosuite_run", EXAMPLES / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    cfg = program_config(run.SUITE, "stack", 20, tmp_path)
    assert cfg["env"]["params"] == {"task": "stack", "segmentation": True}
    assert cfg["benchmark"]["params"] == {"tasks": ["stack"], "validation": 20}
    assert cfg["agent"]["type"] == "code.program" and "def solve(robot)" in cfg["agent"]["params"]["source"]
    assert cfg["env"]["remote"]["runtime"] == "robosuite-sim" and "PYTHONPATH" in cfg["env"]["remote"]["env"]
    llm = code_llm_config(run.SUITE, "lift", 3, tmp_path)
    assert llm["agent"]["params"]["model"]["model"] == "qwen3.5:35b-a3b" and llm["trace"]["record_privileged"] is False


# ---------------------------------------------------------------- simulator
def _remote(params):
    from hylo.core.remote.client import RemoteEnvironment
    from hylo.envs.robosuite_contract.runs import Suite, remote

    suite = Suite("robosuite", "", SIM.get("runtime") or "", ("hylo.envs.robosuite_contract", "hylo.envs.robosuite"),
                  EXAMPLES, {})
    env_vars = {**(SIM.get("env") or {}), **remote(suite)["env"]}
    return RemoteEnvironment("hylo.envs.robosuite.env:RobosuiteEnv", params, runtime=SIM.get("runtime"),
                             python=SIM.get("python"), env=env_vars)


@needs_sim
@pytest.mark.parametrize("key", ["lift", "two_arm_lift"])
def test_contract_geometry_matches_the_simulator(key):
    """FK of the exported model = the simulator's hands; object origins project onto their own segment at a
    consistent depth; joint targets are tracked; the hand closes; the same seed gives the same scene."""
    from hylo.core.task import TaskInstance
    from hylo.core.types import Action
    from hylo.manip.kinematics import Kinematics

    env = _remote({"task": key, "segmentation": True, "privileged_obs": True, "image_size": 128})
    try:
        spec = env.spec()
        kin = Kinematics(spec.robot)
        inst = TaskInstance(spec=robosuite_tasks([key])[0], seed=10_003)
        obs = env.reset(inst)
        priv, segs = obs.extra["privileged"], obs.extra["segments"]
        for a in spec.robot.arms:
            assert Path(a.model).is_file()
            fk = kin[a.name].fk(obs.state[f"q.{a.name}"])
            assert np.allclose(fk, np.asarray(priv["ee"][a.name]) @ np.asarray(a.hand.tcp_in_ee), atol=1e-6)
        hits = 0
        for o in priv["objects"].values():
            for cam, meta in obs.extra["cameras"].items():
                K, T = np.asarray(meta["K"]), np.asarray(meta["T_world_cam"])
                p = np.linalg.inv(T) @ np.r_[o["pos"], 1.0]
                if p[2] <= 0:
                    continue
                u, v = (K @ p[:3])[:2] / p[2]
                if 0 <= round(u) < 128 and 0 <= round(v) < 128:
                    window = obs.images[f"{cam}.seg"][max(0, round(v) - 2):round(v) + 3, max(0, round(u) - 2):round(u) + 3]
                    if any(o["label"] in (segs.get(str(i)) or "") for i in window.ravel()):
                        hits += 1
                        # depth at the origin's pixel: the object's surface around it (a hollow pot: its inside)
                        assert abs(obs.images[f"{cam}.depth"][round(v), round(u)] - p[2]) < 0.15
        assert hits >= 2
        again = env.reset(inst)
        assert np.allclose(again.extra["privileged"]["objects"][next(iter(priv["objects"]))]["pos"],
                           next(iter(priv["objects"].values()))["pos"])
        target = np.concatenate([np.r_[again.state[f"q.{a.name}"] + [0.1, 0.05, 0, -0.1, 0, 0.1, 0.2], 0.0]
                                 for a in spec.robot.arms])
        for _ in range(25):
            res = env.step(Action(data=target))
        a0 = spec.robot.arms[0].name
        assert np.max(np.abs(res.observation.state[f"q.{a0}"] - target[:7])) < 0.01
        assert res.observation.state[f"hand.{a0}"][0] < 0.05
    finally:
        env.close()


@needs_sim
def test_episode_runs_through_the_kernel(tmp_path):
    from hylo.agents.baseline import RandomAgent
    from hylo.core.kernel import EpisodeRunner
    from hylo.core.task import TaskInstance
    from hylo.core.trace import TraceStore

    env = _remote({"task": "stack", "image_size": 64})
    try:
        spec = robosuite_tasks(["stack"])[0].model_copy(update={"max_steps": 3})
        rec = EpisodeRunner(env, RandomAgent(dim=8), trace=TraceStore(tmp_path)).run(
            TaskInstance(spec=spec, seed=1), episode_id="ep")
        assert rec.outcome.steps == 3 and rec.outcome.status == "failure"
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
