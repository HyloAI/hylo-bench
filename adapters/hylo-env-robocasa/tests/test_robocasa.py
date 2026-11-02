"""RoboCasa adapter: pure tests always run; the simulator test needs the robocasa-sim runtime with RoboCasa's assets
(bench/envs/robocasa/setup.sh) and user-space OSMesa, and skips otherwise."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from hylo.core.profile import load_profile
from hylo.envs.robocasa.suites import robocasa_benchmark, robocasa_tasks
from hylo.envs.robocasa.tasks import TASKS, fixture_label, object_label, part_rules

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "adapters" / "hylo-env-robocasa" / "examples"
SIM_PY = ROOT / "envs" / "robocasa" / ".venv" / "bin" / "python"
ASSETS = ROOT / "envs" / "robocasa" / "src-robocasa" / "robocasa" / "models" / "assets" / "objects" / "objaverse"
MESA = Path.home() / ".local" / "mesa" / "root" / "usr" / "lib" / "x86_64-linux-gnu"
SIM = {"runtime": "robocasa-sim"} if "robocasa-sim" in load_profile().runtimes else None
needs_sim = pytest.mark.skipif(SIM is None or not ASSETS.is_dir() or not (MESA / "libOSMesa.so.8").exists(),
                               reason="needs the robocasa-sim runtime and assets (bench/envs/robocasa/setup.sh), OSMesa")


def test_seven_atomic_tasks_over_four_skills():
    assert len(TASKS) == 7
    assert {t.skill for t in TASKS.values()} == {"pick_place", "open_close", "knob", "lever", "button"}
    specs = robocasa_tasks()
    assert all(s.max_steps in (450, 600, 750) for s in specs) and specs[0].params["task"] in TASKS


def test_splits_are_disjoint_seed_ranges():
    b = robocasa_benchmark(["open_drawer"])
    seeds = [s for v in b.splits.values() for s in v]
    assert len(seeds) == len(set(seeds)) == 70 and b.splits["validation"][0] == 10_000


def test_labels_name_fixtures_objects_and_parts():
    assert fixture_label("HingeCabinet") == "cabinet" and fixture_label("Wall") == ""
    assert fixture_label("SomeNewAppliance") == "some new appliance"
    assert object_label("bottled_water") == "bottled water"
    names = ["stove_1_knob_rear_left_main", "stove_1_knob_front_right", "stove_1_door_handle_main", "stove_1_main"]
    rules = dict((label, rx) for rx, label in part_rules("stove_1_", "stove", names))
    assert {"rear left knob", "front right knob", "stove handle"} <= set(rules)
    sink = dict((label, rx) for rx, label in part_rules("sink_1_", "sink", ["sink_1_handle_main", "sink_1_spout"]))
    assert set(sink) == {"faucet handle", "faucet spout"}
    mw = dict((label, rx) for rx, label in part_rules("mw_", "microwave", ["mw_start_button", "mw_stop_button"]))
    assert set(mw) == {"start button", "stop button"}


def test_env_spec_without_a_simulator():
    from hylo.envs.robocasa.env import RoboCasaEnv

    env = RoboCasaEnv(task="open_drawer", segmentation=True)
    env.panda_model = lambda names: {"model": "/x.xml", "geometry": {"tcp_in_ee": np.eye(4).tolist(),
                                                                      "max_width": 0.079, "tcp_offset": 0.09}}
    spec = env.spec()
    assert spec.name == "robocasa" and spec.action.shape == (8,) and spec.robot.table_height is None
    assert [c.name for c in spec.robot.cameras] == ["agentview_left", "agentview_right", "wrist"]
    assert "drawer handle" in spec.robot.notes
    with pytest.raises(ValueError):
        RoboCasaEnv(task="make_coffee")


def test_programs_pass_the_sandbox_and_cover_every_task():
    pytest.importorskip("hylo.codeagents")
    from hylo.codeagents.sandbox import validate

    spec = importlib.util.spec_from_file_location("robocasa_run", EXAMPLES / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    assert set(run.SUITE.programs) == set(TASKS)
    for name in set(run.SUITE.programs.values()):
        validate((EXAMPLES / "programs" / f"{name}.py").read_text())


@needs_sim
def test_kitchen_episode_follows_the_contract():
    """A RoboCasa kitchen through the remote protocol: FK of the exported Panda = the PandaOmron's arm in the scene,
    the referenced drawer and its handle are labelled, the instruction is RoboCasa's, joint targets are tracked."""
    from hylo.core.remote.client import RemoteEnvironment
    from hylo.core.task import TaskInstance
    from hylo.core.types import Action
    from hylo.envs.robosuite_contract.runs import Suite, remote
    from hylo.manip.kinematics import Kinematics

    suite = Suite("robocasa", "", "robocasa-sim", ("hylo.envs.robosuite_contract", "hylo.envs.robocasa"), EXAMPLES, {})
    env = RemoteEnvironment("hylo.envs.robocasa.env:RoboCasaEnv",
                            {"task": "open_drawer", "segmentation": True, "privileged_obs": True, "image_size": 96},
                            runtime="robocasa-sim", env=remote(suite)["env"])
    try:
        spec = env.spec()
        obs = env.reset(TaskInstance(spec=robocasa_tasks(["open_drawer"])[0], seed=10_001))
        arm = spec.robot.arm("arm")
        fk = Kinematics(spec.robot)["arm"].fk(obs.state["q.arm"])
        assert np.allclose(fk, np.asarray(obs.extra["privileged"]["ee"]["arm"]) @ np.asarray(arm.hand.tcp_in_ee),
                           atol=1e-5)
        labels = set(obs.extra["segments"].values())
        assert {"counter", "robot", "drawer", "drawer handle"} <= labels
        assert "drawer" in obs.instruction.lower()
        target = np.r_[obs.state["q.arm"] + [0.1, 0.05, 0, -0.1, 0, 0.1, 0.2], 0.0]
        for _ in range(25):
            res = env.step(Action(data=target))
        assert np.max(np.abs(res.observation.state["q.arm"] - target[:7])) < 0.01
    finally:
        env.close()
