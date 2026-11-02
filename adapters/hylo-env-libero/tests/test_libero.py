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


# ---------------------------------------------------------------- the robot contract (RFC-0006)
def test_vla_mode_spec_is_unchanged():
    """The default (VLA) mode publishes the same EnvSpec as before the robot contract existed (digest of its JSON)."""
    import hashlib

    from hylo.envs.libero.env import LiberoEnv

    spec = LiberoEnv().spec()
    assert spec.robot is None and spec.action.shape == (7,) and spec.action.semantics.control_mode == "eef_delta_pose"
    assert hashlib.sha256(spec.model_dump_json().encode()).hexdigest()[:16] == "8bc9aee2a1d7ea34"
    assert hashlib.sha256(LiberoEnv(image_size=128).spec().model_dump_json().encode()).hexdigest()[:16] == \
        "7f89754538f68a40"


def test_contract_params_are_checked():
    from hylo.envs.libero.env import LiberoEnv

    with pytest.raises(ValueError):
        LiberoEnv(tool_obs=True)                       # the contract needs joint-position control
    with pytest.raises(ValueError):
        LiberoEnv(control="joint_position", tool_obs=True, cameras=["frontview"])
    with pytest.raises(ValueError):
        LiberoEnv(control="osc_pose")


def test_bddl_objects_get_the_names_instructions_use():
    from hylo.envs.libero.robot import PARTS, object_label

    assert object_label("akita_black_bowl_2") == "black bowl"
    assert object_label("glazed_rim_porcelain_ramekin_1") == "ramekin"
    assert object_label("white_yellow_mug_1") == "yellow and white mug"
    assert object_label("alphabet_soup_1") == "alphabet soup"         # default: the category's words
    assert PARTS["cabinet_top"].format(o="wooden cabinet") == "top drawer of the wooden cabinet"


def _sim_tool():
    """The simulator runtime, importing hylo-core and this adapter from this checkout (the runtime's venv may have
    been built from another one)."""
    import importlib.util
    import os

    def src(module: str) -> str:
        spec = importlib.util.find_spec(module)
        return str(Path(list(spec.submodule_search_locations)[0]).parents[len(module.split(".")) - 1])

    env = dict(SIM.get("env") or {})
    env["PYTHONPATH"] = os.pathsep.join([src("hylo.core"), src("hylo.envs.libero")])
    return {**SIM, "env": env}


needs_sim = pytest.mark.skipif(SIM is None or not (MESA / "libOSMesa.so.8").exists(),
                               reason="needs the libero-sim runtime (envs/libero/setup.sh) and user-space OSMesa")


@needs_sim
def test_robot_contract_geometry_matches_the_simulator():
    """RobotSpec, observation layout, FK of the exported model against the simulator, and camera geometry: privileged
    object positions projected with K and T_world_cam land on that object's segment, at a consistent depth."""
    from hylo.core.remote.client import RemoteEnvironment
    from hylo.core.task import TaskInstance
    from hylo.core.types import Action
    from hylo.manip.kinematics import Kinematics

    env = RemoteEnvironment("hylo.envs.libero.env:LiberoEnv",
                            {"control": "joint_position", "tool_obs": True, "segmentation": True, "privileged_obs": True,
                             "image_size": 128}, **_sim_tool())
    try:
        spec = env.spec()
        robot = spec.robot
        assert robot is not None and spec.action.shape == (8,) and spec.action.semantics.control_mode == "joint_pos"
        arm = robot.arm("arm")
        assert Path(arm.model).is_file() and arm.base_in_world == [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0],
                                                                   [0, 0, 0, 1.0]]
        assert 0.07 < arm.hand.max_width < 0.085 and [c.name for c in robot.cameras] == ["agentview", "wrist"]
        kin = Kinematics(robot)["arm"]
        T_tcp = np.asarray(arm.hand.tcp_in_ee)
        for suite in ("libero_spatial", "libero_object"):              # a table scene and a floor scene
            task = libero_tasks(suite, [0])[0]
            obs = env.reset(TaskInstance(spec=task, seed=1))
            assert obs.state["q.arm"].shape == (7,) and 0.9 < obs.state["hand.arm"][0] <= 1.0
            for cam in ("agentview", "wrist"):
                assert obs.images[cam].shape == (128, 128, 3) and obs.images[f"{cam}.depth"].dtype == np.float32
                assert obs.images[f"{cam}.seg"].shape == (128, 128)
            priv, segs = obs.extra["privileged"], obs.extra["segments"]
            # forward kinematics of the exported model = the simulator's hand
            ee = np.asarray(priv["ee"])
            assert np.allclose(kin.fk(obs.state["q.arm"]), ee @ T_tcp, atol=1e-4)
            # every BDDL object in agentview: its projected origin falls on its own segment, depth near the camera z
            hits = 0
            cam = obs.extra["cameras"]["agentview"]
            K, T = np.asarray(cam["K"]), np.asarray(cam["T_world_cam"])
            for name, o in priv["objects"].items():
                p = np.linalg.inv(T) @ np.r_[np.asarray(o["pos"]) + [0.0, 0.0, 0.01], 1.0]  # just above the base
                u, v = (K @ p[:3])[:2] / p[2]
                if not (0 <= round(u) < 128 and 0 <= round(v) < 128):
                    continue
                window = obs.images["agentview.seg"][max(0, round(v) - 2):round(v) + 3, max(0, round(u) - 2):round(u) + 3]
                labels = {segs.get(str(i)) for i in window.ravel()}
                if o["label"] in labels or any(lab and lab.endswith(o["label"]) for lab in labels):
                    hits += 1
                    d = obs.images["agentview.depth"][round(v), round(u)]
                    assert 0.0 < d < p[2] + 0.05
            assert hits >= 3, (suite, hits)
            # joint-position control tracks a target; the hand closes to a measured opening near 0
            target = obs.state["q.arm"] + np.array([0.15, 0.1, 0.0, -0.1, 0.0, 0.1, 0.2])
            for _ in range(25):
                res = env.step(Action(data=np.r_[target, 0.0]))
            assert np.max(np.abs(res.observation.state["q.arm"] - target)) < 0.01
            assert res.observation.state["hand.arm"][0] < 0.05
    finally:
        env.close()


def test_long_horizon_program_splits_libero_10_instructions_into_steps():
    """examples/programs/libero_long.py: each libero_10 instruction becomes the steps the program will run."""
    src = (ROOT / "adapters" / "hylo-env-libero" / "examples" / "programs" / "libero_long.py").read_text()
    ns: dict = {}
    exec(compile(src, "libero_long.py", "exec"), ns)
    labels = ["alphabet soup", "cream cheese box", "tomato sauce", "butter", "basket", "moka pot", "stove", "stove knob",
              "black bowl", "white cabinet", "bottom drawer of the white cabinet", "white mug", "yellow and white mug",
              "plate", "chocolate pudding", "book", "caddy", "microwave", "microwave door"]
    names = ns["short_names"](labels)

    def steps(text):
        out, last = [], None
        for clause in ns["clauses"](text):
            if clause.split()[0] in ("turn", "close"):
                out.append(clause)
                last = "stove" if clause.startswith("turn on") else last
                continue
            st = ns["place_steps"](clause, names, last)
            out += [(o, t, v, m) for o, t, v, m in st]
            last = st[-1][1] if st else last
        return out

    assert steps("put both the cream cheese box and the butter in the basket") == [
        ("cream cheese box", "basket", "in", {}), ("butter", "basket", "in", {})]
    assert steps("turn on the stove and put the moka pot on it") == ["turn on the stove", ("moka pot", "stove", "on", {})]
    assert steps("put the black bowl in the bottom drawer of the cabinet and close it") == [
        ("black bowl", "bottom drawer of the white cabinet", "in", {}), "close it"]
    assert steps("put the white mug on the left plate and put the yellow and white mug on the right plate") == [
        ("white mug", "plate", "on", {"side": "left"}), ("yellow and white mug", "plate", "on", {"side": "right"})]
    assert steps("put the white mug on the plate and put the chocolate pudding to the right of the plate")[1] == (
        "chocolate pudding", "plate", "on", {"relation": "right_of"})
    assert steps("pick up the book and place it in the back compartment of the caddy") == [
        ("book", "caddy", "in", {"side": "back"})]
    assert steps("put both moka pots on the stove") == [("moka pot", "stove", "on", {})] * 2
