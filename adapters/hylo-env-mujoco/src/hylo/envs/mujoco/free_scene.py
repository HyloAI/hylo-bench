"""`mujoco.free_scene`: a task written as a Python scene program (`hylo.freescene/v0.1`) instead of a scene file.

The program (free_program.py) picks the robot, states the instruction, builds the world with MuJoCo's model editor
or MJCF (free_model.py) and judges success from the simulation state (free_world.py). This module is the
environment: it mixes the free scene into the robot environments themselves, so the robot, table, cameras,
control, observation layout, RobotSpec and snapshots are those of `mujoco.panda_table`, `mujoco.aloha_table` or
`mujoco.g1_table`, unchanged; what it adds is the program's model, reset (home posture, randomize, settle), the
program's success function after every step, privileged free-form state and, for checkers, the demo's invisible
hands.

    env = FreeSceneEnv(scene="runs/scenes_free/cable", privileged_obs=True)       # or registry type mujoco.free_scene
    obs = env.reset(task); obs.extra["privileged"]["cables"]["cable"]["nodes"]

Params: scene (a scene.py or the directory holding it; a task's params["scene"] may switch programs but must keep
the robot), demo_hands (invisible hands for demo(); checkers only, 0 otherwise), and the robot environment's
cameras, width, height, segmentation, privileged_obs, control_hz, depth, mesh_detail.

Checker-only methods (never reachable by an agent through the Environment contract): run_demo(), program_targets(),
contacts(), model_xml().
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from hylo.core.robot import RobotSpec
from hylo.core.task import TaskInstance
from hylo.core.types import Action, EnvSpec, Observation, StepResult

from .aloha_table import AlohaTableEnv
from .arm_table import PARK_DEPTH
from .free_model import build_free_model, export_mjcf, robot_facts
from .free_program import FORMAT, ROBOTS, SceneProgram, SceneProgramError
from .free_world import PENETRATION, DemoHands, SceneWorld, SimEdit, SimView
from .g1_table import G1TableEnv
from .panda_table import FINGER_TRAVEL, HAND_CTRL_MAX, HOME, LOWER, UPPER, PandaTableEnv

SETTLE_S = 0.2


class _FreeScene:
    """What the free scene adds to a robot environment. Robot-specific hooks (implemented below per robot base):
    _robot_body, _fingers_of, _home, _apply, _hold_vector, _robot_state, _base_instance."""

    def _free_init(self, program: SceneProgram, demo_hands: int) -> None:
        self.program = program
        self.scene_path = str(program.path)
        self.robot_facts = robot_facts(program.robot)
        self.n_demo_hands = int(demo_hands)
        self.world: SceneWorld | None = None
        self.hands: DemoHands | None = None
        self.group_instance: dict[str, int] = {}
        self._loaded: str | None = None
        self._spec: Any = None
        self.ep = {"items": {}}

    # -- model ------------------------------------------------------------------------------------------------
    def _load_free(self) -> Any:
        import mujoco

        m, self._scene_names, self._spec = build_free_model(self.program, self.width, self.height, self.mesh_detail,
                                                            self.n_demo_hands)
        self._mj = mujoco
        self._loaded = self.program.digest
        return m

    def _index(self) -> None:
        m = self._model
        robot = {b for b in range(m.nbody) if self._robot_body(m, b)}
        self.world = SceneWorld(m, self._scene_names, robot, self._fingers_of(m), self.program.labels)
        self.hands = DemoHands(self.world, self.n_demo_hands)
        self.geom_instance, self.group_instance = self.world.instances(self.geom_instance, self._base_instance())

    def _close_cams(self) -> None:
        if self._cams is not None:
            self._cams.close()
            self._cams = None

    def model_xml(self) -> str:
        """MJCF of the whole compiled world (robot and scene), e.g. for an offline renderer."""
        if self._model is None:
            self._build()
        return export_mjcf(self._spec, self.program.robot)

    # -- views ------------------------------------------------------------------------------------------------
    def _forward(self) -> None:
        self._mj.mj_forward(self._model, self.data)

    def view(self) -> SimView:
        """The read-only `sim` a success function gets."""
        return SimView(self.world, self.data, self.robot_facts["table_top"])

    def editor(self, demo: bool = False) -> SimEdit:
        """The editing `sim` randomize (and, with demo=True, demo) gets."""
        return SimEdit(self.world, self.data, self.robot_facts["table_top"], self._forward,
                       self.hands if demo else None, self._demo_step if demo else None, self.control_hz)

    def contacts(self, depth: float = PENETRATION) -> list[dict[str, Any]]:
        """Contacts deeper than `depth` involving the program's things (checkers)."""
        return self.world.contacts(self.data, depth)

    def privileged_state(self, json_ready: bool = False) -> dict[str, Any]:
        f = self.robot_facts
        out = {"format": FORMAT, "robot": self.program.robot, "instruction": self.program.instruction,
               "time": float(self.data.time), **self.world.state(self.data),
               "table": {"height": f["table_top"], "x": list(f["table_x"]), "y": list(f["table_y"])},
               "arm_zones": {k: {"x": list(v["x"]), "y": list(v["y"])} for k, v in f["arms"].items()},
               **self._robot_state(), "start": self.ep.get("start", {})}
        return json.loads(json.dumps(out)) if json_ready else out

    def segments(self) -> dict[str, str]:
        return {"1": "table", "2": "robot", **{str(i): self.world.label(g) for g, i in self.group_instance.items()}}

    # -- success ----------------------------------------------------------------------------------------------
    def evaluate_success(self) -> tuple[bool, dict[str, Any]]:
        """The program's success(sim) now: (value, info). Errors and non-bool values are reported in info."""
        try:
            v = self.program.call(self.program.success, self.view())
        except SceneProgramError as exc:
            return False, {"success_error": str(exc)[:600]}
        if isinstance(v, (bool, np.bool_)):
            return bool(v), {}
        try:
            return bool(v), {"success_type": type(v).__name__}
        except Exception:  # noqa: BLE001 - e.g. an array: ambiguous truth value
            return False, {"success_error": f"success() returned {type(v).__name__}, not a bool"}

    # -- episode ----------------------------------------------------------------------------------------------
    def _free_reset(self, task: TaskInstance) -> Observation:
        path = task.spec.params.get("scene")
        if path and str(Path(path).resolve()) not in (self.scene_path, str(Path(self.scene_path).parent)):
            prog = SceneProgram(path)
            if prog.robot != self.program.robot:
                raise ValueError(f"task scene {path} uses robot {prog.robot}; this environment was made for "
                                 f"{self.program.robot}")
            self.program, self.scene_path = prog, str(prog.path)
        if self._model is None or self._loaded != self.program.digest:
            self._close_cams()
            self._build()
        mj, m, d = self._mj, self._model, self.data
        self.task = task
        mj.mj_resetData(m, d)
        self._home()
        self.hands.reset(d)
        mj.mj_forward(m, d)
        if self.program.randomize is not None:
            self.program.call(self.program.randomize, self.editor(), np.random.default_rng(task.seed))
            mj.mj_forward(m, d)
        start = self.contacts()
        mj.mj_step(m, d, nstep=int(SETTLE_S / m.opt.timestep))
        d.time = 0.0
        mj.mj_forward(m, d)
        self._warn = self._divergence_count()
        self.t = 0
        self.ep = {"items": {}, "task": f"free/{self.program.path.parent.name}", "instruction": self.program.instruction,
                   "scene": self.scene_path, "digest": self.program.digest, "start": {"contacts": start}}
        return self._observe()

    def _advance(self, action: np.ndarray) -> bool:
        """One control step; False when the simulation diverged."""
        mj, m, d = self._mj, self._model, self.data
        self._apply(action)
        mj.mj_step(m, d, nstep=self.n_substeps)
        mj.mj_forward(m, d)
        bad = self._divergence_count()
        ok = bad == self._warn and bool(np.all(np.isfinite(d.qpos)))
        self._warn = bad
        return ok

    def _free_step(self, action: Action) -> StepResult:
        self.t += 1
        if not self._advance(np.asarray(action.data, dtype=np.float64).reshape(-1)):
            return StepResult(observation=self._observe(), reward=None, terminated=False, truncated=True,
                              success=None, info={"sim_error": "mujoco divergence warning (BADQPOS/QVEL/QACC)"})
        success, info = self.evaluate_success()
        return StepResult(observation=self._observe(), reward=1.0 if success else 0.0, terminated=bool(success),
                          truncated=False, success=bool(success), info=info)

    def _free_apply_episode(self) -> None:
        if self.ep.get("digest") and self.ep["digest"] != self.program.digest:
            raise RuntimeError(f"{self.scene_path} changed since the snapshot was taken")

    # -- demo (checkers only) ---------------------------------------------------------------------------------
    def _demo_step(self) -> bool:
        if not self._advance(self._hold_vector()):
            raise SceneProgramError("demo: the simulation diverged")
        return self.evaluate_success()[0]

    def run_demo(self, after_s: float = 1.0) -> dict[str, Any]:
        """Run the program's demo(sim) from the current state with the invisible hands (the robot holds still),
        then `after_s` seconds more with every hand released. Returns {ran, error, success_seen, success_at,
        success_held_end, success_end, grabs, steps}."""
        if self.program.demo is None:
            return {"ran": False}
        hands = self.hands
        hands.reset(self.data)
        error = None
        try:
            self.program.call(self.program.demo, self.editor(demo=True))
        except SceneProgramError as exc:
            error = str(exc)[:600]
        held_end = self.evaluate_success()[0]
        if error is None:
            for i in hands.hands:
                hands.release(self.data, i)
            try:
                hands.move(self.data, {}, int(round(after_s * self.control_hz)), self._demo_step)
            except SceneProgramError as exc:
                error = str(exc)
        return {"ran": True, "error": error, "success_seen": hands.success_at is not None,
                "success_at": hands.success_at, "success_held_end": held_end, "success_end": self.evaluate_success()[0],
                "grabs": list(hands.log), "steps": hands.steps}

    def program_targets(self) -> dict[str, list[float]]:
        """targets(sim) of the program ({name: xyz}), {} without one."""
        if self.program.targets is None:
            return {}
        t = self.program.call(self.program.targets, self.view()) or {}
        return {str(k): [float(x) for x in np.asarray(v, float).reshape(3)] for k, v in dict(t).items()}

    # -- Environment contract (shared by the robot classes) ---------------------------------------------------
    def reset(self, task: TaskInstance) -> Observation:
        return self._free_reset(task)

    def step(self, action: Action) -> StepResult:
        return self._free_step(action)


def free_notes(facts: dict[str, Any], robot_notes: str) -> str:
    """The robot environment's notes without its sentences about its own task objects, plus the free scene's."""
    skip = ("objects start", "block", "button", "box", "plate", "success needs")
    keep = [s for s in re.split(r"(?<=\.) ", robot_notes) if not any(w in s.lower() for w in skip)]
    reach = "; ".join(f"arm {k} serves x {list(v['x'])}, y {list(v['y'])}" for k, v in facts["arms"].items())
    return (" ".join(keep) + f" Objects come from a free scene program; {reach}. The task is judged by the program's "
            "own success function (privileged runs can read body poses, cable nodes, flex vertices and joint values "
            "through robot.scene()).")


class FreePandaEnv(_FreeScene, PandaTableEnv):
    """mujoco.free_scene with ROBOT = "panda" (PandaTableEnv's robot, cameras and control)."""

    def __init__(self, program: SceneProgram, demo_hands: int = 0, **kw: Any) -> None:
        PandaTableEnv.__init__(self, **kw)
        self._free_init(program, demo_hands)

    def spec(self) -> EnvSpec:
        return PandaTableEnv.spec(self).model_copy(update={"name": "mujoco.free_scene"})

    def robot_spec(self) -> RobotSpec:
        base = PandaTableEnv.robot_spec(self)
        return base.model_copy(update={"notes": free_notes(self.robot_facts, base.notes)})

    def _build(self) -> None:
        m = self._load_free()
        self._bind_robot(m)                        # PandaTableEnv's arm indices, cameras
        gi = np.zeros(m.ngeom, np.int32)
        for g in range(m.ngeom):
            if self._robot_body(m, int(m.geom_bodyid[g])):
                gi[g] = 2
        gi[m.geom("table").id] = 1
        self.geom_instance = gi
        self._sig = self._mj.mjtState.mjSTATE_INTEGRATION
        self._warn = 0
        self._index()

    def _base_instance(self) -> int:
        return 3

    def _robot_body(self, m: Any, b: int) -> bool:
        return m.body(b).name.startswith("panda/")

    def _fingers_of(self, m: Any) -> set[int]:
        return set(self._fingers)

    def _home(self) -> None:
        d = self.data
        d.qpos[self._qadr], d.qpos[self._finger_qadr] = HOME, FINGER_TRAVEL
        d.ctrl[:7], d.ctrl[7] = HOME, HAND_CTRL_MAX

    def _apply(self, a: np.ndarray) -> None:
        self.data.ctrl[:7] = np.clip(a[:7], LOWER, UPPER)
        self.data.ctrl[7] = HAND_CTRL_MAX * float(np.clip(a[7], 0.0, 1.0))

    def _hold_vector(self) -> np.ndarray:
        return np.concatenate([self.data.ctrl[:7], [self.data.ctrl[7] / HAND_CTRL_MAX]])

    def _robot_state(self) -> dict[str, Any]:
        return {"q": {"arm": self.data.qpos[self._qadr].tolist()}, "hand": {"arm": [self.hand_state()]},
                "tcp": {"arm": self.tcp_pose().tolist()}}

    def _apply_episode(self) -> None:
        self._free_apply_episode()


class _FreeArmTable(_FreeScene):
    """The free scene on an ArmTableEnv robot (ALOHA, G1): its object pool stays in the model, parked."""

    def _build_model(self) -> Any:
        return self._load_free()

    def _build(self) -> None:
        super()._build()                           # ArmTableEnv: arms, the (parked) pool, instance ids
        self._index()

    def _base_instance(self) -> int:
        return 3 + len(self.pool_names)

    def _robot_body(self, m: Any, b: int) -> bool:
        return self._is_robot_body(m, b)

    def _fingers_of(self, m: Any) -> set[int]:
        return set().union(*(a.fingers for a in self.arms))

    def _park(self) -> None:
        episode, self.ep = self.ep, {"items": {}}
        super()._apply_episode()                   # ArmTableEnv: no active items, so every pool item is parked
        self.ep = {**episode, "items": {}}

    def _home(self) -> None:
        d = self.data
        self._park()
        for name in self.items:
            self._place(name, [*self.PARK_XY, PARK_DEPTH], 0.0)
        self._extra_reset()
        for a in self.arms:
            d.qpos[a.qadr], d.ctrl[a.ctrl] = a.home, a.home
            d.ctrl[a.hand_ctrl] = a.hand_to_ctrl(np.asarray(a.hand_open, float))
            self._set_fingers_open(a)

    def _apply(self, a: np.ndarray) -> None:
        self._apply_action(a)

    def _hold_vector(self) -> np.ndarray:
        return np.concatenate([np.concatenate([self.data.ctrl[a.ctrl], a.hand_open]) for a in self.arms])

    def _robot_state(self) -> dict[str, Any]:
        return {"q": {a.name: self.data.qpos[a.qadr].tolist() for a in self.arms},
                "hand": {a.name: self.hand_state(a.name).tolist() for a in self.arms},
                "tcp": {a.name: self.tcp_pose(a.name).tolist() for a in self.arms}}

    def _apply_episode(self) -> None:
        self._free_apply_episode()
        self._park()

    def spec(self) -> EnvSpec:
        return super().spec().model_copy(update={"name": "mujoco.free_scene"})

    def robot_spec(self) -> RobotSpec:
        base = super().robot_spec()
        return base.model_copy(update={"notes": free_notes(self.robot_facts, base.notes)})


class FreeAlohaEnv(_FreeArmTable, AlohaTableEnv):
    """mujoco.free_scene with ROBOT = "aloha" (AlohaTableEnv's two arms, cameras and control)."""

    def __init__(self, program: SceneProgram, demo_hands: int = 0, **kw: Any) -> None:
        AlohaTableEnv.__init__(self, **kw)
        self._free_init(program, demo_hands)


class FreeG1Env(_FreeArmTable, G1TableEnv):
    """mujoco.free_scene with ROBOT = "g1" (G1TableEnv's humanoid, cameras and control)."""

    def __init__(self, program: SceneProgram, demo_hands: int = 0, **kw: Any) -> None:
        G1TableEnv.__init__(self, **kw)
        self._free_init(program, demo_hands)


def FreeSceneEnv(scene: str | None = None, demo_hands: int = 0, **params: Any) -> Any:
    """`mujoco.free_scene`: the environment for the scene program at `scene` (its ROBOT picks the class)."""
    if not scene:
        raise ValueError("mujoco.free_scene needs a scene program: env param `scene` (a scene.py or its directory)")
    program = SceneProgram(scene)
    cls = {"panda": FreePandaEnv, "aloha": FreeAlohaEnv, "g1": FreeG1Env}[program.robot]
    return cls(program, demo_hands=demo_hands, **params)


__all__ = ["FreeAlohaEnv", "FreeG1Env", "FreePandaEnv", "FreeSceneEnv", "ROBOTS", "free_notes"]
