"""`RobosuiteContractEnv`: the robot contract (hylo.robot/v0.1) over a robosuite environment, as a base class. An
adapter subclasses it with what is specific to its suite (which scene to build for a task, the arms' layout, the
labels, success); this class does the contract's side once:

- action: per arm, in RobotSpec.arms order, 7 joint-position targets (rad) then a hand command in [0, 1] (0 closed,
  1 open). robosuite's JOINT_POSITION controller takes the targets as absolute goals (`robosuite15.make`); the hand
  command is a target opening that the gripper's set-point approaches at the gripper's own speed (`control`). Other
  robot parts (a mobile base, a torso) get zero commands: they stay where they are. Arms without fingers ignore the
  hand command;
- observation: `q.<arm>`, `hand.<arm>` (measured opening in command units), RGB-D per camera (upright images, depth
  in metres), `<cam>.seg` and `extra["segments"]` with `segmentation: true`, intrinsics and OpenCV camera poses in
  `extra["cameras"]`, object poses in `extra["privileged"]` with `privileged_obs: true`;
- world frame: the body `world_body` (robot0's base): every pose is relative to it; `extra[<suite>]["base_in_sim"]`
  maps it into the simulator's world;
- reset: `start_episode` (reseed and reset), the layout checked once against what the RobotSpec published, the arms
  moved to `ready_posture()` if the suite gives one, then `settle_steps` steps holding the joints with the hands open;
- step: `success()` ends the episode, and so does `failed()` (a suite's own failure condition).

The import is pure (numpy, hylo-core): simulator code runs only inside methods, in the simulator's venv.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from hylo.core.task import TaskInstance
from hylo.core.types import (Action, ActionSemantics, BoxSpace, EnvCapabilities, EnvSpec, Observation,
                             ObservationSpec, StepResult)

from .control import clip_joints, hand_opening, split_action, step_gripper_setpoint
from .labels import InstanceLabels, LabelRules
from .panda import LOWER, UPPER, ArmFacts, PandaNames, robot_spec


@dataclass(frozen=True)
class ArmLayout:
    """One arm as the contract shows it: its name, robosuite names, base pose in the contract's world frame and the
    simulator body that pose is checked against (default: robosuite's `robot<i>_base`; a mobile manipulator's arm
    starts at `robot<i>_link0`)."""

    name: str
    names: PandaNames
    base_in_world: list[list[float]]
    base_body: str | None = None


IDENTITY = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


class RobosuiteContractEnv:
    """Base class. Subclasses implement the hooks marked "suite" below; the rest is shared."""

    env_name = "robosuite"                    # EnvSpec.name
    env_version = "0.1"
    control_hz = 20.0
    world_body = "robot0_base"
    gap_closed, gap_open = 0.0015, 0.0785      # Panda finger travel sum where robosuite 1.5's servos settle

    def __init__(self, image_size: int = 256, cameras: Sequence[str] = (), segmentation: bool = False,
                 privileged_obs: bool = False, kp: float = 300.0, damping_ratio: float = 1.0,
                 settle_steps: int = 10) -> None:
        self.image_size, self.segmentation, self.privileged_obs = int(image_size), bool(segmentation), bool(privileged_obs)
        self.kp, self.damping_ratio, self.settle_steps = float(kp), float(damping_ratio), int(settle_steps)
        self.cameras = tuple(cameras)
        known = self.camera_map()
        unknown = [c for c in self.cameras if c not in known]
        if unknown or not self.cameras:
            raise ValueError(f"unknown cameras {unknown or '(none given)'}; available: {sorted(known)}")
        self._env: Any = None
        self._robot_spec = None
        self._checked = False
        self._labels: InstanceLabels | None = None
        self.body_names: list[str] = []          # of the scene just built (after a reset)
        self.geom_names: list[str] = []

    # ------------------------------------------------------------------ suite hooks (pure)
    def arm_layout(self) -> list[ArmLayout]:
        """suite: the arms, in action order, with their bases in the contract's world frame."""
        raise NotImplementedError

    def camera_map(self) -> dict[str, tuple[str, str | None]]:
        """suite: contract camera name -> (simulator camera name, arm it is mounted on or None)."""
        raise NotImplementedError

    def table_height(self) -> float | None:
        """suite: the support's height in the contract's world frame (a hint for the tools), or None."""
        return None

    def notes(self, hand: str) -> str:
        """suite: RobotSpec.notes (`hand`: the shared sentence about the hand)."""
        return hand

    # ------------------------------------------------------------------ suite hooks (simulator side)
    def build(self) -> Any:
        """suite: construct the robosuite environment (worker side)."""
        raise NotImplementedError

    def start_episode(self, task: TaskInstance) -> None:
        """suite: put the environment in the episode's initial state (default: reseed, reset)."""
        from .robosuite15 import reseed

        reseed(self._env, task.seed)
        self._env.reset()

    def label_rules(self) -> LabelRules:
        """suite: segmentation rules for the scene just built."""
        return LabelRules()

    def instruction(self) -> str:
        """suite: the episode's instruction."""
        raise NotImplementedError

    def success(self) -> bool:
        return bool(self._env._check_success())

    def failed(self) -> bool:
        """suite: a failure that ends the episode (default: none)."""
        return False

    def objects(self) -> list[tuple[str, str, str]]:
        """suite: (key, root body, label) of the objects reported in extra["privileged"] (default: robosuite's
        mujoco_objects labelled by the label rules' `objects`)."""
        rules = self.label_rules()
        return [(o.name, o.root_body, rules.objects.get(o.name, o.name)) for o in self._env.model.mujoco_objects]

    def measured_table_height(self) -> float | None:
        """suite: the support height measured in the built scene (contract frame), checked against table_height()."""
        return self.table_height()

    def panda_model(self, names: PandaNames) -> dict[str, Any]:
        """The arm's exported model and hand geometry (default: robosuite 1.5's Panda)."""
        from .robosuite15 import panda_model

        return panda_model(names)

    def extra(self) -> dict[str, Any]:
        """suite: more of extra[env_name] in every observation (default: nothing)."""
        return {}

    ready_steps = 40

    def ready_posture(self) -> list[Sequence[float]] | None:
        """suite: joint positions per arm to move to after the reset, in `ready_steps` control steps (default: None,
        keep the simulator's start posture)."""
        return None

    # ------------------------------------------------------------------ spec
    @property
    def arms(self) -> list[ArmLayout]:
        return self.arm_layout()

    def spec(self) -> EnvSpec:
        s = self.image_size
        state, units, low, high, labels = {}, {}, [], [], []
        for a in self.arms:
            state[f"q.{a.name}"], state[f"hand.{a.name}"] = (7,), (1,)
            units[f"q.{a.name}"], units[f"hand.{a.name}"] = "rad", "command (0 closed, 1 open)"
            low += [*LOWER, 0.0]
            high += [*UPPER, 1.0]
            labels += [*(f"{a.name}.joint{k}" for k in range(1, 8)), f"{a.name}.hand"]
        return EnvSpec(
            name=self.env_name, version=self.env_version,
            observation=ObservationSpec(state=state, cameras={c: (s, s, 3) for c in self.cameras}, units=units),
            action=BoxSpace(shape=(len(low),), low=tuple(low), high=tuple(high),
                            semantics=ActionSemantics(control_mode="joint_pos", frame="base", gripper="0_close_1_open",
                                                      dim_labels=tuple(labels))),
            control_hz=self.control_hz,
            capabilities=EnvCapabilities(seedable=True, render=True, privileged_state=True),
            robot=self.robot_spec())

    def robot_spec(self):
        """The RobotSpec from the layout and the exported models (cheap after the first export: no scene is built)."""
        if self._robot_spec is None:
            from .panda import hand_note

            facts, geo = [], {}
            for a in self.arms:
                info = self.panda_model(a.names)
                geo = info["geometry"]
                facts.append(ArmFacts(name=a.name, names=a.names, model=info["model"], base_in_world=a.base_in_world,
                                      tcp_in_ee=geo["tcp_in_ee"], max_width=geo["max_width"]))
            cmap = self.camera_map()
            cams = [{"name": c, "height": self.image_size, "width": self.image_size, "mounted_on": cmap[c][1]}
                    for c in self.cameras]
            hand = hand_note(self.arms[0].names.gripper, geo.get("max_width", 0.0), geo.get("tcp_offset", 0.0))
            self._robot_spec = robot_spec(facts, cams, self.table_height(), self.notes(hand))
        return self._robot_spec

    # ------------------------------------------------------------------ episode
    def reset(self, task: TaskInstance) -> Observation:
        if self._env is None:
            self._env = self.build()
        self.start_episode(task)
        self._bind()
        if not self._checked:
            self.check_layout()
            self._checked = True
        ready = self.ready_posture()
        if ready is not None:                      # glide each arm to the ready posture, hands open
            start = [self._q(i) for i in range(len(self.arms))]
            n = max(1, self.ready_steps)
            for k in range(1, n + 1):
                f = 0.5 - 0.5 * np.cos(np.pi * k / n)
                self._env.step(self.sim_action(np.concatenate(
                    [np.r_[q0 + f * (np.asarray(r, float) - q0), 1.0] for q0, r in zip(start, ready)])))
        hold = np.concatenate([np.r_[self._q(i), 1.0] for i in range(len(self.arms))])
        for _ in range(self.settle_steps):
            self._env.step(self.sim_action(hold))
        return self._observe()

    def _bind(self) -> None:
        """Indices into the model just built (robosuite builds a new one on every hard reset)."""
        from .labels import label_geoms
        from .render import no_reflections

        e = self._env
        no_reflections(e.sim)
        m = e.sim.model._model
        self._robots = [e.robots[0] if len(e.robots) == 1 else e.robots[i] for i in range(len(self.arms))]
        self._qidx = [np.asarray(r._ref_joint_pos_indexes) for r in self._robots]
        self._gidx = [np.asarray((r._ref_gripper_joint_pos_indexes or {}).get("right") or [], dtype=int)
                      for r in self._robots]
        self._world = m.body(self.world_body).id
        import mujoco

        self.body_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)]
        self.geom_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
        if self.segmentation:
            self._labels = label_geoms(self.body_names, m.body_parentid, self.geom_names, m.geom_bodyid,
                                       self.label_rules())

    def check_layout(self) -> None:
        """The scene matches what the RobotSpec published: each arm's base and the support height."""
        T_inv = np.linalg.inv(self.world_pose())
        for a in self.arms:
            got = T_inv @ self._body_pose(a.base_body or a.names.base)
            if not np.allclose(got, a.base_in_world, atol=1e-3):
                raise RuntimeError(f"{a.name}'s base is at {np.round(got, 4).tolist()}, not where the layout says "
                                   f"({a.base_in_world}): the simulator changed its scene layout")
        want, got = self.table_height(), self.measured_table_height()
        if (want is None) != (got is None) or (want is not None and abs(want - got) > 0.002):
            raise RuntimeError(f"support height {got} in the contract's frame, the layout says {want}")

    def _q(self, i: int) -> np.ndarray:
        return np.asarray(self._env.sim.data._data.qpos[self._qidx[i]], dtype=np.float64).copy()

    def hand_state(self, i: int) -> float:
        """Arm i's measured opening in command units; arms without fingers report 1."""
        if len(self._gidx[i]) != 2:
            return 1.0
        return hand_opening(self._env.sim.data._data.qpos[self._gidx[i]], self.gap_closed, self.gap_open)

    def _body_pose(self, name: str) -> np.ndarray:
        from .geometry import body_pose

        return body_pose(self._env.sim.data._data, self._env.sim.model._model.body(name).id)

    def world_pose(self) -> np.ndarray:
        """The contract's world frame (`world_body`) in the simulator's world frame (4x4)."""
        from .geometry import body_pose

        return body_pose(self._env.sim.data._data, self._world)

    def _observe(self) -> Observation:
        from .render import render_instances, render_rgbd

        sim = self._env.sim
        T_world = self.world_pose()
        T_inv = np.linalg.inv(T_world)
        cmap = self.camera_map()
        images: dict[str, np.ndarray] = {}
        cams: dict[str, Any] = {}
        for c in self.cameras:
            rgb, depth, meta = render_rgbd(sim, cmap[c][0], self.image_size, T_inv)
            images[c], images[f"{c}.depth"], cams[c] = rgb, depth, meta
            if self._labels is not None:
                images[f"{c}.seg"] = render_instances(sim, cmap[c][0], self.image_size, self._labels.geom)
        extra: dict[str, Any] = {"cameras": cams, self.env_name: {"base_in_sim": T_world.tolist(), **self.extra()}}
        if self._labels is not None:
            extra["segments"] = self._labels.segments()
        if self.privileged_obs:
            extra["privileged"] = self._objects_state(T_inv)
        state = {}
        for i, a in enumerate(self.arms):
            state[f"q.{a.name}"] = self._q(i)
            state[f"hand.{a.name}"] = np.array([self.hand_state(i)])
        return Observation(state=state, images=images, instruction=self.instruction(),
                           time=float(sim.data._data.time), extra=extra)

    def _objects_state(self, T_inv: np.ndarray) -> dict[str, Any]:
        """Object root poses in the contract's world frame with labels and articulations (hinge and slide joints of
        the object, by name), and each hand's frame."""
        from .geometry import body_pose

        m, d = self._env.sim.model._model, self._env.sim.data._data
        out: dict[str, Any] = {}
        for key, root, label in self.objects():
            b = m.body(root).id
            T = T_inv @ body_pose(d, b)
            item: dict[str, Any] = {"label": label, "pos": T[:3, 3].tolist(), "R": T[:3, :3].tolist()}
            prefix = key + "_"
            joints = {m.joint(j).name[len(prefix):]: float(d.qpos[m.jnt_qposadr[j]]) for j in range(m.njnt)
                      if m.joint(j).name.startswith(prefix) and int(m.jnt_type[j]) in (2, 3)}
            if joints:
                item["joints"] = joints
            out[key] = item
        ee = {a.name: (T_inv @ body_pose(d, m.body(a.names.ee_body).id)).tolist() for a in self.arms}
        return {"objects": out, "ee": ee}

    def step(self, action: Action) -> StepResult:
        _, reward, _, _ = self._env.step(self.sim_action(np.asarray(action.data, dtype=np.float64)))
        success = self.success()
        failed = bool(not success and self.failed())
        return StepResult(observation=self._observe(), reward=float(reward), terminated=success or failed,
                          truncated=False, success=success, info={"failure": True} if failed else {})

    def sim_action(self, a: np.ndarray) -> np.ndarray:
        """Contract action -> robosuite's: per robot, the arm part gets the clipped absolute targets, the gripper part
        0 (keep the set-point, which moves towards the commanded opening here), any other part 0 (stay)."""
        out = []
        for (q, hand), r in zip(split_action(a, len(self.arms)), self._robots):
            vec = np.zeros(r.action_dim)
            idx = r.composite_controller._action_split_indexes
            lo, hi = idx["right"]
            vec[lo:hi] = clip_joints(q, LOWER, UPPER)
            g = r.gripper["right"]
            if int(getattr(g, "dof", 0)):
                step_gripper_setpoint(g, hand)
            out.append(vec)
        return np.concatenate(out)

    def hold_action(self, last: np.ndarray | None) -> np.ndarray:
        """Keep the last target, else the current joints with each hand as it is (open when more than half)."""
        if last is not None:
            return np.asarray(last, dtype=float).copy()
        return np.concatenate([np.r_[self._q(i), 1.0 if self.hand_state(i) > 0.5 else 0.0]
                               for i in range(len(self.arms))])

    def privileged_state(self) -> dict:
        st = self._objects_state(np.linalg.inv(self.world_pose()))
        return {"objects": st["objects"], "ee": st["ee"], "success": self.success()}

    def render(self, camera: str = "", width: int = 256, height: int = 256) -> np.ndarray:
        name = self.camera_map().get(camera, (camera or None, None))[0] if camera else self.camera_map()[self.cameras[0]][0]
        img = self._env.sim.render(camera_name=name, width=width, height=height)
        return np.ascontiguousarray(img[::-1])

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


__all__ = ["ArmLayout", "IDENTITY", "RobosuiteContractEnv"]
