"""LIBERO Environment adapter. Runs inside the simulator venv (hf-libero, robosuite 1.4) behind an
EnvServer; the main process never imports LIBERO.

Conventions match LeRobot's LIBERO integration so LeRobot-trained policies see what they saw in training:
- images: agentview and wrist cameras, rotated 180 degrees (robosuite renders them upside down);
- state: eef position (3), eef axis-angle from the xyzw quaternion (3), gripper qpos (2);
- reset: load the task's stored init state (index = seed % 50), then 10 no-op steps;
- action: 7-D OSC delta pose in [-1, 1], last dim gripper (-1 open, +1 close);
- success: LIBERO's BDDL goal predicate (env.check_success()); the episode ends on success.

The robot contract (RFC-0006, hylo.core.robot) for tool-using agents: params `control: joint_position` and
`tool_obs: true` (the defaults keep the VLA mode above, unchanged).
- action: 7 joint-position targets (rad) + hand command in [0, 1] (0 closed, 1 open), through robosuite's
  JOINT_POSITION controller (delta to the target each step, unscaled; gains `kp`, `damping_ratio`); the hand command
  is a target opening, approached at robosuite's gripper speed (full stroke in 8 steps);
- observation: `q.arm`, `hand.arm` (measured opening in command units), RGB-D per camera (`agentview`, `wrist`;
  upright images, depth in metres), `<cam>.seg` and `extra["segments"]` with `segmentation: true`, intrinsics and
  OpenCV-frame camera poses in `extra["cameras"]`, object poses in `extra["privileged"]` with `privileged_obs: true`;
- `EnvSpec.robot`: the Panda exported from robosuite into the asset cache (see robot.py). The contract's world frame
  is the robot base frame (LIBERO scenes place the robot at different heights).

Perturbed tasks (perturb.py): `perturb` (env param, applied to every task) and `TaskSpec.params["perturb"]` (from the
benchmark's `perturb` param) name the changes: position, task, language, object, camera, noise, robot_init. What was
changed is reported in `extra["perturbation"]` of every observation.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np

from hylo.core.task import TaskInstance
from hylo.core.types import (Action, ActionSemantics, BoxSpace, EnvCapabilities, EnvSpec, Observation,
                                     ObservationSpec, StepResult)

NOOP = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
CONTROLS = ("osc_delta", "joint_position")


def quat2axisangle(q: np.ndarray) -> np.ndarray:
    """xyzw quaternion -> axis-angle (same formula as robosuite / LeRobot's LiberoProcessorStep)."""
    w = float(np.clip(q[3], -1.0, 1.0))
    den = np.sqrt(1.0 - w * w)
    if den < 1e-10:
        return np.zeros(3)
    return np.asarray(q[:3], dtype=np.float64) * 2.0 * np.arccos(w) / den


class LiberoEnv:
    """Params: image_size (square cameras), num_steps_wait (no-op steps after loading the init state); robot contract:
    control ("osc_delta": the VLA mode; "joint_position"), tool_obs (the contract's observation layout and
    EnvSpec.robot; needs joint_position), cameras (subset of agentview, wrist), segmentation (instance ids and labels;
    privileged), privileged_obs (object poses in extra["privileged"]), kp and damping_ratio (joint controller)."""

    def __init__(self, image_size: int = 256, num_steps_wait: int = 10, control: str = "osc_delta",
                 tool_obs: bool = False, cameras: Sequence[str] = ("agentview", "wrist"), segmentation: bool = False,
                 privileged_obs: bool = False, kp: float = 300.0, damping_ratio: float = 1.0,
                 perturb: str | Sequence[str] | None = None, noise_sigma: float = 10.0) -> None:
        self.image_size, self.num_steps_wait = image_size, num_steps_wait
        if control not in CONTROLS:
            raise ValueError(f"control must be one of {CONTROLS}, got {control!r}")
        if tool_obs and control != "joint_position":
            raise ValueError("tool_obs (the robot contract) needs control: joint_position")
        from .robot import CAMERAS

        unknown = [c for c in cameras if c not in CAMERAS]
        if unknown:
            raise ValueError(f"unknown cameras {unknown}; available: {list(CAMERAS)}")
        self.control, self.tool_obs, self.cameras = control, bool(tool_obs), tuple(cameras)
        self.segmentation, self.privileged_obs = bool(segmentation), bool(privileged_obs)
        self.kp, self.damping_ratio = float(kp), float(damping_ratio)
        self._env = None
        self._key: tuple[str, int] | None = None
        self._task: TaskInstance | None = None
        self._robot_spec = None
        from .perturb import kinds

        self.perturb, self.noise_sigma = kinds(perturb), float(noise_sigma)
        self._kinds: tuple[str, ...] = ()
        self._perturbation: dict[str, Any] = {}
        self._saved_rgba: dict[int, np.ndarray] = {}
        self._saved_cams: dict[str, Any] = {}
        self._step_i = 0

    @property
    def joint_mode(self) -> bool:
        return self.control == "joint_position"

    def spec(self) -> EnvSpec:
        s = self.image_size
        if self.joint_mode:
            return self._joint_spec()
        return EnvSpec(
            name="libero", version="0.1",
            observation=ObservationSpec(
                state={"eef_pos": (3,), "eef_axis_angle": (3,), "gripper_qpos": (2,), "eef_quat": (4,), "joint_pos": (7,)},
                cameras={"agentview": (s, s, 3), "wrist": (s, s, 3)},
                units={"eef_pos": "m", "eef_axis_angle": "rad", "gripper_qpos": "m", "joint_pos": "rad"}),
            action=BoxSpace(shape=(7,), low=(-1.0,) * 7, high=(1.0,) * 7,
                            semantics=ActionSemantics(control_mode="eef_delta_pose", frame="base", rotation_repr="axis_angle",
                                                      gripper="binary_close_positive",
                                                      dim_labels=("dx", "dy", "dz", "drx", "dry", "drz", "gripper"))),
            control_hz=20.0,
            # No workspace limit yet: LIBERO scenes use different world frames (eef z ranges from ~0.01 m to ~1.35 m
            # across suites), so a single box would block legitimate motion. Calibrate per scene before enabling.
            capabilities=EnvCapabilities(seedable=True, render=True, privileged_state=True))

    # -- robot contract ---------------------------------------------------------------------------------------------
    def robot_spec(self):
        """The RobotSpec (exports the Panda model into the asset cache on first use; simulator venv only)."""
        if self._robot_spec is None:
            from .robot import robot_spec
            from .sim import export_panda_model, hand_geometry

            path = export_panda_model()
            self._robot_spec = robot_spec(str(path), hand_geometry(path), self.cameras, self.image_size,
                                          self.image_size, 20.0, segmentation=self.segmentation)
        return self._robot_spec

    def _joint_spec(self) -> EnvSpec:
        from .robot import JOINTS, LOWER, UPPER

        s = self.image_size
        if self.tool_obs:
            obs = ObservationSpec(state={"q.arm": (7,), "hand.arm": (1,)}, cameras={c: (s, s, 3) for c in self.cameras},
                                  units={"q.arm": "rad", "hand.arm": "command (0 closed, 1 open)"})
        else:
            obs = ObservationSpec(
                state={"eef_pos": (3,), "eef_axis_angle": (3,), "gripper_qpos": (2,), "eef_quat": (4,), "joint_pos": (7,)},
                cameras={"agentview": (s, s, 3), "wrist": (s, s, 3)},
                units={"eef_pos": "m", "eef_axis_angle": "rad", "gripper_qpos": "m", "joint_pos": "rad"})
        return EnvSpec(
            name="libero", version="0.1", observation=obs,
            action=BoxSpace(shape=(8,), low=(*LOWER, 0.0), high=(*UPPER, 1.0),
                            semantics=ActionSemantics(control_mode="joint_pos", frame="base", gripper="0_close_1_open",
                                                      dim_labels=(*(j.removeprefix("robot0_") for j in JOINTS), "hand"))),
            control_hz=20.0,
            capabilities=EnvCapabilities(seedable=True, render=True, privileged_state=True),
            robot=self.robot_spec() if self.tool_obs else None)

    # -- episode ----------------------------------------------------------------------------------------------------
    def _ensure(self, task: TaskInstance) -> None:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        p = task.spec.params
        variant = "task" in self._task_kinds(task)
        key = (p["suite"], int(p["task_index"]), variant)
        if self._key == key:
            return
        self.close()
        suite = benchmark.get_benchmark_dict()[key[0]]()
        lt = suite.get_task(key[1])
        bddl = os.path.join(get_libero_path("bddl_files"), lt.problem_folder, lt.bddl_file)
        if variant:
            from .perturb import variant_bddl

            bddl = str(variant_bddl(key[0], key[1]))
        if not self.joint_mode:
            self._env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=self.image_size,
                                           camera_widths=self.image_size)
        else:
            # tool_obs renders its own RGB-D (upright, with poses); the VLA observation needs robosuite's camera obs
            self._env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=self.image_size,
                                           camera_widths=self.image_size, controller="JOINT_POSITION",
                                           use_camera_obs=not self.tool_obs)
            # absolute targets: the controller's delta passes unscaled (+-1 rad per step at most); robots are rebuilt
            # from robot_configs on every hard reset
            gains = {"input_max": 1, "input_min": -1, "output_max": 1.0, "output_min": -1.0, "kp": self.kp,
                     "damping_ratio": self.damping_ratio, "interpolation": None}
            for cfg in self._env.env.robot_configs:
                cfg["controller_config"].update(gains)
            for r in self._env.env.robots:
                r.controller_config.update(gains)
        self._init_states = suite.get_task_init_states(key[1])
        self._base_language = lt.language
        if variant:
            import re

            m = re.search(r"\(:language\s+([^)]*)\)", open(bddl).read())
            self._base_language = m.group(1).strip() if m else lt.language
        self._language = self._base_language
        self._key = key

    def _task_kinds(self, task: TaskInstance) -> tuple[str, ...]:
        from .perturb import kinds

        return tuple(dict.fromkeys((*self.perturb, *kinds(task.spec.params.get("perturb")))))

    def reset(self, task: TaskInstance) -> Observation:
        self._ensure(task)
        self._task = task
        self._kinds = self._task_kinds(task)
        if self._env is not None:
            from .perturb import restore_camera, restore_colours

            restore_colours(self._env, self._saved_rgba)
            restore_camera(self._env, self._saved_cams)
        self._env.seed(task.seed)
        self._env.reset()
        raw = self._env.set_init_state(self._init_states[task.seed % len(self._init_states)])
        self._perturbation = self._apply_perturbations(task) if self._kinds else {}
        self._step_i = 0
        if self.joint_mode:
            self._bind()
            hold = np.concatenate([self._q(), [1.0]])     # hold the joints, open the gripper
        for _ in range(self.num_steps_wait):
            raw, _, _, _ = self._env.step(self.joint_action(hold) if self.joint_mode else NOOP)
        self._raw = raw
        return self._observe(raw)

    def _apply_perturbations(self, task: TaskInstance) -> dict[str, Any]:
        """Change the scene just after its init state was loaded (before the settling steps)."""
        from . import perturb as P

        e = self._env.env
        info: dict[str, Any] = {"kinds": list(self._kinds)}
        movable = [n for n in e.objects_dict]
        row = P.variants().get(self._key[0], {}).get(str(self._key[1]), {})
        moved = (row.get("task") or {}).get("object") if "task" in self._kinds else None
        moved = moved or next(iter(row.get("moved") or []), None) or next(
            (n for n in movable if n in (task.spec.success.expr if task.spec.success else "")), None)
        if "task" in self._kinds:
            info["task"] = {"instruction": self._base_language}
        self._language = self._base_language
        if "language" in self._kinds:
            self._language = P.paraphrase(self._base_language, task.seed)
            info["language"] = self._language
        if "position" in self._kinds and moved:
            info.update(P.swap_positions(self._env, moved, movable, task.seed) or {"position": "no partner"})
        if "robot_init" in self._kinds:
            from .robot import LOWER, UPPER

            r = e.robots[0]
            idx = np.asarray(r._ref_joint_pos_indexes)
            d = e.sim.data._data
            d.qpos[idx] = P.jitter_joints(d.qpos[idx], task.seed, LOWER, UPPER)
            e.sim.forward()
            info["robot_init"] = np.round(d.qpos[idx], 3).tolist()
        if "object" in self._kinds:
            info.update(P.recolour(self._env, movable, task.seed, self._saved_rgba))
        if "camera" in self._kinds:
            info.update(P.move_camera(self._env, "agentview", task.seed, self._saved_cams))
        if "noise" in self._kinds:
            info["noise"] = {"sigma": self.noise_sigma}
        return info

    def _rgb(self, img: np.ndarray) -> np.ndarray:
        if "noise" not in self._kinds or self._task is None:
            return img
        from .perturb import noisy

        return noisy(img, int(self._task.seed), self._step_i, self.noise_sigma)

    def _bind(self) -> None:
        """Indices into the (rebuilt) model for the joint mode and the contract's observations."""
        from .robot import CAMERAS
        from .sim import Segments

        e = self._env.env
        m = e.sim.model._model
        r = e.robots[0]
        self._qidx = np.asarray(r._ref_joint_pos_indexes)
        self._gidx = np.asarray(r._ref_gripper_joint_pos_indexes)
        self._base = m.body("robot0_base").id
        self._cam_ids = {c: m.camera(CAMERAS[c]).id for c in self.cameras}
        self._objects = list(e.objects_dict) + list(e.fixtures_dict)
        self._segs = Segments(m, self._objects) if self.segmentation else None

    def _q(self) -> np.ndarray:
        return np.asarray(self._env.env.sim.data._data.qpos[self._qidx], dtype=np.float64).copy()

    def hand_state(self) -> float:
        """Measured opening in command units: the fingers' travel, from where the servos settle closed (0) to where
        they settle open (1)."""
        from .robot import GAP_CLOSED, GAP_OPEN

        g = self._env.env.sim.data._data.qpos[self._gidx]
        return float(np.clip((g[0] - g[1] - GAP_CLOSED) / (GAP_OPEN - GAP_CLOSED), 0.0, 1.0))

    def base_pose(self) -> np.ndarray:
        """The robot base in LIBERO's world frame (4x4)."""
        from .robot import pose

        d = self._env.env.sim.data._data
        return pose(d.xmat[self._base].reshape(3, 3), d.xpos[self._base])

    def _observe(self, raw: dict) -> Observation:
        if self.tool_obs:
            return self._tool_observe()
        quat = np.asarray(raw["robot0_eef_quat"], dtype=np.float64)
        return Observation(
            state={"eef_pos": np.asarray(raw["robot0_eef_pos"], dtype=np.float64), "eef_axis_angle": quat2axisangle(quat),
                   "gripper_qpos": np.asarray(raw["robot0_gripper_qpos"], dtype=np.float64), "eef_quat": quat,
                   "joint_pos": np.asarray(raw["robot0_joint_pos"], dtype=np.float64)},
            images={"agentview": self._rgb(np.ascontiguousarray(raw["agentview_image"][::-1, ::-1])),
                    "wrist": self._rgb(np.ascontiguousarray(raw["robot0_eye_in_hand_image"][::-1, ::-1]))},
            instruction=self._language, time=float(self._env.sim.data.time),
            extra={"perturbation": self._perturbation} if self._perturbation else {})

    def _tool_observe(self) -> Observation:
        from .robot import CAMERAS, camera_pose, depth_metres, intrinsics

        sim = self._env.env.sim
        m, d = sim.model._model, sim.data._data
        s = self.image_size
        T_base = self.base_pose()
        T_inv = np.linalg.inv(T_base)
        images: dict[str, np.ndarray] = {}
        cams: dict[str, Any] = {}
        for c in self.cameras:
            name, cid = CAMERAS[c], self._cam_ids[c]
            rgb, z = sim.render(camera_name=name, width=s, height=s, depth=True)
            images[c] = self._rgb(np.ascontiguousarray(rgb[::-1]))     # OpenGL rows bottom-up -> upright
            images[f"{c}.depth"] = depth_metres(m, np.ascontiguousarray(z[::-1]))
            if self._segs is not None:
                images[f"{c}.seg"] = np.ascontiguousarray(self._segs.render(sim, cid, s, s)[::-1])
            cams[c] = {"K": intrinsics(float(m.cam_fovy[cid]), s, s),
                       "T_world_cam": camera_pose(m, d, cid, T_inv).tolist()}
        extra: dict[str, Any] = {"cameras": cams,
                                 "libero": {"suite": self._key[0], "task_index": self._key[1],
                                            "base_in_sim": T_base.tolist()}}
        if self._segs is not None:
            extra["segments"] = {str(k): v for k, v in self._segs.labels.items()}
        if self._perturbation:
            extra["perturbation"] = self._perturbation
        if self.privileged_obs:
            extra["privileged"] = self._objects_state(T_inv)
        return Observation(state={"q.arm": self._q(), "hand.arm": np.array([self.hand_state()])}, images=images,
                           instruction=self._language, time=float(d.time), extra=extra)

    def _objects_state(self, T_inv: np.ndarray) -> dict[str, Any]:
        """Object root poses in the contract's world frame (the robot base), with their labels."""
        from .robot import object_label

        e = self._env.env
        m, d = e.sim.model._model, e.sim.data._data
        out = {}
        for name in self._objects:
            o = e.objects_dict.get(name) or e.fixtures_dict.get(name)
            b = m.body(o.root_body).id
            p = T_inv[:3, :3] @ d.xpos[b] + T_inv[:3, 3]
            R = T_inv[:3, :3] @ d.xmat[b].reshape(3, 3)
            out[name] = {"label": object_label(name), "pos": p.tolist(), "R": R.tolist()}
            joints = {}
            for j in range(m.njnt):                     # articulations (drawers, knobs, doors): slide / hinge
                jn = m.joint(j).name
                if jn.startswith(name + "_") and int(m.jnt_type[j]) in (2, 3):
                    joints[jn[len(name) + 1:]] = float(d.qpos[m.jnt_qposadr[j]])
            if joints:
                out[name]["joints"] = joints
        from .robot import EE_BODY, pose

        ee = m.body(EE_BODY).id
        return {"objects": out, "ee": (T_inv @ pose(d.xmat[ee].reshape(3, 3), d.xpos[ee])).tolist()}

    def step(self, action: Action) -> StepResult:
        if self.joint_mode:
            a = np.asarray(action.data, dtype=np.float64).reshape(-1)
            raw, reward, done, info = self._env.step(self.joint_action(a))
        else:
            raw, reward, done, info = self._env.step(np.asarray(action.data, dtype=np.float64).tolist())
        self._raw = raw
        self._step_i += 1
        success = bool(self._env.check_success())
        return StepResult(observation=self._observe(raw), reward=float(reward), terminated=bool(done or success),
                          truncated=False, success=success, info={})

    def joint_action(self, a: np.ndarray) -> list[float]:
        """Joint targets + hand command -> robosuite's JOINT_POSITION action (delta to the target, unscaled). The hand
        command is a target opening: robosuite's gripper keeps a finger set-point (`current_action`, -1 closed .. +1
        open per finger) that its own actions only push towards open or closed; here the set-point moves towards the
        commanded opening at robosuite's gripper speed (0.25 per control step, the full stroke in 8 steps) and the
        gripper action is 0 (keep it)."""
        from .robot import LOWER, UPPER

        delta = np.clip(np.clip(a[:7], LOWER, UPPER) - self._q(), -1.0, 1.0)
        c = float(np.clip(a[7], 0.0, 1.0))
        g = self._env.env.robots[0].gripper
        want = np.array([2.0 * c - 1.0, 1.0 - 2.0 * c])
        cur = np.broadcast_to(np.asarray(g.current_action, dtype=np.float64), (2,))
        g.current_action = cur + np.clip(want - cur, -0.25, 0.25)
        return [*delta.tolist(), 0.0]

    def hold_action(self, last: np.ndarray | None) -> np.ndarray:
        """Joint mode: keep the last target, else the current joints with the hand as it is (open when > half)."""
        if last is not None:
            return np.asarray(last, dtype=float).copy()
        return np.concatenate([self._q(), [1.0 if self.hand_state() > 0.5 else 0.0]])

    def privileged_state(self) -> dict:
        out = {"object_state": np.asarray(self._raw.get("object-state", []), dtype=np.float64),
               "eef_pos": np.asarray(self._raw["robot0_eef_pos"], dtype=np.float64)}
        if self.tool_obs:
            out["objects"] = self._objects_state(np.linalg.inv(self.base_pose()))["objects"]
        return out

    def render(self, camera: str = "agentview", width: int = 256, height: int = 256) -> np.ndarray:
        name = {"agentview": "agentview", "wrist": "robot0_eye_in_hand"}.get(camera, camera)
        img = self._env.sim.render(camera_name=name, width=width, height=height)
        return img[::-1] if self.tool_obs else img[::-1, ::-1]

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env, self._key = None, None
            self._saved_rgba.clear()
            self._saved_cams.clear()
