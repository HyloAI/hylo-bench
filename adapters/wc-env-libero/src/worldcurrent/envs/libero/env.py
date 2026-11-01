"""LIBERO Environment adapter. Runs inside the simulator venv (hf-libero, robosuite 1.4) behind an
EnvServer; the main process never imports LIBERO.

Conventions match LeRobot's LIBERO integration so LeRobot-trained policies see what they saw in training:
- images: agentview and wrist cameras, rotated 180 degrees (robosuite renders them upside down);
- state: eef position (3), eef axis-angle from the xyzw quaternion (3), gripper qpos (2);
- reset: load the task's stored init state (index = seed % 50), then 10 no-op steps;
- action: 7-D OSC delta pose in [-1, 1], last dim gripper (-1 open, +1 close);
- success: LIBERO's BDDL goal predicate (env.check_success()); the episode ends on success.
"""

from __future__ import annotations

import os

import numpy as np

from worldcurrent.core.task import TaskInstance
from worldcurrent.core.types import (Action, ActionSemantics, BoxSpace, EnvCapabilities, EnvSpec, Observation,
                                     ObservationSpec, StepResult)

NOOP = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def quat2axisangle(q: np.ndarray) -> np.ndarray:
    """xyzw quaternion -> axis-angle (same formula as robosuite / LeRobot's LiberoProcessorStep)."""
    w = float(np.clip(q[3], -1.0, 1.0))
    den = np.sqrt(1.0 - w * w)
    if den < 1e-10:
        return np.zeros(3)
    return np.asarray(q[:3], dtype=np.float64) * 2.0 * np.arccos(w) / den


class LiberoEnv:
    def __init__(self, image_size: int = 256, num_steps_wait: int = 10) -> None:
        self.image_size, self.num_steps_wait = image_size, num_steps_wait
        self._env = None
        self._key: tuple[str, int] | None = None
        self._task: TaskInstance | None = None

    def spec(self) -> EnvSpec:
        s = self.image_size
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

    def _ensure(self, task: TaskInstance) -> None:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        p = task.spec.params
        key = (p["suite"], int(p["task_index"]))
        if self._key == key:
            return
        self.close()
        suite = benchmark.get_benchmark_dict()[key[0]]()
        lt = suite.get_task(key[1])
        self._env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), lt.problem_folder, lt.bddl_file),
                                       camera_heights=self.image_size, camera_widths=self.image_size)
        self._init_states = suite.get_task_init_states(key[1])
        self._language = lt.language
        self._key = key

    def reset(self, task: TaskInstance) -> Observation:
        self._ensure(task)
        self._task = task
        self._env.seed(task.seed)
        self._env.reset()
        raw = self._env.set_init_state(self._init_states[task.seed % len(self._init_states)])
        for _ in range(self.num_steps_wait):
            raw, _, _, _ = self._env.step(NOOP)
        self._raw = raw
        return self._observe(raw)

    def _observe(self, raw: dict) -> Observation:
        quat = np.asarray(raw["robot0_eef_quat"], dtype=np.float64)
        return Observation(
            state={"eef_pos": np.asarray(raw["robot0_eef_pos"], dtype=np.float64), "eef_axis_angle": quat2axisangle(quat),
                   "gripper_qpos": np.asarray(raw["robot0_gripper_qpos"], dtype=np.float64), "eef_quat": quat,
                   "joint_pos": np.asarray(raw["robot0_joint_pos"], dtype=np.float64)},
            images={"agentview": np.ascontiguousarray(raw["agentview_image"][::-1, ::-1]),
                    "wrist": np.ascontiguousarray(raw["robot0_eye_in_hand_image"][::-1, ::-1])},
            instruction=self._language, time=float(self._env.sim.data.time))

    def step(self, action: Action) -> StepResult:
        raw, reward, done, info = self._env.step(np.asarray(action.data, dtype=np.float64).tolist())
        self._raw = raw
        success = bool(self._env.check_success())
        return StepResult(observation=self._observe(raw), reward=float(reward), terminated=bool(done or success),
                          truncated=False, success=success, info={})

    def privileged_state(self) -> dict:
        return {"object_state": np.asarray(self._raw.get("object-state", []), dtype=np.float64),
                "eef_pos": np.asarray(self._raw["robot0_eef_pos"], dtype=np.float64)}

    def render(self, camera: str = "agentview", width: int = 256, height: int = 256) -> np.ndarray:
        name = {"agentview": "agentview", "wrist": "robot0_eye_in_hand"}.get(camera, camera)
        return self._env.sim.render(camera_name=name, width=width, height=height)[::-1, ::-1]

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env, self._key = None, None
