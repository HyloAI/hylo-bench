"""robosuite Environment adapter with the robot contract (hylo.robot/v0.1). Runs inside the robosuite venv
(bench/envs/robosuite) behind an EnvServer; the main process never imports robosuite. The contract's machinery
(action and observation mapping, model export, cameras, segmentation) is `hylo.envs.robosuite_contract`; this module
only says which robosuite task to build and how it is laid out.

One environment instance serves one task (param `task`, a key of tasks.TASKS): the RobotSpec depends on it (one or
two arms, where the second arm stands, which hand). World frame: robot0's base. A seed fixes the episode: robosuite's
generator is reseeded with it before the reset. Success is robosuite's `_check_success()`; Wipe also ends on
robosuite's own failure check (the arm touching the table, a joint at its limit).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from hylo.core.task import TaskInstance
from hylo.envs.robosuite_contract.env import ArmLayout, RobosuiteContractEnv
from hylo.envs.robosuite_contract.labels import LabelRules
from hylo.envs.robosuite_contract.panda import PandaNames

from .tasks import base_in_world, task as get_task

FIXED_CAMERAS = ("agentview", "frontview", "birdview", "sideview")


class RobosuiteEnv(RobosuiteContractEnv):
    """Params: task (required: a key of tasks.TASKS), image_size (square cameras), cameras (contract names; default:
    the task's: agentview / sideview / frontview / birdview / robotview, `wrist` or `robot<i>_wrist`), segmentation
    (instance ids and labels; privileged), privileged_obs (object poses in extra["privileged"]), kp and damping_ratio
    (joint controller), settle_steps (holding steps after the reset)."""

    env_name = "robosuite"

    def __init__(self, task: str = "", image_size: int = 256, cameras: Sequence[str] | None = None,
                 segmentation: bool = False, privileged_obs: bool = False, kp: float = 300.0,
                 damping_ratio: float = 1.0, settle_steps: int = 10) -> None:
        if not task:
            raise ValueError("robosuite env: set params.task (one of hylo.envs.robosuite.tasks.TASKS)")
        self.task = get_task(task)
        super().__init__(image_size=image_size, cameras=tuple(cameras or self.task.cameras), segmentation=segmentation,
                         privileged_obs=privileged_obs, kp=kp, damping_ratio=damping_ratio, settle_steps=settle_steps)

    # -- layout (pure) ------------------------------------------------------------------------------------------
    def arm_layout(self) -> list[ArmLayout]:
        names = ["arm"] if self.task.arms == 1 else [f"robot{i}" for i in range(self.task.arms)]
        return [ArmLayout(n, PandaNames(i, self.task.gripper), base_in_world(self.task, i)) for i, n in enumerate(names)]

    def camera_map(self) -> dict[str, tuple[str, str | None]]:
        out: dict[str, tuple[str, str | None]] = {c: (c, None) for c in FIXED_CAMERAS
                                                  if not (c == "sideview" and self.task.env.startswith("PickPlace"))}
        out["robotview"] = ("robot0_robotview", None)
        if self.task.arms == 1:
            out["wrist"] = ("robot0_eye_in_hand", "arm")
        else:
            out.update({f"robot{i}_wrist": (f"robot{i}_eye_in_hand", f"robot{i}") for i in range(self.task.arms)})
        return out

    def table_height(self) -> float | None:
        return self.task.table_height

    def notes(self, hand: str) -> str:
        t = self.task
        two = ("Two arms facing each other across the table: robot0 stands at the world origin (its base frame), "
               "robot1 opposite it (ArmSpec.base_in_world). " if t.arms == 2 else "")
        labels = list(dict.fromkeys([*(lab for _, lab in t.labels.attached), *(lab for _, lab in t.labels.parts),
                                     *t.labels.objects.values(), *t.labels.bodies.values(), *t.labels.markers.values()]))
        return (f"robosuite {t.env} (MuJoCo), Franka Panda arm(s) on fixed mounts. World frame: robot0's base frame, "
                "+z up, +x from the base towards the workspace; the table top is at RobotSpec.table_height (about "
                f"0.1 m below the base). {two}Action: per arm 7 joint-position targets (rad) then a hand command "
                f"(0 closed, 1 open) at {self.control_hz:g} Hz. {hand} {t.notes}"
                + (f" Segment labels: 'table', 'robot' and {', '.join(repr(v) for v in labels[:6])}."
                   if self.segmentation else ""))

    # -- simulator side -------------------------------------------------------------------------------------------
    def build(self) -> Any:
        from hylo.envs.robosuite_contract.robosuite15 import make

        robots = ["Panda"] * self.task.arms
        return make(self.task.env, robots if self.task.arms > 1 else robots[0], self.kp, self.damping_ratio,
                    hard_reset=True, **self.task.kwargs)

    def start_episode(self, task: TaskInstance) -> None:
        key = task.spec.params.get("task")
        if key != self.task.key:
            raise ValueError(f"this robosuite env was configured for task {self.task.key!r} (env.params.task) but the "
                             f"episode is {key!r}: run one task per experiment")
        super().start_episode(task)

    def label_rules(self) -> LabelRules:
        return self.task.labels

    def instruction(self) -> str:
        return self.task.instruction

    def failed(self) -> bool:
        return bool(self.task.terminate_on_failure and self._env._check_terminated())

    def objects(self) -> list[tuple[str, str, str]]:
        out = super().objects()
        if self.task.key == "two_arm_peg_in_hole":
            out += [("peg", self._env.peg.root_body, "peg"), ("hole", self._env.hole.root_body, "plate")]
        return out

    def measured_table_height(self) -> float | None:
        e = self._env
        z = None
        if getattr(e, "table_offset", None) is not None:
            z = float(np.asarray(e.table_offset, float)[2])
        elif hasattr(e, "bin2_pos"):
            z = float(np.asarray(e.bin2_pos, float)[2])
        if z is None:
            return None
        return round(float((np.linalg.inv(self.world_pose()) @ np.array([0.0, 0.0, z, 1.0]))[2]), 4)

    def extra(self) -> dict[str, Any]:
        return {"task": self.task.key, "env": self.task.env}


__all__ = ["RobosuiteEnv"]
