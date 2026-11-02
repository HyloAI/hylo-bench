"""RoboCasa Environment adapter with the robot contract (hylo.robot/v0.1). Runs inside the RoboCasa venv
(bench/envs/robocasa: RoboCasa 1.0.1 on robosuite 1.5) behind an EnvServer; the main process never imports either.
The contract's machinery is `hylo.envs.robosuite_contract`; this module builds RoboCasa's kitchens and labels them.

Robot: RoboCasa's PandaOmron (a Panda arm on a mobile base with a torso lift). The contract drives the arm only: the
base and the torso get zero commands and stay where the scene put them. World frame: the arm's mount (body
`robot0_link0`), +x forward, +z up; counters are typically 0.2-0.3 m below it (RobotSpec.table_height is not given:
it varies by kitchen, the tools measure it). After RoboCasa's reset the arm glides from RoboCasa's stretched start
posture to the contract's home posture (`ready_posture`), before the agent's first observation. One environment
instance serves one task (param `task`).
"""

from __future__ import annotations

from typing import Any, Sequence

from hylo.core.task import TaskInstance
from hylo.envs.robosuite_contract.env import IDENTITY, ArmLayout, RobosuiteContractEnv
from hylo.envs.robosuite_contract.labels import LabelRules
from hylo.envs.robosuite_contract.panda import PandaNames

from .tasks import DEFAULT_CAMERAS, fixture_label, object_label, part_rules, task as get_task

CAMERAS = {"agentview_left": "robot0_agentview_left", "agentview_center": "robot0_agentview_center",
           "agentview_right": "robot0_agentview_right", "frontview": "robot0_frontview",
           "robotview": "robot0_robotview", "wrist": "robot0_eye_in_hand"}


class RoboCasaEnv(RobosuiteContractEnv):
    """Params: task (required: a key of tasks.TASKS), image_size, cameras (default agentview_left, agentview_right,
    wrist; also agentview_center, frontview, robotview), segmentation, privileged_obs, kp, damping_ratio,
    settle_steps (as for the robosuite adapter), scene_kwargs (passed to RoboCasa's Kitchen, e.g. layout_and_style_ids,
    obj_instance_split)."""

    env_name = "robocasa"
    world_body = "robot0_link0"

    def __init__(self, task: str = "", image_size: int = 256, cameras: Sequence[str] | None = None,
                 segmentation: bool = False, privileged_obs: bool = False, kp: float = 300.0,
                 damping_ratio: float = 1.0, settle_steps: int = 10, scene_kwargs: dict | None = None) -> None:
        if not task:
            raise ValueError("robocasa env: set params.task (one of hylo.envs.robocasa.tasks.TASKS)")
        self.task = get_task(task)
        self.scene_kwargs = dict(scene_kwargs or {})
        super().__init__(image_size=image_size, cameras=tuple(cameras or DEFAULT_CAMERAS), segmentation=segmentation,
                         privileged_obs=privileged_obs, kp=kp, damping_ratio=damping_ratio, settle_steps=settle_steps)

    # -- layout (pure) ------------------------------------------------------------------------------------------
    def arm_layout(self) -> list[ArmLayout]:
        return [ArmLayout("arm", PandaNames(0, "PandaGripper"), [r[:] for r in IDENTITY], base_body="robot0_link0")]

    def camera_map(self) -> dict[str, tuple[str, str | None]]:
        return {k: (v, "arm" if k == "wrist" else None) for k, v in CAMERAS.items()}

    def notes(self, hand: str) -> str:
        return ("RoboCasa kitchen (MuJoCo, robosuite 1.5): a Franka Panda arm on a mobile base that stays put. World "
                "frame: the arm's mount, +z up, +x forward; the counter in front is usually 0.2-0.3 m below the "
                "mount (measure it: robot.table). Cabinets, drawers and appliances are articulated; doors swing on "
                "hinges, drawers slide, knobs turn about their axis, buttons press in. Action: 7 joint-position "
                f"targets (rad) then a hand command (0 closed, 1 open) at {self.control_hz:g} Hz. {hand} "
                f"{self.task.notes}"
                + (" Segment labels: 'counter' (the support), 'robot', objects by name (e.g. 'bottled water'), the "
                   "fixture the instruction refers to by kind ('drawer', 'cabinet', 'stove', 'sink', 'microwave'; "
                   "others of that kind are 'other drawer' and so on) and its parts ('drawer handle', 'rear left "
                   "knob', 'faucet handle', 'start button')." if self.segmentation else ""))

    def ready_posture(self):
        """RoboCasa starts the arm stretched out (joint 4 at its limit), which the toolkit's planner rejects as a
        self-collision: each episode glides it to the contract's home posture first (40 steps, before the agent's
        first observation)."""
        from hylo.envs.robosuite_contract.panda import HOME

        return [HOME]

    # -- simulator side -------------------------------------------------------------------------------------------
    def build(self) -> Any:
        import robocasa  # noqa: F401 - registers RoboCasa's environments with robosuite

        from hylo.envs.robosuite_contract.robosuite15 import make

        return make(self.task.env, "PandaOmron", self.kp, self.damping_ratio, controller_robot="PandaOmron",
                    **self.scene_kwargs)

    def start_episode(self, task: TaskInstance) -> None:
        key = task.spec.params.get("task")
        if key != self.task.key:
            raise ValueError(f"this robocasa env was configured for task {self.task.key!r} (env.params.task) but the "
                             f"episode is {key!r}: run one task per experiment")
        super().start_episode(task)
        self._rules = None

    def referenced_fixtures(self) -> dict[str, Any]:
        """The fixtures the task holds references to (attributes of the RoboCasa env), by fixture name."""
        from robocasa.models.fixtures import Fixture

        e = self._env
        return {v.name: v for v in vars(e).values() if isinstance(v, Fixture) and v.name in e.fixtures}

    def label_rules(self) -> LabelRules:
        if getattr(self, "_rules", None) is not None:
            return self._rules
        e = self._env
        names = self.body_names + self.geom_names
        refs = self.referenced_fixtures()
        kinds_referenced = {fixture_label(type(f).__name__) for f in refs.values()}
        objects: dict[str, str] = {}
        parts: list[tuple[str, str]] = []
        counters: list[str] = []
        for fname, f in e.fixtures.items():
            kind = fixture_label(type(f).__name__)
            if not kind:
                continue
            if kind == "counter":
                counters.append(fname + "_")
                continue
            if fname in refs:
                objects[fname] = kind
                prefix = f.naming_prefix
                parts += part_rules(prefix, kind, [n for n in names if n.startswith(prefix)])
            else:
                objects[fname] = f"other {kind}" if kind in kinds_referenced else kind
        for cfg in getattr(e, "object_cfgs", []):
            cat = (cfg.get("info") or {}).get("cat")
            if cfg.get("name") in e.objects and cat:
                objects[cfg["name"]] = object_label(cat)
        self._rules = LabelRules(objects=objects, parts=tuple(parts), support=(), support_prefixes=tuple(counters),
                                 support_label="counter")
        return self._rules

    def instruction(self) -> str:
        lang = self._env.get_ep_meta().get("lang") if self._env is not None else None
        return str(lang or self.task.instruction)

    def objects(self) -> list[tuple[str, str, str]]:
        rules = self.label_rules()
        out = [(k, o.root_body, rules.objects.get(k, k)) for k, o in self._env.objects.items()]
        for fname, f in self.referenced_fixtures().items():
            root = f.naming_prefix + "main"
            out.append((fname, root, rules.objects.get(fname, fname)))
        return out

    def extra(self) -> dict[str, Any]:
        meta = self._env.get_ep_meta()
        return {"task": self.task.key, "env": self.task.env, "layout_id": meta.get("layout_id"),
                "style_id": meta.get("style_id")}


__all__ = ["RoboCasaEnv"]
