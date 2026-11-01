"""MuJoCo table-top reaching with language, occlusion and disturbances.

A planar 2-link arm over a table with three coloured pads. Each episode samples pad colours and
positions, a target pad, and a natural-language instruction (direct: "touch the red pad", or
indirect: "touch the pad the colour of a ripe tomato"). Pads are only observable for the first
`visible_steps` steps; afterwards the agent must rely on memory. Optionally the arm is pushed.

Task logic (init distribution, instruction, success / failure predicates) lives here; MuJoCo is the
Simulator. Success = fingertip dwells on the target pad; touching another pad long enough fails.
"""

from __future__ import annotations

import numpy as np

from hylo.core.task import TaskInstance, TaskSpec
from hylo.core.types import (Action, ActionSemantics, BoxSpace, EnvCapabilities, EnvSpec, Observation,
                                     ObservationSpec, StepResult)

L1, L2 = 0.3, 0.25
MAX_JOINT_STEP = 0.05
N_PADS = 3
PAD_RADIUS = 0.035
SUCCESS_DWELL, WRONG_DWELL = 3, 6
CAMERA_SIZE = 384  # square camera observations (pixels)

PALETTE: dict[str, tuple[float, float, float]] = {
    "red": (0.85, 0.1, 0.1), "green": (0.1, 0.7, 0.2), "blue": (0.15, 0.35, 0.9),
    "yellow": (0.95, 0.85, 0.1), "purple": (0.55, 0.2, 0.75), "orange": (0.95, 0.5, 0.1),
}
DIRECT = ["touch the {c} pad", "move your fingertip onto the {c} pad", "go to the {c} marker"]
INDIRECT = {
    "red": "touch the pad that is the colour of a ripe tomato",
    "green": "touch the pad that is the colour of fresh grass",
    "blue": "touch the pad that is the colour of a clear daytime sky",
    "yellow": "touch the pad that is the colour of a ripe banana",
    "purple": "touch the pad that is the colour of an eggplant",
    "orange": "touch the pad that is the colour of a pumpkin",
}

_PADS_XML = "\n".join(
    f'    <geom name="pad{i}" type="cylinder" size="{PAD_RADIUS} 0.004" pos="0.3 0 0.002" '
    f'contype="0" conaffinity="0" rgba="0.5 0.5 0.5 1"/>' for i in range(N_PADS))

MJCF = f"""
<mujoco model="table_reach">
  <compiler angle="radian"/>
  <option timestep="0.01" integrator="implicitfast" gravity="0 0 0"/>
  <visual>
    <global offwidth="960" offheight="720"/>
    <headlight ambient="0.3 0.3 0.3" diffuse="0.35 0.35 0.35" specular="0.05 0.05 0.05"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.80 0.86 0.92" rgb2="0.30 0.36 0.44" width="512" height="3072"/>
    <texture name="wood" type="2d" builtin="flat" rgb1="0.80 0.72 0.60" rgb2="0.74 0.66 0.54" mark="random" random="0.08" width="512" height="512"/>
    <material name="table" texture="wood" texrepeat="3 3" reflectance="0.04" specular="0.1"/>
    <material name="floor" rgba="0.55 0.57 0.60 1" reflectance="0.0"/>
    <material name="arm" rgba="0.22 0.24 0.28 1" specular="0.6" shininess="0.6"/>
  </asset>
  <worldbody>
    <light directional="true" pos="0.6 -0.8 2.5" dir="-0.25 0.35 -1" diffuse="0.55 0.55 0.55" specular="0.1 0.1 0.1" castshadow="true"/>
    <light directional="true" pos="-0.8 0.6 2.0" dir="0.3 -0.2 -1" diffuse="0.2 0.2 0.22" castshadow="false"/>
    <camera name="top" pos="0 0 1.45" xyaxes="1 0 0 0 1 0" fovy="50"/>
    <camera name="front" pos="0 -0.92 1.0" xyaxes="1 0 0 0 0.74 0.67" fovy="48"/>
    <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 -0.75" material="floor" contype="0" conaffinity="0"/>
    <geom name="table" type="box" size="0.65 0.65 0.02" pos="0 0 -0.022" material="table" contype="0" conaffinity="0"/>
    <geom name="table_leg" type="box" size="0.55 0.55 0.36" pos="0 0 -0.4" rgba="0.35 0.30 0.25 1" contype="0" conaffinity="0"/>
{_PADS_XML}
    <body name="base" pos="0 0 0.03">
      <geom type="cylinder" size="0.04 0.03" rgba="0.2 0.2 0.25 1" contype="0" conaffinity="0"/>
      <body name="link1" pos="0 0 0.03">
        <joint name="j1" type="hinge" axis="0 0 1" damping="0.5"/>
        <geom type="capsule" fromto="0 0 0 {L1} 0 0" size="0.022" material="arm" mass="0.5"/>
        <body name="link2" pos="{L1} 0 0">
          <joint name="j2" type="hinge" axis="0 0 1" damping="0.5" limited="true" range="-2.8 2.8"/>
          <geom type="capsule" fromto="0 0 0 {L2} 0 0" size="0.019" rgba="0.42 0.45 0.50 1" mass="0.3"/>
          <geom type="sphere" pos="{L2} 0 0" size="0.022" rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0" mass="0"/>  <!-- visual only: no inertia -->
          <site name="tip" pos="{L2} 0 0" size="0.012" rgba="1 1 1 1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a1" joint="j1" kp="20"/>
    <position name="a2" joint="j2" kp="20" ctrllimited="true" ctrlrange="-2.8 2.8"/>
  </actuator>
</mujoco>
"""


class TableReachEnv:
    def __init__(self, control_hz: float = 20.0) -> None:
        import mujoco

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_string(MJCF)
        self.data = mujoco.MjData(self.model)
        self.control_hz = control_hz
        self.n_substeps = max(1, round(1.0 / (control_hz * self.model.opt.timestep)))
        self._tip = self.model.site("tip").id
        self._pads = [self.model.geom(f"pad{i}").id for i in range(N_PADS)]
        self._task: TaskInstance | None = None

    def spec(self) -> EnvSpec:
        return EnvSpec(
            name="mujoco.table_reach", version="0.1",
            observation=ObservationSpec(
                state={"qpos": (2,), "qvel": (2,), "tip_pos": (2,), "obj_pos": (N_PADS, 2),
                       "obj_rgb": (N_PADS, 3), "objects_visible": (1,)},
                cameras={"top": (CAMERA_SIZE, CAMERA_SIZE, 3)},  # only when the task sets camera_obs
                units={"qpos": "rad", "qvel": "rad/s", "tip_pos": "m", "obj_pos": "m", "obj_rgb": "normalized"}),
            action=BoxSpace(shape=(2,), low=(-1.0, -1.0), high=(1.0, 1.0),
                            semantics=ActionSemantics(control_mode="joint_delta", frame="base", dim_labels=("j1", "j2"),
                                                      max_step=(MAX_JOINT_STEP, MAX_JOINT_STEP))),
            control_hz=self.control_hz,
            capabilities=EnvCapabilities(seedable=True, snapshot=False, render=True, privileged_state=True))

    # -- episode setup -----------------------------------------------------------------------------
    def reset(self, task: TaskInstance) -> Observation:
        mj, p = self._mj, task.spec.params
        self._task = task
        rng = np.random.default_rng(task.seed)
        mj.mj_resetData(self.model, self.data)
        self.colors = list(rng.choice(sorted(PALETTE), size=N_PADS, replace=False))
        self.pad_pos = self._sample_pads(rng)
        for gid, name, xy in zip(self._pads, self.colors, self.pad_pos):
            self.model.geom_pos[gid, :2] = xy
            self.model.geom_rgba[gid] = (*PALETTE[name], 1.0)
        self.target = int(rng.integers(N_PADS))
        phrasing = p.get("phrasing", "direct")
        if phrasing == "mixed":
            phrasing = "indirect" if rng.random() < 0.5 else "direct"
        name = self.colors[self.target]
        self.instruction = INDIRECT[name] if phrasing == "indirect" else DIRECT[int(rng.integers(len(DIRECT)))].format(c=name)
        self.phrasing = phrasing
        self.data.qpos[:] = rng.uniform(-0.4, 0.4, size=2)
        self.data.ctrl[:] = self.data.qpos
        mj.mj_forward(self.model, self.data)
        self._warn = self._divergence_count()
        self.t, self.dwell = 0, np.zeros(N_PADS, dtype=int)
        return self._observe()

    @staticmethod
    def _sample_pads(rng: np.random.Generator) -> np.ndarray:
        pads: list[np.ndarray] = []
        while len(pads) < N_PADS:
            r, th = rng.uniform(0.22, 0.5), rng.uniform(-np.pi, np.pi)
            xy = np.array([r * np.cos(th), r * np.sin(th)])
            if all(np.linalg.norm(xy - q) > 0.18 for q in pads):
                pads.append(xy)
        return np.stack(pads)

    def _divergence_count(self) -> int:
        w = self._mj.mjtWarning
        return sum(int(self.data.warning[k].number) for k in (w.mjWARN_BADQPOS, w.mjWARN_BADQVEL, w.mjWARN_BADQACC))

    @property
    def visible(self) -> bool:
        n = self._task.spec.params.get("visible_steps")  # type: ignore[union-attr]
        return n is None or self.t < n

    def _observe(self) -> Observation:
        vis = self.visible
        rgb = np.array([PALETTE[c] for c in self.colors])
        images, extra = {}, {"phrasing": self.phrasing}
        cams = self._task.spec.params.get("camera_obs") or []  # type: ignore[union-attr]
        if cams and vis:  # cameras see the pads only while they are visible
            images = {c: self.render(c, CAMERA_SIZE, CAMERA_SIZE) for c in cams}
            extra["camera_params"] = {c: self.camera_params(c, CAMERA_SIZE, CAMERA_SIZE) for c in cams}
        return Observation(
            state={"qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy(),
                   "tip_pos": self.data.site_xpos[self._tip, :2].copy(),
                   "obj_pos": self.pad_pos.copy() if vis else np.zeros((N_PADS, 2)),
                   "obj_rgb": rgb if vis else np.zeros((N_PADS, 3)),
                   "objects_visible": np.array([1.0 if vis else 0.0])},
            images=images, instruction=self.instruction, time=float(self.data.time), extra=extra)

    def camera_params(self, camera: str, width: int, height: int) -> dict:
        """Pinhole parameters so consumers can project world points into the image (MuJoCo cameras look along -z)."""
        cid = self.model.camera(camera).id
        return {"pos": self.data.cam_xpos[cid].tolist(), "xmat": self.data.cam_xmat[cid].tolist(),
                "fovy_deg": float(self.model.cam_fovy[cid]), "width": width, "height": height}

    # -- dynamics & task predicates ----------------------------------------------------------------
    def step(self, action: Action) -> StepResult:
        mj = self._mj
        ctrl = self.data.ctrl + np.asarray(action.data, dtype=np.float64) * MAX_JOINT_STEP
        limited = self.model.actuator_ctrllimited.astype(bool)
        ctrl[limited] = np.clip(ctrl[limited], *self.model.actuator_ctrlrange[limited].T)
        self.data.ctrl[:] = ctrl
        push = self._task.spec.params.get("push")  # type: ignore[union-attr]
        if push and self.t == push["step"]:
            self.data.ctrl[:] = self.data.ctrl + np.asarray(push["offset"], dtype=np.float64)  # a shove that moves the servo target
            self.data.qvel[:] = self.data.qvel + np.asarray(push.get("qvel", [0.0, 0.0]), dtype=np.float64)
        mj.mj_step(self.model, self.data, nstep=self.n_substeps)
        self.t += 1
        if self._divergence_count() != self._warn:
            self._warn = self._divergence_count()
            return StepResult(observation=self._observe(), reward=None, terminated=False, truncated=True,
                              success=None, info={"sim_error": "mujoco divergence warning"})
        tip = self.data.site_xpos[self._tip, :2]
        dists = np.linalg.norm(self.pad_pos - tip, axis=1)
        on = dists < PAD_RADIUS
        self.dwell = np.where(on, self.dwell + 1, 0)
        success = bool(self.dwell[self.target] >= SUCCESS_DWELL)
        wrong = [i for i in range(N_PADS) if i != self.target and self.dwell[i] >= WRONG_DWELL]
        info = {"distance": float(dists[self.target]), "phrasing": self.phrasing}
        if wrong:
            info["fail_reason"] = f"touched the {self.colors[wrong[0]]} pad"
        return StepResult(observation=self._observe(), reward=-float(dists[self.target]),
                          terminated=success or bool(wrong), truncated=False, success=success, info=info)

    def privileged_state(self) -> dict:
        return {"qpos": self.data.qpos.copy(), "tip": self.data.site_xpos[self._tip, :2].copy(),
                "pad_pos": self.pad_pos.copy(), "pad_rgb": np.array([PALETTE[c] for c in self.colors]),
                "colors": list(self.colors), "target": self.target, "instruction": self.instruction,
                "phrasing": self.phrasing, "visible": self.visible,
                "distance": float(np.linalg.norm(self.pad_pos[self.target] - self.data.site_xpos[self._tip, :2]))}

    def render(self, camera: str = "front", width: int = 640, height: int = 480) -> np.ndarray:
        """Requires an OpenGL backend (MUJOCO_GL=egl or osmesa)."""
        renderers = self.__dict__.setdefault("_renderers", {})
        if (width, height) not in renderers:
            renderers[(width, height)] = self._mj.Renderer(self.model, height, width)
        r = renderers[(width, height)]
        r.update_scene(self.data, camera=camera)
        return r.render()

    def close(self) -> None:
        for r in self.__dict__.pop("_renderers", {}).values():
            r.close()


def table_tasks() -> tuple[TaskSpec, ...]:
    common = {"visible_steps": 5}
    return (
        TaskSpec(id="mujoco/table_direct", version="1", instruction="touch the <colour> pad",
                 params={**common, "phrasing": "direct"}, max_steps=150, tags=("language", "memory")),
        TaskSpec(id="mujoco/table_indirect", version="1", instruction="touch the pad described by colour",
                 params={**common, "phrasing": "indirect"}, max_steps=150, tags=("language", "memory")),
        TaskSpec(id="mujoco/table_push", version="1", instruction="touch the <colour> pad despite a shove",
                 params={**common, "phrasing": "direct", "push": {"step": 12, "offset": [0.6, -0.6]}},
                 max_steps=70, tags=("memory", "recovery")),   # 3.5 s budget: slow controllers time out
    )


def table_benchmark(tags: list[str] | None = None, train: int = 10, validation: int = 20, heldout: int = 20,
                    vision: bool = False, protocol: str = "standard", rsi_states: int = 100, gate_looks: int = 1):
    """vision=True adds a top-camera image to observations while pads are visible (needs MuJoCo rendering).
    Task ids stay the same so results pair with the state-only variant seed by seed."""
    from hylo.core.benchmark import Benchmark

    from .planar_reach import splits_for

    tasks = tuple(t for t in table_tasks() if not tags or set(tags) & set(t.tags))
    if vision:
        tasks = tuple(t.model_copy(update={"params": {**t.params, "camera_obs": ["top"]}}) for t in tasks)
    return Benchmark(id="mujoco/table-vision-v0" if vision else "mujoco/table-v0", version="0.1", tasks=tasks,
                     splits=splits_for(protocol, train, validation, heldout, rsi_states, gate_looks),
                     metadata={"tags": ",".join(tags or [])})
