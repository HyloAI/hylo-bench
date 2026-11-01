"""MuJoCo planar 2-link arm reaching task. Environment + Task logic are ours; MuJoCo is only the
Simulator (it has no env/task/seed concepts of its own; see research/projects/mujoco.md)."""

from __future__ import annotations

import numpy as np

from hylo.core.task import TaskInstance, TaskSpec
from hylo.core.types import (Action, ActionSemantics, BoxSpace, EnvCapabilities, EnvSpec, Observation,
                          ObservationSpec, StepResult)

L1, L2 = 0.3, 0.25
MAX_JOINT_STEP = 0.05  # rad per control step at |action| = 1
SUCCESS_RADIUS = 0.02

MJCF = f"""
<mujoco model="planar_reach">
  <compiler angle="radian"/>  <!-- MJCF defaults to degrees for joint ranges -->
  <option timestep="0.01" integrator="implicitfast"/>
  <worldbody>
    <body name="link1">
      <joint name="j1" type="hinge" axis="0 0 1" damping="0.5"/>
      <geom type="capsule" fromto="0 0 0 {L1} 0 0" size="0.02" mass="0.5"/>
      <body name="link2" pos="{L1} 0 0">
        <joint name="j2" type="hinge" axis="0 0 1" damping="0.5" limited="true" range="-2.8 2.8"/>
        <geom type="capsule" fromto="0 0 0 {L2} 0 0" size="0.02" mass="0.3"/>
        <site name="tip" pos="{L2} 0 0" size="0.01"/>
      </body>
    </body>
    <site name="target" pos="0.3 0.2 0" size="0.02" rgba="1 0 0 1"/>
  </worldbody>
  <actuator>
    <position name="a1" joint="j1" kp="20"/>
    <position name="a2" joint="j2" kp="20" ctrllimited="true" ctrlrange="-2.8 2.8"/>
  </actuator>
</mujoco>
"""


class PlanarReachEnv:
    def __init__(self, control_hz: float = 20.0) -> None:
        import mujoco  # optional dependency, imported lazily so spec() stays cheap

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_string(MJCF)
        self.data = mujoco.MjData(self.model)
        self.control_hz = control_hz
        self.n_substeps = max(1, round(1.0 / (control_hz * self.model.opt.timestep)))
        self._tip = self.model.site("tip").id
        self._target = self.model.site("target").id
        self._task: TaskInstance | None = None

    def spec(self) -> EnvSpec:
        return EnvSpec(
            name="mujoco.planar_reach", version="0.1",
            observation=ObservationSpec(
                state={"qpos": (2,), "qvel": (2,), "tip_pos": (2,), "target_pos": (2,)},
                units={"qpos": "rad", "qvel": "rad/s", "tip_pos": "m", "target_pos": "m"}),
            action=BoxSpace(shape=(2,), low=(-1.0, -1.0), high=(1.0, 1.0),
                            semantics=ActionSemantics(control_mode="joint_delta", frame="base",
                                                      dim_labels=("j1", "j2"),
                                                      max_step=(MAX_JOINT_STEP, MAX_JOINT_STEP))),
            control_hz=self.control_hz,
            capabilities=EnvCapabilities(seedable=True, snapshot=True, privileged_state=True),
        )

    def reset(self, task: TaskInstance) -> Observation:
        mj = self._mj
        self._task = task
        rng = np.random.default_rng(task.seed)
        mj.mj_resetData(self.model, self.data)
        self.data.qpos[:] = rng.uniform(-0.5, 0.5, size=2)
        target = task.spec.params.get("target")
        if target is None:
            r, th = rng.uniform(0.15, 0.5), rng.uniform(-np.pi, np.pi)
            target = [r * np.cos(th), r * np.sin(th)]
        self.model.site_pos[self._target, :2] = target
        self.data.ctrl[:] = self.data.qpos
        mj.mj_forward(self.model, self.data)
        self._warn = self._divergence_count()
        self.t = 0
        return self._observe()

    def _divergence_count(self) -> int:
        w = self._mj.mjtWarning
        return sum(int(self.data.warning[k].number) for k in (w.mjWARN_BADQPOS, w.mjWARN_BADQVEL, w.mjWARN_BADQACC))

    def _observe(self) -> Observation:
        return Observation(
            state={"qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy(),
                   "tip_pos": self.data.site_xpos[self._tip, :2].copy(),
                   "target_pos": self.model.site_pos[self._target, :2].copy()},
            instruction=self._task.spec.instruction,  # type: ignore[union-attr]
            time=float(self.data.time),
        )

    def step(self, action: Action) -> StepResult:
        mj = self._mj
        delta = np.asarray(action.data, dtype=np.float64) * MAX_JOINT_STEP
        ctrl = self.data.ctrl + delta
        limited = self.model.actuator_ctrllimited.astype(bool)  # only clip actuators that declare a range
        ctrl[limited] = np.clip(ctrl[limited], *self.model.actuator_ctrlrange[limited].T)
        self.data.ctrl[:] = ctrl
        mj.mj_step(self.model, self.data, nstep=self.n_substeps)
        self.t += 1
        # MuJoCo silently resets MjData on divergence and only bumps a warning counter
        # (engine_forward.c); surface it as an infra-level truncation instead of hiding it.
        bad = self._divergence_count()
        if bad != self._warn:
            self._warn = bad
            return StepResult(observation=self._observe(), reward=None, terminated=False, truncated=True,
                              success=None, info={"sim_error": "mujoco divergence warning (BADQPOS/QVEL/QACC)"})
        dist = float(np.linalg.norm(self.data.site_xpos[self._tip, :2] - self.model.site_pos[self._target, :2]))
        success = dist < SUCCESS_RADIUS
        return StepResult(observation=self._observe(), reward=-dist, terminated=success, truncated=False,
                          success=success, info={"distance": dist})

    def privileged_state(self) -> dict:
        return {"qpos": self.data.qpos.copy(), "ctrl": self.data.ctrl.copy(),
                "target": self.model.site_pos[self._target, :2].copy()}

    def snapshot(self) -> bytes:
        mj = self._mj
        sig = mj.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mj.mj_stateSize(self.model, sig))
        mj.mj_getState(self.model, self.data, state, sig)
        return np.concatenate([state, self.model.site_pos[self._target, :2], [self.t]]).tobytes()

    def restore(self, blob: bytes) -> None:
        mj = self._mj
        arr = np.frombuffer(blob, dtype=np.float64)
        sig = mj.mjtState.mjSTATE_INTEGRATION
        mj.mj_setState(self.model, self.data, arr[:-3].copy(), sig)
        self.model.site_pos[self._target, :2] = arr[-3:-1]
        self.t = int(arr[-1])
        mj.mj_forward(self.model, self.data)

    def close(self) -> None:
        pass


def mujoco_tasks() -> tuple[TaskSpec, ...]:
    return (
        TaskSpec(id="mujoco/planar_reach_fixed", version="1", instruction="touch the red target",
                 params={"target": [0.35, 0.2]}, max_steps=150, tags=("reach",)),
        TaskSpec(id="mujoco/planar_reach_random", version="1", instruction="touch the red target",
                 params={}, max_steps=150, tags=("reach",)),
    )


RSI_SEED_OFFSET = 30_000      # RFC-0002 splits use their own seeds, disjoint from train / validation / heldout


def splits_for(protocol: str, train: int, validation: int, heldout: int, rsi_states: int = 100,
               gate_looks: int = 1) -> dict[str, tuple[int, ...]]:
    """protocol "standard": train / validation / heldout. protocol "rsi": search / search_val / gate / heldout
    (RFC-0002, 40 / 20 / 20 / 20 percent of `rsi_states` seeds) for improvement campaigns."""
    from hylo.core.benchmark import rsi_splits, standard_splits

    if protocol == "rsi":
        n = rsi_states
        raw = rsi_splits(n, (int(0.4 * n), int(0.6 * n), int(0.8 * n)), gate_looks=gate_looks)
        return {k: tuple(RSI_SEED_OFFSET + s for s in v) for k, v in raw.items()}
    return standard_splits(train, validation, heldout)


def mujoco_benchmark(train: int = 5, validation: int = 10, heldout: int = 10, protocol: str = "standard",
                     rsi_states: int = 100, gate_looks: int = 1):
    from hylo.core.benchmark import Benchmark

    return Benchmark(id="mujoco/reach-v0", version="0.1", tasks=mujoco_tasks(),
                     splits=splits_for(protocol, train, validation, heldout, rsi_states, gate_looks))
