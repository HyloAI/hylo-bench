"""The stage world (RFC-0008 part 3): a running MuJoCo simulation of a stage that tools act on through actions with
receipts.

    world = World(stage, root=stage_dir)          # compiles the stage; generation 1
    r = world.act({"action_id": "a1", "expected_generation": 1, "robot": "arm", "kind": "tcp",
                   "position": [0.5, 0.0, 0.3]})
    r["status"], r["steps"], r["measured"]
    world.capture(["overview"])                    # RGB-D, segmentation by entity, K, camera pose

Rules:
- Every action names the generation it was decided on. A world that was rebuilt (a new stage revision) or reset has
  a new generation, and an action for an older one is refused as `stale`: observe again first.
- `action_id` makes an action idempotent: sending the same id again returns the stored receipt.
- One action per robot at a time; `act_batch` runs actions of several robots in the same steps (all start together,
  each ends on its own; the receipt of each says when).
- Time passes only inside actions (the clock is manual): the world holds still between tool calls.
- A receipt reports what was measured in the simulation after the action, never what was intended.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import mujoco
import numpy as np

from hylo.core.stage import Stage

from .compose import Composed, RobotInfo, compose
from .controllers import Fly, GripperMove, Hold, JointMove, Walk, ik

_GENERATION = [0]


def _next_generation() -> int:
    _GENERATION[0] += 1
    return _GENERATION[0]


class ActionRefused(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class Robot:
    """One robot of the world: its actuators, joints, base and tool point, found by the "<id>/" prefix."""

    def __init__(self, world: "World", info: RobotInfo) -> None:
        m, self.id, self.pack, self.info = world.model, info.id, info.pack, info
        pre = f"{info.id}/"
        self.actuator_ids = [i for i in range(m.nu) if m.actuator(i).name.startswith(pre)]
        self.actuators = [m.actuator(i).name[len(pre):] for i in self.actuator_ids]
        self.joint_ids = [j for j in range(m.njnt) if m.joint(j).name.startswith(pre) and int(m.jnt_type[j]) in (2, 3)]
        self.joints = [m.joint(j).name[len(pre):] for j in self.joint_ids]
        self.root_body = m.body(self.root_name(m)).id
        self.free_joint = next((j for j in range(m.njnt) if int(m.jnt_type[j]) == 0
                                and int(m.jnt_bodyid[j]) == self.root_body), None)
        self.free_qpos = int(m.jnt_qposadr[self.free_joint]) if self.free_joint is not None else None
        self.free_dof = int(m.jnt_dofadr[self.free_joint]) if self.free_joint is not None else None
        # actuator -> joint (joint transmissions), for targets in joint units and errors
        self.act_joint = {k: int(m.actuator_trnid[a][0]) if int(m.actuator_trntype[a]) == 0 else None
                          for k, a in enumerate(self.actuator_ids)}
        self.torque = self.pack.drive == "torque"
        self.target = np.zeros(len(self.actuator_ids))
        self.home_target: dict[int, float] = {}
        self.gripper_index = self.actuators.index(self.pack.gripper.actuator) if self.pack.gripper and \
            self.pack.gripper.actuator in self.actuators else None
        g = self.pack.gripper
        skip = {self.gripper_index}
        base = {f"{pre}{n}" for n in self.pack.base_joints}
        self.arm_joints = [j for k, j in self.act_joint.items() if j is not None and k not in skip
                           and m.joint(j).name not in base]
        self.base_index = {n: self.actuators.index(n) for n in self.pack.base_joints if n in self.actuators}
        self.legs = []
        for leg in self.pack.legs:
            try:
                self.legs.append({"name": leg.name, "hip": self._act_for_joint(m, pre + leg.hip),
                                  "thigh": self._act_for_joint(m, pre + leg.thigh),
                                  "calf": self._act_for_joint(m, pre + leg.calf)})
            except KeyError:
                pass
        self.calf_sign = 1.0
        if self.legs:
            calf_home = info.home_qpos.get(pre + self.pack.legs[0].calf)
            self.calf_sign = 1.0 if calf_home is None or float(calf_home[0]) <= 0 else -1.0
        self.tcp_site = m.site(pre + self.pack.tcp.site).id if self.pack.tcp and self.pack.tcp.site else None
        self.tcp_body = m.body(pre + self.pack.tcp.body).id if self.pack.tcp and self.pack.tcp.body else None
        self.tcp_offset = np.asarray(self.pack.tcp.offset if self.pack.tcp else (0, 0, 0), float)
        self._g = g

    def root_name(self, m: Any) -> str:
        pre = f"{self.id}/"
        return next(m.body(b).name for b in range(m.nbody) if m.body(b).name.startswith(pre))

    def _act_for_joint(self, m: Any, joint: str) -> int:
        j = m.joint(joint).id
        for k, jj in self.act_joint.items():
            if jj == j:
                return k
        raise KeyError(joint)

    # --------------------------------------------------------------- targets and measurements
    def target_joint(self, k: int, target: float) -> float:
        """A target in joint units (position servos: ctrl / gear)."""
        return target

    def actuator_position(self, d: Any, k: int) -> float | None:
        j = self.act_joint.get(k)
        if j is None:
            return None
        return float(d.qpos[self._m.jnt_qposadr[j]])

    def apply(self, m: Any, d: Any) -> None:
        """Targets -> ctrl (servos directly; torque motors through PD)."""
        self._m = m
        if self.pack.drive == "thrust" or getattr(self, "policy_active", False):
            return
        if not self.torque:
            d.ctrl[self.actuator_ids] = self.target
            return
        kp, kd = self.pack.gains
        for k, a in enumerate(self.actuator_ids):
            j = self.act_joint[k]
            if j is None:
                continue
            q, qd = d.qpos[m.jnt_qposadr[j]], d.qvel[m.jnt_dofadr[j]]
            u = kp * (self.target[k] - q) - kd * qd
            lo, hi = m.actuator_ctrlrange[a] if m.actuator_ctrllimited[a] else (-np.inf, np.inf)
            d.ctrl[a] = float(np.clip(u, lo, hi))

    def base_pose(self, d: Any) -> tuple[np.ndarray, float]:
        p = d.xpos[self.root_body].copy()
        R = d.xmat[self.root_body].reshape(3, 3)
        return p, math.atan2(R[1, 0], R[0, 0])

    def base_up(self, d: Any) -> float:
        return float(d.xmat[self.root_body].reshape(3, 3)[2, 2])

    def root_pose(self, d: Any) -> tuple[np.ndarray, np.ndarray]:
        return d.xpos[self.root_body].copy(), d.xmat[self.root_body].reshape(3, 3).copy()

    def tcp(self, m: Any, d: Any) -> tuple[np.ndarray, np.ndarray] | None:
        if self.tcp_site is not None:
            return d.site_xpos[self.tcp_site].copy(), d.site_xmat[self.tcp_site].reshape(3, 3).copy()
        if self.tcp_body is not None:
            R = d.xmat[self.tcp_body].reshape(3, 3)
            return d.xpos[self.tcp_body] + R @ self.tcp_offset, R.copy()
        return None

    def tcp_jacobian(self, m: Any, d: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        if self.tcp_site is not None:
            mujoco.mj_jacSite(m, d, jacp, jacr, self.tcp_site)
        else:
            p, _ = self.tcp(m, d)
            mujoco.mj_jac(m, d, jacp, jacr, p, self.tcp_body)
        p, R = self.tcp(m, d)
        return p, R, jacp, jacr

    def gripper_opening(self, m: Any, d: Any) -> float | None:
        """Measured opening 0 (closed) .. 1 (open): the command a position servo would need to hold the actuator at
        its current length (gain x ctrl + bias0 + bias1 x length = 0), between the pack's closed and open commands."""
        if self.gripper_index is None:
            return None
        a = self.actuator_ids[self.gripper_index]
        gain, b0, b1 = float(m.actuator_gainprm[a][0]), float(m.actuator_biasprm[a][0]), float(m.actuator_biasprm[a][1])
        if gain == 0 or b1 == 0:
            return None
        ctrl_eq = -(b1 * float(d.actuator_length[a]) + b0) / gain
        g = self.pack.gripper
        return float(np.clip((ctrl_eq - g.closed) / (g.open - g.closed), 0.0, 1.0))

    def state(self, m: Any, d: Any) -> dict[str, Any]:
        out: dict[str, Any] = {"pack": self.pack.name,
                               "joints": {n: round(float(d.qpos[m.jnt_qposadr[j]]), 4)
                                          for n, j in zip(self.joints, self.joint_ids)}}
        p, yaw = self.base_pose(d)
        out["base"] = {"position": np.round(p, 4).tolist(), "yaw_deg": round(math.degrees(yaw), 2),
                       "upright": round(self.base_up(d), 3)}
        t = self.tcp(m, d)
        if t is not None:
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, t[1].ravel())
            out["tcp"] = {"position": np.round(t[0], 4).tolist(), "quat_wxyz": np.round(q, 4).tolist()}
        o = self.gripper_opening(m, d)
        if o is not None:
            out["gripper_opening"] = round(o, 3)
        return out


def _root(m: Any, b: int) -> int:
    while int(m.body_parentid[b]) != 0:
        b = int(m.body_parentid[b])
    return b


class World:
    def __init__(self, stage: Stage, root: str | Path = ".", *, composed: Composed | None = None) -> None:
        self.stage, self.root = stage, Path(root)
        c = composed or compose(stage, self.root)
        self.composed = c
        self.model = c.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.robots = {rid: Robot(self, info) for rid, info in c.robots.items()}
        self.cameras = c.cameras
        self.humans: Any = None
        if c.humans:
            from .crowd import Crowd

            self.humans = Crowd(self, c.humans)
        self.receipts: dict[str, dict[str, Any]] = {}
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> int:
        """Back to the stage's start (home postures, entities at their poses); a new generation."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        for r in self.robots.values():
            r._m = m
            for jname, val in r.info.home_qpos.items():
                try:
                    j = m.joint(jname).id
                except KeyError:
                    continue
                a = m.jnt_qposadr[j]
                d.qpos[a:a + len(val)] = val
            if r.free_qpos is not None and r.info.base_height is not None:
                d.qpos[r.free_qpos + 2] = r.info.pose_z + r.info.base_height
        mujoco.mj_forward(m, d)
        for r in self.robots.values():
            for k, a in enumerate(r.actuator_ids):
                name = m.actuator(a).name
                j = r.act_joint[k]
                if r.torque and j is not None:
                    r.target[k] = float(d.qpos[m.jnt_qposadr[j]])
                elif name in r.info.home_ctrl and (r.info.pack.home or j is None):
                    r.target[k] = r.info.home_ctrl[name]
                elif j is not None:
                    gear = float(m.actuator_gear[a][0]) or 1.0
                    r.target[k] = float(d.qpos[m.jnt_qposadr[j]]) * gear
                    if m.actuator_ctrllimited[a]:
                        r.target[k] = float(np.clip(r.target[k], *m.actuator_ctrlrange[a]))
            r.home_target = {k: float(v) for k, v in enumerate(r.target)}
            if r.pack.drive == "thrust" and r.info.pack.home:
                d.ctrl[r.actuator_ids] = [r.info.home_ctrl.get(m.actuator(a).name, 0.0) for a in r.actuator_ids]
            r.apply(m, d)
        self.step = 0
        self.generation = _next_generation()
        self.receipts.clear()
        return self.generation

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()

    # ------------------------------------------------------------------ stepping
    def _tick(self, controllers: Iterable[Any]) -> None:
        for c in controllers:
            if not c.done:
                c.update(self)
        for r in self.robots.values():
            r.apply(self.model, self.data)
        if self.humans is not None:
            self.humans.update(self.model.opt.timestep)
        mujoco.mj_step(self.model, self.data)
        self.step += 1

    def advance(self, seconds: float, on_step: Any = None) -> int:
        """Let time pass with every robot holding its targets (people keep walking); `on_step(world)` is called
        after each step (recording). Returns the steps taken."""
        n = self._n(seconds)
        holds = [Fly(r, self, r.root_pose(self.data)[0], self.robots[k].base_pose(self.data)[1], n)
                 for k, r in self.robots.items() if r.pack.drive == "thrust"]
        for _ in range(n):
            self._tick(holds)
            if on_step is not None:
                on_step(self)
        return n

    def act(self, action: Mapping[str, Any]) -> dict[str, Any]:
        return self.act_batch([action])[0]

    def act_batch(self, actions: list[Mapping[str, Any]], on_step: Any = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any] | None] = [None] * len(actions)
        live: list[tuple[int, Mapping[str, Any], Any, int]] = []
        busy: set[str] = set()
        for i, a in enumerate(actions):
            aid = str(a.get("action_id") or "")
            if aid and aid in self.receipts:
                out[i] = {**self.receipts[aid], "repeated": True}
                continue
            try:
                if a.get("expected_generation") != self.generation:
                    raise ActionRefused("stale", f"the world is at generation {self.generation}, the action was "
                                                 f"decided on {a.get('expected_generation')}: observe again")
                rid = a.get("robot")
                if rid not in self.robots:
                    raise ActionRefused("unknown_robot", f"no robot {rid!r}; robots: {sorted(self.robots)}")
                if rid in busy:
                    raise ActionRefused("busy", f"robot {rid} already has an action in this batch")
                ctl = self._controller(self.robots[rid], a)
            except ActionRefused as exc:
                out[i] = self._receipt(a, "refused", 0, 0, reason=f"{exc.code}: {exc}")
                continue
            busy.add(rid)
            live.append((i, a, ctl, self.step))
        limit = max([c.n + getattr(c, "settle", 0) for _, _, c, _ in live] or [0]) + 10
        n = 0
        while any(not c.done for _, _, c, _ in live) and n < limit:
            self._tick([c for _, _, c, _ in live])
            n += 1
            if on_step is not None:
                on_step(self)
            for i, a, c, s0 in live:
                if c.done and out[i] is None:
                    out[i] = self._receipt(a, "completed", s0, self.step, measured=c.measured(self))
        for i, a, c, s0 in live:
            if out[i] is None:
                out[i] = self._receipt(a, "completed", s0, self.step, measured=c.measured(self))
        return [o for o in out if o is not None]

    def _receipt(self, a: Mapping[str, Any], status: str, s0: int, s1: int, *, measured: Mapping[str, Any] | None = None,
                 reason: str = "") -> dict[str, Any]:
        rec = {"action_id": a.get("action_id"), "robot": a.get("robot"), "kind": a.get("kind"), "status": status,
               "start_step": s0, "end_step": s1, "steps": s1 - s0, "generation": self.generation,
               "time_s": round(self.data.time, 4), "measured": dict(measured or {})}
        if reason:
            rec["reason"] = reason
        if a.get("action_id"):
            self.receipts[str(a["action_id"])] = rec
        return rec

    def _n(self, seconds: float) -> int:
        return max(1, int(round(float(seconds) / self.model.opt.timestep)))

    def _controller(self, r: Robot, a: Mapping[str, Any]) -> Any:
        kind = a.get("kind")
        if kind not in r.pack.controls:
            raise ActionRefused("unsupported", f"{r.pack.name} accepts {list(r.pack.controls)}, not {kind!r}")
        dur = float(a.get("duration_s") or 1.0)
        m = self.model
        need = {"tcp": ("position",), "fly": ("position",), "policy": ("policy",)}.get(kind, ())
        missing = [k for k in need if a.get(k) is None]
        if missing:
            raise ActionRefused("bad_arguments", f"a {kind} action needs {missing}")
        if kind == "hold":
            if r.pack.drive == "thrust":                 # a drone holds by flying where it is
                p, R = r.root_pose(self.data)
                return Fly(r, self, p, math.atan2(R[1, 0], R[0, 0]), self._n(dur))
            return Hold(r, self._n(dur))
        if kind == "joints":
            goal: dict[int, float] = {}
            for name, v in dict(a.get("targets") or {}).items():
                if name not in r.actuators and name in r.joints:      # a joint name: its actuator
                    jid = r.joint_ids[r.joints.index(name)]
                    k = next((k for k, j in r.act_joint.items() if j == jid), None)
                    if k is None:
                        raise ActionRefused("no_actuator", f"joint {name} has no actuator")
                    name = r.actuators[k]
                if name not in r.actuators:
                    raise ActionRefused("unknown_joint", f"{r.id} has no joint or actuator {name!r}; "
                                                         f"actuators: {r.actuators}")
                k = r.actuators.index(name)
                ai = r.actuator_ids[k]
                v = float(v) * (1.0 if r.torque else float(m.actuator_gear[ai][0]) or 1.0)
                if m.actuator_ctrllimited[ai] and not r.torque:
                    v = float(np.clip(v, *m.actuator_ctrlrange[ai]))
                j = r.act_joint[k]
                if r.torque and j is not None and m.jnt_limited[j]:
                    v = float(np.clip(v, *m.jnt_range[j]))
                goal[k] = v
            if not goal:
                raise ActionRefused("empty", "targets is empty")
            return JointMove(r, goal, self._n(dur), self._n(0.3))
        if kind == "gripper":
            if r.gripper_index is None:
                raise ActionRefused("no_gripper", f"{r.pack.name} has no gripper actuator")
            g = r.pack.gripper
            cmd = a.get("command", "close")
            frac = {"open": 1.0, "close": 0.0}.get(cmd) if isinstance(cmd, str) else float(cmd)
            if frac is None:
                raise ActionRefused("bad_command", "command is open, close or a number 0 (closed) .. 1 (open)")
            return GripperMove(r, r.gripper_index, g.closed + frac * (g.open - g.closed), self._n(dur))
        if kind == "tcp":
            if r.tcp(m, self.data) is None or not r.arm_joints:
                raise ActionRefused("no_tcp", f"{r.pack.name} has no tool point")
            pos = np.asarray(a["position"], float)
            Rt = r.tcp(m, self.data)[1]                  # no orientation given: keep the current one
            if a.get("free_orientation"):
                Rt = None
            if a.get("quat_wxyz") is not None:
                Rt = np.zeros(9)
                q = np.asarray(a["quat_wxyz"], float)
                mujoco.mju_quat2Mat(Rt, q / np.linalg.norm(q))
                Rt = Rt.reshape(3, 3)
            q, pe, re = ik(self, r, pos, Rt)
            if pe > 0.01 or re > 0.05:
                raise ActionRefused("unreachable", f"no joint solution within 1 cm / 3 deg (best {pe * 1000:.0f} mm,"
                                                   f" {math.degrees(re):.1f} deg)")
            goal = {}
            for j, qq in zip(r.arm_joints, q):
                k = next(k for k, jj in r.act_joint.items() if jj == j)
                ai = r.actuator_ids[k]
                goal[k] = float(qq) * (1.0 if r.torque else float(m.actuator_gear[ai][0]) or 1.0)
            return JointMove(r, goal, self._n(dur), self._n(0.3))
        if kind == "walk":
            if not r.legs:
                raise ActionRefused("no_legs", f"{r.pack.name} has no leg layout")
            return Walk(r, self, float(a.get("forward", 0.6)), float(a.get("turn", 0.0)), self._n(dur))
        if kind == "fly":
            if r.free_dof is None:
                raise ActionRefused("no_base", f"{r.pack.name} has no free base")
            return Fly(r, self, np.asarray(a["position"], float), math.radians(float(a.get("yaw_deg", 0.0))),
                       self._n(dur))
        if kind == "policy":
            from .policies import PolicyRun, match, policy

            try:
                spec = policy(str(a.get("policy") or ""))
            except KeyError as exc:
                raise ActionRefused("unknown_policy", str(exc)) from exc
            diffs = match(spec, self, r)
            if diffs:
                raise ActionRefused("mismatch", f"{spec.name} does not match {r.id}: {diffs}")
            cmd = np.asarray(a.get("command") or [0.5, 0.0, 0.0], float)
            return PolicyRun(r, self, spec, cmd, self._n(dur))
        if kind == "base":
            goal = {}
            for key, name in zip(("x", "y", "yaw_deg"), r.pack.base_joints):
                if a.get(key) is not None and name in r.base_index:
                    goal[r.base_index[name]] = math.radians(float(a[key])) if key == "yaw_deg" else float(a[key])
            if not goal:
                raise ActionRefused("empty", "give x, y or yaw_deg")
            return JointMove(r, goal, self._n(dur), self._n(0.3))
        raise ActionRefused("unsupported", f"unknown action kind {kind!r}")

    # ------------------------------------------------------------------ observation
    def entity_of_body(self, b: int) -> str | None:
        while b > 0:
            name = self.model.body(b).name
            if name:
                top = name.split("/")[0]
                if any(e.id == top for e in self.stage.entities):
                    return top
            b = int(self.model.body_parentid[b])
        return None

    def state(self) -> dict[str, Any]:
        m, d = self.model, self.data
        objects = {}
        for e in self.stage.entities:
            if (e.shape or e.mesh) and e.body is not None and e.body.dynamic:
                b = m.body(e.id).id
                objects[e.id] = {"position": np.round(d.xpos[b], 4).tolist(),
                                 "quat_wxyz": np.round(d.xquat[b], 4).tolist(),
                                 "speed": round(float(np.linalg.norm(d.cvel[b][3:])), 4)}
        out = {"generation": self.generation, "revision": self.stage.revision, "step": self.step,
               "time_s": round(float(d.time), 4), "robots": {k: r.state(m, d) for k, r in self.robots.items()},
               "objects": objects}
        if self.humans is not None:
            out["humans"] = self.humans.state()
        return out

    def _renderer(self, w: int, h: int) -> mujoco.Renderer:
        key = (w, h)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height=h, width=w)
        return self._renderers[key]

    def capture(self, cameras: list[str] | None = None, *, depth: bool = True, segmentation: bool = True,
                size: tuple[int, int] | None = None) -> dict[str, dict[str, Any]]:
        """Per camera: rgb, depth (metres along the optical axis), seg (entity index per pixel, see `labels`), K and
        T_world_cam (OpenCV frame: x right, y down, z forward), all from the same step."""
        m, d = self.model, self.data
        names = cameras or list(self.cameras)
        out: dict[str, dict[str, Any]] = {}
        labels = ["background"] + [e.id for e in self.stage.entities]
        index = {name: i for i, name in enumerate(labels)}
        geom_entity = np.array([index.get(self.entity_of_body(int(m.geom_bodyid[g])) or "", 0) for g in range(m.ngeom)])
        for cam in names:
            if cam not in self.cameras:
                raise KeyError(f"no camera {cam!r}; cameras: {list(self.cameras)}")
            info = self.cameras[cam]
            w, h = size or (info["width"], info["height"])
            r = self._renderer(w, h)
            cid = m.camera(cam).id
            r.update_scene(d, camera=cam)
            rec: dict[str, Any] = {"rgb": r.render().copy()}
            if depth and info.get("depth", True):
                r.enable_depth_rendering()
                r.update_scene(d, camera=cam)
                rec["depth"] = r.render().astype(np.float32)
                r.disable_depth_rendering()
            if segmentation:
                r.enable_segmentation_rendering()
                r.update_scene(d, camera=cam)
                seg = r.render()
                r.disable_segmentation_rendering()
                ids, types = seg[..., 0], seg[..., 1]
                ent = np.zeros(ids.shape, np.int32)
                ok = (types == int(mujoco.mjtObj.mjOBJ_GEOM)) & (ids >= 0)
                ent[ok] = geom_entity[ids[ok]]
                rec["seg"] = ent
            f = 0.5 * h / math.tan(math.radians(float(m.cam_fovy[cid])) / 2)
            rec["K"] = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])
            T = np.eye(4)
            T[:3, :3] = d.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
            T[:3, 3] = d.cam_xpos[cid]
            rec["T_world_cam"] = T
            out[cam] = rec
        out["_labels"] = {"labels": labels, "step": self.step, "generation": self.generation}
        return out

    def view(self, *, lookat: Any = (0, 0, 0.5), distance: float = 3.0, azimuth: float = 135.0,
             elevation: float = -25.0, size: tuple[int, int] = (640, 480)) -> np.ndarray:
        """A free viewpoint (for people: the Studio's orbiting view)."""
        cam = mujoco.MjvCamera()
        cam.lookat[:] = lookat
        cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
        r = self._renderer(*size)
        r.update_scene(self.data, camera=cam)
        return r.render().copy()

    def overlaps(self, depth: float = 1e-3, limit: int = 16) -> list[dict[str, Any]]:
        """Contacts penetrating deeper than `depth` metres between different entities (or within an entity that
        is not a robot), deepest first: a diagnosis of the start state, nothing is moved."""
        m, d = self.model, self.data
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist >= -depth:
                continue
            a = self.entity_of_body(int(m.geom_bodyid[c.geom1])) or "world"
            b = self.entity_of_body(int(m.geom_bodyid[c.geom2])) or "world"
            if a == b and a in self.robots:
                continue
            out.append({"a": a, "b": b, "depth_mm": round(-float(c.dist) * 1000, 1),
                        "at": np.round(c.pos, 3).tolist()})
        return sorted(out, key=lambda x: -x["depth_mm"])[:limit]


__all__ = ["ActionRefused", "Robot", "World"]
