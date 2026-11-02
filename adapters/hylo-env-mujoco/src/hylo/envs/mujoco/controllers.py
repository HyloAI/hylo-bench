"""Controllers of the stage world (RFC-0008 part 3): each turns one action into actuator commands, step by step.

A robot's actuators keep a target (`Robot.target`, one value per actuator, in ctrl units for position servos and
in joint units for torque motors, where the world closes a PD loop with the pack's gains). A controller moves those
targets over time and reports when it is done and what it measured:

```text
  JointMove    linear interpolation from the current targets to new joint targets over a duration, then a short
               settle; measured: the largest joint error at the end
  GripperMove  the gripper actuator to its open or closed value (or a fraction); measured: whether the fingers
               stopped short of the command (something is held)
  TcpMove      damped least squares IK of the arm joints for a tool-point pose, then a JointMove
  Walk         an open-loop trot for quadrupeds: diagonal leg pairs in antiphase, thigh swing proportional to the
               commanded forward speed (minus / plus a turn term on the left / right legs), calf flexion lifts the
               swing foot; measured: base displacement, heading change, whether the robot tipped over
  Fly          drones: position PD -> desired force -> attitude PD -> body wrench -> actuator commands through the
               allocation matrix measured from the model (works for body-thrust and per-rotor drones alike)
  BaseMove     mobile bases driven through x, y, yaw joints: interpolation like JointMove
  Hold         keep every target for a while
```
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np


class Controller:
    robot: Any
    done: bool = False
    steps: int = 0

    def update(self, world: Any) -> None:
        raise NotImplementedError

    def measured(self, world: Any) -> dict[str, Any]:
        return {}


class Hold(Controller):
    def __init__(self, robot: Any, n: int) -> None:
        self.robot, self.n = robot, max(1, n)

    def update(self, world: Any) -> None:
        self.steps += 1
        self.done = self.steps >= self.n


class JointMove(Controller):
    def __init__(self, robot: Any, goal: dict[int, float], n: int, settle: int) -> None:
        self.robot, self.goal, self.n, self.settle = robot, goal, max(1, n), settle
        self.start = {i: float(robot.target[i]) for i in goal}

    def update(self, world: Any) -> None:
        self.steps += 1
        a = min(1.0, self.steps / self.n)
        for i, g in self.goal.items():
            self.robot.target[i] = self.start[i] + a * (g - self.start[i])
        self.done = self.steps >= self.n + self.settle

    def measured(self, world: Any) -> dict[str, Any]:
        err = {}
        for i, g in self.goal.items():
            q = self.robot.actuator_position(world.data, i)
            if q is not None:
                err[self.robot.actuators[i]] = abs(q - self.robot.target_joint(i, g))
        worst = max(err.values()) if err else 0.0
        return {"max_joint_error": round(float(worst), 4), "worst_joint": max(err, key=err.get) if err else None}


class GripperMove(Controller):
    def __init__(self, robot: Any, index: int, value: float, n: int) -> None:
        self.robot, self.index, self.value, self.n = robot, index, value, max(1, n)
        self.start = float(robot.target[index])
        self.closing = robot.pack.gripper is not None and abs(value - robot.pack.gripper.closed) < abs(
            value - robot.pack.gripper.open)

    def update(self, world: Any) -> None:
        self.steps += 1
        a = min(1.0, self.steps / (self.n * 0.5))
        self.robot.target[self.index] = self.start + a * (self.value - self.start)
        self.done = self.steps >= self.n

    def measured(self, world: Any) -> dict[str, Any]:
        out = {"command": round(self.value, 4)}
        frac = self.robot.gripper_opening(world.model, world.data)
        if frac is not None:
            out["opening"] = round(frac, 3)
            out["holding"] = bool(self.closing and frac > 0.04)
        return out


class Walk(Controller):
    """Open-loop trot. `vx` in [-1, 1] (fraction of the gait's stride), `turn` in [-1, 1]."""

    def __init__(self, robot: Any, world: Any, vx: float, turn: float, n: int, freq: float = 2.0,
                 thigh_amp: float = 0.25, calf_amp: float = 0.45, sign: float = 1.0) -> None:
        self.robot, self.vx, self.turn, self.n = robot, vx, turn, max(1, n)
        self.freq, self.ta, self.ca, self.sign = freq, thigh_amp, calf_amp, sign
        self.dt = world.model.opt.timestep
        self.home = {i: float(v) for i, v in robot.home_target.items()}
        self.start_pose = robot.base_pose(world.data)

    def update(self, world: Any) -> None:
        self.steps += 1
        t = self.steps * self.dt
        ramp = min(1.0, t / 0.5) * min(1.0, max(0.0, (self.n - self.steps) * self.dt / 0.3))
        turn = self.turn
        if abs(self.turn) < 1e-6:                      # walking straight: hold the start heading
            _, yaw = self.robot.base_pose(world.data)
            err = (self.start_pose[1] - yaw + math.pi) % (2 * math.pi) - math.pi
            turn = float(np.clip(1.5 * err, -0.4, 0.4))
        for k, leg in enumerate(self.robot.legs):
            phase = 2 * math.pi * self.freq * t + (0.0 if leg["name"] in ("FL", "RR", "HR") else math.pi)
            side = 1.0 if leg["name"][-1] == "L" else -1.0
            speed = float(np.clip(self.vx - side * turn, -1.0, 1.0))
            swing = max(0.0, math.sin(phase))
            th, ca = leg["thigh"], leg["calf"]
            self.robot.target[th] = self.home[th] + ramp * self.sign * self.ta * speed * math.cos(phase)
            self.robot.target[ca] = self.home[ca] - ramp * self.ca * swing * self.robot.calf_sign
            self.robot.target[leg["hip"]] = self.home[leg["hip"]]
        self.done = self.steps >= self.n

    def measured(self, world: Any) -> dict[str, Any]:
        p0, y0 = self.start_pose
        p1, y1 = self.robot.base_pose(world.data)
        d = p1 - p0
        fwd = np.array([math.cos(y0), math.sin(y0)])
        up = self.robot.base_up(world.data)
        return {"moved_m": round(float(np.linalg.norm(d[:2])), 3), "forward_m": round(float(d[:2] @ fwd), 3),
                "turned_deg": round(math.degrees((y1 - y0 + math.pi) % (2 * math.pi) - math.pi), 1),
                "upright": bool(up > 0.7), "base_height": round(float(p1[2]), 3)}


class Fly(Controller):
    def __init__(self, robot: Any, world: Any, target: np.ndarray, yaw: float, n: int) -> None:
        self.robot, self.goal, self.yaw, self.n = robot, np.asarray(target, float), yaw, max(1, n)
        m = world.model
        self.mass = float(m.body_subtreemass[robot.root_body])
        self.inertia = np.maximum(m.body_inertia[robot.root_body], 1e-6)
        self.g = float(-m.opt.gravity[2])
        self.alloc = allocation(world.model, world.data, robot)
        lo, hi = m.actuator_ctrlrange[robot.actuator_ids].T
        self.lo, self.hi = lo, hi
        # gains from the drone's authority: the angular acceleration its moment actuators can give (weakest axis)
        span = np.maximum(np.abs(lo), np.abs(hi))
        alpha = float(np.min(np.abs(self.alloc[1:]) @ span / self.inertia))
        self.kR = min(180.0, 0.6 * alpha / math.radians(10.0))                # 10 deg error uses 60% of it
        self.kw = 2.0 * math.sqrt(self.kR)
        w_att = math.sqrt(self.kR)
        w_pos = min(1.8, w_att / 3.0)
        self.kp, self.kd = w_pos ** 2, 2.0 * w_pos
        self.tilt = math.radians(min(20.0, max(3.0, math.degrees(0.15 * alpha / max(self.kR, 1e-9)) * 40)))

    def update(self, world: Any) -> None:
        self.steps += 1
        d = world.data
        p, R = self.robot.root_pose(d)
        v = d.qvel[self.robot.free_dof:self.robot.free_dof + 3]
        w = d.qvel[self.robot.free_dof + 3:self.robot.free_dof + 6]            # body frame
        e = self.goal - p
        e = e * min(1.0, 1.5 / max(np.linalg.norm(e), 1e-9))                  # at most 1.5 m of error at once
        acc = self.kp * e - self.kd * v
        acc[2] = 4.0 * e[2] - 3.0 * v[2]                                     # height does not need attitude
        horiz = np.linalg.norm(acc[:2])
        cap = math.tan(self.tilt) * self.g                                   # at most this much tilt
        if horiz > cap:
            acc[:2] *= cap / horiz
        f = self.mass * (acc + np.array([0.0, 0.0, self.g]))
        zb = f / max(np.linalg.norm(f), 1e-9)
        xc = np.array([math.cos(self.yaw), math.sin(self.yaw), 0.0])
        yb = np.cross(zb, xc)
        yb /= max(np.linalg.norm(yb), 1e-9)
        Rd = np.column_stack([np.cross(yb, zb), yb, zb])
        eR = 0.5 * _vee(Rd.T @ R - R.T @ Rd)
        tau = self.inertia * (-self.kR * eR - self.kw * w)
        thrust = float(f @ R[:, 2])
        want = np.array([thrust, *tau])
        ctrl, *_ = np.linalg.lstsq(self.alloc, want, rcond=None)
        d.ctrl[self.robot.actuator_ids] = np.clip(ctrl, self.lo, self.hi)
        self.done = self.steps >= self.n

    def measured(self, world: Any) -> dict[str, Any]:
        p, _ = self.robot.root_pose(world.data)
        return {"position": np.round(p, 3).tolist(), "error_m": round(float(np.linalg.norm(self.goal - p)), 3)}


def allocation(m: Any, d: Any, robot: Any) -> np.ndarray:
    """(4, nu) body-frame thrust (z) and torques produced per unit command of each actuator, measured by applying
    one actuator at a time to a copy of the state."""
    d2 = mujoco.MjData(m)
    d2.qpos[:] = d.qpos
    out = np.zeros((4, len(robot.actuator_ids)))
    _, R = robot.root_pose(d)
    for k, a in enumerate(robot.actuator_ids):
        d2.ctrl[:] = 0.0
        u = 0.5 * float(m.actuator_ctrlrange[a][1]) if m.actuator_ctrllimited[a] and m.actuator_ctrlrange[a][1] > 0 \
            else 1.0                                   # a command inside the range (a clipped probe would mislead)
        d2.ctrl[a] = u
        mujoco.mj_forward(m, d2)
        f = d2.qfrc_actuator[robot.free_dof:robot.free_dof + 6]
        out[0, k] = (R.T @ f[:3])[2] / u
        out[1:, k] = f[3:] / u
    return out


def _vee(M: np.ndarray) -> np.ndarray:
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def ik(world: Any, robot: Any, target_pos: np.ndarray, target_R: np.ndarray | None, iters: int = 200,
       tol: float = 2e-3) -> tuple[np.ndarray, float, float]:
    """Joint values of the arm joints putting the tool point at a pose (orientation optional): damped least squares
    from the current joints, within the joint limits. Returns (q, position error, rotation error)."""
    m = world.model
    d = mujoco.MjData(m)
    d.qpos[:] = world.data.qpos
    joints = robot.arm_joints
    qadr = np.array([m.jnt_qposadr[j] for j in joints])
    dadr = np.array([m.jnt_dofadr[j] for j in joints])
    lo, hi = m.jnt_range[joints].T
    limited = m.jnt_limited[joints].astype(bool)
    pos_err = rot_err = np.inf
    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        p, R, jacp, jacr = robot.tcp_jacobian(m, d)
        e = target_pos - p
        pos_err = float(np.linalg.norm(e))
        if target_R is not None:
            er = 0.5 * (np.cross(R[:, 0], target_R[:, 0]) + np.cross(R[:, 1], target_R[:, 1])
                        + np.cross(R[:, 2], target_R[:, 2]))
            rot_err = float(np.linalg.norm(er))
            J = np.vstack([jacp[:, dadr], 0.5 * jacr[:, dadr]])
            err = np.concatenate([e, 0.5 * er])
        else:
            rot_err = 0.0
            J, err = jacp[:, dadr], e
        if pos_err < tol and rot_err < 0.02:
            break
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(len(err)), err)
        dq = np.clip(dq, -0.2, 0.2)
        q = d.qpos[qadr] + dq
        q = np.where(limited, np.clip(q, lo, hi), q)
        d.qpos[qadr] = q
    return d.qpos[qadr].copy(), pos_err, rot_err


__all__ = ["Controller", "Fly", "GripperMove", "Hold", "JointMove", "Walk", "allocation", "ik"]
