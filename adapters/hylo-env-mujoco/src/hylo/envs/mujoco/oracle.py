"""`oracle.panda_table`: a privileged scripted agent for `mujoco/manip-v0` (an upper bound, not a method).

It reads object poses and the goal from `extra["privileged"]` (the env must run with `privileged_obs: true`) and
solves each task with damped-least-squares IK on the arm model named by `RobotSpec` (loaded with MuJoCo) and
straight-line TCP motions with minimum-jerk timing. It acts only through ordinary actions (joint targets and hand
commands, one chunk per motion segment), so it proves every task is solvable by the robot as simulated.

When an attempt ends without success (a slipped grasp, a box that turned while pushed), the next `act` plans a
new attempt from the current state: only what is still undone (remaining buttons, unsorted blocks, upper levels).
"""

from __future__ import annotations

import math
from typing import Any, Iterator

import numpy as np

from hylo.core.agent import AgentBase, AgentSpec, EpisodeContext
from hylo.core.robot import ArmSpec
from hylo.core.types import ActionChunk, ActionSemantics, Observation

SAFE_Z = 0.20                    # travel height of the TCP (m)
QD_MAX = 1.2                     # joint speed limit of planned motions (rad/s)
DOWN = np.array([0.0, 0.0, -1.0])


def topdown(psi: float) -> np.ndarray:
    """TCP rotation for a top-down approach with the closing axis at yaw `psi` (+x down, +y closing)."""
    y = np.array([math.cos(psi), math.sin(psi), 0.0])
    return np.stack([DOWN, y, np.cross(DOWN, y)], axis=1)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _rot_err(R: np.ndarray, Rg: np.ndarray) -> np.ndarray:
    """Rotation vector taking R to Rg (world frame)."""
    import mujoco

    qa, qb, qd = np.empty(4), np.empty(4), np.empty(3)
    mujoco.mju_mat2Quat(qa, R.ravel())
    mujoco.mju_mat2Quat(qb, Rg.ravel())
    mujoco.mju_subQuat(qd, qb, qa)            # qb * inv(qa), expressed in the frame of qa
    return R @ qd


class Kinematics:
    """Forward and inverse kinematics of one arm from its ArmSpec (MJCF or URDF loaded by MuJoCo)."""

    def __init__(self, arm: ArmSpec) -> None:
        import mujoco

        self._mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(arm.model)
        self.d = mujoco.MjData(self.m)
        jids = [self.m.joint(j).id for j in arm.joints]
        self.qadr = np.array([self.m.jnt_qposadr[j] for j in jids])
        self.vadr = np.array([self.m.jnt_dofadr[j] for j in jids])
        self.body = self.m.body(arm.ee_frame).id
        self.base = np.asarray(arm.base_in_world, dtype=float)
        self.base_inv = np.linalg.inv(self.base)
        self.tcp_in_ee = np.asarray(arm.hand.tcp_in_ee, dtype=float)
        self.lo, self.hi = np.asarray(arm.lower), np.asarray(arm.upper)
        self._jp, self._jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))

    def _set(self, q: np.ndarray) -> None:
        self.d.qpos[self.qadr] = q
        self._mj.mj_kinematics(self.m, self.d)
        self._mj.mj_comPos(self.m, self.d)

    def fk(self, q: np.ndarray) -> np.ndarray:
        self._set(q)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.d.xmat[self.body].reshape(3, 3), self.d.xpos[self.body]
        return self.base @ T @ self.tcp_in_ee

    def ik(self, T_goal: np.ndarray, q0: np.ndarray, q_rest: np.ndarray | None = None, iters: int = 60,
           tol: tuple[float, float] = (5e-4, 5e-3)) -> tuple[np.ndarray, float, float]:
        """Damped least squares from q0 towards the TCP pose T_goal (world), with a pull towards q_rest in the
        null space. Returns (q, position error m, rotation error rad)."""
        mj = self._mj
        Tg = self.base_inv @ T_goal
        q = np.clip(np.asarray(q0, dtype=float).copy(), self.lo, self.hi)
        rest = q if q_rest is None else np.asarray(q_rest, dtype=float)
        pe = re = np.inf
        for _ in range(iters):
            self._set(q)
            R = self.d.xmat[self.body].reshape(3, 3)
            p = self.d.xpos[self.body] + R @ self.tcp_in_ee[:3, 3]
            Rt = R @ self.tcp_in_ee[:3, :3]
            e_p, e_r = Tg[:3, 3] - p, _rot_err(Rt, Tg[:3, :3])
            pe, re = float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r))
            if pe < tol[0] and re < tol[1]:
                break
            mj.mj_jac(self.m, self.d, self._jp, self._jr, p, self.body)
            J = np.vstack([self._jp[:, self.vadr], self._jr[:, self.vadr]])
            e = np.concatenate([e_p, 0.5 * e_r])
            JJt = J @ J.T + 1e-4 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            null = np.eye(len(q)) - J.T @ np.linalg.solve(JJt, J)
            dq += null @ (0.1 * (rest - q))
            step = float(np.max(np.abs(dq)))
            if step > 0.2:
                dq *= 0.2 / step
            q = np.clip(q + dq, self.lo, self.hi)
        return q, pe, re


def _minjerk(n: int) -> np.ndarray:
    s = np.arange(1, n + 1) / n
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class PlanFailed(RuntimeError):
    pass


class PandaOracle(AgentBase):
    """Params: speed (TCP travel speed, m/s), approach_speed (m/s near objects), settle (steps held after each
    motion)."""

    def __init__(self, speed: float = 0.3, approach_speed: float = 0.12, settle: int = 4) -> None:
        self.speed, self.approach_speed, self.settle = float(speed), float(approach_speed), int(settle)

    def spec(self) -> AgentSpec:
        return AgentSpec(name="oracle.panda_table", version="0.1", required_state=("q.arm", "hand.arm"),
                         action_semantics=ActionSemantics(control_mode="joint_pos", frame="base", gripper="0_close_1_open"),
                         action_dim=8)

    def reset(self, ctx: EpisodeContext) -> None:
        self.ctx = ctx
        robot = ctx.env_spec.robot
        if robot is None:
            raise ValueError("oracle.panda_table needs an environment with a RobotSpec (mujoco.panda_table)")
        arm = robot.arms[0]
        key = arm.model_dump_json()
        if getattr(self, "_kin_key", None) != key:   # one model per agent: evaluation workers are threads
            self.kin, self._kin_key = Kinematics(arm), key
        self.dt = 1.0 / ctx.env_spec.control_hz
        self.home = np.asarray(arm.home)
        self._gen: Iterator[np.ndarray] | None = None
        self._obs: Observation | None = None
        self._cmd: np.ndarray | None = None          # last commanded joints and hand
        self.attempts = 0
        # closing-axis yaw of a comfortable top-down posture facing +x; grasp yaws are chosen near it
        self.q_nom = np.array([0.0, 0.25, 0.0, -2.1, 0.0, 2.35, 0.785])
        R = self.kin.fk(self.q_nom)[:3, :3]
        self.psi0 = math.atan2(R[1, 1], R[0, 1])

    # -- agent loop ---------------------------------------------------------------------------------------------
    def act(self, observation: Observation) -> ActionChunk:
        priv = observation.extra.get("privileged")
        if not isinstance(priv, dict):
            raise ValueError("oracle.panda_table reads extra['privileged']: run mujoco.panda_table with privileged_obs: true")
        self._obs = observation
        for _ in range(2):
            if self._gen is None:
                self.attempts += 1
                self._gen = self._attempt(priv["task"]) if self.attempts <= 6 else iter(())
            try:
                actions = next(self._gen)
                break
            except (StopIteration, PlanFailed) as exc:
                if isinstance(exc, PlanFailed):
                    self.ctx.emit("agent.oracle.plan_failed", {"reason": str(exc), "attempt": self.attempts})
                self._gen = None
        else:
            actions = self._hold(5)                   # nothing left to do: wait for the task predicate
        self._cmd = actions[-1].copy()
        return ActionChunk(actions=actions, provenance={"oracle": priv["task"], "attempt": self.attempts})

    # -- state --------------------------------------------------------------------------------------------------
    @property
    def priv(self) -> dict[str, Any]:
        return self._obs.extra["privileged"]  # type: ignore[union-attr]

    def _q(self) -> np.ndarray:
        q = np.asarray(self._obs.state["q.arm"], dtype=float)  # type: ignore[union-attr]
        if self._cmd is not None and np.max(np.abs(self._cmd[:7] - q)) < 0.08:
            return self._cmd[:7].copy()               # continue from the last target (no jump from tracking lag)
        return q

    def _hand_cmd(self) -> float:
        return float(self._cmd[7]) if self._cmd is not None else float(self._obs.state["hand.arm"][0])  # type: ignore[union-attr]

    def obj(self, name: str) -> dict[str, Any]:
        return self.priv["objects"][name]

    def _hold(self, n: int) -> np.ndarray:
        return np.tile(np.concatenate([self._q(), [self._hand_cmd()]]), (n, 1))

    # -- motions (each returns the chunk for one segment) --------------------------------------------------------
    def _rest(self, xy: np.ndarray) -> np.ndarray:
        rest = self.q_nom.copy()
        b = self.kin.base[:3, 3]
        rest[0] = math.atan2(xy[1] - b[1], xy[0] - b[0])
        return rest

    def line(self, pos: np.ndarray, R: np.ndarray, speed: float | None = None, hand: float | None = None,
             settle: int | None = None) -> np.ndarray:
        """Straight TCP line (position linear, rotation about one axis) with minimum-jerk timing, slowed down
        where the joints would move faster than QD_MAX."""
        q_start = self._q()
        T0 = self.kin.fk(q_start)
        p0, R0 = T0[:3, 3], T0[:3, :3]
        rv = _rot_err(R0, R)
        dist, ang = float(np.linalg.norm(pos - p0)), float(np.linalg.norm(rv))
        v = speed or self.speed
        n = max(2, math.ceil(max(dist / v, ang / 1.2) * 1.4 / self.dt))
        rest = self._rest(pos[:2])
        h = self._hand_cmd() if hand is None else hand
        for _ in range(3):
            q, qs = q_start, []
            for s in _minjerk(n):
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = _axis_angle(rv * s) @ R0, p0 + s * (pos - p0)
                q, pe, re = self.kin.ik(T, q, rest)
                if pe > 0.01 or re > 0.05:
                    raise PlanFailed(f"IK failed near {np.round(T[:3, 3], 3).tolist()} ({pe:.3f} m, {re:.3f} rad)")
                qs.append(q)
            ratio = float(np.max(np.abs(np.diff(np.vstack([q_start, *qs]), axis=0)))) / (QD_MAX * self.dt)
            if ratio <= 1.0:
                break
            n = math.ceil(n * ratio * 1.1)
        out = [np.concatenate([q, [h]]) for q in qs]
        out += [out[-1]] * (self.settle if settle is None else settle)
        return np.asarray(out)

    def solve(self, T: np.ndarray, rest: np.ndarray) -> np.ndarray:
        """IK from several seed postures (current, nominal, folded, stretched); the best that converges."""
        seeds = [self._q(), rest]
        for j2, j4, j6 in ((-0.3, -2.8, 2.5), (0.6, -1.6, 2.2), (0.0, -2.4, 2.0)):
            s = rest.copy()
            s[1], s[3], s[5] = j2, j4, j6
            seeds.append(s)
        best = None
        for seed in seeds:
            q, pe, re = self.kin.ik(T, seed, rest, iters=300)
            if best is None or pe + 0.1 * re < best[1] + 0.1 * best[2]:
                best = (q, pe, re)
            if pe < 0.002 and re < 0.01:
                break
        q, pe, re = best  # type: ignore[misc]
        if pe > 0.005 or re > 0.03:
            raise PlanFailed(f"IK failed at {np.round(T[:3, 3], 3).tolist()} ({pe:.3f} m, {re:.3f} rad)")
        return q

    def joint_move(self, pos: np.ndarray, R: np.ndarray, hand: float | None = None) -> np.ndarray:
        q0 = self._q()
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, pos
        q1 = self.solve(T, self._rest(pos[:2]))
        n = max(2, math.ceil(float(np.max(np.abs(q1 - q0))) / (0.75 * QD_MAX) * 1.4 / self.dt))
        h = self._hand_cmd() if hand is None else hand
        out = [np.concatenate([q0 + s * (q1 - q0), [h]]) for s in _minjerk(n)]
        return np.asarray(out + [out[-1]] * self.settle)

    def gripper(self, cmd: float, steps: int = 12) -> np.ndarray:
        q, h0 = self._q(), self._hand_cmd()
        ramp = np.clip(h0 + (cmd - h0) * np.arange(1, steps + 1) / max(1, steps // 3), min(h0, cmd), max(h0, cmd))
        return np.asarray([np.concatenate([q, [h]]) for h in ramp])

    def tcp(self) -> np.ndarray:
        return self.kin.fk(self._q())

    # -- skills (generators of segments) -------------------------------------------------------------------------
    def grasp_yaw(self, name: str) -> float:
        o = self.obj(name)
        ref = self._ref_yaw(np.asarray(o["pos"][:2]))
        if o["kind"] in ("cyl",):
            return ref
        cands = [o["yaw"] + k * math.pi / 2 for k in range(4)]
        return min(cands, key=lambda a: abs(_wrap(a - ref)))

    def _ref_yaw(self, xy: np.ndarray) -> float:
        b = self.kin.base[:3, 3]
        return self.psi0 + math.atan2(xy[1] - b[1], xy[0] - b[0])

    def _near_yaw(self, psi: float, xy: np.ndarray, period: float = math.pi) -> float:
        ref = self._ref_yaw(xy)
        return psi + period * round(_wrap(ref - psi) / period)

    def travel(self, xy: np.ndarray, R: np.ndarray, z: float = SAFE_Z, hand: float | None = None) -> Iterator[np.ndarray]:
        cur = self.tcp()
        if cur[2, 3] < z - 0.03:                      # leave upwards first
            yield self.line(np.array([cur[0, 3], cur[1, 3], z]), cur[:3, :3], speed=self.approach_speed * 1.5, hand=hand)
        goal = np.array([xy[0], xy[1], z])
        if np.linalg.norm(self._q() - self.home) < 0.05:
            yield self.joint_move(goal, R, hand=hand)
            return
        try:
            chunk = self.line(goal, R, hand=hand)
        except PlanFailed:                            # the straight line leaves the reachable set: go joint-wise
            chunk = self.joint_move(goal, R, hand=hand)
        yield chunk

    def pick(self, name: str) -> Iterator[np.ndarray]:
        for _ in range(3):
            o = self.obj(name)
            p = np.asarray(o["pos"])
            psi = self.grasp_yaw(name)
            R = topdown(psi)
            top = p[2] + o["half_size"][2]
            z = float(np.clip(p[2], 0.018, top - 0.012))
            yield from self.travel(p[:2], R, hand=1.0)
            if self._hand_cmd() < 0.99:
                yield self.gripper(1.0, 8)
            yield self.line(np.array([p[0], p[1], z + 0.06]), R)
            yield self.line(np.array([p[0], p[1], z]), R, speed=self.approach_speed, settle=6)
            yield self.gripper(0.0, 14)
            yield self.line(np.array([p[0], p[1], SAFE_Z]), R, speed=self.approach_speed * 1.5)
            if 0.12 < float(self._obs.state["hand.arm"][0]) < 0.95 and self.obj(name)["pos"][2] > top + 0.05:  # type: ignore[union-attr]
                return
            yield self.gripper(1.0, 8)
        raise PlanFailed(f"could not grasp {name}")

    def place(self, name: str, xy: np.ndarray, z_obj: float, psi: float | None = None, release: float = 1.0,
              drop: float = 0.004) -> Iterator[np.ndarray]:
        """Put the held object `name` with its centre at (xy, z_obj) and let go."""
        T = self.tcp()
        o = np.asarray(self.obj(name)["pos"])
        off = o - T[:3, 3]                            # where the object sits relative to the TCP
        if psi is None:                               # free yaw: the comfortable one at the goal (the wrist
            psi = self._near_yaw(math.atan2(T[1, 1], T[0, 1]), xy)  # follows the base instead of spinning)
        R = topdown(psi)
        dpsi = psi - math.atan2(T[1, 1], T[0, 1])     # the held object turns with the hand about the vertical
        off = np.array([*_rotate(off[:2], dpsi), off[2]])
        goal = np.array([xy[0], xy[1], z_obj + drop]) - off
        yield from self.travel(goal[:2], R)
        yield self.line(np.array([goal[0], goal[1], goal[2] + 0.05]), R)
        yield self.line(goal, R, speed=self.approach_speed, settle=6)
        yield self.gripper(release, 10)
        yield self._hold(4)
        yield self.line(np.array([goal[0], goal[1], SAFE_Z]), R, speed=self.approach_speed * 1.5)

    # -- tasks --------------------------------------------------------------------------------------------------
    def _attempt(self, task: str) -> Iterator[np.ndarray]:
        return getattr(self, f"_task_{task}")()

    def _task_pick_place(self) -> Iterator[np.ndarray]:
        g = self.priv["goal"]
        obj, tgt = g["object"], g["target"]
        yield from self.pick(obj)
        t = self.obj(tgt)
        yield from self.place(obj, np.asarray(t["pos"][:2]), self._top(tgt) + self.obj(obj)["half_size"][2])

    def _task_put_in_bowl(self) -> Iterator[np.ndarray]:
        g = self.priv["goal"]
        obj, bowl = g["object"], g["bowl"]
        yield from self.pick(obj)
        b = self.obj(bowl)
        yield from self.place(obj, np.asarray(b["pos"][:2]), b["pos"][2] + 2 * b["half_size"][2] + self.obj(obj)["half_size"][2],
                              drop=0.02)

    def _task_stack(self) -> Iterator[np.ndarray]:
        order = self.priv["goal"]["order"]
        start = 1
        for lower, upper in zip(order, order[1:]):      # skip levels already in place
            pl, pu = np.asarray(self.obj(lower)["pos"]), np.asarray(self.obj(upper)["pos"])
            if np.linalg.norm(pu[:2] - pl[:2]) < 0.015 and abs(pu[2] - pl[2] - 0.045) < 0.006:
                start += 1
            else:
                break
        for i in range(start, len(order)):
            lower, upper = order[i - 1], order[i]
            yield from self.pick(upper)
            pl = np.asarray(self.obj(lower)["pos"])
            psi_l = self.obj(lower)["yaw"]              # line the block up with the one below
            psi = self._near_yaw(psi_l, pl[:2], math.pi / 2)
            yield from self.place(upper, pl[:2], pl[2] + 2 * self.obj(lower)["half_size"][2], psi=psi, drop=0.003)

    def _task_sort_by_colour(self) -> Iterator[np.ndarray]:
        assign = self.priv["goal"]["assign"]
        by_tray: dict[str, list[str]] = {}
        for obj, tray in assign.items():
            by_tray.setdefault(tray, []).append(obj)
        for tray, objs in by_tray.items():
            t = self.obj(tray)
            c = np.asarray(t["pos"][:2])
            slots = [c] if len(objs) == 1 else [c + [0.027, 0.0], c + [-0.027, 0.0]]
            for obj, slot in zip(objs, slots):
                if self._in_tray(obj, tray):
                    continue
                yield from self.pick(obj)
                psi = self._near_yaw(math.pi / 2, slot)   # close across the slot row, so opening misses the neighbour
                yield from self.place(obj, slot, self._top(tray) + self.obj(obj)["half_size"][2], psi=psi, release=0.75)

    def _in_tray(self, obj: str, tray: str) -> bool:
        p, t = np.asarray(self.obj(obj)["pos"]), np.asarray(self.obj(tray)["pos"])
        return bool(np.all(np.abs(p[:2] - t[:2]) < 0.045) and p[2] < 0.06)

    def _top(self, name: str) -> float:
        o = self.obj(name)
        kind = o["kind"]
        if kind == "plate":
            return o["pos"][2] + 2 * o["half_size"][2]
        if kind == "tray":
            return o["pos"][2] + 0.006
        return o["pos"][2] + o["half_size"][2]

    def _task_press_by_number(self) -> Iterator[np.ndarray]:
        seq = self.priv["goal"]["sequence"]
        done = len(self.priv["pressed"])
        for name in seq[done:]:
            b = self.obj(name)
            p = np.asarray(b["pos"])
            top = p[2] + 2 * b["half_size"][2]
            R = topdown(self._ref_yaw(p[:2]))
            if self._hand_cmd() > 0.01:
                yield self.gripper(0.0, 8)
            yield from self.travel(p[:2], R, z=top + 0.12, hand=0.0)
            yield self.line(np.array([p[0], p[1], top + 0.03]), R)
            yield self.line(np.array([p[0], p[1], top - 0.0]), R, speed=0.06, settle=6)
            yield self.line(np.array([p[0], p[1], top + 0.06]), R, speed=self.approach_speed)

    def _task_push_to_zone(self) -> Iterator[np.ndarray]:
        g = self.priv["goal"]
        box, zone = g["object"], g["zone"]
        for _ in range(4):
            b, z = self.obj(box), self.obj(zone)
            pb, pz = np.asarray(b["pos"]), np.asarray(z["pos"])
            d = pz[:2] - pb[:2]
            local = _rotate(d, -z["yaw"])
            if np.all(np.abs(local) < z["half_size"][0] - 0.04):
                return
            dist = float(np.linalg.norm(d))
            u = d / dist
            rel = _wrap(math.atan2(u[1], u[0]) - b["yaw"])
            ext = b["half_size"][0] * (abs(math.cos(rel)) + abs(math.sin(rel)))
            psi = self._near_yaw(math.atan2(u[1], u[0]) + math.pi / 2, pb[:2])
            R = topdown(psi)
            zp = 0.022
            pre = pb[:2] - u * (ext + 0.045)
            end = pb[:2] - u * (ext + 0.012) + u * dist
            yield from self.travel(pre, R, z=0.12, hand=1.0)
            if self._hand_cmd() < 0.99:
                yield self.gripper(1.0, 8)
            yield self.line(np.array([pre[0], pre[1], zp]), R, speed=self.approach_speed)
            yield self.line(np.array([end[0], end[1], zp]), R, speed=0.08, settle=8)
            back = end - u * 0.04
            yield self.line(np.array([back[0], back[1], zp]), R, speed=self.approach_speed)
            yield self.line(np.array([back[0], back[1], 0.12]), R, speed=self.approach_speed * 1.5)


def _rotate(v: np.ndarray, a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _axis_angle(rv: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(rv))
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


__all__ = ["Kinematics", "PandaOracle", "topdown"]
