"""`oracle.aloha_table`: a privileged scripted agent for `mujoco/bimanual-v0` (an upper bound, not a method).

It reads object poses and the goal from `extra["privileged"]` (env param `privileged_obs: true`) and moves both arms
with ordinary joint-position actions, so it proves each task is solvable by the two simulated arms:

- handover: the arm on the object's side picks the bar near one end and lifts it to the middle of the table, turned
  along y; the other arm grasps the free end from above, the first lets go and leaves, the second puts the bar in
  the zone;
- lift_box: both arms grasp the box's end handles from above at the same time, close together and lift along
  straight lines of identical timing, then hold;
- two_bins: each arm picks the block on its side and drops it in the bin on its side, both at once. Param
  `sequential: true` runs the same motions one arm after the other (the baseline the step limit rules out).
"""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np

from .aloha_table import BASES, LIFT_BODY, LIFT_FIN, LIFT_FIN_Z
from .multi_oracle import ArmScript, MultiArmOracle, Segment, topdown, yaw_of_closing
from .oracle import PlanFailed, _wrap

SAFE_Z = 0.16
HAND_AT = 0.06                          # handover: each gripper this far from the bar's centre along it
HANDOVER_Z = 0.16                       # bar centre height at the handover


def _base_yaw(arm: str) -> float:
    return 0.0 if arm == "left" else math.pi


def _rest(script: ArmScript, pos: np.ndarray) -> np.ndarray:
    b = np.asarray(BASES[script.name])
    r = script.q_nom.copy()
    r[0] = _wrap(math.atan2(pos[1] - b[1], pos[0] - b[0]) - _base_yaw(script.name))
    return r


class AlohaOracle(MultiArmOracle):
    """Params: speed (TCP travel speed, m/s), settle (steps held after each motion), sequential (two_bins: one arm
    after the other)."""

    NAME = "oracle.aloha_table"

    def __init__(self, speed: float = 0.25, settle: int = 4, sequential: bool = False) -> None:
        super().__init__(speed, settle)
        self.sequential = bool(sequential)

    def _scripts(self, robot) -> dict[str, ArmScript]:
        q_nom = np.array([0.0, 0.1, 0.5, 0.0, 1.0, 0.0])
        return {a.name: ArmScript(self, a, q_nom, _rest) for a in robot.arms}

    # -- grasp geometry ---------------------------------------------------------------------------------------------
    def ref_yaw(self, arm: str, xy: np.ndarray) -> float:
        """Closing-axis yaw of the arm's natural top-down posture over `xy` (across its reach direction)."""
        b = BASES[arm]
        return math.atan2(xy[1] - b[1], xy[0] - b[0]) + math.pi / 2

    def near(self, arm: str, psi: float, xy: np.ndarray, period: float = math.pi) -> float:
        ref = self.ref_yaw(arm, xy)
        return psi + period * round(_wrap(ref - psi) / period)

    def closing_yaw(self, arm: str, name: str, xy: np.ndarray) -> float:
        o = self.obj(name)
        R = np.asarray(o["R"])
        if o["kind"] == "bar":                       # across the bar: its local x
            return self.near(arm, math.atan2(R[1, 0], R[0, 0]), xy)
        if o["kind"] == "liftbox":                   # handles: close along the box's long axis
            return self.near(arm, math.atan2(R[1, 0], R[0, 0]), xy)
        return self.near(arm, o["yaw"], xy, math.pi / 2)

    # -- per-arm skills (generators of (arm, rows)) -----------------------------------------------------------------
    def pick(self, arm: str, name: str, along: float = 0.0, lift: float = SAFE_Z,
             psi: float | None = None) -> Iterator[tuple[str, np.ndarray]]:
        s = self.arms[arm]
        for _ in range(2):
            o = self.obj(name)
            R = np.asarray(o["R"])
            p = np.asarray(o["pos"]) + along * R[:, 1]
            if psi is None:
                psi = self.closing_yaw(arm, name, p[:2])
            Rt = topdown(psi)
            top = o["pos"][2] + o["half_size"][2]
            z = top - 0.012
            for rows in s.travel([p[0], p[1], z + 0.08], Rt, hand=1.0, lift_to=SAFE_Z):
                yield arm, rows
            yield arm, s.line([p[0], p[1], z], Rt, speed=0.1, settle=4)
            yield arm, s.gripper(0.0, 14)
            yield arm, s.line([p[0], p[1], lift], Rt, speed=0.15)
            h = float(s.hand_measured()[0])
            if 0.05 < h < 0.95 and self.obj(name)["pos"][2] > top + 0.03:
                return
            yield arm, s.gripper(1.0, 8)
        raise PlanFailed(f"{arm} could not grasp {name}")

    def place(self, arm: str, name: str, xy, z_obj: float, psi: float | None = None,
              drop: float = 0.004) -> Iterator[tuple[str, np.ndarray]]:
        s = self.arms[arm]
        T = s.tcp()
        off = np.asarray(self.obj(name)["pos"]) - T[:3, 3]
        psi0 = yaw_of_closing(T[:3, :3])
        if psi is None:
            psi = self.near(arm, psi0, np.asarray(xy))
        d = psi - psi0
        c, sn = math.cos(d), math.sin(d)
        off = np.array([c * off[0] - sn * off[1], sn * off[0] + c * off[1], off[2]])
        goal = np.array([xy[0], xy[1], z_obj + drop]) - off
        Rt = topdown(psi)
        for rows in s.travel([goal[0], goal[1], goal[2] + 0.06], Rt, lift_to=SAFE_Z):
            yield arm, rows
        yield arm, s.line(goal, Rt, speed=0.1, settle=4)
        yield arm, s.gripper(1.0, 10)
        yield arm, s.line([goal[0], goal[1], goal[2] + 0.08], Rt, speed=0.15, settle=0)

    def pick_place(self, arm: str, name: str, target: str) -> Iterator[tuple[str, np.ndarray]]:
        yield from self.pick(arm, name)
        t = self.obj(target)
        top = t["pos"][2] + 0.006
        yield from self.place(arm, name, np.asarray(t["pos"][:2]), top + self.obj(name)["half_size"][2], drop=0.01)
        yield arm, self.arms[arm].go_home()

    # -- tasks ------------------------------------------------------------------------------------------------------
    def _task_two_bins(self) -> Iterator[Segment]:
        assign = self.priv["goal"]["assign"]
        jobs = {}
        for blk, b in assign.items():
            arm = "left" if self.obj(blk)["pos"][0] < 0 else "right"
            if not self._in_bin(blk, b):
                jobs[arm] = self.pick_place(arm, blk, b)
        run = self.in_turn if self.sequential else self.together
        yield from run(*[jobs[a] for a in ("left", "right") if a in jobs])

    def _in_bin(self, blk: str, b: str) -> bool:
        p, c = np.asarray(self.obj(blk)["pos"]), np.asarray(self.obj(b)["pos"])
        return bool(np.all(np.abs(p[:2] - c[:2]) < 0.05) and p[2] < 0.06)

    def _task_lift_box(self) -> Iterator[Segment]:
        g = self.priv["goal"]
        o = self.obj(g["object"])
        c, R = np.asarray(o["pos"]), np.asarray(o["R"])
        fx = LIFT_BODY[0] + LIFT_FIN[0]
        fins = [c + R @ np.array([sx * fx, 0.0, LIFT_FIN_Z]) for sx in (-1.0, 1.0)]
        fins.sort(key=lambda p: p[0])
        top = LIFT_FIN[2] - 0.016
        goals = {}
        for arm, f in zip(("left", "right"), fins):
            psi = self.closing_yaw(arm, g["object"], f[:2])
            goals[arm] = (np.array([f[0], f[1], f[2] + top]), topdown(psi))
        L, Rr = self.arms["left"], self.arms["right"]
        pre = {a: (p + [0, 0, 0.08], Rt) for a, (p, Rt) in goals.items()}
        seg: Segment = {}
        for a, (p, Rt) in pre.items():
            seg[a] = np.vstack(list(self.arms[a].travel(p, Rt, hand=1.0, lift_to=SAFE_Z)))
        yield seg
        n = max(L.line_steps(*goals["left"], speed=0.08), Rr.line_steps(*goals["right"], speed=0.08))
        yield {a: self.arms[a].line(p, Rt, n=n, settle=6) for a, (p, Rt) in goals.items()}
        yield {a: self.arms[a].gripper(0.0, 16) for a in goals}
        yield {a: self.arms[a].hold(6) for a in goals}
        rise = g["height"] + 0.035
        up = {a: (p + [0, 0, rise], Rt) for a, (p, Rt) in goals.items()}
        n = max(L.line_steps(*up["left"], speed=0.06), Rr.line_steps(*up["right"], speed=0.06))
        yield {a: self.arms[a].line(p, Rt, n=n, settle=0) for a, (p, Rt) in up.items()}
        yield {a: self.arms[a].hold(60) for a in goals}

    def _task_handover(self) -> Iterator[Segment]:
        g = self.priv["goal"]
        bar, zone, src, dst = g["object"], g["zone"], g["from"], g["to"]
        A, B = self.arms[src], self.arms[dst]
        o = self.obj(bar)
        R, c = np.asarray(o["R"]), np.asarray(o["pos"])
        u = R[:, 1]                                          # the bar's long axis
        # At the handover the bar lies along y in the middle of the table, both grippers close along x and each
        # keeps its wrist camera mount (on the TCP's +z side) on its own side, away from the other gripper.
        psi_pick = self.closing_yaw(src, bar, c[:2])
        psi_a = 0.0 if abs(_wrap(psi_pick)) < math.pi / 2 else math.pi
        dpsi = _wrap(psi_a - psi_pick)
        a_side = -math.cos(psi_a)                            # A's grasp ends up on this side (y) of the centre
        u_turned = _rotz(dpsi) @ u
        sgn_a = 1.0 if u_turned[1] * a_side > 0 else -1.0
        hy = float(np.clip(c[1], -0.04, 0.14))
        wait = np.array([-0.04 if dst == "left" else 0.04, hy - a_side * HAND_AT, HANDOVER_Z + 0.14])
        gen_a = self.pick(src, bar, along=sgn_a * HAND_AT, lift=HANDOVER_Z + 0.02, psi=psi_pick)
        rows_b = np.vstack(list(B.travel(wait, topdown(psi_a + math.pi), hand=1.0)))
        first = next(gen_a)
        yield {src: first[1], dst: rows_b}
        for arm, rows in gen_a:
            yield {arm: rows}
        # carry: the bar's centre to the handover point, turned by dpsi about the vertical
        o = self.obj(bar)
        c = np.asarray(o["pos"])
        T = A.tcp()
        Rz = _rotz(_wrap(psi_a - yaw_of_closing(T[:3, :3])))
        H = np.array([0.0, hy, HANDOVER_Z])
        yield {src: A.line(H + Rz @ (T[:3, 3] - c), Rz @ T[:3, :3], speed=0.15, settle=8)}
        # B grasps the free end from above
        o = self.obj(bar)
        R, c = np.asarray(o["R"]), np.asarray(o["pos"])
        Ta = A.tcp()
        a_dir = np.sign((Ta[:3, 3] - c) @ R[:, 1]) or 1.0
        pb = c - a_dir * HAND_AT * R[:, 1]
        zb = c[2] + o["half_size"][2] - 0.012
        Rb = topdown(psi_a + math.pi)
        yield {dst: np.vstack(list(B.travel([pb[0], pb[1], zb + 0.07], Rb, hand=1.0)))}
        yield {dst: B.line([pb[0], pb[1], zb], Rb, speed=0.08, settle=6)}
        yield {dst: B.gripper(0.0, 16), src: A.hold(16)}
        if not 0.05 < float(B.hand_measured()[0]) < 0.95:
            raise PlanFailed(f"{dst} missed the bar at the handover")
        yield {src: A.gripper(1.0, 12), dst: B.hold(12)}
        Ta = A.tcp()
        yield {src: A.line(Ta[:3, 3] + [0.0, 0.0, 0.08], Ta[:3, :3], speed=0.12, settle=2)}
        yield {src: A.go_home()}
        z = self.obj(zone)
        yield from ({k: v} for k, v in self.place(dst, bar, np.asarray(z["pos"][:2]),
                                                  0.0005 + self.obj(bar)["half_size"][2], drop=0.006))
        yield {dst: B.go_home()}


def _rotz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


__all__ = ["AlohaOracle"]
