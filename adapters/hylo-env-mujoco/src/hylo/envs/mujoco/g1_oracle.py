"""`oracle.g1_table`: a privileged scripted agent for `mujoco/humanoid-v0` (an upper bound, not a method).

It reads object poses and the goal from `extra["privileged"]` (env param `privileged_obs: true`) and moves the G1's
arms and fingers with ordinary joint-position actions:

- pick_place: the hand on the block's side comes down over it with the claw open (thumb behind, fingers in front),
  closes to the power grasp, carries the block over the plate and opens;
- two_hands: each hand picks the block on its side and puts it on the plate on its side, both at once (param
  `sequential: true` runs them one after the other);
- press_button: the nearer hand closes and pushes the cap down with the underside of its curled fingers from above
  (a pointing index finger needs the hand turned fingers-down, which this arm reaches over too little of the table to
  be reliable).
"""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np

from .g1_table import _OPEN, _POWER, BTN_TRAVEL, preset, tcp_in_ee
from .multi_oracle import ArmScript, MultiArmOracle, Segment, topdown
from .oracle import PlanFailed

LIFT_Z = 0.12                     # carrying height above the table (TCP)
FIST_BELOW = 0.019                # the closed hand's lowest point under the TCP (palm down)


def _pose(R, p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


class G1Oracle(MultiArmOracle):
    """Params: speed, settle, sequential (two_hands: one hand after the other, the baseline the limit rules out)."""

    NAME = "oracle.g1_table"
    LINE_TOL = (0.012, 0.2)           # the hand's orientation may drift a little along a line
    SOLVE_TOL = (0.006, 0.06)
    IK_RESTARTS = 16
    NEAREST_IK = True                 # a seven-joint arm has far-apart solutions; swinging between them hits the body

    def __init__(self, speed: float = 0.25, settle: int = 4, sequential: bool = False) -> None:
        super().__init__(speed, settle)
        self.sequential = bool(sequential)

    def _scripts(self, robot) -> dict[str, ArmScript]:
        out = {}
        for a in robot.arms:
            sgn = 1.0 if a.name == "left" else -1.0
            q_nom = np.array([-0.6, 0.15 * sgn, 0.0, 0.6, 0.0, 0.3, 0.0])
            out[a.name] = ArmScript(self, a, q_nom)
        self.ee_tcp = {a.name: np.asarray(tcp_in_ee(a.name)) for a in robot.arms}
        return out

    def hand(self, arm: str, closure) -> np.ndarray:
        return np.asarray(preset(arm, closure))

    def tcp_from_wrist(self, arm: str, T_wrist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T = T_wrist @ self.ee_tcp[arm]
        return T[:3, 3], T[:3, :3]

    def table_top(self) -> float:
        return self.robot.table_height or 0.0

    def reachable_psi(self, arm: str, points, psis) -> float:
        """The first closing-axis yaw (of `psis`) at which a top-down claw reaches every point."""
        s = self.arms[arm]
        for psi in psis:
            R = topdown(psi)
            try:
                for p in points:
                    T = np.eye(4)
                    T[:3, :3], T[:3, 3] = R, p
                    s.solve(T, s.rest(np.asarray(p)))
                return float(psi)
            except PlanFailed:
                continue
        raise PlanFailed(f"{arm}: no top-down claw yaw reaches {np.round(points[0], 3).tolist()}")

    # -- pick and place -------------------------------------------------------------------------------------------
    def _task_pick_place(self) -> Iterator[Segment]:
        g = self.priv["goal"]
        arm = "left" if self.obj(g["object"])["pos"][1] > 0 else "right"
        for a, rows in self._pick_place(arm, g["object"], g["target"]):
            yield {a: rows}

    def _pick_place(self, arm: str, obj: str, tgt: str) -> Iterator[tuple[str, np.ndarray]]:
        """One hand: pick `obj` with the claw from above, put it on `tgt` (generator of (arm, rows))."""
        s = self.arms[arm]
        for _ in range(2):
            o = self.obj(obj)
            p = np.asarray(o["pos"])
            top = self.table_top()
            base = o["yaw"] + (math.pi / 2) * round(-o["yaw"] / (math.pi / 2))     # closing axis nearest +x
            psi = self.reachable_psi(arm, [p + [0, 0, 0.002], p + [0, 0, 0.10]],
                                     [base + k * math.pi / 2 for k in (0, 1, -1)])
            R = topdown(psi)
            yield arm, np.vstack(list(s.travel([p[0], p[1], p[2] + 0.10], R, hand=self.hand(arm, _OPEN),
                                              lift_to=top + LIFT_Z)))
            yield arm, s.line([p[0], p[1], p[2] + 0.002], R, speed=0.08, settle=4)
            yield arm, s.gripper(self.hand(arm, _POWER), 16)
            yield arm, s.line([p[0], p[1], top + LIFT_Z], R, speed=0.12)
            if self.obj(obj)["pos"][2] > top + 0.06:
                break
            yield arm, s.gripper(self.hand(arm, _OPEN), 8)
        else:
            raise PlanFailed(f"{arm} could not pick up {obj}")
        t = self.obj(tgt)
        T = s.tcp()
        off = np.asarray(self.obj(obj)["pos"]) - T[:3, 3]              # the block hangs this far from the TCP
        goal = np.array([t["pos"][0], t["pos"][1], t["pos"][2] + 2 * t["half_size"][2] + self.obj(obj)["half_size"][2]
                         + 0.006]) - np.array([0.0, 0.0, off[2]])
        psi = self.reachable_psi(arm, [goal, goal + [0, 0, 0.08]],
                                 [psi + k * math.pi / 4 for k in (0, 1, -1, 2, -2)])
        R = topdown(psi)
        yield arm, np.vstack(list(s.travel([goal[0], goal[1], top + LIFT_Z], R)))
        yield arm, s.line(goal, R, speed=0.08, settle=4)
        yield arm, s.gripper(self.hand(arm, _OPEN), 10)
        yield arm, s.line([goal[0], goal[1], top + LIFT_Z], R, speed=0.12, settle=0)
        yield arm, s.go_home(hand=self.hand(arm, _OPEN))

    # -- two hands at once ----------------------------------------------------------------------------------------
    def _task_two_hands(self) -> Iterator[Segment]:
        jobs = []
        for obj, tgt in self.priv["goal"]["assign"].items():
            arm = "left" if self.obj(obj)["pos"][1] > 0 else "right"
            jobs.append(self._pick_place(arm, obj, tgt))
        run = self.in_turn if self.sequential else self.together
        yield from run(*jobs)

    # -- press a button with the index finger ---------------------------------------------------------------------
    def _task_press_button(self) -> Iterator[Segment]:
        g = self.priv["goal"]
        if g["button"] in self.priv["state"].get("pressed", []):
            return
        b = self.obj(g["button"])
        p = np.asarray(b["pos"])
        arm = "left" if p[1] > 0 else "right"
        s = self.arms[arm]
        top = p[2] + 2 * b["half_size"][2]
        fist = self.hand(arm, _POWER)
        # the closed hand from above (the posture of a grasp): its lowest point is FIST_BELOW under the TCP
        above = np.array([p[0], p[1], top + FIST_BELOW + 0.05])
        down = np.array([p[0], p[1], top + FIST_BELOW - BTN_TRAVEL + 0.001])
        psi = self.reachable_psi(arm, [above, down], [0.0, math.pi / 4, -math.pi / 4, math.pi / 2, -math.pi / 2])
        R = topdown(psi)
        yield {arm: np.vstack([s.gripper(fist, 8), *s.travel(above, R, hand=fist)])}
        yield {arm: s.line(down, R, speed=0.05, settle=8)}
        yield {arm: s.line(above, R, speed=0.08, settle=4)}


__all__ = ["G1Oracle"]
