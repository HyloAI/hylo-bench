"""Scripted motion for privileged oracles of robots with several arms (`oracle.aloha_table`, `oracle.g1_table`).

An oracle task is a generator of *segments*: each segment is {arm: (T, joints + hand) rows}; arms left out hold their
last command, and the shorter arm of a segment holds its last row until the longer one finishes, so the arms of one
segment start together. Feedback (did the grasp hold?) comes from the observation current when the generator is
resumed. Motions are damped-least-squares IK on each arm's model (`oracle.Kinematics`) along straight TCP lines with
minimum-jerk timing, capped in joint speed, as in `oracle.panda_table`.
"""

from __future__ import annotations

import math
from typing import Any, Iterator

import numpy as np

from hylo.core.agent import AgentBase, AgentSpec, EpisodeContext
from hylo.core.types import ActionChunk, ActionSemantics, Observation

from .oracle import Kinematics, PlanFailed, _axis_angle, _minjerk, _rot_err, _wrap

QD_MAX = 1.2
DOWN = np.array([0.0, 0.0, -1.0])
Segment = dict[str, np.ndarray]


def topdown(psi: float) -> np.ndarray:
    """TCP rotation for a top-down approach with the closing axis at yaw `psi` (+x down, +y closing)."""
    y = np.array([math.cos(psi), math.sin(psi), 0.0])
    return np.stack([DOWN, y, np.cross(DOWN, y)], axis=1)


def yaw_of_closing(R: np.ndarray) -> float:
    return math.atan2(R[1, 1], R[0, 1])


class ArmScript:
    """Motion primitives of one arm. Rows are joint targets followed by the hand command."""

    def __init__(self, oracle: "MultiArmOracle", arm: Any, q_nom: np.ndarray, rest_fn=None) -> None:
        self.o, self.spec, self.name = oracle, arm, arm.name
        self.kin = Kinematics(arm)
        self.n = len(arm.joints)
        self.home = np.asarray(arm.home, float)
        self.q_nom = np.asarray(q_nom, float)
        self.rest_fn = rest_fn
        self.cmd: np.ndarray | None = None

    # -- state ----------------------------------------------------------------------------------------------------
    @property
    def obs(self) -> Observation:
        return self.o._obs  # type: ignore[return-value]

    def q(self) -> np.ndarray:
        q = np.asarray(self.obs.state[f"q.{self.name}"], dtype=float)
        if self.cmd is not None and np.max(np.abs(self.cmd[:self.n] - q)) < 0.08:
            return self.cmd[:self.n].copy()
        return q

    def hand_cmd(self) -> np.ndarray:
        if self.cmd is not None:
            return self.cmd[self.n:].copy()
        return np.asarray(self.obs.state[f"hand.{self.name}"], dtype=float)

    def hand_measured(self) -> np.ndarray:
        return np.asarray(self.obs.state[f"hand.{self.name}"], dtype=float)

    def tcp(self) -> np.ndarray:
        return self.kin.fk(self.q())

    def rest(self, pos: np.ndarray) -> np.ndarray:
        return self.rest_fn(self, pos) if self.rest_fn is not None else self.q_nom.copy()

    def _h(self, hand) -> np.ndarray:
        return self.hand_cmd() if hand is None else np.atleast_1d(np.asarray(hand, float))

    def rows(self, qs, hand) -> np.ndarray:
        h = self._h(hand)
        return np.asarray([np.concatenate([q, h]) for q in qs])

    # -- motions --------------------------------------------------------------------------------------------------
    def hold(self, n: int, hand=None) -> np.ndarray:
        return self.rows([self.q()] * n, hand)

    def line(self, pos, R, speed: float = 0.25, hand=None, settle: int = 4, n: int | None = None) -> np.ndarray:
        """Straight TCP line (position linear, rotation about one axis) with minimum-jerk timing; `n` fixes the
        number of steps (for motions that must stay in step with another arm's)."""
        q_start = self.q()
        T0 = self.kin.fk(q_start)
        p0, R0 = T0[:3, 3], T0[:3, :3]
        pos = np.asarray(pos, float)
        rv = _rot_err(R0, R)
        dist, ang = float(np.linalg.norm(pos - p0)), float(np.linalg.norm(rv))
        fixed = n is not None
        if n is None:
            n = max(2, math.ceil(max(dist / speed, ang / 1.2) * 1.4 * self.o.hz))
        rest = self.rest(pos)
        n0 = n
        for _ in range(4):
            q, qs = q_start, []
            for s in _minjerk(n):
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = _axis_angle(rv * s) @ R0, p0 + s * (pos - p0)
                q, pe, re = self.kin.ik(T, q, rest)
                if pe > self.o.LINE_TOL[0] or re > self.o.LINE_TOL[1]:
                    raise PlanFailed(f"{self.name}: IK failed near {np.round(T[:3, 3], 3).tolist()} ({pe:.3f} m, {re:.3f} rad)")
                qs.append(q)
            ratio = float(np.max(np.abs(np.diff(np.vstack([q_start, *qs]), axis=0)))) / (QD_MAX / self.o.hz)
            if ratio <= 1.0 or fixed:
                break
            if n * ratio > 8 * n0 + 40:                # the joints jump somewhere along the line: not a smooth path
                raise PlanFailed(f"{self.name}: the line to {np.round(pos, 3).tolist()} needs a joint jump")
            n = math.ceil(n * ratio * 1.1)
        out = self.rows(qs, hand)
        if settle:
            out = np.vstack([out, np.repeat(out[-1:], settle, axis=0)])
        self.cmd = out[-1].copy()
        return out

    def line_steps(self, pos, R, speed: float = 0.25) -> int:
        T0 = self.tcp()
        dist = float(np.linalg.norm(np.asarray(pos) - T0[:3, 3]))
        ang = float(np.linalg.norm(_rot_err(T0[:3, :3], R)))
        return max(2, math.ceil(max(dist / speed, ang / 1.2) * 1.4 * self.o.hz))

    def solve(self, T: np.ndarray, rest: np.ndarray) -> np.ndarray:
        seeds = [self.q(), rest, self.q_nom]
        rng = np.random.default_rng(0)
        lo, hi = np.asarray(self.spec.lower), np.asarray(self.spec.upper)
        for k in range(self.o.IK_RESTARTS):                  # near the rest posture, then anywhere in the limits
            seeds.append(np.clip(rest + rng.normal(0, 0.4, self.n), lo, hi) if k % 2 == 0 else rng.uniform(lo, hi))
        best, near = None, None
        q_now = self.q()
        for k, seed in enumerate(seeds):
            q, pe, re = self.kin.ik(T, seed, rest, iters=300)
            if best is None or pe + 0.1 * re < best[1] + 0.1 * best[2]:
                best = (q, pe, re)
            if pe < 0.002 and re < 0.01:                # converged: keep the one nearest the current joints
                d = float(np.max(np.abs(q - q_now)))
                if near is None or d < near[0]:
                    near = (d, (q, pe, re))
                if k < 2 or not self.o.NEAREST_IK:          # from the current joints or the rest posture: good
                    break
        if near is not None:
            best = near[1]
        q, pe, re = best  # type: ignore[misc]
        if pe > self.o.SOLVE_TOL[0] or re > self.o.SOLVE_TOL[1]:
            raise PlanFailed(f"{self.name}: IK failed at {np.round(T[:3, 3], 3).tolist()} ({pe:.3f} m, {re:.3f} rad)")
        return q

    def joints(self, q1, hand=None, settle: int = 4, speed: float = 0.75) -> np.ndarray:
        q0 = self.q()
        q1 = np.asarray(q1, float)
        n = max(2, math.ceil(float(np.max(np.abs(q1 - q0))) / (speed * QD_MAX) * 1.4 * self.o.hz))
        out = self.rows([q0 + s * (q1 - q0) for s in _minjerk(n)], hand)
        out = np.vstack([out, np.repeat(out[-1:], settle, axis=0)])
        self.cmd = out[-1].copy()
        return out

    def joint_move(self, pos, R, hand=None) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, pos
        return self.joints(self.solve(T, self.rest(np.asarray(pos, float))), hand)

    def gripper(self, cmd, steps: int = 12) -> np.ndarray:
        q, h0 = self.q(), self.hand_cmd()
        h1 = np.atleast_1d(np.asarray(cmd, float))
        k = max(1, steps // 3)
        out = np.asarray([np.concatenate([q, h0 + (h1 - h0) * min(1.0, (i + 1) / k)]) for i in range(steps)])
        self.cmd = out[-1].copy()
        return out

    def travel(self, pos, R, hand=None, lift_to: float | None = None) -> Iterator[np.ndarray]:
        """Up to `lift_to` first when the TCP is lower, then a straight line (joint-space move from home or when the
        line leaves the reachable set)."""
        cur = self.tcp()
        pos = np.asarray(pos, float)
        if lift_to is not None and cur[2, 3] < lift_to - 0.03:
            yield self.line([cur[0, 3], cur[1, 3], lift_to], cur[:3, :3], speed=0.18, hand=hand)
        if np.linalg.norm(self.q() - self.home) < 0.05:
            yield self.joint_move(pos, R, hand=hand)
            return
        try:
            yield self.line(pos, R, hand=hand)
        except PlanFailed:
            yield self.joint_move(pos, R, hand=hand)

    def go_home(self, hand=None) -> np.ndarray:
        return self.joints(self.home, hand)


class MultiArmOracle(AgentBase):
    """Base of the privileged oracles for robots with several arms. Subclasses define `_scripts(robot)` and one
    generator method `_task_<task>()` per task."""

    NAME = "oracle.multi_arm"
    MAX_ATTEMPTS = 4
    LINE_TOL = (0.01, 0.1)            # IK residual accepted along straight lines (m, rad)
    SOLVE_TOL = (0.005, 0.03)         # and at the goal of a joint-space move
    IK_RESTARTS = 6
    NEAREST_IK = False                # try every seed and keep the converged solution nearest the current joints

    def __init__(self, speed: float = 0.25, settle: int = 4) -> None:
        self.speed, self.settle = float(speed), int(settle)
        self._key = None

    def spec(self) -> AgentSpec:
        return AgentSpec(name=self.NAME, version="0.1", action_semantics=ActionSemantics(control_mode="joint_pos"))

    def _scripts(self, robot) -> dict[str, ArmScript]:
        raise NotImplementedError

    def reset(self, ctx: EpisodeContext) -> None:
        self.ctx = ctx
        robot = ctx.env_spec.robot
        if robot is None:
            raise ValueError(f"{self.NAME} needs an environment with a RobotSpec")
        key = robot.model_dump_json()
        if self._key != key:                          # one set of models per agent: evaluation workers are threads
            self.robot, self._key = robot, key
            self.arms = self._scripts(robot)
        for s in self.arms.values():
            s.cmd = None
        self.hz = float(ctx.env_spec.control_hz)
        self._gen: Iterator[Segment] | None = None
        self._obs: Observation | None = None
        self.attempts = 0

    @property
    def priv(self) -> dict[str, Any]:
        return self._obs.extra["privileged"]  # type: ignore[union-attr]

    def obj(self, name: str) -> dict[str, Any]:
        return self.priv["objects"][name]

    def act(self, observation: Observation) -> ActionChunk:
        priv = observation.extra.get("privileged")
        if not isinstance(priv, dict):
            raise ValueError(f"{self.NAME} reads extra['privileged']: run the environment with privileged_obs: true")
        self._obs = observation
        seg: Segment | None = None
        for _ in range(2):
            if self._gen is None:
                self.attempts += 1
                self._gen = getattr(self, f"_task_{priv['task']}")() if self.attempts <= self.MAX_ATTEMPTS else iter(())
            try:
                seg = next(self._gen)
                break
            except (StopIteration, PlanFailed) as exc:
                if isinstance(exc, PlanFailed):
                    self.ctx.emit("agent.oracle.plan_failed", {"reason": str(exc), "attempt": self.attempts})
                self._gen = None
        if not seg:
            seg = {name: s.hold(5) for name, s in self.arms.items()}
        return ActionChunk(actions=self._merge(seg), provenance={"oracle": priv["task"], "attempt": self.attempts})

    def _merge(self, seg: Segment) -> np.ndarray:
        T = max(len(v) for v in seg.values())
        cols = []
        for a in self.robot.arms:
            s = self.arms[a.name]
            rows = seg.get(a.name)
            if rows is None or len(rows) == 0:
                rows = s.hold(1)
            if len(rows) < T:
                rows = np.vstack([rows, np.repeat(rows[-1:], T - len(rows), axis=0)])
            s.cmd = rows[-1].copy()
            cols.append(rows)
        return np.hstack(cols)

    # -- helpers for parallel scripts -------------------------------------------------------------------------------
    def together(self, *gens: Iterator[tuple[str, np.ndarray]]) -> Iterator[Segment]:
        """Run per-arm generators side by side: each yields (arm, rows); one segment pairs their next motions."""
        live = list(gens)
        while live:
            seg: Segment = {}
            nxt = []
            for g in live:
                try:
                    arm, rows = next(g)
                except StopIteration:
                    continue
                seg[arm] = rows
                nxt.append(g)
            live = nxt
            if seg:
                yield seg

    def in_turn(self, *gens: Iterator[tuple[str, np.ndarray]]) -> Iterator[Segment]:
        """The same per-arm generators one after the other (the sequential baseline)."""
        for g in gens:
            for arm, rows in g:
                yield {arm: rows}


__all__ = ["ArmScript", "MultiArmOracle", "QD_MAX", "Segment", "topdown", "yaw_of_closing", "_wrap"]
