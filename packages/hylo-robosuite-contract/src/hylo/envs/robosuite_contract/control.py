"""Joint-position control and the hand command on robosuite (pure numpy; robosuite objects are only touched through
their attributes, so this runs with robosuite 1.4 and 1.5).

robosuite's parallel grippers keep a finger set-point (`gripper.current_action`: [-1, +1] closed, [+1, -1] open)
that their own actions only push towards open or closed at the gripper's speed. The contract's hand command is a
target opening in [0, 1] (0 closed, 1 open): the set-point moves towards it at that same speed each control step and
the gripper's own action is 0 (keep the set-point).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def step_gripper_setpoint(gripper: Any, command: float, speed: float | None = None) -> None:
    """Move a robosuite gripper's finger set-point one control step towards the opening `command` (0 closed, 1 open)
    at `speed` (default: the gripper's own `speed`, 0.2 per step for robosuite 1.5's PandaGripper)."""
    c = float(np.clip(command, 0.0, 1.0))
    step = float(speed if speed is not None else getattr(gripper, "speed", 0.2))
    want = np.array([2.0 * c - 1.0, 1.0 - 2.0 * c])        # [-1, +1] closed .. [+1, -1] open
    cur = np.broadcast_to(np.asarray(gripper.current_action, dtype=np.float64), (2,))
    gripper.current_action = cur + np.clip(want - cur, -step, step)


def hand_opening(finger_qpos: Sequence[float], gap_closed: float, gap_open: float) -> float:
    """Measured opening in command units from the two finger joints (finger 2 travels negative): the finger travel
    sum, from where the servos settle closed (0) to where they settle open (1)."""
    g = np.asarray(finger_qpos, dtype=np.float64)
    return float(np.clip((g[0] - g[1] - gap_closed) / (gap_open - gap_closed), 0.0, 1.0))


def clip_joints(q: Sequence[float], lower: Sequence[float], upper: Sequence[float]) -> np.ndarray:
    return np.clip(np.asarray(q, dtype=np.float64), lower, upper)


def split_action(a: Sequence[float], arms: int, joints: int = 7) -> list[tuple[np.ndarray, float]]:
    """Contract action -> [(joint targets, hand command)] per arm (each arm: `joints` targets then one command)."""
    v = np.asarray(a, dtype=np.float64).reshape(-1)
    k = joints + 1
    if len(v) != arms * k:
        raise ValueError(f"action has {len(v)} values; {arms} arm(s) need {arms * k}")
    return [(v[i * k:i * k + joints].copy(), float(v[i * k + joints])) for i in range(arms)]


__all__ = ["clip_joints", "hand_opening", "split_action", "step_gripper_setpoint"]
