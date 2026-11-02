"""The Franka Panda as robosuite models it, for the robot contract (pure Python: numpy and hylo-core).

robosuite 1.5 names arm `i`'s joints `robot{i}_joint1..7` and attaches its gripper as the arm's "right" part with
the prefix `gripper{i}_right_`: the hand frame is the gripper's root body (`gripper{i}_right_right_gripper` for the
PandaGripper), the fingers `gripper{i}_right_finger_joint1/2`, the pads `gripper{i}_right_finger{1,2}_pad_collision`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

LOWER: tuple[float, ...] = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973)
UPPER: tuple[float, ...] = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973)
LIMIT_MARGIN = 0.02            # ArmSpec limits sit this far inside the model's (IK and planning stay off the stops)
FINGER_TRAVEL = 0.04           # each Panda finger, metres, at its joint limit
# robosuite's start posture holds the hand low over the front of the workspace; HOME lifts it above and behind it so
# the fixed cameras see the table
HOME: tuple[float, ...] = (0.0, -0.9, 0.0, -2.7, 0.0, 1.9, 0.7854)
# root body of each gripper model robosuite can attach (None: no gripper)
GRIPPER_BODY = {"PandaGripper": "right_gripper", "WipingGripper": "wiping_gripper", None: "null_gripper"}


@dataclass(frozen=True)
class PandaNames:
    """robosuite 1.5's names for arm `index` with gripper model `gripper` (None: no gripper)."""

    index: int = 0
    gripper: str | None = "PandaGripper"

    @property
    def joints(self) -> tuple[str, ...]:
        return tuple(f"robot{self.index}_joint{k}" for k in range(1, 8))

    @property
    def base(self) -> str:
        return f"robot{self.index}_base"

    @property
    def prefix(self) -> str:
        return f"gripper{self.index}_right_"

    @property
    def ee_body(self) -> str:
        """The body the TCP is defined on: the gripper's root body."""
        return self.prefix + GRIPPER_BODY.get(self.gripper, "right_gripper")

    @property
    def eef_body(self) -> str:
        """robosuite's grip point body (where a peg or a plate is fixed when there are no fingers)."""
        return self.prefix + "eef"

    @property
    def fingered(self) -> bool:
        return self.gripper == "PandaGripper"

    @property
    def finger_joints(self) -> tuple[str, ...]:
        return (f"{self.prefix}finger_joint1", f"{self.prefix}finger_joint2") if self.fingered else ()

    @property
    def pads(self) -> tuple[str, ...]:
        return (f"{self.prefix}finger1_pad_collision", f"{self.prefix}finger2_pad_collision") if self.fingered else ()


@dataclass(frozen=True)
class ArmFacts:
    """What the RobotSpec needs about one arm, measured in the simulator (`export.panda_model`) or given by the
    suite's layout: its contract name, robosuite names, exported model, base pose in the contract's world frame and
    hand geometry."""

    name: str
    names: PandaNames
    model: str
    base_in_world: list[list[float]]
    tcp_in_ee: list[list[float]]
    max_width: float


def robot_spec(arms: Sequence[ArmFacts], cameras: Sequence[dict[str, Any]], table_height: float | None, notes: str,
               home: Sequence[float] = HOME):
    """RobotSpec for Panda arms; the action is, per arm in order, 7 joint targets then one hand command (0 closed,
    1 open; ignored by arms without fingers). `cameras`: [{"name", "height", "width", "mounted_on"}]."""
    from hylo.core.robot import ArmSpec, CameraSpec, HandSpec, RobotSpec

    lo = tuple(round(v + LIMIT_MARGIN, 4) for v in LOWER)
    hi = tuple(round(v - LIMIT_MARGIN, 4) for v in UPPER)
    specs = []
    for k, a in enumerate(arms):
        hand = HandSpec(kind="parallel", tcp_in_ee=a.tcp_in_ee, open=(1.0,), closed=(0.0,), max_width=float(a.max_width),
                        joints=a.names.finger_joints)
        specs.append(ArmSpec(name=a.name, model=str(a.model), joints=a.names.joints, ee_frame=a.names.ee_body,
                             base_in_world=a.base_in_world, lower=lo, upper=hi, home=tuple(home), hand=hand,
                             action_slice=(8 * k, 8 * k + 7), hand_slice=(8 * k + 7, 8 * k + 8)))
    cams = tuple(CameraSpec(name=c["name"], height=int(c["height"]), width=int(c["width"]), depth=True,
                            mounted_on=c.get("mounted_on")) for c in cameras)
    return RobotSpec(arms=tuple(specs), cameras=cams, table_height=table_height, notes=notes)


def hand_note(gripper: str | None, max_width: float, tcp_offset: float, stroke_s: float = 0.5) -> str:
    """One sentence for RobotSpec.notes on what the hand is and where its TCP sits."""
    if gripper == "PandaGripper":
        return (f"Panda parallel gripper: the hand command is a target opening (the full stroke takes about "
                f"{stroke_s:g} s); hand.<arm> is the measured opening (max {max_width:.3f} m between the pads); TCP "
                f"between the finger pads, {tcp_offset:.3f} m along the hand's axis, +x approach, +y closing.")
    if gripper == "WipingGripper":
        return ("Sponge hand (no fingers; the hand command is ignored): TCP at the centre of the sponge's face, +x "
                "pointing out of it.")
    return "No gripper (the hand command is ignored): TCP at robosuite's grip point, +x approach."


__all__ = ["ArmFacts", "FINGER_TRAVEL", "GRIPPER_BODY", "HOME", "LIMIT_MARGIN", "LOWER", "PandaNames", "UPPER",
           "hand_note", "robot_spec"]
