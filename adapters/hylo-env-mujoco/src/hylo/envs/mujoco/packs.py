"""Robot packs (hylo.pack/v0.1): every robot a stage can name, with what is known about driving it.

A pack points at an asset manifest (MuJoCo Menagerie folder at a pinned commit, verified by digest, fetched on
first use; see assets.py) and its model file, and says how the world controls it:

```text
  family     arm, gripper, hand, bimanual, quadruped, humanoid, drone, mobile_manipulator
  base       fixed (bolted to the world or to the entity it is mounted on) or free (a floating base)
  home       the model's keyframe used as the start posture (none: zero joints)
  drive      position (position servos: a target is a ctrl value) or torque (motors: the world closes a PD loop
             with `gains`) or thrust (drones)
  gripper    {actuator, open, closed}: the ctrl values of an open and a closed hand, when it has one
  tcp        {site} or {body, offset}: the tool point an arm moves (`tcp` actions)
  legs       quadrupeds: per leg the hip, thigh and calf joints, for the trot controller
  base_joints  mobile bases driven through joints (x, y, yaw)
  controls   the actions the world accepts for it (joints, gripper, tcp, walk, fly, base, hold)
  status     usable (checked in the world tests) | partial (some controls only; `notes` says which) | unverified
```

Readiness is separate from priority: a pack is usable when its listed controls passed the world's checks on this
workstation (tests/test_world.py), not because it is a common robot.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field

from hylo.core.contract import Contract

from .assets import assets, manifest

PACK_SCHEMA = "hylo.pack/v0.1"
Family = Literal["arm", "gripper", "hand", "bimanual", "quadruped", "humanoid", "drone", "mobile_manipulator"]
Control = Literal["joints", "gripper", "tcp", "walk", "fly", "base", "hold", "policy"]


class Gripper(Contract):
    actuator: str
    open: float
    closed: float


class Tcp(Contract):
    site: str | None = None
    body: str | None = None
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


class Leg(Contract):
    name: str
    hip: str
    thigh: str
    calf: str


class Pack(Contract):
    schema_version: Literal["hylo.pack/v0.1"] = PACK_SCHEMA
    name: str
    title: str
    family: Family
    asset: str                                   # asset manifest name (assets.py)
    model: str                                   # model file inside the asset folder
    base: Literal["fixed", "free"] = "fixed"
    home: str | None = None
    drive: Literal["position", "torque", "thrust"] = "position"
    gains: tuple[float, float] = (0.0, 0.0)      # torque drives: kp, kd
    gripper: Gripper | None = None
    tcp: Tcp | None = None
    legs: tuple[Leg, ...] = ()
    base_joints: tuple[str, ...] = ()            # x, y, yaw joints of a mobile base
    controls: tuple[Control, ...] = ("joints", "hold")
    status: Literal["usable", "partial", "unverified"] = "unverified"
    notes: str = ""
    reach: float | None = Field(default=None, description="arm reach from the base, metres (for reachable checks)")

    def model_path(self) -> Path:
        """The model file, its folder fetched and verified on first use."""
        return assets(self.asset) / self.model

    @property
    def license(self) -> str:
        return manifest(self.asset)["license"]

    @property
    def source(self) -> str:
        m = manifest(self.asset)
        return f"{m['source']}/tree/{m['commit'][:12]}/{m['folder']}"            # Menagerie or another repository


def _legs(names: tuple[str, ...], fmt: str) -> tuple[Leg, ...]:
    return tuple(Leg(name=n, hip=fmt.format(leg=n, part="hip"), thigh=fmt.format(leg=n, part="thigh"),
                     calf=fmt.format(leg=n, part="calf")) for n in names)


ARM = ("joints", "gripper", "tcp", "hold")
UNITREE_LEGS = _legs(("FL", "FR", "RL", "RR"), "{leg}_{part}_joint")
SPOT_LEGS = tuple(Leg(name=n, hip=f"{n.lower()}_hx", thigh=f"{n.lower()}_hy", calf=f"{n.lower()}_kn")
                  for n in ("FL", "FR", "HL", "HR"))

PACKS: dict[str, Pack] = {p.name: p for p in [
    # arms
    Pack(name="panda", title="Franka Emika Panda", family="arm", asset="menagerie_panda", model="panda.xml", home="home",
         gripper=Gripper(actuator="actuator8", open=255.0, closed=0.0), tcp=Tcp(body="hand", offset=(0, 0, 0.1034)),
         controls=ARM, status="usable", reach=0.85),
    Pack(name="ur5e", title="Universal Robots UR5e", family="arm", asset="menagerie_ur5e", model="ur5e.xml",
         home="home", tcp=Tcp(site="attachment_site"), controls=("joints", "tcp", "hold"), status="usable", reach=0.85,
         notes="no gripper in the model: mount one (robotiq_2f85) on its attachment"),
    Pack(name="ur10e", title="Universal Robots UR10e", family="arm", asset="menagerie_ur10e", model="ur10e.xml",
         home="home", tcp=Tcp(site="attachment_site"), controls=("joints", "tcp", "hold"), status="usable", reach=1.3),
    Pack(name="iiwa14", title="KUKA LBR iiwa 14", family="arm", asset="menagerie_iiwa14", model="iiwa14.xml",
         home="home", tcp=Tcp(site="attachment_site"), controls=("joints", "tcp", "hold"), status="usable", reach=0.82),
    Pack(name="xarm7", title="UFACTORY xArm 7", family="arm", asset="menagerie_xarm7", model="xarm7.xml", home="home",
         gripper=Gripper(actuator="gripper", open=0.0, closed=255.0), tcp=Tcp(site="link_tcp"), controls=ARM,
         status="usable", reach=0.7),
    Pack(name="so101", title="RobotStudio SO-101", family="arm", asset="menagerie_so101", model="so101.xml",
         gripper=Gripper(actuator="gripper", open=1.5, closed=-0.15), tcp=Tcp(site="gripperframe"), controls=ARM,
         status="usable", reach=0.35, notes="no home keyframe: starts at zero joints"),
    Pack(name="vx300s", title="Trossen ViperX 300 6DOF", family="arm", asset="menagerie_vx300s", model="vx300s.xml",
         home="home", gripper=Gripper(actuator="gripper", open=0.037, closed=0.002), tcp=Tcp(site="pinch"),
         controls=ARM, status="usable", reach=0.75),
    Pack(name="aloha", title="ALOHA (two ViperX 300)", family="bimanual", asset="menagerie_aloha", model="aloha.xml",
         home="neutral_pose", controls=("joints", "hold"), status="partial",
         notes="joints only: two arms of one model; the gripper actuators are left/gripper and right/gripper"),
    # hands and grippers
    Pack(name="robotiq_2f85", title="Robotiq 2F-85", family="gripper", asset="menagerie_robotiq_2f85",
         model="2f85.xml", gripper=Gripper(actuator="fingers_actuator", open=0.0, closed=255.0),
         tcp=Tcp(site="pinch"), controls=("gripper", "hold"), status="usable",
         notes="mount on an arm's attachment body (mount.body) to give it a hand"),
    Pack(name="leap", title="LEAP Hand (right)", family="hand", asset="menagerie_leap", model="right_hand.xml",
         controls=("joints", "hold"), status="usable"),
    Pack(name="allegro", title="Wonik Allegro Hand (right)", family="hand", asset="menagerie_allegro",
         model="right_hand.xml", controls=("joints", "hold"), status="usable"),
    Pack(name="shadow", title="Shadow Dexterous Hand (right)", family="hand", asset="menagerie_shadow",
         model="right_hand.xml", controls=("joints", "hold"), status="usable",
         notes="coupled finger joints share one actuator (tendons): targets are per actuator"),
    # legged
    Pack(name="go2", title="Unitree Go2", family="quadruped", asset="menagerie_go2", model="go2.xml", base="free",
         home="home", drive="torque", gains=(80.0, 2.0), legs=UNITREE_LEGS, controls=("joints", "walk", "hold"),
         status="usable", notes="torque motors: the world runs a PD loop (kp 80, kd 2)"),
    Pack(name="go1", title="Unitree Go1", family="quadruped", asset="menagerie_go1", model="go1.xml", base="free",
         home="home", legs=UNITREE_LEGS, controls=("joints", "walk", "hold"), status="usable"),
    Pack(name="a1", title="Unitree A1", family="quadruped", asset="menagerie_a1", model="a1.xml", base="free",
         home="home", legs=UNITREE_LEGS, controls=("joints", "walk", "hold"), status="usable"),
    Pack(name="spot", title="Boston Dynamics Spot", family="quadruped", asset="menagerie_spot", model="spot.xml",
         base="free", home="home", legs=SPOT_LEGS, controls=("joints", "walk", "hold"), status="usable"),
    Pack(name="g1", title="Unitree G1 (29 DoF)", family="humanoid", asset="menagerie_g1", model="g1.xml", base="free",
         home="stand", controls=("joints", "hold", "policy"), status="partial",
         notes="holds a posture; standing and walking need a balance policy (base: fixed pins the pelvis)"),
    Pack(name="h1", title="Unitree H1", family="humanoid", asset="menagerie_h1", model="h1.xml", base="free",
         drive="torque", gains=(150.0, 4.0), controls=("joints", "hold", "policy"), status="partial",
         notes="torque motors (PD in the world); needs a balance policy to stand, or base: fixed"),
    Pack(name="t1", title="Booster T1", family="humanoid", asset="menagerie_t1", model="t1.xml", base="free",
         home="home", controls=("joints", "hold", "policy"), status="partial",
         notes="holds a posture; needs a balance policy to stand, or base: fixed"),
    Pack(name="g1_legs", title="Unitree G1, 12 leg joints (the model its walking policy was trained on)",
         family="humanoid", asset="github_rlgym_g1", model="g1_12dof.xml", base="free", drive="torque",
         gains=(100.0, 2.0), controls=("joints", "hold", "policy"), status="usable",
         notes="upper body rigid; walks with the policy g1_walk (policy action); without it, it cannot stand"),
    Pack(name="h1_legs", title="Unitree H1, 10 leg joints (the model its walking policy was trained on)",
         family="humanoid", asset="github_rlgym_h1", model="h1.xml", base="free", drive="torque",
         gains=(150.0, 2.0), controls=("joints", "hold", "policy"), status="usable",
         notes="arms rigid; walks with the policy h1_walk (policy action)"),
    # aerial
    Pack(name="crazyflie", title="Bitcraze Crazyflie 2", family="drone", asset="menagerie_crazyflie", model="cf2.xml",
         base="free", home="hover", drive="thrust", controls=("fly", "hold"), status="usable",
         notes="body thrust and moments: fly takes a target position and yaw"),
    Pack(name="x2", title="Skydio X2", family="drone", asset="menagerie_x2", model="x2.xml", base="free",
         home="hover", drive="thrust", controls=("fly", "hold"), status="usable", notes="four rotor thrusts"),
    # mobile manipulation
    Pack(name="tidybot", title="Stanford TidyBot (holonomic base, Kinova arm)", family="mobile_manipulator",
         asset="menagerie_tidybot", model="tidybot.xml", home="home", base_joints=("joint_x", "joint_y", "joint_th"),
         gripper=Gripper(actuator="fingers_actuator", open=0.0, closed=255.0), tcp=Tcp(site="pinch_site"),
         controls=("joints", "gripper", "tcp", "base", "hold"), status="usable", reach=0.9),
]}


def pack(name: str) -> Pack:
    try:
        return PACKS[name]
    except KeyError:
        raise KeyError(f"no robot pack {name!r}; packs: {sorted(PACKS)}") from None


@lru_cache(maxsize=None)
def describe(name: str) -> dict:
    """What a model needs to know to use a pack (no file access)."""
    p = pack(name)
    return {"name": p.name, "title": p.title, "family": p.family, "base": p.base, "drive": p.drive,
            "controls": list(p.controls), "status": p.status, "notes": p.notes, "has_gripper": p.gripper is not None,
            "reach_m": p.reach}


__all__ = ["PACKS", "PACK_SCHEMA", "Gripper", "Leg", "Pack", "Tcp", "describe", "pack"]
