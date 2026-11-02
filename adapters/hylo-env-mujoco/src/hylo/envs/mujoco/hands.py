"""Dexterous robot hands from MuJoCo Menagerie, each as a HandSpec plus its model, and the floating-hand robot built
from one: the hand on a six-joint base (three slides, then yaw, pitch and roll hinges) that `mujoco.dex_table` drives
like an arm and that the replay-to-sim workflow loads as its hand model.

    hand_names()                -> ["allegro_left", "allegro_right", "g1_left", "g1_right"]
    resolve("allegro", "left")  -> "allegro_left"
    hand_spec("g1_right")       -> HandSpec (finger joints, open / closed commands, presets, TCP)
    floating_hand_file(name)    -> Path of the floating-hand MJCF (written once into the asset cache)

Hands (folders fetched and verified by assets.py, never vendored):
    g1_left / g1_right          Unitree G1 three-finger hands (unitree_g1, BSD-3-Clause): thumb (3 joints), index
                                and middle (2 each); the claw commands and TCP of mujoco.g1_table.
    allegro_left / _right       Wonik Robotics Allegro Hand (wonik_allegro, BSD-2-Clause): four fingers of four
                                joints each.

The floating-hand model marks the human wrist's place on the hand with a site `wrist` on the hand body (used by
retargeting); finger servos are renamed `<joint>_servo`, the base servos `dex_<joint>_servo`.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from hylo.core.robot import HandSpec

from .assets import ALLEGRO, G1_HANDS, assets, cache_root

# -- Unitree G1 (left hand; the right mirrors every joint but thumb_0), as in mujoco.g1_table
_G1_JOINTS = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
_G1_MIRROR = np.array([1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
_G1 = {"open": (0.0, -0.4, 0.0, -0.9, -0.6, -0.9, -0.6), "power": (0.0, 0.6, 0.7, -1.3, -1.2, -1.3, -1.2),
       "pinch": (0.0, 0.5, 0.5, -1.1, -0.9, -1.1, -0.9), "point": (0.0, 0.9, 1.0, 0.0, 0.0, -1.5, -1.6),
       "flat": (0.0,) * 7}
_G1_RANGE_LEFT = ((-1.0472, 1.0472), (-0.724312, 1.0472), (0.0, 1.74533), (-1.5708, 0.0), (-1.74533, 0.0),
                  (-1.5708, 0.0), (-1.74533, 0.0))
_G1_TCP_POS = {"left": (0.097, -0.055, 0.0), "right": (0.097, 0.055, 0.0)}
_G1_TCP_ROT = {"left": ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),       # rows: the TCP's x, y, z axes
               "right": ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0))}

# -- Allegro: per finger (spread, proximal, medial, distal), thumb (rotation, spread, medial, distal). Palm frame:
# fingers along +z, curling towards +x (the palm's face). open: a claw (fingers half curled, thumb swung opposite);
# power: wrapped; pinch: fingertips onto the thumb tip. TCP 5 cm in front of the palm face at mid-finger height:
# +x approach (into the palm face), +y closing (thumb towards the fingers: the palm's +z), +z = x cross y.
_AL_FINGERS = ("ff", "mf", "rf", "th")
_AL = {"open": (0.0, 0.45, 0.25, 0.2) * 3 + (1.25, 0.35, 0.2, 0.2),
       "power": (0.0, 1.35, 1.45, 1.35) * 3 + (1.35, 0.90, 1.25, 1.35),
       "pinch": (0.0, 0.95, 0.75, 0.55) * 3 + (1.35, 0.95, 0.70, 0.75),
       "flat": (0.0, 0.0, 0.0, 0.0) * 3 + (0.3, 0.0, 0.0, 0.0)}
_AL_RANGE = ((-0.47, 0.47), (-0.196, 1.61), (-0.174, 1.709), (-0.227, 1.618)) * 3 + \
    ((0.263, 1.396), (-0.105, 1.163), (-0.189, 1.644), (-0.162, 1.719))
_AL_TCP = [[1, 0, 0, 0.05], [0, 0, -1, 0.0], [0, 1, 0, 0.035], [0, 0, 0, 1]]


@dataclass(frozen=True)
class HandEntry:
    name: str
    title: str
    licence: str
    manifest: str                       # asset manifest of the Menagerie folder
    file: str                           # model file in that folder
    ee: str                             # the hand's body (its frame: HandSpec.tcp_in_ee is relative to it)
    spec: HandSpec
    ranges: tuple[tuple[float, float], ...]   # finger joint ranges (HandSpec.joints order), from the model
    wrist: tuple[float, float, float]   # the human wrist's place on the hand, in the ee frame
    side: str


def _g1(side: str) -> HandEntry:
    sign = np.ones(7) if side == "left" else _G1_MIRROR

    def cmd(v):
        return tuple(round(float(x), 4) + 0.0 for x in np.asarray(v) * sign)

    T = np.eye(4)
    T[:3, :3] = np.asarray(_G1_TCP_ROT[side]).T
    T[:3, 3] = _G1_TCP_POS[side]
    ranges = tuple(r if i == 0 or side == "left" else (-r[1], -r[0]) for i, r in enumerate(_G1_RANGE_LEFT))
    spec = HandSpec(kind="dexterous", tcp_in_ee=T.tolist(), open=cmd(_G1["open"]), closed=cmd(_G1["power"]),
                    max_width=0.08, joints=tuple(f"{side}_hand_{j}_joint" for j in _G1_JOINTS),
                    presets={k: cmd(v) for k, v in _G1.items() if k != "open"})
    return HandEntry(f"g1_{side}", f"Unitree G1 {side} hand", "BSD-3-Clause", G1_HANDS, "g1_with_hands.xml",
                     f"{side}_wrist_yaw_link", spec, ranges, (0.0, 0.0, 0.0), side)


def _allegro(side: str) -> HandEntry:
    joints = tuple(f"{f}j{k}" for f in _AL_FINGERS for k in range(4))
    spec = HandSpec(kind="dexterous", tcp_in_ee=_AL_TCP, open=_AL["open"], closed=_AL["power"], max_width=0.10,
                    joints=joints, presets={k: v for k, v in _AL.items() if k != "open"})
    return HandEntry(f"allegro_{side}", f"Wonik Allegro {side} hand", "BSD-2-Clause", ALLEGRO, f"{side}_hand.xml",
                     "palm", spec, _AL_RANGE, (-0.01, 0.0, -0.095), side)


HANDS = {"g1_left": lambda: _g1("left"), "g1_right": lambda: _g1("right"),
         "allegro_left": lambda: _allegro("left"), "allegro_right": lambda: _allegro("right")}


def hand_names() -> list[str]:
    return sorted(HANDS)


def resolve(name: str, side: str | None = None) -> str:
    """A library name from a full name, or a family ("g1", "allegro") and a side (default right)."""
    if name in HANDS:
        return name
    full = f"{name}_{side or 'right'}"
    if full in HANDS:
        return full
    raise KeyError(f"no hand {name!r}; hands: {hand_names()} (or g1 / allegro with a side)")


def hand_entry(name: str) -> HandEntry:
    return HANDS[resolve(name)]()


def hand_spec(name: str) -> HandSpec:
    return hand_entry(name).spec


# -- the floating hand -----------------------------------------------------------------------------------------
BASE_JOINTS = (("x", "slide", (1, 0, 0)), ("y", "slide", (0, 1, 0)), ("z", "slide", (0, 0, 1)),
               ("yaw", "hinge", (0, 0, 1)), ("pitch", "hinge", (0, 1, 0)), ("roll", "hinge", (1, 0, 0)))
BASE_RANGE = {"x": (-0.4, 0.4), "y": (-0.4, 0.4), "z": (0.0, 0.6), "yaw": (-3.1, 3.1), "pitch": (-1.45, 1.45),
              "roll": (-3.1, 3.1)}
BASE_KP = {"slide": 3000.0, "hinge": 300.0}
BASE_FORCE = {"slide": 200.0, "hinge": 30.0}
# at zero base joints the TCP points straight down (approach -z), closing along +x
DOWN = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]).T    # columns: TCP x, y, z in the world


def base_joint_names() -> tuple[str, ...]:
    return tuple(f"dex_{j}" for j, _, _ in BASE_JOINTS)


def mount_quat(entry: HandEntry) -> np.ndarray:
    """Orientation of the hand body on the base so the TCP points down at zero base joints."""
    import mujoco

    R_tcp = np.asarray(entry.spec.tcp_in_ee, float)[:3, :3]
    R = DOWN @ R_tcp.T
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    return q


def floating_hand_spec(name: str, finger_gain: float = 1.0) -> Any:
    """MjSpec of the floating hand: base chain `dex_x_link` ... `dex_roll_link`, the hand body attached at the end
    (its own arm joints dropped, meshes the hand does not use removed), servos on every joint, gravity
    compensation everywhere, a `wrist` site on the hand body."""
    import mujoco

    e = hand_entry(name)
    folder = assets(e.manifest)
    src = mujoco.MjSpec.from_file(str(folder / e.file))
    meshdir = folder / (src.meshdir or "")
    for m in src.meshes:
        if m.file and not Path(m.file).is_absolute():
            m.file = str((meshdir / m.file).resolve())
    spec = mujoco.MjSpec()
    spec.compiler.autolimits = True
    spec.compiler.degree = False                       # joint ranges in radians
    spec.option.timestep = 0.002
    parent = spec.worldbody
    for j, kind, axis in BASE_JOINTS:
        parent = parent.add_body(name=f"dex_{j}_link", gravcomp=1.0)
        parent.add_joint(name=f"dex_{j}", type=mujoco.mjtJoint.mjJNT_SLIDE if kind == "slide" else
                         mujoco.mjtJoint.mjJNT_HINGE, axis=list(axis), range=list(BASE_RANGE[j]), damping=5.0 if
                         kind == "slide" else 1.0, armature=0.01)
        parent.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.004, 0, 0], mass=0.2, contype=0, conaffinity=0,
                        group=4)
    mount = parent.add_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mount.attach_body(src.body(e.ee), "", "")
    hand = spec.body(e.ee)
    hand.pos, hand.quat = [0.0, 0.0, 0.0], list(mount_quat(e))       # TCP down at zero base joints
    for jt in list(hand.joints):
        spec.delete(jt)
    keep = set(e.spec.joints)
    for a in list(spec.actuators):
        if a.target not in keep:
            spec.delete(a)
            continue
        a.name = f"{a.target}_servo"
        a.gainprm[0] *= finger_gain
        a.biasprm[1] *= finger_gain
    for item in [*spec.keys, *spec.sensors, *spec.tendons, *spec.equalities, *spec.lights]:
        spec.delete(item)
    names = {b.name for b in spec.bodies}
    for ex in list(spec.excludes):
        if ex.bodyname1 not in names or ex.bodyname2 not in names:
            spec.delete(ex)
    used = {g.meshname for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH}
    for m in list(spec.meshes):
        if m.name not in used:
            spec.delete(m)
    for b in spec.bodies:
        b.gravcomp = 1.0
    hand.add_site(name="wrist", pos=list(e.wrist), size=[0.006, 0, 0], group=5)
    for j, kind, _ in BASE_JOINTS:
        a = spec.add_actuator(name=f"dex_{j}_servo", target=f"dex_{j}", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=BASE_KP[kind], kv=0.1 * BASE_KP[kind] ** 0.5 * 2)
        a.ctrlrange = list(BASE_RANGE[j])
        a.forcerange = [-BASE_FORCE[kind], BASE_FORCE[kind]]
        a.forcelimited = 1
    return spec


@lru_cache(maxsize=None)
def floating_hand_file(name: str) -> Path:
    """The floating hand as an MJCF file in the asset cache (rewritten when its content changes); the model a
    RobotSpec names for kinematics."""
    import mujoco

    spec = floating_hand_spec(resolve(name))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = spec.compile()
        xml = _name_nested_defaults(spec.to_xml(), hand_entry(name).ee)
    path = cache_root() / "hylo_dex" / f"{resolve(name)}-{hashlib.sha1(xml.encode()).hexdigest()[:10]}.xml"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(xml)
        tmp.replace(path)
    m = mujoco.MjModel.from_xml_path(str(path))        # it loads on its own and matches the spec
    for attr in ("jnt_range", "dof_damping", "dof_armature", "dof_frictionloss", "actuator_gainprm", "actuator_biasprm",
                 "geom_size", "geom_friction", "geom_contype", "body_mass"):
        if not np.allclose(getattr(m, attr), getattr(ref, attr), atol=1e-6):
            raise RuntimeError(f"{path}: written model differs from the spec in {attr}")
    return path


def _name_nested_defaults(xml: str, root: str) -> str:
    """An attached model's own top default comes out unnamed inside the main one, which MJCF rejects: name it, and
    make it the child class of the attached root body so the elements under it keep those defaults."""
    out, seen, cls = [], 0, None
    for ln in xml.splitlines():
        if ln.strip() == "<default>":
            seen += 1
            if seen > 1:
                cls = f"attached_main_{seen}"
                ln = ln.replace("<default>", f'<default class="{cls}">')
        out.append(ln)
    if cls is not None:
        for i, ln in enumerate(out):
            if f'<body name="{root}"' in ln and "childclass=" not in ln:
                out[i] = ln.replace(f'<body name="{root}"', f'<body name="{root}" childclass="{cls}"', 1)
                break
    return "\n".join(out) + "\n"


__all__ = ["BASE_JOINTS", "DOWN", "HANDS", "HandEntry", "base_joint_names", "floating_hand_file",
           "floating_hand_spec", "hand_entry", "hand_names", "hand_spec", "mount_quat", "resolve"]
