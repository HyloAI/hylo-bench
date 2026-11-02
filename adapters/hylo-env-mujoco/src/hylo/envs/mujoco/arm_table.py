"""Shared machinery of the table environments with several arms (`mujoco.aloha_table`, `mujoco.g1_table`).

A subclass supplies the robot (an MjSpec it attaches to the scene, its arms as `ArmIO` records, its RobotSpec), its
cameras and its object pool; this base runs everything else the same way for every robot:

- one compiled model holds a pool of objects; an episode activates the ones its task needs (colour, size, mass) and
  parks the rest inside the table, so the model never changes shape and snapshots stay cheap;
- control is joint-position targets per arm plus a hand command per arm, at `control_hz`;
- observations follow `hylo.core.robot`: `q.<arm>`, `hand.<arm>` (measured, command units), RGB-D per camera,
  `<cam>.seg` with `segmentation=True`, camera intrinsics and poses in `extra["cameras"]`, and with
  `privileged_obs=True` object poses and the goal in `extra["privileged"]` (oracles only);
- snapshots carry the episode record and the full integration state.

The task family (sampling and success predicates) is a module with `sample_episode(params, rng)` and
`evaluate(env)`, named by the subclass.
"""

from __future__ import annotations

import importlib
import json
import math
import struct
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from hylo.core.robot import RobotSpec
from hylo.core.task import TaskInstance
from hylo.core.types import (Action, ActionSemantics, BoxSpace, EnvCapabilities, EnvSpec, Observation,
                             ObservationSpec, StepResult)

from .panda_table import _Cameras, _cluster, _f, _pose, _quat_z, _xyaxes, _yaw_of

CONTROL_DT_PHYS = 0.002
SETTLE_S = 0.2
PARK_DEPTH = -0.45                         # parked objects wait inside the table body, out of every view

COLOURS: dict[str, tuple[float, float, float]] = {
    "red": (0.85, 0.12, 0.10), "green": (0.15, 0.65, 0.20), "blue": (0.15, 0.35, 0.90),
    "yellow": (0.95, 0.82, 0.10), "purple": (0.55, 0.22, 0.75), "orange": (0.97, 0.50, 0.08),
}
FIXTURE_COLOURS: dict[str, tuple[float, float, float]] = {
    "white": (0.93, 0.93, 0.91), "pink": (0.95, 0.62, 0.72), "cyan": (0.35, 0.80, 0.85), "grey": (0.55, 0.56, 0.58),
}
ALL_COLOURS = {**COLOURS, **FIXTURE_COLOURS}


@dataclass
class ArmIO:
    """How one arm maps onto the compiled model: joints, actuators, finger joints and the hand command."""
    name: str
    joints: tuple[str, ...]                    # model joint names, command order
    actuators: tuple[str, ...]                 # position actuators of those joints, same order
    hand_actuators: tuple[str, ...]            # actuators driven by the hand command
    finger_joints: tuple[str, ...]             # joints measured for the hand state
    hand_to_ctrl: Callable[[np.ndarray], np.ndarray]     # hand command -> actuator targets
    hand_state: Callable[[np.ndarray], np.ndarray]       # finger joint positions -> command units
    hand_open: tuple[float, ...]
    ee_body: str                               # body of the TCP
    tcp_in_ee: list[list[float]]
    finger_bodies: tuple[str, ...] = ()        # bodies whose contacts count as "touching the hand"
    home: tuple[float, ...] = ()
    lower: tuple[float, ...] = ()
    upper: tuple[float, ...] = ()
    hand_lower: tuple[float, ...] = ()         # bounds of the hand command (default 0 .. 1 per value)
    hand_upper: tuple[float, ...] = ()
    # filled at bind time
    qadr: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    vadr: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    ctrl: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    hand_ctrl: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    finger_qadr: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    ee: int = -1
    fingers: set[int] = field(default_factory=set)


def read_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and faces of a binary (or ASCII) STL file, vertices merged."""
    raw = path.read_bytes()
    if raw[:5] == b"solid" and b"facet" in raw[:400]:
        vs = [list(map(float, ln.split()[1:4])) for ln in raw.decode(errors="ignore").splitlines()
              if ln.strip().startswith("vertex")]
        tri = np.asarray(vs, float).reshape(-1, 3, 3)
    else:
        n = int(np.frombuffer(raw, dtype="<u4", count=1, offset=80)[0])
        rec = np.frombuffer(raw, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n, offset=84)
        tri = rec["v"].astype(float)
    flat = tri.reshape(-1, 3)
    v, inv = np.unique(np.round(flat, 7), axis=0, return_inverse=True)
    return v, inv.reshape(-1, 3)


def simplify_visual_meshes(spec: Any, folder: Path, cell: float, min_faces: int = 4000, groups: Sequence[int] = (1, 2),
                           cache_tag: str = "") -> None:
    """Point the visual geoms (contype 0, in `groups`) of large meshes at vertex-clustered copies (cell `cell`
    metres); collision geoms keep the original meshes. The copies are derived data, cached next to the assets."""
    import os

    import mujoco

    out_path = folder.parent / f"{folder.name}.visual{cache_tag}-{cell * 1000:.1f}mm.npz"
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if out_path.is_file():
        z = np.load(out_path)
        cache = {k[:-2]: (z[k], z[k[:-2] + ".f"]) for k in z.files if k.endswith(".v")}
    meshes = {m.name: m for m in spec.meshes}
    made: dict[str, str] = {}
    changed = False
    for g in spec.geoms:
        if g.type != mujoco.mjtGeom.mjGEOM_MESH or g.contype != 0 or g.group not in groups:
            continue
        mesh = meshes.get(g.meshname)
        if mesh is None or not mesh.file:
            continue
        if g.meshname not in made:
            key = Path(mesh.file).name
            if key not in cache:
                path = folder / (spec.meshdir or "") / mesh.file
                if not path.is_file():
                    path = folder / "assets" / mesh.file
                v, f = read_stl(path) if path.suffix.lower() == ".stl" else (None, None)
                if v is None or len(f) < min_faces:
                    cache[key] = (np.zeros((0, 3)), np.zeros((0, 3), int))
                else:
                    scale = np.asarray(mesh.scale, float)
                    cache[key] = _cluster(v * scale, f, cell)
                changed = True
            v, f = cache[key]
            if len(f) == 0:
                continue
            new = spec.add_mesh(name=f"{g.meshname}__vis")
            new.uservert = v.astype(np.float32).ravel().tolist()
            new.userface = f.astype(np.int32).ravel().tolist()
            made[g.meshname] = new.name
        g.meshname = made[g.meshname]
    if changed:
        tmp = out_path.with_name(out_path.name + f".{os.getpid()}.npz")
        np.savez(tmp, **{f"{k}.v": v for k, (v, _) in cache.items()}, **{f"{k}.f": f for k, (_, f) in cache.items()})
        os.replace(tmp, out_path)


def attach_quietly(frame: Any, body: Any, prefix: str = "", suffix: str = "") -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return frame.attach_body(body, prefix, suffix)


def box_inertia(mass: float, half: Sequence[float]) -> np.ndarray:
    x, y, z = (2 * h for h in half)
    return mass / 12.0 * np.array([y * y + z * z, x * x + z * z, x * x + y * y])


class ArmTableEnv:
    """Base of the table environments with several arms. Subclasses set NAME, CAMERA_NAMES, TASK_MODULE, POOL,
    FREE_KINDS, TABLE_TOP and implement `_build_model`, `_arms`, `robot_spec`, `_camera_model_name`."""

    NAME = "mujoco.arm_table"
    CAMERA_NAMES: tuple[str, ...] = ()
    TASK_MODULE = ""
    POOL: dict[str, int] = {}
    FREE_KINDS: tuple[str, ...] = ()
    RESIZABLE: tuple[str, ...] = ()             # single-box free kinds whose size the episode sets
    DENSITY = 500.0                            # kg/m^3 of resizable objects
    TABLE_TOP = 0.0
    PARK_XY = (0.0, 0.0)

    def __init__(self, cameras: Sequence[str] | None = None, width: int = 320, height: int = 240,
                 segmentation: bool = False, privileged_obs: bool = False, control_hz: float = 20.0,
                 depth: bool = True, mesh_detail: str = "fast") -> None:
        cams = tuple(self.CAMERA_NAMES if cameras is None else cameras)
        unknown = [c for c in cams if c not in self.CAMERA_NAMES]
        if unknown:
            raise ValueError(f"unknown cameras {unknown}; available: {list(self.CAMERA_NAMES)}")
        if mesh_detail not in ("fast", "full"):
            raise ValueError(f"mesh_detail must be 'fast' or 'full', got {mesh_detail!r}")
        self.camera_names, self.width, self.height = cams, int(width), int(height)
        self.segmentation, self.privileged_obs, self.depth = bool(segmentation), bool(privileged_obs), bool(depth)
        self.control_hz, self.mesh_detail = float(control_hz), mesh_detail
        self._model = None
        self._cams: _Cameras | None = None
        self.task = None
        self.ep: dict[str, Any] = {}
        self.arms: list[ArmIO] = self._arms()
        self.pool_names = tuple(f"{k}{i}" for k, n in self.POOL.items() for i in range(n))

    # -- subclass hooks -------------------------------------------------------------------------------------------
    def _arms(self) -> list[ArmIO]:
        raise NotImplementedError

    def _build_model(self):
        raise NotImplementedError

    def robot_spec(self) -> RobotSpec:
        raise NotImplementedError

    def _camera_model_name(self, cam: str) -> str:
        return cam

    def _item_geoms(self, m: Any, name: str) -> list[int]:
        """Geoms of a pool item (default: those of its body and its descendants)."""
        b = m.body(name).id
        bodies = {b} | {i for i in range(m.nbody) if self._descends(m, i, b)}
        return [g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies]

    @staticmethod
    def _descends(m: Any, i: int, root: int) -> bool:
        while i > 0:
            i = int(m.body_parentid[i])
            if i == root:
                return True
        return False

    def _is_robot_body(self, m: Any, b: int) -> bool:
        raise NotImplementedError

    def _extra_reset(self) -> None:
        """Posture of joints that are not part of any arm (e.g. a humanoid's legs), after mj_resetData."""

    # -- spec -----------------------------------------------------------------------------------------------------
    @property
    def action_dim(self) -> int:
        return sum(len(a.joints) + len(a.hand_open) for a in self.arms)

    def spec(self) -> EnvSpec:
        lo, hi, labels = [], [], []
        state, units = {}, {}
        for a in self.arms:
            lo += list(a.lower) + list(a.hand_lower or [0.0] * len(a.hand_open))
            hi += list(a.upper) + list(a.hand_upper or [1.0] * len(a.hand_open))
            labels += list(a.joints) + [f"{a.name}/hand{i}" if len(a.hand_open) > 1 else f"{a.name}/hand"
                                        for i in range(len(a.hand_open))]
            state[f"q.{a.name}"], state[f"hand.{a.name}"] = (len(a.joints),), (len(a.hand_open),)
            units[f"q.{a.name}"], units[f"hand.{a.name}"] = "rad", "hand command units"
        robot = self.robot_spec()
        cams = {c: (self.height, self.width, 3) for c in self.camera_names}
        return EnvSpec(
            name=self.NAME, version="0.1",
            observation=ObservationSpec(state=state, cameras=cams, units=units),
            action=BoxSpace(shape=(self.action_dim,), low=tuple(lo), high=tuple(hi),
                            semantics=ActionSemantics(control_mode="joint_pos", frame="base",
                                                      gripper="0_close_1_open", dim_labels=tuple(labels))),
            control_hz=self.control_hz,
            capabilities=EnvCapabilities(seedable=True, snapshot=True, render=True, privileged_state=True),
            robot=robot)

    # -- simulator ------------------------------------------------------------------------------------------------
    @property
    def model(self):
        if self._model is None:
            self._build()
        return self._model

    def _build(self) -> None:
        import mujoco

        self._mj = mujoco
        m = self._build_model()
        self._model, self.data = m, mujoco.MjData(m)
        self.n_substeps = max(1, round(1.0 / (self.control_hz * m.opt.timestep)))
        for a in self.arms:
            jid = [m.joint(j).id for j in a.joints]
            a.qadr = np.array([m.jnt_qposadr[j] for j in jid])
            a.vadr = np.array([m.jnt_dofadr[j] for j in jid])
            a.ctrl = np.array([m.actuator(n).id for n in a.actuators])
            a.hand_ctrl = np.array([m.actuator(n).id for n in a.hand_actuators])
            a.finger_qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in a.finger_joints])
            a.ee = m.body(a.ee_body).id
            a.fingers = {m.body(b).id for b in a.finger_bodies}
        self._all_fingers = np.array(sorted(set().union(*(a.fingers for a in self.arms))), dtype=int)
        self._cam_ids = {c: m.camera(self._camera_model_name(c)).id for c in self.CAMERA_NAMES}
        self.items: dict[str, dict[str, Any]] = {}
        for name in self.pool_names:
            kind = name.rstrip("0123456789")
            b = m.body(name)
            geoms = self._item_geoms(m, name)
            info: dict[str, Any] = {"kind": kind, "body": b.id, "geoms": geoms,
                                    "contype": m.geom_contype[geoms].copy(), "conaffinity": m.geom_conaffinity[geoms].copy(),
                                    "bodies": sorted({int(m.geom_bodyid[g]) for g in geoms} | {b.id})}
            if kind in self.FREE_KINDS:
                j = m.body_jntadr[b.id]
                info["qadr"], info["vadr"] = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
            else:
                info["mocap"] = int(m.body_mocapid[b.id])
            self._bind_item(m, name, info)
            self.items[name] = info
        self.instance = {"table": 1, "robot": 2, **{n: 3 + i for i, n in enumerate(self.pool_names)}}
        gi = np.zeros(m.ngeom, dtype=np.int32)
        for g in range(m.ngeom):
            if self._is_robot_body(m, int(m.geom_bodyid[g])):
                gi[g] = 2
        gi[m.geom("table").id] = 1
        for name, info in self.items.items():
            gi[info["geoms"]] = self.instance[name]
        self.geom_instance = gi
        self._sig = mujoco.mjtState.mjSTATE_INTEGRATION
        self._warn = 0

    def _bind_item(self, m: Any, name: str, info: dict[str, Any]) -> None:
        """Extra indices a kind needs (joints of a lid, a button's slider ...)."""

    # -- episode --------------------------------------------------------------------------------------------------
    def reset(self, task: TaskInstance) -> Observation:
        tasks = importlib.import_module(self.TASK_MODULE, __package__)
        m = self.model
        mj = self._mj
        self.task = task
        rng = np.random.default_rng(task.seed)
        self.ep = tasks.sample_episode(task.spec.params, rng)
        mj.mj_resetData(m, self.data)
        self._apply_episode()
        d = self.data
        for name in self.items:
            it = self.ep["items"].get(name)
            if it is None:
                self._place(name, [*self.PARK_XY, PARK_DEPTH], 0.0)
            else:
                self._place(name, it["pos"], it.get("yaw", 0.0))
        self._extra_reset()
        for a in self.arms:
            d.qpos[a.qadr] = a.home
            d.ctrl[a.ctrl] = a.home
            d.ctrl[a.hand_ctrl] = a.hand_to_ctrl(np.asarray(a.hand_open, float))
            self._set_fingers_open(a)
        mj.mj_forward(m, d)
        mj.mj_step(m, d, nstep=int(SETTLE_S / m.opt.timestep))
        d.time = 0.0
        self._warn = self._divergence_count()
        self.t = 0
        self.ep.setdefault("state", {})
        return self._observe()

    def _set_fingers_open(self, a: ArmIO) -> None:
        """Finger joints at the open hand before settling (default: leave them at the model's zero)."""

    def _apply_episode(self) -> None:
        """Model-side state of the episode (which pool items take part, colours, sizes, masses): restore re-applies."""
        m = self._model
        active = self.ep["items"]
        for name, info in self.items.items():
            it = active.get(name)
            g = info["geoms"]
            if it is None:
                m.geom_contype[g] = 0
                m.geom_conaffinity[g] = 0
                m.body_gravcomp[info["bodies"]] = 1.0
                continue
            m.geom_contype[g] = info["contype"]
            m.geom_conaffinity[g] = info["conaffinity"]
            m.body_gravcomp[info["bodies"]] = 0.0
            rgb = ALL_COLOURS.get(it.get("colour", ""), None)
            if rgb is not None:
                for gid in self._coloured_geoms(name, info):
                    m.geom_rgba[gid, :3] = rgb
            if info["kind"] in self.RESIZABLE and "size" in it:
                half = np.asarray(it["size"], float)
                gid = g[0]
                m.geom_size[gid] = half
                m.geom_rbound[gid] = float(np.linalg.norm(half))
                m.geom_aabb[gid, 3:] = half
                mass = float(it.get("mass", self.DENSITY * 8 * np.prod(half)))
                m.body_mass[info["body"]] = mass
                m.body_inertia[info["body"]] = box_inertia(mass, half)
            elif "mass" in it and info["kind"] in self.FREE_KINDS:
                scale = float(it["mass"]) / max(float(m.body_mass[info["body"]]), 1e-9)
                m.body_mass[info["body"]] *= scale
                m.body_inertia[info["body"]] *= scale
            self._apply_item(name, info, it)

    def _coloured_geoms(self, name: str, info: dict[str, Any]) -> list[int]:
        return list(info["geoms"])

    def _apply_item(self, name: str, info: dict[str, Any], it: dict[str, Any]) -> None:
        """Kind-specific model state of an active item."""

    def _place(self, name: str, pos: Sequence[float], yaw: float) -> None:
        info, d = self.items[name], self.data
        q = np.array(_quat_z(yaw))
        if "qadr" in info:
            a = info["qadr"]
            d.qpos[a:a + 3], d.qpos[a + 3:a + 7] = pos, q
            d.qvel[info["vadr"]:info["vadr"] + 6] = 0.0
        else:
            d.mocap_pos[info["mocap"]], d.mocap_quat[info["mocap"]] = pos, q

    def _divergence_count(self) -> int:
        w = self._mj.mjtWarning
        return sum(int(self.data.warning[k].number) for k in (w.mjWARN_BADQPOS, w.mjWARN_BADQVEL, w.mjWARN_BADQACC))

    # -- step -----------------------------------------------------------------------------------------------------
    def _apply_action(self, a: np.ndarray) -> None:
        d = self.data
        i = 0
        for arm in self.arms:
            n, h = len(arm.joints), len(arm.hand_open)
            d.ctrl[arm.ctrl] = np.clip(a[i:i + n], arm.lower, arm.upper)
            d.ctrl[arm.hand_ctrl] = arm.hand_to_ctrl(a[i + n:i + n + h])
            i += n + h

    def step(self, action: Action) -> StepResult:
        tasks = importlib.import_module(self.TASK_MODULE, __package__)
        mj, m, d = self._mj, self._model, self.data
        a = np.asarray(action.data, dtype=np.float64).reshape(-1)
        self._apply_action(a)
        mj.mj_step(m, d, nstep=self.n_substeps)
        self.t += 1
        bad = self._divergence_count()
        if bad != self._warn:
            self._warn = bad
            return StepResult(observation=self._observe(), reward=None, terminated=False, truncated=True,
                              success=None, info={"sim_error": "mujoco divergence warning (BADQPOS/QVEL/QACC)"})
        self._after_step()
        success, fail, info = tasks.evaluate(self)
        if fail:
            info["fail_reason"] = fail
        return StepResult(observation=self._observe(), reward=1.0 if success else 0.0,
                          terminated=bool(success or fail), truncated=False, success=bool(success), info=info)

    def _after_step(self) -> None:
        """Latches and counters updated after each control step (buttons pressed ...)."""

    def hold_action(self, last: np.ndarray | None) -> np.ndarray:
        if last is not None:
            return np.asarray(last, dtype=float).copy()
        out = []
        for a in self.arms:
            out += list(self.data.qpos[a.qadr]) + list(self.hand_state(a.name))
        return np.asarray(out)

    # -- scene queries (task predicates, oracles) -----------------------------------------------------------------
    def arm_io(self, name: str) -> ArmIO:
        for a in self.arms:
            if a.name == name:
                return a
        raise KeyError(name)

    def pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        info, d = self.items[name], self.data
        if "qadr" in info:
            a = info["qadr"]
            return d.qpos[a:a + 3].copy(), d.qpos[a + 3:a + 7].copy()
        return d.mocap_pos[info["mocap"]].copy(), d.mocap_quat[info["mocap"]].copy()

    def rotation(self, name: str) -> np.ndarray:
        return self.data.xmat[self.items[name]["body"]].reshape(3, 3).copy()

    def speed(self, name: str) -> tuple[float, float]:
        info = self.items[name]
        if "vadr" not in info:
            return 0.0, 0.0
        v = self.data.qvel[info["vadr"]:info["vadr"] + 6]
        return float(np.linalg.norm(v[:3])), float(np.linalg.norm(v[3:]))

    def contacts_with(self, name: str, bodies: np.ndarray) -> bool:
        own = np.asarray(self.items[name]["bodies"], dtype=int)
        m, d = self._model, self.data
        n = d.ncon
        if n == 0 or len(bodies) == 0:
            return False
        b1, b2 = m.geom_bodyid[d.contact.geom1[:n]], m.geom_bodyid[d.contact.geom2[:n]]
        return bool(np.any((np.isin(b1, own) & np.isin(b2, bodies)) | (np.isin(b2, own) & np.isin(b1, bodies))))

    def touching_hand(self, name: str, arm: str | None = None) -> bool:
        fingers = self._all_fingers if arm is None else np.array(sorted(self.arm_io(arm).fingers), dtype=int)
        return self.contacts_with(name, fingers)

    def hand_state(self, arm: str) -> np.ndarray:
        a = self.arm_io(arm)
        return np.asarray(a.hand_state(self.data.qpos[a.finger_qadr]), float)

    def tcp_pose(self, arm: str) -> np.ndarray:
        a, d = self.arm_io(arm), self.data
        T = np.eye(4)
        T[:3, :3] = d.xmat[a.ee].reshape(3, 3)
        T[:3, 3] = d.xpos[a.ee]
        return T @ np.asarray(a.tcp_in_ee)

    # -- observation ----------------------------------------------------------------------------------------------
    def camera_info(self, name: str) -> dict[str, Any]:
        m, d = self._model, self.data
        cid = self._cam_ids[name]
        f = 0.5 * self.height / math.tan(math.radians(float(m.cam_fovy[cid])) / 2)
        K = [[f, 0.0, (self.width - 1) / 2], [0.0, f, (self.height - 1) / 2], [0.0, 0.0, 1.0]]
        R = d.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])   # MuJoCo camera -> OpenCV optical frame
        return {"K": K, "T_world_cam": _pose(R, d.cam_xpos[cid])}

    def _observe(self) -> Observation:
        d = self.data
        images: dict[str, np.ndarray] = {}
        extra: dict[str, Any] = {"task_family": self.ep.get("task", "")}
        if self.camera_names:
            if self._cams is None:
                self._cams = _Cameras(self._model, self.width, self.height, self.segmentation)
            extra["cameras"] = {}
            for c in self.camera_names:
                rgb, depth, seg = self._cams.render(self._model, d, self._cam_ids[c], self.geom_instance)
                images[c] = rgb
                if self.depth:
                    images[f"{c}.depth"] = depth
                if seg is not None:
                    images[f"{c}.seg"] = seg
                extra["cameras"][c] = self.camera_info(c)
            if self.segmentation:
                extra["segments"] = self.segments()
        if self.privileged_obs:
            extra["privileged"] = self.privileged_state(json_ready=True)
        state = {}
        for a in self.arms:
            state[f"q.{a.name}"] = d.qpos[a.qadr].copy()
            state[f"hand.{a.name}"] = self.hand_state(a.name)
        return Observation(state=state, images=images, instruction=self.ep.get("instruction"), time=float(d.time),
                           extra=extra)

    def segments(self) -> dict[str, str]:
        out = {str(self.instance["table"]): "table", str(self.instance["robot"]): "robot"}
        for name, it in self.ep.get("items", {}).items():
            out[str(self.instance[name])] = it["label"]
        return out

    def privileged_state(self, json_ready: bool = False) -> dict[str, Any]:
        objects = {}
        for name, it in self.ep.get("items", {}).items():
            p, q = self.pose(name)
            o = {"label": it["label"], "kind": self.items[name]["kind"], "pos": p.tolist(), "quat": q.tolist(),
                 "yaw": _yaw_of(q), "half_size": list(it["half_size"]), "instance": self.instance[name],
                 "R": self.rotation(name).tolist()}
            o.update(self._privileged_item(name, it))
            objects[name] = o
        out = {"task": self.ep.get("task"), "goal": self.ep.get("goal"), "objects": objects,
               "state": dict(self.ep.get("state", {})), "instruction": self.ep.get("instruction"),
               "q": {a.name: self.data.qpos[a.qadr].tolist() for a in self.arms},
               "hand": {a.name: self.hand_state(a.name).tolist() for a in self.arms},
               "tcp": {a.name: self.tcp_pose(a.name).tolist() for a in self.arms}}
        return json.loads(json.dumps(out)) if json_ready else out

    def _privileged_item(self, name: str, it: dict[str, Any]) -> dict[str, Any]:
        return {}

    def render(self, camera: str | None = None, width: int | None = None, height: int | None = None) -> np.ndarray:
        """An RGB frame from one camera at the configured resolution (width/height must match when given)."""
        camera = camera or self.CAMERA_NAMES[0]
        if (width or self.width) != self.width or (height or self.height) != self.height:
            raise ValueError(f"this environment renders at {self.width}x{self.height}")
        if self._cams is None:
            self._cams = _Cameras(self.model, self.width, self.height, self.segmentation)
        return self._cams.render(self._model, self.data, self._cam_ids[camera], self.geom_instance)[0]

    # -- snapshot -------------------------------------------------------------------------------------------------
    def snapshot(self) -> bytes:
        mj = self._mj
        state = np.empty(mj.mj_stateSize(self._model, self._sig))
        mj.mj_getState(self._model, self.data, state, self._sig)
        head = json.dumps({"ep": self.ep, "t": self.t, "warn": self._warn,
                           "task": self.task.model_dump(mode="json") if self.task else None}).encode()
        return struct.pack("<I", len(head)) + head + state.tobytes()

    def restore(self, blob: bytes) -> None:
        mj = self._mj
        (n,) = struct.unpack("<I", blob[:4])
        head = json.loads(blob[4:4 + n].decode())
        self.ep, self.t, self._warn = head["ep"], int(head["t"]), int(head["warn"])
        self.task = TaskInstance.model_validate(head["task"]) if head["task"] else None
        self._apply_episode()
        state = np.frombuffer(blob[4 + n:], dtype=np.float64).copy()
        mj.mj_setState(self._model, self.data, state, self._sig)
        mj.mj_forward(self._model, self.data)

    def close(self) -> None:
        if self._cams is not None:
            self._cams.close()
            self._cams = None


def camera_xml(cams: dict[str, tuple[Sequence[float], Sequence[float], float]]) -> str:
    return "\n    ".join(f'<camera name="{n}" pos="{_f(p)}" xyaxes="{_xyaxes(np.subtract(t, p), (0, 0, 1))}" fovy="{fov}"/>'
                           for n, (p, t, fov) in cams.items())


__all__ = ["ALL_COLOURS", "ArmIO", "ArmTableEnv", "COLOURS", "FIXTURE_COLOURS", "attach_quietly", "box_inertia",
           "camera_xml", "read_stl", "simplify_visual_meshes"]
