"""`mujoco.panda_scene`: the Panda table of `mujoco.panda_table` with the objects and the success predicate of a
scene file (`hylo.scene/v0.1`, hylo.core.scene), and the benchmark type `mujoco/scenes-v0` that turns a directory
of scene files into tasks with seeded RFC-0002 splits.

The robot, the table, the cameras, segmentation, the observation layout, control and RobotSpec are panda_table's
(this class reuses PandaTableEnv's robot half and its `attach_panda` / `scene_xml`); what differs is the model: one
compiled per scene file, its bodies built from the file's objects (primitives, trays, cups, meshes, compound parts,
children on slide / hinge joints). Free objects get a free joint; fixtures and the roots of articulated fixtures are
mocap bodies, so seeded start poses never need a recompile. Each episode samples the start poses from the seed
(`region`s), rests objects on the table or on their `on` support, applies joint start values, then settles 0.2 s.

The success predicate is evaluated after every control step (`sequence` progress is part of the episode state and
of snapshots). An episode fails early when a movable object falls off the table. With `privileged_obs=True` the
observation carries `extra["privileged"]`: every object's pose, extents, world bounding box, container interior and
joint state, contacts that penetrate, placement problems and the predicate's terms (for checkers and for agents run
with privileged state; env params are part of the experiment digest, so such runs are never ranked with others).

Params: scene (path to a .json / .yaml scene; a task's params["scene"] overrides it), plus PandaTableEnv's
cameras, width, height, segmentation, privileged_obs, control_hz, depth, mesh_detail.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from hylo.core.robot import RobotSpec
from hylo.core.scene import (COLOURS, AllOf, Above, AnyOf, Inside, JointAtLeast, JointAtMost, Lifted, Near, Not, On,
                             Row, Scene, SceneObject, Sequence as Seq, Upright)
from hylo.core.task import TaskInstance, TaskSpec
from hylo.core.types import Action, EnvSpec, Observation, StepResult

from .panda_table import (BASE_POS, CAMERA_NAMES, FINGER_TRAVEL, HAND_CTRL_MAX, HOME, LOWER, TABLE_TOP, TABLE_X,
                          TABLE_Y, UPPER, PandaTableEnv, attach_panda, scene_xml)

SETTLE_S = 0.2
REACH = ((-0.27, 0.2), (-0.32, 0.32))       # object centres the arm serves well (top-down), x and y
CUP_SEGMENTS = 16
LIN_REST, ANG_REST = 0.03, 0.8               # m/s, rad/s: "at rest"
FELL_Z = TABLE_TOP - 0.05
PENETRATION = 0.003                          # m: deeper contacts are reported


def load_scene(source: str | Path | dict | Scene) -> Scene:
    """A Scene from a path (.json, .yaml / .yml), a mapping or a Scene."""
    if isinstance(source, Scene):
        return source
    if isinstance(source, dict):
        return Scene.model_validate(source)
    text = Path(source).read_text()
    if str(source).endswith((".yaml", ".yml")):
        import yaml

        return Scene.model_validate(yaml.safe_load(text))
    return Scene.model_validate_json(text)


def scene_digest(scene: Scene) -> str:
    return hashlib.sha256(scene.model_dump_json().encode()).hexdigest()[:16]


# -- small geometry -----------------------------------------------------------------------------------------------
def _quat_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """(w, x, y, z) of extrinsic x-y-z rotations (radians): R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr, cp, sp, cy, sy = (math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2),
                              math.cos(yaw / 2), math.sin(yaw / 2))
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy])


def _mat(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _euler_quat(deg: Sequence[float]) -> list[float]:
    return _quat_euler(*(math.radians(v) for v in deg)).tolist()


def _corners(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def _rgba(colour: Any) -> list[float]:
    if isinstance(colour, str):
        rgb = COLOURS.get(colour.lower().strip(), COLOURS["grey"])
    else:
        rgb = tuple(float(np.clip(c, 0, 1)) for c in colour)
    return [*rgb, 1.0]


# -- model --------------------------------------------------------------------------------------------------------
def _primitive_geoms(shape: str, size: Sequence[float], wall: float, mesh: str | None) -> list[dict[str, Any]]:
    """Geoms (type, size, pos, quat, mesh) of one shape centred on its bounding box."""
    if shape == "box":
        return [{"type": "box", "size": [s / 2 for s in size]}]
    if shape == "cylinder":
        return [{"type": "cylinder", "size": [size[0], size[1] / 2]}]
    if shape == "capsule":
        return [{"type": "capsule", "size": [size[0], size[1] / 2 - size[0]]}]
    if shape == "sphere":
        return [{"type": "sphere", "size": [size[0]]}]
    if shape == "mesh":
        return [{"type": "mesh", "mesh": mesh, "scale": list(size) * 3 if len(size) == 1 else list(size) or [1, 1, 1]}]
    if shape == "tray":
        sx, sy, sz = (s / 2 for s in size)
        w = wall / 2
        hz = sz - w                                      # wall half height above the floor
        out = [{"type": "box", "size": [sx, sy, w], "pos": [0, 0, -sz + w]}]
        out += [{"type": "box", "size": [w, sy, hz], "pos": [sgn * (sx - w), 0, w]} for sgn in (1, -1)]
        out += [{"type": "box", "size": [sx - wall, w, hz], "pos": [0, sgn * (sy - w), w]} for sgn in (1, -1)]
        return out
    if shape == "cup":
        r, hz = size[0], size[1] / 2
        w = wall / 2
        out = [{"type": "cylinder", "size": [r, w], "pos": [0, 0, -hz + w]}]
        half = (r - w) * math.tan(math.pi / CUP_SEGMENTS) + w
        for i in range(CUP_SEGMENTS):
            a = 2 * math.pi * i / CUP_SEGMENTS
            out.append({"type": "box", "size": [w, half, hz - w], "pos": [(r - w) * math.cos(a), (r - w) * math.sin(a), w],
                        "quat": [math.cos(a / 2), 0, 0, math.sin(a / 2)]})
        return out
    raise ValueError(f"unknown shape {shape!r}")


def _object_geoms(o: SceneObject, base: Path) -> list[dict[str, Any]]:
    def resolve(m: str | None) -> str | None:
        return None if m is None else str((base / m).resolve() if not Path(m).is_absolute() else Path(m))

    if o.shape is not None:
        return [{**g, "rgba": _rgba(o.colour)} for g in _primitive_geoms(o.shape, o.size, o.wall, resolve(o.mesh))]
    out = []
    for p in o.parts:
        q_part = _euler_quat(p.euler_deg)
        for g in _primitive_geoms(p.shape, p.size, 0.0, resolve(p.mesh)):
            out.append({**g, "pos": (np.asarray(p.pos) + _mat(q_part) @ np.asarray(g.get("pos", [0, 0, 0]))).tolist(),
                        "quat": q_part, "rgba": _rgba(p.colour if p.colour is not None else o.colour)})
    return out


def build_scene_model(scene: Scene, base: Path, width: int = 320, height: int = 240, mesh_detail: str = "fast"):
    """Compile the Panda table with the scene's objects. Returns (model, {object: body id})."""
    import mujoco

    G = mujoco.mjtGeom
    types = {"box": G.mjGEOM_BOX, "cylinder": G.mjGEOM_CYLINDER, "capsule": G.mjGEOM_CAPSULE,
             "sphere": G.mjGEOM_SPHERE, "mesh": G.mjGEOM_MESH}
    spec = mujoco.MjSpec.from_string(scene_xml(width, height, items=""))
    attach_panda(spec, mesh_detail)
    by_name = {o.name: o for o in scene.objects}
    bodies: dict[str, Any] = {}
    meshes: dict[tuple, str] = {}

    def add(o: SceneObject) -> Any:
        if o.name in bodies:
            return bodies[o.name]
        if o.parent is None:
            body = spec.worldbody.add_body(name=o.name, pos=[0.0, 0.0, -0.5])
            if o.movable:
                body.add_freejoint(name=f"{o.name}/free")
            else:
                body.mocap = True
        else:
            pose = o.pose
            assert pose is not None
            body = add(by_name[o.parent]).add_body(name=o.name, pos=list(pose.pos),
                                                   quat=_euler_quat((pose.roll_deg, pose.pitch_deg, pose.yaw_deg)))
            if o.joint is not None:
                j = o.joint
                hinge = j.type == "hinge"
                conv = math.radians if hinge else float
                body.add_joint(name=f"{o.name}/joint", type=mujoco.mjtJoint.mjJNT_HINGE if hinge else mujoco.mjtJoint.mjJNT_SLIDE,
                               axis=list(_mat(_euler_quat((pose.roll_deg, pose.pitch_deg, pose.yaw_deg))).T @ np.asarray(j.axis)),
                               pos=list(j.anchor), range=[conv(j.range[0]), conv(j.range[1])], limited=True,
                               damping=j.damping, stiffness=j.spring, frictionloss=j.friction,
                               springref=conv(j.initial if j.spring_ref is None else j.spring_ref), armature=0.002)
        moving = o.movable or o.joint is not None or (o.parent is not None and not _fixed_root(o, by_name))
        for i, g in enumerate(_object_geoms(o, base)):
            kw: dict[str, Any] = {"name": f"{o.name}/g{i}", "type": types[g["type"]], "pos": list(g.get("pos", [0, 0, 0])),
                                  "rgba": g["rgba"], "friction": [o.friction, 0.01, 0.001], "condim": 4 if moving else 3}
            if moving:
                kw["solref"] = [0.005, 1.0]
            if "quat" in g:
                kw["quat"] = list(g["quat"])
            if g["type"] == "mesh":
                key = (g["mesh"], tuple(g["scale"]))
                if key not in meshes:
                    meshes[key] = f"mesh{len(meshes)}"
                    spec.add_mesh(name=meshes[key], file=g["mesh"], scale=list(g["scale"]))
                kw["meshname"] = meshes[key]
            else:
                kw["size"] = list(g["size"]) + [0.0] * (3 - len(g["size"]))
            body.add_geom(**kw)
        bodies[o.name] = body
        return body

    for o in scene.objects:
        add(o)
    m = spec.compile()
    d = mujoco.MjData(m)
    ids = {o.name: m.body(o.name).id for o in scene.objects}
    for o in scene.objects:                          # each object's own body weighs `mass`
        b = ids[o.name]
        if m.body_mass[b] > 0:
            k = o.mass / m.body_mass[b]
            m.body_mass[b] *= k
            m.body_inertia[b] *= k
    mujoco.mj_setConst(m, d)
    return m, ids


def _fixed_root(o: SceneObject, by_name: dict[str, SceneObject]) -> bool:
    """True when o hangs rigidly (no joint on the way) from a fixture."""
    while o.parent is not None:
        if o.joint is not None:
            return False
        o = by_name[o.parent]
    return not o.movable


# -- environment --------------------------------------------------------------------------------------------------
class PandaSceneEnv(PandaTableEnv):
    """`mujoco.panda_scene`. Params: scene (path), and those of `mujoco.panda_table`."""

    def __init__(self, scene: str | None = None, cameras: Sequence[str] | None = CAMERA_NAMES, width: int = 320,
                 height: int = 240, segmentation: bool = False, privileged_obs: bool = False, control_hz: float = 20.0,
                 depth: bool = True, mesh_detail: str = "fast") -> None:
        super().__init__(cameras=cameras, width=width, height=height, segmentation=segmentation,
                         privileged_obs=privileged_obs, control_hz=control_hz, depth=depth, mesh_detail=mesh_detail)
        self.scene_path = str(scene) if scene else None
        self.scene: Scene | None = None
        self._loaded: tuple[str, str] | None = None     # (path, digest) of the compiled model

    # -- spec -------------------------------------------------------------------------------------------------------
    def robot_spec(self) -> RobotSpec:
        base = super().robot_spec()
        (x0, x1), (y0, y1) = REACH
        notes = (
            f"Franka Panda (MuJoCo Menagerie) on a table; world frame: +z up, table top at z={TABLE_TOP}, table x "
            f"[{TABLE_X[0]}, {TABLE_X[1]}], y [{TABLE_Y[0]}, {TABLE_Y[1]}]; arm base (link0) at {list(BASE_POS)}, +x "
            f"points from the base towards the front camera. The arm serves object centres in x [{x0}, {x1}], "
            f"y [{y0}, {y1}] (metres) best. Max hand opening {2 * FINGER_TRAVEL:.2f} m. Action: 7 joint-position "
            f"targets (rad) + hand command (0 closed, 1 open) at {self.control_hz:g} Hz; hand.arm is the measured "
            f"opening in command units. TCP between the finger pads; +x approach, +y closing. Objects come from a "
            f"scene file; success needs objects at rest and released unless the scene says otherwise.")
        return base.model_copy(update={"notes": notes})

    def spec(self) -> EnvSpec:
        return super().spec().model_copy(update={"name": "mujoco.panda_scene"})

    # -- model ------------------------------------------------------------------------------------------------------
    def _load(self, path: str) -> None:
        """Compile the model for the scene at `path` (no-op when it is already loaded)."""
        import mujoco

        scene = load_scene(path)
        key = (str(Path(path).resolve()), scene_digest(scene))
        if self._loaded == key:
            return
        if self._cams is not None:
            self._cams.close()
            self._cams = None
        self._mj = mujoco
        m, ids = build_scene_model(scene, Path(path).resolve().parent, self.width, self.height, self.mesh_detail)
        self._bind_robot(m)
        self.scene, self._loaded, self.body = scene, key, ids
        self.objects = {o.name: o for o in scene.objects}
        children: dict[int, list[int]] = {}
        for b in range(m.nbody):
            children.setdefault(int(m.body_parentid[b]), []).append(b)

        def subtree(b: int) -> list[int]:
            out = [b]
            for c in children.get(b, []):
                if c != b:
                    out += subtree(c)
            return out

        self.items = {}
        for o in scene.objects:
            b = ids[o.name]
            info: dict[str, Any] = {"body": b, "subtree": subtree(b),
                                    "geoms": [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]}
            info["all_geoms"] = [g for g in range(m.ngeom) if m.geom_bodyid[g] in set(info["subtree"])]
            if o.movable:
                j = m.body_jntadr[b]
                info["qadr"], info["vadr"] = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
            elif o.parent is None:
                info["mocap"] = int(m.body_mocapid[b])
            if o.joint is not None:
                j = m.joint(f"{o.name}/joint").id
                info["jqadr"], info["jvadr"], info["hinge"] = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]), o.joint.type == "hinge"
            self.items[o.name] = info
        self.instance = {"table": 1, "robot": 2, **{o.name: 3 + i for i, o in enumerate(scene.objects)}}
        gi = np.zeros(m.ngeom, dtype=np.int32)
        robot = {b for b in range(m.nbody) if m.body(b).name.startswith("panda/")}
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in robot:
                gi[g] = 2
        gi[m.geom("table").id] = 1
        for name, info in self.items.items():
            gi[info["geoms"]] = self.instance[name]
        self.geom_instance = gi
        self._table_geom = m.geom("table").id
        self._robot_bodies = robot
        self._sig = mujoco.mjtState.mjSTATE_INTEGRATION
        self._warn = 0
        self._local_boxes()

    def _build(self) -> None:
        if not self.scene_path:
            raise ValueError("mujoco.panda_scene needs a scene: env param `scene` or task params['scene']")
        self._load(self.scene_path)

    def _local_boxes(self) -> None:
        """Each object's bounding box in its own frame (children included, joints at 0)."""
        mj, m = self._mj, self._model
        d = mj.MjData(m)
        mj.mj_kinematics(m, d)
        for name, info in self.items.items():
            R, p = d.xmat[info["body"]].reshape(3, 3), d.xpos[info["body"]]
            lo, hi = self._aabb(d, info["all_geoms"])
            pts = (_corners(lo, hi) - p) @ R
            info["local_lo"], info["local_hi"] = pts.min(0), pts.max(0)

    def _aabb(self, d: Any, geoms: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        m = self._model
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for g in geoms:
            c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
            R = d.geom_xmat[g].reshape(3, 3)
            ext = np.abs(R) @ h
            w = d.geom_xpos[g] + R @ c
            lo, hi = np.minimum(lo, w - ext), np.maximum(hi, w + ext)
        return lo, hi

    # -- episode ----------------------------------------------------------------------------------------------------
    def reset(self, task: TaskInstance) -> Observation:
        path = task.spec.params.get("scene") or self.scene_path
        if not path:
            raise ValueError("mujoco.panda_scene needs a scene: env param `scene` or task params['scene']")
        self.scene_path = str(path)
        self._load(self.scene_path)
        mj, m = self._mj, self._model
        d = self.data
        self.task = task
        mj.mj_resetData(m, d)
        rng = np.random.default_rng(task.seed)
        placements, problems = self._sample(rng)
        for name, (pos, quat) in placements.items():
            self._set_pose(name, pos, quat)
        for o in self.scene.objects:
            if o.joint is not None:
                v = o.joint.initial
                d.qpos[self.items[o.name]["jqadr"]] = math.radians(v) if o.joint.type == "hinge" else v
        d.qpos[self._qadr] = HOME
        d.qpos[self._finger_qadr] = FINGER_TRAVEL
        d.ctrl[:7] = HOME
        d.ctrl[7] = HAND_CTRL_MAX
        mj.mj_forward(m, d)
        start = self.penetrations()
        mj.mj_step(m, d, nstep=int(SETTLE_S / m.opt.timestep))
        d.time = 0.0
        mj.mj_forward(m, d)
        self._warn = self._divergence_count()
        self.t = 0
        self.ep = {"task": self.scene.name, "instruction": self.scene.instruction, "scene": self.scene_path,
                   "digest": self._loaded[1], "latch": {}, "placement_problems": problems, "start_contacts": start,
                   "placed": {k: [list(map(float, p)), list(map(float, q))] for k, (p, q) in placements.items()}}
        return self._observe()

    def _sample(self, rng: np.random.Generator) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[str]]:
        """Start poses of root objects: fixed poses first, then regions (rejection against what is placed),
        objects resting `on` others after their support."""
        roots = [o for o in self.scene.objects if o.parent is None]
        order: list[SceneObject] = []
        while len(order) < len(roots):
            progressed = False
            for o in roots:
                if o not in order and (o.on is None or self.objects[o.on] in order):
                    order.append(o)
                    progressed = True
            if not progressed:
                order += [o for o in roots if o not in order]
        order.sort(key=lambda o: (o.on is not None, o.region is not None))
        placed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        foot: dict[str, tuple[np.ndarray, np.ndarray]] = {}       # world AABB of each placed object
        problems: list[str] = []
        for o in order:
            info = self.items[o.name]
            lo, hi = info["local_lo"], info["local_hi"]
            support_top = TABLE_TOP
            sup_xy = None
            if o.on is not None and o.on in foot:
                slo, shi = foot[o.on]
                support_top, sup_xy = float(shi[2]), (slo[:2] + shi[:2]) / 2
            tries = 200 if o.region is not None else 1
            for k in range(tries):
                if o.region is not None:
                    xy = np.array([rng.uniform(*o.region.x), rng.uniform(*o.region.y)])
                    yaw, roll, pitch = rng.uniform(*o.region.yaw_deg), 0.0, 0.0
                    z = None
                else:
                    pose = o.pose
                    xy = np.asarray(pose.pos[:2], float) if pose is not None else np.asarray(sup_xy, float)
                    yaw, roll, pitch = (pose.yaw_deg, pose.roll_deg, pose.pitch_deg) if pose else (0.0, 0.0, 0.0)
                    z = pose.pos[2] if pose is not None and len(pose.pos) == 3 else None
                q = _quat_euler(math.radians(roll), math.radians(pitch), math.radians(yaw))
                pts = _corners(lo, hi) @ _mat(q).T
                if z is None:
                    z = support_top - pts[:, 2].min() + 0.0005
                pos = np.array([xy[0], xy[1], z])
                wlo, whi = pts.min(0) + pos, pts.max(0) + pos
                clash = [n for n, (a, b) in foot.items() if n != o.on and o.name != self.objects[n].on
                         and np.all(wlo[:2] < b[:2] + 0.005) and np.all(a[:2] < whi[:2] + 0.005)
                         and wlo[2] < b[2] - 0.002 and a[2] < whi[2] - 0.002]
                if not clash or o.region is None:
                    break
            else:
                problems.append(f"{o.name}: no free start position in its region after {tries} samples "
                                f"(overlaps {', '.join(clash)}); widen the region or move the other objects")
            placed[o.name] = (pos, q)
            foot[o.name] = (wlo, whi)
        return placed, problems

    def _set_pose(self, name: str, pos: Sequence[float], quat: Sequence[float]) -> None:
        info, d = self.items[name], self.data
        if "qadr" in info:
            a = info["qadr"]
            d.qpos[a:a + 3], d.qpos[a + 3:a + 7] = pos, quat
            d.qvel[info["vadr"]:info["vadr"] + 6] = 0.0
        elif "mocap" in info:
            d.mocap_pos[info["mocap"]], d.mocap_quat[info["mocap"]] = pos, quat

    def arrange(self, poses: dict[str, Any] | None = None, joints: dict[str, float] | None = None) -> None:
        """PRIVILEGED, for checkers only (never reachable by an agent): move movable objects to {name: (pos, quat)}
        and set joints {name: value in scene units}, at rest. The next step() simulates from there."""
        d = self.data
        for name, (pos, quat) in (poses or {}).items():
            if not self.objects[name].movable:
                raise ValueError(f"arrange: {name} is not a movable object")
            self._set_pose(name, pos, quat)
        for name, v in (joints or {}).items():
            info = self.items[name]
            d.qpos[info["jqadr"]] = math.radians(v) if info["hinge"] else v
            d.qvel[info["jvadr"]] = 0.0
        self._mj.mj_forward(self._model, d)

    def holds(self, predicate: dict[str, Any]) -> bool:
        """PRIVILEGED, for checkers only: does a predicate (hylo.scene/v0.1 JSON, no sequence) hold now?"""
        from pydantic import TypeAdapter

        from hylo.core.scene import Predicate

        return bool(self._eval(TypeAdapter(Predicate).validate_python(predicate), "check", {}))

    def _apply_episode(self) -> None:
        """Before a restore: the model of the episode's scene."""
        path = self.ep.get("scene")
        if path:
            self.scene_path = path
            self._load(path)
            if self._loaded[1] != self.ep.get("digest"):
                raise RuntimeError(f"{path} changed since the snapshot was taken")

    # -- step -------------------------------------------------------------------------------------------------------
    def step(self, action: Action) -> StepResult:
        mj, m, d = self._mj, self._model, self.data
        a = np.asarray(action.data, dtype=np.float64).reshape(-1)
        d.ctrl[:7] = np.clip(a[:7], LOWER, UPPER)
        d.ctrl[7] = HAND_CTRL_MAX * float(np.clip(a[7], 0.0, 1.0))
        mj.mj_step(m, d, nstep=self.n_substeps)
        mj.mj_forward(m, d)                              # poses and contacts of the state reached (as after restore)
        self.t += 1
        bad = self._divergence_count()
        if bad != self._warn:
            self._warn = bad
            return StepResult(observation=self._observe(), reward=None, terminated=False, truncated=True,
                              success=None, info={"sim_error": "mujoco divergence warning (BADQPOS/QVEL/QACC)"})
        success, info = self.evaluate()
        fail = self.fell()
        if fail:
            info["fail_reason"] = fail
        return StepResult(observation=self._observe(), reward=1.0 if success else 0.0,
                          terminated=bool(success or fail), truncated=False, success=bool(success and not fail),
                          info=info)

    def fell(self) -> str | None:
        for o in self.scene.objects:
            if o.movable:
                p = self.data.xpos[self.items[o.name]["body"]]
                if p[2] < FELL_Z:
                    return f"the {o.label} fell off the table"
        return None

    # -- scene queries ----------------------------------------------------------------------------------------------
    def pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        b = self.items[name]["body"]
        return self.data.xpos[b].copy(), self.data.xquat[b].copy()

    def speed(self, name: str) -> tuple[float, float]:
        v = np.zeros(6)
        self._mj.mj_objectVelocity(self._model, self.data, self._mj.mjtObj.mjOBJ_BODY, self.items[name]["body"], v, 0)
        return float(np.linalg.norm(v[3:])), float(np.linalg.norm(v[:3]))

    def _contact_bodies(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        m, d = self._model, self.data
        n = d.ncon
        g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
        return g1, g2, m.geom_bodyid[g1], m.geom_bodyid[g2]

    def touching_hand(self, name: str) -> bool:
        _, _, b1, b2 = self._contact_bodies()
        own = np.asarray(self.items[name]["subtree"])
        fingers = np.fromiter(self._fingers, dtype=b1.dtype)
        return bool(np.any((np.isin(b1, own) & np.isin(b2, fingers)) | (np.isin(b2, own) & np.isin(b1, fingers))))

    def touching(self, a: str, b: str) -> bool:
        g1, g2, _, _ = self._contact_bodies()
        ga = np.asarray(self.items[a]["all_geoms"])
        gb = np.asarray([self._table_geom]) if b == "table" else np.asarray(self.items[b]["all_geoms"])
        return bool(np.any((np.isin(g1, ga) & np.isin(g2, gb)) | (np.isin(g2, ga) & np.isin(g1, gb))))

    def bounds(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """World bounding box of an object (its children included)."""
        if name == "table":
            return np.array([TABLE_X[0], TABLE_Y[0], TABLE_TOP - 0.75]), np.array([TABLE_X[1], TABLE_Y[1], TABLE_TOP])
        return self._aabb(self.data, self.items[name]["all_geoms"])

    def joint_value(self, name: str) -> float:
        """Joint position in scene units (metres, or degrees for a hinge)."""
        info = self.items[name]
        v = float(self.data.qpos[info["jqadr"]])
        return math.degrees(v) if info["hinge"] else v

    def interior_box(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """(centre, half extents, rotation, floor z offset) of a tray / cup interior, in the world."""
        o = self.objects[name]
        if o.shape not in ("tray", "cup"):
            return None
        p, q = self.pose(name)
        if o.shape == "tray":
            half = np.array([o.size[0] / 2 - o.wall, o.size[1] / 2 - o.wall, (o.size[2] - o.wall) / 2])
        else:
            r = (o.size[0] - o.wall) / math.sqrt(2)          # square inscribed in the opening
            half = np.array([r, r, (o.size[1] - o.wall) / 2])
        R = _mat(q)
        return p + R @ np.array([0, 0, o.wall / 2]), half, R, o.wall

    def released(self, name: str) -> bool:
        lin, ang = self.speed(name)
        return lin < LIN_REST and ang < ANG_REST and not self.touching_hand(name)

    def penetrations(self, depth: float = PENETRATION) -> list[dict[str, Any]]:
        """Contacts deeper than `depth` between scene objects, the table and the robot (worst per pair)."""
        m, d = self._model, self.data
        owner = {}
        for name, info in self.items.items():
            for g in info["geoms"]:
                owner[g] = name
        owner[self._table_geom] = "table"
        worst: dict[tuple[str, str], float] = {}
        n = d.ncon
        dist, g1s, g2s = d.contact.dist[:n], d.contact.geom1[:n], d.contact.geom2[:n]
        for i in np.flatnonzero(dist < -depth):
            g1, g2 = int(g1s[i]), int(g2s[i])
            a = owner.get(g1, "robot" if m.geom_bodyid[g1] in self._robot_bodies else None)
            b = owner.get(g2, "robot" if m.geom_bodyid[g2] in self._robot_bodies else None)
            if a is None or b is None or a == b or {a, b} == {"robot"}:
                continue
            key = tuple(sorted((a, b)))
            worst[key] = max(worst.get(key, 0.0), -float(dist[i]))
        return [{"a": a, "b": b, "depth": round(v, 4)} for (a, b), v in sorted(worst.items())]

    # -- predicate --------------------------------------------------------------------------------------------------
    def evaluate(self) -> tuple[bool, dict[str, Any]]:
        """(success, info) for the scene's predicate now; advances `sequence` progress."""
        cache: dict[str, Any] = {}
        root = self.scene.success
        ok = self._eval(root, "0", cache)
        if isinstance(root, (AllOf, AnyOf)):
            terms = [bool(cache.get(f"0.{i}")) for i in range(len(root.terms))]
            done, of = sum(terms), len(terms)
        elif isinstance(root, Seq):
            done, of = int(self.ep["latch"].get("0", 0)), len(root.terms)
        else:
            done, of = int(ok), 1
        return ok, {"done": done, "of": of}

    def _box(self, name: str, cache: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        if ("box", name) not in cache:
            cache[("box", name)] = self.bounds(name)
        return cache[("box", name)]

    def _rest(self, name: str, need: bool, cache: dict[str, Any]) -> bool:
        if not need:
            return True
        if ("rest", name) not in cache:
            cache[("rest", name)] = self.released(name)
        return cache[("rest", name)]

    def _over(self, a: str, b: str, cache: dict[str, Any]) -> bool:
        c = self.pose(a)[0]
        lo, hi = self._box(b, cache)
        return bool(lo[0] <= c[0] <= hi[0] and lo[1] <= c[1] <= hi[1])

    def _eval(self, p: Any, path: str, cache: dict[str, Any]) -> bool:
        ok = self._eval_node(p, path, cache)
        cache[path] = ok
        return ok

    def _eval_node(self, p: Any, path: str, cache: dict[str, Any]) -> bool:
        if isinstance(p, AllOf):
            return all([self._eval(t, f"{path}.{i}", cache) for i, t in enumerate(p.terms)])
        if isinstance(p, AnyOf):
            return any([self._eval(t, f"{path}.{i}", cache) for i, t in enumerate(p.terms)])
        if isinstance(p, Not):
            return not self._eval(p.term, f"{path}.0", cache)
        if isinstance(p, Seq):
            k = int(self.ep["latch"].get(path, 0))
            while k < len(p.terms) and self._eval(p.terms[k], f"{path}.{k}", cache):
                k += 1
                if k == len(p.terms):
                    break
            self.ep["latch"][path] = k
            return k == len(p.terms)
        if isinstance(p, On):
            alo, _ = self._box(p.a, cache)
            _, bhi = self._box(p.b, cache)
            return (self.touching(p.a, p.b) and alo[2] >= bhi[2] - 0.012 and self._over(p.a, p.b, cache)
                    and self._rest(p.a, p.released, cache))
        if isinstance(p, Above):
            alo, _ = self._box(p.a, cache)
            _, bhi = self._box(p.b, cache)
            return alo[2] >= bhi[2] - 0.005 and self._over(p.a, p.b, cache) and self._rest(p.a, p.released, cache)
        if isinstance(p, Inside):
            c = self.pose(p.a)[0]
            if p.region is not None:
                r = p.region
                inside = r.x[0] <= c[0] <= r.x[1] and r.y[0] <= c[1] <= r.y[1] and (r.z is None or r.z[0] <= c[2] <= r.z[1])
            else:
                box = self.interior_box(p.b)
                if box is None:
                    lo, hi = self._box(p.b, cache)
                    inside = bool(np.all(lo <= c) and np.all(c <= hi))
                else:
                    centre, half, R, _ = box
                    local = R.T @ (c - centre)
                    inside = bool(np.all(np.abs(local[:2]) <= half[:2]) and -half[2] - 0.01 <= local[2] <= half[2] + 0.01)
            return inside and self._rest(p.a, p.released, cache)
        if isinstance(p, Near):
            c = self.pose(p.a)[0]
            t = self.pose(p.b)[0] if p.b is not None else np.array([*p.point, *([] if len(p.point) == 3 else [c[2]])])
            dist = np.linalg.norm((c - t)[:2] if p.xy_only else c - t)
            return bool(dist <= p.max_dist) and self._rest(p.a, p.released, cache)
        if isinstance(p, Upright):
            R = _mat(self.pose(p.a)[1])
            k = "xyz".index(p.axis)
            return abs(R[2, k]) >= math.cos(math.radians(p.max_tilt_deg)) and self._rest(p.a, p.released, cache)
        if isinstance(p, Lifted):
            return bool(self._box(p.a, cache)[0][2] >= TABLE_TOP + p.height)
        if isinstance(p, Row):
            k, j = "xy".index(p.axis), 1 - "xy".index(p.axis)
            cs = np.array([self.pose(n)[0] for n in p.objects])
            if not all(self._rest(n, p.released, cache) for n in p.objects):
                return False
            along = cs[:, k]
            gaps = np.diff(along)
            return bool(np.all(gaps > max(p.min_gap, 1e-3)) and np.all(np.abs(cs[:, j] - cs[:, j].mean()) <= p.max_offset))
        if isinstance(p, (JointAtLeast, JointAtMost)):
            j = self.objects[p.object].joint
            v = self.joint_value(p.object)
            target = p.value if p.value is not None else j.range[0] + p.fraction * (j.range[1] - j.range[0])
            return v >= target if isinstance(p, JointAtLeast) and not isinstance(p, JointAtMost) else v <= target
        raise ValueError(f"unknown predicate {p!r}")

    # -- observation ------------------------------------------------------------------------------------------------
    def segments(self) -> dict[str, str]:
        out = {"1": "table", "2": "robot"}
        if self.scene is not None:
            for o in self.scene.objects:
                out[str(self.instance[o.name])] = o.label
        return out

    def privileged_state(self, json_ready: bool = False) -> dict[str, Any]:
        objects = {}
        for o in self.scene.objects:
            p, q = self.pose(o.name)
            lo, hi = self.bounds(o.name)
            info = self.items[o.name]
            rec: dict[str, Any] = {
                "label": o.label, "shape": o.shape or "compound", "pos": p.tolist(), "quat": q.tolist(),
                "yaw": float(math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))),
                "aabb": [lo.tolist(), hi.tolist()], "movable": o.movable, "fixed": o.fixed, "parent": o.parent,
                "instance": self.instance[o.name], "colour": o.colour if isinstance(o.colour, str) else list(o.colour)}
            rec["extent"] = (info["local_hi"] - info["local_lo"]).tolist()
            rec["local_box"] = [info["local_lo"].tolist(), info["local_hi"].tolist()]
            if o.shape is not None:
                rec["size"] = list(o.size)
            box = self.interior_box(o.name)
            if box is not None:
                rec["interior"] = {"centre": box[0].tolist(), "half": box[1].tolist()}
            if o.joint is not None:
                rec["joint"] = {"type": o.joint.type, "value": self.joint_value(o.name), "range": list(o.joint.range),
                                "units": "deg" if o.joint.type == "hinge" else "m"}
            if o.movable:
                rec["at_rest"] = bool(self.released(o.name))
            objects[o.name] = rec
        out = {"task": self.ep.get("task"), "scene": self.scene_path, "objects": objects,
               "table": {"height": TABLE_TOP, "x": list(TABLE_X), "y": list(TABLE_Y)},
               "success": json.loads(self.scene.success.model_dump_json()),
               "progress": dict(self.ep.get("latch", {})), "penetration": self.penetrations(),
               "start_contacts": list(self.ep.get("start_contacts", [])),
               "placement_problems": list(self.ep.get("placement_problems", [])),
               "placed": {k: {"pos": v[0], "quat": v[1]} for k, v in self.ep.get("placed", {}).items()},
               "q": self.data.qpos[self._qadr].tolist(), "hand": self.hand_state(), "tcp": self.tcp_pose().tolist(),
               "instruction": self.ep.get("instruction")}
        return json.loads(json.dumps(out, default=float)) if json_ready else out


# -- benchmark ----------------------------------------------------------------------------------------------------
def scene_tasks(directory: str | Path, pattern: str = "*.json") -> tuple[TaskSpec, ...]:
    """One task per scene file in `directory` (sorted by file name); a scene that fails validation is an error."""
    root = Path(directory).expanduser().resolve()
    files = sorted(root.glob(pattern)) if root.is_dir() else [root]
    if not files:
        raise ValueError(f"no scene files matching {pattern!r} in {root}")
    out = []
    for f in files:
        s = load_scene(f)
        out.append(TaskSpec(id=f"mujoco/scene_{s.name}", version=scene_digest(s)[:8], instruction=s.instruction,
                            params={"scene": str(f)}, max_steps=s.max_steps, tags=("manipulation", "scene")))
    ids = [t.id for t in out]
    if len(set(ids)) != len(ids):
        raise ValueError(f"scene names must be unique within {root}: {sorted(ids)}")
    return tuple(out)


VALIDATION_SEED_OFFSET = 20_000


def scene_benchmark(directory: str, pattern: str = "*.json", protocol: str = "rsi", rsi_states: int = 50,
                    validation: int = 20, train: int = 10, heldout: int = 20, gate_looks: int = 1):
    """`mujoco/scenes-v0`: the scene files of `directory` as tasks; seeds vary the sampled start poses. protocol
    "rsi": search / search_val / gate / heldout (RFC-0002) plus `validation`; "standard": train / validation /
    heldout. All seed ranges are disjoint."""
    from hylo.core.benchmark import Benchmark

    from .planar_reach import splits_for

    splits = splits_for(protocol, train, validation, heldout, rsi_states, gate_looks)
    if protocol == "rsi":
        splits["validation"] = tuple(range(VALIDATION_SEED_OFFSET, VALIDATION_SEED_OFFSET + validation))
    return Benchmark(id="mujoco/scenes-v0", version="0.1", tasks=scene_tasks(directory, pattern), splits=splits,
                     metadata={"env": "mujoco.panda_scene", "protocol": protocol, "directory": str(directory)})


__all__ = ["PandaSceneEnv", "REACH", "build_scene_model", "load_scene", "scene_benchmark", "scene_digest", "scene_tasks"]
