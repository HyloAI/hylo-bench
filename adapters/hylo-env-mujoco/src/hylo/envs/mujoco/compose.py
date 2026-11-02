"""A stage (hylo.stage/v0.1) as one MuJoCo model, built with MjSpec (RFC-0008 part 2).

Entities are built parents first. Each becomes a body named by its id (robots: the pack's model attached under a
frame with the prefix "<id>/", so a robot's joints, actuators, sites and bodies are "<id>/<name>"):

```text
  shape    a body with one primitive geom; `body.dynamic` adds a free joint; collision `none` makes it visual only
  mesh     a body with a mesh geom (MuJoCo collides with its convex hull); collision `parts` (convex pieces) or
           `voxels` (boxes) from a physicalize file become collision-only geoms and the mesh stays visual only
  robot    the pack's model: fixed base (its free joint removed) attached at the pose, or onto `mount.entity`'s
           body `mount.body`; free base keeps (or gains) a free joint; the pack's physics options carry over
           (elliptic cones, impratio, integrator) when it asks for them
  camera   a camera on the parent's body (or `camera.body` of a robot parent); MuJoCo convention: it looks along
           its -z axis with +y up (Pose of a camera = that frame; `look_at` builds one)
  light    a directional or spot light
  human    a kinematic person (mocap body: capsule legs, torso, head) moved by the crowd controller
```

The home posture of every robot (its keyframe) is applied after compiling, by joint and actuator name, so robots
from different models never mix keyframes. A free-base robot's pose gives x, y and yaw; its base starts at the
pose's z plus the keyframe's base height (without a keyframe: plus the model's own base height).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from hylo.core.stage import Entity, Stage

from .packs import Pack, pack

_GEOM = {"box": mujoco.mjtGeom.mjGEOM_BOX, "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
         "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER, "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
         "plane": mujoco.mjtGeom.mjGEOM_PLANE}
COLLISION_GROUP = 3


@dataclass
class RobotInfo:
    id: str
    pack: Pack
    home_qpos: dict[str, np.ndarray] = field(default_factory=dict)   # joint name (prefixed) -> qpos slice
    home_ctrl: dict[str, float] = field(default_factory=dict)        # actuator name (prefixed) -> ctrl
    base_height: float | None = None                                 # keyframe base height (free base), if any
    free: bool = False
    pose_z: float = 0.0


@dataclass
class Composed:
    spec: mujoco.MjSpec
    robots: dict[str, RobotInfo]
    cameras: dict[str, dict[str, Any]]
    humans: list[str]
    warnings: list[str]


def look_at(eye: Any, target: Any, up: Any = (0.0, 0.0, 1.0)) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) of a MuJoCo camera at `eye` looking at `target` (-z forward, +y up)."""
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    back = eye - target
    back /= np.linalg.norm(back)
    x = np.cross(up, back)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross([0.0, 1.0, 0.0], back)
    x /= np.linalg.norm(x)
    y = np.cross(back, x)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([x, y, back]).ravel())
    return tuple(float(v) for v in q)


def _home(p: Pack, prefix: str) -> tuple[dict[str, np.ndarray], dict[str, float], float | None, bool]:
    """Keyframe values of a pack's model, keyed by prefixed joint / actuator names; base height; free base?"""
    m = mujoco.MjModel.from_xml_path(str(p.model_path()))
    qpos = m.qpos0.copy()
    ctrl = np.zeros(m.nu)
    if p.home:
        k = m.key(p.home).id
        qpos, ctrl = m.key_qpos[k].copy(), m.key_ctrl[k].copy()
    joints: dict[str, np.ndarray] = {}
    free, height = False, None
    for j in range(m.njnt):
        a = int(m.jnt_qposadr[j])
        n = {0: 7, 1: 4, 2: 1, 3: 1}[int(m.jnt_type[j])]
        if int(m.jnt_type[j]) == 0:
            free = True
            height = float(qpos[a + 2]) if p.home else None
            continue
        joints[prefix + m.joint(j).name] = qpos[a:a + n].copy()
    acts = {prefix + m.actuator(i).name: float(ctrl[i]) for i in range(m.nu)}
    if not p.home:                                  # no keyframe: servos hold the zero posture
        for i in range(m.nu):
            if int(m.actuator_trntype[i]) == 0:
                lo, hi = m.actuator_ctrlrange[i] if m.actuator_ctrllimited[i] else (-np.inf, np.inf)
                acts[prefix + m.actuator(i).name] = float(np.clip(0.0, lo, hi))
    return joints, acts, height, free


def _carry_options(parent: mujoco.MjSpec, child: mujoco.MjSpec) -> None:
    if child.option.cone == mujoco.mjtCone.mjCONE_ELLIPTIC:
        parent.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    parent.option.impratio = max(parent.option.impratio, child.option.impratio)
    if child.option.integrator != mujoco.mjtIntegrator.mjINT_EULER:
        parent.option.integrator = child.option.integrator


def compose(stage: Stage, root: str | Path = ".") -> Composed:
    root = Path(root)
    spec = mujoco.MjSpec()
    spec.modelname = stage.name
    spec.option.timestep = stage.timestep
    spec.option.gravity = list(stage.gravity)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 960
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    far = max([float(np.linalg.norm(e.pose.pos[:2])) for e in stage.entities] + [1.0])
    spec.stat.extent = max(3.0, 2.0 * far + 1.0)         # the shadow map covers this much of the scene
    spec.visual.map.shadowclip = 6.0                    # no shadow-map speckles on the far floor
    spec.visual.quality.shadowsize = 4096
    spec.visual.map.znear = 0.01
    spec.add_texture(name="hylo_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.93, 0.95, 0.97], rgb2=[0.76, 0.80, 0.86], width=512, height=3072)
    world = spec.worldbody
    if stage.ground:                                   # a 50 cm checker: scale is readable in every picture
        spec.add_texture(name="hylo_grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=[0.83, 0.85, 0.87], rgb2=[0.76, 0.78, 0.81],
                         width=300, height=300, mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=[0.66, 0.68, 0.72])
        mat = spec.add_material(name="hylo_grid", texrepeat=[1, 1], texuniform=True, reflectance=0.0)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "hylo_grid"
        world.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="hylo_grid")
    if not any(e.light for e in stage.entities):
        world.add_light(name="sun", pos=[0, 0, 4], dir=[0.2, 0.2, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                        diffuse=[0.6, 0.6, 0.6])
    bodies: dict[str, Any] = {}
    robots: dict[str, RobotInfo] = {}
    cameras: dict[str, dict[str, Any]] = {}
    humans: list[str] = []
    warnings: list[str] = []

    def parent_body(e: Entity) -> Any:
        if e.robot and e.robot.mount:
            mt = e.robot.mount
            if mt.body:
                b = spec.body(f"{mt.entity}/{mt.body}")
                if b is None:
                    raise ValueError(f"entity {e.id}: mount body {mt.body!r} not found on {mt.entity}")
                return b
            return bodies[mt.entity]
        if e.parent is None:
            return world
        return bodies[e.parent]

    for e in stage.ordered():
        pos, quat = list(e.pose.pos), list(e.pose.quat)
        parent = parent_body(e)
        if e.robot is not None:
            p = pack(e.robot.pack)
            child = mujoco.MjSpec.from_file(str(p.model_path()))
            _carry_options(spec, child)
            for k in list(child.keys):
                child.delete(k)
            prefix = f"{e.id}/"
            home_q, home_c, height, has_free = _home(p, prefix)
            free = e.robot.base == "free"
            if not free:
                for j in list(child.joints):
                    if j.type == mujoco.mjtJoint.mjJNT_FREE:
                        child.delete(j)
            elif not has_free:
                child.worldbody.first_body().add_freejoint()
            for name, v in e.robot.joints.items():
                home_q[prefix + name] = np.atleast_1d(np.asarray(v, float))
            frame = parent.add_frame(pos=pos, quat=quat)
            spec.attach(child, prefix=prefix, frame=frame)
            robots[e.id] = RobotInfo(e.id, p, home_q, home_c, height, free, pose_z=float(pos[2]))
            bodies[e.id] = spec.body(prefix + child.worldbody.first_body().name) if child.worldbody.first_body() \
                else parent
            continue
        body = parent.add_body(name=e.id, pos=pos, quat=quat)
        bodies[e.id] = body
        if e.shape is not None or e.mesh is not None:
            dyn = e.body is not None and e.body.dynamic
            if dyn:
                body.add_freejoint()
            coll = e.collision.strategy if e.collision else ("primitive" if e.shape else "hull")
            kw: dict[str, Any] = {"friction": [e.body.friction if e.body else 1.0, 0.005, 0.0001]}
            if e.body is not None:
                kw["density"] = e.body.density
            if e.shape is not None:
                g = body.add_geom(name=e.id, type=_GEOM[e.shape.type], size=_size(e.shape.type, e.shape.size),
                                  rgba=list(e.shape.rgba), **kw)
            else:
                mpath = Path(e.mesh.path)
                mpath = mpath if mpath.is_absolute() else root / mpath
                spec.add_mesh(name=f"{e.id}_mesh", file=str(mpath), scale=list(e.mesh.scale))
                g = body.add_geom(name=e.id, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{e.id}_mesh",
                                  rgba=list(e.mesh.rgba or (0.7, 0.7, 0.72, 1.0)), **kw)
            mass = e.body.mass if (dyn and e.body is not None) else None
            if coll in ("none", "parts", "voxels"):
                g.contype, g.conaffinity = 0, 0
                if coll != "none":
                    g.density = 0.0                             # mass comes from the collision pieces
                    _collision_parts(spec, body, e, root, kw, mass)
                elif mass is not None:
                    g.mass = float(mass)
            elif mass is not None:
                g.mass = float(mass)
        elif e.camera is not None:
            target = body
            if e.camera.body:
                rb = spec.body(f"{e.parent}/{e.camera.body}") if e.parent else None
                if rb is None:
                    raise ValueError(f"camera {e.id}: body {e.camera.body!r} not found on {e.parent}")
                target = rb.add_body(name=f"{e.id}_mount", pos=pos, quat=quat)
                body.pos, body.quat = [0, 0, 0], [1, 0, 0, 0]
            target.add_camera(name=e.id, fovy=e.camera.fovy_deg)
            cameras[e.id] = {"width": e.camera.width, "height": e.camera.height, "depth": e.camera.depth}
        elif e.light is not None:
            d = np.asarray(e.light.direction, float)
            body.add_light(name=e.id, dir=list(d / (np.linalg.norm(d) or 1.0)),
                           type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL if e.light.kind == "directional"
                           else mujoco.mjtLightType.mjLIGHT_SPOT, diffuse=[e.light.intensity] * 3,
                           castshadow=True)
        elif e.human is not None:
            _human(body, e)
            body.mocap = True
            humans.append(e.id)
    return Composed(spec, robots, cameras, humans, warnings)


def _size(kind: str, size: tuple[float, ...]) -> list[float]:
    s = list(size)
    if kind == "plane":
        return [s[0], s[1], 0.05]
    return s + [0.0] * (3 - len(s))


def _collision_parts(spec: mujoco.MjSpec, body: Any, e: Entity, root: Path, kw: dict[str, Any],
                     mass: float | None) -> None:
    path = Path(e.collision.parts)
    data = json.loads((path if path.is_absolute() else root / path).read_text())
    scale = np.asarray(e.mesh.scale if e.mesh else (1.0, 1.0, 1.0), float)
    n = len(data.get("parts") or data.get("boxes") or [1])
    if mass is None and data.get("mass_kg"):                 # the mass physicalize measured from the mesh volume
        mass = float(data["mass_kg"]) * float(np.prod(scale))
    if mass is not None:
        kw = {**{k: v for k, v in kw.items() if k != "density"}, "mass": float(mass) / n}
    if e.collision.strategy == "parts":
        for i, part in enumerate(data["parts"]):
            v = (np.asarray(part["vertices"], float) * scale).ravel().tolist()
            spec.add_mesh(name=f"{e.id}_part{i}", uservert=v)
            body.add_geom(name=f"{e.id}_c{i}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{e.id}_part{i}",
                          group=COLLISION_GROUP, rgba=[0.3, 0.6, 0.9, 0.4], **kw)
    else:
        for i, b in enumerate(data["boxes"]):
            c, h = np.asarray(b[:3], float) * scale, np.abs(np.asarray(b[3:6], float) * scale)
            body.add_geom(name=f"{e.id}_c{i}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=c.tolist(), size=h.tolist(),
                          group=COLLISION_GROUP, rgba=[0.3, 0.6, 0.9, 0.4], **kw)


def _human(body: Any, e: Entity) -> None:
    """A person of the given height from capsules: two legs, torso, two arms, head (the body origin on the floor)."""
    h, rgba = e.human.height, list(e.human.rgba)
    skin = [0.85, 0.7, 0.6, 1.0]
    leg = 0.47 * h
    for side in (-1, 1):
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, side * 0.09 * h / 1.75, 0.05,
                                                                  0, side * 0.09 * h / 1.75, leg], size=[0.06, 0, 0],
                      rgba=[0.2, 0.22, 0.3, 1])
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, side * 0.2 * h / 1.75, 0.82 * h - 0.05,
                                                                  0, side * 0.21 * h / 1.75, 0.52 * h],
                      size=[0.045, 0, 0], rgba=rgba)
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, leg + 0.05, 0, 0, 0.8 * h], size=[0.15, 0, 0],
                  rgba=rgba)
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=[0, 0, 0.91 * h], size=[0.1 * h / 1.75, 0, 0], rgba=skin)


def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


__all__ = ["Composed", "RobotInfo", "compose", "look_at", "yaw_quat"]
