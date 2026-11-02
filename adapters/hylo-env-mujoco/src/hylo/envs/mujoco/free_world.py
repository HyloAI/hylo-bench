"""The free-form state of a compiled free scene, and the `sim` views scene programs read and edit it through.

    world = SceneWorld(model, scene_bodies, robot_bodies, finger_bodies, labels)
    world.state(data)            -> {"bodies", "joints", "cables", "flexes", "sites"}   (privileged, JSON-ready)
    world.touching(data, "rope", "hand"); world.contacts(data, depth=0.003)
    SimView(world, data, table_top)                 `sim` in success(sim) and targets(sim): read-only
    SimEdit(world, data, table_top, forward, hands)  `sim` in randomize(sim, rng) and demo(sim): edits too
    DemoHands(world, n)          invisible mocap hands that hold a body through a ball-joint equality (demo only)

Nothing here knows a robot or an episode: SceneWorld indexes the program's part of the model once (rigid bodies,
composite cables found by their `<prefix>B_first` body, flexes and their vertex bodies, hinge / slide joints, one
segmentation group per cable, flex and rigid root), and every query takes the MjData to read.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Sequence

import numpy as np

from .free_model import DEMO_HAND, flex_name
from .free_program import SceneProgramError

PENETRATION = 0.003                       # m: deeper contacts are reported


class SceneWorld:
    """Index of the program's part of a compiled model, and queries over a MjData."""

    def __init__(self, model: Any, scene_bodies: set[str], robot_bodies: set[int], finger_bodies: set[int],
                 labels: dict[str, str] | None = None) -> None:
        m = self.model = model
        self.labels = dict(labels or {})
        self.scene_ids = {int(m.body(n).id) for n in scene_bodies}
        self.robot_ids = set(map(int, robot_bodies))
        self.finger_ids = set(map(int, finger_bodies))
        self.table_geom = int(m.geom("table").id)
        self._children: dict[int, list[int]] = {}
        for b in range(1, m.nbody):
            self._children.setdefault(int(m.body_parentid[b]), []).append(b)
        self.cables: dict[str, list[int]] = {}
        for b in sorted(self.scene_ids):
            mt = re.match(r"^(.*)B_first$", m.body(b).name)
            if mt:
                self.cables[mt.group(1)] = self._chain_ids(mt.group(1))
        self.flexes: dict[str, dict[str, Any]] = {}
        in_flex: set[int] = set()
        for f in range(m.nflex):
            a, k = int(m.flex_vertadr[f]), int(m.flex_vertnum[f])
            vb = [int(b) for b in m.flex_vertbodyid[a:a + k]]
            in_flex |= set(vb) & self.scene_ids
            self.flexes[flex_name(m, f)] = {"id": f, "bodies": vb if len(set(vb)) == len(vb) else None,
                                            "dim": int(m.flex_dim[f])}
        in_cable = {b for v in self.cables.values() for b in v}
        self.rigid = sorted(b for b in self.scene_ids if b not in in_cable and b not in in_flex)
        self.joints: dict[str, dict[str, Any]] = {}
        for j in range(m.njnt):
            b, t = int(m.jnt_bodyid[j]), int(m.jnt_type[j])
            if b in self.rigid and t in (2, 3):                                  # slide, hinge
                self.joints[m.joint(j).name or f"joint{j}"] = {
                    "id": j, "type": "slide" if t == 2 else "hinge", "body": m.body(b).name,
                    "qadr": int(m.jnt_qposadr[j]), "vadr": int(m.jnt_dofadr[j]),
                    "range": [float(v) for v in m.jnt_range[j]] if m.jnt_limited[j] else None}
        groups: dict[str, set[int]] = {p: set(v) for p, v in self.cables.items()}
        for name, f in self.flexes.items():
            if f["bodies"]:
                groups[name] = set(f["bodies"]) & self.scene_ids
        rigid = set(self.rigid)
        for b in self.rigid:
            root = b
            while int(m.body_parentid[root]) in rigid:
                root = int(m.body_parentid[root])
            groups.setdefault(m.body(root).name, set()).add(b)
        self.groups = groups
        self._owner = {b: g for g, members in groups.items() for b in members}

    def _chain_ids(self, prefix: str) -> list[int]:
        m = self.model

        def key(name: str) -> float:
            tail = name[len(prefix) + 2:]
            return -1.0 if tail == "first" else math.inf if tail == "last" else float(tail) if tail.isdigit() else 1e9

        names = [m.body(b).name for b in self.scene_ids if m.body(b).name.startswith(prefix + "B_")]
        return [int(m.body(n).id) for n in sorted(names, key=key)]

    # -- names --------------------------------------------------------------------------------------------------
    def label(self, group: str) -> str:
        """The words for a group (LABELS, else its name with spaces)."""
        return self.labels.get(group, group.replace("_", " ").strip())

    def body_id(self, name: str) -> int:
        try:
            return int(self.model.body(name).id)
        except KeyError:
            names = sorted(self.model.body(b).name for b in self.scene_ids)
            raise KeyError(f"no body {name!r}; scene bodies: {names[:30]}") from None

    def chain(self, prefix: str) -> list[int]:
        """Body ids of a composite cable in chain order, else the scene bodies whose names start with prefix."""
        if prefix in self.cables:
            return list(self.cables[prefix])
        m = self.model
        return sorted(b for b in self.scene_ids if m.body(b).name.startswith(prefix))

    def subtree(self, b: int) -> list[int]:
        out = [b]
        for c in self._children.get(b, []):
            out += self.subtree(c)
        return out

    def instances(self, geom_instance: np.ndarray, base: int) -> tuple[np.ndarray, dict[str, int]]:
        """Segmentation ids: a copy of `geom_instance` with one id per group from `base` on, and {group: id}."""
        ids = {g: base + i for i, g in enumerate(sorted(self.groups))}
        gi = np.asarray(geom_instance, np.int32).copy()
        for g in range(self.model.ngeom):
            owner = self._owner.get(int(self.model.geom_bodyid[g]))
            if owner is not None:
                gi[g] = ids[owner]
        return gi, ids

    def flex_topology(self, name: str) -> np.ndarray:
        """Elements of a flex (E, dim + 1) as vertex indices (triangles for cloth)."""
        m, f = self.model, self.flexes[name]["id"]
        dim, n = int(m.flex_dim[f]), int(m.flex_elemnum[f])
        a = int(m.flex_elemdataadr[f])
        return np.asarray(m.flex_elem[a:a + n * (dim + 1)]).reshape(n, dim + 1)

    # -- queries ------------------------------------------------------------------------------------------------
    def body_bounds(self, data: Any, b: int) -> tuple[np.ndarray, np.ndarray]:
        """World bounding box of a body's geoms and its children's (its frame origin if it has none)."""
        m = self.model
        members = set(self.subtree(b))
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) in members:
                R = data.geom_xmat[g].reshape(3, 3)
                w = data.geom_xpos[g] + R @ m.geom_aabb[g, :3]
                ext = np.abs(R) @ m.geom_aabb[g, 3:]
                lo, hi = np.minimum(lo, w - ext), np.maximum(hi, w + ext)
        if not np.all(np.isfinite(lo)):
            return data.xpos[b].copy(), data.xpos[b].copy()
        return lo, hi

    def thing(self, name: str) -> tuple[set[int], set[int]]:
        """(bodies, flexes) a name stands for: "table", "robot", "hand", a flex, a cable prefix or a body."""
        if name == "table":
            return {0}, set()
        if name == "robot":
            return set(self.robot_ids), set()
        if name == "hand":
            return set(self.finger_ids), set()
        if name in self.flexes:
            return set(), {self.flexes[name]["id"]}
        if name in self.cables:
            return set(self.cables[name]), set()
        try:
            return set(self.subtree(int(self.model.body(name).id))), set()
        except KeyError:
            raise KeyError(f"touching: no body, cable or flex {name!r}") from None

    def touching(self, data: Any, a: str, b: str) -> bool:
        """Is any contact between thing `a` and thing `b` active? (World geoms other than the table count as
        neither.)"""
        m = self.model
        (ba, fa), (bb, fb) = self.thing(a), self.thing(b)

        def side(g: int, f: int) -> tuple[int | None, int | None]:
            if g < 0:
                return None, f
            body = int(m.geom_bodyid[g])
            return (-1 if body == 0 and g != self.table_geom else body), None

        def hit(s: tuple[int | None, int | None], bodies: set[int], flexes: set[int]) -> bool:
            return (s[0] is not None and s[0] in bodies) or (s[1] is not None and s[1] in flexes)

        for i in range(data.ncon):
            c = data.contact[i]
            s0, s1 = side(int(c.geom[0]), int(c.flex[0])), side(int(c.geom[1]), int(c.flex[1]))
            if (hit(s0, ba, fa) and hit(s1, bb, fb)) or (hit(s1, ba, fa) and hit(s0, bb, fb)):
                return True
        return False

    def contacts(self, data: Any, depth: float = PENETRATION) -> list[dict[str, Any]]:
        """Contacts deeper than `depth` involving the program's things (with each other, the table or the robot),
        as [{a, b, depth}] by group name ("table", "robot" for the others)."""
        m = self.model
        out: dict[tuple[str, str], float] = {}

        def who(g: int, f: int) -> tuple[str, bool]:
            if g < 0:
                return flex_name(m, f), True
            b = int(m.geom_bodyid[g])
            if g == self.table_geom:
                return "table", False
            if b in self.robot_ids:
                return "robot", False
            if b in self.scene_ids:
                return self._owner.get(b, m.body(b).name), True
            return m.body(b).name or "world", False

        for i in range(data.ncon):
            c = data.contact[i]
            if c.dist > -depth:
                continue
            (a, sa), (b, sb) = who(int(c.geom[0]), int(c.flex[0])), who(int(c.geom[1]), int(c.flex[1]))
            if (sa or sb) and a != b:
                key = tuple(sorted((a, b)))
                out[key] = max(out.get(key, 0.0), float(-c.dist))
        return [{"a": a, "b": b, "depth": round(v, 4)} for (a, b), v in sorted(out.items())]

    def state(self, data: Any) -> dict[str, Any]:
        """The program's things now (JSON-ready lists): rigid bodies (pose, world box, free / mocap, parent, label),
        hinge and slide joints, cables (nodes in chain order, radius, whether the first end is anchored), flexes
        (vertices, vertex bodies, radius) and named sites."""
        m, d = self.model, data
        bodies = {}
        for b in self.rigid:
            name = m.body(b).name
            lo, hi = self.body_bounds(d, b)
            parent, j = int(m.body_parentid[b]), int(m.body_jntadr[b])
            bodies[name] = {"pos": d.xpos[b].tolist(), "quat": d.xquat[b].tolist(), "bounds": [lo.tolist(), hi.tolist()],
                            "top": float(hi[2]), "free": bool(j >= 0 and int(m.jnt_type[j]) == 0),
                            "mocap": bool(m.body_mocapid[b] >= 0),
                            "parent": m.body(parent).name if parent in self.scene_ids else None, "label": self.label(name)}
        joints = {name: {"type": j["type"], "value": float(d.qpos[j["qadr"]]), "range": j["range"], "body": j["body"],
                         "units": "rad" if j["type"] == "hinge" else "m"} for name, j in self.joints.items()}
        cables = {}
        for p, ids in self.cables.items():
            members = set(ids)
            r = max((float(m.geom_size[g][0]) for g in range(m.ngeom) if int(m.geom_bodyid[g]) in members), default=0.0)
            cables[p] = {"nodes": d.xpos[ids].tolist(), "bodies": [m.body(b).name for b in ids], "radius": r,
                         "anchored": bool(int(m.body_jntadr[ids[0]]) < 0), "label": self.label(p)}
        flexes = {}
        for name, f in self.flexes.items():
            a, k = int(m.flex_vertadr[f["id"]]), int(m.flex_vertnum[f["id"]])
            flexes[name] = {"vertices": d.flexvert_xpos[a:a + k].tolist(), "dim": f["dim"],
                            "bodies": [m.body(b).name for b in f["bodies"]] if f["bodies"] else None,
                            "radius": float(m.flex_radius[f["id"]]), "label": self.label(name)}
        sites = {m.site(s).name: d.site_xpos[s].tolist() for s in range(m.nsite)
                 if int(m.site_bodyid[s]) in self.scene_ids and m.site(s).name}
        return {"bodies": bodies, "joints": joints, "cables": cables, "flexes": flexes, "sites": sites}

    # -- edits --------------------------------------------------------------------------------------------------
    def set_body(self, data: Any, b: int, pos: Sequence[float] | None, quat: Sequence[float] | None) -> None:
        """Move a free-joint or mocap body, at rest (call mj_forward afterwards)."""
        m, d = self.model, data
        q = None if quat is None else np.asarray(quat, float) / np.linalg.norm(quat)
        mocap = int(m.body_mocapid[b])
        if mocap >= 0:
            if pos is not None:
                d.mocap_pos[mocap] = pos
            if q is not None:
                d.mocap_quat[mocap] = q
            return
        j = int(m.body_jntadr[b])
        if j < 0 or int(m.jnt_type[j]) != 0:
            raise SceneProgramError(f"set_body: {m.body(b).name} has no free joint (and is not a mocap body)")
        a, v = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        if pos is not None:
            d.qpos[a:a + 3] = pos
        if q is not None:
            d.qpos[a + 3:a + 7] = q
        d.qvel[v:v + 6] = 0.0

    def set_joint(self, data: Any, name: str, value: float) -> None:
        """Set any named joint's (first) coordinate, at rest (call mj_forward afterwards)."""
        j = self.model.joint(name).id
        data.qpos[self.model.jnt_qposadr[j]] = float(value)
        data.qvel[self.model.jnt_dofadr[j]] = 0.0


class DemoHands:
    """Invisible hands for demo(sim): mocap bodies (free_model.add_demo_hands) whose ball-joint equality is
    re-targeted at run time to the body they grab. `step()` (given by the environment) advances one control step
    and returns whether success(sim) holds; the first such step is remembered."""

    def __init__(self, world: SceneWorld, n: int) -> None:
        m = world.model
        self.world = world
        self.hands = {i: {"mocap": int(m.body_mocapid[m.body(f"{DEMO_HAND}{i}").id]),
                          "eq": int(m.equality(f"{DEMO_HAND}{i}/eq").id)} for i in range(n)}
        self.log: list[dict[str, Any]] = []
        self.steps, self.success_at = 0, None

    def reset(self, data: Any) -> None:
        for h in self.hands.values():
            data.eq_active[h["eq"]] = 0
        self.log, self.steps, self.success_at = [], 0, None

    def grab(self, data: Any, hand: int, body: int, point: Sequence[float] | None = None) -> None:
        if hand not in self.hands:
            raise SceneProgramError(f"demo: hand {hand} does not exist (this check has hands {sorted(self.hands)})")
        m, h = self.world.model, self.hands[hand]
        p = data.xpos[body].copy() if point is None else np.asarray(point, float)
        R, x = data.xmat[body].reshape(3, 3), data.xpos[body]
        m.eq_obj2id[h["eq"]] = body
        m.eq_data[h["eq"], 0:3] = 0.0
        m.eq_data[h["eq"], 3:6] = R.T @ (p - x)
        data.mocap_pos[h["mocap"]] = p
        data.eq_active[h["eq"]] = 1
        self.log.append({"grab": m.body(body).name, "hand": hand, "point": p.tolist()})

    def release(self, data: Any, hand: int) -> None:
        data.eq_active[self.hands[hand]["eq"]] = 0

    def position(self, data: Any, hand: int) -> np.ndarray:
        return data.mocap_pos[self.hands[hand]["mocap"]].copy()

    def move(self, data: Any, targets: dict[int, Sequence[float]], steps: int, step: Callable[[], bool]) -> None:
        """Move hands in straight lines to their targets over `steps` control steps."""
        start = {k: self.position(data, k) for k in targets}
        n = max(1, int(steps))
        for k in range(1, n + 1):
            for i, xyz in targets.items():
                data.mocap_pos[self.hands[i]["mocap"]] = start[i] + (np.asarray(xyz, float) - start[i]) * k / n
            ok = step()
            self.steps += 1
            if ok and self.success_at is None:
                self.success_at = self.steps


class SimView:
    """`sim` in success(sim) and targets(sim): a read-only view of the simulation (world frame, metres, radians)."""

    def __init__(self, world: SceneWorld, data: Any, table_top: float) -> None:
        self._w, self._d, self._top = world, data, float(table_top)

    @property
    def time(self) -> float:
        return float(self._d.time)

    @property
    def table_top(self) -> float:
        return self._top

    def pos(self, body: str) -> np.ndarray:
        """Position (3,) of a body's frame."""
        return self._d.xpos[self._w.body_id(body)].copy()

    def quat(self, body: str) -> np.ndarray:
        """Orientation (w, x, y, z) of a body."""
        return self._d.xquat[self._w.body_id(body)].copy()

    def xmat(self, body: str) -> np.ndarray:
        """Rotation matrix (3, 3) of a body (columns: its x, y, z axes in the world)."""
        return self._d.xmat[self._w.body_id(body)].reshape(3, 3).copy()

    def speed(self, body: str) -> float:
        """Linear speed (m/s) of a body."""
        import mujoco

        v = np.zeros(6)
        mujoco.mj_objectVelocity(self._w.model, self._d, mujoco.mjtObj.mjOBJ_BODY, self._w.body_id(body), v, 0)
        return float(np.linalg.norm(v[3:]))

    def joint(self, name: str) -> float:
        """Value of a hinge (radians) or slide (metres) joint."""
        m = self._w.model
        try:
            j = m.joint(name).id
        except KeyError:
            raise KeyError(f"no joint {name!r}; scene joints: {sorted(self._w.joints)}") from None
        return float(self._d.qpos[m.jnt_qposadr[j]])

    def joint_range(self, name: str) -> tuple[float, float]:
        r = self._w.model.joint(name).range
        return float(r[0]), float(r[1])

    def site(self, name: str) -> np.ndarray:
        return self._d.site_xpos[self._w.model.site(name).id].copy()

    def geom(self, name: str) -> np.ndarray:
        return self._d.geom_xpos[self._w.model.geom(name).id].copy()

    def nodes(self, prefix: str) -> np.ndarray:
        """Positions (N, 3) of a cable's bodies in chain order: a composite cable `prefix` (bodies prefix+"B_first",
        prefix+"B_1", ..., prefix+"B_last"), else every scene body whose name starts with `prefix`."""
        ids = self._w.chain(prefix)
        if not ids:
            raise KeyError(f"no cable or bodies with prefix {prefix!r}; cables: {sorted(self._w.cables)}")
        return self._d.xpos[ids].copy()

    def flex(self, name: str) -> np.ndarray:
        """Vertex positions (N, 3) of a flex (cloth, flex cable, soft body)."""
        try:
            f = self._w.flexes[name]["id"]
        except KeyError:
            raise KeyError(f"no flex {name!r}; flexes: {sorted(self._w.flexes)}") from None
        m = self._w.model
        a, n = int(m.flex_vertadr[f]), int(m.flex_vertnum[f])
        return self._d.flexvert_xpos[a:a + n].copy()

    def bounds(self, body: str) -> tuple[np.ndarray, np.ndarray]:
        """World bounding box (min, max) of a body's geoms and its children's."""
        return self._w.body_bounds(self._d, self._w.body_id(body))

    def bodies(self, prefix: str = "") -> list[str]:
        """Names of the scene's bodies (the program's, not the robot's) starting with `prefix`."""
        m = self._w.model
        return sorted(m.body(b).name for b in self._w.scene_ids if m.body(b).name.startswith(prefix))

    def touching(self, a: str, b: str) -> bool:
        """Do two things touch? Each is a body name (its children included), a cable prefix, a flex name, "table",
        "robot" or "hand" (the fingers of any arm)."""
        return self._w.touching(self._d, a, b)

    def held(self, thing: str) -> bool:
        """Is a body, cable or flex touched by the robot's fingers?"""
        return self._w.touching(self._d, thing, "hand")


class SimEdit(SimView):
    """`sim` in randomize(sim, rng) and demo(sim): the read-only view plus edits and, in demo, invisible hands."""

    def __init__(self, world: SceneWorld, data: Any, table_top: float, forward: Callable[[], None],
                 hands: DemoHands | None = None, step: Callable[[], bool] | None = None, control_hz: float = 20.0) -> None:
        super().__init__(world, data, table_top)
        self._forward, self._hands, self._step, self._hz = forward, hands, step, control_hz

    def set_body(self, body: str, pos: Sequence[float] | None = None, quat: Sequence[float] | None = None) -> None:
        """Move a body with a free joint (or a mocap body) to `pos` / `quat`, at rest."""
        self._w.set_body(self._d, self._w.body_id(body), pos, quat)
        self._forward()

    def set_joint(self, name: str, value: float) -> None:
        """Set a hinge / slide joint's value, at rest."""
        self._w.set_joint(self._d, name, value)
        self._forward()

    def _demo(self) -> DemoHands:
        if self._hands is None or self._step is None:
            raise SceneProgramError("invisible hands exist only in demo(sim), run by a checker")
        return self._hands

    def grab(self, body: str, hand: int = 0, point: Sequence[float] | None = None) -> None:
        """Demo: attach invisible hand `hand` to `body` at world `point` (default: the body's frame origin)."""
        self._demo().grab(self._d, int(hand), self._w.body_id(body), point)

    def move(self, hand: int, xyz: Sequence[float], seconds: float = 1.0) -> None:
        """Demo: move a hand in a straight line to `xyz` over `seconds`, simulating (the robot holds still)."""
        self.move_together({int(hand): xyz}, seconds)

    def move_together(self, targets: dict[int, Sequence[float]], seconds: float = 1.0) -> None:
        """Demo: move several hands at once ({hand: xyz})."""
        hands = self._demo()
        unknown = [k for k in targets if int(k) not in hands.hands]
        if unknown:
            raise SceneProgramError(f"demo: hands {unknown} do not exist")
        hands.move(self._d, {int(k): v for k, v in targets.items()}, int(round(float(seconds) * self._hz)), self._step)

    def release(self, hand: int = 0) -> None:
        self._demo().release(self._d, int(hand))

    def wait(self, seconds: float = 0.5) -> None:
        """Demo: simulate `seconds` with the hands where they are."""
        self.move_together({}, seconds)

    def hand(self, hand: int = 0) -> np.ndarray:
        return self._demo().position(self._d, int(hand))


__all__ = ["DemoHands", "PENETRATION", "SceneWorld", "SimEdit", "SimView"]
