"""People in a stage world (RFC-0008 part 6): kinematic walkers that follow routes, plan around obstacles and stop
for what is in front of them.

Each `human` entity is a mocap body (built by compose.py from capsules). Every step:

```text
  route      the next waypoint of its route (looping or not); at a waypoint it stands for `dwell` seconds
  path       a path to that waypoint on a planar grid (A*, 8 neighbours, no corner cutting, then shortened by line of
             sight) around the floor footprints of every other body, inflated by the person's radius; planned when a
             waypoint is chosen and again after being blocked for half a second
  sensing    three rays at chest height (straight ahead and 25 degrees to each side, 0.7 m): a hit on anything but the
             floor makes the person stop (a robot, a box, another person): people never walk through things
  motion     walking speed along the path, turning at most 180 degrees per second, a small vertical bob
```

Commands: `goto(id, [x, y])` (a one-waypoint route), `stop(id)`, `resume(id)`. `state()` reports each person's
position, heading and whether they are walking, waiting or blocked.
"""

from __future__ import annotations

import heapq
import math
from typing import Any

import mujoco
import numpy as np

RADIUS = 0.3


class Person:
    def __init__(self, eid: str, body: int, mocap: int, spec: Any, start: np.ndarray, yaw: float) -> None:
        self.id, self.body, self.mocap = eid, body, mocap
        self.route = [np.asarray(p, float) for p in spec.route] or [start[:2].copy()]
        self.loop, self.speed, self.dwell, self.height = spec.loop, spec.speed, spec.dwell, spec.height
        self.pos, self.z0, self.yaw = start[:2].copy(), float(start[2]), yaw
        self.k = 0                                      # index of the waypoint walked to
        self.path: list[np.ndarray] = []
        self.wait = 0.0
        self.blocked = 0.0
        self.stopped = False
        self.done = False
        self.walked = 0.0
        self.ignore_people = 0.0                         # seconds left of walking past people (after yielding)


class Crowd:
    def __init__(self, world: Any, ids: list[str]) -> None:
        self.w = world
        m = world.model
        self.ids = list(ids)
        self.people: dict[str, Person] = {}
        for e in world.stage.entities:
            if e.id in self.ids:
                b = m.body(e.id).id
                start = np.asarray(e.pose.pos, float)
                q = e.pose.quat
                yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
                self.people[e.id] = Person(e.id, b, int(m.body_mocapid[b]), e.human, start, yaw)
        self._own = {pid: self._geoms_of(p.body) for pid, p in self.people.items()}
        self._planned = False                           # the first plan waits for the first step (poses known)

    def _geoms_of(self, body: int) -> set[int]:
        return {g for g in range(self.w.model.ngeom) if int(self.w.model.geom_bodyid[g]) == body}

    # ------------------------------------------------------------------ planning
    def _obstacles(self, me: Person) -> list[tuple[float, float, float, float]]:
        """Floor footprints (xmin, ymin, xmax, ymax) of every geom below head height except the floor and me: the
        geom's own bounding box, rotated into the world and projected onto the floor."""
        m, d = self.w.model, self.w.data
        out = []
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
        for g in range(m.ngeom):
            if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE) or g in self._own[me.id]:
                continue
            c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
            R = d.geom_xmat[g].reshape(3, 3)
            pts = d.geom_xpos[g] + (c + corners * h) @ R.T
            if pts[:, 2].min() > me.height:             # above the head
                continue
            lo, hi = pts.min(0), pts.max(0)
            out.append((lo[0], lo[1], hi[0], hi[1]))
        return out

    def _plan(self, p: Person) -> None:
        goal = p.route[p.k]
        obs = self._obstacles(p)
        pts = np.array([p.pos, goal])
        lo = pts.min(0) - 1.5
        hi = pts.max(0) + 1.5
        cell = max(0.1, float(np.sqrt(np.prod(hi - lo) / 4096.0)))
        nx, ny = int(math.ceil((hi[0] - lo[0]) / cell)), int(math.ceil((hi[1] - lo[1]) / cell))
        free = np.ones((nx, ny), bool)
        xs = lo[0] + (np.arange(nx) + 0.5) * cell
        ys = lo[1] + (np.arange(ny) + 0.5) * cell
        for x0, y0, x1, y1 in obs:
            ix = (xs >= x0 - RADIUS) & (xs <= x1 + RADIUS)
            iy = (ys >= y0 - RADIUS) & (ys <= y1 + RADIUS)
            free[np.ix_(ix, iy)] = False

        def idx(q: np.ndarray) -> tuple[int, int]:
            return (int(np.clip((q[0] - lo[0]) // cell, 0, nx - 1)), int(np.clip((q[1] - lo[1]) // cell, 0, ny - 1)))

        s, t = idx(p.pos), idx(goal)
        free[s] = free[t] = True
        path = astar(free, s, t)
        if path is None:                                 # no way around now: keep the old path, wait, plan again
            p.path = p.path or [goal]
            return
        world_pts = [np.array([xs[i], ys[j]]) for i, j in path[1:-1]] + [goal]
        p.path = shorten([p.pos, *world_pts], free, lo, cell)[1:]

    # ------------------------------------------------------------------ stepping
    def _ahead(self, p: Person) -> str | None:
        """What is in front (within 0.6 m, straight ahead or 12 degrees to a side, at knee and chest height):
        "person", "thing" or None."""
        m, d = self.w.model, self.w.data
        geomid = np.zeros(1, np.int32)
        people = {q.body for q in self.people.values()}
        hit = None
        for h in (0.3, 0.55 * p.height):
            origin = np.array([p.pos[0], p.pos[1], p.z0 + h])
            for a in (0.0, 0.2, -0.2):
                v = np.array([math.cos(p.yaw + a), math.sin(p.yaw + a), 0.0])
                dist = mujoco.mj_ray(m, d, origin, v, None, 1, p.body, geomid)
                if 0 <= dist < 0.6:
                    if int(m.geom_bodyid[geomid[0]]) in people:
                        hit = hit or "person"
                    else:
                        return "thing"
        return hit

    def update(self, dt: float) -> None:
        d = self.w.data
        if not self._planned:
            for p in self.people.values():
                self._plan(p)
            self._planned = True
        for p in self.people.values():
            moving = False
            if p.stopped or p.done:
                pass
            elif p.wait > 0:
                p.wait -= dt
            else:
                target = p.path[0] if p.path else p.route[p.k]
                delta = target - p.pos
                dist = float(np.linalg.norm(delta))
                want = math.atan2(delta[1], delta[0]) if dist > 1e-6 else p.yaw
                turn = (want - p.yaw + math.pi) % (2 * math.pi) - math.pi
                p.yaw += float(np.clip(turn, -math.pi * dt, math.pi * dt))       # turn toward the path, even when
                p.yaw = (p.yaw + math.pi) % (2 * math.pi) - math.pi              # blocked: then look again
                ahead = self._ahead(p)
                if ahead == "thing" or (ahead == "person" and p.ignore_people <= 0):
                    p.blocked += dt
                    if ahead == "thing" and p.blocked > 0.5:      # something stands in the way: plan around it
                        self._plan(p)
                        p.blocked = 0.0
                    elif ahead == "person" and p.blocked > 0.8:   # yielded to a person: plan around, walk on
                        self._plan(p)
                        p.blocked, p.ignore_people = 0.0, 1.2
                else:
                    p.blocked = 0.0
                    p.ignore_people = max(0.0, p.ignore_people - dt)
                    if abs(turn) < math.radians(60):
                        step = min(dist, p.speed * dt)
                        p.pos = p.pos + delta / max(dist, 1e-9) * step
                        p.walked += step
                        moving = True
                    if dist < 0.05:
                        if p.path:
                            p.path.pop(0)
                        if not p.path:
                            self._arrive(p)
            bob = 0.02 * abs(math.sin(p.walked * math.pi / 0.7)) if moving else 0.0
            d.mocap_pos[p.mocap] = [p.pos[0], p.pos[1], p.z0 + bob]
            d.mocap_quat[p.mocap] = [math.cos(p.yaw / 2), 0, 0, math.sin(p.yaw / 2)]

    def _arrive(self, p: Person) -> None:
        p.wait = p.dwell
        if p.k + 1 < len(p.route):
            p.k += 1
        elif p.loop and len(p.route) > 1:
            p.k = 0
        else:
            p.done = True
            return
        self._plan(p)

    # ------------------------------------------------------------------ commands and state
    def goto(self, pid: str, xy: Any) -> None:
        p = self.people[pid]
        p.route, p.k, p.loop, p.done, p.stopped = [np.asarray(xy, float)], 0, False, False, False
        self._plan(p)

    def stop(self, pid: str) -> None:
        self.people[pid].stopped = True

    def resume(self, pid: str) -> None:
        self.people[pid].stopped = False

    def position(self, pid: str) -> np.ndarray:
        p = self.people[pid]
        return np.array([p.pos[0], p.pos[1], p.z0 + 0.5 * p.height])

    def state(self) -> dict[str, Any]:
        out = {}
        for pid, p in self.people.items():
            status = ("stopped" if p.stopped else "arrived" if p.done else "waiting" if p.wait > 0
                      else "blocked" if p.blocked > 0 else "walking")
            out[pid] = {"position": np.round(p.pos, 3).tolist(), "yaw_deg": round(math.degrees(p.yaw), 1),
                        "status": status, "waypoint": p.k, "walked_m": round(p.walked, 2)}
        return out


def astar(free: np.ndarray, s: tuple[int, int], t: tuple[int, int]) -> list[tuple[int, int]] | None:
    nx, ny = free.shape
    openq = [(0.0, s)]
    g = {s: 0.0}
    came: dict[tuple[int, int], tuple[int, int]] = {}
    steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414),
             (-1, -1, 1.414)]
    while openq:
        _, cur = heapq.heappop(openq)
        if cur == t:
            out = [cur]
            while cur in came:
                cur = came[cur]
                out.append(cur)
            return out[::-1]
        for dx, dy, c in steps:
            n = (cur[0] + dx, cur[1] + dy)
            if not (0 <= n[0] < nx and 0 <= n[1] < ny) or not free[n]:
                continue
            if dx and dy and not (free[cur[0] + dx, cur[1]] and free[cur[0], cur[1] + dy]):
                continue                                 # no corner cutting
            ng = g[cur] + c
            if ng < g.get(n, math.inf):
                g[n], came[n] = ng, cur
                heapq.heappush(openq, (ng + math.hypot(t[0] - n[0], t[1] - n[1]), n))
    return None


def shorten(pts: list[np.ndarray], free: np.ndarray, lo: np.ndarray, cell: float) -> list[np.ndarray]:
    """Drop waypoints while the straight segment between the kept ones crosses only free cells."""
    def clear(a: np.ndarray, b: np.ndarray) -> bool:
        n = max(2, int(np.linalg.norm(b - a) / (cell / 2)))
        for u in np.linspace(0, 1, n):
            q = a + u * (b - a)
            i, j = int((q[0] - lo[0]) // cell), int((q[1] - lo[1]) // cell)
            if not (0 <= i < free.shape[0] and 0 <= j < free.shape[1]) or not free[i, j]:
                return False
        return True

    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not clear(pts[i], pts[j]):
            j -= 1
        out.append(pts[j])
        i = j
    return out


__all__ = ["Crowd", "astar", "shorten"]
