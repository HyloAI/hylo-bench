"""RoboCasa CloseCabinet: "close the cabinet doors". Perception only.

The cabinet the instruction means is labelled "cabinet" and each door's handle "cabinet handle". An open door stands
out from the cabinet's front, swung about a vertical hinge at its edge. For each handle (one per door) the program
fits the door's plane from the cabinet points around the handle, takes the hinge as the door's edge furthest from the
handle, and sweeps the closed fingertips round the hinge against the door's outer face until the door lies in the
cabinet's front plane (whose normal points at the robot), plus a little."""
import math

import numpy as np

ARM = "arm"


def label_points(robot, view, label):
    """World points of a label from every camera's mask and depth (not robot.objects(): it drops points under the
    counter top, where lower cabinets are)."""
    out = []
    for m in robot.find(view, label):
        cam = view.cams[m.camera]
        if cam.depth is None or cam.K is None:
            continue
        v, u = np.nonzero(m.mask & (cam.depth > 0))
        if len(v) < 3:
            continue
        z = cam.depth[v, u]
        K, T = np.asarray(cam.K), np.asarray(cam.T_world_cam)
        p = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z, np.ones_like(z)], axis=1)
        out.append((p @ T.T)[:, :3])
    return np.concatenate(out) if out else np.zeros((0, 3))


def clusters(points, gap=0.05, min_points=4):
    """Groups of points connected by steps shorter than `gap` (on a 1 cm grid), largest first."""
    if not len(points):
        return []
    keys = np.floor(points / 0.01).astype(np.int64)
    cells = {}
    for i, k in enumerate(map(tuple, keys)):
        cells.setdefault(k, []).append(i)
    reach = int(math.ceil(gap / 0.01))
    seen, groups = set(), []
    for start in cells:
        if start in seen:
            continue
        seen.add(start)
        stack, members = [start], []
        while stack:
            c = stack.pop()
            members += cells[c]
            for dx in range(-reach, reach + 1):
                for dy in range(-reach, reach + 1):
                    for dz in range(-reach, reach + 1):
                        nb = (c[0] + dx, c[1] + dy, c[2] + dz)
                        if nb in cells and nb not in seen:
                            seen.add(nb)
                            stack.append(nb)
        if len(members) >= min_points:
            groups.append(points[members])
    return sorted(groups, key=len, reverse=True)


def rotz_about(T, point, angle):
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    out = np.eye(4)
    out[:3, :3] = R @ T[:3, :3]
    out[:3, 3] = R @ (T[:3, 3] - point) + point
    return out


def facing(x, pos):
    x = np.asarray(x, float) / np.linalg.norm(x)
    y = np.array([0.0, 0.0, 1.0])
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, np.cross(x, y), pos
    return T


def close_door(robot, table, handle, cabinet, view):
    h = handle.mean(axis=0)
    if h[2] < table.height + 0.02:
        table = None                               # a lower cabinet: the planner must not keep the hand above the top
    near = cabinet[np.linalg.norm(cabinet[:, :2] - h[:2], axis=1) < 0.35]
    near = near[np.abs(near[:, 2] - h[2]) < 0.25]
    if len(near) < 30:
        print("too few door points near the handle")
        return False
    c = near.mean(axis=0)
    vt = np.linalg.svd(near - c, full_matrices=False)[2]
    e = vt[0] * np.array([1.0, 1.0, 0.0])          # along the door, horizontal
    e /= np.linalg.norm(e)
    s = (near - c) @ e
    free = 1.0 if (h - c) @ e > 0 else -1.0
    hinge = c + e * (s.min() if free > 0 else s.max())
    radius = float(np.linalg.norm((h - hinge)[:2]))
    front = np.array([-hinge[0], -hinge[1], 0.0])
    front /= np.linalg.norm(front)                 # the cabinet front's normal, roughly towards the robot
    door = (h - hinge) * np.array([1.0, 1.0, 0.0])
    door /= np.linalg.norm(door)
    closed = np.array([-front[1], front[0], 0.0])
    if closed @ door < 0:
        closed = -closed                           # the closed door lies along the front, on the handle's side
    angle = math.atan2(np.cross(door, closed)[2], door @ closed)
    print("door: hinge", np.round(hinge, 3), "radius", round(radius, 3), "to turn", round(angle, 2), "rad")
    if abs(angle) < 0.08:
        return True
    push_side = np.cross(np.array([0.0, 0.0, 1.0]), door) * math.copysign(1.0, angle)   # the way the door moves
    reach = float(robot.hand_points((0.0,))[:, 0].max())
    contact = hinge + door * (0.8 * radius)
    contact[2] = h[2]
    start = contact - push_side * (reach + 0.04)
    g = robot.Goal("reach", ARM, tcp=facing(push_side, start), hand_before=(0.0,), hand=(0.0,))
    plan = robot.plan(g, table, view.points)
    if plan is None:
        print("cannot get behind the door:", robot.why())
        return False
    if robot.run(plan).get("ended"):
        return True
    T = robot.status()["tcp"][ARM].copy()
    T[:3, 3] += push_side * 0.035
    p = robot.plan(robot.Goal("push", ARM, tcp=T, hand=(0.0,)), table)
    if p is not None and robot.run(p).get("ended"):
        return True
    total = angle + math.copysign(0.12, angle)
    for piece in range(6):
        T = rotz_about(robot.status()["tcp"][ARM], hinge, total / 6)
        p = robot.plan(robot.Goal("push", ARM, tcp=T, hand=(0.0,)), table)
        if p is None:
            print("sweep stopped:", robot.why())
            break
        if robot.run(p).get("ended"):
            return True
    T = robot.status()["tcp"][ARM].copy()
    T[:3, 3] -= T[:3, 0] * 0.08
    p = robot.plan(robot.Goal("retreat", ARM, tcp=T), table)
    if p is not None:
        robot.run(p)
    return True


def solve(robot):
    view = robot.look()
    table = robot.table(view)
    handles = clusters(label_points(robot, view, "cabinet handle"))
    cab = label_points(robot, view, "cabinet")
    if not handles or len(cab) < 20:
        print("cannot see the cabinet", "handles" if not handles else "")
        return
    for h in sorted(handles, key=lambda pts: np.linalg.norm(pts.mean(0)[:2])):
        if robot.status().get("ended"):
            return
        close_door(robot, table, h, cab, view)
        view = robot.look()
