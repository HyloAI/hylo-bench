"""robosuite Door: "open the door". Perception only.

The door is latched: its lever handle has to be turned down before the door swings on its hinge. The program finds
the door's plane (the panel's points) and which way it faces the robot, grasps the lever near its free end from the
robot's side with the fingers closing vertically, turns the lever down about its pivot (the end fixed to the panel)
in small arcs, then pulls the handle round the hinge (the panel's edge furthest from the handle) in small arcs."""
import math

import numpy as np

ARM = "arm"


def points_of(robot, view, table, label, min_points=10):
    dets = [d for d in robot.objects(view, robot.find(view, label), table) if len(d.points) >= min_points]
    return np.concatenate([d.points for d in dets]) if dets else None


def rot(axis, angle):
    """Rotation matrix about a unit axis (Rodrigues)."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def about(T, axis, point, angle):
    """Pose T turned by `angle` about the line through `point` along `axis`."""
    R = rot(axis, angle)
    out = np.eye(4)
    out[:3, :3] = R @ T[:3, :3]
    out[:3, 3] = R @ (T[:3, 3] - point) + point
    return out


def arc(robot, axis, point, total, steps, table):
    """Turn the TCP about a line in `steps` straight pieces; stops early if a piece fails."""
    for k in range(steps):
        t = about(robot.status()["tcp"][ARM], axis, point, total / steps)
        p = robot.plan(robot.Goal("push", ARM, tcp=t, hand=(0.0,)), table)
        if p is None:
            print("arc piece", k, "failed:", robot.why())
            return False
        out = robot.run(p)
        if out.get("ended"):
            return True
    return True


def solve(robot):
    view = robot.look()
    table = robot.table(view)
    panel = points_of(robot, view, table, "door")
    handle = points_of(robot, view, table, "door handle", 5)
    if panel is None or handle is None:
        print("cannot see the", "door" if panel is None else "door handle")
        return
    c = panel.mean(axis=0)
    vt = np.linalg.svd(panel - c, full_matrices=False)[2]
    n = vt[2] * np.array([1.0, 1.0, 0.0])
    n /= np.linalg.norm(n)
    if n @ np.array([-c[0], -c[1], 0.0]) < 0:
        n = -n                                      # n: the panel's normal, facing the robot (base at the origin)
    along = np.cross(np.array([0.0, 0.0, 1.0]), n)  # horizontal, in the panel's plane
    # the lever: its pivot sits against the panel, its free end furthest from it along the panel
    off = (handle - c) @ n
    pivot_pts = handle[off < np.percentile(off, 40)]
    pivot = pivot_pts.mean(axis=0)
    lever = handle[off >= np.percentile(off, 60)]
    s = (lever - pivot) @ along
    free_side = 1.0 if np.median(s) > 0 else -1.0
    tip = lever[np.argsort(free_side * s)[-max(3, len(lever) // 5):]].mean(axis=0)
    grip = pivot + 0.6 * (tip - pivot)
    print("door normal", np.round(n, 2), "pivot", np.round(pivot, 3), "grip", np.round(grip, 3))
    # grasp: approach against the panel (TCP +x = -n), fingers closing vertically
    T = np.eye(4)
    x, y = -n, np.array([0.0, 0.0, 1.0])
    T[:3, 0], T[:3, 1], T[:3, 2] = x, y, np.cross(x, y)
    T[:3, 3] = grip
    goals = []
    for k, dy in enumerate((0.0, 0.01, -0.01)):
        g = robot.grasp_at(T.copy())
        g.tcp[:3, 3] += along * free_side * dy
        g.pre = g.tcp.copy()
        g.pre[:3, 3] += n * 0.10
        g.rank = k
        goals.append(g)
    plan = robot.plan(goals, table, view.points)
    if plan is None:
        print("no grasp plan:", robot.why())
        return
    out = robot.run(plan)
    if out.get("ended"):
        return
    print("holding the lever:", robot.holding())
    # turn the lever down about the axis through its pivot along n (sign: the free end goes down)
    down = -1.0 if (np.cross(n, (tip - pivot))[2]) > 0 else 1.0
    arc(robot, n, pivot, down * 1.0, 5, table)
    # pull round the hinge: the vertical line at the panel's edge away from the handle
    s_panel = (panel - c) @ along
    hinge = c + along * (np.min(s_panel) if free_side > 0 else np.max(s_panel))
    h = robot.status()["tcp"][ARM][:3, 3]
    swing = 1.0 if np.cross(np.array([0.0, 0.0, 1.0]), h - hinge) @ n > 0 else -1.0     # the handle moves towards +n
    arc(robot, np.array([0.0, 0.0, 1.0]), hinge, swing * 0.6, 6, table)
    p = robot.plan(robot.Goal("retreat", ARM, tcp=robot.status()["tcp"][ARM], hand=(1.0,)), table)
    if p is not None:
        robot.run(p)
