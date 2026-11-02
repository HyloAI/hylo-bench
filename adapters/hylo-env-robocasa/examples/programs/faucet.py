"""RoboCasa TurnOnSinkFaucet: "turn on the sink faucet". Perception only.

The faucet's handle ("faucet handle") is a lever mounted on or beside the spout ("faucet spout"); the water runs when
the lever is turned from its rest position. The program grasps the lever's free end from above (the end furthest
from the spout's axis), fingers closing across it, then swings it about the vertical axis through the lever's base in
small arcs, one way and, if that cannot be planned, the other; finally it lifts the end a few centimetres (levers that
tilt up instead of swinging)."""
import math

import numpy as np

ARM = "arm"


def points_of(robot, view, table, label, min_points=8):
    """World points of a label from every camera's mask and depth. Not robot.objects(): it drops points on or under
    the support plane, and drawers, lower cabinets and front knobs sit below the counter top."""
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
    pts = np.concatenate(out) if out else np.zeros((0, 3))
    return pts if len(pts) >= min_points else None


def rotz_about(T, point, angle):
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    out = np.eye(4)
    out[:3, :3] = R @ T[:3, :3]
    out[:3, 3] = R @ (T[:3, 3] - point) + point
    return out


def topdown(closing_yaw, pos):
    y = np.array([math.cos(closing_yaw), math.sin(closing_yaw), 0.0])
    x = np.array([0.0, 0.0, -1.0])
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, np.cross(x, y), pos
    return T


def solve(robot):
    view = robot.look()
    table = robot.table(view)
    lever = points_of(robot, view, table, "faucet handle")
    spout = points_of(robot, view, table, "faucet spout")
    if lever is None:
        print("cannot see the faucet handle")
        return
    base_xy = (spout.mean(axis=0) if spout is not None else lever.mean(axis=0))[:2]
    r = np.linalg.norm(lever[:, :2] - base_xy, axis=1)
    end = lever[r > np.percentile(r, 70)].mean(axis=0)
    pivot = lever[r < np.percentile(r, 30)].mean(axis=0)
    d = end[:2] - pivot[:2]
    if np.linalg.norm(d) < 0.01:
        d = end[:2] - base_xy
    yaw = math.atan2(d[1], d[0]) + math.pi / 2
    top = float(np.percentile(lever[:, 2], 95))
    grip = np.array([end[0], end[1], top - 0.01])
    print("lever end", np.round(end, 3), "pivot", np.round(pivot, 3))
    goals = [robot.grasp_at(topdown(yaw + k * math.pi, grip)) for k in (0, 1)]
    for k, g in enumerate(goals):
        g.rank = k
    plan = robot.plan(goals, table, view.points)
    if plan is None:
        print("no grasp plan:", robot.why())
        return
    if robot.run(plan).get("ended"):
        return
    for sign in (1.0, -1.0):
        moved = 0
        for piece in range(4):
            T = rotz_about(robot.status()["tcp"][ARM], pivot, sign * 0.25)
            p = robot.plan(robot.Goal("push", ARM, tcp=T, hand=(0.0,)), table)
            if p is None:
                break
            if robot.run(p).get("ended"):
                return
            moved += 1
        if moved:
            break
    T = robot.status()["tcp"][ARM].copy()
    T[2, 3] += 0.04
    p = robot.plan(robot.Goal("lift", ARM, tcp=T, hand=(0.0,)), table)
    if p is not None and robot.run(p).get("ended"):
        return
    T = robot.status()["tcp"][ARM].copy()
    p = robot.plan(robot.Goal("retreat", ARM, tcp=T, hand=(1.0,)), table)
    if p is not None:
        robot.run(p)
