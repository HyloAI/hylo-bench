"""RoboCasa TurnOnStove: "turn on the rear left burner of the stove". Perception only.

The segmentation names each knob of the stove by the burner it works ("rear left knob", ...). A knob sits either on
the stove's front (its axis horizontal, towards the robot) or on its top (its axis vertical): the program decides by
whether the knob stands above the stove's top surface around it. It grasps the knob along its axis, fingers closing
across it, and twists it about the axis by about 1.2 rad (a burner counts as on once its knob has turned past
0.35 rad either way); if the twist cannot be planned it tries the other way."""
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


def rot(axis, angle):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def along(axis, pos, closing):
    x = -np.asarray(axis, float)
    y = np.asarray(closing, float) - (np.asarray(closing, float) @ x) * x
    y /= np.linalg.norm(y)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, np.cross(x, y), pos
    return T


def below(table, points):
    """None when the target is under the counter top (the planner would keep the hand above it), else the table."""
    return None if float(np.percentile(points[:, 2], 90)) < table.height + 0.02 else table


def solve(robot):
    instruction = robot.status()["instruction"].lower()
    view = robot.look()
    table = robot.table(view)
    knobs = sorted({v for v in view.segments.values() if v.endswith(" knob")}, key=len, reverse=True)
    which = next((k for k in knobs if k[:-len(" knob")] in instruction), None)
    if which is None:
        print("no knob for", repr(instruction), "among", knobs)
        return
    knob = points_of(robot, view, table, which)
    stove = points_of(robot, view, table, "stove")
    if knob is None:
        print("cannot see the", which)
        return
    k = knob.mean(axis=0)
    table = below(table, knob)
    top_mounted = False
    if stove is not None:
        near = stove[np.linalg.norm(stove[:, :2] - k[:2], axis=1) < 0.08]
        top_mounted = len(near) > 10 and k[2] > float(np.percentile(near[:, 2], 80)) - 0.005
    toward = np.array([-k[0], -k[1], 0.0])
    toward /= np.linalg.norm(toward)
    axis = np.array([0.0, 0.0, 1.0]) if top_mounted else toward
    tip = float(np.percentile((knob - k) @ axis, 95))
    grip = k + axis * (tip - 0.012)
    print(which, "at", np.round(k, 3), "on the", "top" if top_mounted else "front")
    closing = np.array([0.0, 0.0, 1.0]) if not top_mounted else np.array([-toward[1], toward[0], 0.0])
    goals = []
    for r, c in enumerate((closing, -closing)):
        g = robot.grasp_at(along(axis, grip, c), hand=(0.0,))
        g.pre = g.tcp.copy()
        g.pre[:3, 3] += axis * 0.10
        g.rank = r
        goals.append(g)
    plan = robot.plan(goals, table, view.points)
    if plan is None:
        print("no grasp plan:", robot.why())
        return
    if robot.run(plan).get("ended"):
        return
    for sign in (1.0, -1.0):
        ok = True
        for piece in range(3):
            T = robot.status()["tcp"][ARM].copy()
            T[:3, :3] = rot(axis, sign * 0.4) @ T[:3, :3]
            p = robot.plan(robot.Goal("push", ARM, tcp=T, hand=(0.0,)), table)
            if p is None:
                ok = False
                break
            if robot.run(p).get("ended"):
                return
        if ok:
            break
        print("twist", sign, "failed:", robot.why())
    T = robot.status()["tcp"][ARM].copy()
    p = robot.plan(robot.Goal("retreat", ARM, tcp=T, hand=(1.0,)), table)
    if p is not None and not robot.run(p).get("ended"):
        T = robot.status()["tcp"][ARM].copy()
        T[:3, 3] += axis * 0.08
        p = robot.plan(robot.Goal("retreat", ARM, tcp=T), table)
        if p is not None:
            robot.run(p)
