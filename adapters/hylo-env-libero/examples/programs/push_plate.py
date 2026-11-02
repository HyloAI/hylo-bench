"""LIBERO: "push the plate to the front of the stove": the front is the stove's side facing the agentview camera;
the plate is pushed by its rim with the closed fingers pointing down, from the side opposite the goal, along the
line to a spot just in front of the stove, then corrected once from a new look (perception only: cameras, depth,
the environment's segmentation)."""
import math

import numpy as np

ARM = "arm"


def instances(robot, view, table, label):
    """{instance id: world points} for every instance of a label (all cameras, all fragments)."""
    masks = robot.find(view, label)
    out = {}
    for inst in sorted({m.instance for m in masks}):
        dets = robot.objects(view, [m for m in masks if m.instance == inst], table)
        pts = [d.points for d in dets if len(d.points) >= 10]
        if pts:
            out[inst] = np.concatenate(pts)
    return out


def tcp_pose(x_axis, y_axis, origin):
    x = np.asarray(x_axis, float)
    x /= np.linalg.norm(x)
    y = np.asarray(y_axis, float)
    y = y - (y @ x) * x
    y /= np.linalg.norm(y)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, np.cross(x, y), origin
    return T


def round_centre(pts, fallback):
    """Centre of a round object (a plate) from its raised rim: a circle fitted to the highest ring of points, which
    stays right when the camera sees only part of it; `fallback` when the fit is implausible."""
    top = float(np.percentile(pts[:, 2], 98))
    rim = pts[pts[:, 2] > top - 0.005]
    if len(rim) < 20:
        return fallback
    x, y = rim[:, 0], rim[:, 1]
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    sol = np.linalg.lstsq(A, x * x + y * y, rcond=None)[0]
    c = np.array([sol[0] / 2, sol[1] / 2])
    r = math.sqrt(max(sol[2] + c @ c, 0.0))
    if not 0.03 < r < 0.15 or np.linalg.norm(c - fallback) > 0.08:
        return fallback
    return c


def camera_axes(view):
    """Horizontal unit vectors: image-right and towards the agentview camera (what 'right' and 'front' mean)."""
    T = np.asarray(view.cams["agentview"].T_world_cam)
    right = np.array([T[0, 0], T[1, 0], 0.0])
    toward = -np.array([T[0, 2], T[1, 2], 0.0])
    return right / np.linalg.norm(right), toward / np.linalg.norm(toward)


def plate_and_goal(robot, view, table):
    plates, stoves = instances(robot, view, table, "plate"), instances(robot, view, table, "stove")
    if not plates or not stoves:
        return None, None, None
    pts = max(plates.values(), key=len)
    pbox = robot.box(pts, table)
    centre = round_centre(pts, pbox.center[:2])
    radius = 0.5 * float(max(pbox.size[0], pbox.size[1]))
    sbox = robot.box(max(stoves.values(), key=len), table)
    right, toward = camera_axes(view)
    c, s = math.cos(sbox.yaw), math.sin(sbox.yaw)
    half = 0.5 * (abs(sbox.size[0] * (c * toward[0] + s * toward[1])) + abs(sbox.size[1] * (-s * toward[0] + c * toward[1])))
    goal = sbox.center[:2] + toward[:2] * (half + radius + 0.035)
    return centre, radius, goal


def push(robot, table, centre, radius, goal):
    """Closed fingertips down against the rim on the side away from the goal (or up to 50 degrees round from it),
    then a straight push that would bring the centre to the goal. False: no reachable contact."""
    d = goal - centre
    dist = float(np.linalg.norm(d))
    u = d / dist
    goals = []
    for k, rot in enumerate((0.0, 0.45, -0.45, 0.9, -0.9)):
        c, s_ = math.cos(rot), math.sin(rot)
        w = np.array([c * u[0] - s_ * u[1], s_ * u[0] + c * u[1]])          # push direction through the contact
        p0 = np.array([*(centre - w * (radius + 0.02)), table.height + 0.018])
        for closing in (np.array([-w[1], w[0], 0.0]), np.array([w[0], w[1], 0.0])):
            start = tcp_pose((0.0, 0.0, -1.0), closing, p0)
            pre = start.copy()
            pre[2, 3] += 0.04
            g = robot.Goal("reach", ARM, tcp=start, pre=pre, hand_before=(0.0,), rank=len(goals))
            goals.append((g, w))
    view = robot.look()
    p = robot.plan([gw[0] for gw in goals], table, view.points)
    if p is None:
        print("no reachable contact on the rim:", robot.why()[:200])
        return False
    w = next(gw[1] for gw in goals if gw[0] is p.goal)
    if robot.run(p).get("ended"):
        return True
    t = robot.status()["tcp"][ARM].copy()
    t[:2, 3] += w * (d @ w)                                # along the contact's direction, the goal's share
    p = robot.plan(robot.Goal("push", ARM, tcp=t), table)
    if p is None:
        print("no push plan:", robot.why())
        return True
    out = robot.run(p)
    if not out.get("ended"):
        t = robot.status()["tcp"][ARM].copy()
        t[2, 3] += 0.10
        p = robot.plan(robot.Goal("retreat", ARM, tcp=t), table)
        if p is not None:
            robot.run(p)
    return True


def carry(robot, table, centre, radius, goal):
    """Fallback: lift the plate by its rim (one finger inside, one outside) and set it down at the goal."""
    view = robot.look()
    found = instances(robot, view, table, "plate")
    pts = max(found.values(), key=len)
    box = robot.box(pts, table)
    top = float(np.percentile(pts[:, 2], 98))
    toward_base = math.atan2(-centre[1], -centre[0])
    goals = []
    for k in range(8):
        a = toward_base + (k // 2 + 1 if k % 2 else -(k // 2)) * math.pi / 4
        radial = np.array([math.cos(a), math.sin(a), 0.0])
        p0 = np.array([centre[0], centre[1], max(top - 0.006, table.height + 0.017)]) + radial * (radius - 0.006)
        g = robot.grasp_at(ARM, tcp_pose((0.0, 0.0, -1.0), radial, p0))
        g.pre = g.tcp.copy()
        g.pre[2, 3] += 0.06
        g.rank = k
        goals.append(g)
    p = robot.plan(goals, table, view.points)
    if p is None:
        print("no rim grasp:", robot.why()[:200])
        return
    if robot.run(p).get("ended"):
        return
    t = robot.status()["tcp"][ARM].copy()
    t[2, 3] += 0.06
    lp = robot.plan(robot.Goal("lift", ARM, tcp=t), table)
    if lp is not None:
        robot.run(lp)
    obj = box.pose.copy()
    obj[:2, 3] = centre
    goal_pose = robot.place_at(ARM, np.linalg.inv(p.goal.tcp) @ obj, box, goal, table=table, hover=0.005)
    pp = robot.plan(goal_pose, table, view.points)
    if pp is None:
        print("no place plan:", robot.why())
        return
    robot.run(pp)


def solve(robot):
    view = robot.look()
    table = robot.table(view)
    for attempt in range(3):
        view = robot.look()
        centre, radius, goal = plate_and_goal(robot, view, table)
        if centre is None:
            print("cannot see the plate or the stove")
            return
        d = goal - centre
        dist = float(np.linalg.norm(d))
        print("plate", np.round(centre, 3), "radius", round(radius, 3), "goal", np.round(goal, 3), "distance", round(dist, 3))
        if dist < 0.025:
            return
        if not push(robot, table, centre, radius, goal):
            print("pushing is blocked; carrying the plate by its rim instead")
            carry(robot, table, centre, radius, goal)
            return
