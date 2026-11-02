"""RoboCasa OpenDrawer / CloseDrawer: "open the left drawer", "close the right drawer". Perception only.

The segmentation names the drawer the instruction means "drawer" (other drawers are "other drawer") and its handle
"drawer handle". The drawer's front faces the robot; its normal comes from the front's points. Open: grasp the handle
from the front, fingers closing vertically across the bar, and pull straight out along the normal (the drawer must
come out almost fully). Close: close the fingers, put them against the drawer's front and push it straight in until
the front is level with the fronts beside it."""
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


def front_normal(points):
    """Horizontal normal of a roughly vertical face, pointing towards the robot (its base is at the origin)."""
    c = points.mean(axis=0)
    vt = np.linalg.svd(points - c, full_matrices=False)[2]
    n = vt[2] * np.array([1.0, 1.0, 0.0])
    if np.linalg.norm(n) < 0.3:                    # the least axis is vertical: the face seen is a top; use the view
        n = np.array([-c[0], -c[1], 0.0])
    n /= np.linalg.norm(n)
    return n if n @ np.array([-c[0], -c[1], 0.0]) > 0 else -n


def facing(n, pos, closing=(0.0, 0.0, 1.0)):
    """TCP pose approaching against the face (+x = -n), fingers closing along `closing`."""
    x = -np.asarray(n, float)
    y = np.asarray(closing, float) - (np.asarray(closing, float) @ x) * x
    y /= np.linalg.norm(y)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, np.cross(x, y), pos
    return T


def straight(robot, T, verb, table, hand=None):
    p = robot.plan(robot.Goal(verb, ARM, tcp=T, hand=hand), table)
    if p is None:
        print(verb, "failed:", robot.why())
        return None
    return robot.run(p)


def below(table, points):
    """The support plane the planner keeps the hand above, or None when the target is under the counter top (a
    drawer): then only the observed points (the counter's edge among them) are avoided."""
    return None if float(np.percentile(points[:, 2], 90)) < table.height + 0.02 else table


def open_drawer(robot, view, table, n, handle):
    table = below(table, handle)
    hc = handle.mean(axis=0)
    out_of_front = float(np.percentile((handle - hc) @ n, 90))
    grip = hc + n * (out_of_front - 0.01)
    goals = []
    for k, closing in enumerate(((0.0, 0.0, 1.0), (0.0, 0.0, -1.0))):
        g = robot.grasp_at(facing(n, grip, closing))
        g.pre = g.tcp.copy()
        g.pre[:3, 3] += n * 0.10
        g.rank = k
        goals.append(g)
    plan = robot.plan(goals, table, view.points)
    if plan is None:
        print("no grasp plan:", robot.why())
        return
    if robot.run(plan).get("ended"):
        return
    print("holding the handle:", robot.holding())
    for pull in (0.22, 0.22):
        T = robot.status()["tcp"][ARM].copy()
        T[:3, 3] += n * pull
        out = straight(robot, T, "push", table, hand=(0.0,))
        if out is None or out.get("ended"):
            return
    T = robot.status()["tcp"][ARM].copy()
    straight(robot, T, "retreat", table, hand=(1.0,))
    T = robot.status()["tcp"][ARM].copy()
    T[:3, 3] += n * 0.08
    straight(robot, T, "retreat", table)


def close_drawer(robot, view, table, n, drawer, neighbours):
    table = below(table, drawer)
    front = float(np.percentile(drawer @ n, 95))                  # how far the drawer's front sticks out
    level = float(np.percentile(neighbours @ n, 60)) if neighbours is not None else front - 0.35
    depth = max(0.05, front - level + 0.02)
    face = drawer[drawer @ n > front - 0.03]
    c = face.mean(axis=0)
    print("drawer front out by", round(front - level, 3), "m; pushing", round(depth, 3))
    reach = float(robot.hand_points((0.0,))[:, 0].max())          # fingertips past the TCP, fingers closed
    start = facing(n, c + n * (reach + 0.04), (0.0, 1.0, 0.0) if abs(n[1]) < 0.9 else (1.0, 0.0, 0.0))
    g = robot.Goal("reach", ARM, tcp=start, pre=None, hand_before=(0.0,), hand=(0.0,))
    plan = robot.plan(g, table, view.points)
    if plan is None:
        print("cannot get in front of the drawer:", robot.why())
        return
    if robot.run(plan).get("ended"):
        return
    T = robot.status()["tcp"][ARM].copy()
    T[:3, 3] -= n * (depth + 0.04)
    out = straight(robot, T, "push", table, hand=(0.0,))
    if out is not None and not out.get("ended"):
        T = robot.status()["tcp"][ARM].copy()
        T[:3, 3] += n * 0.10
        straight(robot, T, "retreat", table)


def solve(robot):
    instruction = robot.status()["instruction"].lower()
    view = robot.look()
    table = robot.table(view)
    drawer = points_of(robot, view, table, "drawer")
    handle = points_of(robot, view, table, "drawer handle", 4)
    if drawer is None:
        print("cannot see the drawer")
        return
    n = front_normal(drawer if handle is None else np.concatenate([drawer, handle]))
    if "close" in instruction:
        close_drawer(robot, view, table, n, drawer, points_of(robot, view, table, "other drawer"))
    elif handle is None:
        print("cannot see the drawer handle")
    else:
        open_drawer(robot, view, table, n, handle)
