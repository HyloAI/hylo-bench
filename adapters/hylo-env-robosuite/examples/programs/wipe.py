"""robosuite Wipe: "wipe the dirt off the table". Perception only.

The hand is a sponge (TCP at the centre of its face, +x out of it). The dirt is a trail of small spots. The program
orders the dirt's points into a path (nearest neighbour from the end nearest the robot), thins it to waypoints a few
centimetres apart, brings the sponge down onto the first one and slides it along the path pressed lightly onto the
table, the sponge kept flat (only the sponge may touch the table). Then it looks again and wipes what is left, up to
three passes."""
import numpy as np

ARM = "arm"
PRESS = 0.004          # sponge face this far below the table top (a light press)
SPACING = 0.03


def dirt_points(robot, view, table):
    dets = robot.objects(view, robot.find(view, "dirt"), table)
    pts = [d.points for d in dets if len(d.points) >= 3]
    return np.concatenate(pts) if pts else None


def path(points):
    """Waypoints through the points: greedy nearest neighbour from the point nearest the robot, every SPACING m."""
    pts = points[:, :2].copy()
    order = [int(np.argmin(np.linalg.norm(pts, axis=1)))]
    left = np.ones(len(pts), bool)
    left[order[0]] = False
    while left.any():
        d = np.linalg.norm(pts - pts[order[-1]], axis=1)
        d[~left] = np.inf
        k = int(np.argmin(d))
        order.append(k)
        left[k] = False
    out = [pts[order[0]]]
    for k in order[1:]:
        if np.linalg.norm(pts[k] - out[-1]) >= SPACING:
            out.append(pts[k])
    return out


def sponge_down(xy, z):
    T = np.eye(4)
    x, y = np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])
    T[:3, 0], T[:3, 1], T[:3, 2] = x, y, np.cross(x, y)
    T[:3, 3] = [xy[0], xy[1], z]
    return T


def slide(robot, xy, z, table):
    p = robot.plan(robot.Goal("push", ARM, tcp=sponge_down(xy, z)), table=None)
    if p is None:
        print("slide failed:", robot.why())
        return None
    return robot.run(p)


def solve(robot):
    for sweep in range(3):
        view = robot.look()
        table = robot.table(view)
        dirt = dirt_points(robot, view, table)
        if dirt is None:
            print("no dirt left")
            return
        way = path(dirt)
        z = table.height - PRESS
        print("pass", sweep, ":", len(dirt), "dirt points,", len(way), "waypoints")
        start = robot.Goal("reach", ARM, tcp=sponge_down(way[0], z + 0.004), pre=sponge_down(way[0], z + 0.08))
        plan = robot.plan(start, None, view.points)
        if plan is None:
            print("cannot reach the start:", robot.why())
            return
        out = robot.run(plan)
        if out.get("ended"):
            return
        for xy in way[1:]:
            out = slide(robot, xy, z, table)
            if out is None or out.get("ended"):
                return
        up = robot.status()["tcp"][ARM].copy()
        up[2, 3] += 0.08
        p = robot.plan(robot.Goal("lift", ARM, tcp=up), None)
        if p is not None and robot.run(p).get("ended"):
            return
