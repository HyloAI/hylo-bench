"""robosuite TwoArmHandover: "pick up the hammer with the arm next to it and hand it over to the other arm".
Perception only, two arms facing each other (robot0 at the origin, robot1 across the table).

The arm nearer the hammer (robot0 in this task) grasps the handle from above near the head, lifts it and carries it
to the middle between the arms, turned so the handle's free end points at the other arm. The other arm grasps the
free end from above; the first lets go and backs off, leaving the hammer held by its handle in the air (success:
robot1 holds the handle, robot0 touches nothing, the hammer more than 10 cm above the table)."""
import math

import numpy as np

HAND_Z = 0.25           # handover height of the handle above the table


def topdown(closing_yaw, pos):
    y = np.array([math.cos(closing_yaw), math.sin(closing_yaw), 0.0])
    x = np.array([0.0, 0.0, -1.0])
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, np.cross(x, y), pos
    return T


def points_of(robot, view, table, label, min_points=8):
    dets = [d for d in robot.objects(view, robot.find(view, label), table) if len(d.points) >= min_points]
    return np.concatenate([d.points for d in dets]) if dets else None


def run_goal(robot, goal, table, scene=None):
    p = robot.plan(goal, table, scene)
    if p is None:
        print(goal.arm, goal.verb, "no plan:", robot.why())
        return None
    return robot.run(p)


def axis_of(points):
    """Unit horizontal direction of a long object's points (first principal axis, z dropped)."""
    c = points.mean(axis=0)
    vt = np.linalg.svd(points - c, full_matrices=False)[2]
    a = vt[0] * np.array([1.0, 1.0, 0.0])
    return a / np.linalg.norm(a)


def solve(robot):
    view = robot.look()
    table = robot.table(view)
    handle = points_of(robot, view, table, "hammer handle")
    head = points_of(robot, view, table, "hammer head")
    if handle is None or head is None:
        print("cannot see the hammer's", "handle" if handle is None else "head")
        return
    bases = {a.name: np.asarray(a.base_in_world)[:3, 3] for a in robot.spec.arms}
    src = min(bases, key=lambda a: np.linalg.norm(handle.mean(0)[:2] - bases[a][:2]))
    dst = next(a for a in bases if a != src)
    u = axis_of(handle)
    if (head.mean(0) - handle.mean(0))[:2] @ u[:2] < 0:
        u = -u                                          # u: from the handle towards the head
    s = (handle - handle.mean(0)) @ u
    near_head = handle[s > np.percentile(s, 55)]
    top = float(np.percentile(handle[:, 2], 95))
    grip = np.array([near_head[:, 0].mean(), near_head[:, 1].mean(), top - 0.01])
    across = math.atan2(u[1], u[0]) + math.pi / 2
    goals = [robot.grasp_at(topdown(across + k * math.pi, grip), arm=src) for k in (0, 1)]
    for k, g in enumerate(goals):
        g.rank = k
    plan = robot.plan(goals, table, view.points)
    if plan is None:
        print("no grasp plan:", robot.why())
        return
    if robot.run(plan).get("ended"):
        return
    if not robot.holding(arm=src):
        print("missed the handle")
        return
    # carry: turn about the vertical so the free end (-u) points at the other arm, centre the grip mid-way
    T = robot.status()["tcp"][src]
    to_dst = (bases[dst] - bases[src])[:2]
    to_dst /= np.linalg.norm(to_dst)
    want = math.atan2(-to_dst[1], -to_dst[0])          # u should point back at src: the free end faces dst
    turn = (want - math.atan2(u[1], u[0]) + math.pi) % (2 * math.pi) - math.pi
    c, s_ = math.cos(turn), math.sin(turn)
    Rz = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    mid = (bases[src] + bases[dst]) / 2
    goal_tcp = np.eye(4)
    goal_tcp[:3, :3] = Rz @ T[:3, :3]
    goal_tcp[:3, 3] = [mid[0] - 0.06 * to_dst[0], mid[1] - 0.06 * to_dst[1], table.height + HAND_Z]
    if run_goal(robot, robot.Goal("lift", src, tcp=goal_tcp, hand=(0.0,)), table) is None:
        return
    # the other arm takes the free end
    for attempt in range(2):
        view = robot.look()
        pts = points_of(robot, view, table, "hammer handle")
        if pts is None:
            print("cannot see the handle in the air")
            return
        tsrc = robot.status()["tcp"][src][:3, 3]
        a = axis_of(pts)
        proj = (pts - tsrc) @ a
        if proj.mean() < 0:                            # a: from the first hand out along the free end
            a, proj = -a, -proj
        far = pts[proj > 0.04] if (proj > 0.04).sum() >= 3 else pts
        pos = far.mean(axis=0)
        pos[2] = float(np.percentile(pts[:, 2], 95)) - 0.01
        yaw = math.atan2(a[1], a[0]) + math.pi / 2
        gl = [robot.grasp_at(topdown(yaw + k * math.pi, pos), arm=dst) for k in (0, 1)]
        for k, g in enumerate(gl):
            g.rank = k
        p = robot.plan(gl, table, view.points)
        if p is None:
            print("no grasp plan for the free end:", robot.why())
            return
        if robot.run(p).get("ended"):
            return
        if robot.holding(arm=dst):
            break
        print("missed the free end")
        t = robot.status()["tcp"][dst].copy()
        t[2, 3] += 0.06
        run_goal(robot, robot.Goal("retreat", dst, tcp=t, hand=(1.0,)), table)
    else:
        return
    t = robot.status()["tcp"][src]
    if run_goal(robot, robot.Goal("retreat", src, tcp=t, hand=(1.0,)), table) is None:
        return
    back = robot.status()["tcp"][src].copy()
    back[:3, 3] -= back[:3, 0] * 0.08                  # straight back along the approach axis (up)
    back[:2, 3] -= 0.06 * to_dst
    run_goal(robot, robot.Goal("retreat", src, tcp=back), table)
