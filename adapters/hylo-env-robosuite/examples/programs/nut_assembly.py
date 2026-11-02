"""robosuite NutAssembly (single nut): "fit the square nut onto the square peg" (or the round ones). Perception only.

The nut is a ring with a flat handle sticking out of one side (the segmentation labels the handle separately). The
program grasps the handle from above, closing across it, lifts the nut, then lowers it so the ring's centre comes down
along the peg's axis: the approach starts above the peg's top and ends with the ring's underside just above the
table, where the hand lets go."""
import math

import numpy as np

ARM = "arm"


def points_of(robot, view, table, label, min_points=10):
    dets = [d for d in robot.objects(view, robot.find(view, label), table) if len(d.points) >= min_points]
    return max(dets, key=lambda d: len(d.points)).points if dets else None


def topdown(closing_yaw, pos):
    y = np.array([math.cos(closing_yaw), math.sin(closing_yaw), 0.0])
    x = np.array([0.0, 0.0, -1.0])
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, np.cross(x, y), pos
    return T


def move(robot, dz, table, hand=None):
    t = robot.status()["tcp"][ARM].copy()
    t[2, 3] += dz
    p = robot.plan(robot.Goal("lift", ARM, tcp=t, hand=hand), table)
    return robot.run(p) if p is not None else None


def solve(robot):
    instruction = robot.status()["instruction"].lower()
    kind = "square" if "square" in instruction else "round"
    nut, handle, peg = f"{kind} nut", f"{kind} nut handle", f"{kind} peg"
    view = robot.look()
    table = robot.table(view)
    peg_pts = points_of(robot, view, table, peg)
    if peg_pts is None:
        print("cannot see the", peg)
        return
    peg_xy = robot.box(peg_pts, table).center[:2]
    peg_top = float(np.percentile(peg_pts[:, 2], 99))
    print(peg, "at", np.round(peg_xy, 3), "top", round(peg_top, 3))
    held = None
    for attempt in range(2):
        view = robot.look()
        ring = points_of(robot, view, table, nut)
        grip = points_of(robot, view, table, handle, 5)
        if ring is None or grip is None:
            print("cannot see the", nut if ring is None else handle)
            return
        ring_c = robot.box(ring, table).center
        grip_c = grip.mean(axis=0)
        u = grip_c[:2] - ring_c[:2]
        u /= np.linalg.norm(u)                      # from the ring's centre out along the handle
        top = float(np.percentile(grip[:, 2], 95))
        pos = np.array([grip_c[0], grip_c[1], top - 0.008])
        across = math.atan2(u[1], u[0]) + math.pi / 2
        goals = [robot.grasp_at(topdown(across + k * math.pi, pos)) for k in (0, 1)]
        for k, g in enumerate(goals):
            g.rank = k
        plan = robot.plan(goals, table, view.points)
        if plan is None:
            print("no grasp plan:", robot.why())
            return
        out = robot.run(plan)
        if out.get("ended"):
            return
        ok = robot.holding()
        move(robot, max(0.12, peg_top - table.height + 0.06), table)
        if ok and robot.holding():
            held = (plan.goal.tcp, ring, ring_c)
            break
        print("grasp attempt", attempt, "missed")
        move(robot, 0.0, table, hand=(1.0,))
    if held is None:
        return
    grasp_tcp, ring, ring_c = held
    # the ring's frame: at its centre, axes of the world (the nut lies flat); its points in that frame
    ring_frame = np.eye(4)
    ring_frame[:3, 3] = ring_c
    obj_in_tcp = np.linalg.inv(grasp_tcp) @ ring_frame
    ring_local = ring - ring_c
    goal = robot.place_at(obj_in_tcp, ring_local, peg_xy, table=table, hover=0.012, support=table.height)
    goal.pre = goal.tcp.copy()
    goal.pre[2, 3] = goal.tcp[2, 3] + (peg_top - table.height) + 0.04       # ring clear above the peg's top
    goal.attached = None                         # the ring goes around the peg on purpose
    plan = robot.plan(goal, table, view.points)
    if plan is None:
        print("no place plan:", robot.why())
        return
    out = robot.run(plan)
    if not out.get("ended"):
        move(robot, 0.10, table)
