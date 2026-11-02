"""robosuite Stack: "stack the red cube on top of the green cube". Perception only: grasp the first cube named, look
again, put it down centred on the second (the toolkit finds the support height from the points under the footprint),
let go and move up. Retries the grasp once from a fresh look."""
import numpy as np


def object_points(robot, view, table, label):
    dets = [d for d in robot.objects(view, robot.find(view, label), table) if len(d.points) >= 10]
    return max(dets, key=lambda d: len(d.points)).points if dets else None


def move_up(robot, dz, table, hand=None):
    t = robot.status()["tcp"]["arm"].copy()
    t[2, 3] += dz
    p = robot.plan(robot.Goal("lift", "arm", tcp=t, hand=hand), table)
    return robot.run(p) if p is not None else None


def named(instruction, labels):
    """Labels in the order the instruction mentions them."""
    text = instruction.lower()
    found = [(text.find(lab), lab) for lab in labels if lab in text]
    return [pair[1] for pair in sorted(found)]


def pick(robot, label, table):
    for attempt in range(2):
        view = robot.look()
        pts = object_points(robot, view, table, label)
        if pts is None:
            print("cannot see", label)
            return None
        box = robot.box(pts, table)
        goals = robot.grasps(pts, table, scene=view.points) or robot.grasps(pts, table)
        plan = robot.plan(goals, table, view.points) if goals else None
        if plan is None:
            print("no grasp plan:", robot.why() or "no candidates")
            return None
        out = robot.run(plan)
        if out.get("ended"):
            return None
        held = robot.holding()
        move_up(robot, 0.10, table)
        if held and robot.holding():
            return plan.goal.tcp, box
        print("grasp attempt", attempt, "missed")
        move_up(robot, 0.0, table, hand=(1.0,))
    return None


def solve(robot):
    instruction = robot.status()["instruction"]
    view = robot.look()
    table = robot.table(view)
    labels = [v for v in view.segments.values() if v not in ("table", "robot")]
    order = named(instruction, labels)
    if len(order) < 2:
        print("cannot parse", instruction, labels)
        return
    top, bottom = order[0], order[1]
    got = pick(robot, top, table)
    if got is None:
        return
    grasp_tcp, box = got
    view = robot.look()
    base = object_points(robot, view, table, bottom)
    if base is None:
        print("cannot see", bottom)
        return
    bbox = robot.box(base, table)
    obj_in_tcp = np.linalg.inv(grasp_tcp) @ box.pose
    support = bbox.bottom + bbox.size[2]
    plan = None
    for yaw in (bbox.yaw, bbox.yaw + np.pi / 2, None):
        goal = robot.place_at(obj_in_tcp, box, bbox.center[:2], yaw=yaw, table=table, hover=0.006, support=support)
        plan = robot.plan(goal, table, view.points)
        if plan is not None:
            break
        print("place failed:", robot.why())
    if plan is None:
        return
    out = robot.run(plan)
    if not out.get("ended"):
        move_up(robot, 0.08, table)
