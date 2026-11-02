"""RoboCasa PickPlaceCounterToSink: "pick the <object> from the counter and place it in the sink". Perception only.

The object is the segment whose label the instruction names. The program grasps it with the toolkit's ranked grasps
(retrying once from a fresh look), lifts it, then lowers it into the sink: the basin's floor is the lowest part of the
sink's points and the target is the middle of that floor; the object is let go a few centimetres above it and the hand
moves up and away."""
import math

import numpy as np

ARM = "arm"


def detections(robot, view, table, label, min_points=10):
    return [d for d in robot.objects(view, robot.find(view, label), table) if len(d.points) >= min_points]


def move(robot, dz, table, hand=None):
    t = robot.status()["tcp"][ARM].copy()
    t[2, 3] += dz
    p = robot.plan(robot.Goal("lift", ARM, tcp=t, hand=hand), table)
    return robot.run(p) if p is not None else None


def named_object(instruction, labels):
    text = instruction.lower()
    hits = [lab for lab in labels if lab in text and lab not in ("counter", "sink", "robot")]
    return max(hits, key=len) if hits else None


def solve(robot):
    instruction = robot.status()["instruction"]
    view = robot.look()
    table = robot.table(view)
    label = named_object(instruction, set(view.segments.values()))
    if label is None:
        print("no segment named in", repr(instruction), sorted(set(view.segments.values())))
        return
    sink = [d.points for d in detections(robot, view, table, "sink", 20)]
    if not sink:
        print("cannot see the sink")
        return
    sp = np.concatenate(sink)
    floor = float(np.percentile(sp[:, 2], 5))
    basin = sp[sp[:, 2] < floor + 0.03]
    goal_xy = basin[:, :2].mean(axis=0)
    print(label, "-> sink floor", round(floor, 3), "at", np.round(goal_xy, 3))
    got = None
    for attempt in range(2):
        view = robot.look()
        dets = detections(robot, view, table, label)
        if not dets:
            print("cannot see", label)
            return
        pts = max(dets, key=lambda d: len(d.points)).points
        box = robot.box(pts, table)
        goals = robot.grasps(pts, table, scene=view.points) or robot.grasps(pts, table)
        plan = robot.plan(goals, table, view.points) if goals else None
        if plan is None:
            print("no grasp plan:", robot.why() or "no candidates")
            return
        if robot.run(plan).get("ended"):
            return
        ok = robot.holding()
        move(robot, 0.15, table)
        if ok and robot.holding():
            got = (plan.goal.tcp, box)
            break
        print("grasp attempt", attempt, "missed")
        move(robot, 0.0, table, hand=(1.0,))
    if got is None:
        return
    grasp_tcp, box = got
    view = robot.look()
    obj_in_tcp = np.linalg.inv(grasp_tcp) @ box.pose
    yaw_now = math.atan2(box.pose[1, 0], box.pose[0, 0])
    plan = None
    for k, (dyaw, shift) in enumerate(((None, 0.0), (0.7, 0.0), (-0.7, 0.0), (None, 0.05), (1.4, 0.05))):
        toward = -goal_xy / np.linalg.norm(goal_xy)
        goal = robot.place_at(obj_in_tcp, box, goal_xy + shift * toward, yaw=None if dyaw is None else yaw_now + dyaw,
                              table=table, hover=0.03, support=floor)
        plan = robot.plan(goal, table, view.points)
        if plan is not None:
            break
        print("place attempt", k, "failed:", robot.why())
    if plan is None:
        return
    if not robot.run(plan).get("ended"):
        move(robot, 0.15, table)
