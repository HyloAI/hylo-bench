"""robosuite Lift: "lift the red cube off the table". Perception only (cameras, depth, the environment's
segmentation): find the cube, take the toolkit's best collision-free grasp, close and lift; one retry from a fresh
look when the hand comes up empty."""
import numpy as np


def object_points(robot, view, table, label):
    dets = [d for d in robot.objects(view, robot.find(view, label), table) if len(d.points) >= 10]
    return max(dets, key=lambda d: len(d.points)).points if dets else None


def move_up(robot, dz, table):
    t = robot.status()["tcp"]["arm"].copy()
    t[2, 3] += dz
    p = robot.plan(robot.Goal("lift", "arm", tcp=t), table)
    return robot.run(p) if p is not None else None


def solve(robot):
    instruction = robot.status()["instruction"]
    labels = [v for v in robot.look().segments.values() if v not in ("table", "robot")]
    label = next((lab for lab in labels if lab in instruction), labels[0])
    for attempt in range(3):
        view = robot.look()
        table = robot.table(view)
        pts = object_points(robot, view, table, label)
        if pts is None:
            print("cannot see", label)
            return
        print(label, "at", np.round(robot.box(pts, table).center, 3))
        goals = robot.grasps(pts, table, scene=view.points) or robot.grasps(pts, table)
        plan = robot.plan(goals, table, view.points) if goals else None
        if plan is None:
            print("no grasp plan:", robot.why() or "no candidates")
            return
        out = robot.run(plan)
        if out.get("ended"):
            return
        held = robot.holding()
        out = move_up(robot, 0.15, table)
        if out is not None and out.get("ended"):
            return
        if held and robot.holding():
            return
        print("attempt", attempt, "missed; opening and retrying")
        p = robot.plan(robot.Goal("open", "arm", tcp=robot.status()["tcp"]["arm"], hand=(1.0,)), table)
        if p is not None:
            robot.run(p)
