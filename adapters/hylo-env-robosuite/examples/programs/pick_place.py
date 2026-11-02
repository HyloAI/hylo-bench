"""robosuite PickPlace (single object): "pick up the X from the source bin and put it in the target bin, in the
compartment marked with a X marker". Perception only (cameras, depth, the environment's segmentation).

The marker is a see-through copy of the object standing in its compartment: it shows where to go but is not solid,
so its points are left out of the planning scene and the object is set down on the target bin's floor (its lowest
points) at the marker's position. After letting go the hand moves well clear: success needs it at least 7 cm away.
"""
import math

import numpy as np

ARM = "arm"


def detections(robot, view, table, label, min_points=10):
    return [d for d in robot.objects(view, robot.find(view, label), table) if len(d.points) >= min_points]


def points_of(robot, view, table, label):
    dets = detections(robot, view, table, label)
    return max(dets, key=lambda d: len(d.points)).points if dets else None


def without(points, remove, voxel=0.01):
    """`points` minus everything within about a voxel of `remove`."""
    if remove is None or not len(remove):
        return points
    keys = {tuple(k) for k in np.floor(remove / voxel).astype(np.int64)}
    near = set()
    for k in keys:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    near.add((k[0] + dx, k[1] + dy, k[2] + dz))
    pk = np.floor(points / voxel).astype(np.int64)
    keep = np.array([tuple(k) not in near for k in pk], dtype=bool)
    return points[keep]


def marker_xy(robot, view, label):
    """Where a see-through marker stands: it writes no depth, so its mask's depth is the floor right behind it.
    Back-projected from the camera looking most steeply down (there the floor behind is nearly under it)."""
    best = None
    for m in robot.find(view, label):
        cam = view.cams[m.camera]
        if cam.depth is None or m.mask.sum() < 20:
            continue
        down = -float(np.asarray(cam.T_world_cam)[2, 2])           # optical axis . (0, 0, -1)
        if best is None or down > best[0]:
            best = (down, cam, m.mask)
    if best is None:
        return None
    cam, mask = best[1], best[2]
    v, u = np.nonzero(mask & (cam.depth > 0))
    z = cam.depth[v, u]
    K, T = np.asarray(cam.K), np.asarray(cam.T_world_cam)
    p = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z, np.ones_like(z)], axis=1)
    w = p @ T.T
    return np.median(w[:, :2], axis=0)


def move(robot, dz, table, hand=None, dxy=(0.0, 0.0)):
    t = robot.status()["tcp"][ARM].copy()
    t[0, 3] += dxy[0]
    t[1, 3] += dxy[1]
    t[2, 3] += dz
    p = robot.plan(robot.Goal("move", ARM, tcp=t, hand=hand), table)
    return robot.run(p) if p is not None else None


def top_grasps(robot, box, pts):
    """Top-down grasps a little below the top, closing across the box's narrow side, then its long side."""
    top = float(np.percentile(pts[:, 2], 98))
    out = []
    for k, extra in enumerate((0.0, math.pi / 2)):
        th = box.yaw + math.pi / 2 + extra
        y = np.array([math.cos(th), math.sin(th), 0.0])
        x = np.array([0.0, 0.0, -1.0])
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2] = x, y, np.cross(x, y)
        T[:3, 3] = [box.center[0], box.center[1], max(top - 0.025, box.bottom + 0.012)]
        g = robot.grasp_at(T, arm=ARM)
        g.rank = k
        out.append(g)
    return out


def pick(robot, label, table, scene_drop):
    for attempt in range(3):
        view = robot.look()
        pts = points_of(robot, view, table, label)
        if pts is None:
            print("cannot see", label)
            return None
        box = robot.box(pts)                           # in a bin: no table under it, only its own points
        print(label, "box", np.round(box.center, 3), np.round(box.size, 3))
        scene = without(view.points, scene_drop(view))
        plan = None
        for make in (lambda: robot.grasps(pts, table, scene=scene), lambda: robot.grasps(pts, table),
                     lambda: top_grasps(robot, box, pts)):
            goals = make()
            plan = robot.plan(goals, table, scene) if goals else None
            if plan is not None:
                break
            print("grasp: no plan yet:", robot.why() or "no candidates")
        if plan is None:
            return None
        out = robot.run(plan)
        if out.get("ended"):
            return None
        held = robot.holding(ARM)
        move(robot, 0.15, table)
        if held and robot.holding(ARM):
            return plan.goal.tcp, box
        print("grasp attempt", attempt, "missed")
        move(robot, 0.0, table, hand=(1.0,))
    return None


def solve(robot):
    instruction = robot.status()["instruction"].lower()
    view = robot.look()
    table = robot.table(view)
    labels = sorted({v for v in view.segments.values() if v not in ("table", "robot")}, key=len, reverse=True)
    marker = next((lab for lab in labels if lab.endswith(" marker") and lab in instruction), None)
    if marker is None:
        print("cannot parse", instruction)
        return
    label = marker[:-len(" marker")]
    markers = [lab for lab in labels if lab.endswith(" marker")]

    def marker_points(v):                     # markers write no depth, but keep this in case a renderer does
        found = [d.points for m in markers for d in detections(robot, v, table, m, 5)]
        return np.concatenate(found) if found else None

    # where to go, measured before the arm is in the way: the marker's centre and the bin floor around it
    goal_xy = marker_xy(robot, view, marker)
    bin_pts = points_of(robot, view, table, "target bin")
    if goal_xy is None or bin_pts is None:
        print("cannot see the", marker if goal_xy is None else "target bin")
        return
    near = bin_pts[np.linalg.norm(bin_pts[:, :2] - goal_xy, axis=1) < 0.09]
    floor = float(np.percentile((near if len(near) > 20 else bin_pts)[:, 2], 5))
    print("marker at", np.round(goal_xy, 3), "bin floor", round(floor, 3))
    got = pick(robot, label, table, marker_points)
    if got is None:
        return
    grasp_tcp, box = got
    view = robot.look()
    scene = without(view.points, marker_points(view))
    obj_in_tcp = np.linalg.inv(grasp_tcp) @ box.pose
    yaw_now = math.atan2(box.pose[1, 0], box.pose[0, 0])
    toward = -goal_xy / np.linalg.norm(goal_xy)          # towards the robot's base: easier to reach
    plan = None
    tries = [(None, 0.0), (0.6, 0.0), (-0.6, 0.0), (None, 0.04), (1.2, 0.04), (-1.2, 0.04), (None, 0.07),
             (math.pi / 2, 0.07)]
    for k, (dyaw, shift) in enumerate(tries):
        goal = robot.place_at(obj_in_tcp, box, goal_xy + shift * toward,
                              yaw=None if dyaw is None else yaw_now + dyaw, table=table, hover=0.02, support=floor)
        goal.pre = goal.tcp.copy()
        goal.pre[2, 3] += 0.05
        plan = robot.plan(goal, table, scene)
        if plan is not None:
            break
        print("place attempt", k, "failed:", robot.why())
    if plan is None:
        return
    out = robot.run(plan)
    if out.get("ended"):
        return
    move(robot, 0.15, table)
