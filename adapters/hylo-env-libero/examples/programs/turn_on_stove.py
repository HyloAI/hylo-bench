"""LIBERO: turn on the stove: grasp the stove knob's ridge from above and twist it counter-clockwise (seen from
above) by about 50 degrees (perception only: cameras, depth, the environment's segmentation)."""
import math

import numpy as np

ARM = "arm"
SCENERY = ("table", "robot", "floor")


def mentioned(text, labels):
    """Labels named in the text, in order of appearance (longer labels win where they overlap)."""
    text = " " + text.lower().replace(",", " ").replace(".", " ") + " "
    found, taken = [], []
    for lab in sorted(set(labels), key=len, reverse=True):
        start = 0
        while True:
            i = text.find(" " + lab + " ", start)
            if i < 0:
                break
            span = (i, i + len(lab) + 1)
            if not any(a < span[1] and span[0] < b for a, b in taken):
                taken.append(span)
                found.append((i, lab))
            start = i + 1
    return [pair[1] for pair in sorted(found)]


def short_names(labels):
    """Names an instruction may use for each label: the label itself, and its shorter word runs ('black bowl' ->
    'bowl', 'cream cheese box' -> 'cream cheese', 'wooden cabinet' -> 'cabinet'); a short name goes to the label
    with the fewest words that contains it."""
    labels = sorted(set(labels), key=lambda lab: (len(lab.split()), lab))
    out = {lab: lab for lab in labels}
    for lab in labels:
        words = lab.split()
        for i in range(len(words)):
            for j in range(i + 1, len(words) + 1):
                sub = " ".join(words[i:j])
                if sub in out or words[i] in ("of", "the", "and") or words[j - 1] in ("of", "the", "and", "box"):
                    continue
                out[sub] = lab
    return out


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


def lift(robot, dz, table):
    t = robot.status()["tcp"][ARM].copy()
    t[2, 3] += dz
    p = robot.plan(robot.Goal("lift", ARM, tcp=t), table)
    return robot.run(p) if p is not None else None


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def turn_knob(robot, knob, table, turn=0.9):
    view = robot.look()
    found = instances(robot, view, table, knob)
    if not found:
        print("cannot see", knob)
        return False
    pts = max(found.values(), key=len)
    top = float(np.percentile(pts[:, 2], 99))
    ridge = pts[pts[:, 2] > top - 0.012]                     # the grip fin on top of the dial
    rbox = robot.box(ridge)
    print("knob top", round(top, 3), "ridge", np.round(rbox.center, 3), np.round(rbox.size, 3), round(rbox.yaw, 2))
    across = np.array([-math.sin(rbox.yaw), math.cos(rbox.yaw), 0.0])
    centre = np.array([rbox.center[0], rbox.center[1], top - 0.012])
    goals = []
    for k, sgn in enumerate((1.0, -1.0)):
        g = robot.grasp_at(ARM, tcp_pose((0, 0, -1), sgn * across, centre))
        g.rank = k
        goals.append(g)
    plan = robot.plan(goals, table, view.points)
    if plan is None:
        print("no grasp plan:", robot.why())
        return False
    out = robot.run(plan)
    if out.get("ended"):
        return True
    print("holding knob:", robot.holding(ARM))
    t = robot.status()["tcp"][ARM].copy()
    for step in range(3):                                   # twist about the vertical through the knob
        t2 = t.copy()
        t2[:3, :3] = rotz(turn / 3) @ t[:3, :3]
        p = robot.plan(robot.Goal("lift", ARM, tcp=t2), table)
        if p is None:
            print("no twist plan:", robot.why())
            break
        out = robot.run(p)
        if out.get("ended"):
            return True
        t = robot.status()["tcp"][ARM].copy()
    p = robot.plan(robot.Goal("retreat", ARM, tcp=robot.status()["tcp"][ARM], hand=(1.0,)), table)
    if p is not None:
        robot.run(p)
    lift(robot, 0.08, table)
    return True


def solve(robot):
    view = robot.look()
    table = robot.table(view)
    labels = [v for v in view.segments.values() if v not in SCENERY]
    knobs = [lab for lab in labels if "knob" in lab]
    if not knobs:
        print("no knob among", labels)
        return
    turn_knob(robot, knobs[0], table)
