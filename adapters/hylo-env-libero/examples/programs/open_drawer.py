"""LIBERO: open a drawer ("open the middle drawer of the cabinet", "open the top drawer ..."): grasp the drawer's
handle from the front and pull it straight out (perception only: cameras, depth, the environment's segmentation)."""
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


def front_normal(robot, drawer_pts, body_pts, table):
    """Horizontal outward normal of a drawer front: the drawer box's thin axis, pointing away from the cabinet."""
    box = robot.box(drawer_pts, table)
    wid = np.array([-math.sin(box.yaw), math.cos(box.yaw), 0.0])
    away = box.center - robot.box(body_pts, table).center
    return wid if wid @ away >= 0 else -wid


def open_drawer(robot, drawer, cabinet, table, pull=0.19, tries=2):
    """Grasp the drawer's handle and pull it straight out; looks again and retries once if it did not move."""
    for attempt in range(tries):
        view = robot.look()
        found = instances(robot, view, table, drawer)
        body = instances(robot, view, table, cabinet)
        if not found or not body:
            print("cannot see", drawer if not found else cabinet)
            return False
        pts = max(found.values(), key=len)
        n = front_normal(robot, pts, max(body.values(), key=len), table)
        proj = pts @ n
        if attempt > 0 and proj.max() - front > 0.12:
            return True                                    # it is open after all
        front = proj.max()
        handle = pts[proj > proj.max() - 0.012]            # what sticks out of the front: the handle
        c = handle.mean(0)
        print("drawer normal", np.round(n, 2), "handle", np.round(c, 3), len(handle), "points")
        side = np.cross([0.0, 0.0, 1.0], n)
        tall = np.ptp(handle[:, 2]) > np.ptp(handle @ side)  # vertical bar: close across it horizontally
        closing = side if tall else np.array([0.0, 0.0, 1.0])
        # open the hand only a little wider than the handle: fully open fingers reach the neighbouring handles
        opening = min(1.0, (float(np.ptp(handle @ closing)) + 0.025) / robot.spec.arm(ARM).hand.max_width)
        goals = []
        for k, (dz, back) in enumerate(((0.0, 0.06), (0.005, 0.06), (0.0, 0.10), (0.01, 0.10))):
            g = robot.grasp_at(ARM, tcp_pose(-n, closing, c + np.array([0.0, 0.0, dz]) + n * 0.002))
            g.pre = g.tcp.copy()
            g.pre[:3, 3] += n * back                       # short straight approach: the hand is tall when the
            g.hand_before = (opening,)                      # pads close vertically
            g.rank = k
            goals.append(g)
        plan = robot.plan(goals, table, view.points)
        if plan is None:
            print("no grasp plan:", robot.why())
            return False
        out = robot.run(plan)
        if out.get("ended"):
            return True
        print("holding handle:", robot.holding(ARM), "hand", np.round(out["hand"][ARM], 2))
        t = robot.status()["tcp"][ARM].copy()
        t[:3, 3] += n * pull
        p = robot.plan(robot.Goal("push", ARM, tcp=t), table)
        if p is None:
            print("no pull plan:", robot.why())
            return False
        out = robot.run(p)
        if out.get("ended"):
            return True
        p = robot.plan(robot.Goal("retreat", ARM, tcp=robot.status()["tcp"][ARM], hand=(opening,)), table)
        if p is not None:
            robot.run(p)
        t = robot.status()["tcp"][ARM].copy()             # back off and up, out of the cameras' way
        t[:3, 3] += n * 0.06 + np.array([0.0, 0.0, 0.12])
        p = robot.plan(robot.Goal("retreat", ARM, tcp=t), table)
        if p is not None:
            robot.run(p)
        view = robot.look()
        now = instances(robot, view, table, drawer)
        moved = max(now.values(), key=len) @ n if now else None
        if moved is not None and moved.max() - front > 0.12:
            return True
        print("drawer moved only", None if moved is None else round(float(moved.max() - front), 3), "m; again")
    return False

def solve(robot):
    instruction = robot.status()["instruction"].lower()
    view = robot.look()
    table = robot.table(view)
    labels = [v for v in view.segments.values() if v not in SCENERY]
    names = short_names(labels)
    drawers = [names[m] for m in mentioned(instruction, names) if "drawer" in names[m]]
    if not drawers:
        print("no drawer named in", instruction, labels)
        return
    drawer = drawers[0]
    cabinet = drawer.split(" of the ", 1)[1] if " of the " in drawer else "cabinet"
    print("drawer:", drawer, "| cabinet:", cabinet)
    open_drawer(robot, drawer, cabinet, table)
