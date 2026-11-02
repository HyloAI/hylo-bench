"""LIBERO pick and place: "pick up the X [qualifier] and place it on/in the Y", "put the X on/in the Y".

Perception only (cameras, depth, the environment's segmentation). Objects narrower than the open hand are grasped
with the toolkit's grasps(); wide hollow objects (bowls) by the rim. A qualifier ("between the plate and the
ramekin", "next to the cookie box", "on the stove", "in the top drawer of the wooden cabinet", "from table center")
picks one of several instances with the same label.
"""
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


def parse(instruction, labels):
    """(object label, qualifier text, verb 'on'/'in', target label)."""
    names = short_names(labels)
    text = instruction.lower().strip().rstrip(".")
    for p in ("pick up the ", "put the ", "place the ", "move the "):
        if text.startswith(p):
            text = text[len(p):]
            break
    # destination: the last " on (top of) the " / " in the " / " into the "
    best = None
    for kw, verb in ((" on top of the ", "on"), (" on the ", "on"), (" in the ", "in"), (" into the ", "in"),
                     (" inside the ", "in")):
        i = text.rfind(kw)
        if i >= 0 and (best is None or i > best[0]):
            best = (i, kw, verb)
    head, verb, tail = text, "on", ""
    if best is not None:
        head, verb, tail = text[:best[0]], best[2], text[best[0] + len(best[1]):]
    head = head.replace(" and place it", "").replace(" and put it", "")
    tail = tail.replace("place it ", "")
    obj = mentioned(head, names)
    tgt = mentioned(tail, names)
    if not obj or not tgt:
        return None
    qual = head[head.find(obj[0]) + len(obj[0]):] if obj[0] in head else ""
    return names[obj[0]], qual.strip(), verb, names[tgt[0]], [names[x] for x in mentioned(qual, names)]


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


def choose(robot, view, table, label, qual, refs):
    """The instance of `label` the qualifier means; returns its points (or None)."""
    cands = instances(robot, view, table, label)
    if not cands:
        return None
    if len(cands) == 1:
        return next(iter(cands.values()))
    centres = {i: robot.box(p, table).center for i, p in cands.items()}
    ref_pts = []
    for r in refs:
        found = instances(robot, view, table, r)
        if found:
            ref_pts.append(max(found.values(), key=len))
    if "between" in qual and len(ref_pts) >= 2:
        goal = (robot.box(ref_pts[0], table).center + robot.box(ref_pts[1], table).center) / 2
    elif ref_pts:
        goal = robot.box(ref_pts[0], table).center
    elif "center" in qual or "centre" in qual:
        ext = np.asarray(table.extent)
        goal = np.array([ext[0].mean(), ext[1].mean(), table.height])
    else:
        return max(cands.values(), key=len)
    best = min(cands, key=lambda i: np.linalg.norm(centres[i][:2] - goal[:2]))
    print("chose instance", best, "at", np.round(centres[best], 3), "for", qual or label)
    return cands[best]


def tcp_pose(x_axis, y_axis, origin):
    x = np.asarray(x_axis, float)
    x /= np.linalg.norm(x)
    y = np.asarray(y_axis, float)
    y = y - (y @ x) * x
    y /= np.linalg.norm(y)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, np.cross(x, y), origin
    return T


def rim_grasps(robot, box, n=12, pre=0.10):
    """Top-down grasps on the rim of a bowl: one finger inside, one outside, closing radially; the side facing
    the robot first."""
    r = 0.5 * float(min(box.size[0], box.size[1]))
    top = box.center[2] + box.size[2] / 2
    c = box.center
    toward_base = math.atan2(-c[1], -c[0])
    goals = []
    for k in range(n):
        a = toward_base + (k // 2 + 1 if k % 2 else -(k // 2)) * 2 * math.pi / n
        radial = np.array([math.cos(a), math.sin(a), 0.0])
        p = np.array([c[0], c[1], top - 0.018]) + radial * (r - 0.008)
        g = robot.grasp_at(ARM, tcp_pose((0, 0, -1), radial, p))
        g.pre = g.tcp.copy()
        g.pre[2, 3] += pre
        g.rank = k
        goals.append(g)
    return goals


def top_grasps(robot, box, pts, depth=0.012):
    """Top-down grasps a little below the highest points, closing across the box's width, then its length."""
    top = float(np.percentile(pts[:, 2], 99))
    goals = []
    for k, extra in enumerate((0.0, math.pi / 2, math.pi / 4, -math.pi / 4)):
        th = box.yaw + math.pi / 2 + extra
        closing = np.array([math.cos(th), math.sin(th), 0.0])
        p = np.array([box.center[0], box.center[1], max(top - depth, box.bottom + 0.01)])
        g = robot.grasp_at(ARM, tcp_pose((0, 0, -1), closing, p))
        g.rank = k
        goals.append(g)
    return goals


def lift(robot, dz, table):
    t = robot.status()["tcp"][ARM].copy()
    t[2, 3] += dz
    p = robot.plan(robot.Goal("lift", ARM, tcp=t), table)
    return robot.run(p) if p is not None else None


def pick(robot, label, qual, refs, table, tries=2):
    """Grasp and lift; returns (grasp tcp, box of the object) or None."""
    for attempt in range(tries):
        view = robot.look()
        pts = choose(robot, view, table, label, qual, refs)
        if pts is None:
            print("cannot see", label)
            return None
        # resting on the table: the box reaches down to it (hidden sides filled); on something else (a drawer, a
        # box, the stove): only as low as its own points
        on_table = pts[:, 2].min() < table.height + 0.02
        box = robot.box(pts, table if on_table else None)
        print(label, "box", np.round(box.center, 3), np.round(box.size, 3))
        wide = box.size[1] > robot.spec.arm(ARM).hand.max_width - 0.01
        plan = None
        # neighbours first kept clear of the open hand; then only the object itself (the planner still avoids
        # everything seen on the way); then shallower top grasps of our own
        if wide:                                           # rim: from 10 cm above, else a short approach
            options = (lambda: rim_grasps(robot, box), lambda: rim_grasps(robot, box, pre=0.04))
        else:
            options = (lambda: robot.grasps(ARM, pts, table, scene=view.points), lambda: robot.grasps(ARM, pts, table),
                       lambda: top_grasps(robot, box, pts))
        for make_goals in options:
            goals = make_goals()
            if goals:
                plan = robot.plan(goals, table, view.points)
            if plan is not None:
                break
            print("grasp: no plan yet:", robot.why() if goals else "no candidates")
        if plan is None:
            print("no grasp plan:", robot.why())
            return None
        out = robot.run(plan)
        if out.get("ended"):
            return None
        held = robot.holding(ARM)
        lift(robot, 0.12, table)
        if held and robot.holding(ARM):
            return plan.goal.tcp, box
        print("grasp attempt", attempt, "missed")
        p = robot.plan(robot.Goal("retreat", ARM, tcp=robot.status()["tcp"][ARM], hand=(1.0,)), table)
        if p is not None:
            robot.run(p)
    return None


def place(robot, grasp_tcp, box, target, verb, table):
    view = robot.look()
    found = instances(robot, view, table, target)
    if not found:
        print("cannot see", target)
        return False
    tpts = max(found.values(), key=len)
    tbox = robot.box(tpts, table)
    xy = tbox.center[:2]
    print("target", target, "box", np.round(tbox.center, 3), np.round(tbox.size, 3))
    hover = 0.02 if verb == "in" else 0.01
    obj_in_tcp = np.linalg.inv(grasp_tcp) @ box.pose
    yaw_now = math.atan2(box.pose[1, 0], box.pose[0, 0])
    plan = None
    # as held first; then turned about the vertical (the wrist may not reach the first), then nearer the robot
    for k, (dyaw, shift) in enumerate(((None, 0.0), (0.6, 0.0), (-0.6, 0.0), (1.2, 0.0), (-1.2, 0.0), (None, 0.03))):
        toward = -xy / (np.linalg.norm(xy) or 1.0)
        goal = robot.place_at(ARM, obj_in_tcp, box, xy + shift * toward, yaw=None if dyaw is None else yaw_now + dyaw,
                              table=table, hover=hover, scene=view.points)
        plan = robot.plan(goal, table, view.points)
        if plan is not None:
            break
        print("place attempt", k, "failed:", robot.why())
    if plan is None:
        print("no place plan")
        return False
    out = robot.run(plan)
    if not out.get("ended"):
        lift(robot, 0.08, table)
    return True


def solve(robot):
    instruction = robot.status()["instruction"]
    view = robot.look()
    table = robot.table(view)
    labels = [v for v in view.segments.values() if v not in SCENERY]
    task = parse(instruction, labels)
    if task is None:
        print("cannot parse", instruction, labels)
        return
    label, qual, verb, target, refs = task
    print("object:", label, "| qualifier:", qual, refs, "|", verb, target)
    got = pick(robot, label, qual, refs, table)
    if got is None:
        return
    place(robot, got[0], got[1], target, verb, table)
