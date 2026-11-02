"""LIBERO long-horizon tasks (libero_10): the instruction is split into steps, each done with the same perception-only
skills as the single-step programs (cameras, depth, the environment's segmentation):

- "put both the A and the B in the basket", "put both moka pots on the stove": one pick-and-place per object;
- "put the X on the left / right plate": the plate further left / right in the agentview image;
- "put the X to the right of the plate": next to the plate on the image's right;
- "place it in the back compartment of the caddy": the caddy's side away from the camera, the object turned to fit;
- "turn on the stove": twist the knob; "... and put the moka pot on it": "it" is the previous target;
- "... in the bottom drawer of the cabinet and close it": place inside the open drawer, then push the drawer shut.
"""
import math

import numpy as np

ARM = "arm"
SCENERY = ("table", "robot", "floor")
VERBS = ("put", "place", "turn", "close", "open", "push", "pick")


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


def centre_of(robot, pts, table):
    on_table = pts[:, 2].min() < table.height + 0.02
    return robot.box(pts, table if on_table else None)


def round_centre(pts, fallback):
    """Centre of a round object (a plate) from its raised rim: a circle fitted to the highest ring of points, which
    stays right when the camera sees only part of it; `fallback` when the fit is implausible."""
    top = float(np.percentile(pts[:, 2], 98))
    rim = pts[pts[:, 2] > top - 0.005]
    if len(rim) < 20:
        return fallback
    x, y = rim[:, 0], rim[:, 1]
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    sol = np.linalg.lstsq(A, x * x + y * y, rcond=None)[0]
    c = np.array([sol[0] / 2, sol[1] / 2])
    r = math.sqrt(max(sol[2] + c @ c, 0.0))
    if not 0.03 < r < 0.15 or np.linalg.norm(c - fallback) > 0.08:
        return fallback
    return c


def camera_axes(view):
    """Horizontal unit vectors: image-right and towards the agentview camera (what 'right' and 'front' mean)."""
    T = np.asarray(view.cams["agentview"].T_world_cam)
    right = np.array([T[0, 0], T[1, 0], 0.0])
    toward = -np.array([T[0, 2], T[1, 2], 0.0])
    return right / np.linalg.norm(right), toward / np.linalg.norm(toward)


def image_u(view, p):
    cam = view.cams["agentview"]
    q = np.linalg.inv(np.asarray(cam.T_world_cam)) @ np.r_[p, 1.0]
    return float((np.asarray(cam.K) @ q[:3])[0] / q[2])


def clauses(instruction):
    """Steps in order: [{"verb", "text"}], with "pick up the X and place it ..." kept together."""
    text = " " + instruction.lower().strip().rstrip(".") + " "
    for a, b in ((" and place it ", " placing it "), (" and put it ", " placing it ")):
        text = text.replace(a, b)
    parts, cur = [], []
    words = text.split()
    for i, w in enumerate(words):
        if w == "and" and i + 1 < len(words) and words[i + 1] in VERBS and cur:
            parts.append(" ".join(cur))
            cur = []
            continue
        cur.append(w)
    parts.append(" ".join(cur))
    return parts


def place_steps(clause, names, last_target):
    """[(object label, target label, verb, modifiers)] for one put/place clause."""
    text = " " + clause + " "
    for p in (" pick up the ", " put the ", " place the ", " put ", " place "):
        if text.startswith(p):
            text = " " + text[len(p):]
            break
    text = text.replace(" placing it ", " ")
    mods = {}
    for kw, key in ((" to the right of the ", "right_of"), (" to the left of the ", "left_of"),
                    (" in front of the ", "front_of")):
        if kw in text:
            mods["relation"] = key
            text = text.replace(kw, " on the ")
    best = None
    for kw, verb in ((" on top of the ", "on"), (" on the ", "on"), (" in the ", "in"), (" into the ", "in"),
                     (" inside the ", "in"), (" on it", "on")):
        i = text.rfind(kw)
        if i >= 0 and (best is None or i > best[0]):
            best = (i, kw, verb)
    if best is None:
        return []
    head, tail = text[:best[0]], text[best[0] + len(best[1]):]
    if best[1] == " on it":
        tgt = last_target
    else:
        found = mentioned(" " + tail + " ", names)
        tgt = names[found[0]] if found else None
        for side in ("left", "right", "back", "front"):
            if f" {side} " in " " + tail + " ":
                mods["side"] = side
    objs = [names[m] for m in mentioned(" " + head + " ", names)]
    if " both " in " " + head + " " and len(objs) == 1:         # "both moka pots": two of the same label
        objs = objs * 2
    return [(o, tgt, best[2], dict(mods)) for o in objs if tgt]


def pick_instance(robot, view, table, label, avoid):
    """Points of an instance of `label` that is not at one of the `avoid` xy positions (already placed)."""
    cands = instances(robot, view, table, label)
    free = [p for p in cands.values()
            if all(np.linalg.norm(robot.box(p, table).center[:2] - a) > 0.06 for a in avoid)]
    return max(free or list(cands.values()), key=len) if cands else None


def side_grasps(robot, box, pts, below=0.04):
    """For a tall object with a handle (a moka pot): horizontal approaches along its long side, closing across the
    narrow side `below` metres under the top; the end away from the handle first (the handle is the sparse end:
    the box centre lies off the body's median towards it)."""
    if box.size[2] < 0.09:
        return []
    top = float(np.percentile(pts[:, 2], 99))
    along = np.array([math.cos(box.yaw), math.sin(box.yaw), 0.0])
    across = np.array([-along[1], along[0], 0.0])
    body = np.median(pts[:, :2], axis=0)
    handle_side = 1.0 if (box.center[:2] - body) @ along[:2] > 0 else -1.0
    goals = []
    for k, end in enumerate((-handle_side, handle_side)):
        approach = -end * along                            # from that end towards the body
        p0 = np.array([body[0], body[1], top - below]) - approach * 0.008
        g = robot.grasp_at(ARM, tcp_pose(approach, across, p0))
        g.rank = k
        goals.append(g)
    return goals


def grasp(robot, label, table, avoid, tries=2):
    for attempt in range(tries):
        view = robot.look()
        pts = pick_instance(robot, view, table, label, avoid)
        if pts is None:
            print("cannot see", label)
            return None
        box = centre_of(robot, pts, table)
        wide = box.size[1] > robot.spec.arm(ARM).hand.max_width - 0.01
        if wide:
            options = (lambda: rim_grasps(robot, box), lambda: rim_grasps(robot, box, pre=0.04))
        else:
            options = (lambda: robot.grasps(ARM, pts, table, scene=view.points), lambda: robot.grasps(ARM, pts, table),
                       lambda: side_grasps(robot, box, pts),
                       # tall objects: around the lid / upper body (a knob on top only touches the palm), then shallow
                       lambda: top_grasps(robot, box, pts, depth=0.03) if box.size[2] > 0.1 else [],
                       lambda: top_grasps(robot, box, pts))
        plan = None
        for make in options:
            goals = make()
            if goals:
                plan = robot.plan(goals, table, view.points)
            if plan is not None:
                break
        if plan is None:
            print("no grasp plan for", label, ":", robot.why())
            return None
        out = robot.run(plan)
        if out.get("ended"):
            return None
        held = robot.holding(ARM)
        lift(robot, 0.12, table)
        if held and robot.holding(ARM):
            return plan.goal.tcp, box
        print("grasp of", label, "missed (attempt", attempt, ")")
        p = robot.plan(robot.Goal("retreat", ARM, tcp=robot.status()["tcp"][ARM], hand=(1.0,)), table)
        if p is not None:
            robot.run(p)
    return None


def target_xy(robot, view, table, target, mods, k_same, n_same=1):
    """Where the object goes: the target's centre, the instance on the named side, beside it, or a compartment;
    k_same: how many objects already went to this target (spread them out)."""
    found = instances(robot, view, table, target)
    if not found:
        return None, None
    right, toward = camera_axes(view)
    pairs = [(centre_of(robot, p, table), p) for p in found.values()]
    side = mods.get("side")
    if side in ("left", "right") and len(pairs) > 1:
        us = [image_u(view, bp[0].center) for bp in pairs]
        box, pts = pairs[int(np.argmin(us) if side == "left" else np.argmax(us))]
    else:
        box, pts = max(pairs, key=lambda bp: len(bp[1]))
    xy = box.center[:2].copy()
    if target.split()[-1] == "plate":                     # round: the rim gives the centre even when cut off
        xy = round_centre(pts, xy)
    rel = mods.get("relation")
    if rel in ("right_of", "left_of"):
        reach = 0.5 * float(np.max(np.abs((box.size[0] * np.array([math.cos(box.yaw), math.sin(box.yaw)]),
                                          box.size[1] * np.array([-math.sin(box.yaw), math.cos(box.yaw)])) @ right[:2])))
        xy = xy + (1 if rel == "right_of" else -1) * right[:2] * (reach + 0.05)
    elif side in ("back", "front"):
        xy = xy + (-1 if side == "back" else 1) * toward[:2] * 0.035
    elif n_same > 1:                                       # several objects for one target: one behind the other
        xy = xy - toward[:2] * min(0.06, 0.3 * box.size[0]) * (2 * k_same - (n_same - 1)) / max(1, n_same - 1)
        # (the first one furthest from the robot, so the arm never reaches over what it has placed)
    return xy, box


def place(robot, grasp_tcp, box, target, verb, mods, table, k_same, n_same=1):
    view = robot.look()
    xy, tbox = target_xy(robot, view, table, target, mods, k_same, n_same)
    if xy is None:
        print("cannot see", target)
        return None
    print("place on/in", target, mods, "at", np.round(xy, 3))
    obj_in_tcp = np.linalg.inv(grasp_tcp) @ box.pose
    yaw_now = math.atan2(box.pose[1, 0], box.pose[0, 0])
    yaw_fit = None
    if mods.get("side") in ("back", "front"):              # a slot: the object's length along the slot
        yaw_fit = tbox.yaw
    hover = 0.02 if verb == "in" else 0.01
    plan = None
    for k, dyaw in enumerate((0.0, 0.6, -0.6, 1.2, -1.2, math.pi / 2)):
        yaw = (yaw_fit if yaw_fit is not None else yaw_now) + dyaw
        goal = robot.place_at(ARM, obj_in_tcp, box, xy, yaw=yaw if (yaw_fit is not None or dyaw) else None,
                              table=table, hover=hover, scene=None if mods.get("relation") else view.points)
        plan = robot.plan(goal, table, view.points)
        if plan is not None:
            break
    if plan is None:
        print("no place plan:", robot.why())
        return None
    out = robot.run(plan)
    if not out.get("ended"):
        lift(robot, 0.10, table)
    return xy


def close_drawer(robot, drawer, cabinet, table):
    """Push an open drawer shut: along the direction from the cabinet body to the drawer, from its front face."""
    view = robot.look()
    dr, body = instances(robot, view, table, drawer), instances(robot, view, table, cabinet)
    if not dr or not body:
        print("cannot see", drawer if not dr else cabinet)
        return False
    pts, bpts = max(dr.values(), key=len), max(body.values(), key=len)
    bbox = robot.box(bpts, table)
    out_dir = robot.box(pts, table).center - bbox.center
    out_dir[2] = 0.0
    axes = [np.array([math.cos(bbox.yaw + a), math.sin(bbox.yaw + a), 0.0]) for a in (0, math.pi / 2, math.pi, -math.pi / 2)]
    n = max(axes, key=lambda a: a @ out_dir)               # snapped to the cabinet's sides
    front = (pts @ n).max()
    face = (bpts @ n).max()
    face_pts = pts[pts @ n > front - 0.02]
    c = face_pts.mean(0)
    c = c - n * ((c @ n) - front)                          # on the front plane
    c[2] = float(np.clip(c[2], table.height + 0.055, None))   # the hand is 7 cm tall here
    side = np.cross([0.0, 0.0, 1.0], n)
    print("close", drawer, "normal", np.round(n, 2), "open by", round(float(front - face), 3))
    goals = []      # the closed fingertips just in front of the drawer front (handle included), the hand tilted
    k = 0           # back over the drawer so its palm clears the fronts above while it pushes
    for tilt in (0.8, 0.0):
        a = np.array([0.0, 0.0, -1.0]) * math.cos(tilt) - n * math.sin(tilt)
        for off in (0.0, 0.04, -0.04):
            p0 = c + n * 0.03 + side * off
            p0[2] = max(float(c[2]), table.height + 0.035)
            start = tcp_pose(a, side, p0)
            pre = start.copy()
            pre[2, 3] += 0.10                              # from straight above
            goals.append(robot.Goal("reach", ARM, tcp=start, pre=pre, hand_before=(0.0,), rank=k))
            k += 1
    p = robot.plan(goals, table, view.points)
    if p is None:
        print("no plan to the drawer front:", robot.why())
        return False
    print("pushing with contact pose", int(p.goal.rank), "of", len(goals), "(0-2 tilted)")
    if robot.run(p).get("ended"):
        return True
    t = robot.status()["tcp"][ARM].copy()
    t[:3, 3] -= n * (max(0.0, float(front - face)) + 0.03)
    p = robot.plan(robot.Goal("push", ARM, tcp=t), table)
    if p is None:
        print("no push plan:", robot.why())
        return False
    robot.run(p)
    t = robot.status()["tcp"][ARM].copy()
    t[:3, 3] += n * 0.08 + np.array([0.0, 0.0, 0.08])
    p = robot.plan(robot.Goal("retreat", ARM, tcp=t), table)
    if p is not None:
        robot.run(p)
    return True


def solve(robot):
    instruction = robot.status()["instruction"]
    view = robot.look()
    table = robot.table(view)
    labels = [v for v in view.segments.values() if v not in SCENERY]
    names = short_names(labels)
    placed: dict = {}                                       # target -> xy of what was put there
    last_target = None
    for clause in clauses(instruction):
        words = clause.split()
        print("step:", clause)
        if words[:2] == ["turn", "on"]:
            knobs = [lab for lab in labels if "knob" in lab]
            if knobs:
                turn_knob(robot, knobs[0], table)
            last_target = names.get(" ".join(words[3:]), last_target) if len(words) > 3 else last_target
            continue
        if words[:1] == ["close"]:
            if last_target and "drawer" in last_target:
                cab = last_target.split(" of the ", 1)[1] if " of the " in last_target else "cabinet"
                close_drawer(robot, last_target, cab, table)
            else:
                print("cannot close", last_target)
            continue
        steps = place_steps(clause, names, last_target)
        for obj, tgt, verb, mods in steps:
            print("  move", obj, "->", verb, tgt, mods)
            avoid = [xy for xys in placed.values() for xy in xys]
            got = grasp(robot, obj, table, avoid)
            if got is None:
                return
            n_same = sum(1 for st in steps if st[1] == tgt)
            xy = place(robot, got[0], got[1], tgt, verb, mods, table, len(placed.get(tgt, [])), n_same)
            if xy is None:
                return
            placed.setdefault(tgt, []).append(np.asarray(xy))
            last_target = tgt
