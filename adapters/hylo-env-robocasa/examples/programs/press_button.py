"""RoboCasa TurnOnMicrowave (and other "press the ... button" tasks): "press the start button on the microwave".
Perception only.

The segmentation labels the referenced appliance's buttons ("start button", "stop button"). The button's face points
at the robot (its normal comes from the appliance's front around it). The fingers close, come to a spot in front of
the button with the hand pointing at it, push straight in until the fingertips are 1.5 cm past the button's surface,
and come back out."""
import numpy as np

ARM = "arm"


def points_of(robot, view, table, label, min_points=8):
    """World points of a label from every camera's mask and depth. Not robot.objects(): it drops points on or under
    the support plane, and drawers, lower cabinets and front knobs sit below the counter top."""
    out = []
    for m in robot.find(view, label):
        cam = view.cams[m.camera]
        if cam.depth is None or cam.K is None:
            continue
        v, u = np.nonzero(m.mask & (cam.depth > 0))
        if len(v) < 3:
            continue
        z = cam.depth[v, u]
        K, T = np.asarray(cam.K), np.asarray(cam.T_world_cam)
        p = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z, np.ones_like(z)], axis=1)
        out.append((p @ T.T)[:, :3])
    pts = np.concatenate(out) if out else np.zeros((0, 3))
    return pts if len(pts) >= min_points else None


def front_normal(points, centre):
    c = points.mean(axis=0)
    vt = np.linalg.svd(points - c, full_matrices=False)[2]
    n = vt[2] * np.array([1.0, 1.0, 0.0])
    if np.linalg.norm(n) < 0.3:
        n = np.array([-centre[0], -centre[1], 0.0])
    n /= np.linalg.norm(n)
    return n if n @ np.array([-centre[0], -centre[1], 0.0]) > 0 else -n


def pointing(n, pos):
    x = -np.asarray(n, float)
    y = np.array([0.0, 0.0, 1.0])
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, np.cross(x, y), pos
    return T


def below(table, points):
    """None when the target is under the counter top (the planner would keep the hand above it), else the table."""
    return None if float(np.percentile(points[:, 2], 90)) < table.height + 0.02 else table


def solve(robot):
    instruction = robot.status()["instruction"].lower()
    view = robot.look()
    table = robot.table(view)
    labels = sorted({v for v in view.segments.values() if v.endswith("button")}, key=len, reverse=True)
    which = next((lab for lab in labels if lab in instruction), labels[0] if labels else None)
    if which is None:
        print("no buttons in view:", sorted(set(view.segments.values())))
        return
    button = points_of(robot, view, table, which)
    if button is None:
        print("cannot see the", which)
        return
    b = button.mean(axis=0)
    table = below(table, button)
    appliance = [v for v in set(view.segments.values()) if v in instruction and not v.endswith("button")]
    around = points_of(robot, view, table, appliance[0]) if appliance else None
    if around is not None:
        around = around[np.linalg.norm(around - b, axis=1) < 0.12]
    n = front_normal(around if around is not None and len(around) > 20 else button, b)
    surface = float(np.percentile((button - b) @ n, 90))
    reach = float(robot.hand_points((0.0,))[:, 0].max())
    print(which, "at", np.round(b, 3), "facing", np.round(n, 2))
    target = b + n * (surface - 0.015 + reach)
    goal = robot.Goal("press", ARM, tcp=pointing(n, target), pre=pointing(n, target + n * 0.10), hand_before=(0.0,),
                      hand=(0.0,))
    plan = robot.plan(goal, table, view.points)
    if plan is None:
        print("no press plan:", robot.why())
        return
    if robot.run(plan).get("ended"):
        return
    back = robot.status()["tcp"][ARM].copy()
    back[:3, 3] += n * 0.10
    p = robot.plan(robot.Goal("retreat", ARM, tcp=back), table)
    if p is not None:
        robot.run(p)
