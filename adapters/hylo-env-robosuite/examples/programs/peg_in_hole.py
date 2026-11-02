"""robosuite TwoArmPegInHole: "insert the peg into the hole in the plate". Perception only, two arms, no grippers:
the peg is fixed to robot0's hand, the plate (a square with a hole in its middle) to robot1's.

The program measures both from the cameras: the plate's centre and normal (its points' least principal axis), the
peg's axis (its points' first principal axis) and tip (the end away from robot0's hand), both relative to robot0's
TCP. It then moves robot0 so the peg's axis lines up with the plate's normal through its centre, the tip 6 cm short
of the plate, and pushes straight along the axis until the tip is 3 cm past it. robot1 holds still."""
import numpy as np


def points_of(robot, view, label, min_points=10):
    dets = [d for d in robot.objects(view, robot.find(view, label), robot.table(view)) if len(d.points) >= min_points]
    return np.concatenate([d.points for d in dets]) if dets else None


def principal(points):
    c = points.mean(axis=0)
    vt = np.linalg.svd(points - c, full_matrices=False)[2]
    return c, vt


def align(a, b):
    """Rotation turning unit vector a onto unit vector b."""
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


def solve(robot):
    view = robot.look()
    peg = points_of(robot, view, "peg")
    plate = points_of(robot, view, "plate")
    if peg is None or plate is None:
        print("cannot see the", "peg" if peg is None else "plate")
        return
    st = robot.status()
    T0 = st["tcp"]["robot0"]
    pc, pv = principal(plate)
    n = pv[2]
    if (T0[:3, 3] - pc) @ n > 0:
        n = -n                                     # n: through the plate, away from robot0
    kc, kv = principal(peg)
    a = kv[0]
    ends = (peg - kc) @ a
    lo, hi = peg[np.argmin(ends)], peg[np.argmax(ends)]
    tip = hi if np.linalg.norm(hi - T0[:3, 3]) > np.linalg.norm(lo - T0[:3, 3]) else lo    # away from the hand
    if (tip - kc) @ a < 0:
        a = -a                                     # a: along the peg towards its tip
    tip_in_tcp = np.linalg.inv(T0) @ np.r_[tip, 1.0]
    R = align(a, n) @ T0[:3, :3]
    print("plate centre", np.round(pc, 3), "normal", np.round(n, 2), "peg tip", np.round(tip, 3))

    def tcp_for(tip_at):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tip_at - R @ tip_in_tcp[:3]
        return T

    goal = robot.Goal("reach", "robot0", tcp=tcp_for(pc + 0.03 * n), pre=tcp_for(pc - 0.06 * n))
    plan = robot.plan(goal, None, view.points)
    if plan is None:
        print("no plan:", robot.why())
        return
    robot.run(plan)
