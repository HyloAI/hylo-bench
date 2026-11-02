"""robosuite TwoArmLift: "lift the pot with both arms, holding it by its two handles". Perception only, two arms.

Each handle is a rectangular loop sticking out of one side of the pot. Each arm takes the handle on its side (nearer
its base): a top-down grasp on the loop's outer bar, the fingers closing along the line from the pot's centre through
the handle (across the bar). Both grasps run at once; then both TCPs rise along straight lines re-timed together so
the pot stays level."""
import numpy as np

ARMS = ("robot0", "robot1")


def topdown(closing, pos):
    y = np.array([closing[0], closing[1], 0.0])
    y /= np.linalg.norm(y)
    x = np.array([0.0, 0.0, -1.0])
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, np.cross(x, y), pos
    return T


def solve(robot):
    view = robot.look()
    table = robot.table(view)
    pot = [d for d in robot.objects(view, robot.find(view, "pot"), table) if len(d.points) >= 30]
    handles = [d for d in robot.objects(view, robot.find(view, "pot handle"), table) if len(d.points) >= 10][:2]
    if not pot or len(handles) < 2:
        print("cannot see the pot and both handles:", len(pot), len(handles))
        return
    centre = robot.box(max(pot, key=lambda d: len(d.points)).points, table).center
    bases = {a: np.asarray(robot.spec.arm(a).base_in_world)[:3, 3] for a in ARMS}
    # each handle to the arm whose base is nearer
    h0, h1 = handles

    def far(h, arm):
        return np.linalg.norm(h.points.mean(0)[:2] - bases[arm][:2])

    straight = far(h0, "robot0") + far(h1, "robot1")
    crossed = far(h1, "robot0") + far(h0, "robot1")
    assign = {"robot0": h0, "robot1": h1} if straight <= crossed else {"robot0": h1, "robot1": h0}
    plans = {}
    for arm, h in assign.items():
        pts = h.points
        radial = pts.mean(0)[:2] - centre[:2]
        radial /= np.linalg.norm(radial)
        reach = (pts[:, :2] - centre[:2]) @ radial
        bar = pts[reach > np.percentile(reach, 75)]             # the loop's outer bar
        top = float(np.percentile(pts[:, 2], 95))
        pos = np.array([bar[:, 0].mean(), bar[:, 1].mean(), top - 0.012])
        goals = [robot.grasp_at(topdown(radial * s, pos), arm=arm) for s in (1.0, -1.0)]
        for k, g in enumerate(goals):
            g.rank = k
        p = robot.plan(goals, table, view.points)
        if p is None:
            print(arm, "no grasp plan:", robot.why())
            return
        plans[arm] = p
    out = robot.run(plans)
    if out.get("error"):
        print("together refused:", out["error"], "- one arm after the other")
        for arm in ARMS:
            p = robot.plan(plans[arm].goal, table, view.points)
            if p is None or robot.run(p).get("ended"):
                return
    print("holding:", {a: robot.holding(arm=a) for a in ARMS})
    if robot.status().get("ended"):
        return
    for rise in (0.18, 0.05):
        st = robot.status()
        ups = {}
        for arm in ARMS:
            t = st["tcp"][arm].copy()
            t[2, 3] += rise
            ups[arm] = robot.plan(robot.Goal("lift", arm, tcp=t, hand=(0.0,)), table)
            if ups[arm] is None:
                print(arm, "no lift plan:", robot.why())
                return
        out = robot.run(ups, sync=True)
        print("lifted", rise, out.get("error"))
        if out.get("ended"):
            return
