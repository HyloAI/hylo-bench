"""Building a free scene's model: the robot environments' table, cameras and robot around a scene program's things.

    model, scene_bodies, spec = build_free_model(program, width, height, mesh_detail, demo_hands)

The table, lights, cameras and robot come from the existing environments, unchanged: `mujoco.panda_table`'s
`scene_xml(items="")` and `attach_panda`, or `mujoco.aloha_table` / `mujoco.g1_table`'s own `build_model(..., spec=)`,
given the spec the program built on (their scene_xml plus the program's things), which attaches its robot exactly as
it does for its own tasks; their parked object pools stay in the model, out of the way. `robot_facts(robot)` gives
the table and each arm's working zone; `SceneHandle` is `scene` in build(scene); `export_mjcf` writes the compiled
world back out as MJCF (for an offline renderer).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .free_program import SceneProgram, SceneProgramError, GuardedModule

DEMO_HAND = "hylo_demo_hand"


def robot_facts(robot: str) -> dict[str, Any]:
    """Where things are for each robot (from the robot environments' own constants)."""
    if robot == "panda":
        from . import panda_table as pt
        from .panda_scene import REACH

        return {"table_top": pt.TABLE_TOP, "table_x": pt.TABLE_X, "table_y": pt.TABLE_Y,
                "arms": {"arm": {"base": list(pt.BASE_POS), "x": REACH[0], "y": REACH[1]}},
                "opening": 2 * pt.FINGER_TRAVEL}
    if robot == "aloha":
        from . import aloha_table as at

        hx, hy = at.TABLE_HALF
        y = at.WORKSPACE[1]
        return {"table_top": at.TABLE_TOP, "table_x": (-hx, hx), "table_y": (-hy, hy),
                "arms": {"left": {"base": list(at.BASES["left"]), "x": (-0.45, at.REACH_X), "y": y},
                         "right": {"base": list(at.BASES["right"]), "x": (-at.REACH_X, 0.45), "y": y}},
                "opening": at.MAX_WIDTH}
    from . import g1_table as gt

    x = gt.WORKSPACE[0]
    return {"table_top": gt.TABLE_TOP, "table_x": gt.TABLE_X, "table_y": gt.TABLE_Y,
            "arms": {"left": {"base": [0.0, 0.0, 0.0], "x": (x[0], x[1] + 0.03), "y": (-0.1, 0.25)},
                     "right": {"base": [0.0, 0.0, 0.0], "x": (x[0], x[1] + 0.03), "y": (-0.25, 0.1)}},
            "opening": 0.08}


class SceneHandle:
    """`scene` in build(scene): the spec to add to (table, lights and cameras are there; the robot is attached
    after build returns) and the facts a builder needs about the robot's table."""

    def __init__(self, spec: Any, robot: str) -> None:
        import mujoco

        self.spec, self.robot, self.mujoco = spec, robot, GuardedModule(mujoco, "mujoco")
        f = robot_facts(robot)
        self.table_top = float(f["table_top"])
        self.table = {"x": tuple(f["table_x"]), "y": tuple(f["table_y"]), "top": self.table_top}
        self.arms = {k: {"base": tuple(v["base"]), "x": tuple(v["x"]), "y": tuple(v["y"])} for k, v in f["arms"].items()}
        self.opening = float(f["opening"])

    def add_mjcf(self, xml: str, pos: Sequence[float] | None = None) -> Any:
        """Add an MJCF document (<mujoco> with worldbody, asset, default, tendon, equality, contact, extension
        ... sections) to the scene, its world frame at `pos` (default: the world origin). Names stay as written."""
        import warnings

        import mujoco

        try:
            child = mujoco.MjSpec.from_string(str(xml))
        except Exception as exc:  # noqa: BLE001
            raise SceneProgramError(f"add_mjcf: the MJCF does not parse: {str(exc)[:500]}") from exc
        frame = self.spec.worldbody.add_frame(pos=[float(v) for v in (pos or (0.0, 0.0, 0.0))])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.spec.attach(child, frame=frame, prefix="")
        return frame

    def rgba(self, colour: str) -> list[float]:
        from hylo.core.scene import COLOURS

        return [*COLOURS.get(colour.lower().strip(), COLOURS["grey"]), 1.0]


def add_demo_hands(spec: Any, n: int) -> None:
    """Invisible mocap bodies, each with an inactive ball-joint equality to the world; demo() re-targets them."""
    import mujoco

    for i in range(n):
        spec.worldbody.add_body(name=f"{DEMO_HAND}{i}", mocap=True, pos=[0.0, 0.0, -1.0])
        eq = spec.add_equality(name=f"{DEMO_HAND}{i}/eq", type=mujoco.mjtEq.mjEQ_CONNECT, name1=f"{DEMO_HAND}{i}",
                               name2="world", objtype=mujoco.mjtObj.mjOBJ_BODY)
        eq.active = False
        eq.solref = [0.02, 1.0]


def build_free_model(program: SceneProgram, width: int = 320, height: int = 240, mesh_detail: str = "fast",
                     demo_hands: int = 0) -> tuple[Any, set[str], Any]:
    """Compile the program's scene with its robot. Returns (model, names of the program's bodies, the MjSpec it
    was compiled from: robot and scene, for exporting MJCF)."""
    import warnings

    import mujoco

    robot = program.robot
    if robot == "panda":
        from .panda_table import attach_panda, scene_xml

        base_xml = scene_xml(width, height, items="")
    else:
        mod = _robot_module(robot)
        base_xml = mod.scene_xml(width, height)
    spec = mujoco.MjSpec.from_string(base_xml)
    before = {b.name for b in spec.bodies}
    n_eq = len(spec.equalities)
    program.call(program.build, SceneHandle(spec, robot))
    if len(spec.actuators):
        raise SceneProgramError("the scene must not add actuators: the robot is the only actor (use joints with "
                                "springs, damping or equality constraints for passive mechanisms)")
    if len(spec.keys):
        raise SceneProgramError("the scene must not add keyframes (the robot's state layout is added later); set "
                                "start positions in build (pos=, joint ref / qpos0) or in randomize(sim, rng)")
    mine = {b.name for b in spec.bodies} - before
    unnamed = [b for b in spec.bodies if not b.name]
    if unnamed:
        raise SceneProgramError(f"{len(unnamed)} body(ies) without a name: give every body a unique name")
    if len(spec.equalities) < n_eq:
        raise SceneProgramError("the scene removed equalities it did not add")
    if demo_hands:
        add_demo_hands(spec, demo_hands)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if robot == "panda":
                attach_panda(spec, mesh_detail)
                model = spec.compile()
            else:
                model = _robot_module(robot).build_model(width, height, mesh_detail, spec=spec)
    except SceneProgramError:
        raise
    except ValueError as exc:                               # MuJoCo compiler errors
        raise SceneProgramError(f"the model does not compile: {str(exc)[:700]}") from exc
    return model, mine, spec


def flex_name(m: Any, f: int) -> str:
    """The name of flex `f` (or flex<f>)."""
    import mujoco

    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_FLEX, f) or f"flex{f}"


def _robot_module(robot: str) -> Any:
    """The arm_table environment module of a two-arm robot."""
    from . import aloha_table, g1_table

    return {"aloha": aloha_table, "g1": g1_table}[robot]


def export_mjcf(spec: Any, robot: str) -> str:
    """MJCF of a compiled free scene (robot and scene) that loads on its own: unprefixed attaches leave unnamed
    default classes behind (and, for a robot attached twice, repeated ones), which are dropped or named; the robot's
    mesh and texture files get absolute paths."""
    import warnings
    import xml.etree.ElementTree as ET

    from .assets import ALOHA, G1_HANDS, PANDA, assets_dir

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        root = ET.fromstring(spec.to_xml())
    top = root.find("default")
    if top is not None:
        seen: set[str] = set()
        for k, child in enumerate(list(top)):
            if child.tag != "default":
                continue
            names = [d.get("class") for d in child.iter("default") if d.get("class")]
            if not child.get("class") and (not names or all(n in seen for n in names)):
                top.remove(child)
                continue
            if not child.get("class"):
                child.set("class", f"attached{k}")
            seen.update(names)
    folder = assets_dir({"panda": PANDA, "aloha": ALOHA, "g1": G1_HANDS}[robot])
    for el in root.iter():
        f = el.get("file")
        if f and not Path(f).is_absolute():
            for cand in (folder / "assets" / f, folder / f):
                if cand.is_file():
                    el.set("file", str(cand))
                    break
    return ET.tostring(root, encoding="unicode")


__all__ = ["DEMO_HAND", "SceneHandle", "add_demo_hands", "build_free_model", "export_mjcf", "flex_name", "robot_facts"]
