"""The `mujoco/manip-v0` task family on `mujoco.panda_table` (RFC-0006 part 3).

Each task samples a layout from the episode seed (positions, yaw, colours, distractors), realises a natural-language
instruction and evaluates its success predicate after every control step. Predicates need the objects at rest and
released (no contact with the fingers), so a block still in the hand above the plate is not a success.

| Task | Success |
|---|---|
| pick_place      | the named object rests on the named plate |
| stack           | two or three blocks stacked in the named order, the bottom one on the table |
| put_in_bowl     | the named object inside the named bowl |
| push_to_zone    | the large box (too wide to grasp) inside the named zone |
| press_by_number | buttons pressed in the order the instruction gives (a wrong press fails at once) |
| sort_by_colour  | every block in the tray of its colour |

Episodes end early with failure when an object falls off the table or a button is pressed out of order.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from hylo.core.task import TaskSpec

from .panda_table import (BASE_POS, BIG_HALF, BLOCK_HALF, BOWL_H, BOWL_R0, BOWL_R1, BTN_CAP_H, BTN_HOUSE, BTN_HOUSE_H, COLOURS,
                          CYL_HALF, CYL_R, FIXTURE_COLOURS, PLATE_HALF, PLATE_R, TABLE_TOP, TRAY_BASE, TRAY_H, TRAY_IN,
                          TRAY_WALL, WORKSPACE, ZONE_HALF)

if TYPE_CHECKING:
    from .panda_table import PandaTableEnv

TASK_KINDS = ("pick_place", "stack", "put_in_bowl", "push_to_zone", "press_by_number", "sort_by_colour")
MAX_STEPS = {"pick_place": 400, "stack": 700, "put_in_bowl": 400, "push_to_zone": 600, "press_by_number": 500,
             "sort_by_colour": 1100}
TEMPLATES = {"pick_place": "put the <object> on the <plate>", "stack": "stack the <blocks> in the named order",
             "put_in_bowl": "put the <object> in the <bowl>", "push_to_zone": "push the large box into the <zone>",
             "press_by_number": "press the numbered buttons in the given order",
             "sort_by_colour": "put each block in the tray of its colour"}
EXTENT = ((-0.27, 0.21), (-0.34, 0.34))      # footprints stay inside this part of the table (x, y)
SMALL_GAP = 0.055                             # room for open fingers beside a graspable object
LIN_REST, ANG_REST = 0.03, 0.8                # m/s, rad/s


# -- layout sampling --------------------------------------------------------------------------------------------
class _Layout:
    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.boxes: list[tuple[float, float, float, float]] = []      # centre and axis-aligned half extents

    def put(self, hx: float, hy: float, gap: float, tries: int = 300) -> tuple[float, float] | None:
        (x0, x1), (y0, y1) = WORKSPACE
        (ex0, ex1), (ey0, ey1) = EXTENT
        lo_x, hi_x = max(x0, ex0 + hx), min(x1, ex1 - hx)
        lo_y, hi_y = max(y0, ey0 + hy), min(y1, ey1 - hy)
        for _ in range(tries):
            x, y = float(self.rng.uniform(lo_x, hi_x)), float(self.rng.uniform(lo_y, hi_y))
            if all(abs(x - bx) >= hx + bhx + gap or abs(y - by) >= hy + bhy + gap for bx, by, bhx, bhy in self.boxes):
                self.boxes.append((x, y, hx, hy))
                return x, y
        return None


def _rot_extent(hx: float, hy: float, yaw: float) -> tuple[float, float]:
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    return hx * c + hy * s, hx * s + hy * c


class _Builder:
    """Fills ep["items"] from the pool, one kind at a time."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng, self.layout = rng, _Layout(rng)
        self.items: dict[str, dict[str, Any]] = {}
        self.used = {"block": 0, "cyl": 0, "plate": 0, "bowl": 0, "tray": 0, "bigbox": 0, "zone": 0, "button": 0}

    def add(self, kind: str, colour: str, label: str, *, gap: float = SMALL_GAP, yaw: float | None = None,
            **extra: Any) -> str | None:
        half = {"block": (BLOCK_HALF,) * 3, "cyl": (CYL_R, CYL_R, CYL_HALF), "plate": (PLATE_R, PLATE_R, PLATE_HALF),
                "bowl": (BOWL_R1 + 0.004, BOWL_R1 + 0.004, BOWL_H / 2), "tray": (TRAY_IN + 2 * TRAY_WALL,) * 2 + (TRAY_H / 2,),
                "bigbox": BIG_HALF, "zone": (ZONE_HALF, ZONE_HALF, 0.0006),
                "button": (BTN_HOUSE, BTN_HOUSE, BTN_HOUSE_H + BTN_CAP_H)}[kind]
        if yaw is None:
            yaw = float(self.rng.uniform(-math.pi / 4, math.pi / 4)) if kind in ("block", "bigbox", "zone") else 0.0
        hx, hy = _rot_extent(half[0], half[1], yaw)
        xy = self.layout.put(hx, hy, gap)
        if xy is None:
            return None
        name = f"{kind}{self.used[kind]}"
        self.used[kind] += 1
        z = TABLE_TOP + (half[2] + 0.0005 if kind in ("block", "cyl", "bigbox") else 0.0)
        top = {"block": 2 * BLOCK_HALF, "cyl": 2 * CYL_HALF, "bigbox": 2 * BIG_HALF[2], "plate": 2 * PLATE_HALF,
               "bowl": BOWL_H, "tray": TRAY_BASE, "zone": 0.0012, "button": 2 * (BTN_HOUSE_H + BTN_CAP_H)}[kind]
        self.items[name] = {"pos": [xy[0], xy[1], z], "yaw": yaw, "colour": colour, "label": label,
                            "half_size": list(half), "top": TABLE_TOP + top, **extra}
        return name


def _pick(rng: np.random.Generator, options: list[str], n: int) -> list[str]:
    return [options[i] for i in rng.choice(len(options), size=n, replace=False)]


def _choice(rng: np.random.Generator, options: list[str]) -> str:
    return options[int(rng.integers(len(options)))]


def _graspables(b: _Builder, colours: list[str], shapes: list[str]) -> list[str] | None:
    names = []
    for c, s in zip(colours, shapes):
        n = b.add(s, c, f"{c} {'block' if s == 'block' else 'cylinder'}")
        if n is None:
            return None
        names.append(n)
    return names


def _fixtures(b: _Builder, kind: str, n: int, gap: float = 0.03) -> list[str] | None:
    colours = _pick(b.rng, sorted(FIXTURE_COLOURS), n)
    names = []
    for c in colours:
        name = b.add(kind, c, kind if n == 1 else f"{c} {kind}", gap=gap)
        if name is None:
            return None
        names.append(name)
    return names


def _pick_place(rng: np.random.Generator) -> dict | None:
    b = _Builder(rng)
    plates = _fixtures(b, "plate", int(rng.integers(1, 3)))
    n = int(rng.integers(2, 5))
    objs = plates and _graspables(b, _pick(rng, sorted(COLOURS), n), [_choice(rng, ["block", "block", "cyl"]) for _ in range(n)])
    if not objs:
        return None
    obj, plate = objs[0], plates[int(rng.integers(len(plates)))]
    o, p = b.items[obj]["label"], b.items[plate]["label"]
    text = _choice(rng, [f"put the {o} on the {p}", f"place the {o} on the {p}", f"pick up the {o} and put it on the {p}"])
    return {"items": b.items, "goal": {"object": obj, "target": plate}, "instruction": text}


def _stack(rng: np.random.Generator) -> dict | None:
    b = _Builder(rng)
    n, extra = int(rng.integers(2, 4)), int(rng.integers(0, 3))
    blocks = _graspables(b, _pick(rng, sorted(COLOURS), n + extra), ["block"] * (n + extra))
    if not blocks:
        return None
    order = blocks[:n]                                   # bottom first
    lab = [b.items[k]["label"] for k in order]
    if n == 2:
        text = _choice(rng, [f"stack the {lab[1]} on the {lab[0]}", f"put the {lab[1]} on top of the {lab[0]}",
                             f"place the {lab[1]} on the {lab[0]}"])
    else:
        text = _choice(rng, [f"stack the blocks with the {lab[0]} at the bottom, the {lab[1]} in the middle and the {lab[2]} on top",
                             f"build a tower: the {lab[0]} at the bottom, then the {lab[1]}, then the {lab[2]} on top",
                             f"put the {lab[1]} on the {lab[0]}, then the {lab[2]} on the {lab[1]}"])
    return {"items": b.items, "goal": {"order": order}, "instruction": text}


def _put_in_bowl(rng: np.random.Generator) -> dict | None:
    b = _Builder(rng)
    bowls = _fixtures(b, "bowl", int(rng.integers(1, 3)))
    n = int(rng.integers(1, 4))
    objs = bowls and _graspables(b, _pick(rng, sorted(COLOURS), n), [_choice(rng, ["block", "cyl"]) for _ in range(n)])
    if not objs:
        return None
    obj, bowl = objs[0], bowls[int(rng.integers(len(bowls)))]
    o, w = b.items[obj]["label"], b.items[bowl]["label"]
    text = _choice(rng, [f"put the {o} in the {w}", f"drop the {o} into the {w}", f"place the {o} inside the {w}"])
    return {"items": b.items, "goal": {"object": obj, "bowl": bowl}, "instruction": text}


def _push_to_zone(rng: np.random.Generator) -> dict | None:
    b = _Builder(rng)
    box_c, *zone_c = _pick(rng, sorted(COLOURS), 3)
    zones = [b.add("zone", c, f"{c} zone", gap=0.02) for c in zone_c]
    box = b.add("bigbox", box_c, f"large {box_c} box", gap=0.04)
    if None in zones or box is None:
        return None
    target = zones[int(rng.integers(2))]
    d = np.subtract(b.items[target]["pos"][:2], b.items[box]["pos"][:2])
    dist = float(np.linalg.norm(d))
    if not 0.2 <= dist <= 0.34:
        return None
    behind = np.asarray(b.items[box]["pos"][:2]) - d / dist * (BIG_HALF[0] * 1.45 + 0.06)
    (ex0, ex1), (ey0, ey1) = EXTENT
    base_dist = math.hypot(behind[0] - BASE_POS[0], behind[1] - BASE_POS[1])
    if not (ex0 <= behind[0] <= ex1 and ey0 <= behind[1] <= ey1) or base_dist < 0.33:
        return None                                      # no room for the hand behind the box
    zl, bl = b.items[target]["label"], b.items[box]["label"]
    text = _choice(rng, [f"push the {bl} into the {zl}", f"slide the large box onto the {zl}",
                         f"push the {bl} until it is inside the {zl}"])
    return {"items": b.items, "goal": {"object": box, "zone": target}, "instruction": text}


def _press_by_number(rng: np.random.Generator) -> dict | None:
    b = _Builder(rng)
    n = int(rng.integers(3, 5))
    digits = [int(x) for x in rng.choice(np.arange(1, 10), size=n, replace=False)]
    buttons = []
    for dgt in digits:
        name = b.add("button", "", f"button {dgt}", gap=0.04, digit=dgt)
        if name is None:
            return None
        buttons.append(name)
    by_digit = dict(zip(digits, buttons))
    mode = _choice(rng, ["explicit", "explicit", "explicit", "ascending", "descending"])
    if mode == "explicit":
        seq = [int(x) for x in rng.choice(digits, size=int(rng.integers(2, 4)), replace=False)]
        s = [str(x) for x in seq]
        listed = ", ".join(s[:-1]) + f" and {s[-1]}"
        text = _choice(rng, [f"press the buttons in this order: {', '.join(s)}",
                             "press " + ", then ".join(f"button {x}" for x in s),
                             f"push the buttons numbered {listed}, in that order"])
    else:
        seq = sorted(digits, reverse=mode == "descending")
        low, high = ("lowest", "highest") if mode == "ascending" else ("highest", "lowest")
        text = _choice(rng, [f"press all the buttons in {mode} order of their numbers",
                             f"press every button, from the {low} number to the {high}"])
    return {"items": b.items, "goal": {"sequence": [by_digit[x] for x in seq], "digits": seq, "mode": mode},
            "instruction": text}


def _sort_by_colour(rng: np.random.Generator) -> dict | None:
    b = _Builder(rng)
    n_col = int(rng.integers(2, 4))
    colours = _pick(rng, sorted(COLOURS), n_col)
    options = [[2, 1], [1, 2], [2, 2]] if n_col == 2 else [[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]]
    counts = options[int(rng.integers(len(options)))]          # three or four blocks, at most two per tray
    trays = {}
    for c in colours:
        t = b.add("tray", c, f"{c} tray", gap=0.03, yaw=0.0)
        if t is None:
            return None
        trays[c] = t
    assign = {}
    for c, k in zip(colours, counts):
        for _ in range(k):
            blk = b.add("block", c, f"{c} block")
            if blk is None:
                return None
            assign[blk] = trays[c]
    text = _choice(rng, ["put each block in the tray of the same colour", "sort the blocks by colour into the matching trays",
                         "place every block into the tray that matches its colour"])
    return {"items": b.items, "goal": {"assign": assign}, "instruction": text}


SAMPLERS = {"pick_place": _pick_place, "stack": _stack, "put_in_bowl": _put_in_bowl, "push_to_zone": _push_to_zone,
            "press_by_number": _press_by_number, "sort_by_colour": _sort_by_colour}


def sample_episode(params: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    kind = params.get("task")
    if kind not in SAMPLERS:
        raise ValueError(f"task params need 'task' in {list(SAMPLERS)}, got {kind!r}")
    for _ in range(500):
        ep = SAMPLERS[kind](rng)
        if ep is not None:
            ep["task"] = kind
            return ep
    raise RuntimeError(f"could not sample a {kind} layout")


# -- success predicates -----------------------------------------------------------------------------------------
def _local_xy(env: "PandaTableEnv", name: str, ref: str) -> np.ndarray:
    p, _ = env.pose(name)
    c, q = env.pose(ref)
    yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    dx, dy = p[0] - c[0], p[1] - c[1]
    return np.array([math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy])


def _settled(env: "PandaTableEnv", name: str) -> bool:
    lin, ang = env.speed(name)
    return lin < LIN_REST and ang < ANG_REST and not env.touching_hand(name)


def _on_surface(env: "PandaTableEnv", name: str, top: float) -> bool:
    half = env.ep["items"][name]["half_size"]
    z = env.pose(name)[0][2]
    return top + min(half) - 0.008 <= z <= top + max(half) + 0.008


def evaluate(env: "PandaTableEnv") -> tuple[bool, str | None, dict[str, Any]]:
    """(success, failure reason or None, info) after a control step."""
    ep = env.ep
    items, goal, kind = ep["items"], ep["goal"], ep["task"]
    for name in items:
        if name.startswith(("block", "cyl", "bigbox")) and env.pose(name)[0][2] < TABLE_TOP - 0.05:
            return False, f"the {items[name]['label']} fell off the table", {"done": 0}
    if kind == "pick_place":
        obj, tgt = goal["object"], goal["target"]
        near = np.linalg.norm(env.pose(obj)[0][:2] - env.pose(tgt)[0][:2]) < PLATE_R - 0.012
        ok = near and _on_surface(env, obj, items[tgt]["top"]) and _settled(env, obj)
        return ok, None, {"done": int(ok), "of": 1}
    if kind == "stack":
        order = goal["order"]
        done = int(_on_surface(env, order[0], TABLE_TOP) and _settled(env, order[0]))
        for lower, upper in zip(order, order[1:]):
            pl, pu = env.pose(lower)[0], env.pose(upper)[0]
            on = np.linalg.norm(pu[:2] - pl[:2]) < 1.2 * BLOCK_HALF and abs(pu[2] - pl[2] - 2 * BLOCK_HALF) < 0.008
            if not (done and on and _settled(env, upper)):
                break
            done += 1
        return done == len(order), None, {"done": done, "of": len(order)}
    if kind == "put_in_bowl":
        obj, bowl = goal["object"], goal["bowl"]
        p, c = env.pose(obj)[0], env.pose(bowl)[0]
        ok = np.linalg.norm(p[:2] - c[:2]) < BOWL_R0 - 0.01 and p[2] < TABLE_TOP + BOWL_H and _settled(env, obj)
        return bool(ok), None, {"done": int(ok), "of": 1}
    if kind == "push_to_zone":
        obj, zone = goal["object"], goal["zone"]
        local = _local_xy(env, obj, zone)
        inside = bool(np.all(np.abs(local) < ZONE_HALF - 0.025))
        ok = inside and _settled(env, obj)
        return ok, None, {"done": int(ok), "of": 1, "distance": float(np.linalg.norm(local))}
    if kind == "press_by_number":
        seq, pressed = goal["sequence"], ep.get("pressed", [])
        for i, name in enumerate(pressed):
            if i >= len(seq) or name != seq[i]:
                return False, f"pressed {items[name]['label']} out of order", {"done": i, "of": len(seq)}
        return len(pressed) == len(seq), None, {"done": len(pressed), "of": len(seq)}
    if kind == "sort_by_colour":
        done = 0
        for obj, tray in goal["assign"].items():
            local = _local_xy(env, obj, tray)
            z = env.pose(obj)[0][2]
            if np.all(np.abs(local) < TRAY_IN - 0.01) and z < TABLE_TOP + TRAY_H + BLOCK_HALF and _settled(env, obj):
                done += 1
        return done == len(goal["assign"]), None, {"done": done, "of": len(goal["assign"])}
    raise ValueError(f"unknown task {kind!r}")


# -- tasks and benchmark ----------------------------------------------------------------------------------------
TAGS = {"pick_place": ("grasp", "place"), "stack": ("grasp", "place", "precision"), "put_in_bowl": ("grasp", "place"),
        "push_to_zone": ("push", "non-prehensile"), "press_by_number": ("press", "reading"),
        "sort_by_colour": ("grasp", "place", "long-horizon")}


def manip_tasks(kinds: list[str] | tuple[str, ...] | None = None) -> tuple[TaskSpec, ...]:
    kinds = tuple(kinds or TASK_KINDS)
    unknown = [k for k in kinds if k not in TASK_KINDS]
    if unknown:
        raise ValueError(f"unknown manipulation tasks {unknown}; known: {list(TASK_KINDS)}")
    return tuple(TaskSpec(id=f"mujoco/manip_{k}", version="1", instruction=TEMPLATES[k], params={"task": k},
                          max_steps=MAX_STEPS[k], tags=("manipulation", *TAGS[k])) for k in kinds)


VALIDATION_SEED_OFFSET = 10_000


def manip_benchmark(tasks: list[str] | None = None, protocol: str = "rsi", rsi_states: int = 50, validation: int = 20,
                    train: int = 10, heldout: int = 20, gate_looks: int = 1):
    """protocol "rsi" (default): RFC-0002 search / search_val / gate / heldout over `rsi_states` seeds (40/20/20/20
    percent) plus a separate `validation` split for plain evaluation; protocol "standard": train / validation /
    heldout. All seed ranges are disjoint."""
    from hylo.core.benchmark import Benchmark

    from .planar_reach import splits_for

    splits = splits_for(protocol, train, validation, heldout, rsi_states, gate_looks)
    if protocol == "rsi":
        splits["validation"] = tuple(range(VALIDATION_SEED_OFFSET, VALIDATION_SEED_OFFSET + validation))
    return Benchmark(id="mujoco/manip-v0", version="0.1", tasks=manip_tasks(tasks), splits=splits,
                     metadata={"env": "mujoco.panda_table", "protocol": protocol, "tasks": ",".join(tasks or TASK_KINDS)})


__all__ = ["MAX_STEPS", "TASK_KINDS", "evaluate", "manip_benchmark", "manip_tasks", "sample_episode"]
