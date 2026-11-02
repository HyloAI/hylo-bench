"""The `mujoco/bimanual-v0` task family on `mujoco.aloha_table`: tasks that one arm cannot do.

Each task samples a layout from the episode seed (positions, yaw, sizes, masses, colours, which side starts) and
realises a natural-language instruction. Each arm reaches only up to about 10 cm past the middle of the table, which
the layouts use to make a second arm necessary.

| Task | Success | Why one arm is not enough |
|---|---|---|
| handover  | the bar rests in the named zone, released | the bar starts where only one arm reaches and the zone lies where only the other reaches: one arm must pass it to the other |
| lift_box  | the box held at least the named height off the table and level (tilt under 0.2 rad) for one second | the box is wider than a gripper opens; its two end handles are 22 cm apart, and lifted by one handle it tips |
| two_bins  | each block in the bin on its side, both released, within the step limit | each block and its bin lie on one arm's side only, and the step limit is shorter than two pick-and-place motions one after the other |

Failures end the episode early: an object falls off the table, or the box is dropped after it was lifted.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from hylo.core.task import TaskSpec

from .aloha_table import (BAR_HALF, BAR_LEN_RANGE, BIN_BASE, BIN_H, BIN_IN, BIN_WALL, BLOCK_RANGE, LIFT_BODY, LIFT_FIN,
                          LIFT_FIN_Z, LIFT_MASS, TABLE_TOP, WORKSPACE, ZONE_HALF)
from .arm_table import COLOURS, FIXTURE_COLOURS

if TYPE_CHECKING:
    from .aloha_table import AlohaTableEnv

TASK_KINDS = ("handover", "lift_box", "two_bins")
# two_bins: the oracle needs about 190 steps with both arms at once and about 330 with one after the other
MAX_STEPS = {"handover": 900, "lift_box": 400, "two_bins": 260}
TEMPLATES = {"handover": "hand the <bar> over to the other arm and put it in the <zone>",
             "lift_box": "lift the <box> with both arms to the named height and hold it level",
             "two_bins": "put each block in the named bin, both arms at once"}
TAGS = {"handover": ("bimanual", "handover", "grasp", "place"), "lift_box": ("bimanual", "coordination", "lift"),
        "two_bins": ("bimanual", "parallel", "grasp", "place", "time-limit")}
LIN_REST, ANG_REST = 0.03, 0.8
HOLD_STEPS = 20                                    # lift_box: steps the box must stay up and level
TILT_MAX = 0.2
SIDE_X = (0.22, 0.33)                              # |x| of objects that only one arm reaches
BASE_CLEAR = 0.355                                 # |x| object footprints stay under (the arm bases start at 0.367)


class _Layout:
    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.boxes: list[tuple[float, float, float, float]] = []

    def put(self, xr: tuple[float, float], yr: tuple[float, float], hx: float, hy: float, gap: float,
            tries: int = 300) -> tuple[float, float] | None:
        (wx0, wx1), (wy0, wy1) = WORKSPACE
        lo_x, hi_x = max(xr[0], wx0 + hx), min(xr[1], wx1 - hx)
        lo_y, hi_y = max(yr[0], wy0 + hy), min(yr[1], wy1 - hy)
        if lo_x > hi_x or lo_y > hi_y:
            return None
        for _ in range(tries):
            x, y = float(self.rng.uniform(lo_x, hi_x)), float(self.rng.uniform(lo_y, hi_y))
            if all(abs(x - bx) >= hx + bhx + gap or abs(y - by) >= hy + bhy + gap for bx, by, bhx, bhy in self.boxes):
                self.boxes.append((x, y, hx, hy))
                return x, y
        return None


def _extent(hx: float, hy: float, yaw: float) -> tuple[float, float]:
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    return hx * c + hy * s, hx * s + hy * c


def _choice(rng: np.random.Generator, options: list[str]) -> str:
    return options[int(rng.integers(len(options)))]


def _pick(rng: np.random.Generator, options: list[str], n: int) -> list[str]:
    return [options[i] for i in rng.choice(len(options), size=n, replace=False)]


def _side_range(sign: float, lo: float = SIDE_X[0], hi: float = SIDE_X[1]) -> tuple[float, float]:
    return (lo, hi) if sign > 0 else (-hi, -lo)


def _item(pos, yaw, colour, label, half, top, **extra) -> dict[str, Any]:
    return {"pos": [float(v) for v in pos], "yaw": float(yaw), "colour": colour, "label": label,
            "half_size": [float(h) for h in half], "top": float(top), **extra}


def _handover(rng: np.random.Generator) -> dict | None:
    lay = _Layout(rng)
    src = _choice(rng, ["left", "right"])
    dst = "right" if src == "left" else "left"
    s_sign = -1.0 if src == "left" else 1.0
    half_len = float(rng.uniform(*BAR_LEN_RANGE))
    half = (BAR_HALF[0], half_len, BAR_HALF[1])
    yaw = float(rng.uniform(-0.3, 0.3))
    ex, ey = _extent(half[0], half[1], yaw)
    xy = lay.put(_side_range(s_sign, SIDE_X[0], min(SIDE_X[1], BASE_CLEAR - ex)), (-0.10, 0.18), ex, ey, 0.03)
    bar_c, *zone_c = _pick(rng, sorted(COLOURS), 3)
    zones = []
    n_zones = 1 + int(rng.random() < 0.5)
    for c in zone_c[:n_zones]:
        zyaw = float(rng.uniform(-0.3, 0.3))
        zx, zy = _extent(ZONE_HALF, ZONE_HALF, zyaw)
        zxy = lay.put((0.29, 0.36) if s_sign < 0 else (-0.36, -0.29), (-0.16, 0.24), zx, zy, 0.01)
        if zxy is None:
            return None
        zones.append((c, zxy, zyaw))
    if xy is None:
        return None
    items = {"bar0": _item([xy[0], xy[1], TABLE_TOP + half[2] + 0.0005], yaw, bar_c, f"{bar_c} bar", half,
                           TABLE_TOP + 2 * half[2], size=list(half), mass=float(rng.uniform(0.03, 0.06)))}
    for i, (c, zxy, zyaw) in enumerate(zones):
        items[f"zone{i}"] = _item([zxy[0], zxy[1], TABLE_TOP], zyaw, c, f"{c} zone", (ZONE_HALF, ZONE_HALF, 0.0006),
                                  TABLE_TOP + 0.0012)
    target = f"zone{int(rng.integers(len(zones)))}"
    b, z = items["bar0"]["label"], items[target]["label"]
    text = _choice(rng, [f"hand the {b} over to the other arm and put it in the {z}",
                         f"pass the {b} from one gripper to the other and place it in the {z}",
                         f"move the {b} into the {z}; you will need both arms",
                         f"use both arms to bring the {b} to the {z}"])
    return {"items": items, "goal": {"object": "bar0", "zone": target, "from": src, "to": dst}, "instruction": text}


def _lift_box(rng: np.random.Generator) -> dict | None:
    colour = _choice(rng, sorted(COLOURS))
    yaw = float(rng.uniform(-0.25, 0.25))
    x, y = float(rng.uniform(-0.03, 0.03)), float(rng.uniform(-0.04, 0.12))
    height = float(_choice(rng, ["0.08", "0.10", "0.12"]))
    half = (LIFT_BODY[0] + 2 * LIFT_FIN[0], LIFT_BODY[1], LIFT_BODY[2])
    mass = float(rng.uniform(0.8, 1.2) * LIFT_MASS)
    items = {"liftbox0": _item([x, y, TABLE_TOP + LIFT_BODY[2] + 0.0005], yaw, colour, f"{colour} box", half,
                               TABLE_TOP + 2 * LIFT_BODY[2], mass=mass)}
    lab, cm = items["liftbox0"]["label"], round(height * 100)
    text = _choice(rng, [f"lift the {lab} with both arms at least {cm} cm off the table and hold it level",
                         f"use both grippers to raise the {lab} {cm} cm and keep it level",
                         f"pick up the {lab} by its two handles and hold it {cm} cm above the table"])
    return {"items": items, "goal": {"object": "liftbox0", "height": height, "hold_steps": HOLD_STEPS,
                                     "tilt_max": TILT_MAX}, "instruction": text}


def _two_bins(rng: np.random.Generator) -> dict | None:
    lay = _Layout(rng)
    blocks_c = _pick(rng, sorted(COLOURS), 2)
    bins_c = _pick(rng, sorted(FIXTURE_COLOURS), 2)
    items, assign, order = {}, {}, []
    o = BIN_IN + 2 * BIN_WALL
    for i, sign in enumerate((-1.0, 1.0)):
        bxy = lay.put(_side_range(sign, 0.17, BASE_CLEAR - o), (-0.14, 0.22), o, o, 0.03)
        h = float(rng.uniform(*BLOCK_RANGE))
        yaw = float(rng.uniform(-math.pi / 4, math.pi / 4))
        ex, _ = _extent(h, h, yaw)
        kxy = lay.put(_side_range(sign, 0.15, BASE_CLEAR - ex), (-0.14, 0.22), ex, ex, 0.05)
        if bxy is None or kxy is None:
            return None
        items[f"bin{i}"] = _item([bxy[0], bxy[1], TABLE_TOP], 0.0, bins_c[i], f"{bins_c[i]} bin", (o, o, BIN_H / 2),
                                 TABLE_TOP + BIN_BASE)
        items[f"block{i}"] = _item([kxy[0], kxy[1], TABLE_TOP + h + 0.0005], yaw, blocks_c[i], f"{blocks_c[i]} block",
                                   (h, h, h), TABLE_TOP + 2 * h, size=[h, h, h])
        assign[f"block{i}"] = f"bin{i}"
        order.append(i)
    if rng.random() < 0.5:
        order.reverse()
    parts = [f"the {items[f'block{i}']['label']} in the {items[f'bin{i}']['label']}" for i in order]
    text = _choice(rng, [f"put {parts[0]} and {parts[1]}, using both arms at the same time",
                         f"at the same time, place {parts[0]} and {parts[1]}",
                         f"quickly put {parts[0]} and {parts[1]}: there is only time if both arms work together"])
    return {"items": items, "goal": {"assign": assign}, "instruction": text}


SAMPLERS = {"handover": _handover, "lift_box": _lift_box, "two_bins": _two_bins}


def sample_episode(params: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    kind = params.get("task")
    if kind not in SAMPLERS:
        raise ValueError(f"task params need 'task' in {list(SAMPLERS)}, got {kind!r}")
    for _ in range(500):
        ep = SAMPLERS[kind](rng)
        if ep is not None:
            ep["task"] = kind
            ep["state"] = {}
            return ep
    raise RuntimeError(f"could not sample a {kind} layout")


# -- success predicates -----------------------------------------------------------------------------------------
def _local_xy(env: "AlohaTableEnv", name: str, ref: str) -> np.ndarray:
    p, _ = env.pose(name)
    c, q = env.pose(ref)
    yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    dx, dy = p[0] - c[0], p[1] - c[1]
    return np.array([math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy])


def _settled(env: "AlohaTableEnv", name: str) -> bool:
    lin, ang = env.speed(name)
    return lin < LIN_REST and ang < ANG_REST and not env.touching_hand(name)


def tilt(env: "AlohaTableEnv", name: str) -> float:
    return float(np.arccos(np.clip(env.rotation(name)[2, 2], -1.0, 1.0)))


def evaluate(env: "AlohaTableEnv") -> tuple[bool, str | None, dict[str, Any]]:
    """(success, failure reason or None, info) after a control step."""
    ep = env.ep
    items, goal, kind, state = ep["items"], ep["goal"], ep["task"], ep.setdefault("state", {})
    for name in items:
        if env.items[name]["kind"] in env.FREE_KINDS and env.pose(name)[0][2] < TABLE_TOP - 0.05:
            return False, f"the {items[name]['label']} fell off the table", {"done": 0}
    if kind == "handover":
        obj, zone = goal["object"], goal["zone"]
        local = _local_xy(env, obj, zone)
        z = env.pose(obj)[0][2]
        inside = bool(np.all(np.abs(local) < ZONE_HALF - 0.015))
        resting = z < TABLE_TOP + items[obj]["half_size"][2] + 0.012
        ok = inside and resting and _settled(env, obj)
        return ok, None, {"done": int(ok), "of": 1, "distance": float(np.linalg.norm(local))}
    if kind == "lift_box":
        obj = goal["object"]
        bottom = env.pose(obj)[0][2] - LIFT_BODY[2] - TABLE_TOP
        t = tilt(env, obj)
        if bottom > 0.04:
            state["lifted"] = 1
        if state.get("lifted") and bottom < 0.01 and not env.touching_hand(obj):
            return False, f"dropped the {items[obj]['label']}", {"done": 0, "of": 1, "height": bottom}
        up = bottom >= goal["height"] and t <= goal["tilt_max"]
        state["held"] = int(state.get("held", 0)) + 1 if up else 0
        ok = state["held"] >= goal["hold_steps"]
        return ok, None, {"done": int(ok), "of": 1, "height": float(bottom), "tilt": t, "held": state["held"]}
    if kind == "two_bins":
        done = 0
        for obj, b in goal["assign"].items():
            local = _local_xy(env, obj, b)
            z = env.pose(obj)[0][2]
            h = items[obj]["half_size"][2]
            if np.all(np.abs(local) < BIN_IN - 0.008) and z < TABLE_TOP + BIN_BASE + h + 0.012 and _settled(env, obj):
                done += 1
        return done == len(goal["assign"]), None, {"done": done, "of": len(goal["assign"])}
    raise ValueError(f"unknown task {kind!r}")


# -- tasks and benchmark ----------------------------------------------------------------------------------------
def bimanual_tasks(kinds: list[str] | tuple[str, ...] | None = None) -> tuple[TaskSpec, ...]:
    kinds = tuple(kinds or TASK_KINDS)
    unknown = [k for k in kinds if k not in TASK_KINDS]
    if unknown:
        raise ValueError(f"unknown bimanual tasks {unknown}; known: {list(TASK_KINDS)}")
    return tuple(TaskSpec(id=f"mujoco/bimanual_{k}", version="1", instruction=TEMPLATES[k], params={"task": k},
                          max_steps=MAX_STEPS[k], tags=("manipulation", *TAGS[k])) for k in kinds)


VALIDATION_SEED_OFFSET = 10_000


def bimanual_benchmark(tasks: list[str] | None = None, protocol: str = "rsi", rsi_states: int = 50,
                       validation: int = 20, train: int = 10, heldout: int = 20, gate_looks: int = 1):
    """Splits as in mujoco/manip-v0: protocol "rsi" (default) gives RFC-0002 search / search_val / gate / heldout over
    `rsi_states` seeds plus a separate `validation` split; "standard" gives train / validation / heldout."""
    from hylo.core.benchmark import Benchmark

    from .planar_reach import splits_for

    splits = splits_for(protocol, train, validation, heldout, rsi_states, gate_looks)
    if protocol == "rsi":
        splits["validation"] = tuple(range(VALIDATION_SEED_OFFSET, VALIDATION_SEED_OFFSET + validation))
    return Benchmark(id="mujoco/bimanual-v0", version="0.1", tasks=bimanual_tasks(tasks), splits=splits,
                     metadata={"env": "mujoco.aloha_table", "protocol": protocol, "tasks": ",".join(tasks or TASK_KINDS)})


__all__ = ["MAX_STEPS", "TASK_KINDS", "bimanual_benchmark", "bimanual_tasks", "evaluate", "sample_episode", "tilt"]
