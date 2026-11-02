"""The `mujoco/humanoid-v0` task family on `mujoco.g1_table`: a humanoid's hands at a table.

| Task | Success |
|---|---|
| pick_place   | the named block rests on the named plate, released (one hand, a multi-finger grasp) |
| two_hands    | each named block on its named plate, both released, within the step limit: each block and its plate lie on one hand's side, and the limit is shorter than two pick-and-place motions one after the other |
| press_button | the button of the named colour pressed down (a pointing index finger); pressing another one fails at once |

Each task samples positions, yaw, sizes, colours and the instruction from the episode seed. Failures end the episode
early: an object falls off the table or a wrong button is pressed.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from hylo.core.task import TaskSpec

from .arm_table import COLOURS, FIXTURE_COLOURS
from .g1_table import (BLOCK_RANGE, BTN_CAP_H, BTN_HOUSE, BTN_HOUSE_H, PLATE_HALF, PLATE_R, TABLE_TOP,
                       WORKSPACE)

if TYPE_CHECKING:
    from .g1_table import G1TableEnv

TASK_KINDS = ("pick_place", "two_hands", "press_button")
# two_hands: the oracle needs about 210 steps with both hands at once and about 415 with one after the other
MAX_STEPS = {"pick_place": 500, "two_hands": 340, "press_button": 300}
TEMPLATES = {"pick_place": "put the <block> on the <plate>",
             "two_hands": "put each block on the named plate, both hands at once", "press_button": "press the <colour> button"}
TAGS = {"pick_place": ("humanoid", "dexterous", "grasp", "place"),
        "two_hands": ("humanoid", "bimanual", "parallel", "grasp", "place", "time-limit"),
        "press_button": ("humanoid", "dexterous", "press")}
LIN_REST, ANG_REST = 0.03, 0.8


class _Layout:
    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.boxes: list[tuple[float, float, float, float]] = []

    def put(self, hx: float, hy: float, gap: float, xr=None, yr=None, tries: int = 300):
        (x0, x1), (y0, y1) = WORKSPACE
        if xr is not None:
            x0, x1 = max(x0, xr[0]), min(x1, xr[1])
        if yr is not None:
            y0, y1 = max(y0, yr[0]), min(y1, yr[1])
        for _ in range(tries):
            x, y = float(self.rng.uniform(x0, x1)), float(self.rng.uniform(y0, y1))
            if all(abs(x - bx) >= hx + bhx + gap or abs(y - by) >= hy + bhy + gap for bx, by, bhx, bhy in self.boxes):
                self.boxes.append((x, y, hx, hy))
                return x, y
        return None


def _choice(rng, options):
    return options[int(rng.integers(len(options)))]


def _pick(rng, options, n):
    return [options[i] for i in rng.choice(len(options), size=n, replace=False)]


def _item(pos, yaw, colour, label, half, top, **extra) -> dict[str, Any]:
    return {"pos": [float(v) for v in pos], "yaw": float(yaw), "colour": colour, "label": label,
            "half_size": [float(h) for h in half], "top": float(top), **extra}


def _pick_place(rng) -> dict | None:
    """The block and its plate on one hand's side (y > 0: the left hand's), so one hand reaches both; a second plate
    and a second block are distractors anywhere."""
    lay = _Layout(rng)
    items = {}
    side = (0.0, 0.25) if rng.random() < 0.5 else (-0.25, 0.0)
    n_plates = int(rng.integers(1, 3))
    plate_c = _pick(rng, sorted(FIXTURE_COLOURS), n_plates)
    for i, c in enumerate(plate_c):
        xy = lay.put(PLATE_R, PLATE_R, 0.03, yr=side if i == 0 else None)
        if xy is None:
            return None
        items[f"plate{i}"] = _item([xy[0], xy[1], TABLE_TOP], 0.0, c, "plate" if n_plates == 1 else f"{c} plate",
                                   (PLATE_R, PLATE_R, PLATE_HALF), TABLE_TOP + 2 * PLATE_HALF)
    n_blocks = int(rng.integers(1, 3))
    block_c = _pick(rng, sorted(COLOURS), n_blocks)
    for i, c in enumerate(block_c):
        h = float(rng.uniform(*BLOCK_RANGE))
        yaw = float(rng.uniform(-math.pi / 4, math.pi / 4))
        xy = lay.put(h * 1.42, h * 1.42, 0.07, yr=side if i == 0 else None)
        if xy is None:
            return None
        items[f"block{i}"] = _item([xy[0], xy[1], TABLE_TOP + h + 0.0005], yaw, c, f"{c} block", (h, h, h),
                                   TABLE_TOP + 2 * h, size=[h, h, h])
    o, p = items["block0"]["label"], items["plate0"]["label"]
    text = _choice(rng, [f"put the {o} on the {p}", f"place the {o} on the {p}", f"pick up the {o} and set it on the {p}"])
    return {"items": items, "goal": {"object": "block0", "target": "plate0"}, "instruction": text}


def _two_hands(rng) -> dict | None:
    lay = _Layout(rng)
    items, assign = {}, {}
    blocks_c = _pick(rng, sorted(COLOURS), 2)
    plates_c = _pick(rng, sorted(FIXTURE_COLOURS), 2)
    for i, (lo, hi) in enumerate(((0.08, 0.25), (-0.25, -0.08))):          # left hand's side, then the right's
        pxy = lay.put(PLATE_R, PLATE_R, 0.02, yr=(lo, hi))
        h = float(rng.uniform(*BLOCK_RANGE))
        kxy = lay.put(h * 1.42, h * 1.42, 0.06, yr=(lo, hi))
        if pxy is None or kxy is None:
            return None
        items[f"plate{i}"] = _item([pxy[0], pxy[1], TABLE_TOP], 0.0, plates_c[i], f"{plates_c[i]} plate",
                                   (PLATE_R, PLATE_R, PLATE_HALF), TABLE_TOP + 2 * PLATE_HALF)
        items[f"block{i}"] = _item([kxy[0], kxy[1], TABLE_TOP + h + 0.0005], float(rng.uniform(-0.6, 0.6)), blocks_c[i],
                                   f"{blocks_c[i]} block", (h, h, h), TABLE_TOP + 2 * h, size=[h, h, h])
        assign[f"block{i}"] = f"plate{i}"
    order = [0, 1] if rng.random() < 0.5 else [1, 0]
    parts = [f"the {items[f'block{i}']['label']} on the {items[f'plate{i}']['label']}" for i in order]
    text = _choice(rng, [f"with both hands at once, put {parts[0]} and {parts[1]}",
                         f"put {parts[0]} and {parts[1]}, one hand each, at the same time",
                         f"quickly place {parts[0]} and {parts[1]}: use both hands together"])
    return {"items": items, "goal": {"assign": assign}, "instruction": text}


def _press_button(rng) -> dict | None:
    lay = _Layout(rng)
    n = int(rng.integers(2, 4))
    colours = _pick(rng, sorted(COLOURS), n)
    items = {}
    for i, c in enumerate(colours):
        xy = lay.put(BTN_HOUSE, BTN_HOUSE, 0.05, xr=(0.22, 0.31))
        if xy is None:
            return None
        items[f"button{i}"] = _item([xy[0], xy[1], TABLE_TOP], 0.0, c, f"{c} button", (BTN_HOUSE, BTN_HOUSE, BTN_HOUSE_H + BTN_CAP_H),
                                    TABLE_TOP + 2 * (BTN_HOUSE_H + BTN_CAP_H))
    target = f"button{int(rng.integers(n))}"
    lab = items[target]["label"]
    text = _choice(rng, [f"press the {lab}", f"push the {lab} down", f"press the {lab} once"])
    return {"items": items, "goal": {"button": target}, "instruction": text}


SAMPLERS = {"pick_place": _pick_place, "two_hands": _two_hands, "press_button": _press_button}


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


def _settled(env: "G1TableEnv", name: str) -> bool:
    lin, ang = env.speed(name)
    return lin < LIN_REST and ang < ANG_REST and not env.touching_hand(name)


def _on_plate(env: "G1TableEnv", obj: str, tgt: str) -> bool:
    items = env.ep["items"]
    p = env.pose(obj)[0]
    near = np.linalg.norm(p[:2] - env.pose(tgt)[0][:2]) < PLATE_R - 0.01
    h = items[obj]["half_size"]
    on = items[tgt]["top"] + min(h) - 0.008 <= p[2] <= items[tgt]["top"] + max(h) * 1.42 + 0.008
    return bool(near and on and _settled(env, obj))


def tilt(env: "G1TableEnv", name: str) -> float:
    return float(np.arccos(np.clip(env.rotation(name)[2, 2], -1.0, 1.0)))


def evaluate(env: "G1TableEnv") -> tuple[bool, str | None, dict[str, Any]]:
    ep = env.ep
    items, goal, kind, state = ep["items"], ep["goal"], ep["task"], ep.setdefault("state", {})
    for name in items:
        if env.items[name]["kind"] in env.FREE_KINDS and env.pose(name)[0][2] < TABLE_TOP - 0.05:
            return False, f"the {items[name]['label']} fell off the table", {"done": 0}
    if kind in ("pick_place", "two_hands"):
        pairs = {goal["object"]: goal["target"]} if kind == "pick_place" else goal["assign"]
        done = sum(int(_on_plate(env, obj, tgt)) for obj, tgt in pairs.items())
        return done == len(pairs), None, {"done": done, "of": len(pairs)}
    if kind == "press_button":
        pressed = state.get("pressed", [])
        wrong = [b for b in pressed if b != goal["button"]]
        if wrong:
            return False, f"pressed the {items[wrong[0]]['label']}", {"done": 0, "of": 1}
        ok = goal["button"] in pressed
        return ok, None, {"done": int(ok), "of": 1}
    raise ValueError(f"unknown task {kind!r}")


def humanoid_tasks(kinds: list[str] | tuple[str, ...] | None = None) -> tuple[TaskSpec, ...]:
    kinds = tuple(kinds or TASK_KINDS)
    unknown = [k for k in kinds if k not in TASK_KINDS]
    if unknown:
        raise ValueError(f"unknown humanoid tasks {unknown}; known: {list(TASK_KINDS)}")
    return tuple(TaskSpec(id=f"mujoco/humanoid_{k}", version="1", instruction=TEMPLATES[k], params={"task": k},
                          max_steps=MAX_STEPS[k], tags=("manipulation", *TAGS[k])) for k in kinds)


VALIDATION_SEED_OFFSET = 10_000


def humanoid_benchmark(tasks: list[str] | None = None, protocol: str = "rsi", rsi_states: int = 50,
                       validation: int = 20, train: int = 10, heldout: int = 20, gate_looks: int = 1):
    """Splits as in mujoco/manip-v0 (protocol "rsi" with a separate validation split, or "standard")."""
    from hylo.core.benchmark import Benchmark

    from .planar_reach import splits_for

    splits = splits_for(protocol, train, validation, heldout, rsi_states, gate_looks)
    if protocol == "rsi":
        splits["validation"] = tuple(range(VALIDATION_SEED_OFFSET, VALIDATION_SEED_OFFSET + validation))
    return Benchmark(id="mujoco/humanoid-v0", version="0.1", tasks=humanoid_tasks(tasks), splits=splits,
                     metadata={"env": "mujoco.g1_table", "protocol": protocol, "tasks": ",".join(tasks or TASK_KINDS)})


__all__ = ["MAX_STEPS", "TASK_KINDS", "evaluate", "humanoid_benchmark", "humanoid_tasks", "sample_episode", "tilt"]
