"""The `mujoco/dexterous-v0` task family on `mujoco.dex_table`: lift an object with a multi-finger hand.

| Task | Success |
|---|---|
| lift_<shape> | the object (box, cylinder, sphere, mug or rock) held by the hand at least 8 cm above where it started |

Each episode samples the object's position (within 6 cm of the origin), yaw and colour from its seed. Knocking the
object off the table ends the episode as a failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from hylo.core.task import TaskSpec

from .arm_table import COLOURS
from .dex_table import SHAPES, TABLE_TOP, WORKSPACE, shape

if TYPE_CHECKING:
    from .dex_table import DexTableEnv

LIFT = 0.08
MAX_STEPS = 400
TASK_KINDS = tuple(f"lift_{s}" for s in SHAPES)
LABELS = {"box": "block", "cylinder": "cylinder", "sphere": "ball", "mug": "mug", "rock": "rock"}


def sample_episode(params: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    kind = params.get("object") or (params.get("task", "").removeprefix("lift_") or None)
    if kind not in SHAPES:
        raise ValueError(f"task params need 'object' in {list(SHAPES)}, got {kind!r}")
    (x0, x1), (y0, y1) = WORKSPACE
    colour = list(COLOURS)[int(rng.integers(len(COLOURS)))]
    name = f"{kind}0"
    shp = shape(kind)
    label = f"{colour} {LABELS[kind]}"
    item = {"pos": [float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)), TABLE_TOP + 0.0005],
            "yaw": float(rng.uniform(-np.pi, np.pi)), "colour": colour, "label": label, "half_size": shp.half_size,
            "top": TABLE_TOP + 2 * shp.half_size[2]}
    return {"task": f"lift_{kind}", "items": {name: item}, "goal": {"object": name, "height": LIFT},
            "instruction": f"lift the {label}", "state": {}}


def evaluate(env: "DexTableEnv") -> tuple[bool, str | None, dict[str, Any]]:
    name = env.ep["goal"]["object"]
    z0 = env.ep["items"][name]["pos"][2]
    z = float(env.pose(name)[0][2])
    if z < TABLE_TOP - 0.05:
        return False, f"the {env.ep['items'][name]['label']} fell off the table", {"lifted_m": 0.0}
    lifted = z - z0
    ok = lifted >= env.ep["goal"]["height"] and env.touching_hand(name)
    return bool(ok), None, {"lifted_m": round(lifted, 4)}


def dex_tasks(kinds: list[str] | tuple[str, ...] | None = None) -> tuple[TaskSpec, ...]:
    kinds = tuple(kinds or TASK_KINDS)
    unknown = [k for k in kinds if k not in TASK_KINDS]
    if unknown:
        raise ValueError(f"unknown dexterous tasks {unknown}; known: {list(TASK_KINDS)}")
    return tuple(TaskSpec(id=f"mujoco/dex_{k}", version="1", instruction="lift the <object>",
                          params={"task": k, "object": k.removeprefix("lift_")}, max_steps=MAX_STEPS,
                          tags=("manipulation", "dexterous", "grasp", "lift")) for k in kinds)


def dexterous_benchmark(tasks: list[str] | None = None, episodes: int = 10, heldout: int = 10):
    """Seeds 0..episodes-1 as the validation split, then a held-out split (protocol "standard")."""
    from hylo.core.benchmark import Benchmark

    splits = {"validation": tuple(range(episodes)), "heldout": tuple(range(1000, 1000 + heldout))}
    return Benchmark(id="mujoco/dexterous-v0", version="0.1", tasks=dex_tasks(tasks), splits=splits,
                     metadata={"env": "mujoco.dex_table", "tasks": ",".join(tasks or TASK_KINDS)})


__all__ = ["LIFT", "MAX_STEPS", "TASK_KINDS", "dex_tasks", "dexterous_benchmark", "evaluate", "sample_episode"]
