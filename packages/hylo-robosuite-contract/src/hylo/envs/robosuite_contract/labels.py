"""Instance ids and human labels for the geoms of a robosuite scene (pure numpy: works on name lists, so suites test
their rules without MuJoCo).

Id 1 is the support (a table, a counter), id 2 the robot, then one id per label in the order the rules list them
(so ids do not depend on the scene's contents); 0 is background (walls, floor, scenery no rule names). Rules are
tried in this order for each geom:

1. `attached`: (regex on the body name, label): things fixed into the robot's own model (a peg on a hand);
2. the robot: bodies whose top-level ancestor (or own name) starts with one of `robot_prefixes`;
3. `parts`: (regex on the geom or body name, label): a handle on a pot, a knob on a stove;
4. `objects`: {name: label} for objects whose bodies are named `<name>_...` (longest name first);
5. `bodies`: {body name: label}, matched against the body or its top-level ancestor (bins, pegs);
6. `markers`: {body name prefix: label} (many small bodies under one label, e.g. dirt spots);
7. `support`: body names (exact, or starting with one of `support_prefixes`) that make up the support.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

ROBOT_PREFIXES = ("robot", "gripper", "mount", "fixed_mount", "mobilebase")


@dataclass(frozen=True)
class LabelRules:
    objects: dict[str, str] = field(default_factory=dict)
    parts: tuple[tuple[str, str], ...] = ()
    bodies: dict[str, str] = field(default_factory=dict)
    markers: dict[str, str] = field(default_factory=dict)
    attached: tuple[tuple[str, str], ...] = ()
    support: tuple[str, ...] = ("table",)
    support_prefixes: tuple[str, ...] = ()
    support_label: str = "table"
    robot_prefixes: tuple[str, ...] = ROBOT_PREFIXES


@dataclass
class InstanceLabels:
    """geom[g] = instance id of geom g; labels = {id: label}."""

    geom: np.ndarray
    labels: dict[int, str]

    def segments(self) -> dict[str, str]:
        """`extra["segments"]` of an observation: {"<id>": "<label>"}."""
        return {str(k): v for k, v in self.labels.items()}


def top_ancestors(body_names: Sequence[str], body_parent: Sequence[int]) -> list[str]:
    """Each body's ancestor just below the world body (itself for top-level bodies; '' for the world)."""
    out = []
    for b in range(len(body_names)):
        a = b
        while a > 0 and int(body_parent[a]) > 0:
            a = int(body_parent[a])
        out.append(body_names[a] if a > 0 else "")
    return out


def label_geoms(body_names: Sequence[str], body_parent: Sequence[int], geom_names: Sequence[str],
                geom_body: Sequence[int], rules: LabelRules) -> InstanceLabels:
    labels: dict[int, str] = {1: rules.support_label, 2: "robot"}
    ids: dict[str, int] = {rules.support_label: 1, "robot": 2}

    def iid(label: str) -> int:
        if label not in ids:
            ids[label] = len(labels) + 1
            labels[ids[label]] = label
        return ids[label]

    for label in [*(lab for _, lab in rules.attached), *(lab for _, lab in rules.parts), *rules.objects.values(),
                  *rules.bodies.values(), *rules.markers.values()]:
        iid(label)
    attached = [(re.compile(p), lab) for p, lab in rules.attached]
    parts = [(re.compile(p), lab) for p, lab in rules.parts]
    objects = sorted(rules.objects, key=len, reverse=True)
    top = top_ancestors(body_names, body_parent)
    support = set(rules.support)
    out = np.zeros(len(geom_names), dtype=np.int32)
    for g, gname in enumerate(geom_names):
        b = int(geom_body[g])
        name, root = body_names[b], top[b]
        hit = next((lab for rx, lab in attached if rx.search(name)), None)
        if hit is None and (root.startswith(rules.robot_prefixes) or name.startswith(rules.robot_prefixes)):
            out[g] = 2
            continue
        if hit is None:
            hit = next((lab for rx, lab in parts if rx.search(gname) or rx.search(name)), None)
        if hit is None:
            obj = next((o for o in objects if name.startswith(o + "_") or name == o), None)
            hit = rules.objects[obj] if obj is not None else None
        if hit is None:
            hit = rules.bodies.get(name) or rules.bodies.get(root)
        if hit is None:
            hit = next((lab for p, lab in rules.markers.items() if name.startswith(p)), None)
        if hit is not None:
            out[g] = iid(hit)
        elif name in support or root in support or (rules.support_prefixes and
                                                    (name.startswith(rules.support_prefixes) or
                                                     root.startswith(rules.support_prefixes))):
            out[g] = 1
    return InstanceLabels(geom=out, labels=labels)


__all__ = ["InstanceLabels", "LabelRules", "ROBOT_PREFIXES", "label_geoms", "top_ancestors"]
