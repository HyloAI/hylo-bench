"""Checks on a stage world (RFC-0008 part 4): goals as predicates over the world's state, answered with reasons.

A check never steps the simulation. Each failed atom says what was expected, what was measured and a reason code,
so an agent (or a skill matched by reason code, RFC-0003) knows what to do next:

```text
  atom          fields                                   fails with
  in_box        entity, min [x,y,z], max [x,y,z]         POSE_OUTSIDE_BOX
  near          a, b (entities or [x,y,z]), max_m        DISTANCE_NOT_MET
  apart         a, b, min_m                              DISTANCE_NOT_MET
  above         a, b, min_m (a's centre higher than b's) NOT_ABOVE
  contact       a, b (entity ids; b "any")               CONTACT_MISSING
  no_contact    a, b                                     CONTACT_PRESENT
  joint_range   robot, joint, min, max                   JOINT_OUT_OF_RANGE
  settled       entity, max_speed (m/s)                  NOT_SETTLED
  upright       entity (a robot base or an object), min_cos  NOT_UPRIGHT
  lifted        entity, min_z                            NOT_LIFTED
  reachable     robot, point, margin_m                   UNREACHABLE (beyond the pack's reach from its base)
  no_penetration  max_mm                                 PENETRATION_PRESENT
  all / any / not   terms (all, any) or term (not)
```

An atom that cannot be evaluated (an entity that does not exist, a robot without a reach) is `unverifiable`
(ENTITY_MISSING, OBSERVATION_UNAVAILABLE): `not` of it stays unverifiable and `all` with it is not true. The spec
gate (`gate`) refuses a goal whose entities are missing and reports SPEC_ALREADY_SATISFIED when the goal already
holds before anything was done.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

TRUE, FALSE, UNKNOWN = "true", "false", "unverifiable"


def _point(world: Any, ref: Any) -> np.ndarray | None:
    if isinstance(ref, (list, tuple)) and len(ref) == 3:
        return np.asarray(ref, float)
    m, d = world.model, world.data
    if ref in world.robots:
        r = world.robots[ref]
        t = r.tcp(m, d)
        return t[0] if t is not None else d.xpos[r.root_body].copy()
    try:
        return d.xpos[m.body(str(ref)).id].copy()
    except KeyError:
        if world.humans is not None and ref in world.humans.ids:
            return world.humans.position(ref)
        return None


def _speed(world: Any, ent: str) -> float | None:
    try:
        b = world.model.body(ent).id
    except KeyError:
        return None
    return float(np.linalg.norm(world.data.cvel[b][3:]))


def _contacts(world: Any) -> set[tuple[str, str]]:
    m, d = world.model, world.data
    out = set()
    for i in range(d.ncon):
        c = d.contact[i]
        a = world.entity_of_body(int(m.geom_bodyid[c.geom1])) or "world"
        b = world.entity_of_body(int(m.geom_bodyid[c.geom2])) or "world"
        out.add((a, b))
        out.add((b, a))
    return out


def _fail(atom: Mapping[str, Any], code: str, expected: Any, measured: Any) -> dict[str, Any]:
    return {"value": FALSE, "failures": [{"atom": dict(atom), "reason": code, "expected": expected,
                                          "measured": measured}]}


def _unknown(atom: Mapping[str, Any], code: str, why: str) -> dict[str, Any]:
    return {"value": UNKNOWN, "failures": [{"atom": dict(atom), "reason": code, "why": why}]}


def evaluate(world: Any, node: Mapping[str, Any]) -> dict[str, Any]:
    """{"value": true | false | unverifiable, "failures": [...]} for a predicate node."""
    op = node.get("op")
    if op in ("all", "any"):
        res = [evaluate(world, t) for t in node.get("terms") or []]
        vals = [r["value"] for r in res]
        fails = [f for r in res for f in r["failures"]]
        if op == "all":
            v = FALSE if FALSE in vals else UNKNOWN if UNKNOWN in vals else TRUE
            return {"value": v, "failures": fails if v != TRUE else []}
        v = TRUE if TRUE in vals else UNKNOWN if UNKNOWN in vals else FALSE
        return {"value": v, "failures": [] if v == TRUE else fails}
    if op == "not":
        r = evaluate(world, node["term"])
        if r["value"] == UNKNOWN:
            return r
        if r["value"] == TRUE:
            return _fail(node, "NEGATION_FAILED", "the inner condition false", "it holds")
        return {"value": TRUE, "failures": []}
    try:
        return _atom(world, node)
    except KeyError as exc:                       # an atom without a field it needs: say which
        return _unknown(node, "BAD_ATOM", f"{node.get('op')} needs the field {exc}")
    except (TypeError, ValueError) as exc:
        return _unknown(node, "BAD_ATOM", f"{node.get('op')}: {exc}")


def _atom(world: Any, a: Mapping[str, Any]) -> dict[str, Any]:
    op = a.get("op")
    ok = {"value": TRUE, "failures": []}
    if op in ("in_box", "lifted", "settled", "upright"):
        p = _point(world, a["entity"])
        if p is None:
            return _unknown(a, "ENTITY_MISSING", f"no entity {a['entity']!r}")
        if op == "in_box":
            lo, hi = np.asarray(a["min"], float), np.asarray(a["max"], float)
            return ok if bool(np.all(p >= lo) and np.all(p <= hi)) else _fail(
                a, "POSE_OUTSIDE_BOX", {"min": lo.tolist(), "max": hi.tolist()}, np.round(p, 4).tolist())
        if op == "lifted":
            return ok if p[2] >= float(a["min_z"]) else _fail(a, "NOT_LIFTED", f"z >= {a['min_z']}", round(float(p[2]), 4))
        if op == "settled":
            s = _speed(world, a["entity"])
            if s is None:
                return _unknown(a, "OBSERVATION_UNAVAILABLE", "no body to measure")
            lim = float(a.get("max_speed", 0.01))
            return ok if s <= lim else _fail(a, "NOT_SETTLED", f"speed <= {lim}", round(s, 4))
        m, d = world.model, world.data
        ent = a["entity"]
        b = world.robots[ent].root_body if ent in world.robots else m.body(ent).id
        cos = float(d.xmat[b].reshape(3, 3)[2, 2])
        lim = float(a.get("min_cos", 0.9))
        return ok if cos >= lim else _fail(a, "NOT_UPRIGHT", f"up axis cos >= {lim}", round(cos, 3))
    if op in ("near", "apart", "above"):
        pa, pb = _point(world, a["a"]), _point(world, a["b"])
        if pa is None or pb is None:
            return _unknown(a, "ENTITY_MISSING", f"no entity {a['a'] if pa is None else a['b']!r}")
        if op == "above":
            dz = float(pa[2] - pb[2])
            lim = float(a.get("min_m", 0.0))
            return ok if dz >= lim else _fail(a, "NOT_ABOVE", f"height difference >= {lim}", round(dz, 4))
        dist = float(np.linalg.norm(pa - pb))
        if op == "near":
            lim = float(a["max_m"])
            return ok if dist <= lim else _fail(a, "DISTANCE_NOT_MET", f"<= {lim} m", round(dist, 4))
        lim = float(a["min_m"])
        return ok if dist >= lim else _fail(a, "DISTANCE_NOT_MET", f">= {lim} m", round(dist, 4))
    if op in ("contact", "no_contact"):
        pairs = _contacts(world)
        known = {e.id for e in world.stage.entities} | {"world", "any"}
        for k in ("a", "b"):
            if a[k] not in known:
                return _unknown(a, "ENTITY_MISSING", f"no entity {a[k]!r}")
        hit = any(x == a["a"] and (a["b"] == "any" or y == a["b"]) for x, y in pairs)
        if op == "contact":
            return ok if hit else _fail(a, "CONTACT_MISSING", "a contact", "none")
        return ok if not hit else _fail(a, "CONTACT_PRESENT", "no contact", "contact")
    if op == "joint_range":
        r = world.robots.get(a["robot"])
        if r is None or a["joint"] not in r.joints:
            return _unknown(a, "ENTITY_MISSING", f"no robot joint {a['robot']}/{a['joint']}")
        j = r.joint_ids[r.joints.index(a["joint"])]
        q = float(world.data.qpos[world.model.jnt_qposadr[j]])
        return ok if float(a["min"]) <= q <= float(a["max"]) else _fail(
            a, "JOINT_OUT_OF_RANGE", [a["min"], a["max"]], round(q, 4))
    if op == "reachable":
        r = world.robots.get(a["robot"])
        p = _point(world, a["point"])
        if r is None or p is None:
            return _unknown(a, "ENTITY_MISSING", "robot or point missing")
        if r.pack.reach is None:
            return _unknown(a, "OBSERVATION_UNAVAILABLE", f"{r.pack.name} has no declared reach")
        base = world.data.xpos[r.root_body]
        dist = float(np.linalg.norm(p - base))
        lim = r.pack.reach + float(a.get("margin_m", 0.0))
        return ok if dist <= lim else _fail(a, "UNREACHABLE", f"<= {lim:.2f} m from the base", round(dist, 3))
    if op == "no_penetration":
        worst = max((o["depth_mm"] for o in world.overlaps(depth=1e-4)), default=0.0)
        lim = float(a.get("max_mm", 2.0))
        return ok if worst <= lim else _fail(a, "PENETRATION_PRESENT", f"<= {lim} mm", worst)
    return _unknown(a, "UNKNOWN_OP", f"unknown op {op!r}")


def referenced(node: Mapping[str, Any]) -> set[str]:
    out: set[str] = set()
    for k in ("entity", "a", "b", "robot"):
        v = node.get(k)
        if isinstance(v, str) and v not in ("any", "world"):
            out.add(v)
    if isinstance(node.get("point"), str):
        out.add(node["point"])
    for t in node.get("terms") or []:
        out |= referenced(t)
    if node.get("term"):
        out |= referenced(node["term"])
    return out


def gate(world: Any, goal: Mapping[str, Any]) -> dict[str, Any]:
    """Before work starts: refuse a goal naming missing entities; flag one that already holds."""
    known = {e.id for e in world.stage.entities}
    missing = sorted(referenced(goal) - known)
    if missing:
        return {"ok": False, "reason": "ENTITY_MISSING", "missing": missing}
    r = evaluate(world, goal)
    if r["value"] == TRUE:
        return {"ok": False, "reason": "SPEC_ALREADY_SATISFIED", "note": "the goal holds before anything was done"}
    return {"ok": True, "now": r["value"], "failures": r["failures"]}


def check(world: Any, goal: Mapping[str, Any]) -> dict[str, Any]:
    r = evaluate(world, goal)
    return {"value": r["value"], "satisfied": r["value"] == TRUE, "failures": r["failures"][:12],
            "reasons": sorted({f["reason"] for f in r["failures"]}), "step": world.step,
            "generation": world.generation}


__all__ = ["check", "evaluate", "gate", "referenced"]
