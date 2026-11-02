# hylo: worker-side (the simulator functions run in the LIBERO simulator venv: robosuite 1.4, mujoco)
"""Perturbed LIBERO tasks: the official scenes with one thing changed, to measure what an agent relies on.

A perturbation is declared on the task (`TaskSpec.params["perturb"]`, from the `libero` benchmark's `perturb` param),
so one environment serves every variant and each variant has its own task id (`libero/<suite>/<name>+<kind>`):

```text
  position   the object the goal moves trades start positions with another movable object of the scene
  task       a different goal in the same scene: another movable object goes where the original goal put its
             object (same relation, on / in); the instruction is rewritten to match; BDDL written to the cache
  language   the instruction is replaced by a paraphrase (paraphrases.json, chosen by seed)
  object     every movable object is recoloured (hue shifted, by seed); shapes and physics unchanged
  camera     the agentview camera is moved about the point it looks at (yaw, pitch, distance), by seed
  noise      Gaussian pixel noise and a glare spot on every RGB image the agent sees (depth unchanged)
  robot_init the arm starts from its joint values plus a seeded offset (up to 0.15 rad per joint)
```

Kinds combine (`perturb: [position, language]`). `task` needs the variant index (`variants.json`, generated in the
simulator venv with `python -m hylo.envs.libero.perturb`), which also lists the movable objects of every task.
The functions that touch the simulator take robosuite / MuJoCo objects and run in the simulator venv only.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

KINDS = ("position", "task", "language", "object", "camera", "noise", "robot_init")
VARIANTS = Path(__file__).with_name("variants.json")
PARAPHRASES = Path(__file__).with_name("paraphrases.json")
GOAL_ATOM = re.compile(r"\((On|In)\s+([^\s()]+)\s+([^\s()]+)\)")


def kinds(spec: Any) -> tuple[str, ...]:
    """Normalise a perturb param (a kind, a list of kinds, or None) and check it."""
    if not spec:
        return ()
    out = (spec,) if isinstance(spec, str) else tuple(spec)
    bad = [k for k in out if k not in KINDS]
    if bad:
        raise ValueError(f"unknown perturbations {bad}; known: {list(KINDS)}")
    return out


def rng(seed: int, kind: str) -> np.random.Generator:
    h = int(hashlib.sha256(f"{seed}:{kind}".encode()).hexdigest()[:12], 16)
    return np.random.default_rng(h)


# ------------------------------------------------------------------------------------------------ BDDL (pure text)


def bddl_section(text: str, name: str) -> list[str]:
    """Object names declared in a BDDL section (:objects or :fixtures)."""
    m = re.search(rf"\(:{name}\s*(.*?)\)\s*\n", text, re.S)
    if not m:
        return []
    names = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if " - " in line:
            names += line.split(" - ")[0].split()
    return names


RECEPTACLES = ("plate", "basket", "tray", "caddy", "rack", "drainer", "storage_box")
DRAWERS = {"top_region": "top drawer", "middle_region": "middle drawer", "bottom_region": "bottom drawer"}


def region_label(target: str, labels: dict[str, str]) -> str | None:
    """How an instruction names a goal target: 'flat_stove_1_cook_region' -> 'stove', 'wooden_cabinet_1_top_side'
    -> 'top of the wooden cabinet', 'white_cabinet_1_bottom_region' -> 'bottom drawer of the white cabinet',
    'desk_caddy_1_back_contain_region' -> 'back compartment of the caddy'. None for a target that is not part of
    a named object or fixture (a region of the table)."""
    m = re.match(r"(.+?_\d+)(?:_(.+))?$", target)
    if not m:
        return None
    base, rest = m.group(1), m.group(2)
    label = labels.get(base) or base.replace("_", " ")
    if rest == "top_side":
        return f"top of the {label}"
    if rest in DRAWERS and "cabinet" in base:
        return f"{DRAWERS[rest]} of the {label}"
    if rest and rest.endswith("_contain_region") and rest != "contain_region":
        return f"{rest.split('_')[0]} compartment of the {label}"
    return label


def task_variant(bddl: str, language: str, labels: dict[str, str]) -> dict[str, Any] | None:
    """A different goal in the same scene, or None unless the goal is exactly one On / In of a movable object onto
    or into a named object or fixture. The replacement is the first movable object (by name) of another category
    than the original that is not itself a receptacle (a plate, a basket, ...)."""
    goal = re.search(r"\(:goal\s*(.*?)\)\s*\)\s*$", bddl, re.S)
    if not goal:
        return None
    atoms = GOAL_ATOM.findall(goal.group(1))
    if len(atoms) != 1 or len(re.findall(r"\((?!And\b)[A-Za-z]+\s", goal.group(1))) != 1:
        return None                                       # other conditions (Open, TurnOn, ...) would be lost
    rel, obj, target = atoms[0]
    movable = bddl_section(bddl, "objects")
    if obj not in movable or region_label(target, labels) is None:
        return None
    cat = re.sub(r"_\d+$", "", obj)
    others = sorted(o for o in movable if re.sub(r"_\d+$", "", o) != cat and o != target
                    and not target.startswith(o + "_") and not any(r in o for r in RECEPTACLES))
    if not others:
        return None
    new = others[0]
    verb = "on" if rel == "On" else "in"
    new_language = f"put the {labels.get(new, new)} {verb} the {region_label(target, labels)}"
    new_goal = f"(And ({rel} {new} {target}))"
    text = bddl.replace(goal.group(0), f"(:goal\n    {new_goal}\n  )\n)")
    text = re.sub(r"\(:language\s+[^)]*\)", f"(:language {new_language})", text)
    return {"object": new, "replaces": obj, "relation": rel, "target": target, "language": new_language,
            "goal": new_goal, "bddl": text}


# ------------------------------------------------------------------------------------------------ simulator side


def free_joint_adr(m: Any, body_name: str) -> int | None:
    import mujoco

    b = m.body(body_name).id
    for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
        if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return int(m.jnt_qposadr[j])
    return None


def swap_positions(env: Any, moved: str, movable: Sequence[str], seed: int) -> dict[str, Any] | None:
    """Trade the x, y of `moved` and another movable object (seeded); heights kept, then forward kinematics."""
    import mujoco

    e = env.env
    m, d = e.sim.model._model, e.sim.data._data
    roots = {n: e.objects_dict[n].root_body for n in movable if n in e.objects_dict}
    adr = {n: free_joint_adr(m, r) for n, r in roots.items()}
    adr = {n: a for n, a in adr.items() if a is not None}
    if moved not in adr or len(adr) < 2:
        return None
    others = sorted(n for n in adr if n != moved)
    other = others[int(rng(seed, "position").integers(len(others)))]
    a, b = adr[moved], adr[other]
    pa, pb = d.qpos[a:a + 2].copy(), d.qpos[b:b + 2].copy()
    d.qpos[a:a + 2], d.qpos[b:b + 2] = pb, pa
    mujoco.mj_forward(m, d)
    return {"swapped": [moved, other]}


def recolour(env: Any, movable: Sequence[str], seed: int, saved: dict[int, np.ndarray]) -> dict[str, Any]:
    """Shift the hue of every visible geom of the movable objects (and drop their textures' influence by tinting).
    `saved` collects original rgba values so `restore_colours` can undo it on the next reset."""
    import colorsys

    e = env.env
    m = e.sim.model._model
    r = rng(seed, "object")
    out = {}
    for n in movable:
        o = e.objects_dict.get(n)
        if o is None:
            continue
        root = m.body(o.root_body).id
        bodies = {root} | {b for b in range(m.nbody) if _under(m, b, root)}
        shift = float(r.uniform(0.25, 0.75))
        tint = np.array(colorsys.hsv_to_rgb(float(r.uniform()), 0.7, 0.9))
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in bodies and m.geom_group[g] in (0, 1):     # visual groups
                saved.setdefault(g, m.geom_rgba[g].copy())
                rgb = m.geom_rgba[g][:3]
                h, s, v = colorsys.rgb_to_hsv(*[float(c) for c in rgb])
                new = np.array(colorsys.hsv_to_rgb((h + shift) % 1.0, max(s, 0.5), v))
                m.geom_rgba[g][:3] = 0.5 * new + 0.5 * tint
                if m.geom_matid[g] >= 0:
                    saved.setdefault(-1 - g, np.array([m.geom_matid[g]], float))
                    m.geom_matid[g] = -1                                     # textures off: the rgba shows
        out[n] = np.round(tint, 2).tolist()
    return {"recoloured": out}


def restore_colours(env: Any, saved: dict[int, np.ndarray]) -> None:
    m = env.env.sim.model._model
    for g, v in saved.items():
        if g >= 0:
            m.geom_rgba[g] = v
        else:
            m.geom_matid[-1 - g] = int(v[0])
    saved.clear()


def _under(m: Any, b: int, root: int) -> bool:
    while b > 0:
        if b == root:
            return True
        b = int(m.body_parentid[b])
    return False


def move_camera(env: Any, cam: str, seed: int, saved: dict[str, Any], *, yaw_deg: float = 15.0,
                pitch_deg: float = 8.0, dolly: float = 0.12) -> dict[str, Any]:
    """Rotate a fixed camera about the point of its view axis nearest the robot base, and move it closer or further."""
    import mujoco
    from scipy.spatial.transform import Rotation as Rot

    e = env.env
    m, d = e.sim.model._model, e.sim.data._data
    cid = m.camera(cam).id
    if cam not in saved:
        saved[cam] = (m.cam_pos[cid].copy(), m.cam_quat[cid].copy())
    pos0, quat0 = saved[cam]
    r = rng(seed, "camera")
    yaw = math.radians(float(r.uniform(-yaw_deg, yaw_deg)))
    pitch = math.radians(float(r.uniform(-pitch_deg, pitch_deg)))
    scale = 1.0 + float(r.uniform(-dolly, dolly))
    R0 = Rot.from_quat([quat0[1], quat0[2], quat0[3], quat0[0]])          # MuJoCo wxyz -> scipy xyzw
    fwd = R0.apply([0.0, 0.0, -1.0])
    base = d.xpos[m.body("robot0_base").id] if "robot0_base" in [m.body(i).name for i in range(m.nbody)] \
        else m.stat.center
    centre = pos0 + float(np.dot(np.asarray(base) - pos0, fwd)) * fwd        # the view axis point nearest the robot
    rel = (pos0 - centre) * scale
    side = np.cross(fwd, [0.0, 0.0, 1.0])
    side = side / (np.linalg.norm(side) or 1.0)
    turn = Rot.from_rotvec([0.0, 0.0, yaw]) * Rot.from_rotvec(side * pitch)
    m.cam_pos[cid] = centre + turn.apply(rel)
    q = (turn * R0).as_quat()
    m.cam_quat[cid] = [q[3], q[0], q[1], q[2]]
    mujoco.mj_forward(m, d)
    return {"camera": {"yaw_deg": round(math.degrees(yaw), 1), "pitch_deg": round(math.degrees(pitch), 1),
                       "distance_scale": round(scale, 3)}}


def restore_camera(env: Any, saved: dict[str, Any]) -> None:
    m = env.env.sim.model._model
    for cam, (p, q) in saved.items():
        cid = m.camera(cam).id
        m.cam_pos[cid], m.cam_quat[cid] = p, q


def noisy(img: np.ndarray, seed: int, step: int, sigma: float = 10.0) -> np.ndarray:
    """Gaussian pixel noise (sigma in 0..255 units, new every step) plus a fixed glare spot (per episode)."""
    h, w = img.shape[:2]
    r = rng(seed * 100003 + step, "noise")
    out = img.astype(np.float32) + r.normal(0.0, sigma, img.shape).astype(np.float32)
    g = rng(seed, "glare")
    cy, cx, rad = g.uniform(0.15, 0.6) * h, g.uniform(0.2, 0.8) * w, g.uniform(0.08, 0.16) * max(h, w)
    yy, xx = np.mgrid[0:h, 0:w]
    spot = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rad ** 2))[..., None]
    out = out * (1 - 0.6 * spot) + 255.0 * 0.6 * spot
    return np.clip(out, 0, 255).astype(np.uint8)


def jitter_joints(q: np.ndarray, seed: int, lower: Sequence[float], upper: Sequence[float],
                  max_rad: float = 0.15) -> np.ndarray:
    r = rng(seed, "robot_init")
    return np.clip(np.asarray(q, float) + r.uniform(-max_rad, max_rad, len(q)), lower, upper)


def paraphrase(language: str, seed: int, table: dict[str, list[str]] | None = None) -> str:
    table = table if table is not None else (json.loads(PARAPHRASES.read_text()) if PARAPHRASES.is_file() else {})
    options = table.get(language) or []
    if not options:
        raise KeyError(f"no paraphrases for {language!r} in {PARAPHRASES.name}")
    return options[int(rng(seed, "language").integers(len(options)))]


def variants() -> dict[str, Any]:
    return json.loads(VARIANTS.read_text()) if VARIANTS.is_file() else {}


def _generate(suites: Iterable[str] = ("libero_spatial", "libero_object", "libero_goal", "libero_10")) -> None:
    """Write variants.json: per task, the movable objects, the goal's moved object and the task variant."""
    from libero.libero import benchmark, get_libero_path

    from .robot import object_label

    out: dict[str, dict[str, Any]] = {}
    for suite in suites:
        b = benchmark.get_benchmark_dict()[suite]()
        rows = {}
        for i in range(b.n_tasks):
            t = b.get_task(i)
            bddl = (Path(get_libero_path("bddl_files")) / t.problem_folder / t.bddl_file).read_text()
            movable = bddl_section(bddl, "objects")
            labels = {n: object_label(n) for n in movable + bddl_section(bddl, "fixtures")}
            goal = re.search(r"\(:goal\s*(.*?)\)\s*\)\s*$", bddl, re.S)
            atoms = GOAL_ATOM.findall(goal.group(1)) if goal else []
            v = task_variant(bddl, t.language, labels)
            rows[str(i)] = {"name": t.name, "language": t.language, "movable": movable,
                            "moved": [a[1] for a in atoms if a[1] in movable],
                            "task": None if v is None else {k: v[k] for k in ("object", "replaces", "relation",
                                                                              "target", "language", "goal")}}
        out[suite] = rows
    VARIANTS.write_text(json.dumps(out, indent=1))
    print(f"wrote {VARIANTS}")


def variant_bddl(suite: str, task_index: int) -> Path:
    """The task variant's BDDL file, written once into the asset cache (simulator venv)."""
    from libero.libero import benchmark, get_libero_path

    from .robot import object_label
    from .sim import cache_root

    b = benchmark.get_benchmark_dict()[suite]()
    t = b.get_task(task_index)
    src = Path(get_libero_path("bddl_files")) / t.problem_folder / t.bddl_file
    bddl = src.read_text()
    labels = {n: object_label(n) for n in bddl_section(bddl, "objects") + bddl_section(bddl, "fixtures")}
    v = task_variant(bddl, t.language, labels)
    if v is None:
        raise ValueError(f"{suite} task {task_index} has no task variant (its goal is not one On / In)")
    folder = cache_root() / "libero" / "variants"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{suite}_{task_index}_task_{hashlib.sha256(v['bddl'].encode()).hexdigest()[:10]}.bddl"
    if not path.is_file():
        path.write_text(v["bddl"])
    return path


if __name__ == "__main__":  # pragma: no cover
    _generate()


__all__ = ["KINDS", "kinds", "noisy", "paraphrase", "region_label", "swap_positions", "task_variant", "variants"]
