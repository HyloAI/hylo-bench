"""Showcase recordings of the stage world (RFC-0008): each scene is built and run through the `mujoco.world` tool
surface (the same calls a model makes), recorded as a video, and its receipts and checks are written to results.json.

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH=$HOME/.local/mesa/root/usr/lib/x86_64-linux-gnu \\
        uv run python bench/adapters/hylo-env-mujoco/examples/world/showcase.py [--out runs/world-showcase] [NAME ...]

Scenes: quadrupeds (Go2, Go1, A1, Spot trotting side by side), humanoids (G1 and H1 with pretrained walking
policies), drones (Crazyflie and X2 flying waypoints), arms (a Panda picks a cube onto a plate, a UR5e with a mounted
Robotiq gripper, a TidyBot moving its base), crowd (people walking routes around crates while a Go2 crosses),
dataset (an export: RGB, depth and segmentation frames with camera poses).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

from hylo.envs.mujoco.compose import look_at
from hylo.envs.mujoco.world_tools import Workspace


def ent(id_, **kw):
    return {"op": "add", "entity": {"id": id_, **kw}}


def robot(id_, pack, x, y, yaw=0.0, base="free", z=0.0, **extra):
    q = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    return ent(id_, robot={"pack": pack, "base": base, **extra}, pose={"pos": [x, y, z], "quat": q})


def box(id_, pos, half, rgba=(0.6, 0.6, 0.62, 1), dynamic=False, mass=None):
    body = {"dynamic": dynamic, **({"mass": mass} if mass else {})}
    return ent(id_, shape={"type": "box", "size": list(half), "rgba": list(rgba)}, body=body, pose={"pos": list(pos)})


def camera(id_, eye, target, w=640, h=480):
    return ent(id_, camera={"width": w, "height": h}, pose={"pos": list(eye), "quat": list(look_at(eye, target))})


class Scene:
    def __init__(self, out: Path, name: str) -> None:
        if (out / "stages" / name).exists():
            shutil.rmtree(out / "stages" / name)
        self.ws = Workspace(out / "stages", name, out=out / "videos" / name)
        self.s, self.log = self.ws.surface, []

    def call(self, tool, **a):
        r = self.s.call(tool, a)
        if not r.ok:
            raise RuntimeError(f"{tool}: {r.error}")
        self.log.append({"tool": tool, "result": r.result})
        return r.result

    def build(self, ops):
        self.call("stage_edit", expected_revision=self.call("stage_show")["revision"], ops=ops)
        return self.call("world_build")["generation"]

    def act(self, gen, actions, record=None):
        acts = [{**a, "expected_generation": gen} for a in actions]
        return self.call("act", actions=acts, **({"record": record} if record else {}))


def quadrupeds(out: Path) -> dict:
    sc = Scene(out, "quadrupeds")
    names = ["go2", "go1", "a1", "spot"]
    g = sc.build([robot(n, n, 0, 1.2 - 0.8 * i) for i, n in enumerate(names)]
                 + [camera("cam", (2.2, 2.6, 1.5), (1.0, -0.2, 0.2))])
    r1 = sc.act(g, [{"robot": n, "kind": "walk", "forward": 0.8, "duration_s": 4} for n in names],
                record={"camera": "cam", "fps": 25})
    r2 = sc.act(g, [{"robot": n, "kind": "walk", "forward": 0.5, "turn": 0.6, "duration_s": 3} for n in names],
                record={"camera": "cam", "fps": 25})
    return {"videos": [r1["video"]["path"], r2["video"]["path"]],
            "receipts": {n: [r1["receipts"][i]["measured"], r2["receipts"][i]["measured"]] for i, n in enumerate(names)}}


def humanoids(out: Path) -> dict:
    sc = Scene(out, "humanoids")
    g = sc.build([robot("g1", "g1_legs", 0, 0.6), robot("h1", "h1_legs", 0, -0.6),
                  camera("cam", (4.2, -2.6, 1.5), (1.4, 0, 0.7))])
    cmd = [{"robot": "g1", "kind": "policy", "policy": "g1_walk", "command": [0.5, 0, 0], "duration_s": 5},
           {"robot": "h1", "kind": "policy", "policy": "h1_walk", "command": [0.5, 0, 0], "duration_s": 5}]
    r1 = sc.act(g, cmd, record={"camera": "cam", "fps": 25})
    turn = [{**c, "command": [0.3, 0, 0.6], "duration_s": 4} for c in cmd]
    r2 = sc.act(g, turn, record={"camera": "cam", "fps": 25})
    return {"videos": [r1["video"]["path"], r2["video"]["path"]],
            "receipts": {"g1": [r1["receipts"][0]["measured"], r2["receipts"][0]["measured"]],
                         "h1": [r1["receipts"][1]["measured"], r2["receipts"][1]["measured"]]}}


def drones(out: Path) -> dict:
    sc = Scene(out, "drones")
    g = sc.build([robot("cf", "crazyflie", -0.3, 0, z=0.1), robot("x2", "x2", 0.3, 0, z=0.1),
                  box("pad", (0, 0, 0.02), (0.6, 0.3, 0.02), (0.3, 0.32, 0.36, 1)),
                  camera("cam", (2.6, -2.4, 1.6), (0, 0, 0.8))])
    path_cf = [[-0.3, 0, 0.8], [-0.6, 0.5, 1.2], [0.0, 0.8, 1.0]]
    path_x2 = [[0.3, 0, 1.0], [0.8, -0.4, 1.4], [0.3, -0.8, 0.9]]
    rec = []
    for a, b in zip(path_cf, path_x2):
        rec.append(sc.act(g, [{"robot": "cf", "kind": "fly", "position": a, "duration_s": 8},
                              {"robot": "x2", "kind": "fly", "position": b, "duration_s": 4}],
                          record={"camera": "cam", "fps": 25}))
    return {"videos": [r["video"]["path"] for r in rec],
            "receipts": {"cf": [r["receipts"][0]["measured"] for r in rec],
                         "x2": [r["receipts"][1]["measured"] for r in rec]}}


def arms(out: Path) -> dict:
    sc = Scene(out, "arms")
    g = sc.build([
        box("table", (0.5, 0, 0.2), (0.35, 0.45, 0.2), (0.55, 0.42, 0.32, 1)),
        robot("panda", "panda", 0, 0, base="fixed", z=0.4),
        box("cube", (0.5, -0.15, 0.42), (0.02, 0.02, 0.02), (0.85, 0.15, 0.12, 1), dynamic=True, mass=0.05),
        ent("plate", shape={"type": "cylinder", "size": [0.07, 0.005], "rgba": [0.9, 0.9, 0.92, 1]},
            body={"dynamic": False}, pose={"pos": [0.5, 0.18, 0.405]}),
        robot("ur", "ur5e", 0.4, 1.2, base="fixed", yaw=-math.pi / 2),
        robot("hand", "robotiq_2f85", 0, 0, base="fixed", mount={"entity": "ur", "body": "wrist_3_link"}),
        robot("tb", "tidybot", -1.2, -0.6, base="fixed"),
        camera("cam", (1.8, -1.6, 1.5), (0.3, 0.2, 0.5))])
    down = [0, 1, 0, 0]
    steps = [("above", {"position": [0.5, -0.15, 0.55], "quat_wxyz": down}),
             ("down", {"position": [0.5, -0.15, 0.43], "quat_wxyz": down}), ("close", None),
             ("up", {"position": [0.5, -0.15, 0.6], "quat_wxyz": down}),
             ("over", {"position": [0.5, 0.18, 0.6], "quat_wxyz": down}),
             ("lower", {"position": [0.5, 0.18, 0.45], "quat_wxyz": down}), ("open", None),
             ("back", {"position": [0.4, 0.0, 0.65], "quat_wxyz": down})]
    receipts, videos = [], []
    for name, args in steps:
        if args is None:
            a = {"robot": "panda", "kind": "gripper", "command": "close" if name == "close" else "open"}
        else:
            a = {"robot": "panda", "kind": "tcp", **args, "duration_s": 1.2}
        side = []
        if name == "above":
            side = [{"robot": "ur", "kind": "joints", "targets": {"shoulder_pan": -1.0, "elbow": 1.2},
                     "duration_s": 2}, {"robot": "tb", "kind": "base", "x": 0.6, "y": 0.3, "yaw_deg": 45,
                                        "duration_s": 3}]
        if name == "close":
            side = [{"robot": "hand", "kind": "gripper", "command": "close"}]
        r = sc.act(g, [a, *side], record={"camera": "cam", "fps": 25})
        receipts.append({name: [x["measured"] for x in r["receipts"]]})
        videos.append(r["video"]["path"])
    check = sc.call("check", goal={"op": "all", "terms": [
        {"op": "near", "a": "cube", "b": "plate", "max_m": 0.06}, {"op": "settled", "entity": "cube", "max_speed": 0.05}]})
    return {"videos": videos, "receipts": receipts, "check": check}


def crowd(out: Path) -> dict:
    sc = Scene(out, "crowd")
    people = [ent(f"p{i}", human={"route": route, "speed": sp, "rgba": col, "dwell": 0.5},
                  pose={"pos": [route[0][0], route[0][1], 0]})
              for i, (route, sp, col) in enumerate([
                  ([[-2.5, -1.0], [2.5, -1.0]], 1.1, [0.35, 0.45, 0.65, 1]),
                  ([[2.5, 1.0], [-2.5, 1.0]], 1.0, [0.65, 0.35, 0.35, 1]),
                  ([[0.0, -2.5], [0.0, 2.5]], 0.9, [0.35, 0.6, 0.4, 1]),
                  ([[-2.0, 2.0], [2.0, -2.0]], 1.2, [0.6, 0.55, 0.3, 1])])]
    g = sc.build([box("crate1", (0.9, -0.2, 0.4), (0.3, 0.3, 0.4)), box("crate2", (-0.9, 0.3, 0.3), (0.25, 0.4, 0.3)),
                  robot("dog", "go1", -2.0, -2.0, yaw=math.pi / 4), *people,
                  camera("cam", (0.0, -5.0, 4.0), (0.0, 0.0, 0.3))])
    r = sc.act(g, [{"robot": "dog", "kind": "walk", "forward": 0.7, "duration_s": 8}], record={"camera": "cam", "fps": 20})
    adv = sc.call("advance", seconds=4, record={"camera": "cam", "fps": 20})
    collide = sc.call("check", goal={"op": "all", "terms": [{"op": "no_contact", "a": f"p{i}", "b": "any"}
                                                            for i in range(4)]})
    return {"videos": [r["video"]["path"], adv["video"]["path"]], "people": adv.get("people"),
            "dog": r["receipts"][0]["measured"], "people_touching_anything": not collide["satisfied"]}


def dataset(out: Path) -> dict:
    sc = Scene(out, "dataset")
    g = sc.build([box("table", (0.5, 0, 0.2), (0.35, 0.45, 0.2), (0.55, 0.42, 0.32, 1)),
                  robot("panda", "panda", 0, 0, base="fixed", z=0.4),
                  *[box(f"b{i}", (0.45 + 0.08 * (i % 2), -0.2 + 0.13 * i, 0.43), (0.025, 0.025, 0.025),
                        rgba, dynamic=True, mass=0.05)
                    for i, rgba in enumerate([(0.9, 0.2, 0.1, 1), (0.1, 0.5, 0.9, 1), (0.2, 0.75, 0.3, 1),
                                              (0.95, 0.8, 0.1, 1)])],
                  camera("top", (0.5, 0, 1.4), (0.5, 0, 0.4), 320, 240), camera("side", (1.4, -0.9, 0.9), (0.4, 0, 0.45),
                                                                               320, 240)])
    ex = sc.call("export", frames=6, every_s=0.4, name="tabletop", actions=[
        {"robot": "panda", "kind": "tcp", "position": [0.5, 0.0, 0.6], "quat_wxyz": [0, 1, 0, 0], "duration_s": 2,
         "expected_generation": g}])
    folder = Path(ex["folder"])
    rows = [json.loads(x) for x in (folder / "samples.jsonl").read_text().splitlines()]
    from PIL import Image

    tiles = []
    for r in [x for x in rows if x["camera"] == "top"][::2][:3]:
        rgb = np.asarray(Image.open(folder / r["rgb"]))
        d = np.load(folder / r["depth"])
        dn = (255 * (1 - np.clip((d - d.min()) / (np.ptp(d) + 1e-9), 0, 1))).astype(np.uint8)
        seg = np.asarray(Image.open(folder / r["seg"])).astype(int)
        pal = (np.array([[40, 40, 40]] + [[(37 * k) % 255, (91 * k) % 255, (173 * k) % 255] for k in range(1, 64)]))
        tiles.append(np.concatenate([rgb, np.repeat(dn[..., None], 3, 2), pal[seg % 64].astype(np.uint8)], 1))
    grid = np.concatenate(tiles, 0)
    Image.fromarray(grid).save(out / "videos" / "dataset_grid.jpg", quality=88)
    return {"folder": str(folder), "samples": len(rows), "grid": str(out / "videos" / "dataset_grid.jpg")}


SCENES = {"quadrupeds": quadrupeds, "humanoids": humanoids, "drones": drones, "arms": arms, "crowd": crowd,
          "dataset": dataset}


def concat(paths: list[str], dst: Path) -> None:
    import imageio.v2 as imageio

    frames = [f for p in paths for f in imageio.mimread(p, memtest=False)]
    imageio.mimwrite(dst, frames, fps=25, quality=7, macro_block_size=8)
    from PIL import Image

    Image.fromarray(frames[len(frames) // 2]).save(dst.with_suffix(".jpg"), quality=85)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--out", default="runs/world-showcase")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    res_path = out / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    for name in a.names or list(SCENES):
        print(f"== {name}", flush=True)
        r = SCENES[name](out)
        if r.get("videos"):
            concat(r["videos"], out / "videos" / f"world_{name}.mp4")
            r["video"] = str(out / "videos" / f"world_{name}.mp4")
        results[name] = r
        res_path.write_text(json.dumps(results, indent=1, default=str))
        print(json.dumps({k: v for k, v in r.items() if k != "videos"}, default=str)[:1500], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
