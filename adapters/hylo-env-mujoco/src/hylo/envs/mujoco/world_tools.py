"""The stage world as typed tools (RFC-0008 part 5): what a model (or a person through the Studio, or a coding agent
over MCP) calls to build a scene, run robots in it, look at it and check it.

Registered as the surface `mujoco.world` (entry point group hylo.surfaces). One `Workspace` holds a stage store, the
stage being worked on and the world built from one of its revisions:

```text
  stage      stage_show, stage_history (read); stage_edit (a patch against the revision you read: compare and swap),
             stage_restore (act)
  packs      packs: the robots a stage can name, with their controls and status
  world      world_build (compile the newest revision: a new generation), world_reset (act); world_state (read)
  robots     act: one action per robot, or several robots in the same steps; every action names the generation
             it was decided on and comes back with a measured receipt
  time       advance: let time pass (people walk, objects settle), optionally recording a video
  people     people: goto / stop / resume a person
  eyes       capture (camera images with depth and segmentation by entity), view (a free viewpoint),
             pixel_to_world (a pixel of a capture as a world point and the entity it shows)
  checks     check (a goal predicate: satisfied, or which atoms failed and why), gate (before starting: refuse a goal
             naming missing entities, flag one already satisfied), overlaps (penetrations at the start)
  data       export (camera frames over time as a dataset: samples.jsonl, rgb png, depth npy, seg png, calibration)
```
Images (capture, view) come back to vision models; files go under the workspace's output folder.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hylo.core.stage import Patch, Stage, StageConflict, StageRefused, StageStore
from hylo.core.tooling import Help, Surface, ToolDef, ToolError, ToolOutput, obj

from . import checks as C
from .packs import PACKS, describe

STR = {"type": "string", "description": "text"}


def vec(n: int, what: str) -> dict[str, Any]:
    return {"type": "array", "items": {"type": "number", "description": "component"}, "minItems": n, "maxItems": n,
            "description": what}


class Workspace:
    def __init__(self, root: str | Path = "runs/stages", stage: str = "lab", out: str | Path | None = None,
                 create: bool = True) -> None:
        self.store = StageStore(root)
        self.name = stage
        if not self.store.exists(stage):
            if not create:
                raise KeyError(f"no stage {stage!r} in {root}")
            self.store.create(Stage(name=stage), note="created by the workspace")
        self.out = Path(out or Path(root) / stage / "outputs")
        self.out.mkdir(parents=True, exist_ok=True)
        self.world: Any = None
        self.captures: dict[str, dict[str, Any]] = {}
        self._n = 0
        self.surface = self._build()

    # ------------------------------------------------------------------ helpers
    @property
    def stage(self) -> Stage:
        return self.store.load(self.name)

    def _world(self) -> Any:
        if self.world is None:
            raise ToolError("no_world", "no world yet: call world_build", retry="never")
        return self.world

    def _id(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    # ------------------------------------------------------------------ stage
    def stage_show(self, a: dict[str, Any]) -> dict[str, Any]:
        st = self.stage
        ents = []
        for e in st.ordered():
            row: dict[str, Any] = {"id": e.id, "kind": e.kind, "pos": [round(v, 4) for v in e.pose.pos]}
            if e.parent:
                row["parent"] = e.parent
            if e.label:
                row["label"] = e.label
            if e.robot:
                row["pack"] = e.robot.pack
                row["base"] = e.robot.base
                if e.robot.mount:
                    row["mount"] = e.robot.mount.model_dump(exclude_none=True)
            if e.shape:
                row["shape"] = {"type": e.shape.type, "size": list(e.shape.size)}
            if e.body:
                row["dynamic"] = e.body.dynamic
            if e.locked:
                row["locked"] = True
            ents.append(row)
        full = a.get("entity")
        out: dict[str, Any] = {"stage": st.name, "revision": st.revision, "entities": ents}
        if full:
            out["entity_json"] = st.get(full).model_dump(mode="json", exclude_none=True)
        if self.world is not None:
            out["world_built_from_revision"] = self.world.stage.revision
        return out

    def stage_history(self, a: dict[str, Any]) -> dict[str, Any]:
        return {"history": self.store.history(self.name)[-int(a.get("last") or 20):]}

    def stage_edit(self, a: dict[str, Any]) -> dict[str, Any]:
        ops = json.loads(json.dumps(a["ops"]))
        for op in ops:                                   # `kind` is what stage_show reports, not a field to set
            if isinstance(op, dict) and isinstance(op.get("entity"), dict):
                op["entity"].pop("kind", None)
            if isinstance(op, dict) and isinstance(op.get("set"), dict):
                op["set"].pop("kind", None)
        try:
            p = Patch.model_validate({"expected_revision": a["expected_revision"], "ops": ops,
                                      "note": a.get("note", ""), "author": "model"})
        except ValueError as exc:
            raise ToolError("bad_patch", _brief(exc), field="ops") from exc
        try:
            st = self.store.commit(self.name, p)
        except StageConflict as exc:
            raise ToolError("conflict", str(exc), field="expected_revision", retry="changed_arguments") from exc
        except StageRefused as exc:
            raise ToolError("refused", str(exc)[:800], field="ops") from exc
        return {"revision": st.revision, "entities": len(st.entities),
                "note": "the world still runs the old revision until world_build"}

    def stage_restore(self, a: dict[str, Any]) -> dict[str, Any]:
        try:
            st = self.store.restore(self.name, int(a["revision"]), author="model")
        except KeyError as exc:
            raise ToolError("unknown_revision", str(exc), field="revision") from exc
        return {"revision": st.revision, "restored": a["revision"]}

    def packs(self, a: dict[str, Any]) -> dict[str, Any]:
        fam = a.get("family")
        return {"packs": [describe(n) for n, p in PACKS.items() if not fam or p.family == fam]}

    def policies(self, a: dict[str, Any]) -> dict[str, Any]:
        from . import policies as P

        out: dict[str, Any] = {"policies": P.describe()}
        if a.get("inspect"):
            spec = P.policy(a["inspect"])
            net, info = P.network(spec.name)
            out["inspect"] = {**info, "joints": list(spec.joints), "observation": P.observation_size(spec)}
            if a.get("robot") and self.world is not None and a["robot"] in self.world.robots:
                out["match"] = P.match(spec, self.world, self.world.robots[a["robot"]]) or "matches"
        return out

    # ------------------------------------------------------------------ world
    def world_build(self, a: dict[str, Any]) -> dict[str, Any]:
        from .world import World

        st = self.stage
        if self.world is not None:
            self.world.close()
        t0 = time.monotonic()
        try:
            self.world = World(st, self.store.root / self.name)
        except Exception as exc:  # noqa: BLE001 - a stage that does not compile is an answer, not a crash
            self.world = None
            raise ToolError("build_failed", f"{type(exc).__name__}: {exc}"[:800], retry="changed_arguments") from exc
        self.captures.clear()
        return {"generation": self.world.generation, "revision": st.revision, "seconds": round(time.monotonic() - t0, 2),
                "robots": {k: {"pack": r.pack.name, "controls": list(r.pack.controls), "actuators": r.actuators}
                           for k, r in self.world.robots.items()},
                "cameras": list(self.world.cameras), "people": list(self.world.humans.ids) if self.world.humans else [],
                "fixed_in_place": [e.id for e in st.entities if (e.shape or e.mesh) and not (e.body and e.body.dynamic)],
                "movable": [e.id for e in st.entities if (e.shape or e.mesh) and e.body and e.body.dynamic],
                "overlaps": self.world.overlaps(),
                "note": "shapes and meshes without body {dynamic: true} are fixed in place: nothing can move them"}

    def world_reset(self, a: dict[str, Any]) -> dict[str, Any]:
        return {"generation": self._world().reset()}

    def world_state(self, a: dict[str, Any]) -> dict[str, Any]:
        w = self._world()
        out = w.state()
        out["stage_revision_now"] = self.stage.revision
        if out["stage_revision_now"] != w.stage.revision:
            out["note"] = "the stage has newer revisions than the world: world_build to use them"
        return out

    def act(self, a: dict[str, Any]) -> dict[str, Any]:
        w = self._world()
        actions = a["actions"]
        for x in actions:
            x.setdefault("action_id", self._id("a"))
        rec = self._recorder(a.get("record"))
        receipts = w.act_batch(actions, on_step=rec.step if rec else None)
        out: dict[str, Any] = {"receipts": receipts, "generation": w.generation}
        if rec:
            out["video"] = rec.finish()
        return out

    def advance(self, a: dict[str, Any]) -> dict[str, Any]:
        w = self._world()
        rec = self._recorder(a.get("record"))
        n = w.advance(float(a["seconds"]), on_step=rec.step if rec else None)
        out: dict[str, Any] = {"steps": n, "time_s": round(float(w.data.time), 3)}
        if w.humans is not None:
            out["people"] = w.humans.state()
        if rec:
            out["video"] = rec.finish()
        return out

    def people(self, a: dict[str, Any]) -> dict[str, Any]:
        w = self._world()
        if w.humans is None or a["person"] not in w.humans.people:
            raise ToolError("unknown_person", f"no person {a['person']!r}", field="person")
        cmd = a["command"]
        if cmd == "goto":
            if a.get("to") is None:
                raise ToolError("bad_arguments", "goto needs `to` [x, y]", field="to")
            w.humans.goto(a["person"], a["to"])
        elif cmd == "stop":
            w.humans.stop(a["person"])
        else:
            w.humans.resume(a["person"])
        return {"people": w.humans.state()}

    # ------------------------------------------------------------------ eyes
    def capture(self, a: dict[str, Any]) -> ToolOutput:
        w = self._world()
        if not w.cameras:
            raise ToolError("no_cameras", "the stage has no camera entities: add one with stage_edit (camera)",
                            retry="never")
        names = a.get("cameras") or list(w.cameras)
        try:
            cap = w.capture(names)
        except KeyError as exc:
            raise ToolError("unknown_camera", str(exc), field="cameras") from exc
        cid = self._id("c")
        self.captures[cid] = cap
        meta = {}
        for n in names:
            c = cap[n]
            seen = sorted({cap["_labels"]["labels"][i] for i in np.unique(c["seg"]).tolist() if i > 0}) \
                if "seg" in c else []
            meta[n] = {"size": [int(c["rgb"].shape[1]), int(c["rgb"].shape[0])],
                       "position": np.round(c["T_world_cam"][:3, 3], 3).tolist(),
                       "looking_along": np.round(c["T_world_cam"][:3, 2], 3).tolist(),
                       "focal_px": round(float(c["K"][0, 0]), 1), "entities_seen": seen}
        return ToolOutput({"capture": cid, "step": w.step, "generation": w.generation, "cameras": meta},
                          images=[cap[n]["rgb"] for n in names], image_notes=[f"{n} ({cid})" for n in names])

    def view(self, a: dict[str, Any]) -> ToolOutput:
        w = self._world()
        img = w.view(lookat=a.get("lookat") or (0, 0, 0.5), distance=float(a.get("distance") or 3.0),
                     azimuth=float(a.get("azimuth") if a.get("azimuth") is not None else 135),
                     elevation=float(a.get("elevation") if a.get("elevation") is not None else -25))
        return ToolOutput({"step": w.step}, images=[img], image_notes=["free view"])

    def pixel_to_world(self, a: dict[str, Any]) -> dict[str, Any]:
        cap = self.captures.get(a["capture"])
        if cap is None:
            raise ToolError("unknown_capture", f"no capture {a['capture']!r}", field="capture")
        c = cap.get(a["camera"])
        if c is None or "depth" not in c:
            raise ToolError("no_depth", f"camera {a['camera']} was not captured with depth", field="camera")
        H, W = c["depth"].shape
        K, T = c["K"], c["T_world_cam"]
        out = []
        for u, v in a["pixels"]:
            if not (0 <= u < W and 0 <= v < H):
                out.append({"pixel": [u, v], "error": "outside the image"})
                continue
            z = float(c["depth"][int(v), int(u)])
            p = T @ np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z, 1.0])
            ent = cap["_labels"]["labels"][int(c["seg"][int(v), int(u)])] if "seg" in c else None
            out.append({"pixel": [u, v], "world": np.round(p[:3], 4).tolist(), "depth": round(z, 4), "entity": ent})
        return {"capture": a["capture"], "points": out}

    def lidar(self, a: dict[str, Any]) -> dict[str, Any]:
        """Range scan from a camera entity's pose: a grid of rays over its field of view (or a full horizontal ring)."""
        import mujoco

        w = self._world()
        cam = a["camera"]
        if cam not in w.cameras:
            raise ToolError("unknown_camera", f"no camera {cam!r}; cameras: {list(w.cameras)}", field="camera")
        m, d = w.model, w.data
        cid = m.camera(cam).id
        R = d.cam_xmat[cid].reshape(3, 3)
        origin = d.cam_xpos[cid].copy()
        rows, cols = int(a.get("rows") or 16), int(a.get("cols") or 64)
        max_m = float(a.get("max_m") or 10.0)
        if a.get("ring"):
            az = np.linspace(-np.pi, np.pi, cols, endpoint=False)
            el = np.radians(np.linspace(-15, 15, rows))
            dirs = [np.array([np.cos(e) * np.cos(t), np.cos(e) * np.sin(t), np.sin(e)]) for e in el for t in az]
        else:
            half = math.radians(float(m.cam_fovy[cid])) / 2
            dirs = []
            for v in np.linspace(-half, half, rows):
                for u in np.linspace(-half, half, cols):
                    dirs.append(R @ np.array([math.tan(u), math.tan(v), -1.0]))
        gid = np.zeros(1, np.int32)
        pts, hits, ents = [], 0, {}
        for v in dirs:
            v = v / np.linalg.norm(v)
            dist = mujoco.mj_ray(m, d, origin, v, None, 1, -1, gid)
            if 0 <= dist <= max_m:
                hits += 1
                pts.append(origin + dist * v)
                e = w.entity_of_body(int(m.geom_bodyid[gid[0]])) or "floor"
                ents[e] = ents.get(e, 0) + 1
        out: dict[str, Any] = {"camera": cam, "rays": len(dirs), "hits": hits, "hits_by_entity": ents,
                               "nearest_m": round(float(min((np.linalg.norm(p - origin) for p in pts), default=-1)), 3)}
        if a.get("save"):
            path = self.out / f"{self._id('scan')}.npy"
            np.save(path, np.asarray(pts, np.float32))
            out["points_file"] = str(path)
        return out

    def camera_fit(self, a: dict[str, Any]) -> dict[str, Any]:
        """A camera pose from 3D points and the pixels they appear at (PnP: linear estimate, then refinement)."""
        X = np.asarray(a["points"], float)
        uv = np.asarray(a["pixels"], float)
        if len(X) != len(uv) or len(X) < 6:
            raise ToolError("bad_arguments", "give at least 6 points and the same number of pixels", field="points")
        f = float(a["focal_px"])
        cx, cy = float(a.get("cx", 0.0)), float(a.get("cy", 0.0))
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        T, rms = pnp(X, uv, K)
        q = np.zeros(4)
        import mujoco

        mujoco.mju_mat2Quat(q, T[:3, :3].ravel())
        return {"T_world_cam": np.round(T, 5).tolist(), "position": np.round(T[:3, 3], 4).tolist(),
                "quat_wxyz_opencv": np.round(q, 5).tolist(), "reprojection_rms_px": round(rms, 3),
                "frame": "OpenCV camera (x right, y down, z forward) in the world"}

    # ------------------------------------------------------------------ checks
    def check(self, a: dict[str, Any]) -> dict[str, Any]:
        return C.check(self._world(), a["goal"])

    def gate(self, a: dict[str, Any]) -> dict[str, Any]:
        return C.gate(self._world(), a["goal"])

    def overlaps(self, a: dict[str, Any]) -> dict[str, Any]:
        return {"overlaps": self._world().overlaps(depth=float(a.get("depth_mm") or 1.0) / 1000)}

    # ------------------------------------------------------------------ data
    def _recorder(self, spec: Mapping[str, Any] | None) -> "Recorder | None":
        if not spec:
            return None
        return Recorder(self._world(), self.out / f"{self._id('video')}.mp4", spec.get("camera"),
                        int(spec.get("fps") or 25), spec.get("view"))

    def export(self, a: dict[str, Any]) -> dict[str, Any]:
        w = self._world()
        names = a.get("cameras") or list(w.cameras)
        if not names:
            raise ToolError("no_cameras", "no cameras to export", retry="never")
        folder = self.out / (a.get("name") or self._id("dataset"))
        (folder / "frames").mkdir(parents=True, exist_ok=True)
        every = max(1, w._n(float(a.get("every_s") or 0.5)))
        frames = int(a.get("frames") or 4)
        from PIL import Image

        rows = []
        calib: dict[str, Any] = {}

        def grab(world: Any) -> None:
            if (world.step % every) or len(rows) >= frames * len(names):
                return
            cap = world.capture(names)
            for n in names:
                c = cap[n]
                k = f"{world.step:07d}_{n}"
                Image.fromarray(c["rgb"]).save(folder / "frames" / f"{k}_rgb.png")
                np.save(folder / "frames" / f"{k}_depth.npy", c["depth"])
                Image.fromarray(c["seg"].astype(np.uint16)).save(folder / "frames" / f"{k}_seg.png")
                calib[n] = {"K": c["K"].tolist()}
                rows.append({"step": world.step, "time_s": round(float(world.data.time), 4), "camera": n,
                             "generation": world.generation, "revision": world.stage.revision,
                             "rgb": f"frames/{k}_rgb.png", "depth": f"frames/{k}_depth.npy",
                             "seg": f"frames/{k}_seg.png", "T_world_cam": np.round(c["T_world_cam"], 6).tolist(),
                             "state": world.state()})

        grab(w)
        if a.get("actions"):
            for x in a["actions"]:
                x.setdefault("action_id", self._id("a"))
            w.act_batch(a["actions"], on_step=grab)
        while len(rows) < frames * len(names):
            w.advance(w.model.opt.timestep * every, on_step=grab)
        (folder / "samples.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        (folder / "calibration.json").write_text(json.dumps({"cameras": calib, "depth_units": "metres along the "
                                                             "optical axis", "frame": "OpenCV (x right, y down, z "
                                                             "forward); T_world_cam per sample",
                                                             "seg": "pixel = index into labels",
                                                             "labels": ["background"] + [e.id for e in w.stage.entities]},
                                                            indent=1))
        return {"folder": str(folder), "samples": len(rows), "cameras": names}

    # ------------------------------------------------------------------ declarations
    def _build(self) -> Surface:
        pred = {"type": "object", "description": "a goal: {op: in_box | near | apart | above | contact | no_contact | "
                                                 "joint_range | settled | upright | lifted | reachable | no_penetration"
                                                 " | all | any | not, ...fields}; see the check tool",
                "additionalProperties": True, "properties": {}}
        action = {"type": "object", "additionalProperties": True, "properties": {}, "description": (
            "{robot, kind, expected_generation, action_id?, duration_s?, ...}. kinds: joints {targets: {joint or "
            "actuator: value}}; gripper {command: open | close | 0..1}; tcp {position [x,y,z], quat_wxyz? (default: keep the "
            "current orientation; an arm at home points its hand down)}; walk "
            "{forward -1..1, turn -1..1}; fly {position, yaw_deg?}; base {x?, y?, yaw_deg?}; policy {policy, command "
            "[vx m/s, vy m/s, yaw rate rad/s]}; hold")}
        record = obj({"camera": {"type": "string", "description": "a camera entity (default: a free view)"},
                      "fps": {"type": "integer", "minimum": 5, "maximum": 60, "description": "frames per second"},
                      "view": {"type": "object", "description": "free view: lookat, distance, azimuth, elevation",
                               "additionalProperties": True, "properties": {}}}) | {
            "description": "record a video of this while it runs (written to the output folder)"}
        T = [
            ToolDef("stage_show", "read", Help(
                "The stage being worked on: its revision and its entities (id, kind, position, robot pack, mount).",
                use_when="before editing (stage_edit needs the revision you read) or to look up entity ids.",
                returns="stage, revision, entities, world_built_from_revision; entity_json for one entity."),
                obj({"entity": {"type": "string", "description": "also return this entity in full"}}), self.stage_show),
            ToolDef("stage_history", "read", Help("The stage's revisions: number, note, digest, entity count."),
                    obj({"last": {"type": "integer", "minimum": 1, "maximum": 200, "description": "how many"}}),
                    self.stage_history),
            ToolDef("stage_edit", "act", Help(
                "Change the stage with a patch: add, remove, update or reparent entities.",
                use_when="building or changing a scene. The world keeps running the old revision until world_build.",
                effect="a new stage revision (compare and swap on expected_revision).",
                returns="the new revision.",
                errors="conflict (someone changed the stage: read it again), refused (unknown or locked entity, an "
                       "invalid result: the message says why), bad_patch.",
                limits=("entity: {id, parent?, label?, pose {pos [x,y,z], quat [w,x,y,z]}, locked?, and ONE of "
                        "shape {type box|sphere|cylinder|capsule|plane, size (box: half extents), rgba}, mesh {path, "
                        "scale}, robot {pack, base fixed|free, mount {entity, body?}, joints {}}, camera {fovy_deg, "
                        "width, height} (looks along its -z, +y up), light, human {route [[x,y]...], loop, speed, "
                        "height, dwell}; body {dynamic, mass?, friction} for shapes and meshes")),
                obj({"expected_revision": {"type": "integer", "minimum": 0, "description": "the revision you read"},
                     "ops": {"type": "array", "minItems": 1, "description": "operations: {op: add, entity} | {op: "
                             "remove, id, cascade?} | {op: update, id, set {...}} | {op: reparent, id, parent}",
                             "items": {"type": "object", "additionalProperties": True, "properties": {},
                                       "description": "one operation"}},
                     "note": {"type": "string", "maxLength": 200, "description": "why"}},
                    ["expected_revision", "ops"]), self.stage_edit),
            ToolDef("stage_restore", "act", Help("Make an old revision the newest again (a new revision).",
                                                 effect="a new stage revision equal to the old one."),
                    obj({"revision": {"type": "integer", "minimum": 0, "description": "revision to restore"}},
                        ["revision"]), self.stage_restore),
            ToolDef("packs", "read", Help(
                "The robots a stage can use (robot.pack), with their family, base, controls and status.",
                returns="packs [{name, title, family, base, controls, status, notes, has_gripper, reach_m}]."),
                obj({"family": {"type": "string", "enum": ["arm", "gripper", "hand", "bimanual", "quadruped",
                                                           "humanoid", "drone", "mobile_manipulator"],
                                "description": "only this family"}}), self.packs),
            ToolDef("policies", "read", Help(
                "Pretrained locomotion policies the world can run (act kind policy: {policy, command [vx, vy, wz]}),"
                " the robot pack each needs; with inspect, the weights file's format and size read from its bytes, "
                "and with robot, every mismatch with that robot.",
                returns="policies [{name, title, pack, joints, license, source, max_command}], inspect, match."),
                obj({"inspect": {"type": "string", "description": "a policy name"},
                     "robot": {"type": "string", "description": "a robot entity to match against"}}), self.policies),
            ToolDef("world_build", "act", Help(
                "Compile the newest stage revision into a running simulation (robots at their home postures).",
                effect="a new world generation; earlier generations' actions become stale.",
                returns="generation, revision, robots with controls and actuator names, cameras, people, overlaps "
                        "(penetrations at the start: fix the stage if any are deep)."),
                obj(), self.world_build),
            ToolDef("world_reset", "act", Help("Put the world back to its start (same revision).",
                                               effect="a new generation."), obj(), self.world_reset),
            ToolDef("world_state", "read", Help(
                "The world now: generation, step, time, every robot (joints, base pose, tool point, gripper opening),"
                " dynamic objects (position, speed), people.",
                use_when="before deciding an action (it needs the generation) and after one."),
                obj(), self.world_state),
            ToolDef("act", "act", Help(
                "Run robot actions. Several actions (different robots) start together and run in the same steps.",
                effect="time passes until every action is done; robots move.",
                returns="receipts [{status completed | refused, steps, measured {...}, reason}].",
                errors="refused with stale (observe again: world_state), unsupported (see the robot's controls), "
                       "unreachable (tcp: no joint solution), busy."),
                obj({"actions": {"type": "array", "minItems": 1, "maxItems": 8, "items": action,
                                 "description": "actions"}, "record": record}, ["actions"]), self.act),
            ToolDef("advance", "act", Help(
                "Let time pass with every robot holding still (people walk, objects settle, drones hover).",
                returns="steps, time, people."),
                obj({"seconds": {"type": "number", "minimum": 0.01, "maximum": 60, "description": "seconds"},
                     "record": record}, ["seconds"]), self.advance),
            ToolDef("people", "act", Help("Send a person somewhere, stop them or let them walk on.",
                                          effect="changes where the person walks next (time passes in act/advance)."),
                    obj({"person": {"type": "string", "description": "a human entity id"},
                         "command": {"type": "string", "enum": ["goto", "stop", "resume"], "description": "command"},
                         "to": vec(2, "goto: [x, y]")}, ["person", "command"]), self.people),
            ToolDef("capture", "read", Help(
                "Images from camera entities (all at the same step), with depth and segmentation kept for "
                "pixel_to_world.",
                returns="capture id, per camera: size, position, viewing direction, focal length, entities seen; "
                        "the images."),
                obj({"cameras": {"type": "array", "items": STR, "description": "camera ids (default all)"}}),
                self.capture),
            ToolDef("view", "read", Help("A picture from a free viewpoint (no camera entity needed).",
                                         returns="the image."),
                    obj({"lookat": vec(3, "point looked at"), "distance": {"type": "number", "minimum": 0.2,
                                                                           "maximum": 30, "description": "metres"},
                         "azimuth": {"type": "number", "description": "degrees"},
                         "elevation": {"type": "number", "minimum": -89, "maximum": 89, "description": "degrees"}}),
                    self.view),
            ToolDef("pixel_to_world", "read", Help(
                "A pixel of a captured image as a world point (from that capture's depth) and the entity it shows.",
                returns="per pixel: world [x,y,z], depth, entity."),
                obj({"capture": {"type": "string", "description": "capture id"}, "camera": STR | {
                    "description": "camera id"},
                     "pixels": {"type": "array", "minItems": 1, "maxItems": 16, "items": vec(2, "[u, v]"),
                                "description": "pixels (u right, v down)"}}, ["capture", "camera", "pixels"]),
                self.pixel_to_world),
            ToolDef("lidar", "read", Help(
                "A range scan from a camera entity's pose: rays over its field of view (or a horizontal ring with "
                "ring: true); hits per entity and the nearest distance; save: true writes the points (npy).",
                returns="rays, hits, hits_by_entity, nearest_m, points_file."),
                obj({"camera": {"type": "string", "description": "camera entity"},
                     "rows": {"type": "integer", "minimum": 1, "maximum": 128, "description": "rays vertically"},
                     "cols": {"type": "integer", "minimum": 1, "maximum": 1024, "description": "rays horizontally"},
                     "ring": {"type": "boolean", "description": "a 360 degree ring instead of the field of view"},
                     "max_m": {"type": "number", "minimum": 0.1, "maximum": 100, "description": "range"},
                     "save": {"type": "boolean", "description": "write the hit points"}}, ["camera"]), self.lidar),
            ToolDef("camera_fit", "read", Help(
                "Estimate a camera's pose from world points and the pixels they appear at (calibration of a real "
                "or simulated camera from known markers).",
                returns="T_world_cam (OpenCV frame), position, reprojection error in pixels.",
                limits="no lens distortion; needs 6 or more well-spread points."),
                obj({"points": {"type": "array", "minItems": 6, "items": vec(3, "[x, y, z]"), "description": "world"},
                     "pixels": {"type": "array", "minItems": 6, "items": vec(2, "[u, v]"), "description": "pixels"},
                     "focal_px": {"type": "number", "minimum": 1, "description": "focal length in pixels"},
                     "cx": {"type": "number", "description": "principal point u"},
                     "cy": {"type": "number", "description": "principal point v"}}, ["points", "pixels", "focal_px"]),
                self.camera_fit),
            ToolDef("check", "read", Help(
                "Evaluate a goal on the world now (no time passes).",
                returns="satisfied, value (true | false | unverifiable), failures [{atom, reason, expected, "
                        "measured}], reasons.",
                limits=("atoms: in_box {entity, min, max}; near {a, b, max_m}; apart {a, b, min_m}; above {a, b, "
                        "min_m}; contact / no_contact {a, b|any}; joint_range {robot, joint, min, max}; settled "
                        "{entity, max_speed}; upright {entity, min_cos}; lifted {entity, min_z}; reachable {robot, "
                        "point, margin_m}; no_penetration {max_mm}; combine with all / any {terms}, not {term}")),
                obj({"goal": pred}, ["goal"]), self.check),
            ToolDef("gate", "read", Help(
                "Check a goal before working on it: refuses one that names missing entities and flags one that "
                "already holds.", returns="ok, reason, failures now."),
                obj({"goal": pred}, ["goal"]), self.gate),
            ToolDef("overlaps", "read", Help("Penetrations between entities now (deepest first).",
                                             returns="overlaps [{a, b, depth_mm, at}]."),
                    obj({"depth_mm": {"type": "number", "minimum": 0.01, "description": "threshold (default 1)"}}),
                    self.overlaps),
            ToolDef("export", "act", Help(
                "Record camera frames over time as a dataset (while actions run or time passes): samples.jsonl "
                "with state and camera pose per frame, RGB png, depth npy (metres), segmentation png, calibration.",
                effect="time passes; files are written.", returns="folder, samples."),
                obj({"cameras": {"type": "array", "items": STR, "description": "cameras (default all)"},
                     "frames": {"type": "integer", "minimum": 1, "maximum": 500, "description": "frames per camera"},
                     "every_s": {"type": "number", "minimum": 0.01, "maximum": 10, "description": "spacing"},
                     "actions": {"type": "array", "items": action, "description": "actions to run while recording"},
                     "name": {"type": "string", "description": "folder name"}}), self.export),
        ]
        return Surface(T, name="mujoco.world")


def _brief(exc: Exception) -> str:
    """A validation error as one short line per problem (field path: message)."""
    errs = getattr(exc, "errors", None)
    if callable(errs):
        lines = []
        for e in errs()[:8]:
            loc = ".".join(str(x) for x in e.get("loc", ()) if x not in ("add", "remove", "update", "reparent"))
            lines.append(f"{loc}: {e.get('msg')}")
        return "; ".join(lines)
    return str(exc)[:600]


def pnp(X: np.ndarray, uv: np.ndarray, K: np.ndarray, iters: int = 30) -> tuple[np.ndarray, float]:
    """Camera pose (T_world_cam, OpenCV frame) from world points and pixels: DLT on normalised coordinates, then
    Gauss-Newton on the reprojection error. Returns (T, rms pixels)."""
    n = len(X)
    xn = (np.linalg.inv(K) @ np.c_[uv, np.ones(n)].T).T[:, :2]
    A = []
    for (x, y, z), (u, v) in zip(X, xn):
        A.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
        A.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])
    P = np.linalg.svd(np.asarray(A))[2][-1].reshape(3, 4)
    if np.linalg.det(P[:, :3]) < 0:
        P = -P
    U, S, Vt = np.linalg.svd(P[:, :3])
    R = U @ Vt
    t = P[:, 3] / S.mean()
    if np.mean((X @ R.T + t)[:, 2]) < 0:                    # points behind: the other sign
        R, t = R, -t
    x = np.concatenate([_rotvec(R), t])

    def project(p: np.ndarray) -> np.ndarray:
        Rm, tv = _rot(p[:3]), p[3:]
        c = X @ Rm.T + tv
        return ((K @ c.T).T[:, :2] / c[:, 2:3])

    for _ in range(iters):
        r = (project(x) - uv).ravel()
        J = np.zeros((len(r), 6))
        for k in range(6):
            dx = np.zeros(6)
            dx[k] = 1e-6
            J[:, k] = ((project(x + dx) - uv).ravel() - r) / 1e-6
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        x = x + step
        if np.linalg.norm(step) < 1e-10:
            break
    rms = float(np.sqrt(np.mean(np.sum((project(x) - uv) ** 2, axis=1))))
    Rcw, tcw = _rot(x[:3]), x[3:]                            # world -> camera
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rcw.T, -Rcw.T @ tcw
    return T, rms


def _rot(v: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def _rotvec(R: np.ndarray) -> np.ndarray:
    th = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    if th < 1e-9:
        return np.zeros(3)
    return th / (2 * math.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


class Recorder:
    """Frames of a camera entity (or a free view) at a frame rate, written as an mp4 when finished."""

    def __init__(self, world: Any, path: Path, camera: str | None, fps: int, view: Mapping[str, Any] | None) -> None:
        self.w, self.path, self.camera, self.fps, self.view = world, path, camera, fps, dict(view or {})
        self.every = max(1, int(round(1.0 / (fps * world.model.opt.timestep))))
        self.frames: list[np.ndarray] = []

    def step(self, world: Any) -> None:
        if world.step % self.every:
            return
        if self.camera:
            self.frames.append(world.capture([self.camera], depth=False, segmentation=False)[self.camera]["rgb"])
        else:
            self.frames.append(world.view(**{k: v for k, v in self.view.items()
                                             if k in ("lookat", "distance", "azimuth", "elevation")}))

    def finish(self) -> dict[str, Any]:
        if not self.frames:
            return {"path": None, "frames": 0}
        import imageio.v2 as imageio

        self.path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(self.path, self.frames, fps=self.fps, quality=7, macro_block_size=8)
        return {"path": str(self.path), "frames": len(self.frames), "seconds": round(len(self.frames) / self.fps, 2)}


class WorldSurface:
    """Entry point for `hylo.surfaces`: `mujoco.world` with params root, stage, out."""

    def __new__(cls, root: str = "runs/stages", stage: str = "lab", out: str | None = None) -> Surface:  # type: ignore[misc]
        ws = Workspace(root, stage, out)
        s = ws.surface
        s.workspace = ws                                   # type: ignore[attr-defined]
        return s


__all__ = ["Recorder", "WorldSurface", "Workspace"]
