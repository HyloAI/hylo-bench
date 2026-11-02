"""One-command evaluation for a robosuite-family suite with the robot contract (main process; pure at import).

An adapter describes itself with a `Suite` and calls `main(suite)` from its `examples/run.py`:

    run.py smoke                       one episode of one task, through `hylo run` (the real code path)
    run.py programs [--tasks ...]      the solve(robot) programs, --n episodes per task (validation seeds)
    run.py code-llm [--tasks ...]      the code.llm agent (one model session at a time by default), zero-shot
    run.py summarize OUT               rewrite OUT/results.md from the runs under OUT
    run.py video OUT [--task T]        re-simulate a recorded episode from its trace into a captioned MP4

Every run is a `hylo run` process on a generated experiment config (kept under OUT/configs/), one task per
experiment (an environment instance serves one task). The simulator process imports hylo-core, this package and the
adapter from the checkout running this script (PYTHONPATH), so a runtime venv built from another checkout still runs
this code. Videos need the workspace's `demo` dependency group (Pillow, imageio-ffmpeg).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

QWEN = {"backend": "ollama", "model": "qwen3.5:35b-a3b", "vision": True, "reasoning_effort": "none",
        "max_tokens": 3000, "timeout_s": 2400, "retries": 1}


@dataclass(frozen=True)
class Suite:
    """What the runner needs to know about an adapter."""

    name: str                                  # "robosuite", "robocasa": env type, benchmark type, run names
    env_ref: str                               # "package.module:Class" of the Environment (for video replay)
    runtime: str                               # profile runtime of the simulator venv
    modules: tuple[str, ...]                   # hylo modules the simulator process imports from this checkout
    programs_dir: Path
    programs: dict[str, str]                   # task -> program file stem
    video_camera: str = "frontview"
    code_llm_tasks: tuple[str, ...] = ()
    two_arm_tasks: tuple[str, ...] = ()
    env_params: dict[str, Any] = field(default_factory=dict)


def src_dir(module: str) -> str:
    """The directory a hylo namespace package was imported from (the parent of `hylo/`)."""
    spec = importlib.util.find_spec(module)
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError(module)
    return str(Path(list(spec.submodule_search_locations)[0]).parents[len(module.split(".")) - 1])


def remote(suite: Suite) -> dict[str, Any]:
    path = os.pathsep.join(dict.fromkeys(src_dir(m) for m in ("hylo.core", *suite.modules)))
    return {"runtime": suite.runtime, "env": {"PYTHONPATH": path}}


def program_config(suite: Suite, task: str, n: int, out: Path, image_every: int = 0, timeout_s: int = 1500) -> dict:
    program = suite.programs[task]
    return {
        "schema_version": "hylo.experiment/v0.1",
        "name": f"{suite.name}-program-{task.replace('_', '-')}",
        "benchmark": {"type": suite.name, "split": "validation", "params": {"tasks": [task], "validation": n}},
        "env": {"type": suite.name, "params": {"task": task, "segmentation": True, **suite.env_params},
                "remote": remote(suite)},
        "agent": {"type": "code.program",
                  "params": {"source": (suite.programs_dir / f"{program}.py").read_text(), "timeout_s": timeout_s}},
        "trace": {"record_privileged": False, "image_every": image_every},
        "output_dir": str(out),
        "tags": [f"{suite.name}-program", task, program],
    }


def code_llm_config(suite: Suite, task: str, n: int, out: Path, model: dict | None = None, max_turns: int = 16) -> dict:
    return {
        "schema_version": "hylo.experiment/v0.1",
        "name": f"{suite.name}-code-llm-{task.replace('_', '-')}",
        "benchmark": {"type": suite.name, "split": "validation", "params": {"tasks": [task], "validation": n}},
        "env": {"type": suite.name, "params": {"task": task, "segmentation": True, **suite.env_params},
                "remote": remote(suite)},
        "agent": {"type": "code.llm", "params": {"model": model or QWEN, "max_turns": max_turns,
                                                 "block_timeout_s": 120, "act_wait_s": 7200}},
        "trace": {"record_privileged": False, "image_every": 0},
        "output_dir": str(out),
        "tags": [f"{suite.name}-code-llm", task],
    }


def run_configs(configs: dict[str, dict], out: Path, workers: dict[str, int], parallel: int) -> dict[str, str]:
    """`hylo run` each config (name -> config), at most `parallel` at once; returns name -> run directory."""
    import yaml

    (out / "configs").mkdir(parents=True, exist_ok=True)
    pending, running, done = list(configs), {}, {}
    while pending or running:
        while pending and len(running) < parallel:
            name = pending.pop(0)
            path = out / "configs" / f"{name}.yaml"
            path.write_text(yaml.safe_dump(configs[name], sort_keys=False))
            log = open(out / "configs" / f"{name}.log", "w")
            cmd = [sys.executable, "-m", "hylo.core.cli", "run", str(path), "--set", f"workers={workers[name]}"]
            running[name] = (subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, text=True), log)
            print(f"{time.strftime('%H:%M:%S')} started {name} ({workers[name]} workers)", flush=True)
        for name, (proc, log) in list(running.items()):
            if proc.poll() is None:
                continue
            stdout = proc.stdout.read() if proc.stdout else ""
            log.close()
            del running[name]
            try:
                info = json.loads(stdout[stdout.index("{"):])
                done[name] = info["run_dir"]
                print(f"{time.strftime('%H:%M:%S')} {name}: success {info.get('success_mean')} ({info['run_dir']})",
                      flush=True)
            except (ValueError, KeyError):
                print(f"{name}: hylo run failed (exit {proc.returncode}); see {out / 'configs' / (name + '.log')}",
                      flush=True)
        time.sleep(2)
    return done


# ---------------------------------------------------------------------------------------------- traces
def read_episode(trace: Path) -> dict[str, Any]:
    """Actions, code.llm turns, the program event and the start / end events of one episode's trace."""
    from hylo.core import codec

    ep: dict[str, Any] = {"actions": [], "turns": [], "program": None, "start": None, "end": None}
    with open(trace) as f:
        for line in f:
            head = line[:48]
            if '"observation"' in head or '"safety.review"' in head or '"step.result"' in head:
                continue
            ev = json.loads(line)
            t, p = ev["type"], ev["payload"]
            if t == "action":
                ep["actions"].append(codec.decode(p["data"]))
            elif t == "agent.code.turn":
                ep["turns"].append({k: p.get(k) for k in ("turn", "step", "code", "output", "error", "latency_s",
                                                          "tokens_in", "tokens_out")})
            elif t == "agent.code.program":
                ep["program"] = p
            elif t == "episode.start":
                ep["start"], ep["t0"] = p, ev.get("wall_time")
            elif t == "episode.end":
                ep["end"], ep["t1"] = p, ev.get("wall_time")
    return ep


def collect(out: Path) -> list[dict[str, Any]]:
    """One row per episode of every run under `out`."""
    rows = []
    for run_dir in sorted(p.parent for p in out.glob("*/evallog.json")):
        cfg = json.loads((run_dir / "manifest.json").read_text())["config"]
        log = json.loads((run_dir / "evallog.json").read_text())
        for s in log["samples"]:
            ep = read_episode(run_dir / s["trace"])
            turns = ep["turns"]
            rows.append({"task": s["task_id"].split("/")[-1], "agent": cfg["agent"]["type"], "seed": s["seed"],
                         "episode_id": s["episode_id"], "run_dir": str(run_dir), "success": s["status"] == "success",
                         "steps": s["steps"], "turns": len(turns),
                         "wall_s": round(ep["t1"] - ep["t0"], 1) if ep.get("t0") and ep.get("t1") else None,
                         "tail": "\n".join(((ep["program"] or {}).get("output") or "").strip().splitlines()[-3:])})
    return rows


def summarize(out: Path, title: str) -> str:
    rows = collect(out)
    (out / "results.json").write_text(json.dumps(rows, indent=1))
    lines = [f"# {title}", "", "| task | agent | success | episodes | steps (mean) | turns (mean) | failed seeds |",
             "|---|---|---|---|---|---|---|"]
    by: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by.setdefault((r["task"], r["agent"]), []).append(r)
    for (task, agent), eps in sorted(by.items()):
        ok = sum(e["success"] for e in eps)
        failed = ", ".join(str(e["seed"]) for e in eps if not e["success"]) or "-"
        lines.append(f"| {task} | {agent} | {ok}/{len(eps)} | {len(eps)} | {sum(e['steps'] for e in eps) / len(eps):.0f}"
                     f" | {sum(e['turns'] for e in eps) / len(eps):.1f} | {failed} |")
    text = "\n".join(lines) + "\n"
    (out / "results.md").write_text(text)
    return text


# ---------------------------------------------------------------------------------------------- video
SIZE, FPS = 512, 20
VERB = re.compile(r"robot\.(\w+)\(")


def _font(size: int, mono: bool = False):
    from PIL import ImageFont

    path = Path("/usr/share/fonts/truetype/dejavu") / ("DejaVuSansMono.ttf" if mono else "DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def replay(suite: Suite, cfg: dict, ep: dict, camera: str) -> tuple[list, dict, str]:
    """Re-simulate an episode from its recorded actions in a fresh simulator: (frames, outcome, instruction)."""
    import numpy as np

    from hylo.core.registry import registry
    from hylo.core.remote.client import RemoteEnvironment
    from hylo.core.types import Action

    start = ep["start"]
    bench = registry.build("benchmarks", {k: v for k, v in cfg["benchmark"].items() if k != "split"})
    inst = next(i for i in bench.instances(start["split"]) if i.spec.id == start["task_id"] and i.seed == start["seed"])
    params = {**cfg["env"]["params"], "image_size": SIZE, "cameras": [camera], "segmentation": False}
    r = remote(suite)
    env = RemoteEnvironment(suite.env_ref, params, runtime=r["runtime"], env=r["env"])
    try:
        obs = env.reset(inst)
        frames, res = [obs.images[camera]], None
        for a in ep["actions"]:
            res = env.step(Action(data=np.asarray(a, dtype=float)))
            frames.append(res.observation.images[camera])
            if res.terminated or res.truncated:
                break
        return frames, {"success": bool(res.success) if res else False, "steps": len(frames) - 1}, obs.instruction or ""
    finally:
        env.close()


def captions(ep: dict, steps: int) -> list[dict]:
    """Per frame: the code.llm turn behind the motion (its tool calls and result), or the program's printed lines."""
    out: list[dict] = [{"title": "", "lines": []}] * (steps + 1)
    if ep["turns"]:
        turns = sorted(ep["turns"], key=lambda t: t.get("step") or 0)
        for i, t in enumerate(turns):
            lo = t.get("step") or 0
            hi = (turns[i + 1].get("step") or 0) if i + 1 < len(turns) else steps + 1
            verbs = list(dict.fromkeys(VERB.findall(t.get("code") or "")))
            said = [ln.strip() for ln in (t.get("output") or "").splitlines() if ln.strip()]
            result = ("error: " + (t.get("error") or "").strip().splitlines()[-1]) if t.get("error") else \
                (said[-1] if said else "")
            cap = {"title": f"model turn {t.get('turn')}", "lines": [("tools: " + ", ".join(verbs[:7])) if verbs else "",
                                                                      ("-> " + result) if result else ""]}
            for s in range(max(0, lo), min(hi, steps + 1)):
                out[s] = cap
    elif ep["program"]:
        said = [ln for ln in (ep["program"].get("output") or "").splitlines() if ln.strip()]
        out = [{"title": "solve(robot) program, no model", "lines": said[:3]}] * (steps + 1)
    return out


def frame(img, header: str, sub: str, cap: dict, step: int, steps: int, banner: tuple[str, tuple] | None = None):
    import textwrap

    import numpy as np
    from PIL import Image, ImageDraw

    im = Image.fromarray(np.asarray(img)).convert("RGBA").resize((SIZE, SIZE))
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    head = textwrap.wrap(header, 60)[:2]
    d.rectangle([0, 0, SIZE, 26 + 19 * len(head)], fill=(16, 20, 28, 200))
    for k, ln in enumerate(head):
        d.text((8, 4 + 19 * k), ln, font=_font(15), fill=(255, 255, 255, 255))
    y = 7 + 19 * len(head)
    d.text((8, y), sub[:62], font=_font(13), fill=(205, 212, 228, 255))
    d.text((SIZE - 8, y), f"step {step}/{steps}", font=_font(13), fill=(205, 212, 228, 255), anchor="ra")
    lines = [ln for ln in cap.get("lines", []) if ln]
    if cap.get("title") or lines:
        top = SIZE - 26 - 16 * max(1, len(lines))
        d.rectangle([0, top, SIZE, SIZE], fill=(16, 20, 28, 205))
        d.text((8, top + 4), cap.get("title", ""), font=_font(14), fill=(255, 200, 96, 255))
        for k, ln in enumerate(lines[:3]):
            d.text((10, top + 22 + 16 * k), textwrap.shorten(ln, 78), font=_font(12, True), fill=(150, 222, 168, 255))
    if banner is not None:
        text, colour = banner
        d.rectangle([SIZE // 2 - 140, SIZE // 2 - 38, SIZE // 2 + 140, SIZE // 2 + 38], fill=(*colour, 225))
        d.text((SIZE // 2, SIZE // 2), text, font=_font(38), fill=(255, 255, 255, 255), anchor="mm")
    return np.asarray(Image.alpha_composite(im, layer).convert("RGB"))


def encode(frames: list, path: Path, max_mb: float) -> float:
    """H.264 MP4 at FPS, re-encoded with stronger compression until it is at most `max_mb` MB; returns MB."""
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    for q in (7, 6, 5, 4, 3):
        with imageio.get_writer(str(path), fps=FPS, codec="libx264", quality=q, macro_block_size=16,
                                ffmpeg_params=["-preset", "slow"]) as w:
            for f in frames:
                w.append_data(f)
        if path.stat().st_size <= max_mb * 1e6:
            break
    return round(path.stat().st_size / 1e6, 2)


def make_video(suite: Suite, run_dir: Path, episode_id: str, out: Path, camera: str | None = None,
               max_mb: float = 4.0) -> dict:
    """Re-simulate a recorded episode and write a captioned MP4; the replayed outcome is compared with the
    recorded one (a mismatch is reported, not hidden)."""
    cfg = json.loads((run_dir / "manifest.json").read_text())["config"]
    ep = read_episode(run_dir / "episodes" / f"{episode_id}.jsonl")
    cam = camera or suite.video_camera
    frames, got, instruction = replay(suite, cfg, ep, cam)
    recorded = bool((ep["end"] or {}).get("success"))
    agent = cfg["agent"]["type"]
    workflow = f"code.llm {cfg['agent']['params'].get('model', {}).get('model', '')}" if agent == "code.llm" else agent
    header = f"{suite.name} {ep['start']['task_id'].split('/')[-1]}: {instruction}"
    sub = f"{workflow} | seed {ep['start']['seed']} | {cam}"
    caps = captions(ep, got["steps"])
    video = [frame(f, header, sub, caps[i], i, got["steps"]) for i, f in enumerate(frames)]
    banner = ("SUCCESS", (40, 150, 70)) if got["success"] else ("FAILURE", (170, 50, 40))
    video += [frame(frames[-1], header, sub, caps[-1], got["steps"], got["steps"], banner)] * (2 * FPS)
    mb = encode(video, out, max_mb)
    return {"out": str(out), "mb": mb, "replayed": got, "recorded_success": recorded,
            "replay_matches": got["success"] == recorded}


# ---------------------------------------------------------------------------------------------- CLI
def _pick_episode(out: Path, task: str | None) -> dict | None:
    """The episode to film: the first success of `task`, else of the task with the most successes."""
    rows = collect(out)
    if task:
        rows = [r for r in rows if r["task"] == task]
    wins: dict[str, int] = {}
    for r in rows:
        wins[r["task"]] = wins.get(r["task"], 0) + int(r["success"])
    for t in sorted(wins, key=lambda k: -wins[k]):
        ok = [r for r in rows if r["task"] == t and r["success"]]
        if ok:
            return ok[0]
    return rows[0] if rows else None


def main(suite: Suite, argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog=f"{suite.name} run", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke", help="one episode of one task through hylo run (not a measurement)")
    s.add_argument("--task", default=next(iter(suite.programs)))
    s.add_argument("--out", default=f"runs/{suite.name}-smoke")
    g = sub.add_parser("programs", help="the solve(robot) programs on --n episodes per task")
    g.add_argument("--tasks", nargs="*", default=list(suite.programs))
    g.add_argument("--n", type=int, default=20)
    g.add_argument("--out", default=f"runs/{suite.name}-programs")
    g.add_argument("--workers", type=int, default=24, help="episode workers in total")
    g.add_argument("--parallel", type=int, default=8, help="tasks at once")
    g.add_argument("--image-every", type=int, default=0)
    c = sub.add_parser("code-llm", help="the code.llm agent, zero-shot")
    c.add_argument("--tasks", nargs="*", default=list(suite.code_llm_tasks))
    c.add_argument("--episodes", type=int, default=3)
    c.add_argument("--out", default=f"runs/{suite.name}-code-llm")
    c.add_argument("--sessions", type=int, default=1, help="model sessions at once (the model server is shared)")
    c.add_argument("--model", help="chat model spec as JSON (default: qwen3.5:35b-a3b via Ollama, vision)")
    c.add_argument("--max-turns", type=int, default=16)
    m = sub.add_parser("summarize")
    m.add_argument("out")
    v = sub.add_parser("video", help="captioned MP4 re-simulated from a recorded episode")
    v.add_argument("out", help="a programs / code-llm output directory, or a single run directory")
    v.add_argument("--task", help="film this task (default: the task with the most successes)")
    v.add_argument("--episode", help="episode id (default: the first success)")
    v.add_argument("--camera", default=None)
    v.add_argument("--file", default=None, help="output MP4 (default: OUT/videos/<agent>_<task>.mp4)")
    v.add_argument("--max-mb", type=float, default=4.0)
    a = p.parse_args(argv)
    title = f"{suite.name} {a.cmd}"
    if a.cmd == "smoke":
        out = Path(a.out).resolve()
        done = run_configs({a.task: program_config(suite, a.task, 1, out)}, out, {a.task: 1}, 1)
        print(summarize(out, f"{suite.name} smoke check (1 episode; not a measurement)"))
        return 0 if done else 1
    if a.cmd == "programs":
        out = Path(a.out).resolve()
        per = max(1, min(a.n, a.workers // max(1, min(a.parallel, len(a.tasks)))))
        cfgs = {t: program_config(suite, t, a.n, out, a.image_every) for t in a.tasks}
        run_configs(cfgs, out, {t: per for t in cfgs}, a.parallel)
        print(summarize(out, f"{suite.name} programs ({a.n} episodes per task)"))
        return 0
    if a.cmd == "code-llm":
        out = Path(a.out).resolve()
        model = json.loads(a.model) if a.model else None
        cfgs = {t: code_llm_config(suite, t, a.episodes, out, model, a.max_turns) for t in a.tasks}
        run_configs(cfgs, out, {t: 1 for t in cfgs}, max(1, a.sessions))
        print(summarize(out, f"{suite.name} code.llm, zero-shot ({a.episodes} episodes per task)"))
        return 0
    if a.cmd == "summarize":
        print(summarize(Path(a.out), title))
        return 0
    out = Path(a.out)
    if (out / "evallog.json").exists():
        row = {"run_dir": str(out), "episode_id": a.episode, "task": a.task or "", "agent": ""}
        if not a.episode:
            p.error("--episode is needed with a single run directory")
    elif a.episode:
        row = next(r for r in collect(out) if r["episode_id"] == a.episode)
    else:
        row = _pick_episode(out, a.task)
        if row is None:
            p.error(f"no episodes under {out}")
    run_dir = Path(row["run_dir"])
    agent = json.loads((run_dir / "manifest.json").read_text())["config"]["agent"]["type"].replace(".", "_")
    target = Path(a.file) if a.file else out / "videos" / f"{agent}_{row['task'] or 'episode'}.mp4"
    print(json.dumps(make_video(suite, run_dir, row["episode_id"], target, a.camera, a.max_mb)))
    return 0


__all__ = ["QWEN", "Suite", "captions", "code_llm_config", "collect", "make_video", "main", "program_config",
           "read_episode", "replay", "run_configs", "summarize"]
