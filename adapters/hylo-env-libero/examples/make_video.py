"""An MP4 of a recorded LIBERO episode (code.program or code.llm with the robot contract), re-simulated from the
trace's actions: a fresh simulator in the libero-sim runtime is reset with the recorded task and init state and
stepped with the recorded `action` events; the replayed outcome is checked against the recorded one. Agentview at
512x512, captioned from the trace (the model's turns for code.llm, the program's printed lines for code.program).

    uv run python bench/adapters/hylo-env-libero/examples/make_video.py RUN_DIR EPISODE_ID OUT.mp4 \\
        [--workflow "code.program"] [--max-mb 5]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hylo.core import codec

SIZE, FPS = 512, 20
VERB = re.compile(r"robot\.(\w+)\(")


def font(size: int, mono: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSansMono.ttf" if mono else "DejaVuSans.ttf"
    for p in (Path("/usr/share/fonts/truetype/dejavu") / name,):
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


F_HEAD, F_TEXT, F_CODE, F_BIG = font(15), font(13), font(12, True), font(40)


def read_trace(run_dir: Path, episode_id: str) -> dict:
    ep: dict = {"actions": [], "turns": [], "program": None, "start": None, "end": None}
    with open(run_dir / "episodes" / f"{episode_id}.jsonl") as f:
        for line in f:
            head = line[:48]
            if '"observation"' in head or '"safety.review"' in head or '"step.result"' in head:
                continue
            ev = json.loads(line)
            t, p = ev["type"], ev["payload"]
            if t == "action":
                ep["actions"].append(codec.decode(p["data"]))
            elif t == "agent.code.turn":
                ep["turns"].append({k: p.get(k) for k in ("turn", "step", "reply", "code", "output", "error")})
            elif t == "agent.code.program":
                ep["program"] = p
            elif t == "episode.start":
                ep["start"] = p
            elif t == "episode.end":
                ep["end"] = p
    return ep


def src_dir(module: str) -> str:
    spec = importlib.util.find_spec(module)
    return str(Path(list(spec.submodule_search_locations)[0]).parents[len(module.split(".")) - 1])


def replay(cfg: dict, start: dict, actions: list) -> tuple[list[np.ndarray], dict, str]:
    from hylo.core.registry import registry
    from hylo.core.remote.client import RemoteEnvironment
    from hylo.core.types import Action

    bench = registry.build("benchmarks", {k: v for k, v in cfg["benchmark"].items() if k != "split"})
    inst = next(i for i in bench.instances(start["split"]) if i.spec.id == start["task_id"] and i.seed == start["seed"])
    params = {**cfg["env"]["params"], "image_size": SIZE, "cameras": ["agentview"], "segmentation": False}
    remote = dict(cfg["env"].get("remote") or {"runtime": "libero-sim"})
    env_vars = dict(remote.get("env") or {})
    env_vars["PYTHONPATH"] = os.pathsep.join([src_dir("hylo.core"), src_dir("hylo.envs.libero")])
    env = RemoteEnvironment("hylo.envs.libero.env:LiberoEnv", params, runtime=remote.get("runtime"), env=env_vars)
    try:
        frames = [env.reset(inst).images["agentview"]]
        res = None
        for a in actions:
            res = env.step(Action(data=np.asarray(a, dtype=float)))
            frames.append(res.observation.images["agentview"])
            if res.terminated or res.truncated:
                break
        return frames, {"success": bool(res.success) if res is not None else False,
                        "steps": len(frames) - 1}, inst.spec.instruction
    finally:
        env.close()


def captions(ep: dict, steps: int) -> list[dict | None]:
    """Per step: the caption shown (the code.llm turn that produced the motion, or the program's output)."""
    out: list[dict | None] = [None] * (steps + 1)
    if ep["turns"]:
        turns = sorted(ep["turns"], key=lambda t: t.get("step") or 0)
        for i, t in enumerate(turns):
            lo = t.get("step") or 0
            hi = (turns[i + 1].get("step") or 0) if i + 1 < len(turns) else steps + 1
            verbs = list(dict.fromkeys(VERB.findall(t.get("code") or "")))
            lines = [ln.strip() for ln in (t.get("output") or "").splitlines() if ln.strip()]
            cap = {"title": f"turn {t['turn']}", "verbs": verbs,
                   "result": ("error: " + (t.get("error") or "").strip().splitlines()[-1]) if t.get("error")
                   else (lines[-1] if lines else ""),
                   "code": [ln for ln in (t.get("code") or "").splitlines()
                            if re.search(r"robot\.(run|plan|grasps|grasp_at|place_at|Goal)|finish\(", ln)][:3]}
            for s in range(max(0, lo), min(hi, steps + 1)):
                out[s] = cap
    elif ep["program"]:
        lines = [ln for ln in (ep["program"].get("output") or "").splitlines() if ln.strip()][:4]
        cap = {"title": "frozen program solve(robot), no model", "verbs": [], "result": "", "code": lines}
        out = [cap] * (steps + 1)
    return out


def overlay(img: np.ndarray, header: str, sub: str, cap: dict | None, step: int, steps: int,
            banner: tuple[str, tuple] | None = None) -> np.ndarray:
    im = Image.fromarray(img).convert("RGBA").resize((SIZE, SIZE), Image.LANCZOS)
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    head = textwrap.wrap(header, 62)[:2]
    d.rectangle([0, 0, SIZE, 24 + 19 * len(head)], fill=(14, 18, 26, 200))
    for k, ln in enumerate(head):
        d.text((8, 4 + 19 * k), ln, font=F_HEAD, fill=(255, 255, 255, 255))
    y = 6 + 19 * len(head)
    d.text((8, y), sub[:60], font=F_TEXT, fill=(200, 210, 225, 255))
    d.text((SIZE - 8, y), f"step {step}/{steps}", font=F_TEXT, fill=(200, 210, 225, 255), anchor="ra")
    if cap is not None:
        top = SIZE - 92
        d.rectangle([0, top, SIZE, SIZE], fill=(14, 18, 26, 205))
        d.text((8, top + 5), cap["title"], font=F_HEAD, fill=(255, 196, 92, 255))
        y = top + 25
        for ln in cap.get("code", [])[:3]:
            d.text((10, y), textwrap.shorten(ln, 74), font=F_CODE, fill=(140, 220, 160, 255))
            y += 15
        tail = ("tools: " + ", ".join(cap["verbs"][:6]) + "   " if cap.get("verbs") else "") + \
            ("-> " + cap["result"] if cap.get("result") else "")
        if tail.strip():
            d.text((8, SIZE - 18), textwrap.shorten(tail, 84), font=F_TEXT, fill=(170, 200, 255, 255))
    if banner is not None:
        text, colour = banner
        d.rectangle([SIZE // 2 - 150, SIZE // 2 - 40, SIZE // 2 + 150, SIZE // 2 + 40], fill=(*colour, 225))
        d.text((SIZE // 2, SIZE // 2), text, font=F_BIG, fill=(255, 255, 255, 255), anchor="mm")
    return np.asarray(Image.alpha_composite(im, layer).convert("RGB"))


def encode(frames: list[np.ndarray], out: Path, max_mb: float) -> None:
    import imageio_ffmpeg

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(frames):
            Image.fromarray(f).save(f"{tmp}/{i:05d}.png")
        for crf in (26, 30, 34, 38):
            subprocess.run([exe, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{tmp}/%05d.png",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "slow", str(out)],
                           check=True)
            if out.stat().st_size <= max_mb * 1e6:
                return


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_dir")
    p.add_argument("episode_id")
    p.add_argument("out")
    p.add_argument("--workflow", default=None, help="caption, e.g. 'code.program' or 'code.llm qwen3.5:35b-a3b'")
    p.add_argument("--max-mb", type=float, default=5.0)
    a = p.parse_args()
    run_dir = Path(a.run_dir)
    cfg = json.loads((run_dir / "manifest.json").read_text())["config"]
    ep = read_trace(run_dir, a.episode_id)
    frames, out, instruction = replay(cfg, ep["start"], ep["actions"])
    recorded = ep["end"] or {}
    if bool(recorded.get("success")) != out["success"]:
        print(f"warning: replay outcome {out} differs from the recorded {recorded}", file=sys.stderr)
    steps = out["steps"]
    caps = captions(ep, steps)
    agent = cfg["agent"]["type"]
    workflow = a.workflow or (f"{agent} {cfg['agent']['params'].get('model', {}).get('model', '')}".strip()
                              if agent == "code.llm" else agent)
    suite = cfg["benchmark"]["params"].get("suite", "")
    header = f"{suite}: {instruction}"
    sub = f"{workflow} | init state {ep['start']['seed'] % 50}"
    video = [overlay(f, header, sub, caps[i], i, steps) for i, f in enumerate(frames)]
    ok = out["success"]
    banner = ("SUCCESS", (40, 150, 70)) if ok else ("FAILURE", (170, 50, 40))
    video += [overlay(frames[-1], header, sub, caps[-1], steps, steps, banner)] * (FPS * 2)
    encode(video, Path(a.out), a.max_mb)
    print(json.dumps({"out": a.out, "mb": round(Path(a.out).stat().st_size / 1e6, 2), **out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
