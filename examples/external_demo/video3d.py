"""3D replay video: re-simulate an episode's recorded actions in a fresh MuJoCo instance, render two
cameras, and caption each frame with what the trace says the agent knew and decided at that step.

Needs an OpenGL backend for MuJoCo (MUJOCO_GL=osmesa or egl). Returns the maximum fingertip
deviation between the re-simulation and the trace, which doubles as a determinism check.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hylo.core.task import TaskInstance
from hylo.core.trace import EpisodeTrace
from hylo.core.types import Action
from hylo.envs.mujoco.table_reach import TableReachEnv, table_tasks

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TOP, BOTTOM = 64, 88  # caption bands (px); 360 + 64 + 88 = 512 keeps H.264 macro blocks aligned


def _font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default(size=size)


def step_annotations(tr: EpisodeTrace) -> list[dict]:
    """One entry per executed action: the agent's latest decision, visibility and shield count."""
    out, decision, visible, interventions = [], None, True, 0
    for e in tr.events:
        t, p = e["type"], e["payload"]
        if t == "agent.decision":
            decision = p
        elif t == "observation":
            visible = bool(p["state"]["objects_visible"][0] > 0.5)
        elif t == "safety.review" and p.get("modified"):
            interventions += 1
        elif t == "action":
            out.append({"decision": decision, "visible": visible, "interventions": interventions})
    return out


def _caption(frame: np.ndarray, *, title: str, instruction: str, ann: dict, step: int, n: int,
             final: str | None) -> Image.Image:
    img = Image.fromarray(frame)
    w, h = img.size
    canvas = Image.new("RGB", (w, h + TOP + BOTTOM), (24, 26, 30))
    canvas.paste(img, (0, TOP))
    d = ImageDraw.Draw(canvas)
    d.text((14, 8), title, font=_font(FONT, 15), fill=(170, 176, 186))
    d.text((14, 30), f"“{instruction}”", font=_font(FONT_BOLD, 19), fill=(255, 255, 255))
    y = TOP + h + 6
    dec = ann["decision"]
    if dec is not None:
        src = dec.get("inputs", {}).get("object_source")
        txt = f"reasoner [{dec.get('trigger')}]: {dec.get('rationale', '')}" + (f"  ({src})" if src else "")
        for line in textwrap.wrap(txt, 120)[:2]:
            d.text((14, y), line, font=_font(FONT, 14), fill=(120, 220, 150))
            y += 18
    status = f"step {step}/{n}    SafetyShield interventions: {ann['interventions']}"
    d.text((14, TOP + h + BOTTOM - 24), status, font=_font(FONT, 14), fill=(200, 200, 200))
    if not ann["visible"]:
        tag, f = "pads hidden from the agent → acting from memory", _font(FONT_BOLD, 15)
        tw = d.textlength(tag, font=f)
        d.rounded_rectangle((w - tw - 32, TOP + 8, w - 12, TOP + 36), radius=8, fill=(150, 40, 40))
        d.text((w - tw - 22, TOP + 12), tag, font=f, fill=(255, 255, 255))
    if final:
        ok = final.startswith("SUCCESS")
        d.rounded_rectangle((w // 2 - 260, TOP + h // 2 - 28, w // 2 + 260, TOP + h // 2 + 28), radius=12,
                            fill=(30, 130, 70) if ok else (170, 40, 40))
        d.text((w // 2, TOP + h // 2), final, font=_font(FONT_BOLD, 24), fill=(255, 255, 255), anchor="mm")
    return canvas


def render_episode_video(tr: EpisodeTrace, out: str | Path, *, title: str = "", size: tuple[int, int] = (480, 360),
                         fps: int = 20) -> tuple[float, Path]:
    import imageio.v2 as imageio

    start = tr.first("episode.start")["payload"]
    end = tr.last("episode.end")["payload"]
    last = tr.last("step.result")
    reason = (last["payload"]["info"].get("fail_reason") if last else None) or end["termination_reason"]
    reason = {"max_steps": "time budget exhausted"}.get(reason, reason)
    spec = next(t for t in table_tasks() if t.id == start["task_id"])
    anns = step_annotations(tr)
    recorded = [np.asarray(e["payload"]["tip"]) for e in tr.of_type("privileged.state")]
    actions = [e["payload"]["data"] for e in tr.of_type("action")]
    env = TableReachEnv()
    obs = env.reset(TaskInstance(spec=spec, seed=start["seed"]))
    w, h = size
    dev = 0.0
    out = Path(out)
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=7, macro_block_size=8)
    try:
        def frame(i: int, final: str | None = None) -> np.ndarray:
            both = np.concatenate([env.render("front", w, h), env.render("top", h, h)], axis=1)
            return np.asarray(_caption(both, title=title, instruction=obs.instruction or "", ann=anns[min(i, len(anns) - 1)],
                                       step=min(i + 1, len(actions)), n=len(actions), final=final))

        for i, a in enumerate(actions):
            env.step(Action(data=a))
            dev = max(dev, float(np.abs(env.data.site_xpos[env._tip, :2] - recorded[i]).max()))
            writer.append_data(frame(i))
        verdict = "SUCCESS" if end["status"] == "success" else f"{end['status'].upper()}: {reason}"
        for _ in range(fps):  # hold the last frame for a second with the outcome
            writer.append_data(frame(len(actions) - 1, final=verdict))
    finally:
        writer.close()
        env.close()
    return dev, out
