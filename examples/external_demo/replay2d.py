"""Render a mujoco.table_reach episode to an animated GIF from its trace alone (no simulator).

Everything drawn comes from trace events: `privileged.state` (arm joints, true pad layout, shown to
the viewer only), `observation` (what the agent could see), `agent.decision` (subgoal + rationale),
`memory.retrieval` and `safety.review`.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from worldcurrent.core.trace import EpisodeTrace  # noqa: E402

L1, L2, PAD_R = 0.3, 0.25, 0.035


def _frames(tr: EpisodeTrace) -> list[dict]:
    frames, decision, retrieval, interventions = [], None, False, 0
    visible = True
    for e in tr.events:
        t, p = e["type"], e["payload"]
        if t == "agent.decision":
            decision = p
        elif t == "memory.retrieval":
            retrieval = True
        elif t == "safety.review" and p.get("modified"):
            interventions += 1
        elif t == "observation":
            visible = bool(p["state"]["objects_visible"][0] > 0.5)
        elif t == "privileged.state":
            frames.append({"priv": p, "decision": decision, "visible": visible, "retrieval": retrieval,
                           "interventions": interventions})
    return frames


def render_episode_gif(tr: EpisodeTrace, out: str | Path, *, title: str = "", fps: int = 15, stride: int = 1) -> Path:
    frames = _frames(tr)[::stride]
    end = tr.last("episode.end")["payload"]
    last = tr.last("step.result")
    reason = (last["payload"]["info"].get("fail_reason") if last else None) or end["termination_reason"]
    reason = {"max_steps": "time budget exhausted"}.get(reason, reason)
    if not frames:
        raise ValueError(f"episode {tr.episode_id} has no privileged.state events")
    fig, ax = plt.subplots(figsize=(5.4, 6.2), dpi=80)
    tips = np.array([f["priv"]["tip"] for f in frames])

    def draw(i: int) -> None:
        ax.clear()
        f = frames[i]
        pv = f["priv"]
        ax.set_xlim(-0.65, 0.65)
        ax.set_ylim(-0.65, 0.75)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.add_patch(Rectangle((-0.65, -0.65), 1.3, 1.3, color="#d1c2a8", zorder=0))
        for j, (xy, rgb) in enumerate(zip(pv["pad_pos"], pv["pad_rgb"])):
            ax.add_patch(Circle(xy, PAD_R, color=rgb, alpha=1.0 if f["visible"] else 0.35, zorder=1))
            if j == pv["target"]:
                ax.add_patch(Circle(xy, PAD_R + 0.02, fill=False, ls="--", lw=1.2, color="k", zorder=1))
        if not f["visible"]:
            ax.text(0, -0.53, "pads hidden from the agent: acting from memory", ha="center", fontsize=8,
                    color="#7a1f1f", zorder=5)
        q1, q2 = pv["qpos"]
        elbow = np.array([L1 * np.cos(q1), L1 * np.sin(q1)])
        tip = np.asarray(pv["tip"])
        tr_xy = tips[: i + 1]
        ax.plot(tr_xy[:, 0], tr_xy[:, 1], color="#555", lw=0.8, alpha=0.6, zorder=2)
        ax.plot([0, elbow[0], tip[0]], [0, elbow[1], tip[1]], "-o", color="#333", lw=4, ms=5, zorder=3)
        d = f["decision"]
        if d is not None:
            ax.plot(*d["output"], marker="x", color="#0b5", ms=10, mew=2.5, zorder=4)
            src = d.get("inputs", {}).get("object_source", "")
            note = textwrap.fill(f"[{d.get('trigger')}] {d.get('rationale', '')}" + (f" ({src})" if src else ""), 62)
            ax.text(-0.63, 0.47, note, fontsize=7, va="bottom", color="#063", zorder=5)
        ax.text(-0.63, 0.70, textwrap.fill(f"“{pv['instruction']}”", 58), fontsize=8.5, va="top", weight="bold")
        status = f"step {i * stride + 1}/{len(frames) * stride}   shield interventions: {f['interventions']}"
        if i == len(frames) - 1:
            status += f"   -> {end['status'].upper()}" + ("" if end["status"] == "success" else f" ({reason})")
        ax.set_title(title, fontsize=9)
        ax.text(0.63, -0.63, status, ha="right", va="bottom", fontsize=7.5, zorder=5)

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000 // fps)
    out = Path(out)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out
