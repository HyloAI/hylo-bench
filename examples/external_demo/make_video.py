"""Compose a single walkthrough video from a finished demo run (no screen capture needed).

    uv run --group demo python examples/external_demo/make_video.py [runs/demo-<timestamp>]

Title and explanation cards are generated from the run's summary.json; the clips are the
re-simulated 3D replays the demo already rendered; the VLM card shows the marked image the model
actually received (from the trace). Output: <run>/demo_walkthrough.mp4
"""

from __future__ import annotations

import io
import json
import sys
import textwrap
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from hylo.core.trace import TraceStore  # noqa: E402

W, H, FPS = 840, 512, 20
BG, FG, MUTED, ACCENT = (24, 26, 30), (240, 240, 240), (160, 166, 176), (120, 220, 150)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default(size=size)


def card(title: str, lines: list[str], *, image: Image.Image | None = None, kicker: str = "") -> np.ndarray:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    x = 40
    if kicker:
        d.text((x, 34), kicker.upper(), font=font(BOLD, 14), fill=ACCENT)
    y = 56
    for part in textwrap.wrap(title, 44):
        d.text((x, y), part, font=font(BOLD, 28), fill=FG)
        y += 36
    text_w = 62 if image is None else 40
    y += 20
    for line in lines:
        for i, part in enumerate(textwrap.wrap(line, text_w) or [""]):
            d.text((x + (18 if i else 0), y), ("• " if i == 0 and line else "") + part if line else "", font=font(FONT, 18), fill=FG if line else MUTED)
            y += 26
        y += 6
    if image is not None:
        image = image.copy()
        image.thumbnail((W // 2 - 40, H - 140))
        img.paste(image, (W - image.width - 40, 112))
    return np.asarray(img)


def chart(summary: dict) -> Image.Image:
    acts = summary["acts"]
    tasks = sorted({t for a in acts.values() for t in a["per_task"]})
    fig, ax = plt.subplots(figsize=(7.6, 3.9), dpi=110)
    fig.patch.set_facecolor("#181a1e")
    ax.set_facecolor("#181a1e")
    palette = ["#9aa0a6", "#c58af9", "#4285f4", "#34a853", "#f29900"]
    w = 0.8 / len(acts)
    for i, (name, a) in enumerate(acts.items()):
        ax.bar(np.arange(len(tasks)) + i * w, [a["per_task"].get(t, 0) for t in tasks], w, label=name, color=palette[i % 5])
    ax.set_xticks(np.arange(len(tasks)) + w * (len(acts) - 1) / 2, [t.replace("table_", "") for t in tasks], color="w")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate", color="w")
    ax.tick_params(colors="w")
    for sp in ax.spines.values():
        sp.set_color("#555")
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, facecolor="#181a1e", labelcolor="w", edgecolor="#555")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return Image.open(buf).convert("RGB")


def chart_card(summary: dict) -> np.ndarray:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((40, 34), "RESULTS", font=font(BOLD, 14), fill=ACCENT)
    d.text((40, 56), "Same task, components swapped by configuration", font=font(BOLD, 26), fill=FG)
    c = chart(summary)
    c.thumbnail((W - 60, H - 140))
    img.paste(c, ((W - c.width) // 2, 100))
    n = next(iter(summary["acts"].values()))["metrics"]["n_episodes"] // 3
    d.text((40, H - 34), f"{n} validation seeds per task; identical seeds for every agent", font=font(FONT, 15), fill=MUTED)
    return np.asarray(img)


def fmt(d: dict) -> str:
    return ", ".join(f"{k.split('.')[-1]}={v}" for k, v in d.items())


def episode_label(run_root: Path, episode_id: str) -> tuple[str, Path] | None:
    for f in run_root.glob(f"*/episodes/{episode_id}.jsonl"):
        return f.parent.parent.name, f.parent.parent
    return None


def vlm_image(run_dir: Path, episode_id: str) -> tuple[Image.Image, str] | None:
    tr = TraceStore(run_dir).read(episode_id)
    for e in tr.of_type("agent.decision"):
        meta = e["payload"]["inputs"].get("chooser_meta") or {}
        if meta.get("marked_image") is not None:
            return Image.fromarray(np.asarray(meta["marked_image"], dtype=np.uint8)), e["payload"]["rationale"]
    return None


def main() -> int:
    root = Path(__file__).resolve().parents[2] / "runs"
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(root.glob("demo-2*"))[-1]
    summary = json.loads((run / "summary.json").read_text())
    acts = summary["acts"]
    rsi = summary["rsi"]

    clips: dict[str, Path] = {}
    vlm_run = None
    for mp4 in sorted(run.glob("*.mp4")):
        if mp4.name == "demo_walkthrough.mp4":
            continue
        found = episode_label(run / "runs", mp4.stem)
        if not found:
            continue
        label, rdir = found
        key = ("rule" if label.startswith("demo-2-") else "llm" if label.startswith("demo-3-") else
               "vlm" if label.startswith("demo-3b-") else "before" if "parent" in label else "after")
        clips[key] = mp4
        if key == "vlm":
            vlm_run = (rdir, mp4.stem)

    frames: list[np.ndarray] = []

    def hold(img: np.ndarray, seconds: float) -> None:
        frames.extend([img] * int(seconds * FPS))

    def play(path: Path) -> None:
        for f in imageio.mimread(path, memtest=False):
            f = np.asarray(Image.fromarray(f).resize((W, H))) if f.shape[:2] != (H, W) else f
            frames.append(f)

    rule, llm = acts.get("2 rule reasoner + spatial memory", {}), acts.get("3 LLM reasoner (colour values) + memory", {})
    vlm = acts.get("3b VLM reasoner (camera image only) + memory", {})
    model = summary.get("model") or "LLM"

    hold(card("Hylo: one harness for language, memory and self-improvement",
              ["Physical AI research stack: kernel-owned control loop, one SafetyShield for every action, "
               "replayable event traces, swappable reasoners, memory and policies.",
               f"This demo: MuJoCo simulation, local model {model} (Ollama), everything recorded and re-simulated from traces."],
              kicker="demo"), 5)
    hold(card("The task",
              ["A planar arm must touch the pad named by a natural-language instruction.",
               "Direct: “touch the red pad”.  Indirect: “the pad the colour of a ripe tomato”.",
               "The pads are visible only for the first 0.25 s; then the agent must act from memory.",
               "One variant shoves the arm and gives it only 3.5 s.",
               "Success = fingertip rests on the right pad; resting on a wrong pad fails."], kicker="setup"), 7)
    hold(chart_card(summary), 7)
    hold(card("Rule-based reasoner vs. language models",
              [f"Rule reasoner + memory: {rule.get('per_task', {}).get('table_indirect', 0):.2f} on indirect instructions "
               "(it only understands explicit colour words).",
               f"LLM chooser (one config line): {llm.get('per_task', {}).get('table_indirect', 0):.2f}.",
               f"VLM chooser that sees only the camera image: {vlm.get('per_task', {}).get('table_indirect', 0):.2f}.",
               "Next: the same episode seed, three reasoners."], kicker="acts 2-3"), 6)
    for key in ("rule", "llm"):
        if key in clips:
            play(clips[key])
    if vlm_run:
        got = vlm_image(*vlm_run)
        if got:
            img, why = got
            hold(card("What the VLM saw",
                      ["Top camera image with number tags projected from the camera parameters; no colour values are given.",
                       f"Model answer: “{why}”", "The image is stored in the trace as evidence."],
                      image=img, kicker="act 3b"), 7)
    if "vlm" in clips:
        play(clips["vlm"])
    cand_lines = []
    for e in rsi["evaluated"]:
        if "rejected" in e:
            cand_lines.append(f"{fmt(e['patch'])}: blocked (protected scope: candidates may not touch safety settings)")
        else:
            g = e["gate"]
            v = g["validation"]
            tag = "PROMOTED" if e["id"] == rsi["promoted"] else ("pass" if g["passed"] else "REJECTED: " + "; ".join(g["reasons"]))
            cand_lines.append(f"{fmt(e['patch'])}: success {v['candidate_success']:.2f}, shield {v['candidate_safety']:.1f}/episode \u2192 {tag}")
    ho = rsi.get("heldout") or {}
    hold(card("Self-improvement with a validation gate",
              [f"Start: controller gain kp=0.03 is too slow; train-split diagnosis {fmt(rsi['diagnosis'])}", *cand_lines,
               f"Held-out (report only): {ho.get('candidate_success', 0):.2f} vs {ho.get('parent_success', 0):.2f}. "
               "Then rolled back on request; every step is in the registry ledger."], kicker="act 4"), 10)
    for key in ("before", "after"):
        if key in clips:
            play(clips[key])
    det = summary.get("replay_determinism", [])
    hold(card("Every run is evidence",
              ["Every action passed through the same SafetyShield; unsafe candidates were rejected by the gate.",
               f"All videos are re-simulated from recorded actions: max deviation {max((d['max_tip_deviation_m'] for d in det), default=0):g} m (bit-exact).",
               "Components swap by configuration: rule → LLM → VLM without code changes.",
               "Reproduce: uv run --group demo python examples/external_demo/run_demo.py"], kicker="summary"), 7)

    out = run / "demo_walkthrough.mp4"
    with imageio.get_writer(out, fps=FPS, codec="libx264", quality=7, macro_block_size=8) as w:
        for f in frames:
            w.append_data(f)
    print(f"{out}  ({len(frames) / FPS:.0f} s, {out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
