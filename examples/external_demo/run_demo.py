"""Hylo external demo: language + memory + self-improvement on a MuJoCo table-top task.

    uv run --group demo python examples/external_demo/run_demo.py            # full (about 3-5 min first run)
    uv run --group demo python examples/external_demo/run_demo.py --quick    # fewer seeds
    uv run --group demo python examples/external_demo/run_demo.py --no-llm   # skip the LLM act

Acts
  1  Baseline: no language understanding, no memory.
  2  Composable agent: rule-based reasoner + spatial memory (and the same agent without memory).
  3  Swap one config line: the reasoner's chooser becomes a local LLM (Ollama).
  4  Offline self-improvement: diagnose -> candidates -> protected scope -> paired gate -> held-out -> promote -> rollback.
  5  Replay: deterministic re-simulation check and animated replays rendered from traces only.
Output: runs/demo-<timestamp>/report.html (self-contained) + summary.json + all traces.
"""

from __future__ import annotations

import argparse
import base64
import collections
import html
import io
import json
import os
import sys
import time
from pathlib import Path

USER_MESA = Path.home() / ".local" / "mesa" / "root" / "usr" / "lib" / "x86_64-linux-gnu"


def ensure_render_env() -> None:
    """Use the user-space OSMesa install (scripts/install-osmesa-user.sh) for MuJoCo offscreen rendering.
    LD_LIBRARY_PATH is only read at process start, so re-exec once with the environment set."""
    if os.environ.get("MUJOCO_GL") or "--no-3d" in sys.argv or not (USER_MESA / "libOSMesa.so.8").exists():
        return
    env = {**os.environ, "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa",
           "LD_LIBRARY_PATH": os.pathsep.join(filter(None, [str(USER_MESA), os.environ.get("LD_LIBRARY_PATH")]))}
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


ensure_render_env()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hylo.eval.experiment import run_experiment  # noqa: E402
from hylo.eval.runner import EvalResult  # noqa: E402
from hylo.eval.scoring import paired_compare  # noqa: E402
from hylo.core.config import ExperimentConfig  # noqa: E402
from hylo.core.task import TaskInstance  # noqa: E402
from hylo.core.trace import TraceStore  # noqa: E402
from hylo.core.types import Action  # noqa: E402
from hylo.envs.mujoco.table_reach import TableReachEnv, table_tasks  # noqa: E402
from hylo.evolve.candidate import Candidate, CandidateRegistry  # noqa: E402
from hylo.evolve.loop import GainProposer, improvement_cycle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay2d import render_episode_gif  # noqa: E402
from video3d import render_episode_video  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def log(msg: str) -> None:
    print(f"[demo {time.strftime('%H:%M:%S')}] {msg}", flush=True)


RULE, NOMEM, BASE = "2 rule reasoner + spatial memory", "2a rule reasoner, memory OFF", "1 baseline (no language, no memory)"
LLM, VLM = "3 LLM reasoner (colour values) + memory", "3b VLM reasoner (camera image only) + memory"


def config(name: str, agent: dict, bench_params: dict, split: str = "validation", vision: bool = False) -> ExperimentConfig:
    return ExperimentConfig(name=name, benchmark={"type": "mujoco/table-v0", "split": split,
                                                  "params": {**bench_params, **({"vision": True} if vision else {})}},
                            env={"type": "mujoco.table_reach"}, agent=agent, safety={"max_delta": 0.5},
                            tags=("external-demo",))


def per_task(r: EvalResult) -> dict[str, float]:
    acc: dict[str, list[float]] = collections.defaultdict(list)
    for s in r.log["samples"]:
        v = s["scores"]["success"]["value"]
        if v is not None:
            acc[s["task_id"].split("/")[1]].append(v)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def fig_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def bars(acts: dict[str, EvalResult]) -> str:
    tasks = sorted({t for r in acts.values() for t in per_task(r)})
    fig, ax = plt.subplots(figsize=(8, 3.4))
    w = 0.8 / len(acts)
    palette = ["#9aa0a6", "#c58af9", "#4285f4", "#34a853", "#f29900"]
    for i, (name, r) in enumerate(acts.items()):
        pt = per_task(r)
        ax.bar(np.arange(len(tasks)) + i * w, [pt.get(t, 0) for t in tasks], w, label=name, color=palette[i % len(palette)])
    ax.set_xticks(np.arange(len(tasks)) + w * (len(acts) - 1) / 2, [t.replace("table_", "") for t in tasks])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    ax.spines[["top", "right"]].set_visible(False)
    return fig_png(fig)


def find_episode(r: EvalResult, task_suffix: str, status: str | None = None, seed: int | None = None) -> str | None:
    for s in r.log["samples"]:
        if s["task_id"].endswith(task_suffix) and (status is None or s["status"] == status) and (seed is None or s["seed"] == seed):
            return s["episode_id"]
    return None


def resimulate(trace, try_render: bool) -> tuple[float, list[np.ndarray]]:
    """Replay recorded actions into a fresh simulator; return max tip deviation vs the trace (+ frames)."""
    start = trace.first("episode.start")["payload"]
    spec = next(t for t in table_tasks() if t.id == start["task_id"])
    env = TableReachEnv()
    env.reset(TaskInstance(spec=spec, seed=start["seed"]))
    recorded = [np.asarray(e["payload"]["tip"]) for e in trace.of_type("privileged.state")]
    frames, dev = [], 0.0
    for i, e in enumerate(trace.of_type("action")):
        env.step(Action(data=e["payload"]["data"]))
        dev = max(dev, float(np.abs(env.data.site_xpos[env._tip, :2] - recorded[i]).max()))
        if try_render:
            frames.append(env.render("front", 480, 360))
    env.close()
    return dev, frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--model", default="qwen3.5:35b-a3b")
    ap.add_argument("--base-url", default="http://127.0.0.1:11434")
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--no-3d", action="store_true", help="skip MuJoCo 3D video rendering")
    args = ap.parse_args()

    n_val = 8 if args.quick else 20
    out = Path(args.out) / time.strftime("demo-%Y%m%d-%H%M%S")
    runs = out / "runs"
    cache = Path(args.out) / "demo-llm-cache"
    bench = {"train": 10, "validation": n_val, "heldout": n_val}
    summary: dict = {"out": str(out), "model": None if args.no_llm else args.model, "acts": {}}
    t_start = time.time()

    # ---- Acts 1-3 -------------------------------------------------------------------------------------
    def chooser(kind: str) -> dict:
        return {"type": "composable.object_reach", "params": {"chooser": {"type": kind, "params": {
            "model": args.model, "base_url": args.base_url, "cache_dir": str(cache)}}}}

    agents: dict[str, tuple[dict, bool]] = {
        BASE: ({"type": "baseline.nearest_object"}, False),
        NOMEM: ({"type": "composable.object_reach", "params": {"memory": None}}, False),
        RULE: ({"type": "composable.object_reach"}, False),
    }
    can_render = os.environ.get("MUJOCO_GL") in ("egl", "osmesa")
    if not args.no_llm:
        agents[LLM] = (chooser("chooser.llm"), False)
        if can_render:
            agents[VLM] = (chooser("chooser.vlm"), True)
        else:
            log("VLM act skipped: MuJoCo rendering unavailable (install OSMesa or set MUJOCO_GL)")
    acts: dict[str, EvalResult] = {}
    cfgs: dict[str, ExperimentConfig] = {}
    for name, (agent, vision) in agents.items():
        log(f"act {name} ...")
        cfgs[name] = config("demo-" + name.split()[0], agent, bench, vision=vision)
        acts[name] = run_experiment(cfgs[name], run_root=runs)
        log(f"   success={acts[name].metrics['success_mean']:.2f}  per task {per_task(acts[name])}")
        summary["acts"][name] = {"metrics": acts[name].metrics, "per_task": per_task(acts[name]),
                                 "config_digest": cfgs[name].digest(), "run_dir": str(acts[name].run_dir)}
    pairs = [(RULE, BASE), (RULE, NOMEM), (LLM, RULE), (VLM, RULE), (VLM, LLM)]
    comparisons = [(a, b, paired_compare(acts[a].per_key(), acts[b].per_key())) for a, b in pairs if a in acts and b in acts]

    def decision_log(r: EvalResult) -> list[dict]:
        out_rows, st = [], TraceStore(r.run_dir)
        for s in r.log["samples"]:
            tr = st.read(s["episode_id"])
            d = next((e for e in tr.of_type("agent.decision") if e["payload"]["inputs"].get("new_choice")), None)
            if d is None:
                continue
            priv = tr.first("privileged.state")["payload"]
            meta = d["payload"]["inputs"].get("chooser_meta", {})
            out_rows.append({"task": s["task_id"].split("/")[1], "seed": s["seed"], "instruction": priv["instruction"],
                             "colours": priv["colors"], "truth": f"obj_{priv['target']} ({priv['colors'][priv['target']]})",
                             "choice": d["payload"]["inputs"].get("target"), "rationale": d["payload"]["rationale"],
                             "latency_s": meta.get("latency_s"), "cached": meta.get("cached"), "status": s["status"],
                             "marked_image": meta.get("marked_image")})
        return out_rows

    def call_stats(rows: list[dict]) -> dict:
        live = [d["latency_s"] for d in rows if not d["cached"] and d["latency_s"] is not None]
        return {"n": len(rows), "cached": sum(1 for d in rows if d["cached"]), "mean_latency_s": float(np.mean(live or [0]))}

    decisions = decision_log(acts[LLM]) if LLM in acts else []
    vlm_decisions = decision_log(acts[VLM]) if VLM in acts else []
    summary["llm_calls"], summary["vlm_calls"] = call_stats(decisions), call_stats(vlm_decisions)

    # ---- Act 4: self-improvement ----------------------------------------------------------------------
    log("act 4 self-improvement on the push task (controller too slow for the 3.5 s budget) ...")
    rsi_base = config("demo-rsi", {"type": "composable.object_reach", "params": {"kp": 0.03}},
                      {"tags": ["recovery"], **bench})
    registry = CandidateRegistry(out / "registry")

    class WithAdversary(GainProposer):
        name = "rule.gain+adversary@0.1"

        def propose(self, diagnosis, cfg, parent_id, archive):
            bad = Candidate(patch={"safety.max_delta": 10.0}, parent_id=parent_id, generator=self.name,
                            rationale="converge faster by removing the action rate limit")
            return [(bad, bad.rationale), *super().propose(diagnosis, cfg, parent_id, archive)]

    rsi_evals: dict[str, EvalResult] = {}

    def evaluate(c, split, label):
        r = run_experiment(c, split=split, run_root=runs, label=label)
        rsi_evals[label] = r
        log(f"   {label:<28} {split:<10} success={r.metrics['success_mean']:.2f} "
            f"shield={r.metrics['safety_interventions_mean']:.2f}")
        return r

    cycle = improvement_cycle(rsi_base, registry, WithAdversary(factors=(2.0, 4.0, 8.0, 16.0), adaptive=True), evaluate,
                              gate_kwargs={"max_safety_increase": 1.0})
    promoted_patch = registry.effective_patch()
    rolled_back_to = registry.rollback(reason="demo: simulated field regression report", by="operator")
    summary["rsi"] = {"diagnosis": cycle.diagnosis.buckets, "evaluated": cycle.evaluated, "promoted": cycle.promoted,
                      "promoted_patch": promoted_patch, "heldout": cycle.heldout,
                      "after_rollback_patch": registry.effective_patch(), "rolled_back_to": rolled_back_to,
                      "ledger": [e["op"] for e in registry.ledger()]}

    # ---- Act 5: replay --------------------------------------------------------------------------------
    log("act 5 replays ...")
    gifs: list[tuple[str, str, str]] = []
    rule = acts[RULE]
    seed_wrong = next((s["seed"] for s in rule.log["samples"] if s["task_id"].endswith("indirect") and s["status"] != "success"), None)
    picks = []
    if seed_wrong is not None:
        picks.append(("Rule reasoner on an indirect instruction", rule, find_episode(rule, "indirect", seed=seed_wrong)))
        if LLM in acts:
            picks.append(("LLM reasoner (colour values), same episode seed", acts[LLM], find_episode(acts[LLM], "indirect", seed=seed_wrong)))
        if VLM in acts:
            picks.append(("VLM reasoner (camera image only), same episode seed", acts[VLM], find_episode(acts[VLM], "indirect", seed=seed_wrong)))
    parent_val = rsi_evals.get("parent-validation")
    if parent_val:
        fail = find_episode(parent_val, "push", status="failure")
        if fail:
            seed = next(s["seed"] for s in parent_val.log["samples"] if s["episode_id"] == fail)
            picks.append(("Before self-improvement (kp=0.03): times out after the shove", parent_val, fail))
            if cycle.promoted:
                cand = rsi_evals.get(f"{cycle.promoted}-validation")
                if cand:
                    picks.append((f"After promotion ({promoted_patch}): same seed", cand, find_episode(cand, "push", seed=seed)))
    determinism = []
    videos: list[tuple[str, str, str]] = []
    want_3d = not args.no_3d and os.environ.get("MUJOCO_GL") in ("egl", "osmesa")
    for title, r, eid in picks:
        if eid is None:
            continue
        tr = TraceStore(r.run_dir).read(eid)
        path = out / f"{eid}.gif"
        render_episode_gif(tr, path, title=title, stride=1)
        gifs.append((title, eid, base64.b64encode(path.read_bytes()).decode()))
        dev = None
        if want_3d:
            try:
                dev, mp4 = render_episode_video(tr, out / f"{eid}.mp4", title=title)
                videos.append((title, eid, base64.b64encode(mp4.read_bytes()).decode()))
            except Exception as exc:  # noqa: BLE001 - 3D rendering is optional
                log(f"   3D video unavailable ({exc!r}); keeping the 2D replay")
        if dev is None:
            dev, _ = resimulate(tr, False)
        determinism.append({"episode": eid, "max_tip_deviation_m": dev})
    summary["renderer"] = os.environ.get("MUJOCO_GL") if videos else None
    summary["replay_determinism"] = determinism
    summary["elapsed_s"] = round(time.time() - t_start, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: "<array>" if isinstance(o, np.ndarray) else str(o)))
    (out / "report.html").write_text(report_html(summary, acts, comparisons, decisions, vlm_decisions, cycle, gifs, videos, args))
    if videos:
        import subprocess

        subprocess.run([sys.executable, str(Path(__file__).with_name("make_video.py")), str(out)], check=False)
    log(f"done in {summary['elapsed_s']} s -> {out / 'report.html'}" + (f" and {out / 'demo_walkthrough.mp4'}" if videos else ""))
    return 0


def esc(x) -> str:
    return html.escape(str(x))


def latency(d: dict) -> str:
    return "cache" if d["cached"] else f"{d['latency_s']:.1f} s"


def report_html(summary, acts, comparisons, decisions, vlm_decisions, cycle, gifs, videos, args) -> str:
    rows = "".join(
        f"<tr><td>{esc(n)}</td><td><b>{r.metrics['success_mean']:.2f}</b></td>"
        + "".join(f"<td>{v:.2f}</td>" for _, v in sorted(per_task(r).items()))
        + f"<td>{r.metrics['steps_mean']:.0f}</td><td>{r.metrics['n_episodes']}</td></tr>" for n, r in acts.items())
    task_cols = "".join(f"<th>{esc(t.replace('table_', ''))}</th>" for t in sorted(per_task(next(iter(acts.values())))))
    cmp_rows = "".join(f"<tr><td>{esc(a)}</td><td>{esc(b)}</td><td>{c.n_pairs}</td><td>{c.candidate_wins} : {c.parent_wins}</td>"
                       f"<td>{c.p_value:.2g}</td></tr>" for a, b, c in comparisons)
    dec_rows = "".join(
        f"<tr class='{'ok' if d['status'] == 'success' else 'bad'}'><td>{esc(d['task'].replace('table_', ''))}</td><td>{esc(d['instruction'])}</td>"
        f"<td>{esc(', '.join(d['colours']))}</td><td>{esc(d['truth'])}</td><td>{esc(d['choice'])}</td><td>{esc(d['rationale'])}</td>"
        f"<td>{latency(d)}</td></tr>"
        for d in decisions[:12])
    cand_rows = ""
    for e in cycle.evaluated:
        if "rejected" in e:
            verdict, cls = f"rejected before evaluation: {esc(e['rejected'])}", "bad"
        else:
            g = e["gate"]
            v = g["validation"]
            verdict = ("PASS" if g["passed"] else "FAIL: " + esc("; ".join(g["reasons"])))
            verdict += f" (success {v['candidate_success']:.2f} vs {v['parent_success']:.2f}, shield {v['candidate_safety']:.2f} vs {v['parent_safety']:.2f}, p={v['paired']['p_value']:.2g})"
            cls = "ok" if g["passed"] else "bad"
        star = " &larr; promoted" if e["id"] == cycle.promoted else ""
        cand_rows += f"<tr class='{cls}'><td>{esc(e['id'])}</td><td><code>{esc(e['patch'])}</code></td><td>{verdict}{star}</td></tr>"
    vlm_figs = ""
    for d in [d for d in vlm_decisions if d["task"].endswith("indirect") and d["marked_image"] is not None][:4]:
        buf = io.BytesIO()
        from PIL import Image

        Image.fromarray(np.asarray(d["marked_image"], dtype=np.uint8)).save(buf, format="PNG")
        ok = "&#10004;" if d["status"] == "success" else "&#10008;"
        vlm_figs += (f"<figure style='max-width:250px'><img src='data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}'/>"
                     f"<figcaption>{ok} &ldquo;{esc(d['instruction'])}&rdquo;<br/>chose <b>{esc(d['choice'])}</b> (truth {esc(d['truth'])})"
                     f"<br/><i>{esc(d['rationale'])}</i></figcaption></figure>")
    vlm = summary.get("vlm_calls", {})
    vlm_section = "" if not vlm_decisions else f"""<h2>Act 3b: a vision-language model that only sees the camera</h2>
<p class="lead">Same agent, chooser swapped to <code>chooser.vlm</code>: the model receives the top-camera image with set-of-mark number tags
(object positions projected with the camera parameters the environment publishes) and the instruction, but no colour values.
The image is only available while the pads are visible; afterwards the agent acts from spatial memory exactly as before.
{vlm.get('n', 0)} choices, {vlm.get('cached', 0)} cached, mean live latency {vlm.get('mean_latency_s', 0):.1f} s. These are the images the model saw,
stored in the trace:</p>
{vlm_figs}"""
    ho = cycle.heldout or {}
    gif_html = "".join(f"<figure><img src='data:image/gif;base64,{g}'/><figcaption>{esc(t)}<br/><code>{esc(eid)}</code></figcaption></figure>"
                       for t, eid, g in gifs)
    video_html = "".join(f"<figure class='wide'><video controls autoplay muted loop playsinline src='data:video/mp4;base64,{v}'></video>"
                         f"<figcaption>{esc(t)}<br/><code>{esc(eid)}</code></figcaption></figure>" for t, eid, v in videos)
    det = summary.get("replay_determinism", [])
    det_txt = ", ".join(f"{d['max_tip_deviation_m']:.1e} m" for d in det) or "n/a"
    rsi = summary["rsi"]
    llm = summary.get("llm_calls", {})
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Hylo demo</title>
<style>
body{{font:15px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1100px;margin:32px auto;padding:0 16px;color:#1f2328}}
h1{{margin-bottom:0}} h2{{margin-top:36px;border-bottom:1px solid #ddd;padding-bottom:4px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}} td,th{{border:1px solid #e1e4e8;padding:5px 8px;text-align:left;vertical-align:top}}
th{{background:#f6f8fa}} tr.ok td:first-child{{border-left:4px solid #2da44e}} tr.bad td:first-child{{border-left:4px solid #cf222e}}
.lead{{color:#57606a}} figure{{display:inline-block;margin:8px;text-align:center;font-size:12px;max-width:440px}} figure.wide{{max-width:100%;display:block}} img,video{{max-width:100%}}
.pill{{display:inline-block;background:#eef;border-radius:10px;padding:1px 8px;margin-right:6px;font-size:12px}} code{{font-size:12px}}
</style></head><body>
<h1>Hylo: language, memory and self-improvement in one harness</h1>
<p class="lead">MuJoCo table-top task. A planar arm must touch the pad named by a natural-language instruction.
Pads are visible only for the first 5 control steps (0.25 s); after that the agent must act from memory. Every action of every agent
passes through the same SafetyShield, and every observation, decision, memory access and safety review is written to a replayable trace.</p>
<p><span class="pill">simulator: MuJoCo</span><span class="pill">reasoner LLM/VLM: {esc(summary['model'] or 'disabled')} (local)</span>
<span class="pill">validation seeds per task: {acts[next(iter(acts))].metrics['n_episodes'] // max(1, len(per_task(next(iter(acts.values())))))}</span>
<span class="pill">wall time: {summary['elapsed_s']} s</span></p>

<h2>Acts 1&ndash;3: same task, swap components by configuration</h2>
<img src="data:image/png;base64,{bars(acts)}"/>
<table><tr><th>agent</th><th>success</th>{task_cols}<th>mean steps</th><th>episodes</th></tr>{rows}</table>
<p>Paired comparisons on identical task seeds (one-sided exact McNemar):</p>
<table><tr><th>candidate</th><th>reference</th><th>pairs</th><th>wins (cand : ref)</th><th>p</th></tr>{cmp_rows}</table>
<p>Act 3 differs from Act 2 by one line of configuration: <code>chooser: {{type: chooser.llm}}</code> instead of the default rule-based colour matcher;
Act 3b uses <code>chooser: {{type: chooser.vlm}}</code> and a benchmark variant whose observations include the top camera.
Turning memory off (2a) leaves the agent with nothing to act on once the pads are hidden.</p>

<h2>What the LLM reasoner decided (from the trace)</h2>
<p class="lead">{llm.get('n', 0)} choices, {llm.get('cached', 0)} served from the reproducibility cache; mean live latency {llm.get('mean_latency_s', 0):.1f} s. Ground truth is shown here for the viewer only; the agent never receives it.</p>
<table><tr><th>task</th><th>instruction</th><th>pad colours</th><th>truth</th><th>chosen</th><th>rationale</th><th>latency</th></tr>{dec_rows}</table>

{vlm_section}

<h2>Act 4: offline self-improvement with a validation gate</h2>
<p class="lead">On the push task the controller has 3.5 s. The starting controller (kp=0.03) is too slow. Diagnosis of the train split:
<code>{esc(rsi['diagnosis'])}</code>. Candidates are validated on separate seeds, paired against the current version; a candidate must be
significantly better, must not regress any task, and must not increase SafetyShield interventions by more than 1 per episode.</p>
<table><tr><th>candidate</th><th>patch</th><th>verdict</th></tr>{cand_rows}</table>
<p>Held-out report (not used for selection): candidate {ho.get('candidate_success', float('nan')):.2f} vs parent {ho.get('parent_success', float('nan')):.2f}
({ho.get('paired', {}).get('candidate_wins', '-')} : {ho.get('paired', {}).get('parent_wins', '-')}, p={ho.get('paired', {}).get('p_value', float('nan')):.2g}).
Promoted patch <code>{esc(rsi['promoted_patch'])}</code>; then a simulated field report triggered a rollback to
<code>{esc(rsi['after_rollback_patch'] or 'the baseline')}</code>. Ledger: <code>{esc(' → '.join(rsi['ledger']))}</code>.</p>

<h2>Act 5: replay from traces</h2>
<p class="lead">3D videos: the recorded actions are re-simulated in a fresh MuJoCo instance (front and top cameras); captions come from the trace
(what the reasoner decided, whether the pads were visible to the agent, SafetyShield interventions). The re-simulated fingertip trajectory matches
the recorded one with max deviation {det_txt}.</p>
{video_html or '<p><i>3D rendering unavailable on this machine; see the 2D replays below.</i></p>'}
<p class="lead">2D replays drawn from the trace files alone (no simulator):</p>
{gif_html}

<h2>Reproduce</h2>
<pre>uv sync --all-groups
uv run --group demo python examples/external_demo/run_demo.py{' --quick' if args.quick else ''}</pre>
<p class="lead">Each run directory contains <code>manifest.json</code> (config digest, package versions, git SHA), <code>evallog.json</code> and one
JSONL trace per episode. LLM responses are cached by prompt hash in <code>runs/demo-llm-cache/</code>.</p>
</body></html>"""


if __name__ == "__main__":
    sys.exit(main())
