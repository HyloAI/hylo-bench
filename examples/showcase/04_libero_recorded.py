"""LIBERO with SmolVLA from recorded runs: success per suite with Wilson intervals, per-task outcomes, an episode strip.

    uv run python examples/showcase/04_libero_recorded.py [--quick] [--out DIR] [--store DIR] [--run]

LIBERO evaluations take hours on a CPU, so this example does not re-run them. It reads RECORDED runs of
lerobot/smolvla_libero (450M, CPU) from a run store and summarises them: for each LIBERO suite the most recent
run that covers all 10 tasks (5 validation init states per task, 50 episodes), success with 95% Wilson
intervals, steps, shield interventions and wall time per episode taken from the traces; per-task outcomes for
LIBERO-10; and a strip of camera frames from one recorded LIBERO-10 episode. Nothing is simulated; every number is
labelled with the run it comes from. These are 50-episode validation runs, not the official 500-episode protocol.

--store: the run store to read (default: this machine's store from `hylo profile show`, else $HYLO_LIBERO_RECORDS).
--run: also a live check: 2 episodes (libero_spatial tasks 0 and 1, init state 0) through the `libero-sim` and
`lerobot` runtimes of this machine's profile (bench's envs/libero/setup.sh), compared with the same two
episodes in the recorded run. SmolVLA on CPU is not bit-reproducible run to run, so outcomes and step counts
may differ from the recording. About 40-70 s; skipped with a message when the runtimes are not set up.
--quick: no figures.

Typical runtime: about 1 s (--run: +40 to 70 s). Requires hylo-bench, matplotlib and Pillow, and a store with recorded
libero-smolvla-* runs (skips with a message otherwise). Output: runs/examples/libero-recorded/ (result.json,
libero_suites.png, libero_strip.png).
"""

from __future__ import annotations

import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

from _common import REPO, Timer, ci, parser, pct, table, write_result

SLUG = "libero-recorded"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def default_store() -> Path | None:
    try:
        from hylo.core.profile import load_profile

        store = load_profile().store
    except Exception:            # no profile on this machine
        store = None
    store = store or os.environ.get("HYLO_LIBERO_RECORDS")
    return Path(store).expanduser() if store else None


def load_runs(store: Path) -> list[dict]:
    runs = []
    for log in sorted(store.glob("libero-smolvla-*/evallog.json")):
        e = json.loads(log.read_text())
        m = json.loads((log.parent / "manifest.json").read_text())
        if not e["benchmark"]["id"].startswith("libero/"):
            continue
        runs.append({"dir": log.parent, "log": e, "manifest": m, "suite": e["benchmark"]["id"].split("/", 1)[1],
                     "tasks": len({s["task_id"] for s in e["samples"]}), "n": len(e["samples"])})
    return runs


def wall_seconds(run: dict) -> list[float]:
    """Per-episode wall time from the first to the last trace event (each worker runs one episode at a time)."""
    out = []
    for s in run["log"]["samples"]:
        lines = (run["dir"] / s["trace"]).read_text().splitlines()
        if lines:
            out.append(json.loads(lines[-1])["wall_time"] - json.loads(lines[0])["wall_time"])
    return out


def summarise(run: dict) -> dict:
    from hylo.core.scorecard import wilson_interval

    samples = run["log"]["samples"]
    succ = [s for s in samples if s["status"] == "success"]
    walls = wall_seconds(run)
    m = run["manifest"]
    return {"suite": run["suite"], "run_id": run["dir"].name, "recorded": "yes",
            "recorded_at": datetime.fromtimestamp(m["created_at"], timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "policy": m["config"]["agent"].get("params", {}).get("params", {}).get("repo_id")
                      or run["log"]["agent"]["components"].get("policy"),
            "code_git_sha": (m.get("code") or {}).get("git_sha"), "config_digest": m.get("config_digest"),
            "split": run["log"]["benchmark"]["split"], "tasks": run["tasks"], "episodes": len(samples),
            "successes": len(succ), "success": len(succ) / len(samples),
            "ci95": wilson_interval(len(succ), len(samples)),
            "steps_mean": statistics.fmean(s["steps"] for s in samples),
            "steps_mean_success": statistics.fmean(s["steps"] for s in succ) if succ else None,
            "interventions_per_episode": run["log"]["results"].get("safety_interventions_mean"),
            "wall_s_per_episode_median": round(statistics.median(walls), 1) if walls else None,
            "per_task": {t.split("/")[-1]: "".join("S" if s["status"] == "success" else "F"
                                                   for s in sorted((x for x in samples if x["task_id"] == t),
                                                                   key=lambda x: x["seed"]))
                         for t in sorted({s["task_id"] for s in samples})}}


def suites_figure(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, blue, surface = "#0b0b0b", "#52514e", "#2a78d6", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(6.4, 2.4), dpi=110)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    for y, r in enumerate(reversed(rows)):
        lo, hi = r["ci95"]
        ax.barh(y, 100 * r["success"], height=0.55, color=blue, zorder=2)
        ax.errorbar(100 * r["success"], y, xerr=[[100 * (r["success"] - lo)], [100 * (hi - r["success"])]],
                    fmt="none", ecolor=ink, elinewidth=1, capsize=3, zorder=3)
        ax.text(100 * hi + 1.5, y, f"{100 * r['success']:.0f}%  ({r['successes']}/{r['episodes']})", va="center",
                fontsize=8, color=ink)
    ax.set_yticks(range(len(rows)), [r["suite"] for r in reversed(rows)], fontsize=8.5, color=ink)
    ax.set_xlim(0, 115)
    ax.set_xlabel("success %, 50 recorded episodes per suite (10 tasks x 5 init states), 95% Wilson interval",
                  fontsize=7.5, color=muted)
    ax.set_title("SmolVLA (lerobot/smolvla_libero) on LIBERO, CPU: recorded runs", fontsize=9.5, color=ink, loc="left")
    ax.grid(axis="x", color="#e4e3df", linewidth=0.6, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=muted, labelsize=8, length=0)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def strip_figure(run: dict, path: Path, frames: int = 6) -> dict:
    """Frames of one recorded successful episode: the hardest task (fewest successes) that has one."""
    from PIL import Image, ImageDraw

    from hylo.core.trace import TraceStore

    samples = run["log"]["samples"]
    rate = {t: sum(s["status"] == "success" for s in samples if s["task_id"] == t) for t in {s["task_id"] for s in samples}}
    ep = min((s for s in samples if s["status"] == "success"), key=lambda s: (rate[s["task_id"]], s["task_id"], s["seed"]))
    obs = [o for o in TraceStore(run["dir"]).read(ep["episode_id"]).of_type("observation") if o["payload"].get("images")]
    pick = [obs[round(i * (len(obs) - 1) / (frames - 1))] for i in range(frames)]
    size, pad, label = 160, 4, 18
    img = Image.new("RGB", (frames * (size + pad) - pad, size + label), (252, 252, 251))
    draw = ImageDraw.Draw(img)
    steps = []
    for i, o in enumerate(pick):
        step = round(o["sim_time"] * 20) if o.get("sim_time") is not None else None    # 20 Hz control
        steps.append(step)
        img.paste(Image.fromarray(o["payload"]["images"]["agentview"]).resize((size, size)), (i * (size + pad), label))
        draw.text((i * (size + pad) + 2, 3), f"t = {o['sim_time']:.1f} s", fill=(82, 81, 78))
    img.save(path, optimize=True)
    return {"episode_id": ep["episode_id"], "task": ep["task_id"].split("/")[-1], "seed": ep["seed"],
            "steps": ep["steps"], "task_successes_of_5": rate[ep["task_id"]], "frames_at_sim_time_s":
            [round(o["sim_time"], 1) for o in pick]}


def live_check(out: Path, recorded: dict | None) -> dict:
    """2 live episodes through this machine's LIBERO runtimes; compared with the same episodes recorded."""
    from hylo.core.config import ExperimentConfig
    from hylo.core.profile import load_profile
    from hylo.eval.experiment import run_experiment

    prof = load_profile()
    missing = [r for r in ("libero-sim", "lerobot") if r not in prof.runtimes
               or not Path(os.path.expanduser(prof.runtimes[r].python)).exists()]
    if missing:
        return {"skipped": f"runtimes not set up: {missing} (run envs/libero/setup.sh)"}
    cfg = ExperimentConfig.load(REPO / "examples" / "configs" / "libero_smolvla.yaml").with_overrides(
        {"benchmark.params.n_validation": 1, "benchmark.params.task_indices": [0, 1], "workers": 2})
    with Timer() as t:
        r = run_experiment(cfg, run_root=out / "live")
    eps = []
    rec = {s["key"]: s for s in recorded["log"]["samples"]} if recorded else {}
    for s in r.log["samples"]:
        old = rec.get(s["key"])
        eps.append({"task": s["task_id"].split("/")[-1], "seed": s["seed"], "live": s["status"], "live_steps": s["steps"],
                    "recorded": old and old["status"], "recorded_steps": old and old["steps"]})
    return {"seconds": round(t.s, 1), "run_dir": str(r.run_dir), "success": r.metrics["success_mean"], "episodes": eps}


def main() -> None:
    ap = parser(__doc__, SLUG)
    ap.add_argument("--store", type=Path, default=default_store(), help="the run store with recorded runs")
    ap.add_argument("--run", action="store_true", help="also a 2-episode live check (needs envs/libero)")
    a = ap.parse_args()
    out = a.out
    if not a.store or not a.store.is_dir():
        print(f"no run store at {a.store}: nothing recorded to summarise (pass --store)")
        write_result(out, {"example": SLUG, "skipped": f"no store at {a.store}"})
        return
    runs = load_runs(a.store)
    full = {}
    for r in runs:                                      # the most recent run covering every task of a suite
        if r["tasks"] == 10 and (r["suite"] not in full or r["manifest"]["created_at"] > full[r["suite"]]["manifest"]["created_at"]):
            full[r["suite"]] = r
    if not full:
        print(f"no recorded libero-smolvla-* runs covering a whole suite under {a.store}")
        write_result(out, {"example": SLUG, "skipped": "no recorded runs"})
        return
    rows = [summarise(full[s]) for s in SUITES if s in full]
    others = [{"run_id": r["dir"].name, "suite": r["suite"], "episodes": r["n"], "tasks": r["tasks"],
               "success": r["log"]["results"]["success_mean"]} for r in runs if r not in full.values()]
    pooled_s, pooled_n = sum(r["successes"] for r in rows), sum(r["episodes"] for r in rows)
    from hylo.core.scorecard import wilson_interval

    pooled = {"successes": pooled_s, "episodes": pooled_n, "success": pooled_s / pooled_n,
              "ci95": wilson_interval(pooled_s, pooled_n)}
    strip = None
    if not a.quick:
        out.mkdir(parents=True, exist_ok=True)
        suites_figure(rows, out / "libero_suites.png")
        if "libero_10" in full:
            strip = strip_figure(full["libero_10"], out / "libero_strip.png")

    print(f"RECORDED runs from {a.store} (nothing re-simulated)\n")
    print(table(["suite", "success %", "95% CI", "succ/ep", "mean steps", "steps (succ.)", "interv./ep",
                 "wall s/ep (median)", "recorded", "run"],
                [(r["suite"], pct(r["success"]), ci(r["ci95"]), f"{r['successes']}/{r['episodes']}",
                  round(r["steps_mean"]), round(r["steps_mean_success"]), r["interventions_per_episode"],
                  r["wall_s_per_episode_median"], r["recorded_at"][:10], r["run_id"].rsplit("-", 1)[-1]) for r in rows]))
    print(f"\npooled over {len(rows)} suites: {pct(pooled['success'])}% ({pooled_s}/{pooled_n}), 95% CI {ci(pooled['ci95'])}")
    if "libero_10" in full:
        l10 = next(r for r in rows if r["suite"] == "libero_10")
        print("\nLIBERO-10 per task (init states 0-4, S = success):")
        print(table(["task", "outcomes", "success"], [(t[:70], o, f"{o.count('S')}/5") for t, o in l10["per_task"].items()]))
    if strip:
        print(f"\nepisode strip: {strip['task']} init {strip['seed']}, success in {strip['steps']} steps "
              f"({strip['task_successes_of_5']}/5 on this task) -> {out / 'libero_strip.png'}")
    if others:
        print(f"\nother recorded runs (partial suites, not summarised): {len(others)}")
    live = None
    if a.run:
        live = live_check(out, full.get("libero_spatial"))
        if "skipped" in live:
            print(f"\nlive check skipped: {live['skipped']}")
        else:
            print(f"\nLIVE check ({live['seconds']} s):")
            print(table(["task", "init", "live", "recorded"], [(e["task"][:60], e["seed"], f"{e['live']} ({e['live_steps']})",
                         f"{e['recorded']} ({e['recorded_steps']})") for e in live["episodes"]]))
    write_result(out, {"example": SLUG, "source": "recorded", "store": str(a.store), "suites": rows,
                       "pooled": pooled, "episode_strip": strip, "other_recorded_runs": others, "live_check": live})


if __name__ == "__main__":
    main()
