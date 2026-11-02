"""Protocol -> scorecards -> leaderboard, and why a scorecard of another protocol digest is not ranked.

    uv run python examples/showcase/01_leaderboard.py [--quick] [--out DIR]

1. Records expert demonstrations on the campaign development seeds and trains reference policies on them
   (quick: policy.knn, policy.bc; full: also policy.flow, policy.tokens).
2. Scores every agent under the built-in `mujoco-reach` protocol (100 held-out episodes): the IK controller at a
   good and at a weak gain, a random agent and the learned policies (full: also bc + temporal ensemble and flow +
   steering). Each run writes a scorecard; the leaderboard ranks them (Markdown table + leaderboard.png).
3. Edits the protocol the way someone might by hand: same name, same version, one measured field changed
   (safety.max_delta: 0.05). Its digest changes, so `bc` scored under it lands on a separate board, `compare` refuses
   to pair it with the real `bc` scorecard, and `run_protocol` refuses the same change as an override.

Typical runtime: --quick about 17 s, full about 50 s (128-core CPU, 24 workers). Requires hylo-bench,
hylo-env-mujoco, hylo-policies, hylo-learn, matplotlib. Output: runs/examples/leaderboard/ (scorecards/,
leaderboard.md, leaderboard.png, result.json).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from _common import EXPERT, Timer, ci, parser, pct, policy, table, train, write_result

SLUG = "leaderboard"
WORKERS = 24


def agents(quick: bool, out: Path) -> dict[str, dict]:
    methods = ("knn", "bc") if quick else ("knn", "bc", "flow", "tokens")
    p = train(out, methods)
    a = {"expert-ik": EXPERT,
         "ik-weak-gain": {"type": "baseline.planar_ik", "params": {"kp": 0.025}},
         "random": {"type": "baseline.random", "params": {"dim": 2}},
         **{m: {"type": "policy", "params": {"policy": policy(p[m])}} for m in methods}}
    if not quick:
        a["bc+temporal-ensemble"] = {"type": "temporal_ensemble", "params": {"policy": policy(p["bc"]), "m": 0.1}}
        a["flow+steering-4"] = {"type": "steered", "params": {"sampler": policy(p["flow"]),
                                                              "steering": {"n_samples": 4, "selector": "medoid"}}}
    return a


def render(board: list, other: list, digest: str, other_digest: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, blue, grey, surface = "#0b0b0b", "#52514e", "#2a78d6", "#a3a29c", "#fcfcfb"
    rows = [(c, blue) for c in board] + [(c, grey) for c in other]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(rows) + 1.3), dpi=110)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ys = list(range(len(rows)))[::-1]
    if other:                                            # a gap between the board and the other digest
        ys = [y + (1 if i < len(board) else 0) for i, y in enumerate(ys)]
    for y, (c, color) in zip(ys, rows):
        m = c.metrics["success_mean"]
        lo, hi = c.success_ci95
        ax.barh(y, 100 * m, height=0.55, color=color, zorder=2)
        ax.errorbar(100 * m, y, xerr=[[100 * (m - lo)], [100 * (hi - m)]], fmt="none", ecolor=ink, elinewidth=1,
                    capsize=3, zorder=3)
        ax.text(100 * hi + 1.5, y, f"{100 * m:.0f}%", va="center", fontsize=8.5, color=ink)
    labels = [c.policy for c, _ in rows]
    ax.set_yticks(ys, labels, fontsize=9, color=ink)
    if other:
        for t in ax.get_yticklabels()[len(board):]:
            t.set_color(muted)
        ax.text(0, ys[len(board)] + 0.6, f"protocol edited by hand: same name and version, digest {other_digest} "
                "(separate board, not ranked)", fontsize=7.5, color=muted)
    ax.set_xlim(0, 112)
    ax.set_xlabel("success %, 100 held-out episodes, 95% Wilson interval", fontsize=8.5, color=muted)
    ax.set_title(f"mujoco-reach v1 ({digest})", fontsize=10, color=ink, loc="left")
    ax.grid(axis="x", color="#e4e3df", linewidth=0.6, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grey)
    ax.tick_params(colors=muted, labelsize=8, length=0)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def main() -> None:
    a = parser(__doc__, SLUG).parse_args()
    from hylo.bench import leaderboard
    from hylo.bench.protocol import load_protocol
    from hylo.bench.run import compare, run_protocol

    out = a.out
    for d in ("scorecards", "runs", "refused"):          # this example's own boards; demos and artifacts are reused
        shutil.rmtree(out / d, ignore_errors=True)
    protocol = load_protocol("mujoco-reach")
    with Timer() as t_all:
        specs = agents(a.quick, out)
        timings = {}
        for name, agent in specs.items():
            with Timer() as t:
                run_protocol(protocol, agent, name, out=out, overrides={"workers": WORKERS})
            timings[name] = round(t.s, 1)

        # the same protocol with one measured field changed: same name, same version, another digest
        edited = protocol.model_copy(update={"safety": {"max_delta": 0.05}})
        bc_edited, _ = run_protocol(edited, specs["bc"], "bc (edited protocol)", out=out, overrides={"workers": WORKERS})
        cards = leaderboard.collect(out / "scorecards")
        boards = leaderboard.boards(cards)
        board = boards[(protocol.name, protocol.version, protocol.digest())]
        other = boards[(edited.name, edited.version, edited.digest())]
        bc = next(c for c in board if c.policy == "bc")
        try:
            compare(bc_edited, bc)
            refused_compare = None
        except ValueError as e:
            refused_compare = str(e)
        try:
            run_protocol(protocol, specs["bc"], "bc-override", out=out / "refused", overrides={"safety.max_delta": 0.05})
            refused_override = None
        except ValueError as e:
            refused_override = str(e)
        same_digest = protocol.model_copy(update={"workers": 4}).digest() == protocol.digest()

    md = leaderboard.markdown(cards)
    (out / "leaderboard.md").write_text(md)
    render(board, other, protocol.digest(), edited.digest(), out / "leaderboard.png")

    print(table(["#", "policy", "success %", "95% CI", "mean steps", "scored in s"],
                [(i, c.policy, pct(c.metrics["success_mean"]), ci(c.success_ci95), round(c.metrics["steps_mean"]),
                  timings.get(c.policy, "")) for i, c in enumerate(board, 1)]))
    print(f"\nprotocol digest {protocol.digest()}; edited protocol (safety.max_delta 0.05) digest {edited.digest()}")
    print(f"boards: {len(boards)}; bc on the edited protocol: {pct(bc_edited.metrics['success_mean'])}% "
          f"(vs {pct(bc.metrics['success_mean'])}%), not ranked with the others")
    print(f"compare(bc edited, bc): refused: {refused_compare}")
    print(f"override safety.max_delta: refused: {refused_override}")
    print(f"workers is not measured: digest unchanged with workers=4: {same_digest}")
    print(f"total {t_all.s:.0f} s; {out / 'leaderboard.png'}")

    write_result(out, {
        "example": SLUG, "quick": a.quick, "protocol": protocol.name, "protocol_digest": protocol.digest(),
        "leaderboard": [{"rank": i, "policy": c.policy, "success": c.metrics["success_mean"],
                         "ci95": c.success_ci95, "steps_mean": c.metrics["steps_mean"],
                         "per_task": {k: v.success for k, v in c.per_task.items()}} for i, c in enumerate(board, 1)],
        "edited_protocol": {"digest": edited.digest(), "change": {"safety": {"max_delta": 0.05}},
                            "bc_success": bc_edited.metrics["success_mean"], "boards": len(boards),
                            "compare_refused": refused_compare, "override_refused": refused_override},
        "workers_change_digest": not same_digest, "seconds_per_agent": timings, "seconds_total": round(t_all.s, 1),
        "figure": str(out / "leaderboard.png")})


if __name__ == "__main__":
    main()
