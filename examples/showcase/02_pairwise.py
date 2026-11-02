"""Pairwise comparison of two scorecards, episode by episode: discordant pairs and the exact McNemar test.

    uv run python examples/showcase/02_pairwise.py [--quick] [--out DIR]

Scores reference policies under `mujoco-reach` (100 held-out episodes) and compares scorecards pairwise with
`hylo.bench.run.compare` (what `hylo bench compare` prints): the episodes both share are paired by task and seed,
and a one-sided exact McNemar test runs on the pairs where exactly one agent succeeded. For contrast the script
also reports whether the two 95% Wilson intervals overlap and an unpaired one-sided Fisher exact test on the same
counts, which ignores the pairing. An A/A control (the same agent scored twice) must give zero discordant pairs.

Pairs: quick: bc + temporal ensemble vs bc, and the A/A control. Full: also flow + steering (4 samples, medoid)
vs bc, and flow + steering vs plain flow. Figure: pairwise.png, the headline pair episode by episode.

Typical runtime: --quick about 10 s, full about 30 s (128-core CPU, 24 workers). Requires hylo-bench,
hylo-env-mujoco, hylo-policies, hylo-learn, matplotlib. Output: runs/examples/pairwise/ (scorecards/,
pairwise.png, result.json).
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

from _common import Timer, ci, parser, pct, policy, table, train, write_result

SLUG = "pairwise"
WORKERS = 24


def fisher_one_sided(a: int, n1: int, b: int, n2: int) -> float:
    """P[X >= a], X hypergeometric: unpaired test that the first rate is higher (ignores which episodes pair up)."""
    k, n = a + b, n1 + n2
    return sum(math.comb(n1, x) * math.comb(n2, k - x) for x in range(a, min(n1, k) + 1)) / math.comb(n, k)


def overlap(c1, c2) -> bool:
    return c1[0] <= c2[1] and c2[0] <= c1[1]


def strip(cand, base, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    ink, muted, blue, fail, orange, surface = "#0b0b0b", "#52514e", "#2a78d6", "#e4e3df", "#eb6834", "#fcfcfb"
    keys = sorted(k for k in cand.episodes if k in base.episodes)
    fig, ax = plt.subplots(figsize=(9, 2.0), dpi=110)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    for i, k in enumerate(keys):
        for row, card in ((1, cand), (0, base)):
            ax.add_patch(Rectangle((i + 0.08, row + 0.1), 0.84, 0.8, color=blue if card.episodes[k] else fail, lw=0))
        if bool(cand.episodes[k]) != bool(base.episodes[k]):
            ax.plot(i + 0.5, 2.25, marker="v", ms=4.5, color=orange if cand.episodes[k] else ink, lw=0)
    n_fixed = sum(1 for k in keys if "fixed" in k)
    ax.axvline(n_fixed, color=muted, lw=0.8)
    ax.text(n_fixed / 2, -0.45, "fixed target (50 seeds)", ha="center", fontsize=8, color=muted)
    ax.text(n_fixed + (len(keys) - n_fixed) / 2, -0.45, "random target (50 seeds)", ha="center", fontsize=8, color=muted)
    ax.set_xlim(0, len(keys))
    ax.set_ylim(-0.8, 2.6)
    ax.set_yticks([0.5, 1.5], [base.policy, cand.policy], fontsize=8.5, color=ink)
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"{cand.policy} vs {base.policy}: blue = success; markers = discordant pairs "
                 f"(orange: only {cand.policy}, black: only {base.policy})", fontsize=8.5, color=ink, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def main() -> None:
    a = parser(__doc__, SLUG).parse_args()
    from hylo.bench.protocol import load_protocol
    from hylo.bench.run import compare, run_protocol

    out = a.out
    for d in ("scorecards", "runs"):
        shutil.rmtree(out / d, ignore_errors=True)
    protocol = load_protocol("mujoco-reach")
    with Timer() as t_all:
        p = train(out, ("bc",) if a.quick else ("bc", "flow"))
        agents = {"bc": {"type": "policy", "params": {"policy": policy(p["bc"])}},
                  "bc (again)": {"type": "policy", "params": {"policy": policy(p["bc"])}},
                  "bc+temporal-ensemble": {"type": "temporal_ensemble", "params": {"policy": policy(p["bc"]), "m": 0.1}}}
        pairs = [("bc+temporal-ensemble", "bc"), ("bc (again)", "bc")]
        if not a.quick:
            agents["flow"] = {"type": "policy", "params": {"policy": policy(p["flow"])}}
            agents["flow+steering-4"] = {"type": "steered", "params": {
                "sampler": policy(p["flow"]), "steering": {"n_samples": 4, "selector": "medoid"}}}
            pairs = [("flow+steering-4", "bc"), *pairs[:1], ("flow+steering-4", "flow"), pairs[1]]
        cards = {n: run_protocol(protocol, s, n, out=out, overrides={"workers": WORKERS})[0] for n, s in agents.items()}

    rows, results = [], []
    for cn, bn in pairs:
        c, b = cards[cn], cards[bn]
        r = compare(c, b)
        both = sum(1 for k in c.episodes if k in b.episodes and c.episodes[k] and b.episodes[k])
        neither = r.n_pairs - both - r.candidate_wins - r.parent_wins
        sc, sb = round(c.metrics["success_mean"] * 100), round(b.metrics["success_mean"] * 100)
        fisher = fisher_one_sided(sc, 100, sb, 100)
        by_task = {t.split("/")[-1]: [sum(1 for k in c.episodes if k.startswith(t) and c.episodes[k] and not b.episodes[k]),
                                      sum(1 for k in c.episodes if k.startswith(t) and b.episodes[k] and not c.episodes[k])]
                   for t in c.per_task}
        results.append({"candidate": cn, "baseline": bn, "candidate_success": c.metrics["success_mean"],
                        "baseline_success": b.metrics["success_mean"], "candidate_ci95": c.success_ci95,
                        "baseline_ci95": b.success_ci95, "intervals_overlap": overlap(c.success_ci95, b.success_ci95),
                        "n_pairs": r.n_pairs, "both_succeed": both, "neither": neither,
                        "candidate_only": r.candidate_wins, "baseline_only": r.parent_wins,
                        "mcnemar_p_one_sided": r.p_value, "fisher_p_one_sided_unpaired": fisher,
                        "discordant_by_task": by_task})
        rows.append((f"{cn} vs {bn}", f"{pct(c.metrics['success_mean'])} vs {pct(b.metrics['success_mean'])}",
                     "yes" if overlap(c.success_ci95, b.success_ci95) else "no", f"{both}/{neither}",
                     f"{r.candidate_wins}–{r.parent_wins}", f"{r.p_value:.2g}", f"{fisher:.2g}"))
    headline = results[0]
    strip(cards[headline["candidate"]], cards[headline["baseline"]], out / "pairwise.png")

    print(table(["pair", "success %", "CIs overlap", "both/neither", "discordant (cand–base)", "McNemar p",
                 "Fisher p (unpaired)"], rows))
    print("\nWilson 95% intervals: " + "; ".join(f"{n} {ci(c.success_ci95)}" for n, c in cards.items()))
    print(f"discordant pairs by task, headline pair: {headline['discordant_by_task']}")
    print(f"total {t_all.s:.0f} s; {out / 'pairwise.png'}")
    write_result(out, {"example": SLUG, "quick": a.quick, "protocol": protocol.name,
                       "protocol_digest": protocol.digest(), "pairs": results, "seconds_total": round(t_all.s, 1),
                       "figure": str(out / "pairwise.png")})


if __name__ == "__main__":
    main()
