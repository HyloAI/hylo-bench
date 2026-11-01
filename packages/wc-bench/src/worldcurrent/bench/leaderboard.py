"""Leaderboards: scorecards of one protocol (version and digest), ranked by success."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from worldcurrent.core.scorecard import Scorecard


def collect(root: str | Path) -> list[Scorecard]:
    cards = []
    for p in sorted(Path(root).rglob("*.json")):
        try:
            cards.append(Scorecard.load(p))
        except (ValueError, OSError):        # not a scorecard
            continue
    return cards


def boards(cards: list[Scorecard]) -> dict[tuple[str, str, str], list[Scorecard]]:
    """(protocol, version, digest) -> scorecards, best first. Different digests are different boards."""
    out: dict[tuple[str, str, str], list[Scorecard]] = defaultdict(list)
    for c in cards:
        out[(c.protocol, c.protocol_version, c.protocol_digest)].append(c)
    return {k: sorted(v, key=lambda c: (-(c.metrics.get("success_mean") or 0.0), c.policy)) for k, v in sorted(out.items())}


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def markdown(cards: list[Scorecard]) -> str:
    parts = []
    for (name, version, digest), rows in boards(cards).items():
        parts.append(f"### {name} v{version}  (`{digest}`)\n")
        parts.append("| # | policy | success % | 95% CI | episodes | mean steps |")
        parts.append("|---|---|---|---|---|---|")
        for i, c in enumerate(rows, 1):
            ci = f"{_pct(c.success_ci95[0])}–{_pct(c.success_ci95[1])}" if c.success_ci95 else "–"
            steps = c.metrics.get("steps_mean")
            parts.append(f"| {i} | {c.policy} | {_pct(c.metrics.get('success_mean'))} | {ci} | "
                         f"{c.metrics.get('n_scored', c.metrics.get('n_episodes'))} | {'–' if steps is None else f'{steps:.0f}'} |")
        parts.append("")
    return "\n".join(parts) if parts else "no scorecards\n"


__all__ = ["boards", "collect", "markdown"]
