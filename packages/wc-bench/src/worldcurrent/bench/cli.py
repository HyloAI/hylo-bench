"""wc bench: benchmark protocols, scorecards and leaderboards.

    wc bench list                                         protocols that ship with wc-bench
    wc bench show PROTOCOL                                its definition and digest
    wc bench run PROTOCOL --policy SPEC --name NAME [--out DIR] [--set env.remote.python=...] [--workers N]
                                                          run a policy: runs in OUT/runs, OUT/scorecards/PROTOCOL/NAME.json
    wc bench leaderboard [DIR ...] [--json]               rank the scorecards found under DIR (default: scorecards)
    wc bench compare CANDIDATE.json BASELINE.json         paired one-sided McNemar on shared episodes

SPEC is an agent spec as JSON, an experiment config YAML (its agent is used) or a plugin name.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

import yaml


def main(argv: Sequence[str]) -> int:
    """benchmark protocols, scorecards and leaderboards"""
    from worldcurrent.core.scorecard import Scorecard

    from . import leaderboard
    from .protocol import builtin_protocols, load_protocol
    from .run import compare, policy_spec, run_protocol

    ap = argparse.ArgumentParser(prog="wc bench", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="action", required=True)
    sub.add_parser("list", help="protocols that ship with wc-bench")
    sh = sub.add_parser("show", help="a protocol's definition and digest")
    sh.add_argument("protocol")
    r = sub.add_parser("run", help="run a policy under a protocol and write its scorecard")
    r.add_argument("protocol", help="protocol name or YAML path")
    r.add_argument("--policy", required=True, help="agent spec JSON, experiment config YAML, or plugin name")
    r.add_argument("--name", required=True, help="the policy's name on the leaderboard")
    r.add_argument("--out", default=".", help="runs go to OUT/runs, scorecards to OUT/scorecards")
    r.add_argument("--workers", type=int)
    r.add_argument("--set", action="append", metavar="KEY=VALUE", help="execution only: env.remote.* (e.g. a venv path)")
    r.add_argument("--notes")
    lb = sub.add_parser("leaderboard", help="rank scorecards per protocol")
    lb.add_argument("dirs", nargs="*", default=["scorecards"])
    lb.add_argument("--json", action="store_true")
    cp = sub.add_parser("compare", help="paired comparison of two scorecards of one protocol")
    cp.add_argument("candidate")
    cp.add_argument("baseline")
    a = ap.parse_args(list(argv))

    if a.action == "list":
        for name, path in sorted(builtin_protocols().items()):
            p = load_protocol(str(path))
            print(f"{name:<20} v{p.version}  {p.digest()}  {p.description.splitlines()[0] if p.description else ''}")
        return 0
    if a.action == "show":
        p = load_protocol(a.protocol)
        print(yaml.safe_dump({"digest": p.digest(), **p.model_dump(mode="json")}, sort_keys=False), end="")
        return 0
    if a.action == "run":
        p = load_protocol(a.protocol)
        overrides = {k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in a.set or [])}
        if a.workers:
            overrides["workers"] = a.workers
        done = [0]

        def progress(s: dict) -> None:
            done[0] += 1
            print(f"[{done[0]}] {s['task_id']} seed={s['seed']} {s['status']} ({s['steps']} steps)", file=sys.stderr, flush=True)

        card, path = run_protocol(p, policy_spec(a.policy), a.name, out=a.out, overrides=overrides, notes=a.notes,
                                  progress=progress)
        print(json.dumps({"scorecard": str(path), "success_mean": card.metrics.get("success_mean"),
                          "success_ci95": card.success_ci95, "n": card.metrics.get("n_scored")}, indent=2))
        return 0
    if a.action == "leaderboard":
        cards = [c for d in a.dirs for c in leaderboard.collect(d)]
        if a.json:
            print(json.dumps([{"protocol": k[0], "version": k[1], "digest": k[2],
                               "rows": [{"policy": c.policy, **c.metrics, "success_ci95": c.success_ci95} for c in v]}
                              for k, v in leaderboard.boards(cards).items()], indent=2))
        else:
            print(leaderboard.markdown(cards), end="")
        return 0
    cand, base = Scorecard.load(a.candidate), Scorecard.load(a.baseline)
    res = compare(cand, base)
    print(json.dumps({"candidate": cand.policy, "baseline": base.policy, **res.model_dump()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
