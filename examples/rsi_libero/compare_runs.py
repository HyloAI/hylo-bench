"""Compare steering variants against the seeded baseline from finished runs (paired, same init states and noise).

    uv run python examples/rsi_libero/compare_runs.py runs/<sweep dir or any dir containing steer-* runs> [...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from hylo.eval.scoring import one_sided_exact_mcnemar


def load(paths: list[str]) -> dict[str, dict]:
    runs = {}
    for root in paths:
        for log in Path(root).rglob("evallog.json"):
            name = log.parent.name.split("-")[1] if log.parent.name.startswith("steer-") else log.parent.name
            data = json.loads(log.read_text())
            runs[name] = {s["key"]: (s["status"] == "success", s["task_id"].split("/")[-1], s.get("steps"))
                          for s in data["samples"]}
    return runs


def main() -> None:
    runs = load(sys.argv[1:] or ["runs"])
    base_name = next((n for n in runs if n.startswith("A_")), None)
    if base_name is None:
        sys.exit("no A_baseline run found")
    base = runs[base_name]
    tasks = sorted({t for _, t, _ in base.values()})
    print(f"{'variant':<20} {'success':>7} {'wins':>5} {'losses':>6} {'p(one-sided)':>12}   per task")
    for name in sorted(runs):
        r = runs[name]
        keys = [k for k in r if k in base]
        succ = sum(r[k][0] for k in keys) / len(keys)
        per = " ".join(f"{sum(r[k][0] for k in keys if r[k][1] == t) / max(1, sum(1 for k in keys if r[k][1] == t)):.1f}"
                       for t in tasks)
        if name == base_name:
            print(f"{name:<20} {succ:7.2f} {'-':>5} {'-':>6} {'-':>12}   {per}")
            continue
        w = sum(1 for k in keys if r[k][0] and not base[k][0])
        lo = sum(1 for k in keys if base[k][0] and not r[k][0])
        print(f"{name:<20} {succ:7.2f} {w:5d} {lo:6d} {one_sided_exact_mcnemar(w, lo):12.4f}   {per}")
    print("\ntasks:", ", ".join(f"{i}={t[:40]}" for i, t in enumerate(tasks)))


if __name__ == "__main__":
    main()
