"""Parallel workers: one protocol run with 1, 4 and 16 workers gives identical results; when it is faster.

    uv run python examples/showcase/03_parallel_workers.py [--quick] [--out DIR]

`workers` is an execution setting: it is left out of the protocol digest and `hylo bench run --workers N` accepts
it as an override. This script checks that claim on `mujoco-reach` (100 held-out episodes): for each worker count
it compares what the scorecard records (protocol digest, metrics, per-task and per-episode success) and, stronger,
every episode's step count and a hash of its full action sequence read back from the traces.

It also measures wall-clock time (process start-up included). The evaluator's workers are threads, each with its
own environment and agent. An agent that computes in Python in-process holds the GIL, so threads do not make it
faster. An agent that runs out of process (the `remote` agent: one agent server process per worker, the way LIBERO
and LeRobot policies run) computes in parallel and the run scales.

Agents. Quick: the IK controller at a weak gain (kp 0.025, about 126 steps per episode), in-process. Full: also
MPC (CEM planning in a world model learned from development runs, about 50 world-model predictions per step), once
in-process and once as a remote agent; the remote run must give the same episodes as the in-process one.

Typical runtime: --quick about 10 s, full about 3.5 min (128-core CPU). Requires hylo-bench, hylo-env-mujoco,
hylo-policies, hylo-learn. Output: runs/examples/parallel-workers/ (scorecards/, result.json).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from _common import Timer, mpc_policy, parser, pct, table, write_result

SLUG = "parallel-workers"


def fingerprint(run_dir: str) -> dict[str, tuple[int, str]]:
    """episode key -> (steps, sha256 of every executed action), from the run's traces."""
    from hylo.core.trace import TraceStore

    log = json.loads((Path(run_dir) / "evallog.json").read_text())
    store, out = TraceStore(run_dir), {}
    for s in log["samples"]:
        h = hashlib.sha256()
        for e in store.read(s["episode_id"]).of_type("action"):
            h.update(e["payload"]["data"].tobytes())
        out[s["key"]] = (s["steps"], h.hexdigest()[:16])
    return out


def main() -> None:
    a = parser(__doc__, SLUG).parse_args()
    from hylo.bench.protocol import load_protocol
    from hylo.bench.run import run_protocol

    out = a.out
    for d in ("scorecards", "runs"):
        shutil.rmtree(out / d, ignore_errors=True)
    protocol = load_protocol("mujoco-reach")
    # (name, agent, worker counts, reference group: runs in one group must give identical episodes)
    plan = [("ik-weak-gain", "in-process", {"type": "baseline.planar_ik", "params": {"kp": 0.025}}, (1, 4, 16), "ik")]
    if not a.quick:
        mpc = mpc_policy(out)
        one_thread = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        plan += [("mpc-cem", "in-process", {"type": "policy", "params": {"policy": mpc}}, (1, 16), "mpc"),
                 ("mpc-cem", "remote", {"type": "remote", "params": {"ref": "hylo.agents.policy_agents:PolicyAgent",
                                                                      "params": {"policy": mpc}, "env": one_thread}},
                  (1, 4, 16), "mpc")]
    rows, results, refs = [], [], {}
    with Timer() as t_all:
        for name, where, agent, counts, group in plan:
            base_s = None
            for w in counts:
                with Timer() as t:
                    card, _ = run_protocol(protocol, agent, f"{name}-{where}-w{w}", out=out, overrides={"workers": w})
                fp = fingerprint(card.run_dir)
                view = {"protocol_digest": card.protocol_digest, "metrics": card.metrics, "episodes": card.episodes,
                        "per_task": {k: v.model_dump() for k, v in card.per_task.items()}}
                ref = refs.setdefault(group, (view, fp))
                base_s = base_s or t.s
                same_card, same_actions = view == ref[0], fp == ref[1]
                results.append({"agent": name, "where": where, "workers": w, "seconds": round(t.s, 2),
                                "speedup_vs_1_worker": round(base_s / t.s, 2), "success": card.metrics["success_mean"],
                                "steps_mean": card.metrics["steps_mean"], "protocol_digest": card.protocol_digest,
                                "scorecard_identical": same_card, "actions_identical": same_actions})
                rows.append((name, where, w, f"{t.s:.1f}", f"{base_s / t.s:.1f}x", pct(card.metrics["success_mean"]),
                             f"{card.metrics['steps_mean']:.2f}", card.protocol_digest,
                             "yes" if same_card else "NO", "yes" if same_actions else "NO"))
    print(table(["agent", "runs", "workers", "wall s", "speedup", "success %", "mean steps", "protocol digest",
                 "same scorecard", "same actions (100 ep.)"], rows))
    ok = all(r["scorecard_identical"] and r["actions_identical"] for r in results)
    print(f"\nevery run identical to its reference (first row of its agent): {ok}; total {t_all.s:.0f} s")
    write_result(out, {"example": SLUG, "quick": a.quick, "protocol": protocol.name,
                       "protocol_digest": protocol.digest(), "all_identical": ok, "runs": results,
                       "seconds_total": round(t_all.s, 1)})


if __name__ == "__main__":
    main()
