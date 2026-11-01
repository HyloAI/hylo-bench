"""First RFC-0002 improvement cycle on LIBERO-10 with a frozen SmolVLA.

search (init 0-19) -> diagnose; proposer -> candidates with hypotheses; search_val (20-29) -> select;
gate (30-39) -> one confirmatory paired test; heldout (40-49) -> report only; registry promote.

    uv run python examples/rsi_libero/run_cycle.py [--workers 24]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from worldcurrent.eval.experiment import run_experiment
from worldcurrent.core.config import ExperimentConfig
from worldcurrent.evolve.candidate import CandidateRegistry
from worldcurrent.evolve.protocol import Archive, ProtocolConfig, run_cycle
from worldcurrent.evolve.steering import SteeringRuleProposer, diagnose_traces

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=str(ROOT / "runs" / "rsi-libero10"))
    args = ap.parse_args()
    out = Path(args.out)
    cfg = ExperimentConfig.load(ROOT / "examples/configs/libero10_steered.yaml").with_overrides({
        "name": "rsi-l10", "workers": args.workers,
        "benchmark.params": {"suite": "libero_10", "protocol": "rsi", "replicates": 1}})
    registry, archive = CandidateRegistry(out / "registry"), Archive(out / "archive.jsonl")
    t0 = time.time()

    def evaluate(c: ExperimentConfig, split: str, label: str):
        t = time.time()
        r = run_experiment(c, split=split, run_root=out / "runs", label=label)
        print(f"[{time.time() - t0:6.0f}s] {label:<34} {split:<10} success={r.metrics['success_mean']:.3f} "
              f"n={r.metrics['n_episodes']} ({time.time() - t:.0f}s)", flush=True)
        return r

    rep = run_cycle(cfg, registry, archive, SteeringRuleProposer(), evaluate, diagnose_traces,
                    protocol=ProtocolConfig(alpha=0.05, stage1_pairs=30, stage1_max_deficit=4, gate_top_k=1))
    print("\ndiagnosis:", json.dumps({k: rep.diagnosis[k] for k in ("success", "buckets", "bucket_rate")}))
    for e in rep.entries:
        last = e.stages[-1] if e.stages else None
        print(f"- {e.candidate.id} [{e.verdict}, stage {e.stage_reached}] {e.hypothesis}")
        if last:
            print(f"    stage {last.stage}: success {last.success} vs parent {last.parent_success}, "
                  f"wins {last.wins} / losses {last.losses} of {last.n_pairs}, p={last.p_value}")
    print("promoted:", rep.promoted, "| held-out:", json.dumps(rep.heldout))
    (out / "cycle_report.json").write_text(rep.model_dump_json(indent=2))
    print(f"elapsed {rep.elapsed_s:.0f}s -> {out / 'cycle_report.json'}")


if __name__ == "__main__":
    main()
