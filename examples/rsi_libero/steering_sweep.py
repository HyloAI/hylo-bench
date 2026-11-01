"""Step 1 of RFC-0002: does steering a frozen SmolVLA help on LIBERO-10?

Runs a seeded baseline and steering variants on the same init states (search split, 0..n-1) with common random
numbers, then compares every variant to the baseline with the paired one-sided exact McNemar test.

    uv run python examples/rsi_libero/steering_sweep.py [--n 10] [--workers 12]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from hylo.eval.experiment import run_experiment
from hylo.eval.scoring import paired_compare
from hylo.core.config import ExperimentConfig

ROOT = Path(__file__).resolve().parents[2]

VARIANTS: dict[str, dict] = {
    "A_baseline_seeded": {"n_samples": 1},
    "B_vote4": {"n_samples": 4, "selector": "gripper_vote_medoid"},
    "C_rules": {"n_samples": 1, "rules": [
        {"name": "stall_backoff", "feature": "still_steps", "value": 25, "action": "backoff",
         "params": {"steps": 8, "dz": 0.5, "open_gripper": 1}, "cooldown": 40, "max_fires": 3},
        {"name": "regrasp_resample", "feature": "grasp_attempts", "value": 3, "action": "resample",
         "params": {"noise_scale": 1.5}, "cooldown": 60, "max_fires": 2}]},
    "D_exec25": {"n_samples": 1, "execution_steps": 25},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="init states per task (search split 0..n-1)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    base = ExperimentConfig.load(ROOT / "examples/configs/libero10_steered.yaml")
    out_dir = ROOT / "runs" / time.strftime("steering-sweep-%Y%m%d-%H%M%S")
    results, summary = {}, {}
    for name, steering in VARIANTS.items():
        if args.only and name not in args.only:
            continue
        cfg = base.with_overrides({"benchmark.params.n_validation": args.n, "workers": args.workers,
                                   "agent.params.params.steering": steering, "agent.params.params.num_threads": 6,
                                   "name": f"steer-{name}"})
        t = time.time()
        r = run_experiment(cfg, run_root=out_dir, label=name)
        results[name] = r
        per_task: dict[str, list] = {}
        for s in r.log["samples"]:
            per_task.setdefault(s["task_id"].split("/")[-1], []).append(s["status"] == "success")
        summary[name] = {"success": r.metrics["success_mean"], "steps": r.metrics["steps_mean"],
                         "interventions": r.metrics["safety_interventions_mean"], "wall_s": round(time.time() - t),
                         "per_task": {k: sum(v) / len(v) for k, v in per_task.items()}, "run_dir": str(r.run_dir),
                         "config_digest": cfg.digest()}
        print(f"{name}: success={r.metrics['success_mean']:.3f} steps={r.metrics['steps_mean']:.0f} "
              f"wall={time.time() - t:.0f}s", flush=True)
    if "A_baseline_seeded" in results:
        base_keys = results["A_baseline_seeded"].per_key()
        for name, r in results.items():
            if name == "A_baseline_seeded":
                continue
            c = paired_compare(r.per_key(), base_keys)
            summary[name]["vs_baseline"] = c.model_dump()
            print(f"{name} vs baseline: wins {c.candidate_wins} / losses {c.parent_wins} of {c.n_pairs} pairs, "
                  f"one-sided McNemar p={c.p_value:.4f}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("summary:", out_dir / "summary.json")


if __name__ == "__main__":
    main()
