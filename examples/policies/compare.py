"""Every reference policy family on MuJoCo planar reach: demonstrations -> learners -> scorecards -> leaderboard.

    uv run python examples/policies/compare.py [--out runs/policies]

1. data (RFC-0002 development splits only): the IK controller's runs (demonstrations) and random-action runs
   (exploration, for the world model);
2. one learner per family -> sealed artifacts, and a world model for MPC;
3. each policy, run by an agent ("policy", ACT-style "temporal_ensemble", "steered"), scored under the
   mujoco-reach protocol (100 held-out episodes, seeds never used for learning); scorecards go to OUT/scorecards.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from hylo.bench import leaderboard
from hylo.bench.protocol import load_protocol
from hylo.bench.run import run_protocol
from hylo.core.config import ExperimentConfig
from hylo.core.learning import TrainJob
from hylo.eval.experiment import run_experiment
from hylo.learn.jobs import run_job

EXPERT = {"type": "baseline.planar_ik", "params": {"kp": 0.3}}
DATA = ExperimentConfig(name="policies", benchmark={"type": "mujoco/reach-v0", "params": {"protocol": "rsi"}},
                        env={"type": "mujoco.planar_reach"}, agent=EXPERT, workers=16)
METHODS = ("knn", "bc", "flow", "tokens")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="runs/policies")
    out = Path(ap.parse_args().out)
    for split in ("search", "search_val"):
        demos = run_experiment(DATA, split=split, run_root=out / "demos", label=f"expert-{split}")
        run_experiment(DATA.with_overrides({"agent": {"type": "baseline.random", "params": {"dim": 2}}}), split=split,
                       run_root=out / "explore", label=f"random-{split}")
        print(f"{split}: {demos.metrics['n_episodes']} demonstrations, expert success {demos.metrics['success_mean']:.2f}")
    for m in METHODS:
        t = time.time()
        man = run_job(TrainJob(learner=f"policy.{m}", inputs=[str(out / "demos")], out=str(out / "artifacts" / m)))
        print(f"trained {m:<7} {man.metrics['samples']} samples, {time.time() - t:.1f}s, digest {man.digest}")
    run_job(TrainJob(learner="world_model.state", inputs=[str(out / "demos"), str(out / "explore")],
                     out=str(out / "artifacts" / "world_model"), params={"env": {"type": "mujoco.planar_reach"}}))
    art = lambda m: {"type": f"policy.{m}", "params": {"path": str(out / "artifacts" / m)}}   # noqa: E731
    agents = {
        "expert-ik": EXPERT,
        **{m: {"type": "policy", "params": {"policy": art(m)}} for m in METHODS},
        "bc+temporal-ensemble": {"type": "temporal_ensemble", "params": {"policy": art("bc"), "m": 0.1}},
        "flow+steering-4": {"type": "steered", "params": {"sampler": art("flow"), "steering": {"n_samples": 4, "selector": "medoid"}}},
        "mpc-cem": {"type": "policy", "params": {"policy": {"type": "policy.mpc", "params": {
            "world_model": str(out / "artifacts" / "world_model"), "objective": {"reach": ["tip_pos", "target_pos"]}}}}},
    }
    protocol = load_protocol("mujoco-reach")
    for name, agent in agents.items():
        t = time.time()
        card, _ = run_protocol(protocol, agent, name, out=out)
        print(f"scored {name:<22} success {card.metrics['success_mean']:.2f}  ({time.time() - t:.0f}s)")
    print("\n" + leaderboard.markdown(leaderboard.collect(out / "scorecards")))


if __name__ == "__main__":
    main()
