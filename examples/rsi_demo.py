"""End-to-end offline improvement loop on MuJoCo planar reach.

    uv run python examples/rsi_demo.py

1. Runs the baseline (the IK controller with a weak gain, kp=0.02) on the train split, diagnoses failures from traces.
2. A rule generator proposes gain candidates; an adversarial generator also tries to loosen safety.
3. Protected-scope check rejects the safety patch; others are validated (paired McNemar on validation).
4. Best passing candidate is reported on held-out (not used for selection) and promoted.
5. A field regression is simulated and the registry rolls back.
"""

import json
import tempfile
from pathlib import Path

from hylo.eval.experiment import run_experiment
from hylo.core.config import ExperimentConfig
from hylo.evolve.candidate import Candidate, CandidateRegistry
from hylo.evolve.loop import GainProposer, improvement_cycle


class WithAdversary(GainProposer):
    name = "rule.gain+adversary@0.1"

    def propose(self, diagnosis, cfg, parent_id, archive):
        bad = Candidate(patch={"safety.max_delta": 10.0}, parent_id=parent_id, generator=self.name,
                        rationale="faster convergence if the rate limit is removed")
        return [(bad, bad.rationale), *super().propose(diagnosis, cfg, parent_id, archive)]


runs_dir = Path(__file__).resolve().parents[1] / "runs"
runs_dir.mkdir(exist_ok=True)
root = Path(tempfile.mkdtemp(prefix="rsi-demo-", dir=runs_dir))
cfg = ExperimentConfig.load(Path(__file__).parent / "configs" / "mujoco_reach.yaml").with_overrides({"agent.params.kp": 0.02})
registry = CandidateRegistry(root / "registry")


def evaluate(c, split, label):
    r = run_experiment(c, split=split, run_root=root / "runs", label=label)
    print(f"  eval {label:<28} split={split:<10} success={r.metrics['success_mean']:.3f} "
          f"safety={r.metrics['safety_interventions_mean']:.2f}")
    return r


print(f"workspace: {root}")
rep = improvement_cycle(cfg, registry, WithAdversary(factors=(2.0, 4.0, 8.0), adaptive=True), evaluate)
print("\ndiagnosis:", rep.diagnosis.buckets)
for e in rep.evaluated:
    verdict = e.get("rejected") or ("PASS" if e["gate"]["passed"] else "FAIL: " + "; ".join(e["gate"]["reasons"]))
    print(f"  {e['id']} {e['patch']} -> {verdict}")
print("held-out (report only):", json.dumps(rep.heldout))
print("promoted:", rep.promoted, "effective patch:", registry.effective_patch())
restored = registry.rollback(reason="simulated field regression", by="operator")
print("rolled back to:", restored, "effective patch:", registry.effective_patch())
print("ledger ops:", [e["op"] for e in registry.ledger()])
