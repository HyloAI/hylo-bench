# Policies: one reference implementation per family, on MuJoCo planar reach

The harness's `wc-policies` ships a small, weights-free implementation of each common policy family, trained from
your own runs. All of them are `policies` plugins (core.policy.ChunkPolicy), so the same agents run any of them,
any LeRobot policy (`lerobot`) or your own plugin, the same way.

| Policy | Family | Idea (reference) |
|---|---|---|
| `policy.knn` | retrieval | nearest demonstrations, averaged chunks (VINN) |
| `policy.bc` | regression BC | MLP observation -> action chunk (BC with ACT-style chunking) |
| `policy.flow` | generative | flow matching from seeded noise (pi0 / SmolVLA action experts) |
| `policy.tokens` | tokenized | binned actions, categorical heads, sampled (RT-1 / OpenVLA discretization) |
| `policy.mpc` | model-based | CEM planning in a learned world model (MPC, TD-MPC's planner) |

| Agent | Runs a policy |
|---|---|
| `policy` | chunk by chunk, seeded noise |
| `temporal_ensemble` | every step, exponentially weighted over overlapping chunks (ACT) |
| `steered` | N seeded samples, select (medoid), recovery rules |

```bash
uv run python examples/policies/compare.py     # data -> every learner -> mujoco-reach scorecards -> leaderboard (~3 min)

# step by step
uv run wc run examples/configs/mujoco_reach.yaml --split search --output runs/policies/demos \
  --set benchmark.params.protocol=rsi                                    # demonstrations (a development split)
uv run wc learn examples/policies/jobs/flow.json                         # a sealed policy_params artifact
uv run wc run examples/configs/mujoco_policy_flow_steered.yaml           # RFC-0002 held-out evaluation
uv run wc bench run mujoco-reach --name flow-steered \
  --policy examples/configs/mujoco_policy_flow_steered.yaml              # a scorecard under the benchmark protocol
```

Result on this machine (`compare.py`; 120 demonstration episodes from the search / search_val splits; the
mujoco-reach protocol: 100 held-out episodes on seeds never used for learning; 95% Wilson intervals):

| # | policy | success % | 95% CI | mean steps |
|---|---|---|---|---|
| 1 | expert (IK controller) | 100.0 | 96.3–100.0 | 36 |
| 2 | mpc-cem (world model from expert + random-action runs) | 86.0 | 77.9–91.5 | 60 |
| 3 | flow + steering, 4 samples | 84.0 | 75.6–89.9 | 54 |
| 4 | bc + temporal ensemble | 83.0 | 74.5–89.1 | 56 |
| 5 | flow | 83.0 | 74.5–89.1 | 56 |
| 6 | bc | 71.0 | 61.5–79.0 | 68 |
| 7 | tokens | 68.0 | 58.3–76.3 | 74 |
| 8 | knn | 64.0 | 54.2–72.7 | 76 |

Neighbouring rows overlap: compare two scorecards with `wc bench compare A.json B.json` (paired McNemar) before
reading a difference into them. These are small models on 8-dimensional state, for baselines and for testing the
stack; for images and real robots use LeRobot or openpi policies.
