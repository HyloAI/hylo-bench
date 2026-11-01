"""Run a policy under a protocol and write its scorecard; compare two scorecards pairwise."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import yaml

from worldcurrent.core.config import ExperimentConfig
from worldcurrent.core.scorecard import Scorecard, TaskScore, wilson_interval
from worldcurrent.eval.experiment import run_experiment
from worldcurrent.eval.runner import EvalResult
from worldcurrent.eval.scoring import PairedComparison, paired_compare

from .protocol import Protocol


def policy_spec(text: str) -> dict[str, Any]:
    """An agent spec from JSON ({"type", "params"}), an experiment config YAML (its agent), or a plugin name."""
    if text.lstrip().startswith("{"):
        return json.loads(text)
    p = Path(text)
    if p.suffix in (".yaml", ".yml", ".json") and p.exists():
        data = yaml.safe_load(p.read_text())
        return ExperimentConfig.model_validate(data).agent.model_dump(mode="json") if "schema_version" in data else data
    return {"type": text, "params": {}}


def scorecard(protocol: Protocol, policy: str, cfg: ExperimentConfig, result: EvalResult, *, notes: str | None = None) -> Scorecard:
    samples = result.log["samples"]
    episodes = result.per_key("success")
    by_task: dict[str, list[float]] = defaultdict(list)
    for s in samples:
        v = s["scores"]["success"]["value"]
        if v is not None:
            by_task[s["task_id"]].append(float(v))
    scored = [v for v in episodes.values() if v is not None]
    m = result.metrics
    manifest = json.loads((Path(result.run_dir) / "manifest.json").read_text()) if (Path(result.run_dir) / "manifest.json").exists() else {}
    return Scorecard(protocol=protocol.name, protocol_version=protocol.version, protocol_digest=protocol.digest(),
                     policy=policy, agent=cfg.agent.model_dump(mode="json"),
                     metrics={k: v for k, v in m.items() if isinstance(v, (int, float)) or v is None},
                     success_ci95=wilson_interval(sum(scored), len(scored)),
                     per_task={t: TaskScore(success=sum(v) / len(v), n=len(v)) for t, v in sorted(by_task.items())},
                     episodes=episodes, run_dir=str(result.run_dir), config_digest=cfg.digest(),
                     code={"git_sha": manifest.get("code", {}).get("git_sha"),
                           "packages": manifest.get("runtime", {}).get("packages", {})}, notes=notes)


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "policy"


def run_protocol(protocol: Protocol, agent: dict[str, Any], policy: str, *, out: str | Path = ".",
                 overrides: dict[str, Any] | None = None, notes: str | None = None,
                 progress: Callable[[dict], None] | None = None) -> tuple[Scorecard, Path]:
    """Run `agent` under `protocol`; runs go to OUT/runs, the scorecard to OUT/scorecards/<protocol>/<policy>.json.

    `overrides` may only change how the run executes (env.remote.*, workers, output_dir): anything else would
    change what is measured, and the scorecard would no longer belong to this protocol."""
    out = Path(out)
    cfg = protocol.experiment(agent, policy, output_dir=str(out / "runs"))
    if overrides:
        bad = [k for k in overrides if not (k.startswith("env.remote.") or k in ("workers", "output_dir"))]
        if bad:
            raise ValueError(f"overrides {bad} would change what the protocol measures; only env.remote.*, workers")
        cfg = cfg.with_overrides(overrides)
    result = run_experiment(cfg, split=protocol.benchmark.split, run_root=cfg.output_dir, label=_slug(policy),
                            progress=progress)
    card = scorecard(protocol, policy, cfg, result, notes=notes)
    return card, card.save(out / "scorecards" / protocol.name / f"{_slug(policy)}.json")


def compare(candidate: Scorecard, baseline: Scorecard) -> PairedComparison:
    """One-sided exact McNemar on the episodes both scorecards share (H1: candidate succeeds more often)."""
    if candidate.protocol_digest != baseline.protocol_digest:
        raise ValueError(f"different protocols: {candidate.protocol}@{candidate.protocol_version} "
                         f"vs {baseline.protocol}@{baseline.protocol_version}")
    return paired_compare(candidate.episodes, baseline.episodes)


__all__ = ["compare", "policy_spec", "run_protocol", "scorecard"]
