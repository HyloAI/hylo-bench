"""Shared helpers for the showcase scripts: output directory, demonstrations and reference-policy training, tables.

Not an example on its own. The scripts import it from their own directory (Python puts a script's directory on
sys.path), so run them as files: `uv run python examples/showcase/NN_slug.py`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO = Path(__file__).resolve().parents[2]          # the bench repository
EXPERT = {"type": "baseline.planar_ik", "params": {"kp": 0.3}}


def parser(doc: str, slug: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=doc.strip().splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="a smaller version (< 60 s)")
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "examples" / slug, help="output directory")
    return ap


def demonstrations(out: Path, *, workers: int = 16) -> Path:
    """Expert (IK controller) runs on the campaign development splits search / search_val (seeds 30000+), never on
    the mujoco-reach protocol's held-out seeds (20000-20049). Returns the root the learners read."""
    from hylo.core.config import ExperimentConfig
    from hylo.eval.experiment import run_experiment

    root = out / "demos"
    if not any(root.glob("*/evallog.json")):
        data = ExperimentConfig(name="showcase-demos", benchmark={"type": "mujoco/reach-v0", "params": {"protocol": "rsi"}},
                                env={"type": "mujoco.planar_reach"}, agent=EXPERT, workers=workers)
        for split in ("search", "search_val"):
            run_experiment(data, split=split, run_root=root, label=f"expert-{split}")
    return root


def train(out: Path, methods: Iterable[str]) -> dict[str, dict[str, Any]]:
    """One reference learner per method (policy.knn / bc / flow / tokens) on the demonstrations: method -> policy spec.
    Training is deterministic; an identical job reuses its sealed artifact."""
    from hylo.core.learning import TrainJob
    from hylo.learn.jobs import run_job

    demos = demonstrations(out)
    specs = {}
    for m in methods:
        man = run_job(TrainJob(learner=f"policy.{m}", inputs=[str(demos)], out=str(out / "artifacts" / m)))
        specs[m] = {"type": f"policy.{m}", "params": {"path": str(out / "artifacts" / m)}, "_digest": man.digest}
    return specs


def mpc_policy(out: Path, *, workers: int = 16) -> dict[str, Any]:
    """policy.mpc: CEM planning in a world model learned from the expert demonstrations plus random-action runs
    (development splits only), as in examples/policies/compare.py."""
    from hylo.core.config import ExperimentConfig
    from hylo.core.learning import TrainJob
    from hylo.eval.experiment import run_experiment
    from hylo.learn.jobs import run_job

    demos, explore = demonstrations(out), out / "explore"
    if not any(explore.glob("*/evallog.json")):
        rand = ExperimentConfig(name="showcase-explore", benchmark={"type": "mujoco/reach-v0", "params": {"protocol": "rsi"}},
                                env={"type": "mujoco.planar_reach"}, agent={"type": "baseline.random", "params": {"dim": 2}},
                                workers=workers)
        for split in ("search", "search_val"):
            run_experiment(rand, split=split, run_root=explore, label=f"random-{split}")
    wm = out / "artifacts" / "world_model"
    run_job(TrainJob(learner="world_model.state", inputs=[str(demos), str(explore)], out=str(wm),
                     params={"env": {"type": "mujoco.planar_reach"}}))
    return {"type": "policy.mpc", "params": {"world_model": str(wm), "objective": {"reach": ["tip_pos", "target_pos"]}}}


def policy(spec: dict[str, Any]) -> dict[str, Any]:
    """A policy spec without the bookkeeping keys train() adds."""
    return {k: v for k, v in spec.items() if not k.startswith("_")}


class Timer:
    def __enter__(self) -> "Timer":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.s = time.perf_counter() - self.t0


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """A Markdown table (also readable in a terminal)."""
    rows = [[_fmt(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    line = lambda cells: "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"   # noqa: E731
    return "\n".join([line(headers), "|" + "|".join("-" * (w + 2) for w in widths) + "|", *map(line, rows)])


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        if x != 0 and abs(x) < 1e-3:
            return f"{x:.2e}"
        return f"{x:.3f}".rstrip("0").rstrip(".")
    return str(x)


def pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def ci(c: tuple[float, float] | None) -> str:
    return "–" if not c else f"{pct(c[0])}–{pct(c[1])}"


def write_result(out: Path, result: dict[str, Any]) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / "result.json"
    p.write_text(json.dumps(result, indent=2, default=str))
    return p


__all__ = ["EXPERT", "REPO", "Timer", "ci", "demonstrations", "mpc_policy", "parser", "pct", "policy", "table", "train", "write_result"]
