"""Run the `code.llm` agent (a chat model writing Python against the toolkit, turn by turn) on LIBERO tasks with the
robot contract, and summarise the traces.

One `hylo run` per task, validation split (init states 0..n-1), at most --parallel model sessions at once (the
local model server is shared). Zero-shot: no demonstration program, no skills.

    uv run python bench/adapters/hylo-env-libero/examples/run_code_llm.py run --out runs/libero-code-llm \\
        --tasks libero_spatial:0 libero_object:0 --episodes 3 [--model JSON] [--parallel 2] [--max-turns 16]
    uv run python bench/adapters/hylo-env-libero/examples/run_code_llm.py summarize runs/libero-code-llm
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

QWEN = {"backend": "ollama", "model": "qwen3.5:35b-a3b", "vision": True, "reasoning_effort": "none",
        "max_tokens": 3000, "timeout_s": 2400, "retries": 1}


def src_dir(module: str) -> str:
    spec = importlib.util.find_spec(module)
    return str(Path(list(spec.submodule_search_locations)[0]).parents[len(module.split(".")) - 1])


def config(suite: str, task: int, a: argparse.Namespace, out: Path) -> dict:
    model = json.loads(a.model) if a.model else QWEN
    return {
        "schema_version": "hylo.experiment/v0.1",
        "name": f"libero-code-llm-{a.tag}-{suite.removeprefix('libero_')}-{task}",
        "benchmark": {"type": "libero", "split": "validation",
                      "params": {"suite": suite, "task_indices": [task], "n_validation": a.episodes, "n_heldout": 0}},
        "env": {"type": "libero", "params": {"control": "joint_position", "tool_obs": True, "segmentation": True},
                "remote": {"runtime": "libero-sim",
                           "env": {"PYTHONPATH": os.pathsep.join([src_dir("hylo.core"), src_dir("hylo.envs.libero")])}}},
        "agent": {"type": "code.llm", "params": {"model": model, "max_turns": a.max_turns, "block_timeout_s": 120,
                                                 "act_wait_s": 7200}},
        "trace": {"record_privileged": False, "image_every": 0},
        "workers": a.episodes,
        "output_dir": str(out),
        "tags": ["code-llm", "libero", suite],
    }


def run(a: argparse.Namespace) -> int:
    out = Path(a.out).resolve()
    (out / "configs").mkdir(parents=True, exist_ok=True)
    jobs = []
    for spec in a.tasks:
        suite, task = spec.split(":")
        jobs.append((suite, int(task)))
    running: dict = {}
    # each run's episodes are its own workers; --parallel caps the model sessions over all runs
    per_run = max(1, a.parallel // max(1, min(len(jobs), a.parallel)))
    while jobs or running:
        while jobs and sum(n for _, _, n in running.values()) + per_run <= a.parallel:
            suite, task = jobs.pop(0)
            cfg = config(suite, task, a, out)
            cfg["workers"] = per_run
            path = out / "configs" / f"{suite}-{task}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            log = open(out / "configs" / f"{suite}-{task}.log", "w")
            cmd = [sys.executable, "-m", "hylo.core.cli", "run", str(path)]
            running[(suite, task)] = (subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, text=True), log, per_run)
            print(f"{time.strftime('%H:%M:%S')} started {suite}:{task} ({per_run} sessions)", flush=True)
        for key, (proc, log, _) in list(running.items()):
            if proc.poll() is None:
                continue
            stdout = proc.stdout.read() if proc.stdout else ""
            log.close()
            del running[key]
            try:
                info = json.loads(stdout[stdout.index("{"):])
                print(f"{time.strftime('%H:%M:%S')} {key}: success {info.get('success_mean')} ({info.get('run_dir')})",
                      flush=True)
            except ValueError:
                print(f"{key}: hylo run failed (exit {proc.returncode})", flush=True)
        time.sleep(5)
    return 0


def episode(trace: Path) -> dict:
    turns, summary, start, end = [], {}, None, None
    with open(trace) as f:
        for line in f:
            head = line[:60]
            if '"agent.code.turn"' in head or '"agent.code.summary"' in head or '"episode.' in head:
                ev = json.loads(line)
                t, p = ev["type"], ev["payload"]
                if t == "agent.code.turn":
                    turns.append(p)
                elif t == "agent.code.summary":
                    summary = p
                elif t == "episode.start":
                    start = ev["wall_time"]
                elif t == "episode.end":
                    end = ev["wall_time"]
    lat = [t["latency_s"] for t in turns if t.get("latency_s")]
    return {"turns": len(turns), "latency_mean_s": round(statistics.mean(lat), 1) if lat else None,
            "tokens_in": sum(t.get("tokens_in") or 0 for t in turns),
            "tokens_out": sum(t.get("tokens_out") or 0 for t in turns),
            "errors": sum(1 for t in turns if t.get("error")),
            "wall_s": round(end - start, 1) if start and end else None,
            "stopped": summary.get("stopped"), "finished": summary.get("finished")}


def summarize(a: argparse.Namespace) -> int:
    root = Path(a.out)
    rows = []
    for run_dir in sorted(p.parent for p in root.glob("*/evallog.json")):
        log = json.loads((run_dir / "evallog.json").read_text())
        for s in log["samples"]:
            rows.append({"task": s["task_id"].removeprefix("libero/"), "run_dir": str(run_dir), "seed": s["seed"],
                         "episode_id": s["episode_id"], "success": s["status"] == "success", "steps": s["steps"],
                         "reason": s.get("termination_reason"), **episode(run_dir / s["trace"])})
    (root / "results.json").write_text(json.dumps(rows, indent=1, default=str))
    lines = ["| task | init state | outcome | robot steps | turns | code errors | s / turn | tokens in / out | "
             "wall min | stopped |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['task']} | {r['seed']} | {'success' if r['success'] else 'failure'} | {r['steps']} | "
                     f"{r['turns']} | {r['errors']} | {r['latency_mean_s']} | {r['tokens_in']} / {r['tokens_out']} | "
                     f"{(r['wall_s'] or 0) / 60:.1f} | {str(r['stopped'])[:40]} |")
    ok = sum(r["success"] for r in rows)
    lines += ["", f"{ok}/{len(rows)} episodes succeeded."]
    (root / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", default="runs/libero-code-llm")
    r.add_argument("--tasks", nargs="*", default=["libero_spatial:0", "libero_object:0"], help="suite:task_index")
    r.add_argument("--episodes", type=int, default=3)
    r.add_argument("--model", help="chat model spec as JSON (default: qwen3.5:35b-a3b via Ollama, vision)")
    r.add_argument("--max-turns", type=int, default=16)
    r.add_argument("--parallel", type=int, default=2, help="model sessions at once")
    r.add_argument("--tag", default="qwen")
    s = sub.add_parser("summarize")
    s.add_argument("out")
    a = p.parse_args()
    return run(a) if a.cmd == "run" else summarize(a)


if __name__ == "__main__":
    sys.exit(main())
