"""Evaluate the LIBERO `solve(robot)` programs through `hylo run` (agent code.program, env libero with the robot
contract: joint-position control, RGB-D, camera poses, segmentation; no privileged state).

One experiment per (suite, program) with that program's tasks, run as separate `hylo run` processes. Writes
results.json and results.md (success per task, steps per episode) under --out.

    uv run python bench/adapters/hylo-env-libero/examples/programs/run_programs.py --out runs/libero-programs \\
        --suites libero_spatial libero_object libero_goal --n 10 --workers 15

The simulator runs in this machine's `libero-sim` runtime (envs/libero/setup.sh). Its interpreter imports
hylo-core and hylo-env-libero from the checkouts this process uses (PYTHONPATH), so a runtime venv built from another
checkout still runs this code. Options: --image-every N keeps every Nth camera frame in the traces (for videos).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

# which program solves which tasks (task_index within the suite)
PROGRAMS = {
    "libero_spatial": {"pick_place": list(range(10))},
    "libero_object": {"pick_place": list(range(10))},
    "libero_goal": {"open_drawer": [0], "pick_place": [1, 2, 4, 6, 8, 9], "drawer_put_inside": [3],
                    "push_plate": [5], "turn_on_stove": [7]},
    "libero_10": {"libero_long": list(range(10))},
}


def src_dir(module: str) -> str:
    """The source directory a hylo namespace package was imported from (the parent of `hylo/`)."""
    spec = importlib.util.find_spec(module)
    path = Path(list(spec.submodule_search_locations)[0])
    return str(path.parents[len(module.split(".")) - 1])


def config(suite: str, program: str, tasks: list[int], n: int, out: Path, image_every: int, image_size: int) -> dict:
    pythonpath = os.pathsep.join([src_dir("hylo.core"), src_dir("hylo.envs.libero")])
    return {
        "schema_version": "hylo.experiment/v0.1",
        "name": f"libero-program-{suite.removeprefix('libero_')}-{program.replace('_', '-')}",
        "benchmark": {"type": "libero", "split": "validation",
                      "params": {"suite": suite, "task_indices": tasks, "n_validation": n, "n_heldout": 0}},
        "env": {"type": "libero",
                "params": {"control": "joint_position", "tool_obs": True, "segmentation": True,
                           "image_size": image_size},
                "remote": {"runtime": "libero-sim", "env": {"PYTHONPATH": pythonpath}}},
        "agent": {"type": "code.program", "params": {"source": (HERE / f"{program}.py").read_text(), "timeout_s": 900}},
        "trace": {"record_privileged": False, "image_every": image_every},
        "output_dir": str(out),
        "tags": ["libero-program", suite, program],
    }


def episodes(run_dir: Path) -> list[dict]:
    log = json.loads((run_dir / "evallog.json").read_text())
    out = []
    for s in log["samples"]:
        prog: dict = {}
        with open(run_dir / s["trace"]) as f:
            for line in f:
                if '"agent.code.program"' in line[:60]:
                    prog = json.loads(line)["payload"]
        out.append({"task": s["task_id"].split("/")[-1], "seed": s["seed"], "success": s["status"] == "success",
                    "steps": s["steps"], "trace": str(run_dir / s["trace"]),
                    "output_tail": "\n".join((prog.get("output") or "").strip().splitlines()[-3:]),
                    "error": (prog.get("error") or "")[-300:] or None})
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--suites", nargs="*", default=list(PROGRAMS))
    p.add_argument("--programs", nargs="*", help="only these programs (default: all for the suites)")
    p.add_argument("--n", type=int, default=10, help="init states per task (0..n-1)")
    p.add_argument("--out", default="runs/libero-programs")
    p.add_argument("--workers", type=int, default=15, help="episode workers in total")
    p.add_argument("--image-every", type=int, default=0)
    p.add_argument("--image-size", type=int, default=256)
    a = p.parse_args()
    out = Path(a.out).resolve()
    (out / "configs").mkdir(parents=True, exist_ok=True)
    jobs = [(s, prog, tasks) for s in a.suites for prog, tasks in PROGRAMS[s].items()
            if not a.programs or prog in a.programs]
    total = sum(len(t) for _, _, t in jobs)
    procs = []
    t0 = time.time()
    for suite, prog, tasks in jobs:
        w = max(1, round(a.workers * len(tasks) / total))
        cfg = out / "configs" / f"{suite}-{prog}.yaml"
        cfg.write_text(yaml.safe_dump(config(suite, prog, tasks, a.n, out, a.image_every, a.image_size), sort_keys=False))
        log = open(out / "configs" / f"{suite}-{prog}.log", "w")
        cmd = [sys.executable, "-m", "hylo.core.cli", "run", str(cfg), "--set", f"workers={w}"]
        procs.append((suite, prog, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, text=True), log))
        print(f"started {suite} / {prog}: tasks {tasks}, {w} workers", flush=True)
    results: dict[str, dict] = {}
    for suite, prog, proc, log in procs:
        stdout, _ = proc.communicate()
        log.close()
        try:
            info = json.loads(stdout[stdout.index("{"):])
        except ValueError:
            print(f"{suite} / {prog}: hylo run failed (exit {proc.returncode})", flush=True)
            continue
        for e in episodes(Path(info["run_dir"])):
            results.setdefault(suite, {}).setdefault(e["task"], {"program": prog, "episodes": []})["episodes"].append(e)
        print(f"{suite} / {prog}: success {info.get('success_mean'):.2f} ({info['run_dir']})", flush=True)
    (out / "results.json").write_text(json.dumps({"n": a.n, "wall_s": round(time.time() - t0), "suites": results},
                                                 indent=1))
    lines = [f"# LIBERO programs ({a.n} init states per task)", "",
             "| suite | task | program | success | steps (mean) | failed init states |", "|---|---|---|---|---|---|"]
    for suite, tasks in results.items():
        ok = n = 0
        for task, r in sorted(tasks.items()):
            eps = r["episodes"]
            s = sum(e["success"] for e in eps)
            ok, n = ok + s, n + len(eps)
            failed = ", ".join(str(e["seed"]) for e in eps if not e["success"]) or "-"
            steps = sum(e["steps"] for e in eps) / len(eps)
            lines.append(f"| {suite} | {task} | {r['program']} | {s}/{len(eps)} | {steps:.0f} | {failed} |")
        lines.append(f"| **{suite}** | all attempted | | **{ok}/{n} = {ok / max(n, 1):.2f}** | | |")
    lines += ["", f"Wall time {time.time() - t0:.0f} s with {a.workers} episode workers."]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
