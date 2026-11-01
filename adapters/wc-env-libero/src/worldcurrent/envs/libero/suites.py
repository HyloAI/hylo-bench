"""LIBERO benchmarks as worldcurrent Benchmarks, built from a bundled task index (suites.json) so the
main process can enumerate tasks without importing LIBERO.

Seeds index LIBERO's 50 stored init states per task (seed % 50). The official protocol evaluates all 50;
splits partition them: validation = init states [0, n_validation), heldout = the next n_heldout.
Regenerate the index inside the simulator venv with `python -m worldcurrent.envs.libero.suites`.
"""

from __future__ import annotations

import json
from pathlib import Path

INDEX = Path(__file__).with_name("suites.json")

# Max steps per suite, as used by LeRobot's LIBERO evaluation (longest demo + margin).
MAX_STEPS = {"libero_spatial": 280, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}


def libero_tasks(suite: str, task_indices: list[int] | None = None):
    from worldcurrent.core.task import InitStateRef, Predicate, TaskSpec

    rows = json.loads(INDEX.read_text())[suite]
    if task_indices is not None:
        rows = [r for r in rows if r["task_index"] in task_indices]
    return tuple(
        TaskSpec(id=f"libero/{suite}/{r['name']}", version="1", instruction=r["language"],
                 init=InitStateRef(uri=f"libero://{suite}/{r['name']}", count=r["n_init_states"]),
                 success=Predicate(kind="expr", name="bddl_goal", expr=r["goal"]),
                 max_steps=MAX_STEPS.get(suite, 400),
                 params={"suite": suite, "task_index": r["task_index"]}, tags=("libero", suite))
        for r in rows)


def libero_benchmark(suite: str = "libero_spatial", n_validation: int = 25, n_heldout: int = 25,
                     task_indices: list[int] | None = None, protocol: str = "standard", replicates: int = 1,
                     rsi_bounds: tuple[int, int, int] = (20, 30, 40), gate_looks: int = 1):
    """protocol="standard": validation / heldout over the official 50 init states.
    protocol="rsi": search / search_val / gate / heldout (RFC-0002), with `replicates` noise replicates per
    init state (seed + 50 r: same init state, different policy noise)."""
    from worldcurrent.eval.benchmark import Benchmark

    tasks = libero_tasks(suite, task_indices)
    if protocol == "rsi":
        from worldcurrent.eval.benchmark import rsi_splits

        return Benchmark(id=f"libero/{suite}", version="0.1", tasks=tasks,
                         splits=rsi_splits(50, tuple(rsi_bounds), replicates, gate_looks),
                         metadata={"protocol": f"RFC-0002 rsi splits {tuple(rsi_bounds)}, replicates={replicates}, "
                                               f"gate_looks={gate_looks}"})
    if n_validation + n_heldout > 50:
        raise ValueError("LIBERO stores 50 init states per task")
    return Benchmark(id=f"libero/{suite}", version="0.1", tasks=tasks,
                     splits={"train": (), "validation": tuple(range(n_validation)),
                             "heldout": tuple(range(n_validation, n_validation + n_heldout))},
                     metadata={"protocol": "LIBERO init states; official = 50 per task (validation + heldout)"})


def _generate() -> None:  # pragma: no cover - runs in the simulator venv
    import re

    from libero.libero import benchmark, get_libero_path

    out: dict[str, list[dict]] = {}
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"):
        b = benchmark.get_benchmark_dict()[suite]()
        rows = []
        for i in range(b.n_tasks):
            t = b.get_task(i)
            bddl = (Path(get_libero_path("bddl_files")) / t.problem_folder / t.bddl_file).read_text()
            goal = re.search(r"\(:goal\s*(.*?)\)\s*\)\s*$", bddl, re.S)
            rows.append({"task_index": i, "name": t.name, "language": t.language,
                         "n_init_states": int(len(b.get_task_init_states(i))),
                         "goal": " ".join(goal.group(1).split()) if goal else ""})
        out[suite] = rows
    INDEX.write_text(json.dumps(out, indent=1))
    print(f"wrote {INDEX} ({sum(len(v) for v in out.values())} tasks)")


if __name__ == "__main__":
    _generate()
