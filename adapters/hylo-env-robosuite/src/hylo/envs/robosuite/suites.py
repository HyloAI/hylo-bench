"""robosuite tasks as a Hylo Benchmark (pure Python: the main process enumerates tasks without importing robosuite).

A seed fixes an episode completely: the adapter reseeds robosuite's random generator with it before the reset, so
object placements (and the hammer's size in the handover, the dirt path in Wipe, the arm's start noise) follow from
the seed alone. Splits use disjoint seed ranges like the other Hylo benchmarks: protocol "rsi" (default) gives the
RFC-0002 search / search_val / gate / heldout splits over `rsi_states` seeds plus a separate `validation` split for
plain evaluation; protocol "standard" gives train / validation / heldout.
"""

from __future__ import annotations

from hylo.envs.robosuite_contract.splits import seeded_splits

from .tasks import TASKS, task


def robosuite_tasks(tasks: list[str] | tuple[str, ...] | None = None):
    from hylo.core.task import Predicate, TaskSpec

    keys = list(tasks or TASKS)
    out = []
    for k in keys:
        t = task(k)
        out.append(TaskSpec(id=f"robosuite/{t.key}", version="1", instruction=t.instruction,
                            success=Predicate(kind="builtin", name=f"robosuite.{t.env}._check_success"),
                            max_steps=t.max_steps, params={"task": t.key, "env": t.env, "arms": t.arms},
                            tags=("robosuite", "bimanual" if t.arms == 2 else "single-arm", *t.tags)))
    return tuple(out)


def robosuite_benchmark(tasks: list[str] | None = None, protocol: str = "rsi", rsi_states: int = 50,
                        validation: int = 20, train: int = 10, heldout: int = 20, gate_looks: int = 1):
    """`tasks`: keys of tasks.TASKS (default: all). An experiment's env names one task (env.params.task), so run one
    task per experiment (its RobotSpec, e.g. where the second arm stands, depends on the task)."""
    from hylo.core.benchmark import Benchmark

    return Benchmark(id="robosuite/standard-v0", version="0.1", tasks=robosuite_tasks(tasks),
                     splits=seeded_splits(protocol, train, validation, heldout, rsi_states, gate_looks),
                     metadata={"env": "robosuite", "protocol": protocol, "tasks": ",".join(tasks or TASKS),
                               "source": "robosuite 1.5.2 (MIT), https://github.com/ARISE-Initiative/robosuite"})


__all__ = ["robosuite_benchmark", "robosuite_tasks"]
