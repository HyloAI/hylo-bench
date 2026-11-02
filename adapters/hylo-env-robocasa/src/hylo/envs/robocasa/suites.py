"""RoboCasa atomic tasks as a Hylo Benchmark (pure Python: the main process enumerates tasks without RoboCasa).

A seed fixes an episode: the adapter reseeds RoboCasa's generator with it before the reset, so the kitchen (layout
and style), the objects and their placement, the robot's spawn offset and the task's choices (which drawer, which
burner) follow from the seed. Splits as in the other Hylo benchmarks: protocol "rsi" (default) gives RFC-0002
search / search_val / gate / heldout over `rsi_states` seeds plus a `validation` split; "standard" gives
train / validation / heldout.
"""

from __future__ import annotations

from hylo.envs.robosuite_contract.splits import seeded_splits

from .tasks import TASKS, task


def robocasa_tasks(tasks: list[str] | tuple[str, ...] | None = None):
    from hylo.core.task import Predicate, TaskSpec

    out = []
    for k in list(tasks or TASKS):
        t = task(k)
        out.append(TaskSpec(id=f"robocasa/{t.key}", version="1", instruction=t.instruction,
                            success=Predicate(kind="builtin", name=f"robocasa.{t.env}._check_success"),
                            max_steps=t.max_steps, params={"task": t.key, "env": t.env},
                            tags=("robocasa", "kitchen", "atomic", t.skill)))
    return tuple(out)


def robocasa_benchmark(tasks: list[str] | None = None, protocol: str = "rsi", rsi_states: int = 50,
                       validation: int = 20, train: int = 10, heldout: int = 20, gate_looks: int = 1):
    """`tasks`: keys of tasks.TASKS (default: all seven). Run one task per experiment (env.params.task)."""
    from hylo.core.benchmark import Benchmark

    return Benchmark(id="robocasa/atomic-v0", version="0.1", tasks=robocasa_tasks(tasks),
                     splits=seeded_splits(protocol, train, validation, heldout, rsi_states, gate_looks),
                     metadata={"env": "robocasa", "protocol": protocol, "tasks": ",".join(tasks or TASKS),
                               "source": "RoboCasa 1.0.1 (MIT), https://github.com/robocasa/robocasa"})


__all__ = ["robocasa_benchmark", "robocasa_tasks"]
