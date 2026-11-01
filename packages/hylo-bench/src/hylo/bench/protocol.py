"""Benchmark protocols (hylo.protocol/v0.1): everything about an evaluation except the policy.

A protocol fixes the benchmark and split (which tasks and init states), the environment, the safety limits, the
trace options, the timing and the seed. The policy is the only free input, so two scorecards of the same protocol
digest are comparable, episode by episode. `timing` decides whether the world waits for the policy (sync) or keeps
moving while it thinks (realtime: what a workcell runs). Where the simulator runs (env.remote) is not measured.

Protocols ship with this package (protocols/*.yaml, by name) or are given as a YAML path. Changing anything
measured is a new version; `workers` and `description` are not measured and do not change the digest.
"""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from hylo.core.config import BenchmarkRef, ComponentRef, EnvRef, ExperimentConfig, TraceOptions
from hylo.core.contract import Contract, JsonAny
from hylo.core.kernel import Timing

PROTOCOL_SCHEMA = "hylo.protocol/v0.1"


class Protocol(Contract):
    schema_version: Literal["hylo.protocol/v0.1"] = PROTOCOL_SCHEMA
    name: str
    version: str
    description: str = ""
    benchmark: BenchmarkRef                    # includes the split that is scored
    env: EnvRef
    safety: dict[str, JsonAny] = Field(default_factory=dict)
    trace: TraceOptions = Field(default_factory=TraceOptions)
    timing: Timing = Field(default_factory=Timing)  # realtime: score what a robot would run (latency, holds)
    seed: int = 0
    workers: int = Field(default=1, ge=1)      # parallel episodes: how fast, not what is measured
    requires: tuple[str, ...] = ()             # what must be installed or set up, for humans

    def digest(self) -> str:
        measured = self.model_dump(mode="json", exclude={"description", "workers", "requires"})
        if "remote" in measured["env"]:                      # where the simulator runs is not measured: hashed as a
            measured["env"]["remote"] = None                 # local environment, so local protocols keep their digests
        if measured["timing"] == Timing().model_dump(mode="json"):
            measured.pop("timing")
        return hashlib.sha256(json.dumps(measured, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]

    def experiment(self, agent: ComponentRef | dict[str, Any], policy: str, *, output_dir: str = "runs") -> ExperimentConfig:
        return ExperimentConfig(name=f"{self.name}--{policy}", benchmark=self.benchmark, env=self.env,
                                agent=agent if isinstance(agent, ComponentRef) else ComponentRef.model_validate(agent),
                                seed=self.seed, safety=self.safety, trace=self.trace, timing=self.timing, workers=self.workers,
                                output_dir=output_dir, tags=("bench", f"protocol:{self.name}@{self.version}"))


def builtin_protocols() -> dict[str, Path]:
    root = resources.files("hylo.bench") / "protocols"
    return {p.name[:-5]: Path(str(p)) for p in root.iterdir() if p.name.endswith(".yaml")}


def load_protocol(name_or_path: str) -> Protocol:
    p = Path(name_or_path)
    if not p.exists():
        known = builtin_protocols()
        if name_or_path not in known:
            raise KeyError(f"unknown protocol {name_or_path!r}; built in: {', '.join(sorted(known))}")
        p = known[name_or_path]
    return Protocol.model_validate(yaml.safe_load(p.read_text()))


__all__ = ["PROTOCOL_SCHEMA", "Protocol", "builtin_protocols", "load_protocol"]
