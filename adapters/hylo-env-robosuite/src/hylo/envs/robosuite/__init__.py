"""robosuite (Zhu et al. 2020) for Hylo: environment adapter with the robot contract and benchmark tasks."""

# contract versions and what is known about these plugins (hylo.core.plugins)
META = {
    "contracts": {"envs": "hylo.env/v0.1", "benchmarks": "hylo.benchmark/v0.1", "robot": "hylo.robot/v0.1"},
    "maturity": "implemented",
    "plugins": {
        "envs/robosuite": {"notes": "robosuite 1.5 (MIT), out of process (bench/envs/robosuite)"},
        "benchmarks/robosuite": {"notes": "robosuite's standard tasks (MIT); seeded placements, RFC-0002 splits"},
    },
}
