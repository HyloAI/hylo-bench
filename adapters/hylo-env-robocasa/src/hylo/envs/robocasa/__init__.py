"""RoboCasa (Nasiriany et al. 2024) for Hylo: kitchen tasks on robosuite with the robot contract."""

# contract versions and what is known about these plugins (hylo.core.plugins)
META = {
    "contracts": {"envs": "hylo.env/v0.1", "benchmarks": "hylo.benchmark/v0.1", "robot": "hylo.robot/v0.1"},
    "maturity": "implemented",
    "plugins": {
        "envs/robocasa": {"notes": "RoboCasa 1.0.1 (MIT) on robosuite 1.5, out of process (bench/envs/robocasa); "
                                   "assets downloaded separately (research use)"},
        "benchmarks/robocasa": {"notes": "RoboCasa atomic kitchen tasks (MIT); seeded scenes, RFC-0002 splits"},
    },
}
