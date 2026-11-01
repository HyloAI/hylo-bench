"""LIBERO (Liu et al. 2023) for Hylo: environment adapter and benchmark suites."""

# contract versions and what is known about these plugins (hylo.core.plugins)
META = {
    "contracts": {"envs": "hylo.env/v0.1", "benchmarks": "hylo.benchmark/v0.1"},
    "maturity": "implemented",
    "plugins": {
        "envs/libero": {"notes": "LIBERO (MIT); assets from lerobot/libero-assets"},
        "benchmarks/libero": {"notes": "LIBERO (MIT)"},
    },
}
