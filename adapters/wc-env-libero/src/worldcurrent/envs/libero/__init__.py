"""LIBERO (Liu et al. 2023) for worldcurrent: environment adapter and benchmark suites."""

# contract versions and what is known about these plugins (worldcurrent.core.plugins)
META = {
    "contracts": {"envs": "wc.env/v0.1", "benchmarks": "wc.benchmark/v0.1"},
    "maturity": "implemented",
    "plugins": {
        "envs/libero": {"notes": "LIBERO (MIT); assets from lerobot/libero-assets"},
        "benchmarks/libero": {"notes": "LIBERO (MIT)"},
    },
}
