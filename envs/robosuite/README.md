# robosuite runtime (CPU)

A separate venv for the robosuite simulator (robosuite 1.5.2, MuJoCo 3.3.1), used out of process by
`hylo-env-robosuite` through the EnvServer protocol (`hylo.core.remote.server`); the main workspace venv never imports
robosuite.

```bash
scripts/install-osmesa-user.sh        # CPU offscreen rendering for MuJoCo, no root
envs/robosuite/setup.sh               # venv (about 0.9 GB) + this checkout's packages + runtime `robosuite-sim`
uv run hylo conformance examples/configs/robosuite_lift_code_llm.yaml
uv run python adapters/hylo-env-robosuite/examples/run.py smoke
```

What `setup.sh` works around (inside this venv only):

| Problem | Fix |
|---|---|
| robosuite 1.5.2 fails on recent MuJoCo releases (joint-type checks, `mj_fullM`) | MuJoCo pinned to 3.3.1 |
| `opencv-python` links `libGL.so.1`; `pynput` (teleoperation) needs X11 headers | excluded with `overrides.txt`, `opencv-python-headless` instead |
| the exported Panda models | written once to `$HYLO_ASSETS/robosuite/panda/` (about 0.2 GB with meshes) by the first `spec()` |

The runtime is registered in this machine's profile (`hylo profile show`) with `MUJOCO_GL=osmesa`, OSMesa's library
path, `OMP_NUM_THREADS=1` and `LP_NUM_THREADS=2`. Results, the contract mapping and the programs:
`adapters/hylo-env-robosuite/examples/README.md`.
