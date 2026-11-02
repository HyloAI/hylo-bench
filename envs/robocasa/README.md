# RoboCasa runtime (CPU)

A separate venv for RoboCasa 1.0.1 (commit 3443cdf) on robosuite master (commit 5ce6643, robosuite 1.5.2+, MuJoCo
3.3.1), used out of process by `hylo-env-robocasa` through the EnvServer protocol; the main workspace venv never
imports either.

```bash
scripts/install-osmesa-user.sh        # CPU offscreen rendering for MuJoCo, no root
envs/robocasa/setup.sh                # clone + venv + assets + runtime `robocasa-sim` (idempotent; resumes downloads)
uv run hylo conformance examples/configs/robocasa_open_drawer_code_llm.yaml
uv run python adapters/hylo-env-robocasa/examples/run.py smoke
```

Sizes (measured on this machine): venv 1.2 GB; RoboCasa checkout 0.1 GB; assets 5.2 GB downloaded, 11.1 GB extracted
(zips removed after extraction). Total on disk about 12.4 GB.

| Asset pack (RoboCasa's links) | zip | needed for |
|---|---|---|
| textures | 0.54 GB | every kitchen |
| generative_textures | 1.18 GB | kitchen styles that use AI-generated textures (scene building fails without them) |
| fixtures_lightwheel | 0.53 GB | appliances and cabinets |
| objects_lightwheel | 0.79 GB | objects (RoboCasa's default registries) |
| objaverse | 2.16 GB | objects (RoboCasa's default registries) |
| aigen_objs | 5.77 GB | **not fetched**: only used when a config asks for `obj_registries` with `aigen` |

What `setup.sh` works around (inside this venv only):

| Problem | Fix |
|---|---|
| RoboCasa declares `lerobot`, `tianshou` (torch) and teleoperation packages it never imports in an environment worker | excluded with `overrides.txt` |
| RoboCasa pins numba 0.61.2, whose llvmlite (LLVM 15) segfaults in a process that has loaded OSMesa's LLVM 20 | numba 0.68.1 (llvmlite on LLVM 20) |
| `opencv-python` links `libGL.so.1` | `opencv-python-headless` |
| downloaded archives are untrusted | each zip is extracted with a member-path check (no absolute or `..` paths) |

Throughput: building a kitchen takes 3-15 s per reset on this CPU. Results, the contract mapping and the programs:
`adapters/hylo-env-robocasa/examples/README.md`.
