# LIBERO + LeRobot runtime (CPU)

A separate venv for components that cannot live in the main workspace venv: the LIBERO simulator
(`hf-libero`, robosuite 1.4) and LeRobot policies (`lerobot[smolvla]`, torch). The main process talks to
them through the EnvServer and AgentServer protocols, so none of these packages is imported there.

```bash
scripts/install-osmesa-user.sh        # CPU offscreen rendering for MuJoCo, no root
envs/libero/setup.sh                  # venv + LIBERO + LeRobot + assets + non-interactive LIBERO config
uv run hylo run examples/configs/libero_smolvla.yaml                                     # libero_spatial, 5 init states/task
uv run hylo run examples/configs/libero_smolvla.yaml --set benchmark.params.suite=libero_goal --set workers=16
```

What `setup.sh` works around (all inside this venv, nothing system-wide):

| Problem | Fix |
|---|---|
| `egl-probe` (via robomimic) needs EGL headers to build | excluded with `overrides.txt`; unused with OSMesa |
| `opencv-python` links `libGL.so.1` | replaced by `opencv-python-headless` |
| LIBERO asks for a dataset path with `input()` on first import | project-local `LIBERO_CONFIG_PATH=envs/libero/config` |
| LIBERO assets are not in the wheel and would download to `~/.cache` | `lerobot/libero-assets` downloaded into `envs/libero/assets` (local research use only; not redistributed) |

Conventions (match LeRobot's LIBERO evaluation, verified against `lerobot-eval` on this machine):
agentview -> `observation.images.camera1`, wrist -> `camera2`, images rotated 180 degrees, state = eef pos (3)
+ axis-angle (3) + gripper qpos (2), 10 no-op steps after loading the init state, `empty_cameras=1`,
max steps 280 / 280 / 300 / 520 for spatial / object / goal / 10.

Throughput on CPU (128 cores): physics 2 ms/step, OSMesa rendering about 65 ms per 256x256 camera, SmolVLA
inference about 1 s per 50-action chunk. With 10 workers a 50-episode suite takes about 2.5-4 minutes.
GPU rendering (`MUJOCO_GL=egl`) needs Mesa EGL and membership in the `render` group; see the external demo README.

## Results and validation (2026-10-30, CPU, 5 init states per task = 50 episodes per suite)

| Suite | Hylo harness | wall time (10 workers) |
|---|---|---|
| libero_spatial | 0.70 | 249 s |
| libero_object | 0.66 | 147 s |
| libero_goal | 0.78 | 208 s |
| libero_10 | 0.42 | 374 s |

Validation: LeRobot's own `lerobot-eval` on the same machine, same checkpoint, same 50 libero_spatial episodes
scored 0.64 (32/50) against our 0.70 (35/50); the difference is not significant (z = -0.64) and the per-task
pattern matches (tasks 4 and 5 are the hardest in both). SmolVLA samples actions stochastically, so individual
episodes differ between runs.

## Robot contract for tool-using agents (RFC-0006)

`env.params: {control: joint_position, tool_obs: true, segmentation: true}` gives joint-position control, RGB-D with
camera intrinsics and poses, labelled instance segmentation and `EnvSpec.robot` (the Panda exported from robosuite
into `$HYLO_ASSETS/libero/panda/`), for `code.program` / `code.llm` agents and the hylo-manip toolkit. The default
mode above is unchanged. Programs, results against SmolVLA, `code.llm` runs and videos:
`adapters/hylo-env-libero/examples/README.md`.

## Exporting traces as training data

```bash
uv run hylo run examples/configs/libero_smolvla.yaml --set trace.image_every=1      # per-step images are needed for video
HF_HOME=envs/hf-home envs/libero/.venv/bin/python -m hylo.policies.lerobot.export \
    --runs runs/<run_dir> --out data/<name> [--only-success]
```
Produces a LeRobotDataset (v3) with `observation.images.image|image2`, `observation.state` (8), `action` (7, as
executed after the SafetyShield), `next.reward|done|success`, plus `hylo_provenance.json` mapping each episode
back to its run, trace and agent version. Verified by reloading with `LeRobotDataset`.
