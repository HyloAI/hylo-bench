# hylo-env-mujoco

MuJoCo environments and task families for Hylo: planar and table reach, the Franka Panda manipulation table
(`mujoco.panda_table`, benchmark `mujoco/manip-v0`), Panda scenes from hylo.scene/v0.1 files, and two tables for
robots with two arms: two ViperX 300 arms (`mujoco.aloha_table`, `mujoco/bimanual-v0`) and the Unitree G1 humanoid
with three-finger hands (`mujoco.g1_table`, `mujoco/humanoid-v0`). Everything runs on a CPU; cameras need an OpenGL
backend (`MUJOCO_GL=egl`, or `osmesa` with `PYOPENGL_PLATFORM=osmesa`).

## Assets

Robot models are the MuJoCo Menagerie folders at one pinned commit (b0655057a9df), fetched on first use into the asset
cache (`$HYLO_ASSETS`, default `~/.cache/hylo/assets`) and verified file by file against a sha256 manifest beside
`assets.py`; nothing is vendored and each folder keeps its upstream LICENSE.

| Manifest | Folder | Licence | Used by |
|---|---|---|---|
| `menagerie_panda.json` | `franka_emika_panda` | Apache-2.0 | `mujoco.panda_table`, `mujoco.panda_scene` |
| `menagerie_aloha.json` | `aloha` (arms, grippers, actuators; not the table frame) | BSD-3-Clause | `mujoco.aloha_table` |
| `menagerie_g1_hands.json` | `unitree_g1` (`g1_with_hands.xml` and its meshes) | BSD-3-Clause | `mujoco.g1_table` |

`assets(name)` returns a verified folder; manifests that pin the same upstream folder share one cache folder and
fetch only the files they miss. Offline: copy the upstream folder to `assets_dir(name)`.

## Tables with two arms

Both environments share `arm_table.py`: one compiled model with a pool of objects that each episode activates
(colour, size, mass) or parks inside the table, joint-position control per arm plus a hand command per arm, the
observation layout of `hylo.core.robot` (`q.<arm>`, `hand.<arm>`, RGB-D per camera, segmentation, camera poses), a
RobotSpec with one ArmSpec per arm, privileged object poses for oracles, and snapshots. Conformance passes for both.

**`mujoco.aloha_table`**: arms `left` and `right` at x = -0.469 and +0.469 facing each other, 6 joints and a parallel
gripper each (pads 0.058 m apart when open; `hand.<arm>` is 0 when the pads touch). Cameras `overhead`, `front`,
`left_wrist`, `right_wrist`. Each arm reaches top-down grasps only about 10 cm past the middle of the table.

| Task | Success | Why two arms |
|---|---|---|
| `handover` | the bar rests in the named zone | it starts where only one arm reaches; the zone is where only the other reaches |
| `lift_box` | the box at least the named height up, level (tilt < 0.2 rad), for one second | wider than a gripper; lifted by one of its two end handles it tips |
| `two_bins` | each block in the bin on its side | each block and its bin on one arm's side; the 260-step limit is shorter than one arm after the other |

**`mujoco.g1_table`**: the G1 stands at the table with its pelvis fixed and its leg and waist joints rigid; arms
`left` and `right`, 7 joints each plus a dexterous hand of 7 finger joints (HandSpec presets `power`, `pinch`,
`point`, `flat`; `open` is a claw ready to grasp from above). Cameras `head` (on the torso) and `front`.

| Task | Success |
|---|---|
| `pick_place` | the named block on the named plate, with one hand |
| `two_hands` | each block on the plate on its side, both hands at once; the 340-step limit is shorter than one hand after the other |
| `press_button` | the button of the named colour pushed down; another button fails at once |

Simulation choices, stated plainly: arm bodies compensate gravity (as servo integral terms do on the real robots);
the G1's index and middle finger servos are limited to 0.7 N m (the model's 1.4 N m let two fingers shove an object
off the thumb), and fingers and palm of one hand do not collide with each other (the palm's convex collision mesh
fills the hollow the fingers fold into).

## Oracles and measured results

`oracle.aloha_table` and `oracle.g1_table` read privileged object poses (env param `privileged_obs: true`) and move
the arms with scripted IK and ordinary actions: the upper bound of each task. Perception-only reference programs (one
`solve(robot)` per task, cameras and segmentation only) live in `hylo-code-agents/examples/programs` and are measured
with `run_programs.py --suite bimanual|humanoid`. Validation split, 20 episodes per task:

| Benchmark | Task | Oracle | Program |
|---|---|---|---|
| mujoco/bimanual-v0 | handover | 1.00 (472 steps) | 1.00 (365 steps) |
| mujoco/bimanual-v0 | lift_box | 1.00 (175) | 1.00 (131) |
| mujoco/bimanual-v0 | two_bins | 1.00 (197) | 1.00 (180) |
| mujoco/humanoid-v0 | pick_place | 1.00 (196) | 0.80 (179) |
| mujoco/humanoid-v0 | two_hands | 0.90 (219) | 0.50 (279) |
| mujoco/humanoid-v0 | press_button | 0.90 (111) | 1.00 (63) |

The sequential baselines (`oracle.aloha_table` and `oracle.g1_table` with `sequential: true`) take 407-489 and
408-424 steps on two_bins and two_hands, past the step limits, which is what makes those tasks need both arms at once.

```
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH=$HOME/.local/mesa/root/usr/lib/x86_64-linux-gnu \
    uv run hylo run bench/examples/configs/mujoco_aloha_handover.yaml           # oracle, validation split
uv run hylo conformance bench/examples/configs/mujoco_g1_two_hands.yaml
uv run python harness/adapters/hylo-code-agents/examples/programs/run_programs.py --suite bimanual --out runs/bim
```

## Worlds

`world.py` builds a stage (`hylo.stage/v0.1`, hylo-core) into one MuJoCo model (`compose.py`, MjSpec) and runs it:
actions per robot with receipts (`controllers.py`: joints, gripper, tool pose by IK, quadruped trot, drone flight,
mobile base, pretrained walking policies from `policies.py`), goal checks with reason codes (`checks.py`), people on
routes (`crowd.py`), cameras with depth and segmentation by entity, dataset export. `packs.py` lists the 24 robot
packs; each names an asset manifest (`menagerie_*.json`, `github_*.json`: source, commit, licence, a digest per file).
`world_tools.py` is the `mujoco.world` tool surface; `scripts/menagerie_manifest.py` writes a manifest for another
Menagerie folder. Recordings and their receipts: `examples/world/showcase.py`. Guide:
[Worlds](https://hyloai.github.io/hylo-docs/user-guide/worlds/).

