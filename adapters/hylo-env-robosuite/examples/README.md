# robosuite with the robot contract

`hylo-env-robosuite` runs robosuite's standard manipulation tasks for tool-using agents (`code.program`, `code.llm`)
through the robot contract `hylo.robot/v0.1`: joint-position control, RGB-D, camera intrinsics and poses in the OpenCV
convention, labelled instance segmentation, privileged state, and robosuite's Panda exported for the hylo-manip
toolkit's kinematics. The contract machinery it shares with the RoboCasa adapter (and that the LIBERO adapter can
adopt) is `bench/packages/hylo-robosuite-contract` (`hylo.envs.robosuite_contract`); this adapter holds only the task
list, labels, layout and splits.

```yaml
env:
  type: robosuite
  params: {task: two_arm_lift, segmentation: true}     # + privileged_obs: true for object poses
  remote: {runtime: robosuite-sim}
agent: {type: code.program, params: {source: "<a solve(robot) program>"}}   # or code.llm
```

Install: `bench/envs/robosuite/setup.sh` (see `bench/envs/robosuite/README.md`).

## Tasks

| key | robosuite env | arms | hand | instruction | step budget |
|---|---|---|---|---|---|
| `lift` | Lift | 1 | Panda gripper | lift the red cube off the table | 200 |
| `stack` | Stack | 1 | Panda gripper | stack the red cube on top of the green cube | 300 |
| `pick_place_milk` / `_bread` / `_cereal` / `_can` | PickPlaceMilk / Bread / Cereal / Can | 1 | Panda gripper | pick up the X from the source bin and put it in the target bin, in the compartment marked with a X marker | 400 |
| `nut_square`, `nut_round` | NutAssemblySquare / Round | 1 | Panda gripper | fit the square (round) nut onto the square (round) peg | 400 |
| `door` | Door | 1 | Panda gripper | open the door | 300 |
| `wipe` | Wipe | 1 | sponge (WipingGripper) | wipe the dirt off the table | 600 |
| `two_arm_lift` | TwoArmLift | 2 | Panda grippers | lift the pot with both arms, holding it by its two handles | 300 |
| `two_arm_handover` | TwoArmHandover | 2 | Panda grippers | pick up the hammer with the arm next to it and hand it over to the other arm | 500 |
| `two_arm_peg_in_hole` | TwoArmPegInHole | 2 | none (peg and plate fixed to the hands) | insert the peg into the hole in the plate | 300 |

Step budgets are this benchmark's choice (robosuite's own horizon is 1000); success is robosuite's `_check_success()`
and the episode ends when it holds (Wipe also ends on robosuite's own failure check: the arm touching the table or a
joint at its limit). Splits (`robosuite_benchmark`): a seed fixes an episode completely (the adapter reseeds
robosuite's generator in place before the reset, so placements, the hammer's size and the dirt trail follow from it);
protocol `rsi` (default) gives RFC-0002 search / search_val / gate / heldout over 50 seeds from 30000 plus a 20-seed
validation split from 10000; protocol `standard` gives train / validation / heldout.

## Contract mapping

| Contract | robosuite |
|---|---|
| world frame | robot0's base frame (body `robot0_base`); robosuite places the base differently per task, so every pose is relative to it. `extra["robosuite"]["base_in_sim"]` maps it into robosuite's world. Two-arm tasks: robot1 stands `base_distance` ahead facing back (TwoArmLift 1.12 m, TwoArmHandover 1.62 m, TwoArmPegInHole 1.2 m); the layout and the table height are published from `tasks.py` and checked against the simulator on the first reset |
| arms | `arm` (single-arm tasks), `robot0` / `robot1` (two-arm tasks) |
| `ArmSpec.model` | robosuite 1.5's Panda with the task's gripper, exported per arm into `$HYLO_ASSETS/robosuite/panda/<digest>/panda.xml` (content-addressed, meshes copied, root body at the origin, finger_joint2 mirrored so both fingers open towards the top of their range, actuators dropped); an index file lets `spec()` skip robosuite after the first export |
| TCP | Panda gripper: midway between the pads (0.093 m along the hand axis), +x approach, +y closing, max_width 0.079 m. Sponge: the centre of its face. No gripper: robosuite's grip point |
| action | per arm 7 joint targets + 1 hand command in [0, 1]; robosuite's JOINT_POSITION controller with `input_type: absolute` (the target is the goal, kp 300, critically damped, gravity compensated); the hand command moves the gripper's set-point at its own speed (0.2 per step, the full stroke in 10 steps); fingerless arms ignore it |
| `q.<arm>`, `hand.<arm>` | joint positions; finger travel scaled so the servos' settled closed / open gaps (0.0015 / 0.0785 m) are 0 / 1 |
| cameras | per task (default `agentview`, `sideview`, `wrist`; PickPlace `agentview`, `birdview`, `robotview`, `wrist`; Door `sideview`, `birdview`, `frontview`, `wrist`; two-arm `frontview`, `agentview`, `birdview`, `robot0_wrist`, `robot1_wrist`): upright RGB, depth in metres, `.seg` instance ids |
| `extra["cameras"]` | K from fovy, `T_world_cam` in the OpenCV frame, every step |
| `extra["segments"]` | 1 table, 2 robot, then the task's labels: "red cube", "green cube", "milk carton" / "bread" / "cereal box" / "can" and their "... marker", "source bin", "target bin", "square nut" + "square nut handle", "square peg", "door" + "door handle" + "door frame", "dirt", "pot" + "pot handle", "hammer" + "hammer handle" + "hammer head", "peg", "plate" |
| privileged (`privileged_obs: true`) | object root poses with labels and their hinge / slide joints (the door's `hinge` and `latch_joint`), each hand's frame |

Rendering on the CPU: planar reflections are off and the offscreen buffer is single-sample (OSMesa pays for every
multisample on each render and read-out); images lose the table's sheen and get aliased edges, nothing else changes.

## Programs (`programs/`)

`solve(robot)` programs written against the toolkit only (cameras, depth, the environment's segmentation; no
privileged state):

- `lift.py`: the toolkit's best collision-free grasp of the cube, close, lift; retries from a fresh look.
- `stack.py`: grasp the first cube named, set it down centred on the second (support height from its box), let go.
- `pick_place.py` (all four objects): the marker is see-through and writes no depth, so its position comes from its
  mask back-projected through the most downward-looking camera (the floor right behind it); marker points are left
  out of the planning scene; the object goes down onto the target bin's floor at the marker, with turned and
  nearer fallbacks for the far corner, and the hand then moves up and clear.
- `nut_assembly.py`: grasp the handle across, lift above the peg, lower the ring's centre along the peg's axis.
- `door.py`: door plane from the panel's points, grasp the lever from the front, turn it down about its pivot in
  small arcs, pull round the hinge (the panel edge away from the handle).
- `wipe.py`: order the dirt into a nearest-neighbour path, slide the flat sponge along it pressed 4 mm into the
  table, up to three passes.
- `two_arm_lift.py`: each arm takes the handle on its side (top-down on the loop's outer bar), both at once, then a
  synchronised straight lift.
- `two_arm_handover.py`: robot0 grasps the handle near the head, carries the hammer to the middle with the free end
  towards robot1, robot1 grasps the free end, robot0 lets go and backs off.
- `peg_in_hole.py`: plate centre and normal and the peg's axis and tip from their points; robot0 lines the peg up
  with the plate's normal and pushes 3 cm through; robot1 holds still.

## Results

Not run yet. Reproduce (each a single command; the runner writes `results.md`, `results.json` and the generated
configs under the output directory):

```bash
uv run python bench/adapters/hylo-env-robosuite/examples/run.py programs --n 20 --out runs/robosuite-programs
uv run python bench/adapters/hylo-env-robosuite/examples/run.py code-llm --out runs/robosuite-code-llm   # lift + two_arm_lift, 3 episodes each, one Ollama session
uv run python bench/adapters/hylo-env-robosuite/examples/run.py video runs/robosuite-programs           # best program task
uv run python bench/adapters/hylo-env-robosuite/examples/run.py video runs/robosuite-programs --task two_arm_lift
uv run python bench/adapters/hylo-env-robosuite/examples/run.py smoke                                   # 1 episode of lift
```

| task | program | episodes | success |
|---|---|---|---|
| lift | lift.py | 20 | not run yet |
| stack | stack.py | 20 | not run yet |
| pick_place_milk | pick_place.py | 20 | not run yet |
| pick_place_bread | pick_place.py | 20 | not run yet |
| pick_place_cereal | pick_place.py | 20 | not run yet |
| pick_place_can | pick_place.py | 20 | not run yet |
| nut_square | nut_assembly.py | 20 | not run yet |
| nut_round | nut_assembly.py | 20 | not run yet |
| door | door.py | 20 | not run yet |
| wipe | wipe.py | 20 | not run yet |
| two_arm_lift | two_arm_lift.py | 20 | not run yet |
| two_arm_handover | two_arm_handover.py | 20 | not run yet |
| two_arm_peg_in_hole | peg_in_hole.py | 20 | not run yet |

| task | code.llm (qwen3.5:35b-a3b, zero-shot) | episodes | success |
|---|---|---|---|
| lift | 16 turns at most | 3 | not run yet |
| two_arm_lift | 16 turns at most | 3 | not run yet |

Videos (`run.py video`: re-simulated from the trace's actions at 512x512, captioned from the trace, the replayed
outcome compared with the recorded one, at most 4 MB): not made yet.

## Licences and sources

robosuite (Zhu et al., "robosuite: A Modular Simulation Framework and Benchmark for Robot Learning") is MIT-licensed,
https://github.com/ARISE-Initiative/robosuite; installed from PyPI (robosuite 1.5.2) with MuJoCo 3.3.1 (Apache-2.0).
The exported Panda model is derived from robosuite's assets and stays in the local asset cache (not redistributed).
This adapter is Apache-2.0.

## Limitations

- One task per environment instance (and so per experiment): the RobotSpec (arms, where robot1 stands, the hand)
  depends on the task.
- The programs are untuned: written against the toolkit and checked by the sandbox and a smoke episode of `lift`, not
  evaluated (see Results). The door, wipe, nut, handover and peg-in-hole programs in particular rely on geometric
  assumptions (lever pivot, hinge side, plate normal) that only an evaluation will confirm.
- The toolkit's collision model does not know the peg and plate fixed into the hands in TwoArmPegInHole.
- Segmentation is privileged (part of the config digest); the toolkit's colour segmenter is the alternative.
- Each step renders every configured camera (RGB-D and segmentation) on the CPU: about 0.1-0.25 s per step.
