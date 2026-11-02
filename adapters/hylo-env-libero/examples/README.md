# LIBERO with the robot contract: programs and code-writing agents

The LIBERO adapter has a second mode for tool-using agents (RFC-0006 part 3): `control: joint_position` +
`tool_obs: true`. The default (VLA) mode, its EnvSpec and the configs that use it are unchanged.

```yaml
env:
  type: libero
  params: {control: joint_position, tool_obs: true, segmentation: true}   # + privileged_obs: true for object poses
  remote: {runtime: libero-sim}
agent: {type: code.program, params: {source: "<a solve(robot) program>"}}  # or code.llm
```

What the robot contract maps to in LIBERO (`src/hylo/envs/libero/env.py`, `robot.py`, `sim.py`):

| Contract | LIBERO |
|---|---|
| world frame | the robot's base frame (`robot0_base`): LIBERO scenes place the base at 0.91 m (kitchen, study), 0.42 m (living room) or 0 m (floor scenes); relative to it the support surface is always within a few cm of z = 0, so one `EnvSpec.robot` fits every scene. `ArmSpec.base_in_world` = identity; `extra["libero"]["base_in_sim"]` has the base in LIBERO's frame |
| `ArmSpec.model` | robosuite's MountedPanda + PandaGripper MJCF, exported from the simulator into `$HYLO_ASSETS/libero/panda/<digest>/` (content-addressed, meshes copied; finger_joint2 flipped so both fingers open towards the top of their range; actuators dropped) |
| TCP | midway between the finger pads, 0.0934 m along the hand axis (robosuite's grip site is at 0.097), +x approach, +y closing; `max_width` 0.079 m (pad gap, open) |
| action | 7 joint targets + hand command in [0, 1]: robosuite's JOINT_POSITION controller with the delta to the target passed unscaled (kp 300, critically damped); the hand command is a target opening, approached at robosuite's gripper speed |
| `q.arm`, `hand.arm` | joint positions; measured finger travel, scaled so the servos' settled closed / open positions are 0 / 1 |
| `agentview`, `wrist` (+ `.depth`, `.seg`) | upright RGB (OpenGL rows flipped), depth in metres from the z-buffer, instance ids from MuJoCo's segmentation render |
| `extra["cameras"]` | K from fovy, T_world_cam in the OpenCV frame, every step (the wrist camera moves) |
| `extra["segments"]` | 1 table (or floor), 2 robot, then one id per BDDL object with a human label (`akita_black_bowl_1` -> "black bowl", `glazed_rim_porcelain_ramekin_1` -> "ramekin", ...) and one per articulated part ("top drawer of the wooden cabinet", "stove knob", "microwave door") |
| success | LIBERO's BDDL predicate (as in the VLA mode); the episode ends when it holds |

Geometry checks (`tests/test_libero.py::test_robot_contract_geometry_matches_the_simulator`, a table and a floor
scene): forward kinematics of the exported model matches the simulator's hand (exact up to the rounded TCP: < 1e-4 m;
3e-13 m for the ee body); the privileged position of each BDDL object, projected with K and T_world_cam, lands on that
object's segment, and the depth there is no further than the projected point; joint targets are tracked to < 0.01 rad.

## Programs (`programs/`)

`solve(robot)` programs written against the toolkit only (cameras, depth, the environment's segmentation; no
privileged state), one per task type:

- `pick_place.py`: "pick up the X [qualifier] and place it on/in the Y", "put the X on/in the Y". Parses the
  instruction against the segment labels; a qualifier ("between the plate and the ramekin", "next to the cookie
  box", "on the stove", "in the top drawer ...", "from table center") picks among same-label instances by distance
  to the named references; narrow objects with `robot.grasps` (falling back to the object's points only, then
  shallower top grasps), bowls by the rim; `robot.place_at` onto what is under the target; place retries with
  other yaws.
- `open_drawer.py`: the handle is what sticks out of the drawer front; grasp it with the hand opened just wider
  than the handle (fully open fingers reach the neighbouring handles), pull along the front's normal, check that the
  drawer moved and retry once.
- `turn_on_stove.py`: grasp the knob's fin from above and twist it 0.9 rad about the vertical.
- `drawer_put_inside.py`: open the drawer (partly), go home for a clear view, pick and place inside.
- `push_plate.py` (libero_goal "push the plate to the front of the stove"): "front" is the stove's side facing the
  agentview camera; the goal is just past the stove's front edge. The plate's centre comes from a circle fitted to its
  rim; the closed fingertips push its rim from the side away from the goal (or up to 50 degrees round, where the hand
  fits next to the cabinet), then up to two corrections from a fresh look; it lifts the plate by its rim only when no
  push contact is reachable (never needed in the 10 episodes).
- `libero_long.py` (libero_10): splits the instruction into steps ("put both the A and the B ...", "... and put the
  Y on it", "... and close it") and runs them with the skills above: several objects for one target go side by side
  (the far one first, so the arm never reaches over what it placed); "left / right plate" is the plate further left
  / right in the agentview image (centre from the rim circle, since plates near the image edge are cut off); "to
  the right of the plate" is beside it towards image-right; "the back compartment of the caddy" is the caddy's side
  away from the camera, the object turned to the slot; an open drawer is pushed shut with the fingertips, the hand
  tilted back over the drawer so its palm clears the fronts above; tall handled objects (moka pots) fall back to side
  grasps, then a top grasp around the lid.

Results (`results/programs.md`; `run_programs.py`, 10 init states per task = 0..9, CPU; run 1: 945 s, run 2: 500 s,
15 workers):

| Suite | Programs | SmolVLA (recorded, 50 episodes: init states 0..4 x 10 tasks) |
|---|---|---|
| libero_spatial | **98/100 = 0.98** (all 10 tasks) | 0.70 |
| libero_object | **96/100 = 0.96** (all 10 tasks) | 0.66 |
| libero_goal | **81/100 = 0.81** (all 10 tasks; task 5 "push the plate to the front of the stove" 4/10) | 0.78 |
| libero_10 | **66/100 = 0.66** (all 10 tasks) | 0.42 |

libero_10 per task: bowl into the bottom drawer and close it 10/10; alphabet soup and tomato sauce into the basket
10/10; white mug on the left plate, yellow-and-white mug on the right plate 10/10; white mug on the plate, pudding to
its right 10/10; book into the caddy's back compartment 8/10; turn on the stove and put the moka pot on it 7/10;
cream cheese box and butter into the basket 6/10; alphabet soup and cream cheese box into the basket 3/10; both moka
pots on the stove 2/10; yellow-and-white mug into the microwave and close it 0/10.

Failures: libero_goal "open the top drawer and put the bowl inside" 0/10 (the opened drawer hangs next to / over
the bowl: 7 episodes find no collision-free rim grasp, 3 grasp the bowl and run out of the 300-step budget while
placing); "open the middle drawer" 7/10 (the pads close short of the handle, the pull does not move the drawer, and
the retry finds no collision-free approach); cream cheese 7/10 (a flat box among other objects: two grasp attempts
miss); one milk carton tipped over when the grasp missed; libero_spatial: one bowl in the drawer missed twice, one bowl
from the cabinet placed but not counted as on the plate. libero_10: the microwave task has no skill for putting
an object inside an enclosure (the mug ends up on the microwave's top) or for swinging a door shut, 0/10; the flat
cream cheese box (1.7 cm tall) is grasped too shallowly to hold (the fingertips stop above the table): 7 misses in
"soup and cream cheese", 3 failures in "cream cheese and butter"; moka pots slip out of the grasp (6 of 8 failures of
"both moka pots", all 3 of "stove and moka pot"), or the second pot ends off the burner region; 2 books were placed
but not counted as in the back compartment. Push the plate (4/10): in 3 failures the pushes hardly move the plate (the fingertips slide over its low rim), in 3
the plate ends 6-12 cm short when the 300-step budget runs out. Steps per episode
113-183 (budgets 280 / 280 / 300); libero_10 173-468 (budget 520). The comparison is not like for like: programs use the environment's segmentation
(privileged; part of the config digest) and were written while looking at these tasks; SmolVLA uses RGB only.

## code.llm (`run_code_llm.py`)

qwen3.5:35b-a3b (Ollama, vision, local CPU, shared with other jobs), zero-shot (no demonstration, no skills), 16
turns at most:

| Task | Episodes | Success | Turns (mean) | Robot steps |
|---|---|---|---|---|
| libero_object: pick up the alphabet soup and place it in the basket | init states 0-2 | **3/3** | 2.7 (1, 3, 4) | 126-165 |
| libero_spatial: pick up the black bowl between the plate and the ramekin and place it on the plate | init states 0-2 | **0/3** | 9.7 (13, 7, 9) | 280, 124, 280 |
| libero_goal: put the wine bottle on the rack (extra task) | init state 0 | **1/1** | 5 | 131 |

Per episode in `results/code_llm.md`. The model writes the same pipeline a program would (look, find, objects,
box, grasps, plan, run, place_at) and succeeds where `robot.grasps` fits the object (soup can, wine bottle). On the
bowl (libero_spatial) it failed three ways: twice it never found a rim grasp (TCP frames taken from the box frame
turned about z, so the approach axis +x was horizontal, and `robot.grasps` asked for a bowl wider than the hand)
until the 280 steps ran out; once it moved the other black bowl, not the one between the plate and the ramekin, to
the plate and called finish(). Init states 1 and 2 were first interrupted when the model server was stopped by the
operator (0 robot steps); those attempts are not counted and the results above come from a fresh run. Turns took 2-30 minutes because the CPU model server was
shared with other jobs (queueing, not generation).

## Videos (`videos/`)

Re-simulated from the traces' actions at 512x512 (`make_video.py`; the replayed outcome is checked against the
recorded one), captioned from the trace.

| Video | Suite | Task | Workflow | Init state | Steps | Outcome |
|---|---|---|---|---|---|---|
| [program_spatial_bowl_between_plate_ramekin.mp4](videos/program_spatial_bowl_between_plate_ramekin.mp4) | libero_spatial | pick up the black bowl between the plate and the ramekin and place it on the plate | code.program `pick_place.py` | 0 | 133 | success |
| [program_object_alphabet_soup_basket.mp4](videos/program_object_alphabet_soup_basket.mp4) | libero_object | pick up the alphabet soup and place it in the basket | code.program `pick_place.py` | 0 | 131 | success |
| [program_goal_turn_on_stove.mp4](videos/program_goal_turn_on_stove.mp4) | libero_goal | turn on the stove | code.program `turn_on_stove.py` | 0 | 124 | success |
| [program_libero10_bowl_in_drawer_and_close.mp4](videos/program_libero10_bowl_in_drawer_and_close.mp4) | libero_10 | put the black bowl in the bottom drawer of the cabinet and close it | code.program `libero_long.py` | 0 | 290 | success |
| [program_goal_open_middle_drawer.mp4](videos/program_goal_open_middle_drawer.mp4) | libero_goal | open the middle drawer of the cabinet | code.program `open_drawer.py` | 0 | 88 | success |
| [llm_object_alphabet_soup_basket.mp4](videos/llm_object_alphabet_soup_basket.mp4) | libero_object | pick up the alphabet soup and place it in the basket | code.llm qwen3.5:35b-a3b, zero-shot (3 turns) | 1 | 165 | success |
| [llm_goal_wine_bottle_rack.mp4](videos/llm_goal_wine_bottle_rack.mp4) | libero_goal | put the wine bottle on the rack | code.llm qwen3.5:35b-a3b, zero-shot (5 turns) | 0 | 131 | success |

No successful code.llm episode on libero_spatial (0/3), so that suite has a program video only.

## Running

```bash
uv run python bench/adapters/hylo-env-libero/examples/programs/run_programs.py --out runs/libero-programs --n 10
uv run python bench/adapters/hylo-env-libero/examples/run_code_llm.py run --out runs/libero-code-llm \
    --tasks libero_spatial:0 libero_object:0 --episodes 3 --parallel 2
uv run python bench/adapters/hylo-env-libero/examples/run_code_llm.py summarize runs/libero-code-llm
uv run python bench/adapters/hylo-env-libero/examples/make_video.py RUN_DIR EPISODE_ID OUT.mp4
```

The runners point the simulator process at this checkout's hylo-core and hylo-env-libero (PYTHONPATH), so a
`libero-sim` runtime whose venv was built from another checkout still runs this code.
