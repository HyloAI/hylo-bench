# RoboCasa with the robot contract

`hylo-env-robocasa` runs RoboCasa's atomic kitchen tasks (RoboCasa 1.0.1 on robosuite 1.5) for tool-using agents
(`code.program`, `code.llm`) through the robot contract `hylo.robot/v0.1`. The contract machinery is shared with the
robosuite adapter (`bench/packages/hylo-robosuite-contract`); this adapter holds the task list, how kitchens are
labelled, the robot's layout and the splits.

```yaml
env:
  type: robocasa
  params: {task: open_drawer, segmentation: true}     # + privileged_obs, scene_kwargs (passed to RoboCasa's Kitchen)
  remote: {runtime: robocasa-sim}
agent: {type: code.program, params: {source: "<a solve(robot) program>"}}   # or code.llm
```

Install: `bench/envs/robocasa/setup.sh` (see `bench/envs/robocasa/README.md`: about 5.2 GB downloaded, 12.4 GB on
disk with the venv).

## Tasks

| key | RoboCasa env | skill | instruction (realised per scene) | step budget |
|---|---|---|---|---|
| `pick_place_counter_to_sink` | PickPlaceCounterToSink | pick and place | pick the <object> from the counter and place it in the sink | 600 |
| `open_drawer` | OpenDrawer | open (drawer) | open the left / right drawer | 750 |
| `close_drawer` | CloseDrawer | close (drawer) | close the left / right drawer | 450 |
| `close_cabinet` | CloseCabinet | close (hinged doors) | close the cabinet doors | 750 |
| `turn_on_stove` | TurnOnStove | turn a knob | turn on the <position> burner of the stove | 450 |
| `turn_on_sink_faucet` | TurnOnSinkFaucet | turn a lever | turn on the sink faucet | 600 |
| `turn_on_microwave` | TurnOnMicrowave | press a button | press the start button on the microwave | 450 |

Step budgets are RoboCasa's own task horizons (v1.0.1). Success is RoboCasa's `_check_success()`; the episode ends
when it holds. A seed fixes the episode: RoboCasa's generator is reseeded in place before the reset, so the kitchen
(layout and style), objects, placements, the robot's spawn offset and the task's choices follow from the seed (the
conformance check confirms same seed, same episode). Splits as for robosuite: RFC-0002 over 50 seeds from 30000 plus
a 20-seed validation split from 10000, or `protocol: standard`.

## Contract mapping

| Contract | RoboCasa |
|---|---|
| robot | RoboCasa's PandaOmron (a Panda arm on a mobile base with a torso lift); the contract drives the arm only, the base and torso get zero commands and stay where the scene put them |
| world frame | the arm's mount (body `robot0_link0`; PandaOmron's `robot0_base` is a placeholder far from the robot); counters are usually 0.2-0.3 m below it. `RobotSpec.table_height` is not given (it varies by kitchen; the tools measure it) |
| reset | RoboCasa's reset, then the arm glides from RoboCasa's stretched start posture (joint 4 at its limit, which the toolkit's planner rejects as a self-collision) to the contract's home posture in 40 steps, then 10 holding steps, all before the agent's first observation |
| `ArmSpec.model`, TCP, action, `q.arm`, `hand.arm` | as for robosuite (the same exported robosuite 1.5 Panda; forward kinematics matches the PandaOmron's arm in the scene to < 1e-5 m) |
| cameras | `agentview_left`, `agentview_right`, `wrist` by default (also `agentview_center`, `frontview`, `robotview`): RoboCasa's robot-mounted cameras |
| `extra["segments"]` | 1 "counter" (every counter: the support), 2 robot, objects by their RoboCasa category ("bottled water", "rolling pin"), fixtures by kind ("sink", "stove", "microwave", "drawer", "cabinet", "fridge", ...). The fixture the task refers to keeps the plain kind; others of that kind are "other drawer", "other cabinet"; parts of the referenced fixture get their own labels: "drawer handle", "cabinet handle", "rear left knob" (one per burner), "faucet handle", "faucet spout", "start button", "stop button" |
| privileged | object root poses with labels, the referenced fixtures with their joints (e.g. the drawer's `slidejoint`), the hand's frame |
| `extra["robocasa"]` | task, env, `layout_id`, `style_id` of the kitchen |

## Programs (`programs/`)

Perception only (cameras, depth, the environment's segmentation). Fixtures under the counter top (drawers, lower
cabinets, front knobs) are below the support plane, where the toolkit's `robot.objects()` drops points, so these
programs back-project the masks themselves and plan without the table plane for such targets (the observed points,
the counter's edge among them, are still avoided).

- `drawer.py` (open / close): front normal from the drawer's points; open: grasp the handle from the front, fingers
  closing vertically, pull straight out in two strokes; close: closed fingertips against the front, push in until it
  is level with the neighbouring fronts.
- `close_cabinet.py`: per handle, the door's plane from the cabinet points around it and the hinge at its far edge;
  the fingertips sweep round the hinge against the door's outside until it lies in the cabinet's front plane.
- `stove_knob.py`: the knob named by the instruction ("rear left knob"); top- or front-mounted from whether it stands
  above the stove top; grasp along its axis and twist 1.2 rad (the other way if that cannot be planned).
- `faucet.py`: grasp the lever's free end from above, swing it about the vertical through its base, then lift it.
- `press_button.py`: closed fingers pointed at the button along the appliance front's normal, push 1.5 cm past its
  face, back out.
- `pick_place.py`: toolkit grasps of the named object, set it down a few centimetres above the middle of the sink
  basin's floor (the lowest sink points).

## Results

Not run yet. Reproduce:

```bash
uv run python bench/adapters/hylo-env-robocasa/examples/run.py programs --n 20 --out runs/robocasa-programs
uv run python bench/adapters/hylo-env-robocasa/examples/run.py code-llm --out runs/robocasa-code-llm   # open_drawer + turn_on_microwave, 3 episodes each
uv run python bench/adapters/hylo-env-robocasa/examples/run.py video runs/robocasa-programs           # best program task
uv run python bench/adapters/hylo-env-robocasa/examples/run.py smoke                                  # 1 episode of open_drawer
```

| task | program | episodes | success |
|---|---|---|---|
| pick_place_counter_to_sink | pick_place.py | 20 | not run yet |
| open_drawer | drawer.py | 20 | not run yet |
| close_drawer | drawer.py | 20 | not run yet |
| close_cabinet | close_cabinet.py | 20 | not run yet |
| turn_on_stove | stove_knob.py | 20 | not run yet |
| turn_on_sink_faucet | faucet.py | 20 | not run yet |
| turn_on_microwave | press_button.py | 20 | not run yet |

| task | code.llm (qwen3.5:35b-a3b, zero-shot) | episodes | success |
|---|---|---|---|
| open_drawer | 16 turns at most | 3 | not run yet |
| turn_on_microwave | 16 turns at most | 3 | not run yet |

Video: not made yet (`run.py video`, as for robosuite).

## Licences and sources

RoboCasa (Nasiriany et al., "RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots", and
RoboCasa365) is MIT-licensed, https://github.com/robocasa/robocasa, installed from commit 3443cdf (v1.0.1) on
robosuite master 5ce6643 (MIT). Its kitchen assets (textures, fixtures, objaverse- and lightwheel-derived objects) are
downloaded from RoboCasa's own links into the local checkout for research use and are not redistributed; check the
asset packs' own terms before any other use. This adapter is Apache-2.0.

## Limitations

- RoboCasa's AI-generated object set (5.8 GB) is not installed: scenes use RoboCasa's default registries (objaverse,
  lightwheel). Tasks or configs that ask for `obj_registries: [aigen]` need it.
- The mobile base is not part of the contract here: tasks that need the robot to move are out of scope.
- The ready posture changes the robot's start configuration relative to RoboCasa's own (the scene is unchanged).
- The programs are untuned: checked by the sandbox and a smoke episode of `open_drawer`, not evaluated.
- Building a kitchen takes 3-15 s per reset on the CPU; each step renders the cameras (RGB-D and segmentation).
- The toolkit's `robot.objects()` drops points under the support plane (hylo-manip change worth making: an option to
  keep them, for under-counter fixtures).
