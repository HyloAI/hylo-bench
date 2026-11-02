"""The robot contract (hylo.robot/v0.1) on robosuite-based simulators, shared by the robosuite and RoboCasa adapters
(and written so the LIBERO adapter, on robosuite 1.4, can adopt it).

Modules (pure: importable in the main process; worker-side: only inside a simulator venv):

- `geometry` (pure): 4x4 poses, pinhole intrinsics, OpenCV camera poses, MuJoCo depth in metres.
- `panda` (pure): the Franka Panda's limits and home posture, robosuite's naming of arm `i` and its gripper, and the
  RobotSpec builder.
- `control` (pure): the hand command as a gripper set-point, the measured hand opening, joint-target clipping.
- `labels` (pure): instance ids and human labels for a scene's geoms from name rules.
- `splits` (pure): seeded RFC-0002 / standard evaluation splits.
- `env` (pure at import): `RobosuiteContractEnv`, the Environment base class (spec, reset, observe, step) that the
  adapters subclass with their task lists, scene builders and label rules.
- `export` (worker-side, MuJoCo): the arm's MJCF exported into a content-addressed asset folder and the TCP measured
  on it.
- `render` (worker-side, MuJoCo): RGB-D and instance segmentation of a camera in the contract's conventions.
- `robosuite15` (worker-side, robosuite 1.5): absolute joint-position controllers, cheap offscreen rendering,
  reseeding, the Panda's XML as a task builds it.
- `runs` (main process): one-command program evaluation, code.llm runs, summaries and captioned videos re-simulated
  from traces, for any adapter that describes itself with a `runs.Suite`.
"""
