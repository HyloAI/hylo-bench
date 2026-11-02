"""MuJoCo environments for Hylo: planar reach, table reach, the Franka Panda manipulation table, Panda scenes loaded
from hylo.scene/v0.1 files, two ViperX arms at a table (ALOHA), the Unitree G1 humanoid at a table and a floating
dexterous hand (G1 or Allegro) above objects of several shapes."""

# contract versions and what is known about these plugins (hylo.core.plugins)
META = {
    "contracts": {"envs": "hylo.env/v0.1", "benchmarks": "hylo.benchmark/v0.1", "agents": "hylo.agent/v0.1",
                  "surfaces": "hylo.tools/v0.1"},
    "maturity": "implemented",
    "plugins": {
        "surfaces/mujoco.world": {
            "notes": "a stage (hylo.stage/v0.1) as a running MuJoCo world with typed tools: stage edits with revisions, "
                     "22 robot packs from MuJoCo Menagerie (fetched at a pinned commit, verified by digest), joint / "
                     "gripper / tool-point / trot / flight / mobile-base actions with receipts, cameras, checks with "
                     "reason codes, people, datasets (RFC-0008)"},
        "envs/mujoco.panda_table": {
            "notes": "Franka Panda from MuJoCo Menagerie (Apache-2.0), fetched at a pinned commit into the asset "
                     "cache ($HYLO_ASSETS) on first use and verified by digest; cameras need MUJOCO_GL=egl or osmesa"},
        "envs/mujoco.panda_scene": {
            "notes": "the panda_table robot, cameras and observation layout with the objects and success predicate "
                     "of a hylo.scene/v0.1 file (params.scene); privileged_obs exposes object poses for checkers and "
                     "privileged agents"},
        "benchmarks/mujoco/scenes-v0": {
            "notes": "a directory of scene files as tasks; seeds vary the sampled start poses (RFC-0002 splits)"},
        "envs/mujoco.aloha_table": {
            "notes": "two ViperX 300 arms from MuJoCo Menagerie ALOHA (BSD-3-Clause), fetched at a pinned commit and "
                     "verified by digest; RobotSpec with arms left and right, wrist cameras on both grippers"},
        "benchmarks/mujoco/bimanual-v0": {
            "notes": "handover, lift_box, two_bins: tasks one arm cannot do (reach split at the middle of the table, "
                     "a box too wide for one gripper, a step limit shorter than two motions in turn); RFC-0002 splits"},
        "agents/oracle.aloha_table": {
            "notes": "privileged upper bound for mujoco/bimanual-v0 (env param privileged_obs: true); scripted IK "
                     "for both arms, ordinary actions only"},
        "envs/mujoco.g1_table": {
            "notes": "Unitree G1 with three-finger hands from MuJoCo Menagerie (BSD-3-Clause), pelvis fixed, legs and "
                     "waist rigid; each arm with its hand is one ArmSpec with a dexterous HandSpec (claw presets)"},
        "benchmarks/mujoco/humanoid-v0": {
            "notes": "pick_place (one hand), two_hands (both hands at once within a step limit), press_button; "
                     "RFC-0002 splits"},
        "agents/oracle.g1_table": {
            "notes": "privileged upper bound for mujoco/humanoid-v0 (env param privileged_obs: true); scripted IK "
                     "and finger presets, ordinary actions only"},
        "envs/mujoco.dex_table": {
            "notes": "a Menagerie dexterous hand (Unitree G1 three-finger, BSD-3-Clause; Wonik Allegro, BSD-2-Clause; "
                     "left or right) on a six-joint floating base above a table with a box, cylinder, sphere, mug and "
                     "rock; one ArmSpec 'hand' with a dexterous HandSpec (params.hand)"},
        "benchmarks/mujoco/dexterous-v0": {
            "notes": "lift_<shape>: hold the object 8 cm above its start with a multi-finger hand"},
        "agents/oracle.panda_table": {
            "notes": "privileged upper bound, not a method: reads simulator object poses and the goal "
                     "(env param privileged_obs: true); scripted IK and straight-line motions, ordinary actions only"},
    },
}
