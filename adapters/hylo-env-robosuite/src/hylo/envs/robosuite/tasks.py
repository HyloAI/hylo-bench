"""The robosuite tasks this adapter offers (pure Python): which robosuite environment each builds, how many arms and
which hand, the instruction, the step budget, the cameras the robot contract shows, the segmentation labels and the
layout the RobotSpec publishes (checked against the simulator on the first reset)."""

from __future__ import annotations

from dataclasses import dataclass, field

from hylo.envs.robosuite_contract.labels import LabelRules

SINGLE_CAMERAS = ("agentview", "sideview", "wrist")
TWO_CAMERAS = ("frontview", "agentview", "birdview", "robot0_wrist", "robot1_wrist")


@dataclass(frozen=True)
class Task:
    key: str                          # the TaskSpec's params["task"]
    env: str                          # robosuite environment class
    instruction: str
    max_steps: int
    arms: int = 1
    gripper: str | None = "PandaGripper"
    cameras: tuple[str, ...] = SINGLE_CAMERAS
    labels: LabelRules = field(default_factory=LabelRules)
    tags: tuple[str, ...] = ()
    kwargs: dict = field(default_factory=dict)
    terminate_on_failure: bool = False        # robosuite's own failure check ends the episode (Wipe)
    notes: str = ""
    # layout in robot0's base frame, as robosuite builds the task: the table top's height and, for two arms, how far
    # robot1's base stands from robot0's (facing it)
    table_height: float | None = -0.112
    base_distance: float = 0.0


PICK_PLACE_OBJECTS = {"milk": ("Milk", "milk carton"), "bread": ("Bread", "bread"), "cereal": ("Cereal", "cereal box"),
                      "can": ("Can", "can")}
PICK_PLACE_LABELS = LabelRules(
    objects={**{name: label for name, label in PICK_PLACE_OBJECTS.values()},
             **{f"Visual{name}": f"{label} marker" for name, label in PICK_PLACE_OBJECTS.values()}},
    bodies={"bin1": "source bin", "bin2": "target bin"})
NUT_LABELS = LabelRules(objects={"SquareNut": "square nut", "RoundNut": "round nut"},
                        parts=((r"^SquareNut_g4", "square nut handle"), (r"^RoundNut_g8", "round nut handle")),
                        bodies={"peg1": "square peg", "peg2": "round peg"})
NUT_NOTES = "A nut is a ring with a flat handle sticking out of one side; the ring must go down over the peg."


def _pick_place(key: str) -> Task:
    name, label = PICK_PLACE_OBJECTS[key]
    return Task(
        key=f"pick_place_{key}", env=f"PickPlace{name}", max_steps=400,
        instruction=f"pick up the {label} from the source bin and put it in the target bin, in the compartment "
                    f"marked with a {label} marker",
        cameras=("agentview", "birdview", "robotview", "wrist"), labels=PICK_PLACE_LABELS, tags=("grasp", "place"),
        notes="The target bin has four compartments; a see-through marker shaped like each object (it writes no "
              "depth) shows its compartment. Success needs the object inside its compartment and the hand at least "
              "7 cm away from it.")


TASKS: dict[str, Task] = {t.key: t for t in [
    Task(key="lift", env="Lift", instruction="lift the red cube off the table", max_steps=200,
         labels=LabelRules(objects={"cube": "red cube"}), tags=("grasp",),
         notes="Success: the cube's bottom more than 4 cm above the table."),
    Task(key="stack", env="Stack", instruction="stack the red cube on top of the green cube", max_steps=300,
         labels=LabelRules(objects={"cubeA": "red cube", "cubeB": "green cube"}), tags=("grasp", "place", "precision"),
         notes="Success: the red cube resting on the green cube and released."),
    *[_pick_place(k) for k in PICK_PLACE_OBJECTS],
    Task(key="nut_square", env="NutAssemblySquare", instruction="fit the square nut onto the square peg",
         max_steps=400, table_height=-0.092, labels=NUT_LABELS, tags=("grasp", "place", "precision"), notes=NUT_NOTES),
    Task(key="nut_round", env="NutAssemblyRound", instruction="fit the round nut onto the round peg", max_steps=400,
         table_height=-0.092, labels=NUT_LABELS, tags=("grasp", "place", "precision"), notes=NUT_NOTES),
    Task(key="door", env="Door", instruction="open the door", max_steps=300,
         cameras=("sideview", "birdview", "frontview", "wrist"),
         labels=LabelRules(objects={"Door": "door"},
                           parts=((r"^Door_(handle|latch)", "door handle"), (r"^Door_[rl]_frame", "door frame"))),
         tags=("articulated", "door"),
         notes="The door is latched: its lever handle must be turned down before the door swings open on its hinge "
               "(privileged joints 'hinge' and 'latch_joint'). Success: the door turned more than 0.3 rad."),
    Task(key="wipe", env="Wipe", instruction="wipe the dirt off the table", max_steps=600, gripper="WipingGripper",
         table_height=-0.012, labels=LabelRules(markers={"contact": "dirt"}), tags=("contact", "wipe"),
         terminate_on_failure=True,
         notes="The hand is a sponge. A dirt spot is wiped when the sponge presses on it (it then disappears). The "
               "episode fails when another part of the arm touches the table or a joint reaches its limit; success: "
               "every spot wiped."),
    Task(key="two_arm_lift", env="TwoArmLift", instruction="lift the pot with both arms, holding it by its two handles",
         max_steps=300, arms=2, base_distance=1.12, cameras=TWO_CAMERAS,
         labels=LabelRules(objects={"pot": "pot"}, parts=((r"^pot_handle", "pot handle"),)), tags=("bimanual", "grasp"),
         notes="Success: the pot's bottom more than 10 cm above the table."),
    Task(key="two_arm_handover", env="TwoArmHandover",
         instruction="pick up the hammer with the arm next to it and hand it over to the other arm", max_steps=500,
         arms=2, base_distance=1.62, cameras=TWO_CAMERAS,
         labels=LabelRules(objects={"hammer": "hammer"},
                           parts=((r"^hammer_handle", "hammer handle"), (r"^hammer_(head|neck|face|claw)", "hammer head"))),
         tags=("bimanual", "grasp", "handover"),
         notes="Success: robot1 holds the hammer by its handle, robot0 touches it nowhere, and the hammer is more than "
               "10 cm above the table."),
    Task(key="two_arm_peg_in_hole", env="TwoArmPegInHole", instruction="insert the peg into the hole in the plate",
         max_steps=300, arms=2, base_distance=1.2, table_height=None, gripper=None, cameras=TWO_CAMERAS,
         labels=LabelRules(attached=((r"^peg_", "peg"), (r"^hole_", "plate"))), tags=("bimanual", "insertion"),
         notes="No grippers: the peg is fixed to robot0's hand and the plate to robot1's. Success: the peg's axis "
               "through the hole, aligned within about 0.26 rad, its tip within a few centimetres of the plate."),
]}


def task(key: str) -> Task:
    if key not in TASKS:
        raise ValueError(f"unknown robosuite task {key!r}; known: {sorted(TASKS)}")
    return TASKS[key]


def base_in_world(t: Task, i: int) -> list[list[float]]:
    """Arm i's base in the contract's world frame (robot0's base): robot1 stands `base_distance` ahead, facing back."""
    if i == 0:
        return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    return [[-1.0, 0.0, 0.0, t.base_distance], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


__all__ = ["PICK_PLACE_OBJECTS", "SINGLE_CAMERAS", "TASKS", "TWO_CAMERAS", "Task", "base_in_world", "task"]
