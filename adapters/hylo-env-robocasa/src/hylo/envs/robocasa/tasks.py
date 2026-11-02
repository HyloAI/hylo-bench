"""The RoboCasa atomic tasks this adapter offers and how their scenes are labelled (pure Python).

Seven atomic tasks over the four skills RoboCasa's atomic set covers on a counter: pick and place, opening or closing
an articulated fixture (a drawer, a hinged cabinet door), turning a knob or lever (a stove burner, a sink faucet) and
pressing a button (a microwave). Step budgets are RoboCasa's own task horizons (v1.0.1).

Labels: objects by their RoboCasa category ("bottled_water" -> "bottled water"); fixtures by kind ("sink", "stove",
"drawer", "cabinet", ...); a fixture the task refers to keeps the plain kind and other fixtures of the same kind get
"other <kind>" (the instruction's "the left drawer" is the one labelled "drawer"); parts of referenced fixtures get
their own labels (`PARTS`); counters are the support (id 1, "counter").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_CAMERAS = ("agentview_left", "agentview_right", "wrist")


@dataclass(frozen=True)
class Task:
    key: str                     # the TaskSpec's params["task"]
    env: str                     # RoboCasa environment class
    instruction: str             # template; the realised instruction comes from the scene (ep_meta["lang"])
    max_steps: int               # RoboCasa's task horizon
    skill: str
    notes: str = ""


TASKS: dict[str, Task] = {t.key: t for t in [
    Task("pick_place_counter_to_sink", "PickPlaceCounterToSink",
         "pick the <object> from the counter and place it in the sink", 600, "pick_place",
         "The object stands on the counter next to the sink; success: it rests inside the sink basin and the hand "
         "is clear of it."),
    Task("open_drawer", "OpenDrawer", "open the <left|right> drawer", 750, "open_close",
         "The drawer slides out along its front's normal when pulled by its handle; success: open far enough."),
    Task("close_drawer", "CloseDrawer", "close the <left|right> drawer", 450, "open_close",
         "The drawer starts pulled out; push its front back in."),
    Task("close_cabinet", "CloseCabinet", "close the cabinet doors", 750, "open_close",
         "The hinged cabinet doors (one or two) start open; swing each shut. Success: every door closed."),
    Task("turn_on_stove", "TurnOnStove", "turn on the <position> burner of the stove", 450, "knob",
         "Each burner has a knob on the stove's front or top; a knob turns about its own axis. Success: the named "
         "burner's knob turned on."),
    Task("turn_on_sink_faucet", "TurnOnSinkFaucet", "turn on the sink faucet", 600, "lever",
         "The faucet's handle is a lever on the spout; lift or swing it. Success: the water on."),
    Task("turn_on_microwave", "TurnOnMicrowave", "press the start button on the microwave", 450, "button",
         "The start and stop buttons are on the microwave's front panel; press the start button in."),
]}

# fixture class -> label ("" : not labelled, i.e. background)
FIXTURE_LABELS = {
    "Counter": "counter", "Sink": "sink", "Stove": "stove", "Stovetop": "stove", "Microwave": "microwave",
    "Drawer": "drawer", "HingeCabinet": "cabinet", "SingleCabinet": "cabinet", "OpenCabinet": "shelf",
    "PanelCabinet": "cabinet", "HousingCabinet": "cabinet", "Fridge": "fridge", "FridgeFrenchDoor": "fridge",
    "FridgeSideBySide": "fridge", "FridgeBottomFreezer": "fridge", "Dishwasher": "dishwasher", "Oven": "oven",
    "Toaster": "toaster", "ToasterOven": "toaster oven", "CoffeeMachine": "coffee machine", "StandMixer": "stand mixer",
    "ElectricKettle": "electric kettle", "Blender": "blender", "Hood": "range hood", "DishRack": "dish rack",
    "Accessory": "", "WallAccessory": "", "Stool": "stool", "Wall": "", "Floor": "", "Box": "", "Window": "",
}

# parts of a referenced fixture: (regex on the geom or body name after the fixture's prefix, label; {k}: the kind)
PARTS = (
    (r"knob_((?:front|rear|center)_(?:left|right|center))", "{pos} knob"),
    (r"start_button", "start button"),
    (r"stop_button", "stop button"),
    (r"^handle", "faucet handle"),
    (r"^spout", "faucet spout"),
    (r"(left|right)_door_handle|door_handle|_handle", "{k} handle"),
)


def fixture_label(class_name: str) -> str:
    if class_name in FIXTURE_LABELS:
        return FIXTURE_LABELS[class_name]
    return re.sub(r"(?<!^)(?=[A-Z])", " ", class_name).lower()


def object_label(category: str) -> str:
    return str(category).replace("_", " ").strip()


def part_rules(prefix: str, kind: str, part_names: list[str]) -> list[tuple[str, str]]:
    """(regex, label) segmentation rules for the parts of one referenced fixture, from the body and geom names that
    start with its prefix."""
    out: dict[str, str] = {}
    for name in part_names:
        rest = name[len(prefix):]
        for rx, template in PARTS:
            m = re.search(rx, rest)
            if m:
                pos = m.group(1).replace("_", " ") if template.startswith("{pos}") else ""
                label = template.format(pos=pos, k=kind)
                out.setdefault("^" + re.escape(name) + "$", label)
                break
    return list(out.items())


def task(key: str) -> Task:
    if key not in TASKS:
        raise ValueError(f"unknown RoboCasa task {key!r}; known: {sorted(TASKS)}")
    return TASKS[key]


__all__ = ["DEFAULT_CAMERAS", "FIXTURE_LABELS", "PARTS", "TASKS", "Task", "fixture_label", "object_label",
           "part_rules", "task"]
