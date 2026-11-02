"""One command per job for robosuite with the robot contract (shared runner: hylo.envs.robosuite_contract.runs).

    uv run python bench/adapters/hylo-env-robosuite/examples/run.py smoke                  # 1 episode, lift
    uv run python bench/adapters/hylo-env-robosuite/examples/run.py programs --n 20        # every task, 20 episodes
    uv run python bench/adapters/hylo-env-robosuite/examples/run.py code-llm               # 2 tasks x 3 episodes
    uv run python bench/adapters/hylo-env-robosuite/examples/run.py video runs/robosuite-programs
    uv run python bench/adapters/hylo-env-robosuite/examples/run.py video runs/robosuite-programs --task two_arm_lift

Needs the robosuite-sim runtime (bench/envs/robosuite/setup.sh) and, for code-llm, a reachable Ollama with
qwen3.5:35b-a3b.
"""

import sys
from pathlib import Path

from hylo.envs.robosuite_contract.runs import Suite, main

HERE = Path(__file__).resolve().parent

SUITE = Suite(
    name="robosuite", env_ref="hylo.envs.robosuite.env:RobosuiteEnv", runtime="robosuite-sim",
    modules=("hylo.envs.robosuite_contract", "hylo.envs.robosuite"), programs_dir=HERE / "programs",
    programs={"lift": "lift", "stack": "stack", "pick_place_milk": "pick_place", "pick_place_bread": "pick_place",
              "pick_place_cereal": "pick_place", "pick_place_can": "pick_place", "nut_square": "nut_assembly",
              "nut_round": "nut_assembly", "door": "door", "wipe": "wipe", "two_arm_lift": "two_arm_lift",
              "two_arm_handover": "two_arm_handover", "two_arm_peg_in_hole": "peg_in_hole"},
    video_camera="frontview", code_llm_tasks=("lift", "two_arm_lift"),
    two_arm_tasks=("two_arm_lift", "two_arm_handover", "two_arm_peg_in_hole"))

if __name__ == "__main__":
    sys.exit(main(SUITE))
