"""One command per job for RoboCasa with the robot contract (shared runner: hylo.envs.robosuite_contract.runs).

    uv run python bench/adapters/hylo-env-robocasa/examples/run.py smoke                   # 1 episode, open_drawer
    uv run python bench/adapters/hylo-env-robocasa/examples/run.py programs --n 20         # 7 tasks, 20 episodes
    uv run python bench/adapters/hylo-env-robocasa/examples/run.py code-llm                # 2 tasks x 3 episodes
    uv run python bench/adapters/hylo-env-robocasa/examples/run.py video runs/robocasa-programs

Needs the robocasa-sim runtime and RoboCasa's assets (bench/envs/robocasa/setup.sh) and, for code-llm, a reachable
Ollama with qwen3.5:35b-a3b.
"""

import sys
from pathlib import Path

from hylo.envs.robosuite_contract.runs import Suite, main

HERE = Path(__file__).resolve().parent

SUITE = Suite(
    name="robocasa", env_ref="hylo.envs.robocasa.env:RoboCasaEnv", runtime="robocasa-sim",
    modules=("hylo.envs.robosuite_contract", "hylo.envs.robocasa"), programs_dir=HERE / "programs",
    programs={"open_drawer": "drawer", "close_drawer": "drawer", "close_cabinet": "close_cabinet",
              "turn_on_stove": "stove_knob", "turn_on_sink_faucet": "faucet", "turn_on_microwave": "press_button",
              "pick_place_counter_to_sink": "pick_place"},
    video_camera="agentview_center", code_llm_tasks=("open_drawer", "turn_on_microwave"))

if __name__ == "__main__":
    sys.exit(main(SUITE))
