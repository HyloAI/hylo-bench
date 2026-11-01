# External demo: language, memory and self-improvement in one harness

A MuJoCo table-top task. A planar arm must touch the pad named by a natural-language instruction. Pads are visible only for
the first 5 control steps (0.25 s); after that the agent must act from memory. Every action of every agent passes through
the same SafetyShield, and every observation, decision, memory access and safety review is written to a replayable trace.

```bash
uv sync --all-groups
uv run --group demo python examples/external_demo/run_demo.py           # about 2 min (LLM/VLM calls are cached after the first run)
uv run --group demo python examples/external_demo/run_demo.py --quick   # fewer seeds
uv run --group demo python examples/external_demo/run_demo.py --no-llm  # without the LLM act
```

Requires a local Ollama server with `qwen3.5:35b-a3b` (text + vision) (override with `--model` / `--base-url`; any OpenAI-compatible
endpoint works through `chooser.llm` with `api: openai`). Output in `runs/demo-<timestamp>/`:

- `demo_walkthrough.mp4`: about 1 minute walkthrough video (title and result cards + the 3D replays), built by `make_video.py`
- `report.html`: self-contained report (charts, paired statistics, LLM/VLM decision logs, gate verdicts, embedded videos)
- `*.mp4` / `*.gif`: individual 3D and 2D replays, `summary.json`, and every episode trace under `runs/`

`uv run --group demo python examples/external_demo/make_video.py runs/demo-<timestamp>` rebuilds the walkthrough from an existing run.

| Act | What is shown | Result (20 validation seeds per task, 2026-10-30) |
|---|---|---|
| 1 | Baseline: no language understanding, no memory | 0.00 |
| 2a | Rule-based reasoner, memory off | 0.00 |
| 2 | Rule-based reasoner + spatial memory | 0.77 (direct 1.00, **indirect 0.30**, push 1.00) |
| 3 | Same agent, chooser swapped to a local LLM by one config line (sees colour values as text) | **1.00** on all three tasks; about 2-3 s per decision |
| 3b | Chooser swapped to a VLM that sees **only the top-camera image** (set-of-mark number tags, no colour values) | **1.00** on all three tasks; about 1.9 s per decision; the marked images are stored in the trace |
| 4 | Offline self-improvement on the push task (3.5 s budget) | adversarial `safety.*` candidate blocked; kp x16 rejected for +5.2 shield interventions/episode; kp 0.03 -> 0.24 promoted (validation 0.05 -> 1.00, held-out 0.00 -> 0.95, report-only); then rolled back |
| 5 | 3D videos (front + top camera) re-simulated from recorded actions, captioned from the trace; 2D replays drawn from traces only | max fingertip deviation 0.0 m (bit-exact) |

## Rendering

The report embeds 3D MP4 replays (front + top camera) re-simulated from the recorded actions, with captions taken from
the trace, plus 2D replays drawn from the traces alone. 3D rendering needs an OpenGL backend for MuJoCo:

- **CPU, no root (installed on this machine):** `scripts/install-osmesa-user.sh` extracts Mesa OSMesa into
  `~/.local/mesa`. `run_demo.py` detects it and re-executes itself with `MUJOCO_GL=osmesa`; about 11 ms per 640x480 frame.
- **GPU (AMD, needs root once):**
  ```bash
  sudo apt-get install -y libegl1 libgl1-mesa-dri libgbm1
  sudo usermod -aG render,video $USER      # log out and back in
  MUJOCO_GL=egl uv run --group demo python examples/external_demo/run_demo.py
  ```
- `--no-3d` skips 3D rendering; the 2D replays and the determinism check still run. Without rendering the VLM act is skipped.
