#!/usr/bin/env bash
# Build the robosuite venv used by out-of-process EnvServer workers (hylo-env-robosuite). CPU only; rendering uses
# OSMesa from scripts/install-osmesa-user.sh (no root needed). About 0.9 GB; no asset download (robosuite ships its
# meshes in the wheel).
set -euo pipefail
cd "$(dirname "$0")/../.."                       # the bench repository
ROOT=$(pwd)
HARNESS=${HYLO_HARNESS:-../harness}                 # hylo-harness, checked out next to this repository
PY=envs/robosuite/.venv/bin/python
[ -x $PY ] || uv venv -q envs/robosuite/.venv --python 3.12
# robosuite 1.5.2 with MuJoCo 3.3.1 (newer MuJoCo releases change APIs robosuite 1.5.2 calls); opencv-python and
# pynput (teleoperation) excluded: the first links libGL, the second needs X11 headers.
uv pip install --python $PY "robosuite==1.5.2" "mujoco==3.3.1" opencv-python-headless imageio "websockets>=13" \
  --override envs/robosuite/overrides.txt
uv pip install -q --python $PY -e packages/hylo-robosuite-contract -e adapters/hylo-env-robosuite -e "$HARNESS/packages/hylo-core"
# Register the runtime experiment configs name (env.remote: {runtime: robosuite-sim}) in this machine's profile
# (core.profile; $HYLO_PROFILE or ~/.config/hylo/profile.yaml).
$PY -m hylo.core.profile set-runtime robosuite-sim --python "$ROOT/$PY" \
  --env MUJOCO_GL=osmesa --env PYOPENGL_PLATFORM=osmesa --env "LD_LIBRARY_PATH=~/.local/mesa/root/usr/lib/x86_64-linux-gnu" \
  --env OMP_NUM_THREADS=1 --env LP_NUM_THREADS=2
echo "robosuite venv ready: $PY (runtime robosuite-sim)"
