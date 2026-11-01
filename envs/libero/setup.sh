#!/usr/bin/env bash
# Build the LIBERO + LeRobot venv used by out-of-process EnvServer / AgentServer workers.
# CPU only. Rendering uses OSMesa from scripts/install-osmesa-user.sh (no root needed).
set -euo pipefail
cd "$(dirname "$0")/../.."                       # the bench repository
ROOT=$(pwd)
HARNESS=${HYLO_HARNESS:-../harness}                 # hylo-harness, checked out next to this repository
PY=envs/libero/.venv/bin/python
uv venv -q envs/libero/.venv --python 3.12
# egl-probe (pulled in by robomimic) needs EGL headers to build and is unused with OSMesa.
uv pip install --python $PY "hf-libero>=0.1.4,<0.2" "lerobot[smolvla]>=0.6" --override envs/libero/overrides.txt
# opencv-python links libGL; the headless build does not.
uv pip uninstall -q --python $PY opencv-python || true
uv pip install -q --python $PY opencv-python-headless "websockets>=13"   # websockets: ws transport (core.remote.ws)
uv pip install -q --python $PY -e adapters/hylo-env-libero \
  -e "$HARNESS/packages/hylo-core" -e "$HARNESS/packages/hylo-eval" -e "$HARNESS/packages/hylo-agents" -e "$HARNESS/packages/hylo-memory" \
  -e "$HARNESS/packages/hylo-learn" -e "$HARNESS/adapters/hylo-policy-lerobot"
# LIBERO assets (research use only; not redistributed) and a non-interactive LIBERO config.
$PY -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='lerobot/libero-assets', repo_type='dataset', local_dir='envs/libero/assets')"
PKG=$($PY -c "import importlib.util,os; print(os.path.join(list(importlib.util.find_spec('libero').submodule_search_locations)[0], 'libero'))")
ln -sfn "$ROOT/envs/libero/assets" "$PKG/assets"
mkdir -p envs/libero/config
cat > envs/libero/config/config.yaml <<CFG
benchmark_root: $PKG
bddl_files: $PKG/bddl_files
init_states: $PKG/init_files
datasets: $PKG/../datasets
assets: $ROOT/envs/libero/assets
CFG
# Register the runtimes experiment configs name (remote: {runtime: libero-sim}, agent params runtime: lerobot) in this
# machine's profile (core.profile; $HYLO_PROFILE or ~/.config/hylo/profile.yaml).
$PY -m hylo.core.profile set-runtime libero-sim --python "$ROOT/$PY" \
  --env MUJOCO_GL=osmesa --env PYOPENGL_PLATFORM=osmesa --env "LD_LIBRARY_PATH=~/.local/mesa/root/usr/lib/x86_64-linux-gnu" \
  --env "LIBERO_CONFIG_PATH=$ROOT/envs/libero/config" --env OMP_NUM_THREADS=1 --env LP_NUM_THREADS=2
$PY -m hylo.core.profile set-runtime lerobot --python "$ROOT/$PY" \
  --env "HF_HOME=$ROOT/envs/hf-home" --env HF_HUB_OFFLINE=1 --env TOKENIZERS_PARALLELISM=false
echo "LIBERO venv ready: $PY (runtimes libero-sim, lerobot)"
