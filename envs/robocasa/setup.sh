#!/usr/bin/env bash
# Build the RoboCasa venv used by out-of-process EnvServer workers (hylo-env-robocasa): RoboCasa 1.0.1 on robosuite
# 1.5 (master), CPU only, OSMesa rendering (scripts/install-osmesa-user.sh). Sizes: venv about 1.2 GB, RoboCasa
# checkout 0.1 GB, assets 5.2 GB downloaded (11.2 GB once extracted, zips removed). RoboCasa's AI-generated object
# set (5.8 GB more) is not fetched: the scenes are built from the objaverse and lightwheel registries only
# (RoboCasa's default obj_registries), so nothing else is needed.
set -euo pipefail
cd "$(dirname "$0")/../.."                       # the bench repository
ROOT=$(pwd)
HARNESS=${HYLO_HARNESS:-../harness}                 # hylo-harness, checked out next to this repository
DIR=envs/robocasa
PY=$DIR/.venv/bin/python
ROBOCASA_COMMIT=3443cdf3dd20f9ae8ed41adba442059b506891d0       # robocasa main, 2026-10-30 (v1.0.1)
ROBOSUITE_COMMIT=5ce6643f3092639d08f7b0f90ed1c6a84f50552c      # robosuite master, 2026-07-11 (1.5.2+)
[ -d $DIR/src-robocasa/.git ] || git clone -q https://github.com/robocasa/robocasa $DIR/src-robocasa
git -C $DIR/src-robocasa checkout -q $ROBOCASA_COMMIT
[ -x $PY ] || uv venv -q $DIR/.venv --python 3.12
# overrides.txt drops RoboCasa's training / teleoperation dependencies (lerobot, tianshou: torch) and pins numba to a
# release whose LLVM matches OSMesa's (see the file)
uv pip install --python $PY "robosuite @ git+https://github.com/ARISE-Initiative/robosuite@$ROBOSUITE_COMMIT" \
  -e $DIR/src-robocasa opencv-python-headless imageio "websockets>=13" --override $DIR/overrides.txt
uv pip install -q --python $PY -e packages/hylo-robosuite-contract -e adapters/hylo-env-robocasa \
  -e "$HARNESS/packages/hylo-core" --override $DIR/overrides.txt
# Assets (RoboCasa's own download links; research use, not redistributed), each zip in its own folder first and
# extracted with a path check, into the checkout where RoboCasa looks for them.
ASSETS=$DIR/src-robocasa/robocasa/models/assets
mkdir -p $DIR/downloads $ASSETS/objects
fetch() {   # name, box id, destination
  [ -f "$DIR/downloads/$1.done" ] && return
  curl -sSL -C - --retry 5 -o "$DIR/downloads/$1.zip" "https://utexas.box.com/shared/static/$2.zip"
  $PY -I - "$DIR/downloads/$1.zip" "$3" <<'PYEOF'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
for info in z.infolist():
    n = info.filename
    if n.startswith("/") or ".." in n.split("/") or n == "README.md":
        continue
    z.extract(info, sys.argv[2])
PYEOF
  rm -f "$DIR/downloads/$1.zip" && touch "$DIR/downloads/$1.done"
}
fetch textures 4i85ileasdvstmlln5sbvzptz7keuoy1 $ASSETS                       # 0.54 GB
fetch generative_textures ebaad09k82tmfmlq6ohdkmrh8izl9vn5 $ASSETS            # 1.18 GB (some kitchen styles use them)
fetch fixtures_lightwheel idbncsadpnaz1jfl4i6m8qejawk7p9pi $ASSETS            # 0.53 GB
fetch objects_lightwheel vckqvvkh1z8t69k8qcpcmee6k66stii4 $ASSETS/objects     # 0.79 GB
fetch objaverse 03eionyo8fk3a9dsksq9jb8du5lqfw8h $ASSETS/objects              # 2.16 GB
# Register the runtime experiment configs name (env.remote: {runtime: robocasa-sim}) in this machine's profile.
$PY -m hylo.core.profile set-runtime robocasa-sim --python "$ROOT/$PY" \
  --env MUJOCO_GL=osmesa --env PYOPENGL_PLATFORM=osmesa --env "LD_LIBRARY_PATH=~/.local/mesa/root/usr/lib/x86_64-linux-gnu" \
  --env OMP_NUM_THREADS=1 --env LP_NUM_THREADS=2 --env NUMBA_CACHE_DIR="$ROOT/$DIR/numba-cache"
echo "RoboCasa venv ready: $PY (runtime robocasa-sim)"
