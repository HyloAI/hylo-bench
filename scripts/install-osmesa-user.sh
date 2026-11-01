#!/usr/bin/env bash
# Install Mesa OSMesa (CPU offscreen OpenGL) into the user's home without root, for MuJoCo rendering.
# Ubuntu 24.04: downloads the distribution .deb files and extracts them into ~/.local/mesa.
# Use with: MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH=~/.local/mesa/root/usr/lib/x86_64-linux-gnu
# (examples/external_demo/run_demo.py sets these automatically when the install exists.)
set -euo pipefail
DEST="${HOME}/.local/mesa"
mkdir -p "${DEST}/debs"
cd "${DEST}/debs"
apt-get download libosmesa6
ldconfig -p | grep -q "libLLVM.so.20" || apt-get download libllvm20
for f in *.deb; do dpkg-deb -x "$f" "${DEST}/root"; done
LIB="${DEST}/root/usr/lib/x86_64-linux-gnu"
if LD_LIBRARY_PATH="${LIB}" ldd "${LIB}/libOSMesa.so.8" | grep -q "not found"; then
  echo "missing libraries:"; LD_LIBRARY_PATH="${LIB}" ldd "${LIB}/libOSMesa.so.8" | grep "not found"; exit 1
fi
echo "OSMesa installed in ${LIB}"
