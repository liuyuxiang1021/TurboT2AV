#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIXI_RUN=(pixi run --manifest-path "$ROOT_DIR")
DISPLAY_NUM="${DISPLAY_NUM:-100}"
XPRA_PORT="${XPRA_PORT:-14500}"
PROFILE_DIR="${PROFILE_DIR:-$ROOT_DIR/profile}"
SESSION_NAME="${SESSION_NAME:-tavgbench-firefox}"
CONDA_PREFIX="$("${PIXI_RUN[@]}" python -c 'import os; print(os.environ.get("CONDA_PREFIX", ""))')"
XPRA_BIN="${XPRA_BIN:-$("${PIXI_RUN[@]}" which xpra)}"
FIREFOX_BIN="${FIREFOX_BIN:-$("${PIXI_RUN[@]}" which firefox)}"
XVFB_BIN="${XVFB_BIN:-$(find "$CONDA_PREFIX" -path '*/Xvfb' | head -n 1)}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/bin/FirefoxApp:${CONDA_PREFIX}/x86_64-conda-linux-gnu/sysroot/usr/lib64:${CONDA_PREFIX}/x86_64-conda-linux-gnu/sysroot/usr/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$PROFILE_DIR"

exec "${PIXI_RUN[@]}" xpra start ":${DISPLAY_NUM}" \
  --daemon=no \
  --bind-tcp="0.0.0.0:${XPRA_PORT}" \
  --html=on \
  --mdns=no \
  --notifications=no \
  --pulseaudio=no \
  --opengl=no \
  --exit-with-children=yes \
  --xvfb="${XVFB_BIN} -screen 0 1920x1080x24 -nolisten tcp -noreset -auth \$XAUTHORITY" \
  --start-child="$FIREFOX_BIN --no-remote --profile $PROFILE_DIR https://www.youtube.com" \
  --session-name="$SESSION_NAME"
