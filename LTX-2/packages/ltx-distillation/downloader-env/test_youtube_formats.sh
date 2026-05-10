#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO_URL="${VIDEO_URL:-https://www.youtube.com/watch?v=P2RPnCJZMBo}"
COOKIES_FILE="${COOKIES_FILE:-/data/datasets/turbodiff_datasets_and_ckpt/tavgbench/www.youtube.com_cookies.txt}"
EXTRACTOR_ARGS="${EXTRACTOR_ARGS:-}"
YT_DLP_FORMAT="${YT_DLP_FORMAT:-bv*+ba/b}"
BGUTIL_BASE_URL="${BGUTIL_BASE_URL:-}"
NODE_RUNTIME="${NODE_RUNTIME:-$ROOT_DIR/.pixi/envs/default/bin/node}"

CMD=(
  pixi run yt-dlp
  --verbose
  --cookies "$COOKIES_FILE"
  --simulate
  --list-formats
  -f "$YT_DLP_FORMAT"
  --no-js-runtimes
  --js-runtimes "node:${NODE_RUNTIME}"
)

if [[ -n "$EXTRACTOR_ARGS" ]]; then
  CMD+=(--extractor-args "$EXTRACTOR_ARGS")
elif [[ -n "$BGUTIL_BASE_URL" ]]; then
  CMD+=(--extractor-args "youtubepot-bgutilhttp:base_url=${BGUTIL_BASE_URL}")
fi

CMD+=("$VIDEO_URL")

cd "$ROOT_DIR"
"${CMD[@]}"
