#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIXI_ENV_DIR="${PIXI_ENV_DIR:-/home/jovyan/codes/turbodiff/new_Turbo/OmniForcing/LTX-2/packages/ltx-distillation/downloader-env/.pixi/envs/default}"
PYTHON_BIN="${PYTHON_BIN:-$PIXI_ENV_DIR/bin/python}"

CAPTIONS_FILE="${CAPTIONS_FILE:-/data/datasets/turbodiff_datasets_and_ckpt/tavgbench/release_captions.txt}"
COMPLETED_FILES="${COMPLETED_FILES:-/data/datasets/turbodiff_datasets_and_ckpt/tavgbench/video_clips/completed_files.txt}"
OUTPUT_FILE="${OUTPUT_FILE:-/data/datasets/turbodiff_datasets_and_ckpt/tavgbench/pending_release_captions.txt}"

cd "$ROOT_DIR"
"$PYTHON_BIN" "$ROOT_DIR/src/ltx_distillation/tools/build_pending_tavgbench_list.py" \
  --captions_file "$CAPTIONS_FILE" \
  --completed_files "$COMPLETED_FILES" \
  --output_file "$OUTPUT_FILE"
