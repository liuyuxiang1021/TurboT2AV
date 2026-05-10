#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISTILLATION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LTX2_ROOT="$(cd "${DISTILLATION_ROOT}/../.." && pwd)"

PATH_PREFIX="${LTX2_ROOT}/.pixi/envs/default/bin"
export PATH="${PATH_PREFIX}:${PATH}"
export PYTHONPATH="${DISTILLATION_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

CAPTIONS_FILE="${CAPTIONS_FILE:-/data/datasets/turbodiff_datasets_and_ckpt/tavgbench/release_captions.txt}"
VIDEO_DIR="${VIDEO_DIR:-/data/datasets/turbodiff_datasets_and_ckpt/tavgbench/video_clips}"
OUTPUT_FILE="${OUTPUT_FILE:-/data/datasets/turbodiff_datasets_and_ckpt/tavgbench/omniforcing_video_caption_manifest.jsonl}"

cd "${DISTILLATION_ROOT}"
python -m ltx_distillation.tools.build_video_caption_manifest \
  --captions_file "${CAPTIONS_FILE}" \
  --video_dir "${VIDEO_DIR}" \
  --output_file "${OUTPUT_FILE}"
