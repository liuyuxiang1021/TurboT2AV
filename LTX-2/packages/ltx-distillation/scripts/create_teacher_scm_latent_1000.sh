#!/bin/bash
# =============================================================================
# Teacher-generated pseudo-SCM latent dataset (1000 prompts)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.pixi/envs/default/bin/python"

if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN="python3"
fi

CONFIG_PATH="${CONFIG_PATH:-${PACKAGE_ROOT}/configs/stage1_bidirectional_rcm.yaml}"
PROMPTS_FILE="${PROMPTS_FILE:-/data/datasets/turbodiff_datasets_and_ckpt/tavgbench/release_prompts.txt}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
START_INDEX="${START_INDEX:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_ID="${SHARD_ID:-0}"
SHARD_STRATEGY="${SHARD_STRATEGY:-modulo}"
OUTPUT_LMDB="${OUTPUT_LMDB:-/data/datasets/turbodiff_datasets_and_ckpt/my_omniforcing/scm_latent_teacher_1000}"
PREVIEW_DIR="${PREVIEW_DIR:-/data/datasets/turbodiff_datasets_and_ckpt/my_omniforcing/scm_latent_teacher_1000_preview}"
PREVIEW_COUNT="${PREVIEW_COUNT:-16}"
MODE="${MODE:-native_rf}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
SEED="${SEED:-12345}"
OVERWRITE="${OVERWRITE:-1}"
RESUME="${RESUME:-0}"
MAP_SIZE="${MAP_SIZE:-500000000000}"

echo "=============================================="
echo "Teacher pseudo-SCM latent generation"
echo "=============================================="
echo "Config:        ${CONFIG_PATH}"
echo "Prompts:       ${PROMPTS_FILE}"
echo "Num prompts:   ${NUM_PROMPTS}"
echo "Shard:         ${SHARD_ID}/${NUM_SHARDS}"
echo "Strategy:      ${SHARD_STRATEGY}"
echo "Mode/steps:    ${MODE} / ${NUM_INFERENCE_STEPS}"
echo "Output LMDB:   ${OUTPUT_LMDB}"
echo "Preview dir:   ${PREVIEW_DIR}"
echo "Resume:        ${RESUME}"
echo "Overwrite:     ${OVERWRITE}"
echo "=============================================="

CMD=(
  "${PYTHON_BIN}" -m ltx_distillation.tools.create_teacher_scm_latent_lmdb
  --config_path "${CONFIG_PATH}"
  --prompts_file "${PROMPTS_FILE}"
  --num_prompts "${NUM_PROMPTS}"
  --start_index "${START_INDEX}"
  --num_shards "${NUM_SHARDS}"
  --shard_id "${SHARD_ID}"
  --shard_strategy "${SHARD_STRATEGY}"
  --output_lmdb "${OUTPUT_LMDB}"
  --preview_dir "${PREVIEW_DIR}"
  --preview_count "${PREVIEW_COUNT}"
  --mode "${MODE}"
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
  --seed "${SEED}"
  --map_size "${MAP_SIZE}"
)

if [ "${OVERWRITE}" = "1" ]; then
  CMD+=(--overwrite)
fi

if [ "${RESUME}" = "1" ]; then
  CMD+=(--resume)
fi

cd "${PACKAGE_ROOT}"
"${CMD[@]}"
