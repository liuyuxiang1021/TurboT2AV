#!/bin/bash

set -euo pipefail

if [ -n "${VENV_PATH:-}" ] && [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
fi

LMDB_ROOT="${LMDB_ROOT:-/data/datasets/turbodiff_datasets_and_ckpt/tavgbench/omniforcing_latent}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/data/datasets/turbodiff_datasets_and_ckpt/omniforcing/ltx-2-19b-dev.safetensors}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/datasets/turbodiff_datasets_and_ckpt/tavgbench/latent_decode_check}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
SAMPLES_PER_SHARD="${SAMPLES_PER_SHARD:-}"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"
VIDEO_FPS="${VIDEO_FPS:-24}"
AUDIO_SAMPLE_RATE="${AUDIO_SAMPLE_RATE:-48000}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"

echo "=============================================="
echo "SCM Latent Decode Verification"
echo "=============================================="
echo "LMDB root:     ${LMDB_ROOT}"
echo "Checkpoint:    ${CHECKPOINT_PATH}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "Num samples:   ${NUM_SAMPLES}"
echo "Per shard:     ${SAMPLES_PER_SHARD:-<none>}"
echo "Max samples:   ${MAX_SAMPLES}"
echo "Device/Dtype:  ${DEVICE} / ${DTYPE}"
echo "=============================================="

CMD=(
    python -m ltx_distillation.tools.verify_scm_latent_decode
    --lmdb_root "${LMDB_ROOT}"
    --checkpoint_path "${CHECKPOINT_PATH}"
    --output_dir "${OUTPUT_DIR}"
    --num_samples "${NUM_SAMPLES}"
    --max_samples "${MAX_SAMPLES}"
    --video_fps "${VIDEO_FPS}"
    --audio_sample_rate "${AUDIO_SAMPLE_RATE}"
    --seed "${SEED}"
    --device "${DEVICE}"
    --dtype "${DTYPE}"
)
if [ -n "${SAMPLES_PER_SHARD}" ]; then
    CMD+=(--samples_per_shard "${SAMPLES_PER_SHARD}")
fi

"${CMD[@]}"
