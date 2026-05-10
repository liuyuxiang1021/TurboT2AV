#!/bin/bash
# =============================================================================
# SCM Latent LMDB Creation - 8 GPU Parallel Shards
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISTILLATION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -n "${VENV_PATH:-}" ] && [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
    echo "Activated venv: ${VENV_PATH}"
fi

NUM_SHARDS="${NUM_SHARDS:-8}"
START_GPU="${START_GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_BATCHES="${PREFETCH_BATCHES:-4}"
SOURCE_INDEX_START="${SOURCE_INDEX_START:-}"
SOURCE_INDEX_END="${SOURCE_INDEX_END:-}"
MANIFEST_PATH="${MANIFEST_PATH:-/data/datasets/turbodiff_datasets_and_ckpt/tavgbench/omniforcing_video_caption_manifest.jsonl}"
CAPTIONS_PATH="${CAPTIONS_PATH:-/data/datasets/turbodiff_datasets_and_ckpt/tavgbench/release_captions.txt}"
VIDEO_DIR="${VIDEO_DIR:-/data/datasets/turbodiff_datasets_and_ckpt/tavgbench/video_clips}"

echo "=============================================="
echo "SCM Latent LMDB Creation (Parallel)"
echo "=============================================="
echo "Shards / GPUs: ${NUM_SHARDS}"
echo "Start GPU:     ${START_GPU}"
echo "Batch size:    ${BATCH_SIZE}"
echo "Workers:       ${NUM_WORKERS}"
echo "Prefetch:      ${PREFETCH_BATCHES}"
echo "Source range:  ${SOURCE_INDEX_START:-<none>} .. ${SOURCE_INDEX_END:-<none>}"
echo "Manifest:      ${MANIFEST_PATH}"
echo "Output root:   ${OUTPUT_LMDB:-/data/datasets/turbodiff_datasets_and_ckpt/my_omniforcing/scm_latent}"
echo "=============================================="

PATH_PREFIX="${DISTILLATION_ROOT}/../../.pixi/envs/default/bin"
export PATH="${PATH_PREFIX}:${PATH}"
export PYTHONPATH="${DISTILLATION_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -m ltx_distillation.tools.build_video_caption_manifest \
    --captions_file "${CAPTIONS_PATH}" \
    --video_dir "${VIDEO_DIR}" \
    --output_file "${MANIFEST_PATH}"

pids=()

for ((i=0; i<NUM_SHARDS; i++)); do
    gpu_id=$((START_GPU + i))
    echo "[launch] shard ${i}/${NUM_SHARDS} on cuda:${gpu_id}"
    (
        export CUDA_VISIBLE_DEVICES="${gpu_id}"
        export DEVICE="cuda:0"
        export MANIFEST_PATH="${MANIFEST_PATH}"
        export CAPTIONS_PATH="${CAPTIONS_PATH}"
        export VIDEO_DIR="${VIDEO_DIR}"
        export SYNC_MANIFEST="0"
        export BATCH_SIZE="${BATCH_SIZE}"
        export NUM_WORKERS="${NUM_WORKERS}"
        export PREFETCH_BATCHES="${PREFETCH_BATCHES}"
        export SOURCE_INDEX_START="${SOURCE_INDEX_START}"
        export SOURCE_INDEX_END="${SOURCE_INDEX_END}"
        export SHARD_ID="${i}"
        export NUM_SHARDS="${NUM_SHARDS}"
        export OVERWRITE="${OVERWRITE:-0}"
        export RESUME="${RESUME:-1}"
        "${SCRIPT_DIR}/create_scm_latent_lmdb.sh"
    ) &
    pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

if [ "${status}" -ne 0 ]; then
    echo "[error] at least one shard process failed" >&2
    exit "${status}"
fi

echo "=============================================="
echo "All shard processes completed successfully."
echo "Point scm_data_path to the output root directory."
echo "=============================================="
