#!/bin/bash
# =============================================================================
# SCM Latent LMDB Creation
# =============================================================================
# Encodes real video/audio samples into clean latents for faithful SCM training.
#
# Manifest format (JSONL):
#   {"prompt": "...", "video_path": "/abs/path/sample.mp4"}
#   {"prompt": "...", "video_path": "/abs/path/sample.mp4", "audio_path": "/abs/path/sample.wav"}

set -euo pipefail

if [ -n "${VENV_PATH:-}" ] && [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
    echo "Activated venv: ${VENV_PATH}"
fi

MANIFEST_PATH="${MANIFEST_PATH:-}"
CAPTIONS_PATH="${CAPTIONS_PATH:-}"
VIDEO_DIR="${VIDEO_DIR:-}"
SYNC_MANIFEST="${SYNC_MANIFEST:-0}"
OUTPUT_LMDB="${OUTPUT_LMDB:-/data/datasets/turbodiff_datasets_and_ckpt/my_omniforcing/scm_latent}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/data/datasets/turbodiff_datasets_and_ckpt/omniforcing/ltx-2-19b-dev.safetensors}"

NUM_FRAMES="${NUM_FRAMES:-121}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-512}"
VIDEO_WIDTH="${VIDEO_WIDTH:-768}"
VIDEO_FPS="${VIDEO_FPS:-24}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PREFETCH_BATCHES="${PREFETCH_BATCHES:-2}"
PIN_MEMORY="${PIN_MEMORY:-0}"
MAP_SIZE="${MAP_SIZE:-500000000000}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SOURCE_INDEX_START="${SOURCE_INDEX_START:-}"
SOURCE_INDEX_END="${SOURCE_INDEX_END:-}"
ALLOW_MISSING_AUDIO="${ALLOW_MISSING_AUDIO:-0}"
OVERWRITE="${OVERWRITE:-0}"
RESUME="${RESUME:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_ID="${SHARD_ID:-0}"

echo "=============================================="
echo "SCM Latent LMDB Creation"
echo "=============================================="
if [ -n "${MANIFEST_PATH}" ]; then
    echo "Manifest:      ${MANIFEST_PATH}"
else
    echo "Captions:      ${CAPTIONS_PATH}"
    echo "Video dir:     ${VIDEO_DIR}"
fi
echo "Output:        ${OUTPUT_LMDB}"
echo "Checkpoint:    ${CHECKPOINT_PATH}"
echo "Frames:        ${NUM_FRAMES}"
echo "Resolution:    ${VIDEO_HEIGHT}x${VIDEO_WIDTH}"
echo "Video FPS:     ${VIDEO_FPS}"
echo "Device/Dtype:  ${DEVICE} / ${DTYPE}"
echo "Batch size:    ${BATCH_SIZE}"
echo "Workers:       ${NUM_WORKERS}"
echo "Prefetch:      ${PREFETCH_BATCHES}"
echo "Pin memory:    ${PIN_MEMORY}"
echo "Shard:         ${SHARD_ID}/${NUM_SHARDS}"
echo "Source range:  ${SOURCE_INDEX_START:-<none>} .. ${SOURCE_INDEX_END:-<none>}"
echo "Resume:        ${RESUME}"
echo "=============================================="

if [ "${SYNC_MANIFEST}" = "1" ]; then
    if [ -z "${MANIFEST_PATH}" ] || [ -z "${CAPTIONS_PATH}" ] || [ -z "${VIDEO_DIR}" ]; then
        echo "[error] SYNC_MANIFEST=1 requires MANIFEST_PATH, CAPTIONS_PATH, and VIDEO_DIR" >&2
        exit 1
    fi
    python -m ltx_distillation.tools.build_video_caption_manifest \
        --captions_file "${CAPTIONS_PATH}" \
        --video_dir "${VIDEO_DIR}" \
        --output_file "${MANIFEST_PATH}"
fi

CMD=(
    python -m ltx_distillation.tools.create_scm_latent_lmdb
    --output_lmdb "${OUTPUT_LMDB}"
    --checkpoint_path "${CHECKPOINT_PATH}"
    --num_frames "${NUM_FRAMES}"
    --video_height "${VIDEO_HEIGHT}"
    --video_width "${VIDEO_WIDTH}"
    --video_fps "${VIDEO_FPS}"
    --dtype "${DTYPE}"
    --device "${DEVICE}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --prefetch_batches "${PREFETCH_BATCHES}"
    --map_size "${MAP_SIZE}"
    --num_shards "${NUM_SHARDS}"
    --shard_id "${SHARD_ID}"
)

if [ -n "${MANIFEST_PATH}" ]; then
    CMD+=(--manifest_path "${MANIFEST_PATH}")
else
    CMD+=(--captions_path "${CAPTIONS_PATH}" --video_dir "${VIDEO_DIR}")
fi

if [ -n "${MAX_SAMPLES}" ]; then
    CMD+=(--max_samples "${MAX_SAMPLES}")
fi

if [ -n "${SOURCE_INDEX_START}" ]; then
    CMD+=(--source_index_start "${SOURCE_INDEX_START}")
fi

if [ -n "${SOURCE_INDEX_END}" ]; then
    CMD+=(--source_index_end "${SOURCE_INDEX_END}")
fi

if [ "${ALLOW_MISSING_AUDIO}" = "1" ]; then
    CMD+=(--allow_missing_audio)
fi

if [ "${OVERWRITE}" = "1" ]; then
    CMD+=(--overwrite)
fi

if [ "${RESUME}" = "1" ]; then
    CMD+=(--resume)
fi

if [ "${PIN_MEMORY}" = "1" ]; then
    CMD+=(--pin_memory)
fi

"${CMD[@]}"

echo "=============================================="
echo "SCM latent LMDB creation complete!"
echo "Output saved to: ${OUTPUT_LMDB}"
echo "=============================================="
