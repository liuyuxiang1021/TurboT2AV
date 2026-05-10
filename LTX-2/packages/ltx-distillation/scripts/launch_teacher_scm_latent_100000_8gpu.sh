#!/bin/bash
# =============================================================================
# Launch teacher-generated pseudo-SCM latent generation on 8 GPUs via tmux.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NUM_SHARDS="${NUM_SHARDS:-8}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
SESSION_PREFIX="${SESSION_PREFIX:-teacher_scm_100k}"
LOG_DIR="${LOG_DIR:-/data/datasets/turbodiff_datasets_and_ckpt/my_omniforcing/logs}"
OUTPUT_LMDB="${OUTPUT_LMDB:-/data/datasets/turbodiff_datasets_and_ckpt/my_omniforcing/scm_latent_teacher_native_rf_100000_shards}"
PREVIEW_DIR="${PREVIEW_DIR:-/data/datasets/turbodiff_datasets_and_ckpt/my_omniforcing/scm_latent_teacher_native_rf_100000_preview}"
PREVIEW_COUNT="${PREVIEW_COUNT:-0}"
SHARD_STRATEGY="${SHARD_STRATEGY:-modulo}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [ "${#GPUS[@]}" -ne "${NUM_SHARDS}" ]; then
  echo "GPU_LIST length (${#GPUS[@]}) must equal NUM_SHARDS (${NUM_SHARDS})" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_LMDB}"
if [ "${PREVIEW_COUNT}" != "0" ]; then
  mkdir -p "${PREVIEW_DIR}"
fi

for (( shard=0; shard<NUM_SHARDS; shard++ )); do
  mkdir -p "${OUTPUT_LMDB}/shard_$(printf '%05d' "${shard}")"
  if [ "${PREVIEW_COUNT}" != "0" ]; then
    mkdir -p "${PREVIEW_DIR}/shard_$(printf '%05d' "${shard}")"
  fi
done

for (( shard=0; shard<NUM_SHARDS; shard++ )); do
  gpu="${GPUS[$shard]}"
  session="${SESSION_PREFIX}_g${gpu}"
  log_path="${LOG_DIR}/${session}.log"

  tmux has-session -t "${session}" 2>/dev/null && tmux kill-session -t "${session}"

cmd=$(cat <<EOF
cd "${PACKAGE_ROOT}" && CUDA_VISIBLE_DEVICES=${gpu} NUM_SHARDS=${NUM_SHARDS} SHARD_ID=${shard} SHARD_STRATEGY="${SHARD_STRATEGY}" OUTPUT_LMDB="${OUTPUT_LMDB}" PREVIEW_DIR="${PREVIEW_DIR}" PREVIEW_COUNT="${PREVIEW_COUNT}" ./scripts/create_teacher_scm_latent_100000.sh 2>&1 | tee "${log_path}"
EOF
)
  tmux new-session -d -s "${session}" "${cmd}"
  echo "Launched ${session} on GPU ${gpu} -> ${log_path}"
done

echo "All shards launched."
