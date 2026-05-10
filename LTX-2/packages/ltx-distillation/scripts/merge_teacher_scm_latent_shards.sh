#!/bin/bash
# =============================================================================
# Merge teacher pseudo-SCM shard LMDBs into a single LMDB.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.pixi/envs/default/bin/python"

if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN="python3"
fi

SHARDS_ROOT="${SHARDS_ROOT:-/data/datasets/turbodiff_datasets_and_ckpt/my_omniforcing/scm_latent_teacher_native_rf_100000_shards}"
OUTPUT_LMDB="${OUTPUT_LMDB:-/data/datasets/turbodiff_datasets_and_ckpt/my_omniforcing/scm_latent_teacher_native_rf_100000}"
MAP_SIZE="${MAP_SIZE:-2000000000000}"
COMMIT_INTERVAL="${COMMIT_INTERVAL:-8}"
OVERWRITE="${OVERWRITE:-0}"
RESUME="${RESUME:-1}"

echo "=============================================="
echo "Merge teacher pseudo-SCM LMDB shards"
echo "=============================================="
echo "Shards root:   ${SHARDS_ROOT}"
echo "Output LMDB:   ${OUTPUT_LMDB}"
echo "Resume:        ${RESUME}"
echo "Overwrite:     ${OVERWRITE}"
echo "=============================================="

CMD=(
  "${PYTHON_BIN}" -m ltx_distillation.tools.merge_teacher_scm_lmdb_shards
  --shards_root "${SHARDS_ROOT}"
  --output_lmdb "${OUTPUT_LMDB}"
  --map_size "${MAP_SIZE}"
  --commit_interval "${COMMIT_INTERVAL}"
)

if [ "${OVERWRITE}" = "1" ]; then
  CMD+=(--overwrite)
fi

if [ "${RESUME}" = "1" ]; then
  CMD+=(--resume)
fi

cd "${PACKAGE_ROOT}"
"${CMD[@]}"
