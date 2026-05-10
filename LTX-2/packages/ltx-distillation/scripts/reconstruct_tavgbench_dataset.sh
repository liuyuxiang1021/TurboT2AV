#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIXI_ENV_DIR="${PIXI_ENV_DIR:-/home/jovyan/codes/turbodiff/new_Turbo/OmniForcing/LTX-2/packages/ltx-distillation/downloader-env/.pixi/envs/default}"
BGUTIL_BASE_URL="${BGUTIL_BASE_URL:-http://127.0.0.1:4416}"

if [ -n "${VENV_PATH:-}" ] && [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
    echo "Activated venv: ${VENV_PATH}"
fi

CAPTIONS_FILE="${CAPTIONS_FILE:-/data/datasets/turbodiff_datasets_and_ckpt/tavgbench/release_captions.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/datasets/turbodiff_datasets_and_ckpt/tavgbench}"
RAW_CACHE_DIR="${RAW_CACHE_DIR:-$OUTPUT_DIR/raw_videos}"
CLIPS_DIR="${CLIPS_DIR:-$OUTPUT_DIR/video_clips}"
COMPLETED_INDEX_PATH="${COMPLETED_INDEX_PATH:-}"
MANIFEST_PATH="${MANIFEST_PATH:-$OUTPUT_DIR/video_clips_manifest.jsonl}"
FAILURES_PATH="${FAILURES_PATH:-$OUTPUT_DIR/video_clips_failures.jsonl}"
START_INDEX="${START_INDEX:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_ID="${SHARD_ID:-0}"
YT_DLP_BIN="${YT_DLP_BIN:-$PIXI_ENV_DIR/bin/yt-dlp}"
FFMPEG_BIN="${FFMPEG_BIN:-$PIXI_ENV_DIR/bin/ffmpeg}"
YT_DLP_FORMAT="${YT_DLP_FORMAT:-bv*+ba/b}"
YT_DLP_EXTRACTOR_ARGS="${YT_DLP_EXTRACTOR_ARGS:-youtubepot-bgutilhttp:base_url=$BGUTIL_BASE_URL}"
YT_DLP_VERBOSE="${YT_DLP_VERBOSE:-0}"
YT_DLP_NO_JS_RUNTIMES="${YT_DLP_NO_JS_RUNTIMES:-1}"
YT_DLP_JS_RUNTIMES="${YT_DLP_JS_RUNTIMES:-node:$PIXI_ENV_DIR/bin/node}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-3}"
SLEEP_BETWEEN_DOWNLOADS="${SLEEP_BETWEEN_DOWNLOADS:-0}"
COOKIES_FILE="${COOKIES_FILE:-}"
COOKIES_FROM_BROWSER="${COOKIES_FROM_BROWSER:-}"
OVERWRITE="${OVERWRITE:-0}"
COPY_CODECS="${COPY_CODECS:-0}"
CACHE_SOURCE_VIDEOS="${CACHE_SOURCE_VIDEOS:-0}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-$PIXI_ENV_DIR/bin/python}"

CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/src/ltx_distillation/tools/reconstruct_tavgbench_dataset.py"
  --captions_file "$CAPTIONS_FILE"
  --output_dir "$OUTPUT_DIR"
  --raw_cache_dir "$RAW_CACHE_DIR"
  --clips_dir "$CLIPS_DIR"
  --manifest_path "$MANIFEST_PATH"
  --failures_path "$FAILURES_PATH"
  --start_index "$START_INDEX"
  --num_shards "$NUM_SHARDS"
  --shard_id "$SHARD_ID"
  --yt_dlp_bin "$YT_DLP_BIN"
  --ffmpeg_bin "$FFMPEG_BIN"
  --yt_dlp_format "$YT_DLP_FORMAT"
  --download_retries "$DOWNLOAD_RETRIES"
  --sleep_between_downloads "$SLEEP_BETWEEN_DOWNLOADS"
)

if [[ -n "${COMPLETED_INDEX_PATH}" ]]; then
  CMD+=(--completed_index_path "$COMPLETED_INDEX_PATH")
fi

if [[ -n "${YT_DLP_EXTRACTOR_ARGS}" ]]; then
  CMD+=(--yt_dlp_extractor_args "$YT_DLP_EXTRACTOR_ARGS")
fi

if [[ "$YT_DLP_VERBOSE" == "1" ]]; then
  CMD+=(--yt_dlp_verbose)
fi

if [[ "$YT_DLP_NO_JS_RUNTIMES" == "1" ]]; then
  CMD+=(--yt_dlp_no_js_runtimes)
fi

if [[ -n "${YT_DLP_JS_RUNTIMES}" ]]; then
  IFS=',' read -ra _jsr <<< "$YT_DLP_JS_RUNTIMES"
  for runtime in "${_jsr[@]}"; do
    if [[ -n "$runtime" ]]; then
      CMD+=(--yt_dlp_js_runtimes "$runtime")
    fi
  done
fi

if [[ -n "${COOKIES_FILE}" ]]; then
  CMD+=(--cookies_file "$COOKIES_FILE")
fi

if [[ -n "${COOKIES_FROM_BROWSER}" ]]; then
  CMD+=(--cookies_from_browser "$COOKIES_FROM_BROWSER")
fi

if [[ -n "${NUM_SAMPLES}" ]]; then
  CMD+=(--num_samples "$NUM_SAMPLES")
fi

if [[ "$OVERWRITE" == "1" ]]; then
  CMD+=(--overwrite)
fi

if [[ "$COPY_CODECS" == "1" ]]; then
  CMD+=(--copy_codecs)
fi

if [[ "$CACHE_SOURCE_VIDEOS" == "1" ]]; then
  CMD+=(--cache_source_videos)
fi

if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=(--dry_run)
fi

cd "$ROOT_DIR"
"${CMD[@]}"
