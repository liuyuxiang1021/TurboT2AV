#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-/data/datasets/turbodiff_datasets_and_ckpt/tavgbench}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"
CLIPS_DIR="${CLIPS_DIR:-$OUTPUT_DIR/video_clips}"
MANIFEST_DIR="${MANIFEST_DIR:-$OUTPUT_DIR/manifests}"
FAILURES_DIR="${FAILURES_DIR:-$OUTPUT_DIR/failures}"
COOKIES_FILE="${COOKIES_FILE:-/data/datasets/turbodiff_datasets_and_ckpt/tavgbench/www.youtube.com_cookies.txt}"
NUM_WORKERS="${NUM_WORKERS:-8}"
START_INDEX="${START_INDEX:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-5}"
SLEEP_BETWEEN_DOWNLOADS="${SLEEP_BETWEEN_DOWNLOADS:-0}"
SESSION_PREFIX="${SESSION_PREFIX:-tavgbench_reconstruct}"
BGUTIL_SESSION="${BGUTIL_SESSION:-tavgbench_bgutil}"

mkdir -p "$LOG_DIR" "$CLIPS_DIR" "$MANIFEST_DIR" "$FAILURES_DIR"

if ! tmux ls 2>/dev/null | rg -q "^${BGUTIL_SESSION}:"; then
  tmux new-session -d -s "$BGUTIL_SESSION" \
    "cd '$ROOT_DIR/downloader-env' && ./launch_bgutil_provider.sh 2>&1 | tee '$LOG_DIR/bgutil_provider.log'"
  sleep 2
fi

for ((i=0; i<NUM_WORKERS; i++)); do
  session="${SESSION_PREFIX}_$(printf '%02d' "$i")"
  manifest_path="$MANIFEST_DIR/manifest_shard_$(printf '%02d' "$i").jsonl"
  failures_path="$FAILURES_DIR/failures_shard_$(printf '%02d' "$i").jsonl"
  log_path="$LOG_DIR/reconstruct_shard_$(printf '%02d' "$i").log"

  tmux kill-session -t "$session" 2>/dev/null || true

  env_cmd=(
    "COOKIES_FILE='$COOKIES_FILE'"
    "OUTPUT_DIR='$OUTPUT_DIR'"
    "CLIPS_DIR='$CLIPS_DIR'"
    "MANIFEST_PATH='$manifest_path'"
    "FAILURES_PATH='$failures_path'"
    "START_INDEX='$START_INDEX'"
    "NUM_SHARDS='$NUM_WORKERS'"
    "SHARD_ID='$i'"
    "DOWNLOAD_RETRIES='$DOWNLOAD_RETRIES'"
    "SLEEP_BETWEEN_DOWNLOADS='$SLEEP_BETWEEN_DOWNLOADS'"
  )
  if [[ -n "$NUM_SAMPLES" ]]; then
    env_cmd+=("NUM_SAMPLES='$NUM_SAMPLES'")
  fi

  command_str="cd '$ROOT_DIR' && env $(printf '%s ' "${env_cmd[@]}") bash ./scripts/reconstruct_tavgbench_dataset.sh 2>&1 | tee '$log_path'"
  tmux new-session -d -s "$session" "$command_str"
done

tmux ls 2>/dev/null | rg "^(${BGUTIL_SESSION}|${SESSION_PREFIX}_)"
