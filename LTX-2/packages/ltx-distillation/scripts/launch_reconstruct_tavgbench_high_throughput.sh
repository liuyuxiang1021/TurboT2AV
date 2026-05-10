#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-/data/datasets/turbodiff_datasets_and_ckpt/tavgbench}"
COOKIES_FILE="${COOKIES_FILE:-/data/datasets/turbodiff_datasets_and_ckpt/tavgbench/www.youtube.com_cookies.txt}"

CLIPS_DIR="${CLIPS_DIR:-$OUTPUT_DIR/video_clips}"
COMPLETED_INDEX_PATH="${COMPLETED_INDEX_PATH:-$OUTPUT_DIR/completed_filenames.txt}"
RUN_TAG="${RUN_TAG:-fullscan_16p32w}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs_${RUN_TAG}}"
MANIFEST_DIR="${MANIFEST_DIR:-$OUTPUT_DIR/manifests_${RUN_TAG}}"
FAILURES_DIR="${FAILURES_DIR:-$OUTPUT_DIR/failures_${RUN_TAG}}"

NUM_PROVIDERS="${NUM_PROVIDERS:-16}"
WORKERS_PER_PROVIDER="${WORKERS_PER_PROVIDER:-32}"
START_PORT="${START_PORT:-4416}"
START_INDEX="${START_INDEX:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-5}"
SLEEP_BETWEEN_DOWNLOADS="${SLEEP_BETWEEN_DOWNLOADS:-0}"
RESUME_FROM_LOGS="${RESUME_FROM_LOGS:-1}"

BGUTIL_PREFIX="${BGUTIL_PREFIX:-tavgbench_bgutil_${RUN_TAG}}"
SESSION_PREFIX="${SESSION_PREFIX:-tavgbench_reconstruct_${RUN_TAG}}"

mkdir -p "$LOG_DIR" "$MANIFEST_DIR" "$FAILURES_DIR" "$CLIPS_DIR"

find "$CLIPS_DIR" -maxdepth 1 -type f -name '*.mp4' -printf '%f\n' | sort > "$COMPLETED_INDEX_PATH"

total_workers=$((NUM_PROVIDERS * WORKERS_PER_PROVIDER))

compute_resume_start_index() {
  local manifest_path="$1"
  local failures_path="$2"
  local default_start="$3"
  python3 - "$manifest_path" "$failures_path" "$default_start" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
failures_path = Path(sys.argv[2])
default_start = int(sys.argv[3])

max_idx = default_start - 1
for path in (manifest_path, failures_path):
    if not path.exists():
        continue
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                idx = obj.get("source_line_index")
                if isinstance(idx, int) and idx > max_idx:
                    max_idx = idx
    except Exception:
        pass

print(max_idx + 1)
PY
}

for ((p=0; p<NUM_PROVIDERS; p++)); do
  port=$((START_PORT + p))
  provider_session=$(printf "%s_%02d" "$BGUTIL_PREFIX" "$p")
  provider_log="$LOG_DIR/bgutil_provider_$(printf '%02d' "$p").log"

  tmux kill-session -t "$provider_session" 2>/dev/null || true
  tmux new-session -d -s "$provider_session" \
    "cd '$ROOT_DIR/downloader-env' && PORT=$port ./launch_bgutil_provider.sh > '$provider_log' 2>&1"
done

sleep 5

for ((w=0; w<total_workers; w++)); do
  provider_idx=$((w / WORKERS_PER_PROVIDER))
  port=$((START_PORT + provider_idx))
  worker_session=$(printf "%s_%03d" "$SESSION_PREFIX" "$w")
  manifest_path="$MANIFEST_DIR/manifest_shard_$(printf '%03d' "$w").jsonl"
  failures_path="$FAILURES_DIR/failures_shard_$(printf '%03d' "$w").jsonl"
  log_path="$LOG_DIR/reconstruct_shard_$(printf '%03d' "$w").log"
  worker_start_index="$START_INDEX"

  if [[ "$RESUME_FROM_LOGS" == "1" ]]; then
    worker_start_index="$(compute_resume_start_index "$manifest_path" "$failures_path" "$START_INDEX")"
  fi

  tmux kill-session -t "$worker_session" 2>/dev/null || true

  env_cmd=(
    "COOKIES_FILE='$COOKIES_FILE'"
    "OUTPUT_DIR='$OUTPUT_DIR'"
    "CLIPS_DIR='$CLIPS_DIR'"
    "COMPLETED_INDEX_PATH='$COMPLETED_INDEX_PATH'"
    "MANIFEST_PATH='$manifest_path'"
    "FAILURES_PATH='$failures_path'"
    "START_INDEX='$worker_start_index'"
    "NUM_SHARDS='$total_workers'"
    "SHARD_ID='$w'"
    "DOWNLOAD_RETRIES='$DOWNLOAD_RETRIES'"
    "SLEEP_BETWEEN_DOWNLOADS='$SLEEP_BETWEEN_DOWNLOADS'"
    "BGUTIL_BASE_URL='http://127.0.0.1:$port'"
  )
  if [[ -n "$NUM_SAMPLES" ]]; then
    env_cmd+=("NUM_SAMPLES='$NUM_SAMPLES'")
  fi

  command_str="cd '$ROOT_DIR' && env $(printf '%s ' "${env_cmd[@]}") bash ./scripts/reconstruct_tavgbench_dataset.sh > '$log_path' 2>&1"
  tmux new-session -d -s "$worker_session" "$command_str"
done

echo "run_tag=$RUN_TAG"
echo "providers=$NUM_PROVIDERS workers_per_provider=$WORKERS_PER_PROVIDER total_workers=$total_workers"
echo "clips_dir=$CLIPS_DIR"
echo "log_dir=$LOG_DIR"
echo "manifest_dir=$MANIFEST_DIR"
echo "failures_dir=$FAILURES_DIR"
tmux ls 2>/dev/null | rg "^(${BGUTIL_PREFIX}_|${SESSION_PREFIX}_)"
