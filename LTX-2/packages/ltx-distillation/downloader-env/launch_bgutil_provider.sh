#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROVIDER_ROOT="${PROVIDER_ROOT:-$ROOT_DIR/provider-src/bgutil-ytdlp-pot-provider}"
PORT="${PORT:-4416}"

SERVER_DIR="$PROVIDER_ROOT/server"
if [[ ! -f "$SERVER_DIR/build/main.js" ]]; then
  echo "bgutil provider is not built yet. Run setup_bgutil_provider.sh first." >&2
  exit 1
fi

cd "$SERVER_DIR"
exec "$ROOT_DIR/.pixi/envs/default/bin/node" build/main.js --port "$PORT"
