#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROVIDER_VERSION="${PROVIDER_VERSION:-1.3.1}"
PROVIDER_ROOT="${PROVIDER_ROOT:-$ROOT_DIR/provider-src/bgutil-ytdlp-pot-provider}"

mkdir -p "$(dirname "$PROVIDER_ROOT")"

if [[ ! -d "$PROVIDER_ROOT/.git" ]]; then
  git clone --single-branch --branch "$PROVIDER_VERSION" \
    https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git \
    "$PROVIDER_ROOT"
else
  git -C "$PROVIDER_ROOT" fetch --tags --depth=1 origin "$PROVIDER_VERSION"
  git -C "$PROVIDER_ROOT" checkout "$PROVIDER_VERSION"
fi

cd "$ROOT_DIR"
pixi install >/dev/null

cd "$PROVIDER_ROOT/server"
"$ROOT_DIR/.pixi/envs/default/bin/npm" ci
"$ROOT_DIR/.pixi/envs/default/bin/npx" tsc

echo "Provider ready at: $PROVIDER_ROOT/server"
