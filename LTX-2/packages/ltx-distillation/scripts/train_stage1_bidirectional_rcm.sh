#!/bin/bash
# =============================================================================
# Stage 1: Bidirectional rCM Distillation (DMD + SCM)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
    CONFIG_PATH="$1"
    shift
else
    CONFIG_PATH="configs/stage1_bidirectional_rcm.yaml"
fi

exec "${SCRIPT_DIR}/train_stage1_bidirectional_dmd.sh" "$CONFIG_PATH" "$@"
