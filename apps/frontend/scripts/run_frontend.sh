#!/usr/bin/env bash
set -euo pipefail

# This script lives in apps/frontend/scripts
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

cd "$ROOT_DIR"

FRONTEND_PORT="${FRONTEND_PORT:-3000}"

exec python -m streamlit run apps/frontend/ui.py \
  --server.port "$FRONTEND_PORT"
