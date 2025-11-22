#!/usr/bin/env bash
set -euo pipefail

# This script lives in apps/backend/scripts
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

cd "$ROOT_DIR/apps/backend/src"

PORT="${PORT:-8000}"

exec uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --reload
