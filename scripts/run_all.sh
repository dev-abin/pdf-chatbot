#!/usr/bin/env bash
set -euo pipefail

# This script lives in scripts/
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Optional: activate venv once for all processes
# if [[ -f "$ROOT_DIR/chatenv/bin/activate" ]]; then
#   source "$ROOT_DIR/chatenv/bin/activate"
# fi

echo "[run_all] Starting Ollama..."
bash "$ROOT_DIR/scripts/run_ollama.sh" &

echo "[run_all] Starting backend..."
bash "$ROOT_DIR/apps/backend/scripts/run_backend.sh" &

echo "[run_all] Starting frontend..."
bash "$ROOT_DIR/apps/frontend/scripts/run_frontend.sh" &

# Ensure logs dir + file exists
mkdir -p "$ROOT_DIR/logs"
touch "$ROOT_DIR/logs/app.log"

echo "[run_all] Tailing logs/app.log..."
cd "$ROOT_DIR"
tail -f logs/app.log
