#!/usr/bin/env bash
set -euo pipefail

# This script lives in scripts/
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

# Simple: just start the Ollama server in foreground
exec ollama serve
