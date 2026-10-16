#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo "Install uv 0.11.19 first." >&2; exit 1; }
python3 scripts/lock_dependencies.py --sync "$@"
