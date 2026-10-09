#!/usr/bin/env bash
# OpenTPU-Benchmark Execution Script
# Usage: ./make_tpu.sh [args]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect Python interpreter
if [[ -f "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
else
    echo "Error: Python 3 not found." >&2
    exit 1
fi

"$PYTHON_BIN" -m opentpu.main "$@"
