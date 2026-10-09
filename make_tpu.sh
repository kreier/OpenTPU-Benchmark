#!/usr/bin/env bash
# OpenTPU-Benchmark Execution Script
# Parity with make.sh: compiles executable to bin/OpenTPU-Benchmark and runs it
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p bin

# Detect Python interpreter
if [[ -f "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
else
    echo "Error: Python 3 not found." >&2
    exit 1
fi

# Package standalone executable into bin/OpenTPU-Benchmark if needed or changed
mkdir -p build/zipapp_src
rm -rf build/zipapp_src/*
cp -r opentpu build/zipapp_src/
cat << 'EOF' > build/zipapp_src/__main__.py
from opentpu.main import main
if __name__ == "__main__":
    main()
EOF

"$PYTHON_BIN" -m zipapp build/zipapp_src -o bin/OpenTPU-Benchmark -p "/usr/bin/env python3"
chmod +x bin/OpenTPU-Benchmark

# Run benchmark executable
"$PYTHON_BIN" bin/OpenTPU-Benchmark "$@"
