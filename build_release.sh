#!/usr/bin/env bash
# Build script for OpenTPU-Benchmark release artifacts (v0.1.0)
# Produces:
#   1. dist/OpenTPU-Benchmark (standalone executable zipapp)
#   2. dist/opentpu_benchmark-0.1.0-py3-none-any.whl (pip wheel)
#   3. dist/opentpu_benchmark-0.1.0.tar.gz (source distribution)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Building OpenTPU-Benchmark Release Artifacts ==="

# Clean build and dist dirs
rm -rf build dist bin
mkdir -p build/zipapp_src dist bin

# 1. Build standalone executable (zipapp)
echo "Packaging standalone executable (zipapp)..."
cp -r opentpu build/zipapp_src/
cat << 'EOF' > build/zipapp_src/__main__.py
from opentpu.main import main
if __name__ == "__main__":
    main()
EOF

python3 -m zipapp build/zipapp_src -o dist/OpenTPU-Benchmark -p "/usr/bin/env python3"
chmod +x dist/OpenTPU-Benchmark
# Also create symlink or copy to dist/opentpu and bin/OpenTPU-Benchmark
cp dist/OpenTPU-Benchmark dist/opentpu
cp dist/OpenTPU-Benchmark bin/OpenTPU-Benchmark

# 2. Build Python wheel and sdist
echo "Building standard pip wheel and sdist..."
if [[ -f "$SCRIPT_DIR/.venv/bin/python" ]]; then
    "$SCRIPT_DIR/.venv/bin/python" -m build
else
    python3 -m build
fi

echo "=== Build Complete! Generated Artifacts in dist/: ==="
ls -lh dist/
