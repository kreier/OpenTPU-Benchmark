#!/usr/bin/env bash
# OpenTPU-Benchmark Quick Runner for Google Colab
# Downloads the 36 KiB standalone executable binary and runs it directly.
set -e

if command -v curl &> /dev/null; then
    curl -sSL -o OpenTPU-Benchmark https://github.com/kreier/OpenTPU-Benchmark/releases/latest/download/OpenTPU-Benchmark
elif command -v wget &> /dev/null; then
    wget -q -O OpenTPU-Benchmark https://github.com/kreier/OpenTPU-Benchmark/releases/latest/download/OpenTPU-Benchmark
else
    echo "Error: Neither curl nor wget found." >&2
    exit 1
fi

chmod +x OpenTPU-Benchmark
./OpenTPU-Benchmark "$@"
