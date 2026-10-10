#!/usr/bin/env bash
# OpenTPU-Benchmark Quick Runner for Google Colab / Linux
# https://kreier.github.io/OpenTPU-Benchmark/
set -e

BIN_NAME="OpenTPU-Benchmark"
RELEASE_URL="https://github.com/kreier/OpenTPU-Benchmark/releases/latest/download/${BIN_NAME}"

if command -v curl &> /dev/null; then
    curl -sSL -o "${BIN_NAME}" "${RELEASE_URL}"
elif command -v wget &> /dev/null; then
    wget -q -O "${BIN_NAME}" "${RELEASE_URL}"
else
    echo "Error: Neither curl nor wget found." >&2
    exit 1
fi

chmod +x "${BIN_NAME}"
./"${BIN_NAME}" "$@"
