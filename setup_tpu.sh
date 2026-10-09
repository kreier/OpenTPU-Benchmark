#!/usr/bin/env bash
# Setup script for Cloud TPU VM (e.g. TPU v5e-1)
set -e

echo "=== Setting up OpenTPU-Benchmark on Cloud TPU ==="

# Check if running in a venv, otherwise create one
if [[ -z "$VIRTUAL_ENV" ]]; then
    if [[ ! -d ".venv" ]]; then
        echo "Creating virtual environment .venv..."
        python3 -m venv .venv
    fi
    source .venv/bin/activate
fi

echo "Installing pip requirements and TPU JAX runtime..."
pip install --upgrade pip
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -r requirements.txt

echo "Verifying TPU runtime..."
python3 -c "import jax; print('JAX version:', jax.__version__); print('Discovered devices:', jax.devices())"

echo "=== Setup complete! Run with: ./make_tpu.sh ==="
