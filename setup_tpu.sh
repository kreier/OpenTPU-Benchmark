#!/usr/bin/env bash
# Setup script for Cloud TPU VM and Google Colab
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Checking TPU & JAX Environment ==="

# Check if TPU-enabled JAX is already functional
if python3 -c "import jax; devs = [d.platform for d in jax.devices()]; print(f'Detected devices: {jax.devices()}'); assert 'tpu' in devs" 2>/dev/null; then
    echo "TPU-enabled JAX is already configured and functional. Skipping installation."
    echo "Run benchmark with: ./make_tpu.sh"
    exit 0
fi

# Detect Google Colab environment
IS_COLAB=false
if python3 -c "import google.colab" 2>/dev/null || [[ -d "/content" ]]; then
    IS_COLAB=true
    echo "Running inside Google Colab."
fi

if [[ "$IS_COLAB" == false ]] && [[ -z "$VIRTUAL_ENV" ]]; then
    if [[ ! -d ".venv" ]]; then
        echo "Creating virtual environment .venv..."
        if ! python3 -m venv .venv 2>/dev/null; then
            echo "Note: python3 -m venv failed (ensurepip missing). Continuing with current environment."
        fi
    fi
    if [[ -f ".venv/bin/activate" ]]; then
        source .venv/bin/activate
    fi
fi

echo "Installing/updating TPU JAX runtime..."
pip install --upgrade pip 2>/dev/null || true
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html 2>/dev/null || pip install jax jaxlib
pip install -r requirements.txt

echo "Verifying runtime..."
python3 -c "import jax; print('JAX version:', jax.__version__); print('Discovered devices:', jax.devices())"

echo "=== Setup complete! Run with: ./make_tpu.sh ==="
