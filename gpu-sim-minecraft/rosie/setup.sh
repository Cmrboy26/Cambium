#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$SIM_ROOT/simulator/build-rosie-cuda"
VENV_DIR="$SIM_ROOT/.venv-rosie"
CUDA_ARCHITECTURES="${MCSIM_CUDA_ARCHITECTURES:-75}"

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 is unavailable on this Rosie node."
    exit 1
fi

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install \
    --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1
"$VENV_DIR/bin/python" -m pip install --no-deps \
    -r "$SIM_ROOT/simulator/training/requirements.txt"

mkdir -p "$SCRIPT_DIR/logs"

echo "Python setup complete: $VENV_DIR"
echo "The native CUDA build runs inside the allocated GPU job."
echo "Python: $VENV_DIR/bin/python"
echo "CUDA architecture: $CUDA_ARCHITECTURES"
echo "Submit from $SIM_ROOT with:"
echo "  sbatch rosie/train.sbatch"
