#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$SIM_ROOT/.venv-rosie"

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 is unavailable on this host."
    exit 1
fi

if [[ -e "$VENV_DIR" || -L "$VENV_DIR" ]]; then
    if [[ ! -f "$VENV_DIR/pyvenv.cfg" || ! -x "$VENV_DIR/bin/python" ]]; then
        echo "ERROR: Existing environment is incomplete or its interpreter is unavailable: $VENV_DIR" >&2
        echo "Move it aside and rerun setup to create a machine-local environment. No files were removed." >&2
        exit 1
    fi
    if ! "$VENV_DIR/bin/python" -c 'import sys; assert sys.prefix != sys.base_prefix, "Interpreter is not a virtual environment"; import pip'; then
        echo "ERROR: Existing environment failed validation: $VENV_DIR" >&2
        echo "Move it aside and rerun setup. No files were removed." >&2
        exit 1
    fi
    echo "Reusing Python environment: $VENV_DIR"
else
    python3 -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install \
    --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1
"$VENV_DIR/bin/python" -m pip install \
    -r "$SIM_ROOT/simulator/training/requirements.txt"

mkdir -p "$SCRIPT_DIR/logs"

echo "Python setup complete: $VENV_DIR"
echo "Python: $VENV_DIR/bin/python"
echo "Rosie training: bash rosie/submit.sh"
echo "Local WSL2 training: bash rosie/submit.sh --local"
echo "Local CUDA builds also require nvcc, CMake 3.24+, and a WSL-visible NVIDIA GPU."
