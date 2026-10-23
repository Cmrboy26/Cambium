#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$SIM_ROOT/.venv-rosie"
PYTHON_BIN="${ROSIE_PYTHON:-}"

if [[ -e "$VENV_DIR" || -L "$VENV_DIR" ]]; then
    if [[ ! -f "$VENV_DIR/pyvenv.cfg" || ! -x "$VENV_DIR/bin/python" ]]; then
        echo "ERROR: Existing environment is incomplete or its interpreter is unavailable: $VENV_DIR" >&2
        echo "Move it aside and rerun setup to create a machine-local environment. No files were removed." >&2
        exit 1
    fi
    if ! "$VENV_DIR/bin/python" -c 'import sys; assert sys.prefix != sys.base_prefix, "Interpreter is not a virtual environment"'; then
        echo "ERROR: Existing environment failed validation: $VENV_DIR" >&2
        echo "Move it aside and rerun setup. No files were removed." >&2
        exit 1
    fi
    VENV_PYTHON_VERSION="$("$VENV_DIR/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "$VENV_PYTHON_VERSION" != "3.12" ]]; then
        echo "ERROR: Existing environment uses Python $VENV_PYTHON_VERSION; Rosie setup requires Python 3.12 for torch==2.5.1/cu124." >&2
        echo "Move it aside, then rerun with Python 3.12. No files were removed." >&2
        exit 1
    fi
    if ! "$VENV_DIR/bin/python" -c 'import pip' >/dev/null 2>&1; then
        echo "pip is missing from the existing environment; attempting recovery with ensurepip."
        if ! "$VENV_DIR/bin/python" -m ensurepip --upgrade || ! "$VENV_DIR/bin/python" -c 'import pip'; then
            echo "ERROR: Existing environment failed validation: $VENV_DIR" >&2
            echo "Could not restore pip. Move it aside and rerun setup. No files were removed." >&2
            exit 1
        fi
    fi
    echo "Reusing Python environment: $VENV_DIR"
else
    if [[ -z "$PYTHON_BIN" ]]; then
        if command -v python3.12 >/dev/null 2>&1; then
            PYTHON_BIN="python3.12"
        elif command -v python3 >/dev/null 2>&1; then
            PYTHON_BIN="python3"
        else
            echo "ERROR: Python 3.12 is unavailable. Install it with venv support or set ROSIE_PYTHON." >&2
            exit 1
        fi
    fi
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        echo "ERROR: ROSIE_PYTHON is unavailable: $PYTHON_BIN" >&2
        exit 1
    fi
    PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "$PYTHON_VERSION" != "3.12" ]]; then
        echo "ERROR: Rosie setup requires Python 3.12 for torch==2.5.1/cu124; $PYTHON_BIN is Python $PYTHON_VERSION." >&2
        echo "Install Python 3.12 with venv support or set ROSIE_PYTHON=/path/to/python3.12." >&2
        exit 1
    fi
    "$PYTHON_BIN" -m venv "$VENV_DIR"
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
