#!/usr/bin/env bash
#SBATCH --job-name=cambium-sim
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=24G
#SBATCH --time=0-2:00
#SBATCH --output=rosie/logs/gpu-sim-%j.out
#SBATCH --error=rosie/logs/gpu-sim-%j.err

set -euo pipefail

SIM_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
CUDA_ARCHITECTURES="${MCSIM_CUDA_ARCHITECTURES:-75}"
BUILD_DIR="${SIM_BUILD_DIR:-$SIM_ROOT/simulator/build-rosie-cuda-$CUDA_ARCHITECTURES}"
VENV_DIR="$SIM_ROOT/.venv-rosie"
LIBRARY="$BUILD_DIR/minecraft_sim_training.so"

SIM_ENVS="${SIM_ENVS:-256}"
SIM_UPDATES="${SIM_UPDATES:-1}"
SIM_ROLLOUT_STEPS="${SIM_ROLLOUT_STEPS:-16}"
SIM_EPOCHS="${SIM_EPOCHS:-4}"
SIM_MINIBATCH_SIZE="${SIM_MINIBATCH_SIZE:-1024}"
SIM_THREADS="${SIM_THREADS:-4}"
SIM_SEED="${SIM_SEED:-1}"
SIM_SCHEMA="${SIM_SCHEMA:-v1}"
SIM_ENVIRONMENT="${SIM_ENVIRONMENT:-}"
SIM_REPLAY_EVERY="${SIM_REPLAY_EVERY:-0}"
SIM_REPLAY_DIR="${SIM_REPLAY_DIR:-}"
SIM_OUTPUT="${SIM_OUTPUT:-$SIM_ROOT/simulator/training/runs/rosie-${SLURM_JOB_ID}}"

cd "$SIM_ROOT"
module load cuda/current openmpi/current StdEnv

if [[ ! -d "$SIM_ROOT/simulator/training" ]]; then
    echo "ERROR: Submit this job from the gpu-sim-minecraft directory."
    exit 1
fi

if [[ -n "$SIM_ENVIRONMENT" && ! -f "$SIM_ENVIRONMENT" ]]; then
    echo "ERROR: Environment file not found from $SIM_ROOT: $SIM_ENVIRONMENT"
    exit 1
fi
if [[ "$SIM_REPLAY_EVERY" -lt 0 ]]; then
    echo "ERROR: SIM_REPLAY_EVERY must be zero or greater."
    exit 1
fi
if [[ -n "$SIM_REPLAY_DIR" && "$SIM_REPLAY_EVERY" -eq 0 ]]; then
    echo "ERROR: Set SIM_REPLAY_EVERY to a positive value to enable replay output."
    exit 1
fi

PYTHON="$VENV_DIR/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    if ! command -v python3 >/dev/null 2>&1; then
        echo "ERROR: No runnable venv Python or system python3 on $(hostname)."
        exit 1
    fi
    PYTHON="$(command -v python3)"
    PYTHON_VERSION="$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    SITE_PACKAGES="$VENV_DIR/lib/python${PYTHON_VERSION}/site-packages"
    if [[ ! -d "$SITE_PACKAGES" ]]; then
        echo "ERROR: Venv Python is unavailable and compatible packages were not found at $SITE_PACKAGES. Run bash rosie/setup.sh first."
        exit 1
    fi
    export PYTHONPATH="$SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
    echo "Using system Python $PYTHON with shared packages from $SITE_PACKAGES"
fi

for tool in cmake nvcc; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "ERROR: $tool is unavailable on this allocated compute node after loading Rosie modules."
        exit 1
    fi
done

cmake_version="$(cmake --version | awk 'NR == 1 {print $3}')"
if [[ "$(printf '%s\n' 3.24 "$cmake_version" | sort -V | head -n 1)" != 3.24 ]]; then
    echo "ERROR: CMake 3.24+ is required; found $cmake_version."
    exit 1
fi

nvidia-smi
"$PYTHON" -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda); assert torch.cuda.is_available(), "CUDA is unavailable"'

echo "Building native CUDA library for architecture $CUDA_ARCHITECTURES on $(hostname)"
cmake -S "$SIM_ROOT/simulator" -B "$BUILD_DIR" \
    -DMCSIM_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
    -DBUILD_TESTING=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --parallel "${SLURM_CPUS_PER_TASK:-8}"

if [[ ! -f "$LIBRARY" ]]; then
    echo "ERROR: native training library was not produced: $LIBRARY"
    exit 1
fi

echo "Training on $(hostname): envs=$SIM_ENVS updates=$SIM_UPDATES rollout_steps=$SIM_ROLLOUT_STEPS epochs=$SIM_EPOCHS minibatch=$SIM_MINIBATCH_SIZE schema=$SIM_SCHEMA"
TRAIN_ARGS=(
    --device cuda \
    --schema "$SIM_SCHEMA" \
    --envs "$SIM_ENVS" \
    --updates "$SIM_UPDATES" \
    --rollout-steps "$SIM_ROLLOUT_STEPS" \
    --epochs "$SIM_EPOCHS" \
    --minibatch-size "$SIM_MINIBATCH_SIZE" \
    --threads "$SIM_THREADS" \
    --seed "$SIM_SEED" \
    --library "$LIBRARY" \
    --output "$SIM_OUTPUT"
)
if [[ -n "$SIM_ENVIRONMENT" ]]; then
    TRAIN_ARGS+=(--environment "$SIM_ENVIRONMENT")
fi
if [[ "$SIM_REPLAY_EVERY" -gt 0 ]]; then
    TRAIN_ARGS+=(--replay-every "$SIM_REPLAY_EVERY")
    if [[ -n "$SIM_REPLAY_DIR" ]]; then
        TRAIN_ARGS+=(--replay-dir "$SIM_REPLAY_DIR")
    fi
fi

"$PYTHON" -B -m simulator.training.train "${TRAIN_ARGS[@]}"
