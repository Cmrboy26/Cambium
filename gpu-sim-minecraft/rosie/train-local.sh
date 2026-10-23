#!/usr/bin/env bash
set -euo pipefail

SIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_ARCHITECTURES="${MCSIM_CUDA_ARCHITECTURES:-86}"
BUILD_DIR="${SIM_BUILD_DIR:-$SIM_ROOT/simulator/build-local-cuda-$CUDA_ARCHITECTURES}"
VENV_DIR="$SIM_ROOT/.venv-rosie"
SIM_ENVS="${SIM_ENVS:-256}"
SIM_UPDATES="${SIM_UPDATES:-1}"
SIM_ROLLOUT_STEPS="${SIM_ROLLOUT_STEPS:-64}"
SIM_HIDDEN_DIM="${SIM_HIDDEN_DIM:-}"
SIM_EPOCHS="${SIM_EPOCHS:-}"
SIM_MINIBATCH_SIZE="${SIM_MINIBATCH_SIZE:-}"
SIM_LR="${SIM_LR:-}"
SIM_GAMMA="${SIM_GAMMA:-}"
SIM_GAE_LAMBDA="${SIM_GAE_LAMBDA:-}"
SIM_CLIP_EPS="${SIM_CLIP_EPS:-}"
SIM_VF_COEF="${SIM_VF_COEF:-}"
SIM_ENT_COEF="${SIM_ENT_COEF:-}"
SIM_MAX_GRAD_NORM="${SIM_MAX_GRAD_NORM:-}"
SIM_THREADS="${SIM_THREADS:-4}"
SIM_PROFILE_EVERY="${SIM_PROFILE_EVERY:-0}"
SIM_PLAN="${SIM_PLAN:-}"
SIM_REPLAY_EVERY="${SIM_REPLAY_EVERY:-0}"
SIM_REPLAY_EVERY_UPDATES="${SIM_REPLAY_EVERY_UPDATES:-0}"
SIM_REPLAY_DIR="${SIM_REPLAY_DIR:-}"
SIM_OUTPUT="${SIM_OUTPUT:-$SIM_ROOT/simulator/training/runs/local-$(date +%Y%m%d-%H%M%S)}"
SIM_RESUME="${SIM_RESUME:-}"
SIM_INITIALIZE="${SIM_INITIALIZE:-}"

cd "$SIM_ROOT"
if [[ ! -d "$SIM_ROOT/simulator/training" ]]; then
    echo "ERROR: Could not locate simulator/training from $SIM_ROOT."
    exit 1
fi
if [[ -z "$SIM_PLAN" && -z "$SIM_RESUME" ]]; then
    SIM_PLAN="simulator/setups/curricula/shield-axe-curriculum.json"
fi
if [[ -n "$SIM_PLAN" && ! -f "$SIM_PLAN" ]]; then
    echo "ERROR: Training plan not found: $SIM_PLAN"
    exit 1
fi
if [[ -n "$SIM_RESUME" && -n "$SIM_INITIALIZE" ]]; then
    echo "ERROR: Choose SIM_RESUME or SIM_INITIALIZE, not both."
    exit 1
fi
for checkpoint in "$SIM_RESUME" "$SIM_INITIALIZE"; do
    if [[ -n "$checkpoint" && ! -f "$checkpoint" ]]; then
        echo "ERROR: Checkpoint not found: $checkpoint"
        exit 1
    fi
done
if [[ "$SIM_REPLAY_EVERY" -lt 0 || ! "$SIM_REPLAY_EVERY_UPDATES" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "ERROR: Replay intervals must be nonnegative integers."
    exit 1
fi
if [[ "$SIM_REPLAY_EVERY" -gt 0 && "$SIM_REPLAY_EVERY_UPDATES" -gt 0 ]]; then
    echo "ERROR: Choose episode-based or update-based replay output, not both."
    exit 1
fi
if [[ -n "$SIM_REPLAY_DIR" && "$SIM_REPLAY_EVERY" -eq 0 && "$SIM_REPLAY_EVERY_UPDATES" -eq 0 ]]; then
    echo "ERROR: Set a positive replay interval when specifying SIM_REPLAY_DIR."
    exit 1
fi

for tool in cmake nvcc flock nvidia-smi; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "ERROR: $tool is required for local CUDA training. In WSL2, install the CUDA toolkit and verify nvidia-smi."
        exit 1
    fi
done
PYTHON="$VENV_DIR/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Training environment is missing. Run bash rosie/setup.sh in this WSL checkout."
    exit 1
fi
if [[ "$SIM_THREADS" == auto ]]; then
    SIM_THREADS="$("$PYTHON" -c 'import os; print(min(4, os.cpu_count() or 1))')"
fi
"$PYTHON" -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda); assert torch.cuda.is_available(), "CUDA is unavailable"'
nvidia-smi

cmake_version="$(cmake --version | awk 'NR == 1 {print $3}')"
if [[ "$(printf '%s\n' 3.24 "$cmake_version" | sort -V | head -n 1)" != 3.24 ]]; then
    echo "ERROR: CMake 3.24+ is required; found $cmake_version."
    exit 1
fi

PRIVATE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/cambium-local-$$.XXXXXX")"
trap 'rm -rf -- "$PRIVATE_DIR"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir -p "$BUILD_DIR"
BUILD_DIR="$(cd "$BUILD_DIR" && pwd -P)"
LIBRARY="$BUILD_DIR/minecraft_sim_training.so"
exec {BUILD_LOCK_FD}>>"$BUILD_DIR/.build.lock"
flock --exclusive "$BUILD_LOCK_FD"
echo "Building native CUDA library for architecture $CUDA_ARCHITECTURES"
cmake -S "$SIM_ROOT/simulator" -B "$BUILD_DIR" \
    -DMCSIM_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
    -DBUILD_TESTING=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --parallel "$SIM_THREADS"
if [[ ! -f "$LIBRARY" ]]; then
    echo "ERROR: Native training library was not produced: $LIBRARY"
    exit 1
fi
cp -- "$LIBRARY" "$PRIVATE_DIR/minecraft_sim_training.so"
LIBRARY="$PRIVATE_DIR/minecraft_sim_training.so"
chmod a-w "$LIBRARY"
flock --unlock "$BUILD_LOCK_FD"
exec {BUILD_LOCK_FD}>&-

TRAIN_ARGS=(
    --device cuda
    --updates "$SIM_UPDATES"
    --threads "$SIM_THREADS"
    --profile-every "$SIM_PROFILE_EVERY"
    --library "$LIBRARY"
    --output "$SIM_OUTPUT"
)
if [[ -n "$SIM_ENVS" ]]; then TRAIN_ARGS+=(--envs "$SIM_ENVS"); fi
if [[ -n "$SIM_ROLLOUT_STEPS" ]]; then TRAIN_ARGS+=(--rollout-steps "$SIM_ROLLOUT_STEPS"); fi
if [[ -n "$SIM_PLAN" ]]; then TRAIN_ARGS+=(--plan "$SIM_PLAN"); fi
if [[ -n "$SIM_RESUME" ]]; then TRAIN_ARGS+=(--resume "$SIM_RESUME"); fi
if [[ -n "$SIM_INITIALIZE" ]]; then TRAIN_ARGS+=(--initialize "$SIM_INITIALIZE"); fi
if [[ -n "$SIM_EPOCHS" ]]; then TRAIN_ARGS+=(--epochs "$SIM_EPOCHS"); fi
if [[ -n "$SIM_MINIBATCH_SIZE" ]]; then TRAIN_ARGS+=(--minibatch-size "$SIM_MINIBATCH_SIZE"); fi
if [[ -n "$SIM_HIDDEN_DIM" ]]; then TRAIN_ARGS+=(--hidden-dim "$SIM_HIDDEN_DIM"); fi
if [[ -n "$SIM_LR" ]]; then TRAIN_ARGS+=(--lr "$SIM_LR"); fi
if [[ -n "$SIM_GAMMA" ]]; then TRAIN_ARGS+=(--gamma "$SIM_GAMMA"); fi
if [[ -n "$SIM_GAE_LAMBDA" ]]; then TRAIN_ARGS+=(--gae-lambda "$SIM_GAE_LAMBDA"); fi
if [[ -n "$SIM_CLIP_EPS" ]]; then TRAIN_ARGS+=(--clip-eps "$SIM_CLIP_EPS"); fi
if [[ -n "$SIM_VF_COEF" ]]; then TRAIN_ARGS+=(--vf-coef "$SIM_VF_COEF"); fi
if [[ -n "$SIM_ENT_COEF" ]]; then TRAIN_ARGS+=(--ent-coef "$SIM_ENT_COEF"); fi
if [[ -n "$SIM_MAX_GRAD_NORM" ]]; then TRAIN_ARGS+=(--max-grad-norm "$SIM_MAX_GRAD_NORM"); fi
if [[ "$SIM_REPLAY_EVERY" -gt 0 ]]; then TRAIN_ARGS+=(--replay-every "$SIM_REPLAY_EVERY"); fi
if [[ "$SIM_REPLAY_EVERY_UPDATES" -gt 0 ]]; then TRAIN_ARGS+=(--replay-every-updates "$SIM_REPLAY_EVERY_UPDATES"); fi
if [[ -n "$SIM_REPLAY_DIR" ]]; then TRAIN_ARGS+=(--replay-dir "$SIM_REPLAY_DIR"); fi

echo "Training locally on $(hostname): envs=$SIM_ENVS updates=$SIM_UPDATES rollout_steps=$SIM_ROLLOUT_STEPS plan=$SIM_PLAN"
"$PYTHON" -B -m simulator.training.train "${TRAIN_ARGS[@]}"
