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

SIM_ENVS="${SIM_ENVS:-}"
SIM_UPDATES="${SIM_UPDATES:-1}"
SIM_ROLLOUT_STEPS="${SIM_ROLLOUT_STEPS:-}"
SIM_EPOCHS="${SIM_EPOCHS:-}"
SIM_MINIBATCH_SIZE="${SIM_MINIBATCH_SIZE:-}"
SIM_HIDDEN_DIM="${SIM_HIDDEN_DIM:-}"
SIM_LR="${SIM_LR:-}"
SIM_THREADS="${SIM_THREADS:-4}"
SIM_PROFILE_EVERY="${SIM_PROFILE_EVERY:-0}"
SIM_SAVE_EVERY="${SIM_SAVE_EVERY:-10}"
SIM_PLAN="${SIM_PLAN:-}"
SIM_SWEEP="${SIM_SWEEP:-}"
SIM_RESUME="${SIM_RESUME:-}"
SIM_INITIALIZE="${SIM_INITIALIZE:-}"
SIM_START_STAGE="${SIM_START_STAGE:-}"
SIM_REPLAY_EVERY="${SIM_REPLAY_EVERY:-0}"
SIM_REPLAY_EVERY_UPDATES="${SIM_REPLAY_EVERY_UPDATES:-0}"
SIM_REPLAY_DIR="${SIM_REPLAY_DIR:-}"
SIM_OUTPUT="${SIM_OUTPUT:-$SIM_ROOT/simulator/training/runs/rosie-${SLURM_JOB_ID}}"

cd "$SIM_ROOT"
if [[ ":${LOADEDMODULES:-}:" != *":StdEnv:"* ]]; then
    module load StdEnv
fi

if [[ ! -d "$SIM_ROOT/simulator/training" ]]; then
    echo "ERROR: Submit this job from the gpu-sim-minecraft directory."
    exit 1
fi

if [[ -n "${SIM_ENVIRONMENT:-}" || -n "${SIM_SCHEMA:-}" || -n "${SIM_SEED:-}" ]]; then
    echo "ERROR: SIM_ENVIRONMENT/SIM_SCHEMA/SIM_SEED were replaced by the environment, policy and seed in SIM_PLAN."
    exit 1
fi
if [[ -n "$SIM_SWEEP" && ( ! -f "$SIM_SWEEP" || -n "$SIM_PLAN" || -n "$SIM_RESUME" || -n "$SIM_INITIALIZE" || -n "$SIM_START_STAGE" ) ]]; then
    echo "ERROR: SIM_SWEEP needs an existing manifest and cannot be combined with plan/resume/initialize/start-stage."
    exit 1
fi
if [[ -z "$SIM_PLAN" && -z "$SIM_RESUME" && -z "$SIM_SWEEP" ]]; then
    SIM_PLAN="simulator/setups/curricula/shield-axe-curriculum.json"
fi
if [[ -z "$SIM_RESUME" ]]; then
    SIM_ENVS="${SIM_ENVS:-256}"
    SIM_ROLLOUT_STEPS="${SIM_ROLLOUT_STEPS:-16}"
fi
if [[ -n "$SIM_PLAN" && ! -f "$SIM_PLAN" ]]; then
    echo "ERROR: Training plan not found from $SIM_ROOT: $SIM_PLAN"
    exit 1
fi
if [[ -n "$SIM_RESUME" && -n "$SIM_INITIALIZE" ]]; then
    echo "ERROR: Choose SIM_RESUME or SIM_INITIALIZE, not both."
    exit 1
fi
if [[ -n "$SIM_START_STAGE" && -n "$SIM_RESUME" ]]; then
    echo "ERROR: SIM_START_STAGE is for new runs and cannot be combined with SIM_RESUME."
    exit 1
fi
for checkpoint in "$SIM_RESUME" "$SIM_INITIALIZE"; do
    if [[ -n "$checkpoint" && ! -f "$checkpoint" ]]; then
        echo "ERROR: Checkpoint not found from $SIM_ROOT: $checkpoint"
        exit 1
    fi
done
if [[ ! "$SIM_PROFILE_EVERY" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "ERROR: SIM_PROFILE_EVERY must be a nonnegative integer."
    exit 1
fi
if [[ ! "$SIM_SAVE_EVERY" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: SIM_SAVE_EVERY must be a positive integer."
    exit 1
fi
if [[ "$SIM_REPLAY_EVERY" -lt 0 ]]; then
    echo "ERROR: SIM_REPLAY_EVERY must be zero or greater."
    exit 1
fi
if [[ ! "$SIM_REPLAY_EVERY_UPDATES" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "ERROR: SIM_REPLAY_EVERY_UPDATES must be a nonnegative integer."
    exit 1
fi
if [[ "$SIM_REPLAY_EVERY" -gt 0 && "$SIM_REPLAY_EVERY_UPDATES" -gt 0 ]]; then
    echo "ERROR: Choose episode-based or update-based replays, not both."
    exit 1
fi
if [[ -n "$SIM_REPLAY_DIR" && "$SIM_REPLAY_EVERY" -eq 0 && "$SIM_REPLAY_EVERY_UPDATES" -eq 0 ]]; then
    echo "ERROR: Set SIM_REPLAY_EVERY or SIM_REPLAY_EVERY_UPDATES to enable replay output."
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

for tool in cmake nvcc flock; do
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

PRIVATE_DIR="$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/cambium-${SLURM_JOB_ID}.XXXXXX")"
trap 'rm -rf -- "$PRIVATE_DIR"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$BUILD_DIR"
BUILD_DIR="$(cd "$BUILD_DIR" && pwd -P)"
LIBRARY="$BUILD_DIR/minecraft_sim_training.so"
exec {BUILD_LOCK_FD}>>"$BUILD_DIR/.build.lock"
echo "Waiting for shared CUDA build lock: $BUILD_DIR/.build.lock"
flock --exclusive "$BUILD_LOCK_FD"

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

cp -- "$LIBRARY" "$PRIVATE_DIR/minecraft_sim_training.so"
LIBRARY="$PRIVATE_DIR/minecraft_sim_training.so"
chmod a-w "$LIBRARY"
flock --unlock "$BUILD_LOCK_FD"
exec {BUILD_LOCK_FD}>&-
echo "Released build lock; training uses private library: $LIBRARY"

if [[ -n "$SIM_SWEEP" ]]; then
    echo "Running shared-GPU sweep: $SIM_SWEEP"
    "$PYTHON" -B -m simulator.training.sweep run "$SIM_SWEEP" --library "$LIBRARY" \
        --allocated-cpus "${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_PER_GPU:-8}}" &
    SWEEP_PID=$!
    stop_sweep() {
        trap '' INT TERM
        kill "-$1" "$SWEEP_PID" 2>/dev/null || true
        wait "$SWEEP_PID" || true
        exit "$2"
    }
    trap 'stop_sweep INT 130' INT
    trap 'stop_sweep TERM 143' TERM
    SWEEP_RESULT=0
    wait "$SWEEP_PID" || SWEEP_RESULT=$?
    trap 'exit 130' INT
    trap 'exit 143' TERM
    if ! "$PYTHON" -B -m simulator.training.plot_sweep "$SIM_SWEEP"; then
        echo "ERROR: Sweep plot failed; regenerate it with simulator.training.plot_sweep."
        if (( SWEEP_RESULT == 0 )); then SWEEP_RESULT=1; fi
    fi
    exit "$SWEEP_RESULT"
fi

echo "Training on $(hostname): envs=$SIM_ENVS updates=$SIM_UPDATES rollout_steps=$SIM_ROLLOUT_STEPS plan=$SIM_PLAN resume=$SIM_RESUME"
TRAIN_ARGS=(
    --device cuda \
    --updates "$SIM_UPDATES" \
    --threads "$SIM_THREADS" \
    --profile-every "$SIM_PROFILE_EVERY" \
    --save-every "$SIM_SAVE_EVERY" \
    --library "$LIBRARY" \
    --output "$SIM_OUTPUT"
)
if [[ -n "$SIM_ENVS" ]]; then TRAIN_ARGS+=(--envs "$SIM_ENVS"); fi
if [[ -n "$SIM_ROLLOUT_STEPS" ]]; then TRAIN_ARGS+=(--rollout-steps "$SIM_ROLLOUT_STEPS"); fi
if [[ -n "$SIM_PLAN" ]]; then
    TRAIN_ARGS+=(--plan "$SIM_PLAN")
fi
if [[ -n "$SIM_RESUME" ]]; then TRAIN_ARGS+=(--resume "$SIM_RESUME"); fi
if [[ -n "$SIM_INITIALIZE" ]]; then TRAIN_ARGS+=(--initialize "$SIM_INITIALIZE"); fi
if [[ -n "$SIM_START_STAGE" ]]; then TRAIN_ARGS+=(--start-stage "$SIM_START_STAGE"); fi
if [[ -n "$SIM_EPOCHS" ]]; then TRAIN_ARGS+=(--epochs "$SIM_EPOCHS"); fi
if [[ -n "$SIM_MINIBATCH_SIZE" ]]; then TRAIN_ARGS+=(--minibatch-size "$SIM_MINIBATCH_SIZE"); fi
if [[ -n "$SIM_HIDDEN_DIM" ]]; then TRAIN_ARGS+=(--hidden-dim "$SIM_HIDDEN_DIM"); fi
if [[ -n "$SIM_LR" ]]; then TRAIN_ARGS+=(--lr "$SIM_LR"); fi
for setting in GAMMA GAE_LAMBDA CLIP_EPS VF_COEF ENT_COEF MAX_GRAD_NORM; do
    variable="SIM_$setting"
    if [[ -n "${!variable:-}" ]]; then
        flag="${setting,,}"
        TRAIN_ARGS+=("--${flag//_/-}" "${!variable}")
    fi
done
if [[ "$SIM_REPLAY_EVERY" -gt 0 ]]; then
    TRAIN_ARGS+=(--replay-every "$SIM_REPLAY_EVERY")
fi
if [[ "$SIM_REPLAY_EVERY_UPDATES" -gt 0 ]]; then
    TRAIN_ARGS+=(--replay-every-updates "$SIM_REPLAY_EVERY_UPDATES")
fi
if [[ -n "$SIM_REPLAY_DIR" ]]; then
    TRAIN_ARGS+=(--replay-dir "$SIM_REPLAY_DIR")
fi

"$PYTHON" -B -m simulator.training.train "${TRAIN_ARGS[@]}"
