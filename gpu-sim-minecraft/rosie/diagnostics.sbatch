#!/usr/bin/env bash
#SBATCH --job-name=cambium-diagnostics
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=rosie/logs/diagnostics-%j.out
#SBATCH --error=rosie/logs/diagnostics-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
SOURCE_ROOT="${MCSIM_DIAGNOSTIC_SOURCE:?Set the frozen diagnostic source directory}"
MODE="${MCSIM_DIAGNOSTIC_MODE:?Choose evaluation or ppo}"
BUNDLE="${MCSIM_EVALUATION_BUNDLE:?Set the frozen evaluation bundle}"
OUTPUT_ROOT="${MCSIM_DIAGNOSTIC_OUTPUT:?Set a new diagnostic job output directory}"
if [[ "$MODE" != evaluation && "$MODE" != ppo ]]; then
    echo "ERROR: MCSIM_DIAGNOSTIC_MODE must be evaluation or ppo."
    exit 1
fi
if [[ ! -f "$SOURCE_ROOT/simulator/CMakeLists.txt" || ! -f "$BUNDLE/manifest.json" || -e "$OUTPUT_ROOT" ]]; then
    echo "ERROR: Frozen source/bundle is missing, or output already exists."
    exit 1
fi
if [[ "$MODE" == ppo && ! -f "${MCSIM_PPO_CHECKPOINT:-}" ]]; then
    echo "ERROR: PPO diagnostics need a frozen full research checkpoint."
    exit 1
fi

if [[ ":${LOADEDMODULES:-}:" != *":StdEnv:"* ]]; then
    module load StdEnv
fi
PYTHON="$PROJECT_ROOT/.venv-rosie/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(command -v python3)"
    PYTHON_VERSION="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    SITE_PACKAGES="$PROJECT_ROOT/.venv-rosie/lib/python${PYTHON_VERSION}/site-packages"
    if [[ ! -d "$SITE_PACKAGES" ]]; then
        echo "ERROR: Compatible virtual-environment packages are unavailable."
        exit 1
    fi
    export PYTHONPATH="$SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
fi
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$SOURCE_ROOT"
"$PYTHON" -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda); assert torch.cuda.is_available(), "CUDA unavailable"'

PRIVATE_DIR="$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/cambium-diagnostics-${SLURM_JOB_ID}.XXXXXX")"
trap 'rm -rf -- "$PRIVATE_DIR"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir -p "$OUTPUT_ROOT"
BUILD_ARGS=(-DMCSIM_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release)
for dependency in nlohmann_json fast_csv; do
    cache="$PROJECT_ROOT/simulator/build-research/_deps/${dependency}-src"
    if [[ -d "$cache" ]]; then
        upper="${dependency^^}"
        BUILD_ARGS+=("-DFETCHCONTENT_SOURCE_DIR_${upper}=$cache")
    fi
done
cmake -S "$SOURCE_ROOT/simulator" -B "$PRIVATE_DIR/build" "${BUILD_ARGS[@]}"
cmake --build "$PRIVATE_DIR/build" --parallel "${SLURM_CPUS_PER_TASK:-8}"
LIBRARY="$PRIVATE_DIR/build/minecraft_sim_training.so"
if [[ "$MODE" == evaluation ]]; then
    "$PYTHON" -B -m simulator.training.evaluate_checkpoints run "$BUNDLE" \
        --episodes 256 --worlds 256 --device cuda --seed 1729 --library "$LIBRARY"
    cp -- "$BUNDLE/results.json" "$BUNDLE/report.md" "$OUTPUT_ROOT/"
else
    ctest --test-dir "$PRIVATE_DIR/build" --output-on-failure
    "$PYTHON" -B -m simulator.training.benchmark_ppo \
        --checkpoint "$MCSIM_PPO_CHECKPOINT" --device cuda --library "$LIBRARY" \
        --envs 2 --rollout-steps 4 --max-samples 8 --minibatch-sizes 3,5 \
        --threads 2 --repeats 1 --warmup 0 --seed 12345 \
        --output "$OUTPUT_ROOT/real-capture-smoke"
    "$PYTHON" -B -m simulator.training.benchmark_ppo \
        --checkpoint "$MCSIM_PPO_CHECKPOINT" --device cuda --library "$LIBRARY" \
        --envs 8192 --rollout-steps 128 --max-samples 1048576 \
        --minibatch-sizes 1024,4096,8192,16384 --threads 4 \
        --repeats 3 --warmup 1 --seed 12345 --randomize-order --profile-trace \
        --output "$OUTPUT_ROOT/ppo"
fi
