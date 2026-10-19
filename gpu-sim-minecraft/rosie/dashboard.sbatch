#!/usr/bin/env bash
#SBATCH --job-name=mcsim-dashboard
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0-08:00
#SBATCH --output=rosie/logs/dashboard-%j.out
#SBATCH --error=rosie/logs/dashboard-%j.err

set -euo pipefail

SIM_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
VENV_DIR="$SIM_ROOT/.venv-rosie"
RUNS_DIR="${SIM_RUNS_DIR:-$SIM_ROOT/simulator/training/runs}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8767}"
LOCAL_DASHBOARD_PORT="${LOCAL_DASHBOARD_PORT:-8768}"
SSH_USER="${ROSIE_SSH_USER:-$USER}"
SSH_JUMP="${ROSIE_SSH_JUMP:-${SSH_USER}@dh-mgmt3.hpc.msoe.edu}"

cd "$SIM_ROOT"
module load StdEnv

if [[ ! -d "$SIM_ROOT/simulator/training" ]]; then
    echo "ERROR: Submit this job from the gpu-sim-minecraft directory."
    exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "ERROR: Python setup is missing. Run bash rosie/setup.sh first."
    exit 1
fi

mkdir -p "$RUNS_DIR"
NODE_HOST="$(hostname -f)"

echo "Dashboard runs directory: $RUNS_DIR"
echo "Dashboard URL on this node: http://127.0.0.1:$DASHBOARD_PORT"
echo "From your local machine, connected to GlobalVPN, start this tunnel:"
echo "  ssh -N -J $SSH_JUMP -L ${LOCAL_DASHBOARD_PORT}:127.0.0.1:${DASHBOARD_PORT} ${SSH_USER}@${NODE_HOST}"
echo "Open http://127.0.0.1:$LOCAL_DASHBOARD_PORT on your local machine."
echo "Stop the dashboard with: scancel $SLURM_JOB_ID"

exec "$VENV_DIR/bin/python" -B -m simulator.training.dashboard \
    --runs-dir "$RUNS_DIR" \
    --port "$DASHBOARD_PORT"