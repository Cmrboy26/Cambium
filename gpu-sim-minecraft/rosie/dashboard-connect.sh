#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    printf 'Usage: %s [TRAINING_JOB_ID]\n' "$0"
    printf 'On Rosie, open the dashboard through an SSH tunnel; elsewhere, host the local checkout directly.\n'
    printf 'An explicit job ID uses that Rosie allocation. Set ROSIE_SSH_JUMP and ROSIE_PROJECT_DIR to connect remotely.\n'
    exit 0
fi
if [[ "$#" -gt 1 ]]; then
    printf 'Usage: %s [TRAINING_JOB_ID]\n' "$0" >&2
    exit 2
fi

if command -v python3 >/dev/null 2>&1; then
    LOCAL_PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
    LOCAL_PYTHON="python"
else
    printf 'ERROR: Python 3 is required on this computer.\n' >&2
    exit 1
fi

CURRENT_HOST="$(hostname -f 2>/dev/null || hostname)"
JOB_ID="${1:-${ROSIE_JOB_ID:-}}"
PROJECT_DIR="${ROSIE_PROJECT_DIR:-}"
SSH_JUMP="${ROSIE_SSH_JUMP:-}"
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! "$CURRENT_HOST" =~ \.hpc\.msoe\.edu$ \
    && ! "$CURRENT_HOST" =~ ^dh-mgmt[1-4]$ && -z "$SSH_JUMP" \
        && -z "$JOB_ID" && -z "$PROJECT_DIR" ]]; then
    if [[ -n "${ROSIE_RUNS_DIR:-}" ]]; then
        RUNS_DIR="$ROSIE_RUNS_DIR"
    else
        RUNS_DIR="$SCRIPT_ROOT/simulator/training/runs"
    fi
    PORT="${ROSIE_DASHBOARD_PORT:-0}"
    cd "$SCRIPT_ROOT"
    printf 'Starting the dashboard from this local checkout.\n'
    printf 'Runs directory: %s\nPress Ctrl+C to stop the dashboard.\n' "$RUNS_DIR"
    exec "$LOCAL_PYTHON" -B -m simulator.training.dashboard --runs-dir "$RUNS_DIR" --port "$PORT"
fi

if ! command -v ssh >/dev/null 2>&1; then
    printf 'ERROR: ssh is required to connect to Rosie from this computer.\n' >&2
    exit 1
fi

SSH_USER="${ROSIE_SSH_USER:-${USER:-${USERNAME:-}}}"
if [[ -z "$SSH_USER" ]]; then
    printf 'ERROR: Set ROSIE_SSH_USER to your Rosie SSH account.\n' >&2
    exit 1
fi
if [[ "$SSH_USER" != *@* ]]; then
    SSH_USER="${SSH_USER}@ad.msoe.edu"
fi

if [[ -z "$SSH_JUMP" ]]; then
    if [[ "$CURRENT_HOST" =~ ^dh-mgmt[1-4](\.hpc\.msoe\.edu)?$ ]]; then
        SSH_JUMP="$CURRENT_HOST"
    elif [[ -t 0 ]]; then
        read -r -p 'Rosie management node (enter 1-4 or full host, e.g. dh-mgmt1.hpc.msoe.edu): ' SSH_JUMP
    else
        printf 'ERROR: Run this from a Rosie management node or set ROSIE_SSH_JUMP.\n' >&2
        exit 1
    fi
fi
if [[ -z "$SSH_JUMP" ]]; then
    printf 'ERROR: A Rosie management host is required.\n' >&2
    exit 1
fi
SSH_JUMP="${SSH_JUMP##*@}"
if [[ "$SSH_JUMP" =~ ^[1-4]$ ]]; then
    SSH_JUMP="dh-mgmt${SSH_JUMP}.hpc.msoe.edu"
elif [[ "$SSH_JUMP" =~ ^dh-mgmt[1-4]$ ]]; then
    SSH_JUMP="${SSH_JUMP}.hpc.msoe.edu"
elif [[ ! "$SSH_JUMP" =~ ^dh-mgmt[1-4]\.hpc\.msoe\.edu$ ]]; then
    printf 'ERROR: Enter 1-4 or a host from dh-mgmt1.hpc.msoe.edu through dh-mgmt4.hpc.msoe.edu.\n' >&2
    exit 1
fi
SSH_JUMP_TARGET="$SSH_USER@$SSH_JUMP"
if [[ -z "$PROJECT_DIR" && "$CURRENT_HOST" =~ ^dh-mgmt[1-4](\.hpc\.msoe\.edu)?$ ]]; then
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
LOG_FILE="$(mktemp)"
DASHBOARD_SSH_PID=""
TUNNEL_PID=""

cleanup() {
    if [[ -n "$TUNNEL_PID" ]]; then
        kill "$TUNNEL_PID" 2>/dev/null || true
        wait "$TUNNEL_PID" 2>/dev/null || true
    fi
    if [[ -n "$DASHBOARD_SSH_PID" ]]; then
        kill "$DASHBOARD_SSH_PID" 2>/dev/null || true
        wait "$DASHBOARD_SSH_PID" 2>/dev/null || true
    fi
    rm -f "$LOG_FILE"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

REMOTE_INFO="$(ssh -T -l "$SSH_USER" "$SSH_JUMP" "bash -s -- $(printf '%q' "$JOB_ID") $(printf '%q' "$PROJECT_DIR")" <<'REMOTE'
set -euo pipefail
job_id="${1:-}"
if [[ -z "$job_id" ]]; then
    work_dir="${2:-}"
    if [[ -z "$work_dir" ]]; then
        echo 'ERROR: Set ROSIE_PROJECT_DIR to the remote gpu-sim-minecraft checkout, or run this script on Rosie to use its own checkout.' >&2
        exit 1
    fi
    if [[ "$work_dir" == "~/"* ]]; then
        work_dir="$HOME/${work_dir:2}"
    fi
    if [[ ! -f "$work_dir/simulator/training/dashboard.py" ]]; then
        echo "ERROR: Dashboard not found in Rosie project directory: $work_dir" >&2
        echo 'Set ROSIE_PROJECT_DIR to the remote gpu-sim-minecraft checkout.' >&2
        exit 1
    fi
    work_dir="$(cd "$work_dir" && pwd)"
    printf '%s\t%s\t%s\n' '-' '-' "$work_dir"
    exit 0
fi

job_record="$(squeue --noheader --jobs="$job_id" --states=RUNNING --format='%i|%j|%N')"
if [[ -z "$job_record" ]]; then
    echo "ERROR: Job $job_id is not running." >&2
    exit 1
fi
IFS='|' read -r _ job_name nodes <<< "$job_record"

job_details="$(scontrol show job --oneliner "$job_id")"
job_command="$(sed -n 's/.* Command=\([^ ]*\).*/\1/p' <<< "$job_details")"
if [[ "$job_command" != */rosie/train.sbatch && "$job_command" != rosie/train.sbatch ]]; then
    echo "ERROR: Job $job_id ('$job_name') was not submitted with rosie/train.sbatch." >&2
    exit 1
fi

node="$(scontrol show hostnames "$nodes" | head -n 1)"
work_dir="$(sed -n 's/.* WorkDir=\([^ ]*\).*/\1/p' <<< "$job_details")"
if [[ -z "$node" || -z "$work_dir" ]]; then
    echo "ERROR: Could not determine the node or working directory for job $job_id." >&2
    exit 1
fi
printf '%s\t%s\t%s\n' "$job_id" "$node" "$work_dir"
REMOTE
)"
IFS=$'\t' read -r JOB_ID NODE WORK_DIR <<< "$REMOTE_INFO"

LOCAL_PORT="$("$LOCAL_PYTHON" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
RUNS_DIR="$WORK_DIR/simulator/training/runs"
HOST_OPTIONS=()
if [[ "$JOB_ID" == "-" ]]; then
    NODE="$SSH_JUMP"
    HOST_DESCRIPTION="Rosie management node $NODE"
    printf -v REMOTE_INNER 'cd %q && exec python3 -B -m simulator.training.dashboard --runs-dir %q --port 0' \
        "$WORK_DIR" "$RUNS_DIR"
else
    HOST_OPTIONS=(-J "$SSH_JUMP_TARGET" -o IdentitiesOnly=yes)
    HOST_DESCRIPTION="Rosie job $JOB_ID"
    printf -v REMOTE_INNER 'cd %q && exec srun --jobid=%q --overlap --nodes=1 --ntasks=1 --cpus-per-task=1 python3 -B -m simulator.training.dashboard --runs-dir %q --port 0' \
        "$WORK_DIR" "$JOB_ID" "$RUNS_DIR"
fi
printf -v REMOTE_COMMAND 'bash -c %q' "$REMOTE_INNER"

# A remote PTY terminates the dashboard when its owning SSH connection closes.
ssh -tt "${HOST_OPTIONS[@]}" -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -l "$SSH_USER" "$NODE" \
    "$REMOTE_COMMAND" >"$LOG_FILE" 2>&1 &
DASHBOARD_SSH_PID=$!

REMOTE_PORT=""
for attempt in {1..30}; do
    if ! kill -0 "$DASHBOARD_SSH_PID" 2>/dev/null; then
        cat "$LOG_FILE" >&2
        printf 'ERROR: Dashboard failed to start on %s.\n' "$HOST_DESCRIPTION" >&2
        exit 1
    fi
    REMOTE_PORT="$(sed -n 's/.*Training dashboard: http:\/\/127\.0\.0\.1:\([0-9][0-9]*\).*/\1/p' "$LOG_FILE" | tail -n 1)"
    if [[ -n "$REMOTE_PORT" ]]; then
        break
    fi
    sleep 1
done
if [[ -z "$REMOTE_PORT" ]]; then
    cat "$LOG_FILE" >&2
    printf 'ERROR: Timed out waiting for the dashboard to start on %s.\n' "$HOST_DESCRIPTION" >&2
    exit 1
fi

ssh -N -T "${HOST_OPTIONS[@]}" -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -L "127.0.0.1:$LOCAL_PORT:127.0.0.1:$REMOTE_PORT" -l "$SSH_USER" "$NODE" &
TUNNEL_PID=$!
sleep 1
if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
    printf 'ERROR: Could not establish the SSH tunnel.\n' >&2
    exit 1
fi

URL="http://127.0.0.1:$LOCAL_PORT"
"$LOCAL_PYTHON" - "$URL" <<'PY'
import json
import sys
import time
from urllib.request import urlopen

for attempt in range(10):
    try:
        with urlopen(sys.argv[1] + "/api/runs", timeout=2) as response:
            if not isinstance(json.load(response).get("runs"), list):
                raise ValueError("Dashboard response has no run list")
        break
    except (OSError, ValueError) as error:
        failure = str(error)
        time.sleep(1)
else:
    sys.exit("ERROR: Dashboard tunnel did not respond: " + failure)
PY
printf 'Dashboard on %s is ready: %s\n' "$HOST_DESCRIPTION" "$URL"
printf 'Monitoring all runs in %s (refreshes every 3 seconds).\n' "$RUNS_DIR"
printf 'Press Ctrl+C to stop the dashboard and tunnel.\n'
if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$URL" >/dev/null 2>&1 || true
elif command -v open >/dev/null 2>&1; then
    open "$URL" >/dev/null 2>&1 || true
fi
while kill -0 "$DASHBOARD_SSH_PID" 2>/dev/null && kill -0 "$TUNNEL_PID" 2>/dev/null; do
    sleep 1
done
if ! kill -0 "$DASHBOARD_SSH_PID" 2>/dev/null; then
    cat "$LOG_FILE" >&2
    printf 'ERROR: Dashboard disconnected from %s.\n' "$HOST_DESCRIPTION" >&2
    exit 1
fi
wait "$TUNNEL_PID"