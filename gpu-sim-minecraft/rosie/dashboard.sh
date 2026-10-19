#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    printf 'Usage: %s [TRAINING_JOB_ID]\n' "$0"
    printf 'Connect to a running Rosie training job and serve its dashboard locally.\n'
    exit 0
fi
if [[ "$#" -gt 1 ]]; then
    printf 'Usage: %s [TRAINING_JOB_ID]\n' "$0" >&2
    exit 2
fi

if ! command -v ssh >/dev/null 2>&1; then
    printf 'ERROR: ssh is required on this computer.\n' >&2
    exit 1
fi
if command -v python3 >/dev/null 2>&1; then
    LOCAL_PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
    LOCAL_PYTHON="python"
else
    printf 'ERROR: Python 3 is required on this computer.\n' >&2
    exit 1
fi

SSH_USER="${ROSIE_SSH_USER:-${USER:-${USERNAME:-}}}"
if [[ -z "$SSH_USER" ]]; then
    printf 'ERROR: Set ROSIE_SSH_USER to your Rosie SSH account.\n' >&2
    exit 1
fi
SSH_JUMP="${ROSIE_SSH_JUMP:-${SSH_USER}@dh-mgmt3.hpc.msoe.edu}"
JOB_ID="${1:-${ROSIE_JOB_ID:-}}"
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
trap cleanup EXIT INT TERM

REMOTE_INFO="$(ssh -T "$SSH_JUMP" "bash -s -- $(printf '%q' "$JOB_ID")" <<'REMOTE'
set -euo pipefail
job_id="${1:-}"
if [[ -z "$job_id" ]]; then
    mapfile -t matches < <(squeue --noheader --user="$USER" --states=RUNNING --format='%i|%j|%N' |
        awk -F'|' '$2 == "cambium-sim" { print $1 "|" $3 }')
    if [[ "${#matches[@]}" -eq 0 ]]; then
        echo 'ERROR: No running cambium-sim job found. Start training first, or pass its job ID.' >&2
        exit 1
    elif [[ "${#matches[@]}" -gt 1 ]]; then
        echo 'ERROR: Multiple running cambium-sim jobs found. Pass the desired job ID:' >&2
        printf '  %s\n' "${matches[@]}" >&2
        exit 1
    fi
    IFS='|' read -r job_id nodes <<< "${matches[0]}"
else
    job_record="$(squeue --noheader --jobs="$job_id" --format='%i|%j|%N')"
    if [[ -z "$job_record" ]]; then
        echo "ERROR: Job $job_id is not running." >&2
        exit 1
    fi
    IFS='|' read -r _ job_name nodes <<< "$job_record"
    if [[ "$job_name" != "cambium-sim" ]]; then
        echo "ERROR: Job $job_id is '$job_name', not a cambium-sim training job." >&2
        exit 1
    fi
fi

node="$(scontrol show hostnames "$nodes" | head -n 1)"
job_details="$(scontrol show job --oneliner "$job_id")"
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
PYTHON_PATH="$WORK_DIR/.venv-rosie/bin/python"
RUNS_DIR="$WORK_DIR/simulator/training/runs"
printf -v REMOTE_INNER 'cd %q && module load StdEnv && exec srun --jobid=%q --overlap --nodes=1 --ntasks=1 --cpus-per-task=1 %q -B -m simulator.training.dashboard --runs-dir %q --port 0' \
    "$WORK_DIR" "$JOB_ID" "$PYTHON_PATH" "$RUNS_DIR"
printf -v REMOTE_COMMAND 'bash -lc %q' "$REMOTE_INNER"

ssh -T -J "$SSH_JUMP" -o BatchMode=yes "$SSH_USER@$NODE" "$REMOTE_COMMAND" >"$LOG_FILE" 2>&1 &
DASHBOARD_SSH_PID=$!

REMOTE_PORT=""
for attempt in {1..30}; do
    if ! kill -0 "$DASHBOARD_SSH_PID" 2>/dev/null; then
        cat "$LOG_FILE" >&2
        printf 'ERROR: Dashboard failed to start on job %s.\n' "$JOB_ID" >&2
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
    printf 'ERROR: Timed out waiting for the dashboard to start on job %s.\n' "$JOB_ID" >&2
    exit 1
fi

ssh -N -T -J "$SSH_JUMP" -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 -L "127.0.0.1:$LOCAL_PORT:127.0.0.1:$REMOTE_PORT" \
    "$SSH_USER@$NODE" &
TUNNEL_PID=$!
sleep 1
if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
    printf 'ERROR: Could not establish the SSH tunnel.\n' >&2
    exit 1
fi

URL="http://127.0.0.1:$LOCAL_PORT"
printf 'Dashboard for Rosie job %s is ready: %s\n' "$JOB_ID" "$URL"
printf 'Press Ctrl+C to stop the dashboard and tunnel.\n'
if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$URL" >/dev/null 2>&1 || true
elif command -v open >/dev/null 2>&1; then
    open "$URL" >/dev/null 2>&1 || true
fi
wait "$TUNNEL_PID"