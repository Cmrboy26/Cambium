#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DRY_RUN=false
case "${1:-}" in
    --dry-run) DRY_RUN=true ;;
    -h|--help)
        printf 'Usage: bash rosie/submit.sh [--dry-run]\nInteractive training submission; Enter accepts each default.\n'
        exit 0 ;;
    '') ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 1 ;;
esac
if (( $# > 1 )); then
    printf 'Expected at most one option.\n' >&2
    exit 1
fi

prompt() {
    local variable="$1" label="$2" default="$3" pattern="$4" answer
    while true; do
        printf '%s [%s]: ' "$label" "$default" >&2
        if ! IFS= read -r answer; then
            printf '\nInput closed; nothing submitted.\n' >&2
            exit 1
        fi
        answer="${answer:-$default}"
        if [[ "$answer" =~ $pattern ]]; then
            printf -v "$variable" '%s' "$answer"
            return
        fi
        printf 'Invalid value; please try again.\n' >&2
    done
}

printf 'Rosie training submission\n\nGPU presets: h100, v100, teaching (T4)\n'
prompt GPU 'GPU preset' h100 '^(h100|v100|teaching)$'
case "$GPU" in
    h100) PARTITION=dgxh100; GRES=gpu:h100:1; ARCH=90; ENVS_DEFAULT=8192; ROLLOUT_DEFAULT=128; TIME_DEFAULT=1-00:00:00 ;;
    v100) PARTITION=dgx; GRES=gpu:v100:1; ARCH=70; ENVS_DEFAULT=2048; ROLLOUT_DEFAULT=128; TIME_DEFAULT=1-00:00:00 ;;
    teaching) PARTITION=teaching; GRES=gpu:t4:1; ARCH=75; ENVS_DEFAULT=256; ROLLOUT_DEFAULT=64; TIME_DEFAULT=0-02:00:00 ;;
esac

shopt -s nullglob
PLANS=(simulator/setups/curricula/*.json)
if (( ${#PLANS[@]} == 0 )); then
    printf 'No training plans found.\n' >&2
    exit 1
fi
DEFAULT_PLAN=1
printf '\nTraining plans:\n'
for index in "${!PLANS[@]}"; do
    printf '  %d) %s\n' "$((index + 1))" "${PLANS[index]##*/}"
    if [[ "${PLANS[index]##*/}" == shield-axe-curriculum.json ]]; then
        DEFAULT_PLAN=$((index + 1))
    fi
done
while true; do
    prompt PLAN_INDEX 'Plan number' "$DEFAULT_PLAN" '^[1-9][0-9]{0,3}$'
    if (( PLAN_INDEX <= ${#PLANS[@]} )); then break; fi
    printf 'Choose one of the listed plans.\n' >&2
done
PLAN="${PLANS[PLAN_INDEX - 1]}"
PYTHON="$ROOT/.venv-rosie/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(command -v python3 || true)"
fi
if [[ -z "$PYTHON" ]]; then
    printf 'Python 3 is required to read training plan defaults.\n' >&2
    exit 1
fi
if ! DEFAULTS="$("$PYTHON" - "$PLAN" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    plan = json.load(stream)
if plan.get("format") != "minecraft_sim_training_plan" or plan.get("version") != 2:
    raise SystemExit("Expected a version-2 training plan")
stages = plan.get("stages", [])
budgets = [stage.get("advance", {}).get("updates") for stage in stages]
updates = sum(budgets) if budgets and all(type(value) is int and value > 0 for value in budgets) else 1500
learner = {"hidden_dim": 256, "lr": 0.0003, "epochs": 4, "minibatch_size": 1024,
           **plan.get("learner", {})}
print(updates, learner["hidden_dim"], learner["epochs"], learner["minibatch_size"], learner["lr"])
PY
)"; then
    printf 'Could not read training plan defaults; nothing submitted.\n' >&2
    exit 1
fi
read -r UPDATES_DEFAULT HIDDEN_DEFAULT EPOCHS_DEFAULT MINIBATCH_DEFAULT LR_DEFAULT <<< "$DEFAULTS"

printf '\nTraining settings (world counts are starting points, not measured tuning):\n'
NAME="${PLAN##*/}"
prompt JOB_NAME 'Job name' "${NAME%.json}" '^[a-zA-Z0-9][a-zA-Z0-9_.-]*$'
prompt TIME 'Wall-time limit (days-HH:MM:SS or HH:MM:SS)' "$TIME_DEFAULT" '^([0-9]+-)?[0-9]+:[0-5][0-9]:[0-5][0-9]$'
prompt ENVS 'Parallel worlds' "$ENVS_DEFAULT" '^[1-9][0-9]{0,5}$'
prompt UPDATES 'Maximum optimizer updates' "$UPDATES_DEFAULT" '^[1-9][0-9]{0,8}$'
prompt ROLLOUT 'Steps per rollout' "$ROLLOUT_DEFAULT" '^[1-9][0-9]{0,5}$'
prompt HIDDEN 'Hidden width' "$HIDDEN_DEFAULT" '^[1-9][0-9]{0,4}$'
prompt EPOCHS 'PPO epochs' "$EPOCHS_DEFAULT" '^[1-9][0-9]{0,3}$'
prompt MINIBATCH 'Minibatch size' "$MINIBATCH_DEFAULT" '^[1-9][0-9]{0,7}$'
prompt LR 'Learning rate' "$LR_DEFAULT" '^([0-9]*\.[0-9]*[1-9][0-9]*|[1-9][0-9]*(\.[0-9]+)?)([eE][+-]?[0-9]+)?$'
prompt CPUS 'Allocated CPU cores' 8 '^[1-9][0-9]{0,2}$'
prompt MEMORY 'Host RAM (not GPU memory)' 24G '^[1-9][0-9]*[KMGT]$'
while true; do
    prompt THREADS 'Trainer CPU threads' "$(( CPUS < 4 ? CPUS : 4 ))" '^[1-9][0-9]{0,2}$'
    if (( THREADS <= CPUS )); then break; fi
    printf 'Trainer threads must not exceed allocated cores.\n' >&2
done
prompt REPLAY 'Replay interval in completed duels (0 disables)' 50000 '^(0|[1-9][0-9]{0,8})$'
printf '\nCUDA build directory: simulator/build-rosie-cuda-%s\n' "$ARCH"
printf 'Do not overlap jobs using this build directory. Enter their job IDs below to queue after them.\n'
prompt DEPENDENCY 'Wait for job IDs (colon-separated), or none' none '^(none|[1-9][0-9]*(:[1-9][0-9]*)*)$'

EXPORTS="ALL,MCSIM_CUDA_ARCHITECTURES=$ARCH,SIM_PLAN=$PLAN,SIM_ENVS=$ENVS,SIM_UPDATES=$UPDATES,SIM_ROLLOUT_STEPS=$ROLLOUT,SIM_HIDDEN_DIM=$HIDDEN,SIM_EPOCHS=$EPOCHS,SIM_MINIBATCH_SIZE=$MINIBATCH,SIM_LR=$LR,SIM_THREADS=$THREADS,SIM_REPLAY_EVERY=$REPLAY"
COMMAND=(sbatch --parsable "--job-name=$JOB_NAME" "--partition=$PARTITION" "--gres=$GRES"
    "--time=$TIME" "--cpus-per-gpu=$CPUS" "--mem=$MEMORY" "--export=$EXPORTS")
if [[ "$DEPENDENCY" != none ]]; then
    COMMAND+=("--dependency=afterany:$DEPENDENCY")
fi
COMMAND+=(rosie/train.sbatch)

printf '\nWorking directory: %s\nFresh run; existing SIM_* settings and SBATCH_* overrides will not be inherited.\n' "$ROOT"
printf 'Submission command:\n  '
printf '%q ' "${COMMAND[@]}"
printf '\n\n'
if "$DRY_RUN"; then
    printf 'Dry run: nothing submitted.\n'
    exit 0
fi
prompt CONFIRM 'Submit this job? (y/n)' n '^[yYnN]$'
if [[ "$CONFIRM" != [yY] ]]; then
    printf 'Cancelled; nothing submitted.\n'
    exit 0
fi
if ! command -v sbatch >/dev/null 2>&1; then
    printf 'sbatch is unavailable; run this script on Rosie.\n' >&2
    exit 1
fi

while IFS= read -r variable; do
    case "$variable" in SIM_*|SBATCH_*) unset "$variable" ;; esac
done < <(compgen -e)
mkdir -p rosie/logs
JOB_ID="$("${COMMAND[@]}")"
JOB_ID="${JOB_ID%%;*}"
printf 'Submitted job %s\nRun: simulator/training/runs/rosie-%s\nLog: rosie/logs/gpu-sim-%s.out\n' "$JOB_ID" "$JOB_ID" "$JOB_ID"