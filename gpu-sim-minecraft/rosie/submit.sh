#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DRY_RUN=false
PROFILE_EVERY=0
SHARED_GPU=false
MAX_PARALLEL=0
while (( $# )); do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --shared-gpu) SHARED_GPU=true; shift ;;
        --max-parallel)
            if [[ ! "${2:-}" =~ ^(0|[1-9][0-9]*)$ ]]; then
                printf 'Expected a nonnegative integer after --max-parallel (0 means all).\n' >&2
                exit 1
            fi
            MAX_PARALLEL="$2"
            shift 2 ;;
        --profile-every)
            if [[ ! "${2:-}" =~ ^(0|[1-9][0-9]*)$ ]]; then
                printf 'Expected a nonnegative integer after --profile-every.\n' >&2
                exit 1
            fi
            PROFILE_EVERY="$2"
            shift 2 ;;
        -h|--help)
            printf 'Usage: bash rosie/submit.sh [--dry-run] [--profile-every N] [--shared-gpu [--max-parallel N]]\nInteractive training submission; Enter accepts each default.\nType multiple at one training/resource setting to submit a single-variable sweep.\n--shared-gpu runs training-setting trials in one allocation, all concurrently by default.\nProfiling defaults to off (0); the interval applies to every submitted job.\n'
            exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 1 ;;
    esac
done
if ! "$SHARED_GPU" && (( MAX_PARALLEL != 0 )); then
    printf -- '--max-parallel requires --shared-gpu.\n' >&2
    exit 1
fi

SWEEP_VARIABLE=""
SWEEP_LABEL=""
SWEEP_VALUES=()

valid_value() {
    local value="$1" pattern="$2" maximum="$3"
    if [[ ! "$value" =~ $pattern ]]; then
        printf 'Invalid value; please try again.\n' >&2
        return 1
    fi
    if [[ -n "$maximum" ]] && (( value > maximum )); then
        printf 'Trainer threads must not exceed allocated cores (%s).\n' "$maximum" >&2
        return 1
    fi
}

prompt() {
    local variable="$1" label="$2" default="$3" pattern="$4" answer
    local allow_multiple="${5:-false}" maximum="${6:-}" duplicate existing
    while true; do
        printf '%s [%s]: ' "$label" "$default" >&2
        if ! IFS= read -r answer; then
            printf '\nInput closed; nothing submitted.\n' >&2
            exit 1
        fi
        if [[ "$answer" == multiple && "$allow_multiple" == true ]]; then
            if [[ -n "$SWEEP_VARIABLE" ]]; then
                printf 'Only one setting can use multiple; %s is already selected. Enter a single value.\n' "$SWEEP_LABEL" >&2
                continue
            fi
            while true; do
                printf '  %s value (Enter to finish): ' "$label" >&2
                if ! IFS= read -r answer; then
                    printf '\nInput closed; nothing submitted.\n' >&2
                    exit 1
                fi
                if [[ -z "$answer" ]]; then
                    if (( ${#SWEEP_VALUES[@]} )); then break; fi
                    printf 'Enter at least one value before finishing.\n' >&2
                    continue
                fi
                if ! valid_value "$answer" "$pattern" "$maximum"; then continue; fi
                duplicate=false
                for existing in "${SWEEP_VALUES[@]}"; do
                    if [[ "$answer" == "$existing" ]]; then duplicate=true; break; fi
                done
                if "$duplicate"; then
                    printf 'That value is already included.\n' >&2
                    continue
                fi
                SWEEP_VALUES+=("$answer")
            done
            SWEEP_VARIABLE="$variable"
            SWEEP_LABEL="$label"
            printf -v "$variable" '%s' "${SWEEP_VALUES[0]}"
            return
        fi
        answer="${answer:-$default}"
        if valid_value "$answer" "$pattern" "$maximum"; then
            printf -v "$variable" '%s' "$answer"
            return
        fi
    done
}

printf 'Rosie training submission\n\nGPU presets: h100, v100, teaching (T4)\n'
prompt GPU 'GPU preset' h100 '^(h100|v100|teaching)$'
case "$GPU" in
    h100) PARTITION=dgxh100; GRES=gpu:h100:1; ARCH=90; ENVS_DEFAULT=8192; ROLLOUT_DEFAULT=128; TIME_DEFAULT=1-00:00:00 ;;
    v100) PARTITION=dgx; GRES=gpu:v100:1; ARCH=70; ENVS_DEFAULT=2048; ROLLOUT_DEFAULT=128; TIME_DEFAULT=1-00:00:00 ;;
    teaching) PARTITION=teaching; GRES=gpu:t4:1; ARCH=75; ENVS_DEFAULT=256; ROLLOUT_DEFAULT=64; TIME_DEFAULT=0-02:00:00 ;;
esac

shopt -s nullglob globstar
PLANS=(simulator/setups/curricula/**/*.json)
if (( ${#PLANS[@]} == 0 )); then
    printf 'No training plans found.\n' >&2
    exit 1
fi
DEFAULT_PLAN=1
printf '\nTraining plans:\n'
for index in "${!PLANS[@]}"; do
    printf '  %d) %s\n' "$((index + 1))" "${PLANS[index]#simulator/setups/curricula/}"
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
    printf 'Python 3 with the training requirements is required to validate training plans.\n' >&2
    exit 1
fi
printf '\nValidating selected training plan: %s\n' "$PLAN"
if ! "$PYTHON" -m simulator.training.validate_plans "$PLAN"; then
    printf 'Selected training plan validation failed; nothing submitted.\n' >&2
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
printf 'Type multiple at one training/resource setting; enter values one per line, then an empty line.\n'
NAME="${PLAN##*/}"
prompt JOB_NAME 'Job/run name' "${NAME%.json}" '^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,119}$'
prompt TIME 'Wall-time limit (days-HH:MM:SS or HH:MM:SS)' "$TIME_DEFAULT" '^([0-9]+-)?[0-9]+:[0-5][0-9]:[0-5][0-9]$' true
prompt ENVS 'Parallel worlds' "$ENVS_DEFAULT" '^[1-9][0-9]{0,5}$' true
prompt UPDATES 'Maximum optimizer updates' "$UPDATES_DEFAULT" '^[1-9][0-9]{0,8}$' true
prompt ROLLOUT 'Steps per rollout' "$ROLLOUT_DEFAULT" '^[1-9][0-9]{0,5}$' true
prompt HIDDEN 'Hidden width' "$HIDDEN_DEFAULT" '^[1-9][0-9]{0,4}$' true
prompt EPOCHS 'PPO epochs' "$EPOCHS_DEFAULT" '^[1-9][0-9]{0,3}$' true
prompt MINIBATCH 'Minibatch size' "$MINIBATCH_DEFAULT" '^[1-9][0-9]{0,7}$' true
prompt LR 'Learning rate' "$LR_DEFAULT" '^([0-9]*\.[0-9]*[1-9][0-9]*|[1-9][0-9]*(\.[0-9]+)?)([eE][+-]?[0-9]+)?$' true
prompt CPUS 'Allocated CPU cores' 8 '^[1-9][0-9]{0,2}$' true
prompt MEMORY 'Host RAM (not GPU memory)' 24G '^[1-9][0-9]*[KMGT]$' true
THREAD_LIMIT="$CPUS"
if "$SHARED_GPU" && [[ -n "$SWEEP_VARIABLE" && "$SWEEP_VARIABLE" != THREADS ]]; then
    ACTIVE_TRIALS="${#SWEEP_VALUES[@]}"
    if (( MAX_PARALLEL > 0 && MAX_PARALLEL < ACTIVE_TRIALS )); then ACTIVE_TRIALS="$MAX_PARALLEL"; fi
    THREAD_LIMIT=$(( CPUS / ACTIVE_TRIALS ))
    if (( THREAD_LIMIT < 1 )); then
        printf 'Allocate at least one CPU core per simultaneous trial, or use --max-parallel.\n' >&2
        exit 1
    fi
fi
if [[ "$SWEEP_VARIABLE" == CPUS ]]; then
    for cores in "${SWEEP_VALUES[@]}"; do
        if (( cores < THREAD_LIMIT )); then THREAD_LIMIT="$cores"; fi
    done
fi
prompt THREADS 'Trainer CPU threads' "$(( THREAD_LIMIT < 4 ? THREAD_LIMIT : 4 ))" '^[1-9][0-9]{0,2}$' true "$THREAD_LIMIT"
prompt REPLAY 'Replay interval in completed duels (0 disables)' 50000 '^(0|[1-9][0-9]{0,8})$' true
printf '\nCUDA build directory: simulator/build-rosie-cuda-%s\n' "$ARCH"
printf 'Builds use a shared lock; each job trains from a private library copy.\n'
printf 'For jobs submitted with the older launcher, enter their IDs to avoid overlapping an unlocked build.\n'
prompt DEPENDENCY 'Wait for job IDs (colon-separated), or none' none '^(none|[1-9][0-9]*(:[1-9][0-9]*)*)$'

if "$SHARED_GPU"; then
    case "$SWEEP_VARIABLE" in
        ENVS) PARAMETER=envs ;; UPDATES) PARAMETER=updates ;; ROLLOUT) PARAMETER=rollout_steps ;;
        HIDDEN) PARAMETER=hidden_dim ;; EPOCHS) PARAMETER=epochs ;; MINIBATCH) PARAMETER=minibatch_size ;;
        LR) PARAMETER=lr ;; THREADS) PARAMETER=threads ;; REPLAY) PARAMETER=replay_every ;;
        *) printf 'Shared-GPU mode requires multiple at one training setting, not allocation CPU/RAM/wall-time.\n' >&2; exit 1 ;;
    esac
    SWEEP_DIRECTORY="$ROOT/simulator/training/sweeps/$JOB_NAME"
    SWEEP_OPTIONS="$("$PYTHON" - "$ENVS" "$UPDATES" "$ROLLOUT" "$HIDDEN" "$EPOCHS" "$MINIBATCH" "$LR" "$THREADS" "$REPLAY" "$PROFILE_EVERY" <<'PY'
import json
import sys
keys = ("envs", "updates", "rollout_steps", "hidden_dim", "epochs", "minibatch_size", "lr", "threads", "replay_every", "profile_every")
options = {key: float(value) if key == "lr" else int(value) for key, value in zip(keys, sys.argv[1:])}
print(json.dumps({"device": "cuda", **options}))
PY
)"
    PREPARE=("$PYTHON" -B -m simulator.training.sweep prepare --plan "$PLAN" --directory "$SWEEP_DIRECTORY"
        --runs "$ROOT/simulator/training/runs" --name "$JOB_NAME" --parameter "$PARAMETER"
        --values "${SWEEP_VALUES[@]}" --options "$SWEEP_OPTIONS" --max-parallel "$MAX_PARALLEL" --allocated-cpus "$CPUS")
    "${PREPARE[@]}" --dry-run
    SHARED_COMMAND=(sbatch --parsable "--job-name=$JOB_NAME" "--partition=$PARTITION" "--gres=$GRES"
        "--time=$TIME" "--cpus-per-gpu=$CPUS" "--mem=$MEMORY"
        "--export=ALL,MCSIM_CUDA_ARCHITECTURES=$ARCH,SIM_SWEEP=$SWEEP_DIRECTORY/sweep.json")
    if [[ "$DEPENDENCY" != none ]]; then SHARED_COMMAND+=("--dependency=afterany:$DEPENDENCY"); fi
    SHARED_COMMAND+=(rosie/train.sbatch)
    printf '\nShared-GPU sweep: %s, %d trials, max parallel=%s (0 means all).\n' "$PARAMETER" "${#SWEEP_VALUES[@]}" "$MAX_PARALLEL"
    printf 'Allocation: %s, CPUs=%s, RAM=%s; GPU memory is shared, not guaranteed per trial.\n' "$GRES" "$CPUS" "$MEMORY"
    printf 'Training threads per trial: %s (or selected values for a thread sweep).\n' "$THREADS"
    printf 'Submission command:\n  '
    printf '%q ' "${SHARED_COMMAND[@]}"
    printf '\n'
    if "$DRY_RUN"; then printf 'Dry run: nothing submitted or written.\n'; exit 0; fi
    prompt CONFIRM 'Submit 1 shared-GPU sweep job? (y/n)' n '^[yYnN]$'
    if [[ "$CONFIRM" != [yY] ]]; then printf 'Cancelled; nothing submitted.\n'; exit 0; fi
    if ! command -v sbatch >/dev/null 2>&1; then printf 'sbatch is unavailable.\n' >&2; exit 1; fi
    "${PREPARE[@]}"
    while IFS= read -r variable; do
        case "$variable" in SIM_*|SBATCH_*) unset "$variable" ;; esac
    done < <(compgen -e)
    mkdir -p rosie/logs
    if ! JOB_ID="$("${SHARED_COMMAND[@]}")"; then
        printf 'Submission failed; frozen sweep retained at %s.\n' "$SWEEP_DIRECTORY" >&2
        exit 1
    fi
    JOB_ID="${JOB_ID%%;*}"
    if [[ ! "$JOB_ID" =~ ^[1-9][0-9]*$ ]]; then printf 'Unexpected sbatch job ID: %s\n' "$JOB_ID" >&2; exit 1; fi
    printf 'Submitted job %s\nSweep: %s/sweep.json\nComparison: %s/comparison.png\n' "$JOB_ID" "$SWEEP_DIRECTORY" "$SWEEP_DIRECTORY"
    exit 0
fi

RUN_NAMES=("$JOB_NAME")
if [[ -n "$SWEEP_VARIABLE" ]]; then
    RUN_NAMES=()
    for value in "${SWEEP_VALUES[@]}"; do
        suffix="${value//:/-}"
        suffix="${suffix//+/}"
        RUN_NAMES+=("$JOB_NAME-$suffix")
    done
fi
declare -A SEEN_RUNS=()
for run_name in "${RUN_NAMES[@]}"; do
    if [[ ! "$run_name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,119}$ ]]; then
        printf 'Run name must be at most 120 filename-safe characters: %s\nNothing submitted.\n' "$run_name" >&2
        exit 1
    fi
    if [[ -n "${SEEN_RUNS[$run_name]:-}" || -e "simulator/training/runs/$run_name" || -L "simulator/training/runs/$run_name" ]]; then
        printf 'Run name already exists or is duplicated: %s\nChoose an unused name; nothing submitted.\n' "$run_name" >&2
        exit 1
    fi
    SEEN_RUNS[$run_name]=1
done

prepare_command() {
    local index="$1" dependency="$2"
    if [[ -n "$SWEEP_VARIABLE" ]]; then
        printf -v "$SWEEP_VARIABLE" '%s' "${SWEEP_VALUES[index]}"
    fi
    RUN_OUTPUT="simulator/training/runs/${RUN_NAMES[index]}"
    EXPORTS="ALL,MCSIM_CUDA_ARCHITECTURES=$ARCH,SIM_PLAN=$PLAN,SIM_OUTPUT=$RUN_OUTPUT,SIM_ENVS=$ENVS,SIM_UPDATES=$UPDATES,SIM_ROLLOUT_STEPS=$ROLLOUT,SIM_HIDDEN_DIM=$HIDDEN,SIM_EPOCHS=$EPOCHS,SIM_MINIBATCH_SIZE=$MINIBATCH,SIM_LR=$LR,SIM_THREADS=$THREADS,SIM_REPLAY_EVERY=$REPLAY,SIM_PROFILE_EVERY=$PROFILE_EVERY"
    COMMAND=(sbatch --parsable "--job-name=${RUN_NAMES[index]}" "--partition=$PARTITION" "--gres=$GRES"
        "--time=$TIME" "--cpus-per-gpu=$CPUS" "--mem=$MEMORY" "--export=$EXPORTS")
    if [[ "$dependency" != none ]]; then
        COMMAND+=("--dependency=afterany:$dependency")
    fi
    COMMAND+=(rosie/train.sbatch)
}

printf '\nWorking directory: %s\nFresh run; existing SIM_* settings and SBATCH_* overrides will not be inherited.\n' "$ROOT"
if [[ -n "$SWEEP_VARIABLE" ]]; then
    printf 'Sweep: %s (%d jobs). Jobs can train concurrently; shared builds are locked.\n' "$SWEEP_LABEL" "${#RUN_NAMES[@]}"
fi
for index in "${!RUN_NAMES[@]}"; do
    prepare_command "$index" "$DEPENDENCY"
    printf 'Submission command (%s):\n  ' "${RUN_NAMES[index]}"
    printf '%q ' "${COMMAND[@]}"
    printf '\n\n'
done
if "$DRY_RUN"; then
    printf 'Dry run: nothing submitted.\n'
    exit 0
fi
prompt CONFIRM "Submit ${#RUN_NAMES[@]} job(s)? (y/n)" n '^[yYnN]$'
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
for index in "${!RUN_NAMES[@]}"; do
    prepare_command "$index" "$DEPENDENCY"
    if ! JOB_ID="$("${COMMAND[@]}")"; then
        printf 'Submission failed for %s; no further jobs submitted. Already submitted jobs were left unchanged.\n' "${RUN_NAMES[index]}" >&2
        exit 1
    fi
    JOB_ID="${JOB_ID%%;*}"
    if [[ ! "$JOB_ID" =~ ^[1-9][0-9]*$ ]]; then
        printf 'Unexpected sbatch job ID: %s; stopping submissions.\n' "$JOB_ID" >&2
        exit 1
    fi
    printf 'Submitted job %s\nRun: %s\nLog: rosie/logs/gpu-sim-%s.out\n' "$JOB_ID" "$RUN_OUTPUT" "$JOB_ID"
done