#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DRY_RUN=false
LOCAL=false
PROFILE_EVERY=0
SHARED_GPU=false
MAX_PARALLEL=0
LINE_MODE=false
INITIALIZE=""
RESUME=""
START_STAGE=""
SAVE_EVERY=10
LEARNER_OVERRIDES=()
while (( $# )); do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --local) LOCAL=true; shift ;;
        --line-mode) LINE_MODE=true; shift ;;
        --shared-gpu) SHARED_GPU=true; shift ;;
        --initialize|--resume|--start-stage|--save-every)
            if (( $# < 2 )); then printf 'Expected a value after %s.\n' "$1" >&2; exit 1; fi
            case "$1" in
                --initialize) INITIALIZE="$2" ;;
                --resume) RESUME="$2" ;;
                --start-stage) START_STAGE="$2" ;;
                --save-every)
                    if [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
                        printf 'Expected a positive integer after --save-every.\n' >&2
                        exit 1
                    fi
                    SAVE_EVERY="$2" ;;
            esac
            shift 2 ;;
        --gamma|--gae-lambda|--clip-eps|--vf-coef|--ent-coef|--max-grad-norm)
            if (( $# < 2 )); then printf 'Expected a value after %s.\n' "$1" >&2; exit 1; fi
            variable="${1#--}"
            variable="${variable^^}"
            LEARNER_OVERRIDES+=(--override "${variable//-/_}" "$2")
            shift 2 ;;
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
            printf 'Usage: bash rosie/submit.sh [--local] [--dry-run] [--line-mode] [--profile-every N] [--shared-gpu [--max-parallel N]] [--resume CHECKPOINT] [--initialize CHECKPOINT] [--start-stage NAME] [--save-every N]\n--local builds and trains on this WSL/Linux host; use a local checkout with CUDA installed.\nInteractive settings: Up/Down to navigate, type to edit, Enter on Submit.\nUse comma-separated values in one field for a single-variable sweep.\n--line-mode retains sequential prompts (also used for piped input).\n--initialize loads compatible weights into a new run; --resume restores optimizer and curriculum progress.\n--start-stage skips earlier stages in a new run; it cannot be combined with --resume.\n--save-every sets the rolling checkpoint cadence in optimizer updates (default 10).\nPPO overrides: --gamma, --gae-lambda, --clip-eps, --vf-coef, --ent-coef, --max-grad-norm VALUE.\nA comma-separated override also works in line mode.\n--shared-gpu runs training-setting trials in one allocation, all concurrently by default.\nVerbose timing profiling defaults to off (0); its interval applies to every submitted job.\n'
            exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 1 ;;
    esac
done
if ! "$SHARED_GPU" && (( MAX_PARALLEL != 0 )); then
    printf -- '--max-parallel requires --shared-gpu.\n' >&2
    exit 1
fi
if "$LOCAL" && "$SHARED_GPU"; then
    printf -- '--local cannot be combined with --shared-gpu. Local sweeps run sequentially.\n' >&2
    exit 1
fi
DEFAULT_START_STAGE="$START_STAGE"
if "$SHARED_GPU" && { [[ -n "$RESUME" || -n "$INITIALIZE" || -n "$START_STAGE" ]] || (( SAVE_EVERY != 10 )); }; then
    printf 'Checkpoint controls, stage selection, and custom save cadence are not supported with --shared-gpu.\n' >&2
    exit 1
fi

SWEEP_VARIABLE=""
SWEEP_LABEL=""
SWEEP_VALUES=()

read_form_values() {
    local key value
    while IFS=$'\t' read -r key value; do
        case "$key" in
            SWEEP_VARIABLE)
                if [[ -n "$SWEEP_VARIABLE" ]]; then printf 'Only one setting can be swept.\n' >&2; exit 1; fi
                SWEEP_VARIABLE="$value" ;;
            SWEEP_LABEL) SWEEP_LABEL="$value" ;;
            SWEEP_VALUE) SWEEP_VALUES+=("$value") ;;
            JOB_NAME|TIME|ENVS|UPDATES|ROLLOUT|HIDDEN|EPOCHS|MINIBATCH|LR|CPUS|MEMORY|THREADS|REPLAY|PROFILE_EVERY|SAVE_EVERY|RESUME|INITIALIZE|START_STAGE|DEPENDENCY|GAMMA|GAE_LAMBDA|CLIP_EPS|VF_COEF|ENT_COEF|MAX_GRAD_NORM)
                printf -v "$key" '%s' "$value" ;;
            *) printf 'Unexpected form setting; nothing submitted.\n' >&2; exit 1 ;;
        esac
    done <<< "$1"
}

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

if "$LOCAL"; then
    ARCH="${MCSIM_CUDA_ARCHITECTURES:-86}"
    if [[ ! "$ARCH" =~ ^[0-9]+$ ]]; then
        printf 'MCSIM_CUDA_ARCHITECTURES must be a numeric CUDA architecture.\n' >&2
        exit 1
    fi
    ENVS_DEFAULT="${MCSIM_LOCAL_ENVS:-256}"
    ROLLOUT_DEFAULT="${MCSIM_LOCAL_ROLLOUT_STEPS:-64}"
    TIME_DEFAULT=00:00:00
    printf 'Local CUDA training (architecture %s; default 256 worlds for an 8 GB GPU).\n' "$ARCH"
else
    printf 'Rosie training submission\n\nGPU presets: h100, v100, teaching (T4)\n'
    prompt GPU 'GPU preset' h100 '^(h100|v100|teaching)$'
    case "$GPU" in
        h100) PARTITION=dgxh100; GRES=gpu:h100:1; ARCH=90; ENVS_DEFAULT=8192; ROLLOUT_DEFAULT=128; TIME_DEFAULT=1-00:00:00 ;;
        v100) PARTITION=dgx; GRES=gpu:v100:1; ARCH=70; ENVS_DEFAULT=2048; ROLLOUT_DEFAULT=128; TIME_DEFAULT=1-00:00:00 ;;
        teaching) PARTITION=teaching; GRES=gpu:t4:1; ARCH=75; ENVS_DEFAULT=256; ROLLOUT_DEFAULT=64; TIME_DEFAULT=0-02:00:00 ;;
    esac
fi

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
if "$LOCAL" && ! "$DRY_RUN"; then
    for tool in cmake nvcc flock nvidia-smi; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            printf 'ERROR: %s is required for local CUDA training. Set up CUDA inside WSL2 first.\n' "$tool" >&2
            exit 1
        fi
    done
    if ! "$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'; then
        printf 'ERROR: The selected Python environment cannot access CUDA. Run bash rosie/setup.sh and verify WSL GPU support.\n' >&2
        exit 1
    fi
    nvidia-smi --query-gpu=name --format=csv,noheader
fi
if ! DEFAULTS="$("$PYTHON" - "$PLAN" "$START_STAGE" <<'PY'
import json
import sys
from simulator.training.compiler import LEARNER_DEFAULTS

with open(sys.argv[1], encoding="utf-8") as stream:
    plan = json.load(stream)
if plan.get("format") != "minecraft_sim_training_plan" or plan.get("version") != 2:
    raise SystemExit("Expected a version-2 training plan")
stages = plan.get("stages", [])
start_stage = sys.argv[2]
start_index = next((index for index, stage in enumerate(stages) if stage.get("name") == start_stage), 0)
budgets = [stage.get("advance", {}).get("updates") for stage in stages[start_index:]]
updates = sum(budgets) if budgets and all(type(value) is int and value > 0 for value in budgets) else 1500
learner = {**LEARNER_DEFAULTS, **plan.get("learner", {})}
keys = ("hidden_dim", "epochs", "minibatch_size", "lr", "gamma", "gae_lambda", "clip_eps", "vf_coef", "ent_coef", "max_grad_norm")
print(updates, *(learner[key] for key in keys))
PY
)"; then
    printf 'Could not read training plan defaults; nothing submitted.\n' >&2
    exit 1
fi
read -r UPDATES_DEFAULT HIDDEN_DEFAULT EPOCHS_DEFAULT MINIBATCH_DEFAULT LR_DEFAULT GAMMA GAE_LAMBDA CLIP_EPS VF_COEF ENT_COEF MAX_GRAD_NORM <<< "$DEFAULTS"

FORM_USED=false
FORM_ARGS=(--plan "$PLAN" --envs-default "$ENVS_DEFAULT" --rollout-default "$ROLLOUT_DEFAULT"
    --time-default "$TIME_DEFAULT" --max-parallel "$MAX_PARALLEL"
    --profile-every-default "$PROFILE_EVERY" --save-every-default "$SAVE_EVERY"
    --resume-default "$RESUME" --initialize-default "$INITIALIZE" --start-stage-default "$START_STAGE"
    "${LEARNER_OVERRIDES[@]}")
if "$LOCAL"; then FORM_ARGS+=(--local); fi
if [[ -t 0 && -t 2 ]] && ! "$LINE_MODE"; then
    if "$SHARED_GPU"; then FORM_ARGS+=(--shared-gpu); fi
    if ! FORM_DATA="$("$PYTHON" -B -m simulator.training.submit_form "${FORM_ARGS[@]}")"; then exit 1; fi
    read_form_values "$FORM_DATA"
    FORM_USED=true
elif (( ${#LEARNER_OVERRIDES[@]} )); then
    if ! FORM_DATA="$("$PYTHON" -B -m simulator.training.submit_form "${FORM_ARGS[@]}" --learner-only)"; then exit 1; fi
    read_form_values "$FORM_DATA"
fi

if [[ -n "$RESUME" && -n "$INITIALIZE" ]]; then
    printf 'Choose --resume or --initialize, not both; nothing submitted.\n' >&2
    exit 1
fi
if [[ -n "$RESUME" && -n "$START_STAGE" ]]; then
    printf 'A full resume restores its checkpoint stage; --start-stage is only for a new run.\nNothing submitted.\n' >&2
    exit 1
fi
if [[ -n "$RESUME" && -n "$SWEEP_VARIABLE" ]]; then
    printf 'A full resume cannot be combined with a sweep; nothing submitted.\n' >&2
    exit 1
fi
for checkpoint in "$RESUME" "$INITIALIZE"; do
    if [[ "$checkpoint" == *,* ]]; then
        printf 'Checkpoint paths cannot contain commas because Slurm uses commas in --export.\nNothing submitted.\n' >&2
        exit 1
    fi
    if [[ -n "$checkpoint" && ! -f "$checkpoint" ]]; then
        if [[ -n "$RESUME" && "$checkpoint" == "$RESUME" ]]; then
            printf 'Resume checkpoint not found from %s: %s\nNothing submitted.\n' "$ROOT" "$checkpoint" >&2
        else
            printf 'Initialization checkpoint not found from %s: %s\nNothing submitted.\n' "$ROOT" "$checkpoint" >&2
        fi
        exit 1
    fi
done
if [[ -n "$START_STAGE" ]] && ! "$PYTHON" - "$PLAN" "$START_STAGE" <<'PY'
import sys
from simulator.training.plans import load_plan

plan = load_plan(sys.argv[1])
stages = [stage["name"] for stage in plan["stages"]]
if sys.argv[2] not in stages:
    raise SystemExit(f"Unknown start stage {sys.argv[2]!r}; choose one of: {', '.join(stages)}")
PY
then
    printf 'Selected start stage is invalid; nothing submitted.\n' >&2
    exit 1
fi
if [[ "$START_STAGE" != "$DEFAULT_START_STAGE" && "$UPDATES" == "$UPDATES_DEFAULT" ]]; then
    if ! UPDATED_UPDATES_DEFAULT="$("$PYTHON" - "$PLAN" "$START_STAGE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    stages = json.load(stream)["stages"]
start_stage = sys.argv[2]
start_index = next((index for index, stage in enumerate(stages) if stage["name"] == start_stage), 0)
budgets = [stage.get("advance", {}).get("updates") for stage in stages[start_index:]]
print(sum(budgets) if budgets and all(type(value) is int and value > 0 for value in budgets) else 1500)
PY
)"; then
        printf 'Could not recalculate the update cap for the selected start stage; nothing submitted.\n' >&2
        exit 1
    fi
    UPDATES="$UPDATED_UPDATES_DEFAULT"
fi
if "$SHARED_GPU" && { [[ -n "$RESUME" || -n "$INITIALIZE" || -n "$START_STAGE" ]] || (( SAVE_EVERY != 10 )); }; then
    printf 'Checkpoint controls, stage selection, and custom save cadence are not supported with --shared-gpu.\n' >&2
    exit 1
fi

if ! "$FORM_USED"; then
if "$LOCAL"; then
printf '\nLocal training settings (world counts are starting points, not measured tuning):\n'
else
printf '\nTraining settings (world counts are starting points, not measured tuning):\n'
fi
printf 'Type multiple at one training/resource setting; enter values one per line, then an empty line.\n'
NAME="${PLAN##*/}"
prompt JOB_NAME 'Job/run name' "${NAME%.json}" '^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,119}$'
if "$LOCAL"; then
prompt ENVS 'Parallel worlds' "$ENVS_DEFAULT" '^[1-9][0-9]{0,5}$' true
prompt UPDATES 'Maximum optimizer updates' "$UPDATES_DEFAULT" '^[1-9][0-9]{0,8}$' true
prompt ROLLOUT 'Steps per rollout' "$ROLLOUT_DEFAULT" '^[1-9][0-9]{0,5}$' true
prompt HIDDEN 'Hidden width' "$HIDDEN_DEFAULT" '^[1-9][0-9]{0,4}$' true
prompt EPOCHS 'PPO epochs' "$EPOCHS_DEFAULT" '^[1-9][0-9]{0,3}$' true
prompt MINIBATCH 'Minibatch size' "$MINIBATCH_DEFAULT" '^[1-9][0-9]{0,7}$' true
prompt LR 'Learning rate' "$LR_DEFAULT" '^([0-9]*\.[0-9]*[1-9][0-9]*|[1-9][0-9]*(\.[0-9]+)?)([eE][+-]?[0-9]+)?$' true
prompt THREADS 'Trainer CPU threads (auto allowed)' auto '^(auto|[1-9][0-9]{0,2})$' true
prompt REPLAY 'Replay interval in completed duels (0 disables)' 50000 '^(0|[1-9][0-9]{0,8})$' true
else
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
fi
fi

if "$SHARED_GPU"; then
    case "$SWEEP_VARIABLE" in
        ENVS) PARAMETER=envs ;; UPDATES) PARAMETER=updates ;; ROLLOUT) PARAMETER=rollout_steps ;;
        HIDDEN) PARAMETER=hidden_dim ;; EPOCHS) PARAMETER=epochs ;; MINIBATCH) PARAMETER=minibatch_size ;;
        LR) PARAMETER=lr ;; THREADS) PARAMETER=threads ;; REPLAY) PARAMETER=replay_every ;;
        GAMMA|GAE_LAMBDA|CLIP_EPS|VF_COEF|ENT_COEF|MAX_GRAD_NORM) PARAMETER="${SWEEP_VARIABLE,,}" ;;
        *) printf 'Shared-GPU mode requires multiple at one training setting, not allocation CPU/RAM/wall-time.\n' >&2; exit 1 ;;
    esac
    SWEEP_DIRECTORY="$ROOT/simulator/training/sweeps/$JOB_NAME"
    SWEEP_OPTIONS="$("$PYTHON" - "$ENVS" "$UPDATES" "$ROLLOUT" "$HIDDEN" "$EPOCHS" "$MINIBATCH" "$LR" "$THREADS" "$REPLAY" "$PROFILE_EVERY" "$GAMMA" "$GAE_LAMBDA" "$CLIP_EPS" "$VF_COEF" "$ENT_COEF" "$MAX_GRAD_NORM" <<'PY'
import json
import sys
from simulator.training.compiler import LEARNER_DEFAULTS
keys = ("envs", "updates", "rollout_steps", "hidden_dim", "epochs", "minibatch_size", "lr", "threads", "replay_every", "profile_every",
        "gamma", "gae_lambda", "clip_eps", "vf_coef", "ent_coef", "max_grad_norm")
floats = set(LEARNER_DEFAULTS) - {"hidden_dim", "epochs", "minibatch_size"}
options = {key: float(value) if key in floats else int(value) for key, value in zip(keys, sys.argv[1:])}
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
    if "$FORM_USED"; then CONFIRM=y; else prompt CONFIRM 'Submit 1 shared-GPU sweep job? (y/n)' n '^[yYnN]$'; fi
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
    EXPORTS="ALL,MCSIM_CUDA_ARCHITECTURES=$ARCH,SIM_PLAN=$PLAN,SIM_OUTPUT=$RUN_OUTPUT,SIM_ENVS=$ENVS,SIM_UPDATES=$UPDATES,SIM_SAVE_EVERY=$SAVE_EVERY,SIM_RESUME=$RESUME,SIM_INITIALIZE=$INITIALIZE,SIM_START_STAGE=$START_STAGE,SIM_ROLLOUT_STEPS=$ROLLOUT,SIM_HIDDEN_DIM=$HIDDEN,SIM_EPOCHS=$EPOCHS,SIM_MINIBATCH_SIZE=$MINIBATCH,SIM_LR=$LR,SIM_THREADS=$THREADS,SIM_REPLAY_EVERY=$REPLAY,SIM_PROFILE_EVERY=$PROFILE_EVERY,SIM_GAMMA=$GAMMA,SIM_GAE_LAMBDA=$GAE_LAMBDA,SIM_CLIP_EPS=$CLIP_EPS,SIM_VF_COEF=$VF_COEF,SIM_ENT_COEF=$ENT_COEF,SIM_MAX_GRAD_NORM=$MAX_GRAD_NORM"
    COMMAND=(sbatch --parsable "--job-name=${RUN_NAMES[index]}" "--partition=$PARTITION" "--gres=$GRES"
        "--time=$TIME" "--cpus-per-gpu=$CPUS" "--mem=$MEMORY" "--export=$EXPORTS")
    if [[ "$dependency" != none ]]; then
        COMMAND+=("--dependency=afterany:$dependency")
    fi
    COMMAND+=(rosie/train.sbatch)
}

prepare_local_command() {
    local index="$1"
    if [[ -n "$SWEEP_VARIABLE" ]]; then
        printf -v "$SWEEP_VARIABLE" '%s' "${SWEEP_VALUES[index]}"
    fi
    RUN_OUTPUT="simulator/training/runs/${RUN_NAMES[index]}"
    LOCAL_ENV=(
        "MCSIM_CUDA_ARCHITECTURES=$ARCH"
        "SIM_PLAN=$PLAN"
        "SIM_OUTPUT=$RUN_OUTPUT"
        "SIM_ENVS=$ENVS"
        "SIM_UPDATES=$UPDATES"
        "SIM_SAVE_EVERY=$SAVE_EVERY"
        "SIM_ROLLOUT_STEPS=$ROLLOUT"
        "SIM_HIDDEN_DIM=$HIDDEN"
        "SIM_EPOCHS=$EPOCHS"
        "SIM_MINIBATCH_SIZE=$MINIBATCH"
        "SIM_LR=$LR"
        "SIM_THREADS=$THREADS"
        "SIM_REPLAY_EVERY=$REPLAY"
        "SIM_REPLAY_EVERY_UPDATES=0"
        "SIM_REPLAY_DIR="
        "SIM_PROFILE_EVERY=$PROFILE_EVERY"
        "SIM_RESUME=$RESUME"
        "SIM_INITIALIZE=$INITIALIZE"
        "SIM_START_STAGE=$START_STAGE"
        "SIM_SWEEP="
        "SIM_GAMMA=$GAMMA"
        "SIM_GAE_LAMBDA=$GAE_LAMBDA"
        "SIM_CLIP_EPS=$CLIP_EPS"
        "SIM_VF_COEF=$VF_COEF"
        "SIM_ENT_COEF=$ENT_COEF"
        "SIM_MAX_GRAD_NORM=$MAX_GRAD_NORM"
    )
    COMMAND=(env "${LOCAL_ENV[@]}" bash rosie/train-local.sh)
}

if "$LOCAL"; then
    printf '\nLocal working directory: %s\nFresh run; inherited SIM_* settings are replaced.\n' "$ROOT"
else
    printf '\nWorking directory: %s\nFresh run; existing SIM_* settings and SBATCH_* overrides will not be inherited.\n' "$ROOT"
fi
if [[ -n "$SWEEP_VARIABLE" ]]; then
    printf 'Sweep: %s (%d jobs). Jobs can train concurrently; shared builds are locked.\n' "$SWEEP_LABEL" "${#RUN_NAMES[@]}"
fi
for index in "${!RUN_NAMES[@]}"; do
    if "$LOCAL"; then
        prepare_local_command "$index"
        printf 'Local command (%s):\n  ' "${RUN_NAMES[index]}"
    else
        prepare_command "$index" "$DEPENDENCY"
        printf 'Submission command (%s):\n  ' "${RUN_NAMES[index]}"
    fi
    printf '%q ' "${COMMAND[@]}"
    printf '\n\n'
done
if "$DRY_RUN"; then
    if "$LOCAL"; then printf 'Dry run: nothing built or launched.\n'; else printf 'Dry run: nothing submitted.\n'; fi
    exit 0
fi
if "$FORM_USED"; then
    CONFIRM=y
elif "$LOCAL"; then
    prompt CONFIRM "Start ${#RUN_NAMES[@]} local run(s)? (y/n)" n '^[yYnN]$'
else
    prompt CONFIRM "Submit ${#RUN_NAMES[@]} job(s)? (y/n)" n '^[yYnN]$'
fi
if [[ "$CONFIRM" != [yY] ]]; then
    if "$LOCAL"; then printf 'Cancelled; nothing launched.\n'; else printf 'Cancelled; nothing submitted.\n'; fi
    exit 0
fi

if "$LOCAL"; then
    for index in "${!RUN_NAMES[@]}"; do
        prepare_local_command "$index"
        printf 'Starting local run %s. Press Ctrl+C to stop it.\n' "${RUN_NAMES[index]}"
        if ! "${COMMAND[@]}"; then
            printf 'Local run failed for %s; stopping the remaining sweep.\n' "${RUN_NAMES[index]}" >&2
            exit 1
        fi
        printf 'Finished local run %s\n' "${RUN_NAMES[index]}"
    done
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