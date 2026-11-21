# GPU Simulator on Rosie

Train the standalone Minecraft 26.3 simulator on Rosie with SLURM and a GPU.
This workflow is separate from Cambium's legacy Minecraft 1.8.9 server and bots.

## Setup

On Rosie, clone Cambium into your own account, then run from that checkout:

```sh
cd gpu-sim-minecraft
bash rosie/setup.sh
pwd
```

Setup requires Python 3.12 for its pinned PyTorch wheel. It prefers `python3.12`;
set `ROSIE_PYTHON` to select a specific Python 3.12 interpreter. Note the `pwd`
output: local Rosie tools need this remote `gpu-sim-minecraft` directory.

## Train

Choose GPU and plan, then edit training settings with Up/Down and type values.
Ctrl+U clears a value; Enter on the bottom Submit row submits:

```sh
bash rosie/submit.sh
squeue -u "$USER"
tail -f rosie/logs/gpu-sim-<JOB_ID>.out
```

The wizard validates the selected plan before submitting; an invalid plan stops
with no job submitted. `bash rosie/submit.sh --dry-run` also validates and
previews the job without submitting.
Some research plans require checkpoints from earlier runs; a fresh checkout
does not include those files. See [checkpoint prerequisites](../simulator/setups/README.md#checkpoint-prerequisites).
The job/run name appears in the dashboard. Sweep one field with comma-separated
values (e.g. entropy `0,0.005,0.01`). Jobs get `-<value>` suffixes and train
concurrently with locked builds and private libraries. [Options](AGENTS.md#interactive-submission).
The form includes optional checkpoint initialization/resume, curriculum start
stage, full-checkpoint save cadence, and verbose timing-profile interval fields.
Alternatively, `--initialize PATH`, `--resume PATH`, `--start-stage NAME`,
`--save-every N`, and `--profile-every N` set their initial values. Resume restores
optimizer and curriculum progress from a full checkpoint and cannot be combined
with initialization or stage selection.
Use `--line-mode` for the original sequential prompts and `multiple` entry.
Use `--shared-gpu` for one job and an overlaid PNG; `--max-parallel N` caps concurrency. [Details](AGENTS.md#shared-gpu-sweeps).
For a one-update CUDA smoke test, use `sbatch rosie/train.sbatch` instead.
Set `SIM_VERIFY_GENERATED_TERRAIN=1` when submitting `rosie/train.sbatch` to
run the generated Bridge terrain CPU/CUDA parity test against the job's freshly
built library before training. A failed test stops the job.

## Train Locally in WSL2

Use a checkout inside WSL's Linux filesystem, install the Windows NVIDIA WSL
driver and a CUDA toolkit with `nvcc` inside WSL, then run `bash rosie/setup.sh`
and `bash rosie/submit.sh --local`. The local wizard uses the same plan/settings,
defaults to architecture 86 and 256 worlds for an 8 GB GPU, and runs sweeps
sequentially. A checkout open through the remote Rosie VS Code session still
runs on Rosie; pull/sync the changes into your WSL checkout to use your local GPU.
See [local training details](AGENTS.md#local-wsl2-training).

## Local Replays And Duels

On your Windows computer, use a local Cambium checkout, install the mod and
follow [local Rosie setup](../scripts/README.md#rosie-setup-on-a-new-machine).
For passwordless SSH, run once with GlobalVPN connected:

```powershell
.\rosie\setup-ssh-key.ps1
```

Enter any prompts only in the terminal. [Key setup options](AGENTS.md#ssh-key-setup).

On your local computer, install the updated mod and use a Python environment
with the training requirements (CPU inference needs no native simulator build).
With GlobalVPN connected, configure your Minecraft game folder once. The
prompts ask for your assigned node, your account and the remote path from `pwd`:

```powershell
.\scripts\rosie-play.ps1 -Setup
```

Use your assigned management node. On Windows, Minecraft opens a login terminal
when needed and reuses it for later operations. Enter any password or host-key
confirmation **in that terminal** and leave it open; credentials never go in chat.
Ctrl+C disconnects. Other platforms need a local terminal running
`scripts/rosie-play.ps1 -Connect -MinecraftDirectory <game>` first.
The connection verifies the runs directory, which is created by the first training
run; rerun setup to correct a wrong path.
In a private, disposable singleplayer world:

```text
/simrosie runs
/simrosie replay rosie-123
/simrosie duel rosie-123
/simduel join
/simrosie stop
```

Replay downloads start playback; duels download weights and run inference locally.
Stop an active Rosie operation before starting another. Omit the run to select
the newest saved run, not the best policy. `/simrosie replays <run>` lists samples.
Training must enable `SIM_REPLAY_EVERY`; no running allocation is needed to fetch
saved outputs. [Offline use and configuration](AGENTS.md#local-artifact-launcher).

## Dashboard

With GlobalVPN connected, run this **on your local computer** from its
`gpu-sim-minecraft` directory. PowerShell needs Git Bash available as `bash`:

```powershell
$env:ROSIE_SSH_JUMP = Read-Host 'Assigned Rosie management node (1-4)'
$env:ROSIE_SSH_USER = Read-Host 'Rosie account'
$env:ROSIE_PROJECT_DIR = Read-Host 'Remote gpu-sim-minecraft directory (pwd on Rosie)'
bash ./rosie/dashboard-connect.sh --remote
```

The dashboard lists every run under `simulator/training/runs` and refreshes
every three seconds, including new or changing runs. For a local checkout's
runs, use `bash ./rosie/dashboard-connect.sh --local` instead. It chooses a
free port automatically. **No training job or Slurm
allocation is required or submitted.** Press Ctrl+C to stop the dashboard.

An optional job ID after `--remote` hosts the dashboard inside that running
allocation and discovers its checkout. On Rosie itself, the command forwards
localhost on that node, not your computer. See [AGENTS.md](AGENTS.md) for
GPU selection, experiment options, output paths, and remote connection details.
