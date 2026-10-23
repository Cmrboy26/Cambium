# GPU Simulator on Rosie

Train the standalone Minecraft 26.3 simulator on Rosie with SLURM and a GPU.
This workflow is separate from Cambium's legacy Minecraft 1.8.9 server and bots.

## Setup

Run on Rosie from the Cambium repository:

```sh
cd gpu-sim-minecraft
bash rosie/setup.sh
```

Setup requires Python 3.12 for its pinned PyTorch wheel. It prefers `python3.12`;
set `ROSIE_PYTHON` to select a specific Python 3.12 interpreter.

## Train

Choose GPU and plan, then edit training settings with Up/Down and type values.
Ctrl+U clears a value; Enter on the bottom Submit row submits:

```sh
bash rosie/submit.sh
squeue -u "$USER"
tail -f rosie/logs/gpu-sim-<JOB_ID>.out
```

The selected plan is validated; `bash rosie/submit.sh --dry-run` previews without submitting.
The job/run name appears in the dashboard. Sweep one field with comma-separated
values (e.g. entropy `0,0.005,0.01`). Jobs get `-<value>` suffixes and train
concurrently with locked builds and private libraries. [Options](AGENTS.md#interactive-submission).
Use `--line-mode` for the original sequential prompts and `multiple` entry.
Use `--shared-gpu` for one job and an overlaid PNG; `--max-parallel N` caps concurrency. [Details](AGENTS.md#shared-gpu-sweeps).
For a one-update CUDA smoke test, use `sbatch rosie/train.sbatch` instead.

## Train Locally in WSL2

Use a checkout inside WSL's Linux filesystem, install the Windows NVIDIA WSL
driver and a CUDA toolkit with `nvcc` inside WSL, then run `bash rosie/setup.sh`
and `bash rosie/submit.sh --local`. The local wizard uses the same plan/settings,
defaults to architecture 86 and 256 worlds for an 8 GB GPU, and runs sweeps
sequentially. A checkout open through the remote Rosie VS Code session still
runs on Rosie; pull/sync the changes into your WSL checkout to use the 3060 Ti.
See [local training details](AGENTS.md#local-wsl2-training).

## Local Replays And Duels

For passwordless SSH on Windows, run once with GlobalVPN connected:

```powershell
.\rosie\setup-ssh-key.ps1 -SshHost 2 -SshUser reissmannc
```

Enter any prompts only in the terminal. [Key setup options](AGENTS.md#ssh-key-setup).

On your local computer, install the updated mod and use a Python environment
with the training requirements (CPU inference needs no native simulator build).
With GlobalVPN connected, configure your Minecraft game folder once:

```powershell
.\scripts\rosie-play.ps1 -Setup -MinecraftDirectory 'C:\path\to\game' -SshHost 2 -SshUser reissmannc
```

Use your assigned management node. On Windows, Minecraft opens a login terminal
when needed and reuses it for later operations. Enter any password or host-key
confirmation **in that terminal** and leave it open; credentials never go in chat.
Ctrl+C disconnects. Other platforms need a local terminal running
`scripts/rosie-play.ps1 -Connect -MinecraftDirectory <game>` first.
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

With GlobalVPN connected, run this **on your local computer** from the
`gpu-sim-minecraft` directory. In PowerShell, Git Bash must be installed and
available as `bash`:

```powershell
bash ./rosie/dashboard-connect.sh
```

The dashboard lists every run under `simulator/training/runs` and refreshes
every three seconds, including new or changing runs. On Rosie the script uses
an SSH tunnel; from a local WSL checkout it hosts that checkout's runs directly,
without SSH. It chooses a free port automatically. **No training job or Slurm
allocation is required or submitted.** Press Ctrl+C to stop the dashboard.

Set `ROSIE_PROJECT_DIR` to your remote checkout when connecting from your
computer without a job ID. On Rosie itself, the command uses its own
checkout and forwards localhost on that node, not your computer. An optional
`<JOB_ID>` still hosts the dashboard inside a running training allocation.

For a remote Rosie checkout launched from another computer, set
`ROSIE_SSH_JUMP` and `ROSIE_PROJECT_DIR` to retain tunnel behavior. See
[AGENTS.md](AGENTS.md) for GPU selection, experiment options, output paths, and
remote connection details.
