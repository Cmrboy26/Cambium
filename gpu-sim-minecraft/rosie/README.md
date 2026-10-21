# GPU Simulator on Rosie

Train the standalone Minecraft 26.3 simulator on Rosie with SLURM and a GPU.
This workflow is separate from Cambium's legacy Minecraft 1.8.9 server and bots.

## Setup

Run on Rosie from the Cambium repository:

```sh
cd gpu-sim-minecraft
bash rosie/setup.sh
```

## Train

Walk through GPU, plan and training settings; press Enter to accept defaults:

```sh
bash rosie/submit.sh
squeue -u "$USER"
tail -f rosie/logs/gpu-sim-<JOB_ID>.out
```

The selected plan is validated; `bash rosie/submit.sh --dry-run` previews without submitting.
The job/run name appears in the dashboard. Sweep one setting with `multiple`, then
values one per line and an empty line. Jobs get `-<value>` suffixes and train
concurrently with locked builds and private libraries. [Options](AGENTS.md#interactive-submission).
Use `--shared-gpu` for one job and an overlaid PNG; `--max-parallel N` caps concurrency. [Details](AGENTS.md#shared-gpu-sweeps).
For a one-update CUDA smoke test, use `sbatch rosie/train.sbatch` instead.

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
.\scripts\rosie-play.ps1 -Connect -MinecraftDirectory 'C:\path\to\game'
```

Use your assigned management node. Enter the SSH password and any host-key
confirmation **in the local terminal**, then leave that terminal open. Minecraft
reuses this connection; the password is not stored or entered in chat. Ctrl+C
disconnects. With working key/agent authentication, `-Connect` is optional.
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
every three seconds, including new or changing runs. The script hosts it on
your Rosie management node, chooses both ports automatically, and opens the
local URL through an SSH tunnel. **No training job or Slurm allocation is
required or submitted.** Saved runs remain accessible when training stops.
Press Ctrl+C to stop the remote dashboard and local tunnel.

Remote checkout: `~/minecraftlol/Cambium/gpu-sim-minecraft`.
Override with `ROSIE_PROJECT_DIR`. On Rosie itself, the command uses its own
checkout and forwards localhost on that node, not your computer. An optional
`<JOB_ID>` still hosts the dashboard inside a running training allocation.

See [AGENTS.md](AGENTS.md) for GPU selection, experiment options, output paths,
tunneling details, and local Linux checks. If `ROSIE_SSH_JUMP` is unset, the
script derives the management host when run on Rosie. When run locally, it
prompts for your assigned node number (`1`-`4`) or full hostname, such as
`dh-mgmt1.hpc.msoe.edu`. Set `ROSIE_SSH_JUMP` to skip the prompt; it accepts the
same number or hostname. Set
`ROSIE_SSH_USER` to override the account; by default, the local username is
suffixed with `@ad.msoe.edu`.
