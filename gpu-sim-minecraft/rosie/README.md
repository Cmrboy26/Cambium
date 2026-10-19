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

Submit a short CUDA smoke test from `gpu-sim-minecraft`:

```sh
sbatch rosie/train.sbatch
squeue -u "$USER"
tail -f rosie/logs/gpu-sim-<JOB_ID>.out
```

## Local Replays And Duels

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

With a training job running and GlobalVPN connected, run this from the
`gpu-sim-minecraft` directory. In PowerShell, Git Bash must be installed and
available as `bash`:

```powershell
bash ./rosie/dashboard-connect.sh
```

In a Git Bash terminal, use `bash rosie/dashboard-connect.sh` instead.

The dashboard lists every run under `simulator/training/runs` and refreshes
every three seconds, including new or changing runs. The script starts it as an
overlapping step on a running job's compute node, chooses both ports
automatically, and opens the local URL. If multiple jobs share a working
directory, it uses one to host the dashboard; pass a job ID only when jobs use
different working directories: `bash rosie/dashboard-connect.sh <JOB_ID>`. The
dashboard server runs on Rosie, not on the local computer. Press Ctrl+C to stop
the remote dashboard step and local tunnel; no separate dashboard allocation
is submitted.

See [AGENTS.md](AGENTS.md) for GPU selection, experiment options, output paths,
tunneling details, and local Linux checks. If `ROSIE_SSH_JUMP` is unset, the
script derives the management host when run on Rosie. When run locally, it
prompts for your assigned node number (`1`-`4`) or full hostname, such as
`dh-mgmt1.hpc.msoe.edu`. Set `ROSIE_SSH_JUMP` to skip the prompt; it accepts the
same number or hostname. Set
`ROSIE_SSH_USER` to override the account; by default, the local username is
suffixed with `@ad.msoe.edu`.
