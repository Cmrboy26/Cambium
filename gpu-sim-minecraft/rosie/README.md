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

## Dashboard

With a training job running, run this from a local Bash terminal while connected
to GlobalVPN:

```sh
bash rosie/dashboard.sh
```

The script finds the running `cambium-sim` job, starts the read-only dashboard
as an overlapping step on its compute node, chooses both ports automatically,
and opens the local URL. Pass a job ID when multiple training jobs are running:
`bash rosie/dashboard.sh <JOB_ID>`. Press Ctrl+C to stop the dashboard and
tunnel; no separate dashboard allocation is submitted.

See [AGENTS.md](AGENTS.md) for GPU selection, experiment options, output paths,
tunneling details, and local Linux checks. Set `ROSIE_SSH_USER` or
`ROSIE_SSH_JUMP` to override the SSH account or jump host.
