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

Start the dashboard as a separate job:

```sh
sbatch rosie/dashboard.sbatch
tail -f rosie/logs/dashboard-<JOB_ID>.out
```

Connect to GlobalVPN, run the SSH tunnel command printed in the log, and open
`http://127.0.0.1:8768`. Stop the dashboard with `scancel <JOB_ID>`.

See [AGENTS.md](AGENTS.md) for GPU selection, experiment options, output paths,
tunneling details, and local Linux checks.
