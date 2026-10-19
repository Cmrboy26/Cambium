# Rosie GPU Simulator Workflow

Follow the repository [development guide](../AGENTS.md). This folder owns
cluster-specific launchers; general simulator and training behavior remains in
the simulator guides.

## Scope And Prerequisites

This workflow trains the standalone Minecraft Java 26.3 simulator. It does not
start the legacy 1.8.9 server, HeadlessMC clients, or FastAPI backend. Submit
jobs from `gpu-sim-minecraft` so `SLURM_SUBMIT_DIR`, relative environment paths,
and `rosie/logs` resolve as expected.

`setup.sh` creates `.venv-rosie` and installs CUDA-enabled PyTorch 2.5.1/cu124
plus the training requirements. It runs on the management node and does not
compile CUDA. The first training job configures CMake and builds on its allocated
compute node, which needs network access for the pinned FetchContent dependencies.

The training job checks `.venv-rosie/bin/python`. On `dh-dgxh100-*`, the venv's
management-node interpreter symlink may be unavailable. In that case, the job
uses the compute-node `python3` and adds the venv's matching
`lib/python<major>.<minor>/site-packages` to `PYTHONPATH`. The Rosie H100 setup
was verified with system Python 3.12.3, PyTorch 2.5.1+cu124, and an H100 80GB.
If the system Python minor version differs from the venv packages, this fallback
fails explicitly because the matching site-packages directory is absent.

## Training Resources And Build Trees

`train.sbatch` defaults to `teaching`, one T4, 8 CPUs, 24 GB and a two-hour
limit. It builds CUDA architecture 75 by default. Override resources on the
`sbatch` command line for another partition. H100 requires architecture 90:

```sh
sbatch --partition=dgxh100 --gres=gpu:h100:1 \
  --export=ALL,MCSIM_CUDA_ARCHITECTURES=90,SIM_ENVS=8192,SIM_UPDATES=500,SIM_ROLLOUT_STEPS=128,SIM_EPOCHS=2,SIM_MINIBATCH_SIZE=16384,SIM_HIDDEN_DIM=512 \
  rosie/train.sbatch
```

One requested H100 is used by this single-process trainer; requesting all eight
does not enable multi-GPU training. Builds are isolated by architecture:
`simulator/build-rosie-cuda-75` for T4 and
`simulator/build-rosie-cuda-90` for H100. This avoids different architectures
reconfiguring a shared CMake tree. Do not run concurrent jobs against the same
architecture/build directory.

Supported training environment variables:

- `SIM_ENVS`: parallel worlds; default 256.
- `SIM_UPDATES`: optimizer updates; default 1.
- `SIM_ROLLOUT_STEPS`: steps per rollout; default 16.
- `SIM_EPOCHS`, `SIM_MINIBATCH_SIZE`: PPO optimizer settings; defaults 4/1024.
- `SIM_HIDDEN_DIM`: width of the policy's hidden layers; default 256. Try 512 as a moderate increase; larger networks cost more optimizer time and are not guaranteed to train better.
- `SIM_THREADS`, `SIM_SEED`, `SIM_SCHEMA`: CPU threads, random seed, and `v1`/`v2` schema.
- `SIM_ENVIRONMENT`: JSON scenario path relative to `gpu-sim-minecraft` or an absolute path.
- `SIM_OUTPUT`: run directory; default `simulator/training/runs/rosie-<JOB_ID>`.
- `SIM_REPLAY_EVERY`: replay interval in completed duels; default 0 (disabled).
- `SIM_REPLAY_DIR`: optional replay destination; default `<run>/replays` when replay is enabled.

For the close-duel scenario, ensure the JSON exists in the Rosie checkout, then
submit for up to 24 hours (within the partition's 21-day maximum):

```sh
sbatch --partition=dgxh100 --gres=gpu:h100:1 --time=1-00:00:00 \
  --export=ALL,MCSIM_CUDA_ARCHITECTURES=90,SIM_ENVS=8192,SIM_UPDATES=500,SIM_ROLLOUT_STEPS=128,SIM_EPOCHS=2,SIM_MINIBATCH_SIZE=16384,SIM_ENVIRONMENT=simulator/environments/research-duel-close-rebalancedv2.json,SIM_REPLAY_EVERY=50000 \
  rosie/train.sbatch
```

Windows DLL paths and `%APPDATA%`/`$env:APPDATA` do not exist on Rosie. With
`SIM_REPLAY_EVERY=50000`, files go to
`simulator/training/runs/rosie-<JOB_ID>/replays` unless `SIM_REPLAY_DIR` is set.
Replay cadence counts completed duels, not optimizer updates. CUDA training
still stages observations and rewards through the CPU.
The hidden size is part of checkpoint compatibility; changing it starts a new
model rather than resuming a checkpoint with a different hidden dimension.

SLURM output and error logs are `rosie/logs/gpu-sim-<JOB_ID>.out` and `.err`.
The trainer writes metrics and checkpoints under its run directory. A one-update
smoke job is a wiring check, not a useful training duration; check its logs before
scaling up.

## Dashboard And SSH Tunnel

`dashboard.sbatch` requests one CPU, 2 GB, and eight hours by default. It reads
`simulator/training/runs` and binds only to `127.0.0.1:8767`. The job log prints
an SSH `ProxyJump` command using `dh-mgmt3.hpc.msoe.edu` and the allocated compute
node. On the local machine, connect to GlobalVPN, run the printed command, and
open `http://127.0.0.1:8768`; local 8768 forwards to remote 8767. It is not
necessary to expose an unauthenticated dashboard port to the network.

- `LOCAL_DASHBOARD_PORT`: local tunnel port printed in the command; default 8768.
- `DASHBOARD_PORT`: dashboard's remote loopback port; default 8767.
- `SIM_RUNS_DIR`: runs directory to display; default `simulator/training/runs`.
- `ROSIE_SSH_JUMP`: override the default management-host jump target if needed.
- `ROSIE_SSH_USER`: override the SSH account prefix printed by the job.

Logs use `rosie/logs/dashboard-<JOB_ID>.out` and `.err`. Ctrl+C while tailing a
log only stops `tail`; stop the service with `scancel <JOB_ID>`. If SSH policy
blocks direct compute-node access through `ProxyJump`, use the site's approved
tunneling method.

## Local Linux Checks

The PowerShell scripts under `scripts/` target Windows. To build and run a small
CPU check on Linux from `gpu-sim-minecraft`:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r simulator/training/requirements.txt
cmake -S simulator -B simulator/build -DMCSIM_ENABLE_CUDA=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build simulator/build --parallel
ctest --test-dir simulator/build --output-on-failure
python -B -m simulator.training.train --device cpu --envs 4 --updates 1 --rollout-steps 16
```

Do not use `sudo`; use an unprivileged venv and build folder. For local CUDA,
install CUDA-enabled PyTorch and configure with `-DMCSIM_ENABLE_CUDA=ON` and the
GPU's `CMAKE_CUDA_ARCHITECTURES` value.