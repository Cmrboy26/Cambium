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

With a `cambium-sim` training job running, start
`bash ./rosie/dashboard-connect.sh`
from PowerShell in the local `gpu-sim-minecraft` directory (Git Bash must be
installed and available as `bash`), or `bash rosie/dashboard-connect.sh` from
Git Bash.
Connect to GlobalVPN first. The script finds the unique running training job,
or selects one automatically when several share a working directory; pass the
job ID when they use different working directories.
It starts the read-only Python dashboard with `srun --jobid ... --overlap` on
the allocated compute node, so it does not submit another job or request
another node. Both the remote and local loopback ports are selected
automatically; the script prints and opens the local URL. Ctrl+C stops the
dashboard step and tunnel, not the training job.

The dashboard binds only to loopback and is reached through SSH `ProxyJump` via
the assigned `dh-mgmt1` through `dh-mgmt4` host; do not expose its port to the
network. If `ROSIE_SSH_JUMP` is unset, the script derives the host when run on
a Rosie management node; from a local computer it prompts for the node number
(`1`-`4`) or full hostname (for example, `dh-mgmt1.hpc.msoe.edu`). The variable
accepts either form. The local computer needs Git Bash, SSH and Python 3
(`python3` or `python`). Set
`ROSIE_SSH_USER` to specify/override the Rosie account, `ROSIE_SSH_JUMP` to
avoid the host prompt, or `ROSIE_JOB_ID` to select a training job without a
positional argument. The default SSH account appends `@ad.msoe.edu` to the local
username. If SSH policy blocks direct
compute-node access through `ProxyJump`, use the site's approved tunneling
method.

## Local Artifact Launcher

`scripts/rosie-play.ps1 -Setup` configures the **local** Minecraft instance;
it does not start a Rosie job or modify the remote installation. The account,
management host/SSH alias, remote runs directory, local repository and selected
Python executable are saved in game `config/cambium-rosie.json`. Use the local
checkout and a Python environment with training requirements for duels. Re-run
setup after moving the checkout or changing interpreters. No credentials are
stored; use the local SSH agent or a configured key. Keep the config trusted:
the mod executes its Python interpreter from its repository directory.

Connect GlobalVPN and verify normal `ssh -l <account> <management-host>` access
first, including host-key verification. In-game requests use `BatchMode=yes`,
a ten-second connect timeout and SSH keepalives. Interactive passwords, MFA
prompts and passphrase entry are not supported in Minecraft; handle these through
your site's approved key/agent/connection-reuse configuration in a terminal.
Do not disable host-key checking. The selected `ssh` command/agent must be
available to Minecraft's process, not only a separate terminal environment.

The default remote root is
`~/minecraftlol/Cambium/gpu-sim-minecraft/simulator/training/runs`; override with
`-RemoteRunsDirectory` at setup. SSH runs a bundled, standard-library Python 3
artifact reader on the management host. Only saved files are read, not training
or inference. No public port, file server or active SLURM allocation is needed.
This assumes that the run root is visible on that management host. Custom
`SIM_REPLAY_DIR` locations must be collected into the run's `replays` directory
before using this launcher.

The helper is `python -B -m simulator.training.rosie <action>
--minecraft-directory <game>`; actions are `setup`, `runs`, `replays`, `replay`,
`sync`, `duel`. `--run` and `--replay` default to `latest`; `--offline` uses only
cached artifacts. `--json` emits newline-separated events for the mod. Run/replay
basenames are limited to 120 ASCII letters, digits, dots, underscores or hyphens,
starting with a letter/digit. Paths and remote arguments are encoded, not shell
interpolated; resolved remote files must remain inside the configured run root.

`replay` downloads only the selected sample to game `sim-replays`, prefixing its
filename with the run ID. `duel` caches `latest.pt` and available config/metrics
in game `sim-rosie-cache/<run>`, then serves CPU inference on an automatically
allocated loopback port. `sync` caches the checkpoint and **all** replay files
for that run; consider disk usage before syncing large runs. Transfers reuse
matching size/mtime files, stream to unique `.part` files, verify length/SHA-256,
then atomically replace destinations. A failed transfer does not replace an
existing artifact. Remote reads pin a file descriptor across atomic trainer
checkpoint replacements. Replay limit is 64 MiB, other artifact limit 512 MiB.

For offline duels: `scripts/rosie-play.ps1 -Action duel -Run <run> -Offline`;
use `/simduel arena <printed-port>`. Ctrl+C stops the terminal-owned service.
Cached replays can be loaded normally with `/simreplay load latest`. The cache
is not a historical model archive: syncing newer weights replaces cached
`latest.pt`, and neither local nor remote `latest` means best-performing.

Tests: `python -B -m unittest simulator.training.test_rosie -v` is stdlib-only
and uses local subprocesses to exercise the actual remote reader. The existing
`test_training.InferenceChecks` also launches a real offline policy process.
Neither proves local Windows SSH authentication or in-game behavior. See
[in-game commands and lifecycle](../cambiummod/AGENTS.md#rosie-artifact-commands)
and [launcher parameters](../scripts/AGENTS.md#rosie-sync-and-play).

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